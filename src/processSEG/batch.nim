## Batch jobs over a whole file: `bruteStack`, and `writeStack` to save a
## stack as SEG-Y (or `stackKeys` and `stackOverrides` to write it yourself
## through parseSEG's writer).
##
## A brute stack reads the traces through parseSEG's prefetcher and, on its
## decode workers (decision D13: the prefetcher's `transform` hook, with
## `uniform = true`), runs on each chunk while it is in cache:
##
## 1. datum statics (optional, `DatumStatics`);
## 2. the pre-stack flow, a `Pipeline` (mutes, filters, gain);
## 3. NMO with a stretch mute, onto the stack's time axis.
##
## The calling thread takes the corrected chunks in file order and stacks
## each trace into its bin, with a per-sample fold (the span between its
## first and last non-zero samples, as `addLive`).
##
## ```nim
## var f = openSeismic("line.sgy", keys = true)
## var opts = initBruteStackOptions(flow = initPipeline([bandPassOp(
##   Corners(f1: 4, f2: 8, f3: 60, f4: 80))]))
## opts.workers = 4
## let res = bruteStack(f, float32, cdpBinning(firstCdp = 1, count = 800),
##   velocities, opts, progress = proc (done, total: int64): bool = true)
## let part = bruteStack(f, float32, 0 ..< 5000, binning, velocities, opts)
## let side = bruteStack(f, float32, picked, binning, velocities, opts) # a list
## # res.data: [bins, samples]; res.fold; res.binning.location(bin);
## # res.binTraces[bin]: a fold map; res.cmpX[bin], res.cmpY[bin]: its place
## ```
##
## **Velocities** are read at each bin's location (`Binning.location`): a
## 2D field by the bin's CDP number, a 3D field by its inline and
## crossline numbers. A `VelocityFunction` serves every bin. A field must
## reach the stack's last sample (`tMaxMs` ≥ `startMs + (samples − 1)·dtMs`;
## `ValueError` otherwise): past its end its velocities would stay constant.
##
## **Traces.** Only live traces are stacked (`ChunkMeta.isLive`). A live
## trace is skipped, and counted in `StackResult.skipped`, when it falls
## outside every bin, has a NaN offset or `t0Ms`, a NaN static (with
## `statics`), no sample interval in its headers, or a NaN or infinite
## sample, or when the stretch mute leaves nothing of it. Traces that could
## not be decoded are counted in `failed` (their diagnostics are dropped:
## read them with parseSEG if needed). The live traces with an interval
## must all have the same one (a mixed SEG-D record: select the traces of
## one channel set), and the stack and the flow are sized by the longest
## of them. The traces need not be sorted. `traces` is a range of the file
## or a list of trace indices (a map selection, one component, one channel
## set), stacked in list order; a trace listed twice counts twice.
##
## **Reproducibility.** The traces are summed in the order of `traces`,
## whatever the number of workers (the workers correct the traces; one
## thread stacks them): a stack is the same bit for bit for 1 or 16
## workers, run after run. A `perPanel` normalization in the flow depends
## on `chunkTraces`, nothing else does.
##
## **Cost.** A `VelocityField` is read once per bin and worker, as long as
## the bins a worker is working on fit its cache of velocity columns
## (`velocityCacheBytes`, 8 MiB per worker by default: 698 bins of 1501
## samples, kept by bin number): CDP-sorted and shot-sorted input alike,
## the latter while a shot's traces fall in fewer bins than that. Beyond
## it a column is computed again when its bin comes back (`samples`
## interpolations): raise the cache for 3D shots that spread over
## thousands of bins. The cache never changes the result.
##
## **Threads.** `bruteStack` blocks its caller: run it on a background
## thread of your own. The job only reads the file (its map, index and
## keys). While it runs, other threads may read the same `SeismicFile`
## (`readTracesInto`, `traceKeys`, views, the prefetchers they have) and,
## with parseSEG 0.10.0 or later, create and close prefetchers on it. (A
## prefetcher pins the file's map, and the job has one of its own; before
## 0.10.0 the pin count is not thread-safe, so two threads must not create
## or close prefetchers on one file at the same time.) No thread may
## change or close the file until `bruteStack` has returned or raised: no
## `applySurvey` or `applyGeometry`, no header edit, no `close`. Cancel
## through `progress` and wait for the call to end first.
##
## **Progress and cancel.** `progress(done, total)` is called on the
## calling thread after every chunk with the **traces** done and the total
## (parseSEG's `Progress` type, which elsewhere counts bytes), and
## once at the end with `done == total`; returning false stops the workers
## and raises parseSEG's `SegCancelledError`.
##
## **Memory.** The job holds the stack once, `bins × samples × (sizeof(T)
## + 4)` bytes, whatever the number of workers: the sums are normalized in
## place and become the result. Besides it: the prefetch ring
## (`maxRingBytes`, at least `workers + 1` chunks, plus the chunk the
## consumer holds; a chunk is `chunkTraces` rows of the longer of the input
## trace and the stack) and each worker's flow scratch (`bruteStackMemory`
## estimates the total).

import std/[atomics, math, options]
import parseSEG
import core/[rows, types, workspace]
import pipeline
import stack/[binning, nmo, stack]
import statics/elevation
import velocity/[model, moveout]

# For `BruteStackOptions.statics`: callers write `some(datumStatics(...))`.
export options

type
  BruteStackOptions* = object
    ## The processing and the job's resources. Start from
    ## `initBruteStackOptions` and change fields.
    flow*: Pipeline ## Pre-stack flow, after statics, before NMO. A
                      ## `perPanel` normalization works per prefetch chunk
                      ## (`chunkTraces`, cut where the interval changes), so its
                      ## factors depend on the chunking: prefer `perTrace` here.
    statics*: Option[DatumStatics] ## Datum statics, applied first:
                                     ## `some(datumStatics(...))`; `none`: no statics.
    staticsInterp*: ShiftInterp ## Interpolation of the statics shifts.
    nmo*: NmoParams ## NMO interpolation and stretch mute.
    norm*: FoldNorm ## Normalization of the stack.
    samples*: int ## Stack samples; 0: the longest input trace's.
    startMs*: float64 ## Time of the stack's sample 0.
    workers*: int ## Prefetch (and processing) threads.
    chunkTraces*: int ## Traces per prefetch chunk.
    maxRingBytes*: int ## Prefetch ring cap (parseSEG's default: 4 MiB).
    velocityCacheBytes*: int ## Each worker's cache of `VelocityField`
                               ## columns, one per bin it met last (8 MiB;
                               ## 0: one column).

  StackResult*[T: SampleType] = object
    ## A brute stack: `data[bin, j]` at time `startMs + j·dtMs`.
    data*: Tensor[T]       ## `[bins, samples]`, normalized.
    fold*: Tensor[int32]   ## `[bins, samples]`, the per-sample fold.
    binning*: Binning      ## The bins (`location` gives each bin's numbers).
    dtMs*: float64         ## Sample interval, in ms.
    intervalUs*: float64   ## The same in µs, as the input's keys hold it, for
                           ## the writer (`stackKeys`): `dtMs · 1000` is not
                           ## always the whole microseconds `dtMs` came from.
                           ## 0 (a result built by hand): `dtMs · 1000` is used.
    startMs*: float64      ## Time of sample 0, in ms.
    binTraces*: seq[int32] ## Traces stacked into each bin: a fold map.
    cmpX*, cmpY*: seq[float64]
      ## The mean midpoint of the traces stacked into each bin: where a bin
      ## of header numbers lies on the map. NaN for a bin with no trace, or
      ## whose traces have no midpoint.
    stacked*: int ## Traces stacked.
    skipped*: int ## Live traces not stacked (see the module documentation).
    failed*: int ## Traces that could not be decoded.

func cacheSlots(cacheBytes, bins, samples: int): int =
  ## The velocity columns of a worker's cache: what `cacheBytes` holds, at
  ## least one, no more than there are bins.
  clamp(cacheBytes div max(samples, 1) div sizeof(float64), 1, max(bins, 1))

proc badSize(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func mulSize(a, b: int): int =
  # a · b of two sizes; a ValueError when it does not fit an int, in every
  # build mode (a danger build has no overflow checks).
  if b != 0 and a > high(int) div b: badSize("the size overflows")
  a * b

func addSize(a, b: int): int =
  if a > high(int) - b: badSize("the size overflows")
  a + b

proc initBruteStackOptions*(flow = initPipeline(), nmo = initNmoParams(),
    norm = fnFold, workers = 1): BruteStackOptions =
  ## Options with the flow `flow`, NMO `nmo`, normalization `norm` and
  ## `workers` threads; no statics, the input's length from 0 ms, chunks of
  ## 256 traces, a 4 MiB ring and 8 MiB of velocity columns per worker.
  BruteStackOptions(flow: flow, nmo: nmo, norm: norm, workers: workers,
    chunkTraces: 256, maxRingBytes: 4 shl 20, velocityCacheBytes: 8 shl 20,
    staticsInterp: siLinear)

func bruteStackMemory*(T: typedesc[SampleType], bins, samples, inSamples,
    workers: int, chunkTraces = 256, maxRingBytes = 4 shl 20,
    velocityCacheBytes = 8 shl 20): int =
  ## An estimate of the memory a `bruteStack` holds, in bytes: the stack
  ## (`stackMemory`, once: it becomes the result), the prefetch ring and
  ## the consumer's chunk (chunks of `chunkTraces` rows of `max(inSamples,
  ## samples)` samples), and each worker's scratch (taken as 16 lane rows
  ## of a 4× padded trace, which bounds the FFT filters' scratch, plus
  ## 1 MiB for a percentile histogram, an output trace and the cache of
  ## velocity columns: up to `velocityCacheBytes` and no more than every
  ## bin's, as a `VelocityField` takes; a function takes one column. A flow
  ## of several long filters may need more: `plan.workspaceSize` is
  ## exact). Raises `ValueError` for a negative
  ## size, `workers`, `chunkTraces` or `maxRingBytes` that are not
  ## positive, or a total beyond `int`.
  if inSamples < 0 or workers <= 0 or chunkTraces <= 0 or maxRingBytes <= 0 or
      velocityCacheBytes < 0:
    badSize("inSamples and velocityCacheBytes must be ≥ 0 and workers, " &
      "chunkTraces and maxRingBytes positive")
  let stack = stackMemory(T, bins, samples)
  let chunk = mulSize(mulSize(chunkTraces, max(inSamples, samples)), sizeof(T))
  let ring = addSize(max(maxRingBytes, mulSize(addSize(workers, 1), chunk)), chunk)
  let cache = mulSize(mulSize(cacheSlots(velocityCacheBytes, bins, samples),
    samples), 8)
  let perWorker = addSize(addSize(addSize(mulSize(16 * 4 * 8, inSamples),
    1 shl 20), mulSize(8, samples)), cache)
  addSize(addSize(stack, ring), mulSize(workers, perWorker))

type
  VelKind = enum velFunction, velField

  WorkerState[T] = object
    ws: Workspace
    # The cache of velocity columns: `job.slots` columns of `job.samples`
    # sloths (1/V²) on the stack's axis; slot i holds bin `slotBin[i]`.
    sloth: seq[float64]
    slotBin: seq[int]
    row: seq[T] # one NMO-corrected trace
    failed: bool
    msg: array[240, char]
    msgLen: int

  Job[T] = object
    plan: ptr PipelinePlan[T]
    binning: ptr Binning
    velKind: VelKind
    fun: ptr VelocityFunction
    field: ptr VelocityField
    useStatics: bool
    statics: DatumStatics
    staticsInterp: ShiftInterp
    nmo: NmoParams
    dtMs, startMs: float64
    intervalUs: float64
    inSamples: int          # the flow's row width: the longest live input trace
    samples: int            # the stack's
    slots: int              # velocity columns each worker caches
    states: ptr UncheckedArray[WorkerState[T]]
    anyFailed: Atomic[bool] # a worker failed: the consumer stops

proc note[T](job: ptr Job[T], s: var WorkerState[T], msg: string) {.raises: [].} =
  job.anyFailed.store(true)
  if s.failed: return
  s.failed = true
  s.msgLen = min(msg.len, s.msg.len)
  for i in 0 ..< s.msgLen: s.msg[i] = msg[i]

proc slothOf[T](job: ptr Job[T], s: var WorkerState[T],
    bin: int): ptr UncheckedArray[float64] {.raises: [ValueError].} =
  # The bin's sloths on the stack's axis, from the worker's cache: its RMS
  # velocities, read and inverted once for every trace of the bin the
  # worker meets while the column stays in its slot. A column depends on
  # the bin only, so what the cache holds never shows in the result. A
  # function has one column for every bin.
  let key = if job.velKind == velFunction: 0 else: bin
  let slot = key mod job.slots
  result = cast[ptr UncheckedArray[float64]](addr s.sloth[slot * job.samples])
  if s.slotBin[slot] != key:
    template column: untyped = result.toOpenArray(0, job.samples - 1)
    case job.velKind
    of velFunction:
      job.fun[].vrmsInto(job.startMs, job.dtMs, column)
    of velField:
      let loc = job.binning[].location(bin)
      case job.field[].kind
      of fk2D: job.field[].vrmsInto(loc.cdp, job.startMs, job.dtMs, column)
      of fk3D: job.field[].vrmsInto(loc.inline, loc.crossline, job.startMs,
          job.dtMs, column)
    vrmsToSloth(column)
    s.slotBin[slot] = key

func stackable[T](job: ptr Job[T], k: TraceKeys, bin: int): bool {.inline.} =
  # A live trace is corrected and stacked: it has a bin, an offset, a start
  # time, the job's interval and, with statics, a static.
  bin >= 0 and bin < job.binning[].len and not k.offset.isNaN and
    not k.t0Ms.isNaN and k.intervalUs == job.intervalUs and
    not (job.useStatics and job.statics.staticMs(k).isNaN)

proc correctChunk[T](worker: int, samples: ptr UncheckedArray[T], rows, cols: int,
    traces: TraceRows, failed: openArray[int], user: pointer) {.nimcall,
    gcsafe, raises: [].} =
  # The prefetcher's transform, on a decode worker: statics and the flow on
  # the chunk, then every stackable row NMO-corrected onto the stack's axis,
  # in place (the first `job.samples` columns of its row). Every other row
  # is zeroed there, so the consumer finds no live span in it. Nothing here
  # touches the stack: the consumer adds the rows, in chunk order.
  let job = cast[ptr Job[T]](user)
  template s: untyped = job.states[worker]
  if s.failed or rows == 0 or job.samples == 0: return
  try:
    let meta = initChunkMeta(traces, job.dtMs, 0, failed)
    let n = job.inSamples
    # A stack longer than the input widens the rows (cols > n). The flow
    # sees what it was compiled for, rows of n samples: they are moved to
    # the front of the chunk first, and corrected from the last row down, so
    # that a wide output row never covers an input row still to come.
    if cols > n:
      for r in 1 ..< rows:
        moveMem(addr samples[r * n], addr samples[r * cols], n * sizeof(T))
    let b = rowBlock(samples.toOpenArray(0, rows * n - 1), rows, n)
    if job.useStatics:
      applyStatics(b, meta, job.statics, job.staticsInterp) # NaN: untouched
    {.cast(gcsafe).}:
      job.plan[].apply(b, meta, s.ws)
    for r in countdown(rows - 1, 0):
      let dst = cast[ptr UncheckedArray[T]](addr samples[r * cols])
      let k = meta.keys(r)
      let bin = if meta.isLive(r): job.binning[].binOf(k) else: -1
      if job.stackable(k, bin):
        var sloth: ptr UncheckedArray[float64]
        {.cast(gcsafe).}:
          sloth = slothOf(job, s, bin)
        discard nmoSlothInto(b.row(r, meta.liveSamples(r, n)),
          meta.sampleTimeMs(r, 0), s.row, job.startMs, job.dtMs, k.offset,
          sloth.toOpenArray(0, job.samples - 1), job.nmo)
        copyMem(dst, addr s.row[0], job.samples * sizeof(T))
      else:
        zeroMem(dst, job.samples * sizeof(T))
  except CatchableError as e:
    job.note(s, e.msg)

proc stackChunk[T](acc: var StackAccumulator[T], c: PrefetchChunk[T],
    binning: Binning, dtMs: float64, stacked, skipped: var int) =
  # On the calling thread: adds the corrected rows of a chunk to their bins,
  # in row order. Which rows are live and where they go is decided from the
  # keys again (the transform has no channel to say so but the samples).
  if c.count == 0: return
  let meta = initChunkMeta(c.rows, dtMs, 0, c.failedRows)
  let b = rowBlock(c.samples)
  for r in 0 ..< c.count:
    if meta.isLive(r):
      let bin = binning.binOf(meta.keys(r))
      var added = false
      # No key reaches `add` with a bin outside the stack: its Defect would
      # end the process.
      if bin >= 0 and bin < acc.bins:
        let span = liveSpan(b.row(r, acc.samples))
        added = span.len > 0 and acc.add(bin, b.row(r, acc.samples), span.a,
          span.b + 1)
      if added:
        acc.addMidpoint(bin, meta.keys(r).cmpX, meta.keys(r).cmpY)
        inc stacked
      else:
        inc skipped

type TraceSel = object
  ## The traces of a job: a range of the file, or a list.
  listed: bool
  span: Slice[int]
  list: seq[int]

func len(sel: TraceSel): int =
  if sel.listed: sel.list.len else: max(sel.span.b - sel.span.a + 1, 0)

proc liveTraces(rows: TraceRows): (float64, int) =
  # The one sample interval of the live traces (in µs) and their longest
  # length; traces without an interval are left out (and skipped later).
  var us = NaN
  var longest = 0
  for i in 0 ..< rows.len:
    let k = rows.keysAt(i)
    let x = k.intervalUs
    if k.status in {tsLive, tsUnknown} and x > 0 and x < Inf:
      if us.isNaN: us = x
      elif x != us:
        raise newException(ValueError, "the live traces have different sample " &
          "intervals (" & $us & " and " & $x & " µs, trace " & $rows.traceOf(i) &
          "): stack the traces of one interval (one channel set)")
      longest = max(longest, rows.numSamples(i))
  if us.isNaN:
    raise newException(ValueError, "no live traces with a sample interval to stack")
  (us, longest)

proc checkOptions(o: BruteStackOptions) =
  if o.workers <= 0 or o.chunkTraces <= 0 or o.maxRingBytes <= 0:
    raise newException(ValueError, "workers, chunkTraces and maxRingBytes " &
      "must be positive")
  if o.velocityCacheBytes < 0:
    raise newException(ValueError, "velocityCacheBytes must be ≥ 0, got " &
      $o.velocityCacheBytes)
  if o.samples < 0:
    raise newException(ValueError, "samples must be ≥ 0, got " & $o.samples)
  if o.startMs.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "startMs must be finite, got " & $o.startMs)
  if o.nmo.maxStretch.isNaN or o.nmo.maxStretch <= 0:
    raise newException(ValueError, "opts.nmo must come from initNmoParams")
  if o.statics.isSome and not (o.statics.get.vReplacement > 0):
    raise newException(ValueError, "opts.statics must hold a datumStatics(...)")

proc checkCovers(field: VelocityField, startMs, dtMs: float64, samples: int) =
  # Past its last grid time a field's velocity stays constant: a field
  # shorter than the stack would correct the deep part with the velocity of
  # its end (2749 instead of 3747 m/s at 6 s for one that stops at 4 s).
  let lastMs = startMs + float64(max(samples - 1, 0)) * dtMs
  if lastMs > field.tMaxMs + 1e-6 * dtMs:
    raise newException(ValueError, "the velocity field ends at " &
      $field.tMaxMs & " ms, the stack at " & $lastMs & " ms: build the field " &
      "with tMaxMs ≥ " & $lastMs & " (or stack fewer samples)")

proc runStack[T](f: SeismicFile, sel: TraceSel, binning: Binning,
    job: var Job[T], opts: BruteStackOptions, progress: Progress): StackResult[T] =
  if not f.hasKeys:
    raise newException(ValueError, "bruteStack needs stored keys: open the " &
      "file with keys = true (or call ensureKeys)")
  checkOptions(opts)
  # `rowsOf` refuses traces outside the file.
  let (intervalUs, inSamples) = liveTraces(
    if sel.listed: f.rowsOf(sel.list) else: f.rowsOf(sel.span))
  let dtMs = intervalUs / 1000
  let samples = if opts.samples > 0: opts.samples else: inSamples
  if job.velKind == velField:
    checkCovers(job.field[], opts.startMs, dtMs, samples)
  let plan = opts.flow.compile(T, dtMs, max(inSamples, 1))
  # Everything the workers use is allocated here, on the calling thread.
  job.slots =
    if job.velKind == velFunction: 1
    else: cacheSlots(opts.velocityCacheBytes, binning.len, samples)
  var states = newSeq[WorkerState[T]](opts.workers)
  for s in states.mitems:
    s.ws = initWorkspace()
    s.ws.reserve(plan)
    s.sloth = newSeq[float64](job.slots * samples)
    s.slotBin = newSeq[int](job.slots)
    for b in s.slotBin.mitems: b = -1
    s.row = newSeq[T](samples)
  job.plan = addr plan
  job.binning = addr binning
  job.useStatics = opts.statics.isSome
  if job.useStatics: job.statics = opts.statics.get
  job.staticsInterp = opts.staticsInterp
  job.nmo = opts.nmo
  job.dtMs = dtMs
  job.intervalUs = intervalUs
  job.startMs = opts.startMs
  job.inSamples = inSamples
  job.samples = samples
  job.states = cast[ptr UncheckedArray[WorkerState[T]]](addr states[0])
  # One accumulator, on this thread: the workers hand over corrected chunks,
  # which are added in file order, so the sums are the same for any number
  # of workers. (An accumulator per worker, merged at the end, held
  # `workers + 1` stacks and was slower with 8 workers than with 1 on a
  # stack of 40 000 bins: 846 against 583 ms, 5.7 against 1.3 GB.)
  var acc = initStackAccumulator[T](binning.len, samples, dtMs, opts.startMs)
  result = StackResult[T](binning: binning, dtMs: dtMs, intervalUs: intervalUs,
    startMs: opts.startMs)
  let total = int64(sel.len)
  var done = 0'i64
  block run:
    # (The list form takes its own copy of the list.)
    var p =
      if sel.listed:
        newPrefetcher[T](f, sel.list, chunkTraces = opts.chunkTraces,
          depth = opts.workers + 2, cols = max(inSamples, samples),
          workers = opts.workers, maxRingBytes = opts.maxRingBytes,
          uniform = true, transform = correctChunk[T], user = addr job)
      else:
        newPrefetcher[T](f, sel.span, chunkTraces = opts.chunkTraces,
          depth = opts.workers + 2, cols = max(inSamples, samples),
          workers = opts.workers, maxRingBytes = opts.maxRingBytes,
          uniform = true, transform = correctChunk[T], user = addr job)
    defer: p.close()
    var c: PrefetchChunk[T]
    while p.nextInto(c) and not job.anyFailed.load:
      done += int64(c.count)
      result.failed += c.failed
      stackChunk(acc, c, binning, dtMs, result.stacked, result.skipped)
      if progress != nil and done < total and not progress(done, total):
        raiseCancelled("bruteStack cancelled")
  for s in states:
    if s.failed:
      var msg = newString(s.msgLen)
      for i in 0 ..< s.msgLen: msg[i] = s.msg[i]
      raise newException(ValueError, "bruteStack: " & msg)
  result.binTraces = newSeq[int32](acc.bins)
  result.cmpX = newSeq[float64](acc.bins)
  result.cmpY = newSeq[float64](acc.bins)
  for bin in 0 ..< acc.bins:
    result.binTraces[bin] = int32(acc.tracesAt(bin))
    result.cmpX[bin] = acc.cmpXAt(bin)
    result.cmpY[bin] = acc.cmpYAt(bin)
  acc.finishInto(result.data, result.fold, opts.norm)
  if progress != nil: discard progress(total, total)

proc checkFunction(velocity: VelocityFunction) =
  let v = velocity.vrmsAt(0)
  if v.isNaN or v <= 0:
    raise newException(ValueError, "the velocity function must come from " &
      "constantVelocity, gradientVelocity, rmsVelocity or intervalVelocity")

proc bruteStack*(f: SeismicFile, T: typedesc[SampleType], traces: Slice[int],
    binning: Binning, velocity: VelocityField, opts = initBruteStackOptions(),
    progress: Progress = nil): StackResult[T] =
  ## A brute stack of the traces `traces` into the bins of `binning`, with
  ## NMO velocities from `velocity` at each bin (a 2D field by the bin's
  ## CDP, which a 2D binning gives; a 3D field by its inline and
  ## crossline). Blocks until done; see the module documentation. Raises
  ## `ValueError` when the file has no keys, a trace is outside the file,
  ## the live traces have no single sample interval, the field does not
  ## suit the binning or ends before the stack's last sample
  ## (`field.tMaxMs`), an option is invalid or the flow does not compile,
  ## and `SegCancelledError` when `progress` cancels.
  checkSuits(velocity, binning)
  var job = Job[T](velKind: velField, field: addr velocity)
  runStack[T](f, TraceSel(span: traces), binning, job, opts, progress)

proc bruteStack*(f: SeismicFile, T: typedesc[SampleType], traces: Slice[int],
    binning: Binning, velocity: VelocityFunction,
    opts = initBruteStackOptions(), progress: Progress = nil): StackResult[T] =
  ## A brute stack with one velocity function for every bin; see the field
  ## form.
  checkFunction(velocity)
  var job = Job[T](velKind: velFunction, fun: addr velocity)
  runStack[T](f, TraceSel(span: traces), binning, job, opts, progress)

proc bruteStack*(f: SeismicFile, T: typedesc[SampleType],
    traces: openArray[int], binning: Binning, velocity: VelocityField,
    opts = initBruteStackOptions(), progress: Progress = nil): StackResult[T] =
  ## A brute stack of a list of traces (a map selection, one component, one
  ## channel set), read and stacked in list order: `traces[i]` is a trace
  ## index, and a trace listed twice is stacked twice. `progress` counts
  ## list positions. The list is copied. Otherwise as the range form.
  checkSuits(velocity, binning)
  var job = Job[T](velKind: velField, field: addr velocity)
  runStack[T](f, TraceSel(listed: true, list: @traces), binning, job, opts,
    progress)

proc bruteStack*(f: SeismicFile, T: typedesc[SampleType],
    traces: openArray[int], binning: Binning, velocity: VelocityFunction,
    opts = initBruteStackOptions(), progress: Progress = nil): StackResult[T] =
  ## A brute stack of a list of traces with one velocity function; see the
  ## other list form.
  checkFunction(velocity)
  var job = Job[T](velKind: velFunction, fun: addr velocity)
  runStack[T](f, TraceSel(listed: true, list: @traces), binning, job, opts,
    progress)

proc bruteStack*(f: SeismicFile, T: typedesc[SampleType], binning: Binning,
    velocity: VelocityField, opts = initBruteStackOptions(),
    progress: Progress = nil): StackResult[T] =
  ## A brute stack of the whole file; see the form with `traces`.
  bruteStack(f, T, 0 ..< f.len, binning, velocity, opts, progress)

proc bruteStack*(f: SeismicFile, T: typedesc[SampleType], binning: Binning,
    velocity: VelocityFunction, opts = initBruteStackOptions(),
    progress: Progress = nil): StackResult[T] =
  ## A brute stack of the whole file with one velocity function.
  bruteStack(f, T, 0 ..< f.len, binning, velocity, opts, progress)

proc checkBin[T](res: StackResult[T], bin: int) =
  if res.data.rank != 2 or res.fold.shape != res.data.shape:
    raise newException(ValueError, "an inconsistent stack: data and fold differ in shape")
  if bin notin 0 ..< res.data.shape[0]:
    raise newException(ValueError, "bin " & $bin & " is outside 0 ..< " & $res.data.shape[0])

proc maxOf(x: ptr UncheckedArray[int32], n: int): int32 {.noinline, raises: [].} =
  ## The largest of `x[0 ..< n]`, 0 at least (an integer max: it vectorizes).
  {.push boundChecks: off.}
  for j in 0 ..< n: result = max(result, x[j])
  {.pop.}

proc maxFold*[T](res: StackResult[T], bin: int): int32 =
  ## The largest per-sample fold of bin `bin`; 0 for a bin nothing was
  ## stacked into. Raises `ValueError` for a bin outside the stack or an
  ## inconsistent result.
  checkBin(res, bin)
  let n = res.fold.shape[1]
  if res.fold.is_C_contiguous:
    # The fold's own buffer: through `Tensor.[]` this took 9.2 ms for 1396
    # bins of 1501 samples in a release build.
    let fold = cast[ptr UncheckedArray[int32]](res.fold.unsafe_raw_offset(
        aligned = false))
    result = maxOf(cast[ptr UncheckedArray[int32]](addr fold[bin * n]), n)
  else:
    for j in 0 ..< n: result = max(result, res.fold[bin, j])

proc keysOf[T](res: StackResult[T], bin: int, fold: int32): TraceKeys =
  # The keys of bin `bin`, whose largest fold is `fold`.
  let loc = res.binning.location(bin)
  result = initTraceKeys()
  result.ensemble = if loc.cdp.isNaN: float64(bin + 1) else: loc.cdp
  result.inline = loc.inline
  result.crossline = loc.crossline
  # A bin with a geometry is at its centre; a bin of header numbers has
  # none, and is where its traces' midpoints are on average.
  if loc.x.isNaN and bin < res.cmpX.len and bin < res.cmpY.len:
    result.cmpX = res.cmpX[bin]
    result.cmpY = res.cmpY[bin]
  else:
    result.cmpX = loc.x
    result.cmpY = loc.y
  result.offset = 0
  result.t0Ms = res.startMs
  # The input's own microseconds: 1001 µs is 1.001 ms, and 1.001 · 1000 is
  # 1000.9999999999999.
  result.intervalUs = if res.intervalUs > 0: res.intervalUs else: res.dtMs * 1000
  result.status = if fold > 0: tsLive else: tsDead

func overridesOf(fold: int32): array[1, HeaderOverride] =
  [headerOverride("trace.numHorzStacked", min(fold, int32(high(int16))))]

proc stackKeys*[T](res: StackResult[T], bin: int): TraceKeys =
  ## The keys of bin `bin`'s stacked trace (decision D14: processSEG
  ## describes its results, parseSEG writes them): the CDP number (or the
  ## 1-based bin index for 3D bins), inline and crossline, as the midpoint
  ## the bin centre (line and grid binnings) or the mean midpoint of the
  ## bin's traces (`cmpX`, `cmpY`: header CDP and inline/crossline
  ## binnings; NaN for an empty bin), offset 0, `startMs` as `t0Ms`,
  ## `intervalUs` as the
  ## interval (`dtMs · 1000` when it is 0), and `tsDead` for a bin with no
  ## fold, else `tsLive`; every other key unknown. Pass them, with
  ## `stackOverrides`, to parseSEG's
  ## `writeTrace(w, keys, samples, overrides)` to write a stack your own way
  ## (other options, extra overrides, a selection of bins); `writeStack`
  ## does exactly that for every bin. Raises `ValueError` as `maxFold`.
  res.keysOf(bin, res.maxFold(bin))

proc stackOverrides*[T](res: StackResult[T], bin: int): array[1, HeaderOverride] =
  ## The SEG-Y header fields of bin `bin` that `TraceKeys` cannot hold: the
  ## fold, as the largest per-sample fold in `numHorzStacked` (bytes 33-34,
  ## clamped to int16). Raises `ValueError` as `maxFold`.
  overridesOf(res.maxFold(bin))

proc writeStack*[T](path: string, res: StackResult[T],
    opts = SegyOutputOptions(traceSorting: 4)): DiagLog =
  ## Writes a stack as SEG-Y at `path`, one trace per bin in bin order, for
  ## a stack of any input standard (parseSEG's writer of traces from
  ## `TraceKeys`; `opts` sets the revision, byte order, sample format and
  ## textual header). Each trace holds its bin's location (CDP number, or
  ## the 1-based bin index for 3D bins; inline and crossline; the bin
  ## centre, or for bins of header numbers the mean midpoint of their
  ## traces, as the CDP X/Y), offset 0, `startMs` as the delay, the interval
  ## (`intervalUs`, exact for the input's whole microseconds), and
  ## as the fold (`numHorzStacked`, bytes 33-34) the largest per-sample fold;
  ## a bin with no fold is a dead trace (code 2). The binary header's trace
  ## sorting is `opts.traceSorting`, and 4 (CDP stack) when that is 0 (so
  ## `SegyOutputOptions(revision: ...)` still says stack). `openSeismic`
  ## reads the keys back. Each trace is `stackKeys` with `stackOverrides`.
  ##
  ## Returns the writer's diagnostics: a failed rename of the finished
  ## file into place is a fatal entry there, not a raise (check
  ## `isFatal`). Raises `ValueError` for `opts` the writer refuses or an
  ## empty stack, `SegFormatError` for a value the sample format cannot
  ## hold, `IOError` when the output cannot be created or written; no
  ## output is left on a raise.
  if res.data.rank != 2 or res.data.size == 0 or res.fold.shape != res.data.shape:
    raise newException(ValueError, "writeStack: an empty or inconsistent stack")
  let bins = res.data.shape[0]
  let data = rowBlock(res.data)
  var o = opts
  if o.traceSorting == 0: o.traceSorting = 4
  var w = openSeismicWriter(path, o)
  for bin in 0 ..< bins:
    let fold = res.maxFold(bin) # once per bin, for the keys and the header
    w.writeTrace(res.keysOf(bin, fold), data.row(bin), overrides = overridesOf(fold))
  w.close()
