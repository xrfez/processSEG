## Batch jobs over a whole file: `bruteStack`, and `writeStack` to save a
## stack as SEG-Y.
##
## A brute stack reads the traces through parseSEG's prefetcher and, on its
## decode workers (decision D13: the prefetcher's `transform` hook, with
## `uniform = true`), runs on each chunk while it is in cache:
##
## 1. datum statics (optional, `DatumStatics`);
## 2. the pre-stack flow, a `Pipeline` (mutes, filters, gain);
## 3. NMO with a stretch mute, onto the stack's time axis;
## 4. stacking into the trace's bin, with a per-sample fold (`addLive`).
##
## ```nim
## var f = openSeismic("line.sgy", keys = true)
## var opts = initBruteStackOptions(flow = initPipeline([bandPassOp(
##   Corners(f1: 4, f2: 8, f3: 60, f4: 80))]))
## opts.workers = 4
## let res = bruteStack(f, float32, cdpBinning(firstCdp = 1, count = 800),
##   velocities, opts, progress = proc (done, total: int64): bool = true)
## let part = bruteStack(f, float32, 0 ..< 5000, binning, velocities, opts)
## # res.data: [bins, samples]; res.fold; res.binning.location(bin)
## ```
##
## **Velocities** are read at each bin's location (`Binning.location`): a
## 2D field by the bin's CDP number, a 3D field by its inline and
## crossline numbers. A `VelocityFunction` serves every bin.
##
## **Traces.** Only live traces are stacked (`ChunkMeta.isLive`). A live
## trace is skipped, and counted in `StackResult.skipped`, when it falls
## outside every bin, has a NaN offset or `t0Ms`, a NaN static (with
## `useStatics`), no sample interval in its headers, or a NaN or infinite
## sample, or when the stretch mute leaves nothing of it. Traces that could
## not be decoded are counted in `failed` (their diagnostics are dropped:
## read them with parseSEG if needed). The live traces with an interval
## must all have the same one (a mixed SEG-D record: select the traces of
## one channel set), and the stack and the flow are sized by the longest
## of them. The traces need not be sorted.
##
## **Reproducibility.** With several workers, which worker stacks which
## chunk, and so the order of the float sums, changes from run to run: two
## runs agree to rounding, not bit for bit. One worker is deterministic.
##
## **Cost.** A `VelocityField` is read for a bin whenever a worker moves to
## another bin (the last bin's column is cached), so CDP-sorted input reads
## it once per CDP and shot-sorted input about once per trace
## (`samples` interpolations each).
##
## **Progress and cancel.** `progress(done, total)` is called on the
## calling thread after every chunk with the **traces** done and the total
## (parseSEG's `Progress` type, which elsewhere counts bytes), and
## once at the end with `done == total`; returning false stops the workers
## and raises parseSEG's `SegCancelledError`.
##
## **Memory.** Each worker owns a full stack accumulator, merged at the
## end, so the job holds `workers` accumulators, `bins × samples ×
## (sizeof(T) + 4)` bytes each, plus the result (the same again), the
## prefetch ring (`maxRingBytes`, at least `workers + 1` chunks, plus the
## chunk the consumer holds) and each worker's flow scratch
## (`bruteStackMemory` estimates the total). Use fewer workers for a large
## 3D stack.

import std/[atomics, math]
import parseSEG
import core/[rows, types, workspace]
import pipeline
import stack/[binning, nmo, stack]
import statics/elevation
import velocity/model

type
  BruteStackOptions* = object
    ## The processing and the job's resources. Start from
    ## `initBruteStackOptions` and change fields.
    flow*: Pipeline ## Pre-stack flow, after statics, before NMO. A
                      ## `perPanel` normalization works per prefetch chunk
                      ## (`chunkTraces`, cut where the interval changes), so its
                      ## factors depend on the chunking: prefer `perTrace` here.
    useStatics*: bool ## Apply `statics` first.
    statics*: DatumStatics ## Datum statics, when `useStatics`.
    staticsInterp*: ShiftInterp ## Interpolation of the statics shifts.
    nmo*: NmoParams ## NMO interpolation and stretch mute.
    norm*: FoldNorm ## Normalization of the stack.
    samples*: int ## Stack samples; 0: the longest input trace's.
    startMs*: float64 ## Time of the stack's sample 0.
    workers*: int ## Prefetch (and processing) threads.
    chunkTraces*: int ## Traces per prefetch chunk.
    maxRingBytes*: int ## Prefetch ring cap (parseSEG's default: 4 MiB).

  StackResult*[T: SampleType] = object
    ## A brute stack: `data[bin, j]` at time `startMs + j·dtMs`.
    data*: Tensor[T]     ## `[bins, samples]`, normalized.
    fold*: Tensor[int32] ## `[bins, samples]`, the per-sample fold.
    binning*: Binning    ## The bins (`location` gives each bin's numbers).
    dtMs*: float64       ## Sample interval, in ms.
    startMs*: float64    ## Time of sample 0, in ms.
    stacked*: int        ## Traces stacked.
    skipped*: int        ## Live traces not stacked (see the module documentation).
    failed*: int         ## Traces that could not be decoded.

proc initBruteStackOptions*(flow = initPipeline(), nmo = initNmoParams(),
    norm = fnFold, workers = 1): BruteStackOptions =
  ## Options with the flow `flow`, NMO `nmo`, normalization `norm` and
  ## `workers` threads; no statics, the input's length from 0 ms, chunks of
  ## 256 traces.
  BruteStackOptions(flow: flow, nmo: nmo, norm: norm, workers: workers,
    chunkTraces: 256, maxRingBytes: 4 shl 20, staticsInterp: siLinear)

func bruteStackMemory*(T: typedesc[SampleType], bins, samples, inSamples,
    workers: int, chunkTraces = 256, maxRingBytes = 4 shl 20): int =
  ## An estimate of the memory a `bruteStack` holds, in bytes: `workers`
  ## accumulators and the result (`stackMemory` each), the prefetch ring
  ## and the consumer's chunk, and each worker's scratch (taken as 16 lane
  ## rows of a 4× padded trace, which bounds the FFT filters' scratch, plus
  ## 1 MiB for a percentile histogram; a flow of several long filters may
  ## need more: `plan.workspaceSize` is exact). Raises `OverflowDefect` for
  ## sizes beyond `int`.
  let stacks = (workers + 1) * stackMemory(T, bins, samples)
  let chunk = chunkTraces * inSamples * sizeof(T)
  let ring = max(maxRingBytes, (workers + 1) * chunk) + chunk
  let scratch = workers * (16 * 4 * inSamples * 8 + (1 shl 20) + 2 * samples * 8)
  stacks + ring + scratch

type
  VelKind = enum velFunction, velField

  WorkerState[T] = object
    ws: Workspace
    acc: StackAccumulator[T]
    vel: seq[float64]
    row: seq[T]
    lastBin: int
    stacked, skipped: int
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
    states: ptr UncheckedArray[WorkerState[T]]
    anyFailed: Atomic[bool] # a worker failed: the consumer stops

proc note[T](job: ptr Job[T], s: var WorkerState[T], msg: string) {.raises: [].} =
  job.anyFailed.store(true)
  if s.failed: return
  s.failed = true
  s.msgLen = min(msg.len, s.msg.len)
  for i in 0 ..< s.msgLen: s.msg[i] = msg[i]

proc velocityAt[T](job: ptr Job[T], s: var WorkerState[T], bin: int) {.
    raises: [ValueError].} =
  # Fills s.vel with the bin's RMS velocities on the stack's axis.
  if bin == s.lastBin or (job.velKind == velFunction and s.lastBin >= 0):
    return
  s.lastBin = bin
  case job.velKind
  of velFunction:
    job.fun[].vrmsInto(job.startMs, job.dtMs, s.vel)
  of velField:
    let loc = job.binning[].location(bin)
    case job.field[].kind
    of fk2D: job.field[].vrmsInto(loc.cdp, job.startMs, job.dtMs, s.vel)
    of fk3D: job.field[].vrmsInto(loc.inline, loc.crossline, job.startMs,
        job.dtMs, s.vel)

proc stackChunk[T](worker: int, samples: ptr UncheckedArray[T], rows, cols: int,
    traces: TraceRows, failed: openArray[int], user: pointer) {.nimcall,
    gcsafe, raises: [].} =
  let job = cast[ptr Job[T]](user)
  template s: untyped = job.states[worker]
  if s.failed or rows == 0: return
  try:
    let meta = initChunkMeta(traces, job.dtMs, 0, failed)
    let b = rowBlock(samples.toOpenArray(0, rows * cols - 1), rows, cols)
    if job.useStatics:
      applyStatics(b, meta, job.statics, job.staticsInterp) # NaN: untouched
    {.cast(gcsafe).}:
      job.plan[].apply(b, meta, s.ws)
    for r in 0 ..< rows:
      if not meta.isLive(r): continue
      let k = meta.keys(r)
      let bin = job.binning[].binOf(k)
      if bin < 0 or k.offset.isNaN or k.t0Ms.isNaN or
          k.intervalUs != job.intervalUs or
          (job.useStatics and job.statics.staticMs(k).isNaN):
        inc s.skipped
        continue
      {.cast(gcsafe).}:
        velocityAt(job, s, bin)
      let n = meta.liveSamples(r, cols)
      discard nmoInto(b.row(r, n), meta.sampleTimeMs(r, 0), s.row, job.startMs,
        job.dtMs, k.offset, s.vel, job.nmo)
      if liveSpan(s.row).len > 0 and s.acc.addLive(bin, s.row): inc s.stacked
      else: inc s.skipped
  except CatchableError as e:
    job.note(s, e.msg)

proc liveTraces(f: SeismicFile, traces: Slice[int]): (float64, int) =
  # The one sample interval of the live traces (in µs) and their longest
  # length; traces without an interval are left out (and skipped later).
  let rows = f.rowsOf(traces)
  var us = NaN
  var longest = 0
  for i in 0 ..< rows.len:
    let k = rows.keysAt(i)
    if k.status notin {tsLive, tsUnknown}: continue
    let x = k.intervalUs
    if not (x > 0 and x < Inf): continue
    if us.isNaN: us = x
    elif x != us:
      raise newException(ValueError, "the live traces have different sample " &
        "intervals (" & $us & " and " & $x & " µs, trace " & $(traces.a + i) &
        "): stack the traces of one interval (one channel set)")
    longest = max(longest, rows.numSamples(i))
  if us.isNaN:
    raise newException(ValueError, "no live traces with a sample interval to stack")
  (us, longest)

proc checkOptions(o: BruteStackOptions) =
  if o.workers <= 0 or o.chunkTraces <= 0 or o.maxRingBytes <= 0:
    raise newException(ValueError, "workers, chunkTraces and maxRingBytes " &
      "must be positive")
  if o.samples < 0:
    raise newException(ValueError, "samples must be ≥ 0, got " & $o.samples)
  if o.startMs.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "startMs must be finite, got " & $o.startMs)
  if o.nmo.maxStretch.isNaN or o.nmo.maxStretch <= 0:
    raise newException(ValueError, "opts.nmo must come from initNmoParams")
  if o.useStatics and not (o.statics.vReplacement > 0):
    raise newException(ValueError, "opts.statics must come from datumStatics")

proc runStack[T](f: SeismicFile, traces: Slice[int], binning: Binning,
    job: var Job[T], opts: BruteStackOptions, progress: Progress): StackResult[T] =
  if not f.hasKeys:
    raise newException(ValueError, "bruteStack needs stored keys: open the " &
      "file with keys = true (or call ensureKeys)")
  checkOptions(opts)
  let (intervalUs, inSamples) = liveTraces(f, traces)
  let dtMs = intervalUs / 1000
  let samples = if opts.samples > 0: opts.samples else: inSamples
  let plan = opts.flow.compile(T, dtMs, max(inSamples, 1))
  # Everything the workers use is allocated here, on the calling thread.
  var states = newSeq[WorkerState[T]](opts.workers)
  for s in states.mitems:
    s.ws = initWorkspace()
    s.ws.reserve(plan)
    s.acc = initStackAccumulator[T](binning.len, samples, dtMs, opts.startMs)
    s.vel = newSeq[float64](samples)
    s.row = newSeq[T](samples)
    s.lastBin = -1
  job.plan = unsafeAddr plan
  job.binning = unsafeAddr binning
  job.useStatics = opts.useStatics
  job.statics = opts.statics
  job.staticsInterp = opts.staticsInterp
  job.nmo = opts.nmo
  job.dtMs = dtMs
  job.intervalUs = intervalUs
  job.startMs = opts.startMs
  job.states = cast[ptr UncheckedArray[WorkerState[T]]](addr states[0])
  let total = int64(traces.len)
  var done = 0'i64
  var failed = 0
  block run:
    var p = newPrefetcher[T](f, traces, chunkTraces = opts.chunkTraces,
      depth = opts.workers + 2, cols = inSamples, workers = opts.workers,
      maxRingBytes = opts.maxRingBytes, uniform = true,
      transform = stackChunk[T], user = addr job)
    defer: p.close()
    var c: PrefetchChunk[T]
    while p.nextInto(c):
      done += int64(c.count)
      failed += c.failed
      if job.anyFailed.load: break
      if progress != nil and done < total and not progress(done, total):
        raiseCancelled("bruteStack cancelled")
  for s in states:
    if s.failed:
      var msg = newString(s.msgLen)
      for i in 0 ..< s.msgLen: msg[i] = s.msg[i]
      raise newException(ValueError, "bruteStack: " & msg)
  for i in 1 ..< states.len:
    states[0].acc.merge(states[i].acc)
  result = StackResult[T](binning: binning, dtMs: dtMs, startMs: opts.startMs,
    data: newTensor[T](binning.len, samples),
    fold: newTensor[int32](binning.len, samples))
  states[0].acc.stackInto(result.data, opts.norm)
  states[0].acc.foldInto(result.fold)
  result.failed = failed
  for s in states:
    result.stacked += s.stacked
    result.skipped += s.skipped
  if progress != nil: discard progress(total, total)

proc checkField(binning: Binning, velocity: VelocityField) =
  if velocity.len == 0:
    raise newException(ValueError, "the velocity field is empty")
  if velocity.kind == fk2D and binning.is3D:
    raise newException(ValueError, "a 2D velocity field needs a 2D binning " &
      "(cdpBinning or lineBinning)")
  if velocity.kind == fk3D and not binning.is3D:
    raise newException(ValueError, "a 3D velocity field needs a 3D binning " &
      "(gridBinning or inlineCrosslineBinning)")

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
  ## `ValueError` when the file has no keys, the live traces have no single
  ## sample interval, the field does not suit the binning, an option is
  ## invalid or the flow does not compile, and `SegCancelledError` when
  ## `progress` cancels.
  checkField(binning, velocity)
  var job = Job[T](velKind: velField, field: unsafeAddr velocity)
  runStack[T](f, traces, binning, job, opts, progress)

proc bruteStack*(f: SeismicFile, T: typedesc[SampleType], traces: Slice[int],
    binning: Binning, velocity: VelocityFunction,
    opts = initBruteStackOptions(), progress: Progress = nil): StackResult[T] =
  ## A brute stack with one velocity function for every bin; see the field
  ## form.
  checkFunction(velocity)
  var job = Job[T](velKind: velFunction, fun: unsafeAddr velocity)
  runStack[T](f, traces, binning, job, opts, progress)

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

proc stackKeys[T](res: StackResult[T], bin: int, live: bool): TraceKeys =
  # A stacked trace's keys: the bin's numbers and centre, zero offset.
  let loc = res.binning.location(bin)
  result = initTraceKeys()
  result.ensemble = if loc.cdp.isNaN: float64(bin + 1) else: loc.cdp
  result.inline = loc.inline
  result.crossline = loc.crossline
  result.cmpX = loc.x
  result.cmpY = loc.y
  result.offset = 0
  result.t0Ms = res.startMs
  result.intervalUs = res.dtMs * 1000
  result.status = if live: tsLive else: tsDead

proc writeStack*[T](path: string, res: StackResult[T],
    opts = SegyOutputOptions(traceSorting: 4)): DiagLog =
  ## Writes a stack as SEG-Y at `path`, one trace per bin in bin order, for
  ## a stack of any input standard (parseSEG's writer of traces from
  ## `TraceKeys`; `opts` sets the revision, byte order, sample format and
  ## textual header). Each trace holds its bin's location (CDP number, or
  ## the 1-based bin index for 3D bins; inline and crossline; the bin
  ## centre as the CDP X/Y), offset 0, `startMs` as the delay, `dtMs`, and
  ## as the fold (`numHorzStacked`, bytes 33-34) the largest per-sample fold;
  ## a bin with no fold is a dead trace (code 2). The binary header's trace
  ## sorting is `opts.traceSorting`, and 4 (CDP stack) when that is 0 (so
  ## `SegyOutputOptions(revision: ...)` still says stack). `openSeismic`
  ## reads the keys back.
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
  let n = res.data.shape[1]
  let data = rowBlock(res.data)
  var o = opts
  if o.traceSorting == 0: o.traceSorting = 4
  var w = openSeismicWriter(path, o)
  for bin in 0 ..< bins:
    var most = 0'i32
    for j in 0 ..< n: most = max(most, res.fold[bin, j])
    w.writeTrace(stackKeys(res, bin, most > 0), data.row(bin),
      overrides = [headerOverride("trace.numHorzStacked", min(most, int32(high(int16))))])
  w.close()
