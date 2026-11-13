## Brute stack (batch): a synthetic pre-stack SEG-Y line (split-spread
## shots over topography, signed offsets, header CDPs) stacked with NMO,
## statics and a flow; the fold per bin, the peaks of the reflectors, the
## same bits for 1 to 8 workers and as a stack by hand in file order (also
## onto a longer time axis), a trace subset, a 2D field, progress and
## cancel, errors, and parseSEG's demo files (post-stack SEG-Y reproduced
## exactly, SEG-D and SEG-2 runs). Memory bound arithmetic.

import std/[assertions, math, options, os, random, strutils]
import processSEG
import parseSEG/io/diskio
from parseSEG/model/traceindex import mindex # the tests corrupt the index
import parseSEG/segy/[samples as segysamples, textheader, binheader,
  traceheader, writer as segywriter]
import utils/[demo, synth]

const
  dt = 2.0
  ns = 1001
  shots = 10
  t0s = [400.0, 900, 1500]

let vel = gradientVelocity(1800, 500)

type Line = object
  path: string
  traces: int
  foldOf: seq[int] # traces per CDP 1 ..
  firstCdp: float64

proc writeLine(path: string, topo: bool): Line =
  ## Shots every 50 m along x into a split spread of 2 × 12 channels at
  ## 25 m (near offset 12.5 m): CDPs every 12.5 m, numbered from 1. With
  ## `topo`, the stations sit up to ±20 m about the datum of 100 m and the
  ## events carry the elevation static at 2000 m/s.
  var f = SegyFile(byteOrder: boBig, format: some(sfFloat32))
  f.text = initTextHeader(teEbcdic)
  f.binary = SegyBinaryHeader(sampleInterval: uint16(dt * 1000),
    samplesPerTrace: ns, formatCode: 5, measurementSystem: 1, segyMajor: 1,
    fixedLengthTraces: 1)
  let sink = openDiskSink(path)
  result.path = path
  var rng = initRand(17)
  var x = newSeq[float32](ns)
  let rel = splitSpread(12, 25, 12.5)
  var maxCdp = 0
  try:
    var w = openSegyWriter(sink, f, initSegyWriteOptions(f))
    for s in 0..<shots:
      let srcX = 10_000 + 50 * float64(s)
      let srcZ = if topo: 100 + float64(rng.rand(-20..20)) else: 100.0
      for i, r in rel:
        let rcvX = srcX + r
        let rcvZ = if topo: 100 + float64(rng.rand(-20..20)) else: 100.0
        let cdp = int((srcX + r / 2 - 10_006.25 + 300) / 12.5) + 1
        maxCdp = max(maxCdp, cdp)
        if result.foldOf.len < cdp + 1: result.foldOf.setLen(cdp + 1)
        inc result.foldOf[cdp]
        var k = initTraceKeys()
        k.offset = r
        k.srcZ = srcZ
        k.rcvZ = rcvZ
        let shift = elevationShiftMs(k, 100, 2000)
        for j in 0..<ns: x[j] = 0
        for t0 in t0s:
          x.addRicker(dt, hyperbolicMs(t0, vel.vrmsAt(t0), r) + shift, 20)
        var t = SegyTrace(index: result.traces, numSamples: ns,
          sampleInterval: dt * 1000)
        t.header = SegyTraceHeader(traceSeqLine: int32(result.traces + 1),
          traceSeqFile: int32(result.traces + 1), fieldRecord: int32(s + 1),
          traceNumberInField: int32(i + 1), ensembleNumber: int32(cdp),
          traceIdCode: 1, offset: int32(r), receiverElevation: int32(rcvZ),
          sourceSurfaceElevation: int32(srcZ), elevationScalar: 1,
          coordinateScalar: 1, sourceX: int32(srcX), groupX: int32(rcvX),
          coordinateUnits: 1, numSamples: ns, sampleInterval: uint16(dt * 1000))
        t.samples = newSeq[byte](ns * 4)
        encodeSamples(sfFloat32, x, t.samples, boBig)
        w.writeTrace(t)
        inc result.traces
    w.close()
    doAssert w.log.len == 0, $w.log[0]
  finally:
    sink.close()
  result.foldOf.setLen(maxCdp + 1)

let dir = getTempDir() / "processseg_tbatch"
createDir(dir)
let flat = writeLine(dir / "flat.sgy", topo = false)
let topo = writeLine(dir / "topo.sgy", topo = true)
# CDP numbers start at 1; the shots cover CDPs 1 .. foldOf.high.
let cdps = flat.foldOf.len - 1
let binning = cdpBinning(firstCdp = 1, count = cdps)

proc peakNear(res: StackResult[float32], bin: int, t0: float64): (float64, float32) =
  let c = int(t0 / dt)
  var best = c - 5
  for j in c - 5 .. c + 5:
    if res.data[bin, j] > res.data[bin, best]: best = j
  (float64(best) * dt, res.data[bin, best])

block stack_of_a_synthetic_line:
  var f = openSeismic(flat.path, keys = true)
  doAssert f.len == flat.traces
  let opts = initBruteStackOptions(nmo = initNmoParams(siCubic,
      maxStretch = 0.5))
  let res = bruteStack(f, float32, binning, vel, opts)
  doAssert res.data.shape == [cdps, ns] and res.fold.shape == [cdps, ns]
  doAssert res.dtMs == dt and res.startMs == 0
  doAssert res.stacked == flat.traces and res.skipped == 0
  for b in 0..<cdps:
    let want = flat.foldOf[b + 1]
    # Deep, below any stretch mute, every trace of the CDP counts.
    doAssert res.fold[b, int(1500 / dt)] == want, $b
    if want >= 4:
      for t0 in t0s:
        let (t, a) = res.peakNear(b, t0)
        doAssert abs(t - t0) <= dt, $b & " " & $t0 & " " & $t
        doAssert abs(a - 1) <= 0.03, $b & " " & $t0 & " " & $a
  doAssert res.binning.location(0).cdp == 1

block same_bits_for_any_worker_count:
  # The workers correct the traces; the calling thread stacks them in file
  # order: the sums, and so the stack, do not depend on how many workers
  # there are or on which took which chunk (15 chunks here). With an
  # accumulator per worker they agreed to rounding only.
  for (path, withFlow) in [(flat.path, false), (topo.path, true)]:
    var f = openSeismic(path, keys = true)
    var opts = initBruteStackOptions()
    opts.chunkTraces = 16
    if withFlow:
      opts.flow = initPipeline([bandPassOp(Corners(f1: 3, f2: 6, f3: 60,
          f4: 80)), agcOp(300)])
      opts.statics = some(datumStatics(100, 2000))
      opts.nmo = initNmoParams(siCubic, maxStretch = 0.5, taperMs = 20)
      opts.norm = fnSqrtFold
    let field = initVelocityField([1.0, float64(cdps) / 2, float64(cdps)], [vel,
        gradientVelocity(1900, 500), vel], dtMs = 4, tMaxMs = 2400)
    let one = bruteStack(f, float32, binning, vel, opts)
    let oneField = bruteStack(f, float32, binning, field, opts)
    doAssert one.stacked == flat.traces and one.data != oneField.data
    for workers in [2, 4, 8]:
      opts.workers = workers
      for run in 0 ..< 2:
        let res = bruteStack(f, float32, binning, vel, opts)
        doAssert res.data == one.data and res.fold == one.fold, $workers
        doAssert res.stacked == one.stacked and res.skipped == one.skipped
      let byField = bruteStack(f, float32, binning, field, opts)
      doAssert byField.data == oneField.data and byField.fold == oneField.fold

block velocity_cache_never_shows:
  # Each worker caches the field's columns of the bins it met last. The
  # line is shot-sorted (a shot's 24 traces fall in 24 CDPs), so a cache of
  # one or three columns is overwritten all the time and the default one
  # holds every bin: the stack is the same bit for bit.
  var f = openSeismic(flat.path, keys = true)
  let field = initVelocityField([1.0, float64(cdps) / 2, float64(cdps)], [vel,
      gradientVelocity(1900, 500), vel], dtMs = 4, tMaxMs = 2400)
  var opts = initBruteStackOptions()
  opts.chunkTraces = 16
  doAssert opts.velocityCacheBytes == 8 shl 20
  let want = bruteStack(f, float32, binning, field, opts)
  for cacheBytes in [0, 3 * ns * 8, 7 * ns * 8 + 5]:
    for workers in [1, 3]:
      opts.workers = workers
      opts.velocityCacheBytes = cacheBytes
      let res = bruteStack(f, float32, binning, field, opts)
      doAssert res.data == want.data and res.fold == want.fold, $cacheBytes
  opts.velocityCacheBytes = -1
  doAssertRaises ValueError: discard bruteStack(f, float32, binning, field, opts)

proc byHand(f: var SeismicFile, flow: Pipeline, chunkTraces, samples: int,
    startMs: float64, binning: Binning): Tensor[float32] =
  ## The stack of the whole file with the public kernels, trace by trace in
  ## file order: datum statics, the flow in blocks of `chunkTraces` rows,
  ## linear NMO onto the stack's axis, addLive.
  let n = f.len
  var raw = newTensor[float32](n, ns)
  discard f.readTracesInto(0 ..< n, raw)
  let meta = chunkMeta(f, 0 ..< n)
  raw.applyStatics(meta, datumStatics(100, 2000))
  let plan = flow.compile(float32, dt, ns)
  var ws = initWorkspace()
  ws.reserve(plan)
  let b = rowBlock(raw)
  var a = 0
  while a < n:
    let rows = a ..< min(n, a + chunkTraces)
    plan.apply(b.sub(rows), meta.sub(rows), ws)
    a += chunkTraces
  var velocities = newSeq[float64](samples)
  vel.vrmsInto(startMs, dt, velocities)
  var row = newSeq[float32](samples)
  var acc = initStackAccumulator[float32](binning.len, samples, dt, startMs)
  for r in 0 ..< n:
    discard nmoInto(b.row(r), meta.sampleTimeMs(r, 0), row, startMs, dt,
      meta.keys(r).offset, velocities, initNmoParams())
    doAssert acc.addLive(binning.binOf(meta.keys(r)), row)
  result = newTensor[float32](binning.len, samples)
  acc.stackInto(result)

block equals_the_stack_by_hand:
  # bruteStack is the stack of the traces in file order, bit for bit: on
  # the input's time axis, and on a longer one that starts earlier (the
  # chunk rows are then wider than the traces the flow sees).
  var f = openSeismic(topo.path, keys = true)
  let flow = initPipeline([bandPassOp(Corners(f1: 3, f2: 6, f3: 60, f4: 80))])
  for (samples, startMs) in [(ns, 0.0), (ns + 300, -100.0), (ns div 2, 200.0)]:
    let want = byHand(f, flow, 16, samples, startMs, binning)
    for workers in [1, 3]:
      var opts = initBruteStackOptions(flow = flow, workers = workers)
      opts.chunkTraces = 16
      opts.statics = some(datumStatics(100, 2000))
      opts.samples = samples
      opts.startMs = startMs
      let res = bruteStack(f, float32, binning, vel, opts)
      doAssert res.data.shape == [cdps, samples] and res.startMs == startMs
      doAssert res.stacked == topo.traces
      doAssert res.data == want, $samples & " " & $workers

block subset_and_field:
  var f = openSeismic(flat.path, keys = true)
  var opts = initBruteStackOptions(workers = 3)
  opts.chunkTraces = 32 # several chunks per worker
  let three = bruteStack(f, float32, binning, vel, opts)
  # The first three shots only.
  let part = bruteStack(f, float32, 0 ..< 72, binning, vel, opts)
  doAssert part.stacked == 72
  # A 2D field by CDP gives the same stack as its one function.
  let field = initVelocityField([1.0, float64(cdps)], [vel, vel], dtMs = 4,
    tMaxMs = 2400)
  let byField = bruteStack(f, float32, binning, field, opts)
  doAssert byField.fold == three.fold
  var worst = 0.0
  for b in 0..<cdps:
    for j in int(300 / dt)..<ns:
      worst = max(worst, abs(float64(byField.data[b, j] - three.data[b, j])))
  doAssert worst <= 5e-3, $worst

block fold_map_and_midpoints:
  # The result holds, per bin, the traces stacked into it and the mean of
  # their midpoints: a fold map without the fold tensor, and a place on
  # the map for bins by header numbers, which have no geometry of their
  # own (stackKeys had no cmpX/cmpY for them).
  var f = openSeismic(flat.path, keys = true)
  var opts = initBruteStackOptions(workers = 3)
  opts.chunkTraces = 16
  let res = bruteStack(f, float32, binning, vel, opts)
  doAssert res.binTraces.len == cdps and res.cmpX.len == cdps and res.cmpY.len == cdps
  var count = newSeq[int](cdps)
  var sumX, sumY = newSeq[float64](cdps)
  for t in 0 ..< f.len:
    let k = f.traceKeys(t)
    let b = binning.binOf(k)
    inc count[b]
    sumX[b] += k.cmpX
    sumY[b] += k.cmpY
  var total = 0
  for b in 0 ..< cdps:
    doAssert res.binTraces[b] == flat.foldOf[b + 1] and count[b] == flat.foldOf[b + 1]
    total += res.binTraces[b]
    if count[b] > 0:
      doAssert abs(res.cmpX[b] - sumX[b] / float64(count[b])) <= 1e-9
      doAssert abs(res.cmpY[b] - sumY[b] / float64(count[b])) <= 1e-9
      # Midpoints 12.5 m apart along the line (headers hold whole metres).
      doAssert abs(res.cmpX[b] - (10_006.25 - 300 + 12.5 * float64(b))) <= 0.5
      let k = res.stackKeys(b)
      doAssert k.cmpX == res.cmpX[b] and k.cmpY == res.cmpY[b] and k.ensemble ==
        float64(b + 1)
    else:
      doAssert res.cmpX[b].isNaN and res.stackKeys(b).cmpX.isNaN
  doAssert total == res.stacked
  let one = bruteStack(f, float32, binning, vel) # the same for any workers
  doAssert one.binTraces == res.binTraces
  for b in 0 ..< cdps:
    doAssert one.cmpX[b] == res.cmpX[b] or (one.cmpX[b].isNaN and res.cmpX[b].isNaN)
  # A bin past the line's CDPs: no trace, no midpoint.
  let wider = bruteStack(f, float32, cdpBinning(1, cdps + 3), vel)
  doAssert wider.binTraces[cdps + 2] == 0 and wider.cmpX[cdps + 2].isNaN
  doAssert wider.stackKeys(cdps + 2).cmpX.isNaN
  # Written and read back: the stacked traces carry their midpoints.
  let path = dir / "midpoints.sgy"
  doAssert not writeStack(path, res).isFatal
  var st = openSeismic(path, keys = true)
  var checked = 0
  for b in 0 ..< cdps:
    if res.binTraces[b] > 0:
      doAssert abs(st.traceKeys(b).cmpX - res.cmpX[b]) <= 0.01, $st.traceKeys(b).cmpX
      doAssert abs(st.traceKeys(b).cmpY - res.cmpY[b]) <= 0.01
      inc checked
    else:
      doAssert st.traceKeys(b).status == tsDead
  doAssert checked > cdps div 2
  st.close()
  # Bins with a geometry keep their centre as the midpoint.
  let line = lineBinning(10_006.25 - 300, 0, 90, 12.5, cdps)
  let byLine = bruteStack(f, float32, line, vel)
  doAssert byLine.binTraces[4] == 0 and byLine.binTraces[20] > 0
  for b in [4, 20]:
    doAssert byLine.stackKeys(b).cmpX == line.location(b).x
  doAssert abs(byLine.cmpX[20] - line.location(20).x) <= 0.5
  # A result built by hand, without them: NaN, as before.
  let hand = StackResult[float32](binning: binning, dtMs: 2,
    data: newTensor[float32](cdps, 3), fold: newTensor[int32](cdps, 3))
  doAssert hand.stackKeys(0).cmpX.isNaN

block trace_list:
  # A list of traces (a map selection, one side of the spread, one channel
  # set) is stacked in list order.
  var f = openSeismic(flat.path, keys = true)
  var opts = initBruteStackOptions(workers = 2)
  opts.chunkTraces = 16
  let whole = bruteStack(f, float32, binning, vel, opts)
  var all: seq[int]
  for t in 0 ..< f.len: all.add t
  let listed = bruteStack(f, float32, all, binning, vel, opts)
  doAssert listed.data == whole.data and listed.fold == whole.fold
  doAssert listed.stacked == whole.stacked
  # The first three shots, as a list and as a range; an array will do.
  let part = bruteStack(f, float32, all[0 ..< 72], binning, vel, opts)
  let span = bruteStack(f, float32, 0 ..< 72, binning, vel, opts)
  doAssert part.stacked == 72 and part.data == span.data and part.fold == span.fold
  doAssert bruteStack(f, float32, [3, 4, 5], binning, vel).stacked == 3
  # One side of the split spread, picked by key: half the fold everywhere.
  var positive: seq[int]
  for t in 0 ..< f.len:
    if f.traceKeys(t).offset > 0: positive.add t
  doAssert positive.len == flat.traces div 2
  let side = bruteStack(f, float32, positive, binning, vel, opts)
  doAssert side.stacked == positive.len
  let deep = int(1500 / dt) # below any stretch mute: every trace counts
  var foldSum = 0
  for b in 0 ..< cdps: foldSum += side.fold[b, deep]
  doAssert foldSum == positive.len
  # Any order (the sums then differ in their last bits only), and a trace
  # listed twice counts twice.
  var back = all
  for i in 0 ..< back.len div 2: swap(back[i], back[back.high - i])
  let reversed = bruteStack(f, float32, back, binning, vel, opts)
  doAssert reversed.fold == whole.fold
  for b in 0 ..< cdps:
    for j in 0 ..< ns:
      doAssert abs(reversed.data[b, j] - whole.data[b, j]) <= 1e-5
  let once = bruteStack(f, float32, [100], binning, vel)
  let twice = bruteStack(f, float32, [100, 100], binning, vel)
  doAssert twice.stacked == 2 and twice.data == once.data
  let bin = binning.binOf(f.traceKeys(100))
  doAssert once.fold[bin, deep] == 1 and twice.fold[bin, deep] == 2
  # Progress counts the list's traces.
  var last: (int64, int64)
  let rec: Progress = proc (done, total: int64): bool {.gcsafe.} =
    last = (done, total)
    true
  discard bruteStack(f, float32, positive, binning, vel, opts, rec)
  doAssert last == (int64(positive.len), int64(positive.len))
  doAssertRaises ValueError: discard bruteStack(f, float32, [0, f.len], binning, vel)
  doAssertRaises ValueError: discard bruteStack(f, float32, [-1], binning, vel)
  doAssertRaises ValueError:
    discard bruteStack(f, float32, newSeq[int](), binning, vel) # nothing to stack
  let field = initVelocityField([1.0, float64(cdps)], [vel, vel], dtMs = 4,
    tMaxMs = 2400)
  doAssert bruteStack(f, float32, positive, binning, field, opts).stacked ==
    positive.len

block statics_and_flow:
  var f = openSeismic(topo.path, keys = true)
  var opts = initBruteStackOptions(flow = initPipeline([bandPassOp(Corners(
      f1: 3, f2: 6, f3: 60, f4: 80))]))
  let raw = bruteStack(f, float32, binning, vel, opts)
  opts.statics = some(datumStatics(100, 2000))
  opts.staticsInterp = siCubic
  let fixed = bruteStack(f, float32, binning, vel, opts)
  # With statics, every full-fold CDP's reflectors peak at t0; without,
  # the stack is smeared.
  var sumRaw, sumFixed = 0.0
  for b in 0..<cdps:
    if topo.foldOf[b + 1] < 4: continue
    for t0 in t0s:
      let (t, a) = fixed.peakNear(b, t0)
      doAssert abs(t - t0) <= dt and abs(a - 1) <= 0.1, $b & " " & $t0 & " " & $a
      sumFixed += a
      sumRaw += raw.peakNear(b, t0)[1]
  doAssert sumRaw < 0.75 * sumFixed, $sumRaw & " " & $sumFixed

proc progressAndCancel() =
  # A proc, so the callbacks capture locals (gcsafe closures).
  var f = openSeismic(flat.path, keys = true)
  var opts = initBruteStackOptions(workers = 2)
  opts.chunkTraces = 40
  var calls: seq[(int64, int64)]
  let rec: Progress = proc (done, total: int64): bool {.gcsafe.} =
    calls.add (done, total)
    true
  discard bruteStack(f, float32, binning, vel, opts, rec)
  doAssert calls.len >= 2 and calls[^1] == (int64(flat.traces), int64(flat.traces))
  for i in 1..<calls.len: doAssert calls[i][0] > calls[i - 1][0]
  var n = 0
  let stop: Progress = proc (done, total: int64): bool {.gcsafe.} =
    inc n
    false
  doAssertRaises SegCancelledError:
    discard bruteStack(f, float32, binning, vel, opts, stop)
  doAssert n == 1
  # The file is still usable after a cancel.
  doAssert bruteStack(f, float32, binning, vel, opts).stacked == flat.traces

progressAndCancel()

type StackJob = object
  ## A brute stack for a background thread: what it reads, where it puts
  ## its result.
  f: ptr SeismicFile
  binning: ptr Binning
  velocity: ptr VelocityFunction
  res: ptr StackResult[float32]
  ok: ptr bool

proc runStackJob(job: StackJob) {.thread.} =
  try:
    var opts = initBruteStackOptions(workers = 2)
    opts.chunkTraces = 8
    {.cast(gcsafe).}:
      job.res[] = bruteStack(job.f[], float32, job.binning[], job.velocity[], opts)
    job.ok[] = true
  except CatchableError:
    job.ok[] = false

block other_threads_read_the_file:
  # The thread contract: the job runs on a background thread while this
  # one keeps reading the same file (samples, keys, views); the result is
  # the one a stack on this thread gives. (Creating prefetchers here
  # meanwhile needs the parseSEG release that locks the map's pin count.)
  var f = openSeismic(flat.path, keys = true)
  var opts = initBruteStackOptions(workers = 2)
  opts.chunkTraces = 8
  let want = bruteStack(f, float32, binning, vel, opts)
  var res: StackResult[float32]
  var ok = false
  var thread: Thread[StackJob]
  createThread(thread, runStackJob, StackJob(f: addr f, binning: addr binning,
    velocity: addr vel, res: addr res, ok: addr ok))
  var buf = newTensor[float32](24, ns)
  var first0 = newTensor[float32](24, ns)
  discard f.readTracesInto(0 ..< 24, first0)
  var live = 0
  for pass in 0 ..< 40:
    let first = (pass mod shots) * 24
    discard f.readTracesInto(first ..< first + 24, buf)
    if first == 0: doAssert buf == first0
    let rows = f.rowsOf(first ..< first + 24)
    for r in 0 ..< rows.len:
      if binning.binOf(rows.keysAt(r)) >= 0 and f.traceKeys(first + r).offset != 0:
        inc live
  joinThread(thread)
  doAssert ok and live == 40 * 24
  doAssert res.data == want.data and res.fold == want.fold
  doAssert res.stacked == want.stacked and res.binTraces == want.binTraces

block errors:
  var nokeys = openSeismic(flat.path)
  doAssertRaises ValueError: discard bruteStack(nokeys, float32, binning, vel)
  var f = openSeismic(flat.path, keys = true)
  var opts = initBruteStackOptions(workers = 0)
  doAssertRaises ValueError: discard bruteStack(f, float32, binning, vel, opts)
  opts = initBruteStackOptions()
  doAssert opts.statics.isNone # no statics unless asked
  opts.statics = some(DatumStatics()) # not from datumStatics
  try:
    discard bruteStack(f, float32, binning, vel, opts)
    doAssert false, "a default DatumStatics must be refused"
  except ValueError as e:
    doAssert "opts.statics" in e.msg, e.msg
  opts = initBruteStackOptions()
  opts.nmo = NmoParams()
  doAssertRaises ValueError: discard bruteStack(f, float32, binning, vel, opts)
  # A flow that does not compile (corners beyond Nyquist at 2 ms).
  opts = initBruteStackOptions(flow = initPipeline([filterOp(lowCut(300, 400))]))
  doAssertRaises ValueError: discard bruteStack(f, float32, binning, vel, opts)
  let field3 = initVelocityField([1.0, 2], [1.0, 2], [vel, vel, vel, vel],
    dtMs = 4, tMaxMs = 1000)
  doAssertRaises ValueError: discard bruteStack(f, float32, binning, field3)
  let field2 = initVelocityField([1.0], [vel], dtMs = 4, tMaxMs = 1000)
  doAssertRaises ValueError:
    discard bruteStack(f, float32, inlineCrosslineBinning(1, 2, 1, 2), field2)
  doAssertRaises ValueError:
    discard bruteStack(f, float32, binning, VelocityFunction())
  # A field must reach the stack's last sample (2000 ms here): past its end
  # its velocities stay constant, which is not what a short field means.
  proc fieldTo(tMaxMs: float64): VelocityField =
    initVelocityField([1.0, float64(cdps)], [vel, vel], dtMs = 4, tMaxMs = tMaxMs)
  doAssert fieldTo(1996).tMaxMs == 1996
  try:
    discard bruteStack(f, float32, binning, fieldTo(1996))
    doAssert false, "a field that ends before the stack must be refused"
  except ValueError as e:
    doAssert "1996" in e.msg and "2000" in e.msg and "tMaxMs" in e.msg, e.msg
  doAssert bruteStack(f, float32, binning, fieldTo(2000)).stacked == flat.traces
  opts = initBruteStackOptions()
  opts.samples = 500 # the stack ends at 998 ms
  doAssert bruteStack(f, float32, binning, fieldTo(1000), opts).stacked > 0
  opts.startMs = 4 # ... at 1002 ms
  doAssertRaises ValueError: discard bruteStack(f, float32, binning, fieldTo(1000), opts)
  doAssertRaises ValueError:
    discard bruteStack(f, float32, [0, 1, 2], binning, fieldTo(1000))
  doAssertRaises ValueError: discard bruteStack(f, float32, 0 .. f.len, binning, vel)

block demo_files:
  let d = demoDir()
  # survey.sgy is post-stack (offset 0, fold 1): the stack is the input.
  var s = openSeismic(d / "survey.sgy", keys = true)
  let bins = inlineCrosslineBinning(1001, 1, 1, 200)
  let res = bruteStack(s, float32, bins, constantVelocity(2000))
  doAssert res.stacked == 200 and res.skipped == 0
  var raw = newTensor[float32](200, 2001)
  discard s.readTracesInto(0 ..< 200, raw)
  doAssert res.data == raw
  doAssert res.binning.location(5).crossline == 6
  # shot.sgd: SEG-D with auxiliary traces of other intervals and two
  # seismic ones with coordinates (one bin per midpoint, along x).
  var shot = openSeismic(d / "shot.sgd", keys = true)
  var live = 0
  var cmp: seq[float64]
  for t in 0..<shot.len:
    let k = shot.traceKeys(t)
    if k.status in {tsLive, tsUnknown}:
      inc live
      cmp.add k.cmpX
  doAssert live == 2
  let line = lineBinning(cmp[0], shot.traceKeys(3).cmpY, 90, cmp[1] - cmp[0], 2,
    maxOffLine = 1)
  let sres = bruteStack(shot, float32, line, constantVelocity(3000),
    initBruteStackOptions(nmo = initNmoParams(maxStretch = Inf)))
  doAssert sres.stacked + sres.skipped == live and sres.stacked == 2
  doAssert sres.dtMs == 1
  # The SEG-D stack round-trips through openSeismic as SEG-Y (parseSEG F3:
  # traces from keys, no SEG-Y template), with its keys and samples.
  let stackPath = getTempDir() / "processseg_tbatch_shot_stack.sgy"
  discard writeStack(stackPath, sres)
  var st = openSeismic(stackPath, keys = true)
  doAssert st.kind == skSegy and st.len == 2
  doAssert st.index.segy.binary.traceSorting == 4
  let back = st.readTraces(0 ..< 2)
  doAssert back == sres.data
  for bin in 0 ..< 2:
    let k = st.traceKeys(bin)
    let loc = line.location(bin)
    doAssert k.ensemble == loc.cdp and k.cmpX == loc.x and k.cmpY == loc.y
    doAssert k.offset == 0 and k.t0Ms == 0 and k.intervalUs == 1000
    doAssert k.status == tsLive
    var most = 0'i32
    for j in 0 ..< sres.fold.shape[1]: most = max(most, sres.fold[bin, j])
    doAssert most == 1
    doAssert st.traceField(bin, "trace.numHorzStacked").toFloat == 1
  st.close()
  # A bin with no fold is written as a dead trace; Rev 1 and a time origin.
  var empty = sres
  empty.fold = sres.fold.clone()
  empty.fold[1, _] = 0'i32
  empty.startMs = -4
  discard writeStack(stackPath, empty, SegyOutputOptions(revision: SegyRevision(major: 1)))
  var st1 = openSeismic(stackPath, keys = true)
  doAssert st1.traceKeys(0).status == tsLive and st1.traceKeys(1).status == tsDead
  doAssert st1.traceKeys(0).t0Ms == -4 and st1.index.segy.revision.major == 1
  doAssert st1.index.segy.binary.traceSorting == 4 # 0 in the options: a stack
  st1.close()
  var bad = sres
  bad.fold = newTensor[int32](1, 1)
  doAssertRaises(ValueError): discard writeStack(stackPath, bad)
  # D14: the same file written by hand from stackKeys and stackOverrides.
  let handPath = getTempDir() / "processseg_tbatch_hand_stack.sgy"
  block:
    var w = openSeismicWriter(handPath, SegyOutputOptions(traceSorting: 4))
    for bin in 0 ..< empty.data.shape[0]:
      w.writeTrace(empty.stackKeys(bin), empty.data[bin, _].toFlatSeq,
        overrides = empty.stackOverrides(bin))
    doAssert not w.close().isFatal
  discard writeStack(stackPath, empty)
  doAssert readFile(handPath) == readFile(stackPath)
  removeFile(handPath)
  doAssert empty.maxFold(1) == 0 and empty.maxFold(0) == 1
  doAssertRaises(ValueError): discard empty.stackKeys(2)
  doAssertRaises(ValueError): discard bad.stackOverrides(0)
  # line.sg2: SEG-2 has no CDP numbers, so header-CDP bins take none.
  var sg2 = openSeismic(d / "line.sg2", keys = true)
  let gres = bruteStack(sg2, float32, cdpBinning(1, 10), constantVelocity(1500))
  doAssert gres.stacked == 0 and gres.skipped == sg2.len

block max_fold_of_any_layout:
  # maxFold reads the fold's buffer; a fold that is a strided view still
  # gives its own values.
  var res = StackResult[float32](binning: cdpBinning(1, 2), dtMs: 2,
    data: newTensor[float32](2, 3))
  var wide = newTensor[int32](2, 6)
  for b in 0 ..< 2:
    for j in 0 ..< 6: wide[b, j] = int32(10 * b + (if j mod 2 == 0: j else: 50))
  res.fold = wide[_, 0..4|2] # columns 0, 2, 4
  doAssert not res.fold.is_C_contiguous and res.fold.shape == [2, 3]
  doAssert res.maxFold(0) == 4 and res.maxFold(1) == 14
  res.fold = res.fold.clone()
  doAssert res.fold.is_C_contiguous
  doAssert res.maxFold(0) == 4 and res.maxFold(1) == 14
  doAssert res.stackKeys(1).status == tsLive and res.stackKeys(1).ensemble == 2
  res.fold[0, _] = 0'i32
  doAssert res.maxFold(0) == 0 and res.stackKeys(0).status == tsDead

block decode_failures_counted:
  # Traces whose samples lie past the end of the file fail to decode: they
  # are counted, not stacked.
  var f = openSeismic(flat.path, keys = true)
  for t in [5, 6, 100]: f.mindex.entries[t].offset = int64(getFileSize(flat.path))
  let res = bruteStack(f, float32, binning, vel)
  doAssert res.failed == 3 and res.stacked == flat.traces - 3 and res.skipped == 0

block skipped_traces:
  # A NaN static (unknown elevation) with statics on, and a trace with no
  # interval, are skipped; a trace fully stretch-muted is skipped too.
  var f = openSeismic(flat.path, keys = true)
  f.mindex.keys[7].rcvZ = NaN
  f.mindex.keys[8].intervalUs = NaN
  var opts = initBruteStackOptions()
  opts.statics = some(datumStatics(100, 2000))
  let res = bruteStack(f, float32, binning, vel, opts)
  doAssert res.skipped == 2 and res.stacked == flat.traces - 2
  opts = initBruteStackOptions(nmo = initNmoParams(maxStretch = 1e-6))
  opts.samples = 50 # 100 ms: every trace (none at zero offset) is muted
  let muted = bruteStack(f, float32, binning, vel, opts)
  doAssert muted.stacked == 0 and muted.skipped == flat.traces

block interval_kept_exact:
  # A whole-microsecond interval whose value in ms is not a binary fraction
  # (1001 µs: 1.001 ms · 1000 is 1000.9999999999999, and 1472 of the 65535
  # SEG-Y intervals are like it): the stack keeps the input's µs and
  # stackKeys hands those to the writer.
  let path = dir / "odd.sgy"
  block:
    var w = openSeismicWriter(path, SegyOutputOptions())
    var x = newSeq[float32](200)
    x[50] = 1
    for cdp in 1 .. 2:
      for i in 0 ..< 3:
        var k = initTraceKeys()
        k.ensemble = float64(cdp)
        k.offset = 10 * float64(i)
        k.t0Ms = 0
        k.intervalUs = 1001
        k.status = tsLive
        w.writeTrace(k, x)
    doAssert not w.close().isFatal
  var f = openSeismic(path, keys = true)
  doAssert f.traceKeys(0).intervalUs == 1001
  let res = bruteStack(f, float32, cdpBinning(1, 2), constantVelocity(2000))
  doAssert res.stacked == 6 and res.dtMs == 1.001 and res.dtMs * 1000 != 1001
  doAssert res.intervalUs == 1001
  doAssert res.stackKeys(0).intervalUs == 1001
  let stackPath = dir / "odd_stack.sgy"
  doAssert not writeStack(stackPath, res).isFatal
  var st = openSeismic(stackPath, keys = true)
  doAssert st.len == 2 and st.traceKeys(1).intervalUs == 1001
  st.close()
  f.close()
  # A result built by hand, with no intervalUs: from its dtMs.
  let hand = StackResult[float32](binning: cdpBinning(1, 2), dtMs: 2,
    data: newTensor[float32](2, 3), fold: newTensor[int32](2, 3))
  doAssert hand.intervalUs == 0 and hand.stackKeys(0).intervalUs == 2000

block memory_bound:
  # One stack, whatever the number of workers (there were workers + 1).
  let m = bruteStackMemory(float32, bins = 1000, samples = 2001,
    inSamples = 2001, workers = 4)
  let chunk = 256 * 2001 * 4
  let ring = max(4 shl 20, 5 * chunk) + chunk
  # Per worker: the flow's scratch, an output trace, and the velocity
  # columns of the bins it met last (8 MiB of them: 524 of 2001 samples).
  let cache = ((8 shl 20) div (2001 * 8)) * 2001 * 8
  doAssert cache <= 8 shl 20 and cache > (8 shl 20) - 2001 * 8
  let scratch = 4 * (16 * 4 * 2001 * 8 + (1 shl 20) + 2001 * 8 + cache)
  doAssert m == stackMemory(float32, 1000, 2001) + ring + scratch
  # ... of every bin when there are few, and of one bin at least.
  let small = stackMemory(float32, 3, 2001) + max(4 shl 20, 2 * chunk) + chunk +
    16 * 4 * 2001 * 8 + (1 shl 20) + 2001 * 8
  doAssert bruteStackMemory(float32, 3, 2001, 2001, 1) == small + 3 * 2001 * 8
  doAssert bruteStackMemory(float32, 3, 2001, 2001, 1,
    velocityCacheBytes = 0) == small + 2001 * 8
  doAssertRaises ValueError:
    discard bruteStackMemory(float32, 3, 2001, 2001, 1, velocityCacheBytes = -1)
  let big = stackMemory(float32, 100_000, 2001)
  doAssert bruteStackMemory(float32, 100_000, 2001, 2001, workers = 8) -
    bruteStackMemory(float32, 100_000, 2001, 2001, workers = 1) < big div 4
  # Sizes beyond int are a ValueError in every build mode, as bad
  # arguments are (they were an OverflowDefect, or wrapped in danger).
  doAssertRaises ValueError:
    discard bruteStackMemory(float32, high(int) div 16, 2001, 2001, 4)
  doAssertRaises ValueError:
    discard bruteStackMemory(float32, 10, 10, high(int) div 8, 4)
  doAssertRaises ValueError:
    discard bruteStackMemory(float32, 10, high(int) div 4, 10, 1)
  doAssertRaises ValueError:
    discard bruteStackMemory(float32, 10, 10, 10, high(int) div 2)
  doAssertRaises ValueError:
    discard bruteStackMemory(float32, 10, 10, 10, 1, chunkTraces = high(int) div 3)
  doAssertRaises ValueError:
    discard bruteStackMemory(float32, 10, 10, 10, 1, maxRingBytes = high(int))
  doAssertRaises ValueError: discard bruteStackMemory(float32, -1, 10, 10, 1)
  doAssertRaises ValueError: discard bruteStackMemory(float32, 10, 10, 10, 0)
  doAssertRaises ValueError:
    discard bruteStackMemory(float32, 10, 10, 10, 1, chunkTraces = 0)
  # A stack longer than the input widens the chunks.
  doAssert bruteStackMemory(float32, 10, 4000, 2001, 1) -
    bruteStackMemory(float32, 10, 2001, 2001, 1) >= 2 * 256 * 1999 * 4
  # The scratch term bounds what a typical flow reserves.
  let plan = initPipeline([bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
    agcOp(500), normalizeOp(normPercentile, perPanel)]).compile(float32, 2, 2001)
  let w = plan.workspaceSize
  doAssert w.f32 * 4 + w.f64 * 8 + w.u32 * 4 <= 16 * 4 * 2001 * 8 + (1 shl 20)

removeDir(dir)
