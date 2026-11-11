## Brute stack (batch): a synthetic pre-stack SEG-Y line (split-spread
## shots over topography, signed offsets, header CDPs) stacked with NMO,
## statics and a flow; the fold per bin, the peaks of the reflectors, 1 vs
## 3 workers, a trace subset, a 2D field, progress and cancel, errors, and
## parseSEG's demo files (post-stack SEG-Y reproduced exactly, SEG-D and
## SEG-2 runs). Memory bound arithmetic.

import std/[assertions, math, options, os, random]
import processSEG
import parseSEG/io/diskio
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

block workers_and_subset:
  var f = openSeismic(flat.path, keys = true)
  var opts = initBruteStackOptions()
  opts.chunkTraces = 32 # several chunks per worker
  let one = bruteStack(f, float32, binning, vel, opts)
  opts.workers = 3
  let three = bruteStack(f, float32, binning, vel, opts)
  doAssert three.stacked == one.stacked and three.fold == one.fold
  for b in 0..<cdps:
    for j in 0..<ns:
      doAssert abs(three.data[b, j] - one.data[b, j]) <= 1e-5
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

block statics_and_flow:
  var f = openSeismic(topo.path, keys = true)
  var opts = initBruteStackOptions(flow = initPipeline([bandPassOp(Corners(
      f1: 3, f2: 6, f3: 60, f4: 80))]))
  let raw = bruteStack(f, float32, binning, vel, opts)
  opts.useStatics = true
  opts.statics = datumStatics(100, 2000)
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

block errors:
  var nokeys = openSeismic(flat.path)
  doAssertRaises ValueError: discard bruteStack(nokeys, float32, binning, vel)
  var f = openSeismic(flat.path, keys = true)
  var opts = initBruteStackOptions(workers = 0)
  doAssertRaises ValueError: discard bruteStack(f, float32, binning, vel, opts)
  opts = initBruteStackOptions()
  opts.useStatics = true # without a datum
  doAssertRaises ValueError: discard bruteStack(f, float32, binning, vel, opts)
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
  # line.sg2: SEG-2 has no CDP numbers, so header-CDP bins take none.
  var sg2 = openSeismic(d / "line.sg2", keys = true)
  let gres = bruteStack(sg2, float32, cdpBinning(1, 10), constantVelocity(1500))
  doAssert gres.stacked == 0 and gres.skipped == sg2.len

block decode_failures_counted:
  # Traces whose samples lie past the end of the file fail to decode: they
  # are counted, not stacked.
  var f = openSeismic(flat.path, keys = true)
  for t in [5, 6, 100]: f.index.entries[t].offset = int64(getFileSize(flat.path))
  let res = bruteStack(f, float32, binning, vel)
  doAssert res.failed == 3 and res.stacked == flat.traces - 3 and res.skipped == 0

block skipped_traces:
  # A NaN static (unknown elevation) with statics on, and a trace with no
  # interval, are skipped; a trace fully stretch-muted is skipped too.
  var f = openSeismic(flat.path, keys = true)
  f.index.keys[7].rcvZ = NaN
  f.index.keys[8].intervalUs = NaN
  var opts = initBruteStackOptions()
  opts.useStatics = true
  opts.statics = datumStatics(100, 2000)
  let res = bruteStack(f, float32, binning, vel, opts)
  doAssert res.skipped == 2 and res.stacked == flat.traces - 2
  opts = initBruteStackOptions(nmo = initNmoParams(maxStretch = 1e-6))
  opts.samples = 50 # 100 ms: every trace (none at zero offset) is muted
  let muted = bruteStack(f, float32, binning, vel, opts)
  doAssert muted.stacked == 0 and muted.skipped == flat.traces

block memory_bound:
  let m = bruteStackMemory(float32, bins = 1000, samples = 2001,
    inSamples = 2001, workers = 4)
  doAssert m >= 5 * stackMemory(float32, 1000, 2001)
  let chunk = 256 * 2001 * 4
  let ring = max(4 shl 20, 5 * chunk) + chunk
  let scratch = 4 * (16 * 4 * 2001 * 8 + (1 shl 20) + 2 * 2001 * 8)
  doAssert m == 5 * stackMemory(float32, 1000, 2001) + ring + scratch
  # The scratch term bounds what a typical flow reserves.
  let plan = initPipeline([bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
    agcOp(500), normalizeOp(normPercentile, perPanel)]).compile(float32, 2, 2001)
  let w = plan.workspaceSize
  doAssert w.f32 * 4 + w.f64 * 8 + w.u32 * 4 <= 16 * 4 * 2001 * 8 + (1 shl 20)

removeDir(dir)
