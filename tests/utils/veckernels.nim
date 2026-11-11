## Kernel instances with fixed symbols for `tests/tvectorized.nim`, which
## builds this module and checks their disassembly for packed SIMD.

import processSEG
import synth

template wrap(name, kernel: untyped, T: typedesc, args: varargs[untyped]) =
  proc name(x: var openArray[T]) {.exportc: astToStr(name), noinline.} =
    kernel(x, args)

wrap(vec_demean_f32, demean, float32)
wrap(vec_demean_f64, demean, float64)
wrap(vec_detrend_f32, detrend, float32)
wrap(vec_scale_f32, scale, float32, 0.5)
wrap(vec_clip_f32, clip, float32, -1.0, 1.0)
wrap(vec_reversePolarity_f32, reversePolarity, float32)
wrap(vec_muteTop_f32, muteTop, float32, 300.5, 20.0)

proc vec_traceStats_f32(x: openArray[float32]): TraceStats {.exportc: "vec_traceStats_f32", noinline.} =
  traceStats(x)

proc vec_traceStats_f64(x: openArray[float64]): TraceStats {.exportc: "vec_traceStats_f64", noinline.} =
  traceStats(x)

proc vec_agc_f32(x: var openArray[float32], plan: AgcPlan,
    ws: var Workspace) {.exportc: "vec_agc_f32", noinline.} =
  agc(x, plan, ws)

proc agcOnce() =
  let plan = initAgcPlan(20, 2, 100)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var x = newSeq[float32](100)
  x[3] = 1
  vec_agc_f32(x, plan, ws)

proc vec_curve_f32(x: var openArray[float32], plan: GainCurvePlan[float32],
    startMs: float64) {.exportc: "vec_curve_f32", noinline.} =
  x.applyGainCurve(plan, startMs)

proc curveOnce() =
  let plan = initGainCurvePlan(float32, powerGain(2), 2, 100)
  var x = newSeq[float32](100)
  vec_curve_f32(x, plan, 0)
  vec_curve_f32(x, plan, 1)

proc vec_taperPads_f32(x: var openArray[float32],
    ramp: openArray[float32]) {.exportc: "vec_taperPads_f32", noinline.} =
  taperPads(x, ramp)

proc vec_fftLanes_f32(p: FftPlan[float32], x: var openArray[float32],
    re, im: var openArray[float32], ws: var Workspace) {.exportc: "vec_fftLanes_f32", noinline.} =
  p.forwardLanes(x, re, im, ws)
  p.inverseLanes(re, im, x, ws)

proc vec_fftLanes_f64(p: FftPlan[float64], x: var openArray[float64],
    re, im: var openArray[float64], ws: var Workspace) {.exportc: "vec_fftLanes_f64", noinline.} =
  p.forwardLanes(x, re, im, ws)
  p.inverseLanes(re, im, x, ws)

proc fftOnce(T: typedesc) =
  let p = initFftPlan(T, 60)
  var ws = initWorkspace()
  ws.reserve(p.workspaceSize)
  var x = newSeq[T](p.len * fftLanes)
  var re, im = newSeq[T](p.bins * fftLanes)
  when T is float32: vec_fftLanes_f32(p, x, re, im, ws)
  else: vec_fftLanes_f64(p, x, re, im, ws)

proc vec_filter_f32(b: RowBlock[float32], meta: ChunkMeta, plan: FilterPlan[
    float32], ws: var Workspace) {.exportc: "vec_filter_f32", noinline.} =
  b.applyFilter(meta, plan, ws)

proc filterOnce() =
  let plan = initFilterPlan(float32, lowCut(10, 40), 2, 50)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var g = cmpGather[float32](cmpX = 0, offsets = [0.0, 0.0], 50, 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  vec_filter_f32(rowBlock(g.data), meta, plan, ws)

proc vec_iir_f32(b: RowBlock[float32], meta: ChunkMeta, plan: IirPlan[float32],
    ws: var Workspace) {.exportc: "vec_iir_f32", noinline.} =
  b.applyFilter(meta, plan, ws)

proc iirOnce() =
  let plan = initIirPlan(float32, butterBandPass(10, 100), 2, 50)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var g = cmpGather[float32](cmpX = 0, offsets = [0.0, 0.0], 50, 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  vec_iir_f32(rowBlock(g.data), meta, plan, ws)

proc vec_stackAdd_f32(acc: var StackAccumulator[float32],
    x: openArray[float32]): bool {.exportc: "vec_stackAdd_f32", noinline.} =
  acc.add(0, x)

proc stackOnce() =
  var acc = initStackAccumulator[float32](1, 100, 2)
  var x = newSeq[float32](100)
  discard vec_stackAdd_f32(acc, x)

proc nmoOnce() =
  # moveoutCols (non-generic, in the nmo module's own C code).
  var src = newSeq[float32](300)
  var dst = newSeq[float32](300)
  var vel = newSeq[float64](300)
  for v in vel.mitems: v = 2000
  discard nmoInto(src, 0, dst, 0, 2, 500, vel, initNmoParams())
  echo dst[0]

proc shotFitOnce() =
  # l1Fit (non-generic, in qc/shotfit): the normal-equation sums.
  var keys = newSeq[TraceKeys](40)
  var counts = newSeq[int](40)
  var picks = newSeq[FirstBreakPick](40)
  for i in 0 ..< 40:
    keys[i] = initTraceKeys()
    keys[i].offset = 25.0 * float64(i + 1)
    keys[i].t0Ms = 0
    counts[i] = 100
    picks[i] = FirstBreakPick(timeMs: 10 + keys[i].offset / 2, col: 1, snrDb: 30,
      confidence: 1)
  let meta = initChunkMeta(rowsOf(keys, counts), 2)
  var fit: ShotFit
  fitShot(meta, picks, ShotFitParams(correlate: false), fit)
  echo fit.used

when isMainModule:
  nmoOnce()
  shotFitOnce()
  stackOnce()
  fftOnce(float32)
  fftOnce(float64)
  filterOnce()
  iirOnce()
  agcOnce()
  curveOnce()
  var x = newSeq[float32](1000)
  var y = newSeq[float64](1000)
  let ramp = newSeq[float32](100)
  vec_demean_f32(x)
  vec_demean_f64(y)
  vec_detrend_f32(x)
  vec_scale_f32(x)
  vec_clip_f32(x)
  vec_reversePolarity_f32(x)
  vec_muteTop_f32(x)
  vec_taperPads_f32(x, ramp)
  echo x[0], y[0], vec_traceStats_f32(x), vec_traceStats_f64(y)

