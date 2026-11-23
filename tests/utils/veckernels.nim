## Kernel instances with fixed symbols for `tests/tvectorized.nim`, which
## builds this module and checks their disassembly for packed SIMD.

import processSEG
import synth, veccorrelate, vecsignature

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

proc vec_scaleInto_f32(src: openArray[float32], dst: var openArray[float32]) {.
    exportc: "vec_scaleInto_f32", noinline.} =
  scaleInto(src, dst, 0.25) # calibration: a row times its factor, src → dst

proc vec_scaleInto_f64(src: openArray[float64], dst: var openArray[float64]) {.
    exportc: "vec_scaleInto_f64", noinline.} =
  scaleInto(src, dst, 0.25)

proc vec_traceStats_f32(x: openArray[float32]): TraceStats {.exportc: "vec_traceStats_f32", noinline.} =
  traceStats(x)

proc vec_traceStats_f64(x: openArray[float64]): TraceStats {.exportc: "vec_traceStats_f64", noinline.} =
  traceStats(x)

proc vec_agc_f32(x: var openArray[float32], plan: AgcPlan,
    ws: var Workspace) {.exportc: "vec_agc_f32", noinline.} =
  agc(plan, x, ws)

proc agcOnce() =
  let plan = initAgcPlan(20, 2, 100)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var x = newSeq[float32](100)
  x[3] = 1
  vec_agc_f32(x, plan, ws)

proc vec_curve_f32(x: var openArray[float32], plan: GainCurvePlan[float32],
    startMs: float64) {.exportc: "vec_curve_f32", noinline.} =
  plan.applyGainCurve(x, startMs)

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
  plan.applyFilter(b, meta, ws)

proc filterOnce() =
  let plan = initFilterPlan(float32, lowCut(10, 40), 2, 50)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var g = cmpGather[float32](cmpX = 0, offsets = [0.0, 0.0], 50, 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  vec_filter_f32(rowBlock(g.data), meta, plan, ws)

proc vec_iir_f32(b: RowBlock[float32], meta: ChunkMeta, plan: IirPlan[float32],
    ws: var Workspace) {.exportc: "vec_iir_f32", noinline.} =
  plan.applyFilter(b, meta, ws)

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

proc vec_amplitude_f32(b: RowBlock[float32], meta: ChunkMeta,
    dst: var openArray[TraceAmplitude], ws: var Workspace) {.exportc: "vec_amplitude_f32",
    noinline.} =
  b.traceAmplitudeInto(meta, AmplitudeParams(), dst, ws)

proc amplitudeOnce() =
  # sumAndSquares, extremes, countBeyond and spikeCount (amplitude QC).
  var g = cmpGather[float32](cmpX = 0, offsets = [0.0, 100.0], 200, 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var ws = initWorkspace()
  ws.reserve(traceAmplitudeWorkspace(200))
  var dst = newSeq[TraceAmplitude](2)
  vec_amplitude_f32(rowBlock(g.data), meta, dst, ws)
  echo dst[0].rmsDb

proc vec_recordStack_f32(acc: var RecordStack[float32], x: openArray[float32],
    ws: var Workspace): bool {.exportc: "vec_recordStack_f32", noinline.} =
  acc.add(0, x, ws)

proc recordStackOnce() =
  # addVertical and addWeighted (vibroseis record stacks).
  var x = newSeq[float32](100)
  x[3] = 1
  for mode in RecordStackMode:
    var acc = initRecordStack[float32](1, 100, 2, RecordStackParams(mode: mode))
    var ws = initWorkspace()
    ws.reserve(acc.workspaceSize)
    discard vec_recordStack_f32(acc, x, ws)

proc vec_stackMerge_f32(acc: var StackAccumulator[float32],
    other: StackAccumulator[float32], dst: var Tensor[float32]) {.
    exportc: "vec_stackMerge_f32", noinline.} =
  acc.merge(other)
  acc.stackInto(dst, fnFold)
  acc.stackInto(dst, fnSqrtFold)

proc stackMergeOnce() =
  # mergeInto and normalize (CMP stacks).
  var acc = initStackAccumulator[float32](2, 100, 2)
  let other = initStackAccumulator[float32](2, 100, 2)
  var dst = zeros[float32](2, 100)
  vec_stackMerge_f32(acc, other, dst)

proc vec_shift_f32(x: var openArray[float32], shift: float64,
    interp: ShiftInterp) {.exportc: "vec_shift_f32", noinline.} =
  x.shiftSamples(shift, interp)

proc shiftOnce() =
  # lerpInto and cubicInto (fractional statics shifts).
  var x = newSeq[float32](1000)
  vec_shift_f32(x, 2.5, siLinear)
  vec_shift_f32(x, -2.5, siCubic)

proc spectrogramOnce() =
  # laneMagnitude and amplitudesFinite (fft/spectrogram), both sample types.
  let sg = initSpectrogramPlan(float32, windowLen = 32, hop = 8, dtMs = 2.0,
    samples = 64)
  var ws = initWorkspace()
  ws.reserve(sg.averageWorkspace)
  var x = newSeq[float32](64)
  x[3] = 1
  var amp = newSeq[float64](sg.frames * sg.bins)
  sg.spectrogramInto(x, amp, ws)
  var g = cmpGather[float32](0, newSeq[float64](2), 64, 2)
  g.data[0, 3] = 1
  let meta = initChunkMeta(g.rows, dtMs = 2.0)
  discard sg.averageSpectrogramInto(g.data, meta, amp, ws)
  let sg64 = initSpectrogramPlan(float64, windowLen = 32, hop = 8, dtMs = 2.0,
    samples = 64)
  var ws64 = initWorkspace()
  ws64.reserve(sg64.averageWorkspace)
  var y = newSeq[float64](64)
  y[3] = 1
  var amp64 = newSeq[float64](sg64.frames * sg64.bins)
  sg64.spectrogramInto(y, amp64, ws64)
  var g64 = cmpGather[float64](0, newSeq[float64](2), 64, 2)
  g64.data[0, 3] = 1
  let meta64 = initChunkMeta(g64.rows, dtMs = 2.0)
  discard sg64.averageSpectrogramInto(g64.data, meta64, amp64, ws64)
  echo amp[0], amp64[0]

when isMainModule:
  spectrogramOnce()
  signatureOnce()
  amplitudeOnce()
  correlateOnce()
  recordStackOnce()
  stackMergeOnce()
  shiftOnce()
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
  var x2 = newSeq[float32](1000)
  var y2 = newSeq[float64](1000)
  vec_scaleInto_f32(x, x2)
  vec_scaleInto_f64(y, y2)
  x[1] = x2[1]
  y[1] = y2[1]
  echo x[0], y[0], vec_traceStats_f32(x), vec_traceStats_f64(y)

