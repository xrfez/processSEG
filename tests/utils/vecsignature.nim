## Vibrator signature kernels for `tests/tvectorized.nim`, in a module of
## their own (as `veccorrelate`): their plans transform with the one-trace
## FFT, whose instances (scalar by design) would otherwise sit among the
## lanes instances that `veckernels` checks.

import processSEG

proc vec_hilbert_f32(hp: HilbertPlan[float32], x: openArray[float32],
    h: var openArray[float32], ws: var Workspace) {.exportc: "vec_hilbert_f32", noinline.} =
  hp.hilbertInto(x, h, ws)
  hp.envelopeInto(x, h, ws)

proc vec_hilbert_f64(hp: HilbertPlan[float64], x: openArray[float64],
    h: var openArray[float64], ws: var Workspace) {.exportc: "vec_hilbert_f64", noinline.} =
  hp.hilbertInto(x, h, ws)
  hp.envelopeInto(x, h, ws)

proc vec_track_f32(tp: SweepTrackPlan[float32], x: openArray[float32], r: SweepReference,
    t: var SweepTrack, ws: var Workspace) {.exportc: "vec_track_f32", noinline.} =
  tp.trackInto(x, r, t, ws)

proc vec_track_f64(tp: SweepTrackPlan[float64], x: openArray[float64], r: SweepReference,
    t: var SweepTrack, ws: var Workspace) {.exportc: "vec_track_f64", noinline.} =
  tp.trackInto(x, r, t, ws)

proc trackOnce(T: typedesc, harmonics: int) =
  # scan: the running sums of every harmonic in 8 (or 16) lanes.
  let spec = linearSweep(10, 60, 1000, 2)
  let tp = initSweepTrackPlan(T, TrackParams(harmonics: harmonics), 2, spec.sweepLen)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  var r = initSweepReference(tp)
  var t = initSweepTrack(tp)
  let x = spec.sweepSamples(T)
  tp.referenceInto(x, r, ws)
  when T is float32: vec_track_f32(tp, x, r, t, ws)
  else: vec_track_f64(tp, x, r, t, ws)
  echo t.phaseDeg[20]

proc vec_cross_f32(cp: CrossCorrelationPlan[float32], x, r: openArray[float32],
    lags: var openArray[float32], ws: var Workspace) {.exportc: "vec_cross_f32", noinline.} =
  cp.crossCorrelateInto(x, r, lags, ws)

proc vec_cross_f64(cp: CrossCorrelationPlan[float64], x, r: openArray[float64],
    lags: var openArray[float64], ws: var Workspace) {.exportc: "vec_cross_f64", noinline.} =
  cp.crossCorrelateInto(x, r, lags, ws)

proc crossOnce(T: typedesc) =
  # crossKernel: x·conj(y), bin by bin.
  let cp = initCrossCorrelationPlan(T, 20, 2, 200)
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var x, r = newSeq[T](200)
  x[10] = 1
  r[12] = 1
  var lags = newSeq[T](cp.lags)
  when T is float32: vec_cross_f32(cp, x, r, lags, ws)
  else: vec_cross_f64(cp, x, r, lags, ws)
  echo lags[3]

proc vec_weightedSum_f32(m, b: openArray[float32], dst: var openArray[float32]) {.
    exportc: "vec_weightedSum_f32", noinline.} =
  groundForceInto(m, b, mass(2450, muKilogram), mass(1180, muKilogram), dst)

proc vec_weightedSum_f64(m, b: openArray[float64], dst: var openArray[float64]) {.
    exportc: "vec_weightedSum_f64", noinline.} =
  groundForceInto(m, b, mass(2450, muKilogram), mass(1180, muKilogram), dst)

proc forceOnce() =
  # weightedSumKernel: Mr·a_M + Mb·a_B (vibqc).
  var a, b, f = newSeq[float32](100)
  var c, d, g = newSeq[float64](100)
  a[1] = 1
  c[1] = 1
  vec_weightedSum_f32(a, b, f)
  vec_weightedSum_f64(c, d, g)
  echo f[1], g[1]

proc signatureOnce*() =
  forceOnce()
  crossOnce(float32)
  crossOnce(float64)
  # scan (sweeptrack), 8 and 16 lanes, both sample types.
  trackOnce(float32, 5)
  trackOnce(float32, 12)
  trackOnce(float64, 5)
  # rotateBins (the spectrum times −i) and envelopeKernel (fft/hilbert).
  let a = initHilbertPlan(float32, 100)
  let b = initHilbertPlan(float64, 100)
  var ws = initWorkspace()
  ws.reserve(a.workspaceSize)
  ws.reserve(b.workspaceSize)
  var x, h = newSeq[float32](100)
  var y, g = newSeq[float64](100)
  x[3] = 1
  y[3] = 1
  vec_hilbert_f32(a, x, h, ws)
  vec_hilbert_f64(b, y, g, ws)
  echo h[0], g[0]
