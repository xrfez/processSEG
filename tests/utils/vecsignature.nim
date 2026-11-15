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

proc signatureOnce*() =
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
