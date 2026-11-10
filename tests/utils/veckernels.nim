## Kernel instances with fixed symbols for `tests/tvectorized.nim`, which
## builds this module and checks their disassembly for packed SIMD.

import processSEG

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

when isMainModule:
  fftOnce(float32)
  fftOnce(float64)
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
  echo x[0], y[0]
