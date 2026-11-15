## The Hilbert transform and envelope: H[cos] = sin and
## H[sin] = −cos on and between bins (Bedrosian: a smooth envelope slower
## than the carrier passes through), the envelope of an amplitude-modulated
## sinusoid and of a tapered sweep, float32 and float64, short inputs, NaN,
## refusals and no allocation.

import std/[assertions, math]
import processSEG
import utils/allocs

const dt = 0.002 # s

proc gauss(j, n: int): float64 =
  ## A smooth envelope, exp(−36) at both ends of `n` samples: its
  ## spectrum is far below any carrier tested, so H passes it through.
  let u = (float64(j) - float64(n - 1) / 2) / (float64(n) / 12)
  exp(-u * u)

proc worstOver[T](a: openArray[T], b: openArray[float64]): float64 =
  for j in 0 ..< a.len: result = max(result, abs(float64(a[j]) - b[j]))

block cos_to_sin:
  # A Gaussian-tapered cosine at frequencies on a bin of the FFT and between
  # bins: H is the same envelope times the sine, to rounding.
  let n = 3001
  let hp = initHilbertPlan(float64, n)
  doAssert hp.samples == n and hp.nfft >= 2 * n
  var ws = initWorkspace()
  ws.reserve(hp.workspaceSize)
  for f in [1000 / (float64(hp.nfft) * 2) * 70, 37.3, 61.0, 123.45]: # Hz; the first on a bin
    var x, wantH, wantNeg = newSeq[float64](n)
    var xs = newSeq[float64](n)
    for j in 0 ..< n:
      let g = gauss(j, n)
      let p = 2 * PI * f * float64(j) * dt + 0.3
      x[j] = g * cos(p)
      xs[j] = g * sin(p)
      wantH[j] = g * sin(p)
      wantNeg[j] = -g * cos(p)
    var h = newSeq[float64](n)
    hp.hilbertInto(x, h, ws)
    doAssert worstOver(h, wantH) < 1e-12, $f & ": " & $worstOver(h, wantH)
    hp.hilbertInto(xs, h, ws)
    doAssert worstOver(h, wantNeg) < 1e-12, $f & ": " & $worstOver(h, wantNeg)
    var env = newSeq[float64](n)
    hp.envelopeInto(x, env, ws)
    var g = newSeq[float64](n)
    for j in 0 ..< n: g[j] = gauss(j, n)
    doAssert worstOver(env, g) < 1e-12, $f & ": " & $worstOver(env, g)

block float32_form:
  let n = 2000
  let hp = initHilbertPlan(float32, n)
  var ws = initWorkspace()
  ws.reserve(hp.workspaceSize)
  var x = newSeq[float32](n)
  var want = newSeq[float64](n)
  for j in 0 ..< n:
    let g = gauss(j, n)
    x[j] = float32(g * cos(2 * PI * 41.7 * float64(j) * dt))
    want[j] = g * sin(2 * PI * 41.7 * float64(j) * dt)
  var h = newSeq[float32](n)
  hp.hilbertInto(x, h, ws)
  doAssert worstOver(h, want) < 2e-5, $worstOver(h, want)
  var env = newSeq[float32](n)
  var g = newSeq[float64](n)
  for j in 0 ..< n: g[j] = gauss(j, n)
  hp.envelopeInto(x, env, ws)
  doAssert worstOver(env, g) < 2e-5, $worstOver(env, g)

block amplitude_modulation:
  # (1 + 0.5·cos Ωt)·cos ωt with Ω < ω: the envelope is the modulation.
  let n = 4001
  let hp = initHilbertPlan(float64, n)
  var ws = initWorkspace()
  ws.reserve(hp.workspaceSize)
  var x, want = newSeq[float64](n)
  for j in 0 ..< n:
    let t = float64(j) * dt
    let a = gauss(j, n) * (1 + 0.5 * cos(2 * PI * 3 * t))
    x[j] = a * cos(2 * PI * 45 * t)
    want[j] = a
  var env = newSeq[float64](n)
  hp.envelopeInto(x, env, ws)
  doAssert worstOver(env, want) < 1e-10, $worstOver(env, want)

block tapered_sweep_envelope:
  # A tapered sweep: inside, the envelope is the taper (to 1e-3, the sweep's
  # own spectral leakage near its start); its peak is 1.
  let spec = linearSweep(8, 80, 6000, 2, initSweepTaper(500, 500, tpCos2))
  let x = spec.sweepSamples(float64)
  let hp = initHilbertPlan(float64, x.len)
  var ws = initWorkspace()
  ws.reserve(hp.workspaceSize)
  var env = newSeq[float64](x.len)
  hp.envelopeInto(x, env, ws)
  var worst = 0.0
  for j in 0 ..< x.len:
    let t = float64(j) * 2
    if t > 250 and t < 5750:
      let w = (if t < 500: taperWeight(tpCos2, t / 500) elif t > 5500:
        taperWeight(tpCos2, (6000 - t) / 500) else: 1.0)
      worst = max(worst, abs(env[j] - w))
  doAssert worst < 1e-3, $worst

block short_inputs_and_nan:
  # A trace shorter than the plan: zeros after it. An empty one: nothing.
  let hp = initHilbertPlan(float64, 1000)
  var ws = initWorkspace()
  ws.reserve(hp.workspaceSize)
  let n = 600
  var x, want = newSeq[float64](n)
  for j in 0 ..< n:
    x[j] = gauss(j, n) * cos(2 * PI * 50 * float64(j) * dt)
    want[j] = gauss(j, n) * sin(2 * PI * 50 * float64(j) * dt)
  var h = newSeq[float64](n)
  hp.hilbertInto(x, h, ws)
  doAssert worstOver(h, want) < 1e-12
  var none: seq[float64]
  hp.hilbertInto(none, none, ws)
  hp.envelopeInto(none, none, ws)
  x[10] = NaN
  hp.hilbertInto(x, h, ws)
  doAssert h[0].isNaN and h[^1].isNaN

block refusals_and_defects:
  doAssertRaises(ValueError): discard initHilbertPlan(float64, 0)
  doAssertRaises(ValueError): discard initHilbertPlan(float64, 100, nfft = 150)
  doAssertRaises(ValueError): discard initHilbertPlan(float64, 100, nfft = 202) # 2·101
  doAssert initHilbertPlan(float64, 100, nfft = 240).nfft == 240
  let hp = initHilbertPlan(float64, 10)
  var ws = initWorkspace()
  ws.reserve(hp.workspaceSize)
  var x = newSeq[float64](11)
  var h = newSeq[float64](11)
  doAssertRaises(AssertionDefect): hp.hilbertInto(x, h, ws) # longer than the plan
  var y = newSeq[float64](10)
  doAssertRaises(AssertionDefect): hp.envelopeInto(y, h, ws) # output length
  var small = initWorkspace()
  doAssertRaises(AssertionDefect): hp.hilbertInto(y, y, small)

block no_allocation:
  let hp = initHilbertPlan(float32, 4000)
  var ws = initWorkspace()
  ws.reserve(hp.workspaceSize)
  var x, h = newSeq[float32](4000)
  for j in 0 ..< x.len: x[j] = float32(sin(float64(j) * 0.3))
  hp.hilbertInto(x, h, ws)
  let n = allocsIn:
    hp.hilbertInto(x, h, ws)
    hp.envelopeInto(x, h, ws)
  doAssert n == 0, $n & " allocations"
