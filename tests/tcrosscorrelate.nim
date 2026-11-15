## A signal against its reference: the correlation over lags
## equals a direct one, a delay peaks at its lag (and so does the envelope),
## the lag window's phase is −360·f·τ and 0 for the reference itself, the
## comparison of a scaled, delayed copy reads its gain and its delay, the
## three normalizations, rows without energy, refusals and no allocation.

import std/[assertions, math, random]
import processSEG
import utils/allocs

proc direct(x, r: seq[float64], m: int): float64 =
  ## Σ x[j + m]·r[j].
  for j in 0 ..< r.len:
    let i = j + m
    if i >= 0 and i < x.len: result += x[i] * r[j]

proc delayed(spec: SweepSpec, n: int, tauMs: float64, amp = 1.0): seq[float64] =
  ## The sweep delayed by `tauMs` (any fraction of a sample), with its taper.
  result = newSeq[float64](n)
  let T = spec.lengthMs
  for j in 0 ..< n:
    let t = float64(j) * spec.dtMs - tauMs
    if t < 0 or t > T: continue
    var w = 1.0
    let s = spec.taper.startMs
    let e = spec.taper.endMs
    if t < s: w = taperWeight(spec.taper.shape, t / s)
    if T - t < e: w = min(w, taperWeight(spec.taper.shape, (T - t) / e))
    result[j] = amp * w * sin(spec.phaseAt(t))

let spec = linearSweep(10, 80, 4000, 1, initSweepTaper(250, 250, tpCos2))
let n = spec.sweepLen + 1000
var pilot = spec.sweepSamples(float64)
pilot.setLen(n)

proc argmax(x: seq[float64]): int =
  for i in 1 ..< x.len:
    if x[i] > x[result]: result = i

block equals_direct_correlation:
  var rng = initRand(3)
  var x = newSeq[float64](900)
  var r = newSeq[float64](700) # rows may be shorter than the plan
  for v in x.mitems: v = rng.gauss(0, 1)
  for v in r.mitems: v = rng.gauss(0, 1)
  let cp = initCrossCorrelationPlan(float64, maxLagMs = 120, dtMs = 2, samples = 1000)
  doAssert cp.maxLag == 60 and cp.lags == 121 and cp.lagMs(0) == -120 and cp.lagMs(60) == 0
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var lags = newSeq[float64](cp.lags)
  cp.crossCorrelateInto(x, r, lags, ws, cnNone)
  var worst, peak = 0.0
  for i in 0 ..< cp.lags:
    let d = direct(x, r, i - cp.maxLag)
    worst = max(worst, abs(lags[i] - d))
    peak = max(peak, abs(d))
  doAssert worst < 1e-10 * peak, $worst
  # Normalized: by √(Σx²·Σr²) and by Σr².
  var ex, er = 0.0
  for v in x: ex += v * v
  for v in r: er += v * v
  var c = newSeq[float64](cp.lags)
  cp.crossCorrelateInto(x, r, c, ws) # cnCoefficient
  for i in 0 ..< cp.lags: doAssert abs(c[i] - lags[i] / sqrt(ex * er)) < 1e-12
  cp.crossCorrelateInto(x, r, c, ws, cnReference)
  for i in 0 ..< cp.lags: doAssert abs(c[i] - lags[i] / er) < 1e-12
  # Lags nearly as long as the rows: still no wrap.
  var y = newSeq[float64](1000)
  for v in y.mitems: v = rng.gauss(0, 1)
  let wide = initCrossCorrelationPlan(float64, maxLagMs = 1800, dtMs = 2, samples = 1000)
  var ws2 = initWorkspace()
  ws2.reserve(wide.workspaceSize)
  var all = newSeq[float64](wide.lags)
  wide.crossCorrelateInto(y, y, all, ws2, cnNone)
  worst = 0
  for i in 0 ..< wide.lags: worst = max(worst, abs(all[i] - direct(y, y, i - wide.maxLag)))
  doAssert worst < 1e-9, $worst

block lag_spectrum_is_the_tapered_dft:
  # The documented window: a Tukey taper, sin² ramps over 10 % of the lags at
  # each end, lag zero the phase origin.
  let cp = initCrossCorrelationPlan(float64, maxLagMs = 50, dtMs = 1, samples = 400)
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var rng = initRand(9)
  var lags = newSeq[float64](cp.lags)
  for v in lags.mitems: v = rng.gauss(0, 1)
  var amp, phase = newSeq[float64](cp.bins)
  cp.lagSpectrumInto(lags, amp, phase, ws)
  let nl = cp.lags
  let ramp = int(round(0.1 * float64(nl)))
  for k in 0 ..< cp.bins:
    var re, im = 0.0
    for i in 0 ..< nl:
      let d = min(i, nl - 1 - i)
      let w = (if d >= ramp: 1.0 else: sin(PI / 2 * float64(d + 1) / float64(ramp + 1)) ^ 2)
      let a = -2 * PI * float64(k * (i - cp.maxLag)) / float64(cp.nfft)
      re += w * lags[i] * cos(a)
      im += w * lags[i] * sin(a)
    doAssert abs(amp[k] - hypot(re, im)) < 1e-9 * max(1.0, amp[k])
    doAssert abs(wrapDeg(phase[k] - radToDeg(arctan2(im, re)))) < 1e-6

block delay_peaks_at_its_lag:
  # The pilot 12 ms late: the correlation and its envelope peak at +12 ms,
  # the coefficient there is 1 (the same shape); scaled by 2.5, 2.5 by the
  # reference's energy.
  let cp = initCrossCorrelationPlan(float64, maxLagMs = 500, dtMs = 1, samples = n)
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var lags, env = newSeq[float64](cp.lags)
  let s = delayed(spec, n, 12)
  cp.crossCorrelateInto(s, pilot, lags, env, ws)
  doAssert cp.lagMs(argmax(lags)) == 12 and cp.lagMs(argmax(env)) == 12
  doAssert abs(lags[argmax(lags)] - 1) < 1e-6, $lags[argmax(lags)]
  doAssert abs(env[argmax(env)] - 1) < 1e-3, $env[argmax(env)]
  cp.crossCorrelateInto(delayed(spec, n, 12, amp = 2.5), pilot, lags, ws, cnReference)
  doAssert abs(lags[argmax(lags)] - 2.5) < 1e-5, $lags[argmax(lags)]
  # 90° out of phase (a quarter period at no single frequency: the Hilbert
  # transform of the pilot): the correlation is near zero at lag 0, the
  # envelope still peaks there.
  let hp = initHilbertPlan(float64, n)
  var hws = initWorkspace()
  hws.reserve(hp.workspaceSize)
  var quad = newSeq[float64](n)
  hp.hilbertInto(pilot, quad, hws)
  cp.crossCorrelateInto(quad, pilot, lags, env, ws)
  doAssert abs(lags[cp.maxLag]) < 1e-3 and cp.lagMs(argmax(env)) == 0
  doAssert abs(env[cp.maxLag] - 1) < 2e-3, $env[cp.maxLag]

block lag_spectrum_phase:
  # In band (12 .. 75 Hz) the phase is −360·f·τ for a delay of 3.5 ms (not a
  # whole sample), and 0 for the reference itself.
  let cp = initCrossCorrelationPlan(float64, maxLagMs = 500, dtMs = 1, samples = n)
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var lags = newSeq[float64](cp.lags)
  var amp, phase = newSeq[float64](cp.bins)
  cp.crossCorrelateInto(pilot, pilot, lags, ws)
  cp.lagSpectrumInto(lags, amp, phase, ws)
  var seen = 0
  for k in 0 ..< cp.bins:
    let f = cp.binHz(k)
    if f > 12 and f < 75:
      doAssert abs(phase[k]) < 0.01, $f & ": " & $phase[k]
      inc seen
  doAssert seen > 50
  let tau = 3.5
  cp.crossCorrelateInto(delayed(spec, n, tau), pilot, lags, ws)
  cp.lagSpectrumInto(lags, amp, phase, ws)
  for k in 0 ..< cp.bins:
    let f = cp.binHz(k)
    if f > 12 and f < 75:
      let want = wrapDeg(-360 * f * tau / 1000)
      doAssert abs(wrapDeg(phase[k] - want)) < 0.5, $f & ": " & $phase[k] & " vs " & $want
  # The amplitude is the band's: low outside it.
  var inBand, outBand = 0.0
  for k in 0 ..< cp.bins:
    let f = cp.binHz(k)
    if f > 20 and f < 70: inBand = max(inBand, amp[k])
    if f > 120: outBand = max(outBand, amp[k])
  doAssert outBand < 1e-3 * inBand

block comparison_gain_and_delay:
  # 0.4 × the pilot 2.5 ms late against the pilot: −7.96 dB and −360·f·τ in
  # band; the pilot against itself: 0 dB, 0°.
  let cp = initCrossCorrelationPlan(float64, maxLagMs = 400, dtMs = 1, samples = n)
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var ratio, phase = newSeq[float64](cp.bins)
  cp.compareInto(delayed(spec, n, 2.5, amp = 0.4), pilot, ratio, phase, ws)
  var seen = 0
  for k in 0 ..< cp.bins:
    let f = cp.binHz(k)
    if f > 15 and f < 75:
      doAssert abs(ratio[k] - 20 * log10(0.4)) < 0.1, $f & ": " & $ratio[k]
      doAssert abs(wrapDeg(phase[k] + 360 * f * 2.5 / 1000)) < 0.5, $f & ": " & $phase[k]
      inc seen
  doAssert seen > 40
  cp.compareInto(pilot, pilot, ratio, phase, ws)
  for k in 0 ..< cp.bins:
    let f = cp.binHz(k)
    if f > 15 and f < 75: doAssert abs(ratio[k]) < 1e-9 and abs(phase[k]) < 1e-6

block no_energy_and_nan:
  let cp = initCrossCorrelationPlan(float64, maxLagMs = 50, dtMs = 1, samples = 500)
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var zeros = newSeq[float64](500)
  var x = newSeq[float64](500)
  x[10] = 1
  var lags, env = newSeq[float64](cp.lags)
  cp.crossCorrelateInto(x, zeros, lags, env, ws)
  doAssert lags[0].isNaN and env[cp.maxLag].isNaN
  cp.crossCorrelateInto(x, zeros, lags, ws, cnReference)
  doAssert lags[cp.maxLag].isNaN
  cp.crossCorrelateInto(x, zeros, lags, ws, cnNone)
  doAssert lags[cp.maxLag] == 0
  var ratio, phase = newSeq[float64](cp.bins)
  cp.compareInto(x, zeros, ratio, phase, ws)
  doAssert ratio[3].isNaN and phase[3].isNaN
  x[20] = NaN
  cp.crossCorrelateInto(x, x, lags, ws)
  doAssert lags[cp.maxLag].isNaN
  var amp = newSeq[float64](cp.bins)
  lags[0] = NaN
  cp.lagSpectrumInto(lags, amp, phase, ws)
  doAssert amp[1].isNaN

block float32_plan:
  let cp = initCrossCorrelationPlan(float32, maxLagMs = 300, dtMs = 1, samples = n)
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var p32, s32 = newSeq[float32](n)
  let s = delayed(spec, n, 7)
  for j in 0 ..< n:
    p32[j] = float32(pilot[j])
    s32[j] = float32(s[j])
  var lags, env = newSeq[float32](cp.lags)
  cp.crossCorrelateInto(s32, p32, lags, env, ws)
  var best = 0
  for i in 1 ..< cp.lags:
    if lags[i] > lags[best]: best = i
  doAssert cp.lagMs(best) == 7 and abs(lags[best] - 1) < 1e-4

block refusals_and_shapes:
  doAssertRaises(ValueError): discard initCrossCorrelationPlan(float64, 100, 0, 1000)
  doAssertRaises(ValueError): discard initCrossCorrelationPlan(float64, 100, 2, 1)
  doAssertRaises(ValueError): discard initCrossCorrelationPlan(float64, 1, 2, 1000) # under a sample
  doAssertRaises(ValueError): discard initCrossCorrelationPlan(float64, 2000, 2, 1000)
  # Lags past an int's range: a ValueError too (before: an OverflowDefect, or
  # in -d:danger a plan of maxLag int.low).
  for lag in [1e19, 1e300]:
    doAssertRaises(ValueError): discard initCrossCorrelationPlan(float64, lag, 1, 1000)
  doAssertRaises(ValueError): discard initCrossCorrelationPlan(float64, 100, 2, 1000,
      taperFraction = 0.6)
  doAssertRaises(ValueError): discard initCrossCorrelationPlan(float64, 100, 2, 1000, nfft = 64)
  doAssertRaises(ValueError): discard initCrossCorrelationPlan(float64, 100, 2, 1000, nfft = 107)
  let cp = initCrossCorrelationPlan(float64, 100, 2, 1000, nfft = 120)
  doAssert cp.nfft == 120 and cp.bins == 61 and abs(cp.binHz(1) - 1000 / 240) < 1e-12
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var x = newSeq[float64](1001)
  var y = newSeq[float64](1000)
  var lags = newSeq[float64](cp.lags)
  doAssertRaises(AssertionDefect): cp.crossCorrelateInto(x, y, lags, ws)
  var short = newSeq[float64](cp.lags - 1)
  doAssertRaises(AssertionDefect): cp.crossCorrelateInto(y, y, short, ws)

block no_allocation:
  let cp = initCrossCorrelationPlan(float32, maxLagMs = 200, dtMs = 1, samples = n)
  var ws = initWorkspace()
  ws.reserve(cp.workspaceSize)
  var p32 = newSeq[float32](n)
  for j in 0 ..< n: p32[j] = float32(pilot[j])
  var lags, env = newSeq[float32](cp.lags)
  var amp, phase, ratio = newSeq[float64](cp.bins)
  cp.crossCorrelateInto(p32, p32, lags, env, ws)
  let count = allocsIn:
    cp.crossCorrelateInto(p32, p32, lags, env, ws)
    cp.lagSpectrumInto(lags, amp, phase, ws)
    cp.compareInto(p32, p32, ratio, phase, ws)
  doAssert count == 0, $count & " allocations"
