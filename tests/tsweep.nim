## Sweep generation (task 10.2): the instantaneous frequency of every law
## (from the phase derivative and a short-time spectrum), the dB/octave and
## dB/Hz spectral slopes, the Klauder wavelet's width, phase accuracy, and
## validation.

import std/[assertions, math]
import processSEG

proc instFreq(spec: SweepSpec): seq[float64] =
  ## Mean frequency over each sample interval from the analytic phase:
  ## the sweep at φ₀ = 0 (sine) and at 90° (cosine).
  var s = spec
  s.taper = noTaper()
  let a = s.sweepSamples(float64)
  s.startPhaseDeg = 90
  let b = s.sweepSamples(float64)
  result = newSeq[float64](a.len - 1)
  for j in 0 ..< a.len - 1:
    var d = arctan2(a[j + 1], b[j + 1]) - arctan2(a[j], b[j])
    while d < 0: d += 2 * PI
    while d >= 2 * PI: d -= 2 * PI
    result[j] = d / (2 * PI) / (spec.dtMs / 1000)

proc checkLaw(spec: SweepSpec, tol: float64) =
  let f = instFreq(spec)
  var worst = 0.0
  for j in 0 ..< f.len:
    # The mean of the law over the interval (Simpson), as the phase step is.
    let a = float64(j) * spec.dtMs
    let want = (spec.frequencyAt(a) + 4 * spec.frequencyAt(a + 0.5 * spec.dtMs) +
      spec.frequencyAt(a + spec.dtMs)) / 6
    worst = max(worst, abs(f[j] - want))
  doAssert worst < tol, $spec.law & ": instantaneous frequency off by " & $worst & " Hz"

block instantaneous_frequency_follows_the_law:
  let taper = initSweepTaper(200, 200)
  checkLaw(linearSweep(8, 80, 6000, 2, taper), 1e-6)
  checkLaw(linearSweep(90, 6, 6000, 2, taper), 1e-6) # down-sweep
  checkLaw(exponentialSweep(5, 120, 8000, 2, taper), 1e-4)
  checkLaw(dbPerOctaveSweep(6, 100, 8000, 2, 3, taper), 1e-4)
  checkLaw(dbPerOctaveSweep(100, 6, 8000, 2, -4.5, taper), 1e-4)
  checkLaw(dbPerHzSweep(6, 100, 8000, 2, 0.1, taper), 1e-4)
  checkLaw(dbPerHzSweep(6, 100, 8000, 1, -0.2, taper), 1e-4)
  checkLaw(tableSweep([SweepPoint(tMs: 0, fHz: 10), SweepPoint(tMs: 1500, fHz: 20),
    SweepPoint(tMs: 4000, fHz: 90)], 2), 1e-6)
  # The law's end points.
  let d = dbPerHzSweep(6, 100, 8000, 2, 0.1)
  doAssert abs(d.frequencyAt(0) - 6) < 1e-12 and abs(d.frequencyAt(8000) - 100) < 1e-9
  let o = dbPerOctaveSweep(6, 100, 8000, 2, 3)
  doAssert abs(o.frequencyAt(0) - 6) < 1e-12 and abs(o.frequencyAt(8000) - 100) < 1e-9
  # A +3 dB/octave boost dwells on the high end: it gets there early and
  # spends the second half above the midband (f ∝ √t here: 70.8 Hz).
  doAssert abs(o.frequencyAt(4000) - sqrt((36.0 + 10000) / 2)) < 0.05

block short_time_spectrum_peak:
  let spec = dbPerOctaveSweep(8, 100, 10000, 2, 3)
  let x = spec.sweepSamples(float64)
  const win = 256
  let sp = initSpectrumPlan(float64, win, 2, taperFraction = 0.5, nfft = 2048)
  var ws = initWorkspace()
  ws.reserve(sp.workspaceSize)
  var amp, ph = newSeq[float64](sp.bins)
  for c in countup(600, 4400, 400):
    sp.amplitudeInto(x.toOpenArray(c - win div 2, c + win div 2 - 1), amp, ph, ws)
    var best = 0
    for k in 1 ..< sp.bins:
      if amp[k] > amp[best]: best = k
    let want = spec.frequencyAt(float64(c) * 2)
    # One bin (0.24 Hz) plus the frequency change over the window's centre.
    doAssert abs(sp.binHz(best) - want) < 1.0, "at " & $c & ": " & $sp.binHz(best) &
      " Hz, want " & $want

proc powerSpectrum(x: seq[float64], n: int): seq[float64] =
  let p = initFftPlan(float64, n)
  var ws = initWorkspace()
  ws.reserve(p.workspaceSize(batched = false))
  var buf = newSeq[float64](n)
  for j in 0 ..< x.len: buf[j] = x[j]
  var re, im = newSeq[float64](p.bins)
  p.forward(buf, re, im, ws)
  result = newSeq[float64](p.bins)
  for k in 0 ..< p.bins: result[k] = re[k] * re[k] + im[k] * im[k]

proc bandDb(pw: seq[float64], binHz, lo, hi: float64): float64 =
  ## Mean power in [lo, hi] Hz, in dB (amplitude scale: 10·log10 of power).
  var s = 0.0
  var c = 0
  for k in 0 ..< pw.len:
    let f = float64(k) * binHz
    if f >= lo and f <= hi:
      s += pw[k]
      inc c
  10 * log10(s / float64(c))

block db_per_octave_slope:
  for boost in [-3.0, 3.0, 6.0]:
    let spec = dbPerOctaveSweep(5, 200, 16000, 1, boost, initSweepTaper(50, 50))
    let n = 32768
    let pw = powerSpectrum(spec.sweepSamples(float64), n)
    let binHz = 1000 / (float64(n) * 1)
    # Third-octave bands at 40, 80, 160 Hz, away from the tapered ends (a
    # +6 dB/octave sweep is past 20 Hz 16 ms in, inside the start ramp).
    var levels: seq[float64]
    for fc in [40.0, 80.0, 160.0]:
      levels.add bandDb(pw, binHz, fc / pow(2, 1 / 6), fc * pow(2, 1 / 6))
    for i in 1 ..< levels.len:
      let slope = levels[i] - levels[i - 1]
      doAssert abs(slope - boost) < 0.5, "boost " & $boost & ": " & $slope & " dB/octave"

block db_per_hz_slope:
  let boost = 0.1
  let spec = dbPerHzSweep(5, 125, 16000, 1, boost, initSweepTaper(300, 300))
  let n = 32768
  let pw = powerSpectrum(spec.sweepSamples(float64), n)
  let binHz = 1000 / float64(n)
  let a = bandDb(pw, binHz, 25, 35)
  let b = bandDb(pw, binHz, 85, 95)
  doAssert abs((b - a) / 60 - boost) < 0.01, $((b - a) / 60) & " dB/Hz"

block klauder_wavelet:
  # The autocorrelation of a linear sweep: peak at lag 0, envelope (with the
  # quadrature sweep) first null near 1/(f2 − f1).
  let spec = linearSweep(10, 60, 4000, 1)
  let s = spec.sweepSamples(float64)
  var q = spec
  q.startPhaseDeg = 90
  let c = q.sweepSamples(float64)
  proc corr(a, b: seq[float64], lag: int): float64 =
    for j in 0 ..< a.len - lag: result += a[j + lag] * b[j]
  let r0 = corr(s, s, 0)
  var envPrev = Inf
  var null = -1
  for lag in 1 .. 60:
    let r = corr(s, s, lag)
    doAssert r < r0
    # The analytic sweep c + i·s: |Σ z(t + τ)·z*(t)| is twice the envelope.
    let env = hypot(r + corr(c, c, lag), corr(s, c, lag) - corr(c, s, lag)) / 2
    if env > envPrev and null < 0: null = lag - 1
    envPrev = env
  # 1/(60 − 10) Hz = 20 ms = 20 samples.
  doAssert null >= 18 and null <= 22, "first envelope null at " & $null & " ms"
  # Zero phase: the autocorrelation is symmetric, R(−τ) = R(τ).
  for lag in 1 .. 30:
    var neg = 0.0
    for j in lag ..< s.len: neg += s[j - lag] * s[j]
    doAssert abs(neg - corr(s, s, lag)) < 1e-9 * r0

block phase_accuracy:
  # Linear: the samples are sin of the closed-form phase.
  let lin = linearSweep(8, 80, 8000, 2, startPhaseDeg = 30)
  let x = lin.sweepSamples(float64)
  for j in [0, 1, 1000, 2500, 4000]:
    doAssert abs(x[j] - sin(lin.phaseAt(float64(j) * 2))) < 1e-12
  # Exponential through Gauss–Legendre against its closed form (sweepSamples
  # integrates numerically only for dB laws: compare a dB/octave sweep of
  # −3.0103 dB/octave, which is exponential, with the exponential one).
  let e = exponentialSweep(5, 120, 8000, 2)
  let d = dbPerOctaveSweep(5, 120, 8000, 2, -20 * log10(2.0) / 2)
  let xe = e.sweepSamples(float64)
  let xd = d.sweepSamples(float64)
  for j in 0 ..< xe.len:
    doAssert abs(xe[j] - xd[j]) < 1e-9, $j
    doAssert abs(xe[j] - sin(e.phaseAt(float64(j) * 2))) < 1e-9
  doAssert e.segySweepType == 3 and d.segySweepType == 3 and lin.segySweepType == 1
  # A two-point table is the linear sweep.
  let t = tableSweep([SweepPoint(tMs: 0, fHz: 8), SweepPoint(tMs: 8000, fHz: 80)], 2)
  let xt = t.sweepSamples(float64)
  let xl = linearSweep(8, 80, 8000, 2).sweepSamples(float64)
  for j in 0 ..< xt.len: doAssert abs(xt[j] - xl[j]) < 1e-9
  doAssert t.segySweepType == 4
  # float32 and float64 agree.
  let x32 = lin.sweepSamples(float32)
  for j in 0 ..< x.len: doAssert abs(float64(x32[j]) - x[j]) < 1e-6

block lengths_and_amplitude:
  let s = linearSweep(8, 80, 8000, 2, amplitude = 2.5, taper = initSweepTaper(100, 100))
  doAssert s.sweepLen == 4001
  let x = s.sweepSamples(float32)
  doAssert x[0] == 0 and x[^1] == 0
  var peak = 0.0
  for v in x: peak = max(peak, abs(float64(v)))
  doAssert peak <= 2.5 and peak > 2.49
  doAssert linearSweep(8, 80, 0.3, 0.1).sweepLen == 4 # 0.3 / 0.1 is 2.9999999999999996
  var buf = newSeq[float64](10)
  doAssertRaises(ValueError): s.sweepInto(buf)

block validation:
  doAssertRaises(ValueError): discard linearSweep(8, 210, 8000, 2) # > 0.8 · 250
  doAssertRaises(ValueError): discard linearSweep(8, 260, 8000, 2, maxNyquistFraction = 1)
  discard linearSweep(8, 240, 8000, 2, maxNyquistFraction = 1) # allowed
  doAssertRaises(ValueError): discard linearSweep(0, 80, 8000, 2)
  doAssertRaises(ValueError): discard linearSweep(8, 80, 1, 2) # shorter than dt
  doAssertRaises(ValueError): discard linearSweep(8, 80, 8000, 0)
  doAssertRaises(ValueError): discard linearSweep(8, 80, 8001, 2) # not whole samples
  doAssertRaises(ValueError): discard linearSweep(8, 80, 1e12, 2) # over 2³¹ samples
  doAssertRaises(ValueError): discard linearSweep(8, 80, 1e300, 2)
  doAssertRaises(ValueError): discard linearSweep(8, 80, 1000, 2, initSweepTaper(600, 500))
  doAssertRaises(ValueError): discard dbPerHzSweep(8, 80, 8000, 2, 10) # 720 dB
  doAssertRaises(ValueError): discard dbPerOctaveSweep(8, 80, 8000, 2, NaN)
  doAssertRaises(ValueError):
    discard tableSweep([SweepPoint(tMs: 1, fHz: 8), SweepPoint(tMs: 8000, fHz: 80)], 2)
  doAssertRaises(ValueError):
    discard tableSweep([SweepPoint(tMs: 0, fHz: 8), SweepPoint(tMs: 0, fHz: 80)], 2)
  doAssertRaises(ValueError): discard tableSweep([SweepPoint(tMs: 0, fHz: 8)], 2)
  var bad = linearSweep(8, 80, 8000, 2)
  bad.amplitude = NaN
  doAssertRaises(ValueError): bad.validate
