## Butterworth biquad cascades (filter/iir): the magnitude matches the
## frequency-domain response (zero phase) and the analytic |H| (minimum
## phase), the −3 dB / −6 dB corners, causality, symmetry, the steady-state
## start-up against DC and drift, rows of varying live length, and no
## allocation in a steady state.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

const
  dt = 2.0
  ns = 4000 # 2⁵·5³, for the FFT of the impulse responses

func db(g: float64): float64 = 20 * log10(g)
func near(a, b, tol: float64): bool = abs(a - b) <= tol

func maxAbs[T](x: openArray[T]): float64 =
  for v in x: result = max(result, abs(float64(v)))

func maxDiff[T](a, b: openArray[T], lo = 0, hi = -1): float64 =
  let h = if hi < 0: a.len - 1 else: hi
  for i in lo..h: result = max(result, abs(float64(a[i]) - float64(b[i])))

proc filtered[T, P](x: seq[T], plan: P): seq[T] =
  result = x
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  plan.applyFilter(result, ws)

proc clean(n = 2001): seq[float64] =
  result = zeroSignal[float64](n)
  result.addRicker(dt, 900, 25)
  result.addRicker(dt, 2400, 15, amp = -0.6)

proc spectrumOf(h: seq[float64]): seq[float64] =
  ## |FFT| of an impulse response (centred anywhere: the modulus ignores the
  ## shift).
  let p = initFftPlan(float64, h.len)
  var ws = initWorkspace()
  ws.reserve(p.workspaceSize(batched = false))
  var re, im = newSeq[float64](p.bins)
  p.forward(h, re, im, ws)
  result = newSeq[float64](p.bins)
  for k in 0..<p.bins: result[k] = hypot(re[k], im[k])

block design_and_corners:
  let lp = initIirPlan(float64, butterHighCut(60, 24), dt, 100)
  doAssert lp.order == 2 and lp.phase == ipZero
  doAssert near(db(lp.magnitudeAt(60)), db(0.5), 1e-9), "zero phase: −6 dB"
  let mp = initIirPlan(float64, butterHighCut(60, 24), dt, 100, phase = ipMinimum)
  doAssert mp.order == 4
  doAssert near(db(mp.magnitudeAt(60)), -3.0103, 1e-4), "minimum phase: −3 dB"
  doAssert near(mp.magnitudeAt(0), 1, 1e-12) and mp.magnitudeAt(250) < 1e-12
  # An odd order (18 dB/octave minimum phase: 3 poles) and a band-pass.
  let odd = initIirPlan(float64, butterLowCut(4, 18), dt, 100, phase = ipMinimum)
  doAssert odd.order == 3 and near(db(odd.magnitudeAt(4)), -3.0103, 1e-4)
  doAssert odd.magnitudeAt(0) < 1e-12 and near(odd.magnitudeAt(250), 1, 1e-12)
  let bp = initIirPlan(float64, butterBandPass(4, 60, 24, 48), dt, 100)
  doAssert bp.order == 6, $bp.order # 24/12 + 48/12 poles
  for f in [0.5, 2.0, 4.0, 10.0, 30.0, 60.0, 90.0, 200.0]:
    doAssert near(bp.magnitudeAt(f), bp.spec.response(f, dt), 1e-9),
      "zero phase = the spec's response at " & $f
  # Validation.
  doAssertRaises ValueError: discard initIirPlan(float64, lowCut(1, 2), dt, 100)
  doAssertRaises ValueError:
    discard initIirPlan(float64, butterLowCut(4).withNotch(50, 2), dt, 100)
  doAssertRaises ValueError:
    discard initIirPlan(float64, butterLowCut(4).withZeroDc, dt, 100)
  doAssertRaises ValueError: discard initIirPlan(float64, butterLowCut(4, 18), dt, 100)
  doAssertRaises ValueError:
    discard initIirPlan(float64, butterLowCut(4, 20), dt, 100, phase = ipMinimum)
  doAssertRaises ValueError: discard initIirPlan(float64, butterLowCut(4), dt, 0)
  doAssertRaises ValueError: discard initIirPlan(float64, butterHighCut(300), dt, 100)

block matches_frequency_domain_magnitude:
  # The impulse response of the zero-phase cascade has the spec's response,
  # which the FFT filter applies too: both filters give the same output.
  for s in [butterBandPass(4, 60, 24), butterLowCut(2, 24), butterHighCut(80, 48),
      butterBandPass(8, 100, 36 + 12, 24)]:
    let iir = initIirPlan(float64, s, dt, ns, trend = trNone)
    var x = zeroSignal[float64](ns)
    x.addSpike(ns div 2)
    let h = filtered(x, iir)
    let mag = spectrumOf(h)
    let df = 1000 / (float64(ns) * dt)
    var e = 0.0
    for k in 0..<mag.len: e = max(e, abs(mag[k] - s.response(float64(k) * df, dt)))
    doAssert e < 1e-4, $s.lowHz & "–" & $s.highHz & ": " & $e
    let fd = filtered(x, initFilterPlan(float64, s, dt, ns, trend = trNone))
    doAssert maxDiff(h, fd) < 1e-4 * maxAbs(fd), $maxDiff(h, fd)
  # Minimum phase: |H| of slope/6 poles, which is the square root of the
  # response of a spec with twice the slope.
  for (s, s2) in [(butterBandPass(4, 60, 24), butterBandPass(4, 60, 48)),
      (butterLowCut(3, 18), butterLowCut(3, 36))]:
    let iir = initIirPlan(float64, s, dt, ns, phase = ipMinimum, trend = trNone)
    var x = zeroSignal[float64](ns)
    x.addSpike(iir.requiredMargin + 1) # past the reach of the start pad's reflection
    let mag = spectrumOf(filtered(x, iir))
    let df = 1000 / (float64(ns) * dt)
    var e = 0.0
    for k in 0..<mag.len:
      let f = float64(k) * df
      e = max(e, abs(mag[k] - sqrt(s2.response(f, dt))))
      e = max(e, abs(mag[k] - iir.magnitudeAt(f)))
    doAssert e < 1e-4, $e

block causal_and_symmetric:
  var x = zeroSignal[float64](2001)
  x.addSpike(1000)
  let s = butterBandPass(4, 60, 24)
  let m = filtered(x, initIirPlan(float64, s, dt, 2001, phase = ipMinimum, trend = trNone))
  doAssert maxAbs(m.toOpenArray(0, 999)) == 0, "nothing before the spike"
  doAssert m[1000] > 0
  let z = filtered(x, initIirPlan(float64, s, dt, 2001, trend = trNone))
  var e = 0.0
  for k in 1..1000: e = max(e, abs(z[1000 + k] - z[1000 - k]))
  doAssert e < 1e-12, $e

block steady_state_start:
  # A large DC and a drift under a 2 Hz low-cut: the first 10 % match the
  # filtered clean signal. With nothing removed, the steady-state start and
  # the reflection alone handle the DC.
  let c = clean()
  for phase in [ipZero, ipMinimum]:
    for tr in [trNone, trMean, trLinear]:
      let plan = initIirPlan(float64, butterLowCut(2, 24), dt, 2001, phase, tr)
      let reference = filtered(c, plan)
      var dc = c
      dc.addDc(1000)
      let e = maxDiff(filtered(dc, plan), reference, 0, 199)
      doAssert e < 1e-6, $phase & " " & $tr & " DC: " & $e
      var drift = c
      drift.addDrift(dt, slopePerS = 20, start = 5)
      let d = maxDiff(filtered(drift, plan), reference, 0, 199)
      # Zero phase: the reflection carries a drift through; minimum phase
      # needs trLinear (its steady-state start assumes a constant).
      if tr == trLinear: doAssert d < 1e-9, $phase & " drift: " & $d
      elif phase == ipZero: doAssert d < 0.05, $tr & " drift: " & $d
  # A high-cut keeps the DC and the drift (gain 1 at 0 Hz), in both phases;
  # minimum phase delays the drift by its group delay at 0 Hz.
  for phase in [ipZero, ipMinimum]:
    var dc = c
    dc.addDrift(dt, slopePerS = 3, start = 7)
    let plan = initIirPlan(float64, butterHighCut(60), dt, 2001, phase)
    let y = filtered(dc, plan)
    let noTrend = filtered(c, plan)
    let (t0, t1) = (y[0] - noTrend[0], y[^1] - noTrend[^1])
    doAssert near(t1 - t0, 3 * 4.0, 1e-6), "the slope is kept"
    if phase == ipZero: doAssert near(t0, 7, 1e-6)
    else:
      # The delay: the ramp comes out late by G′(1)/G(1) samples, here a
      # few samples of 2 ms at 3 per second.
      doAssert t0 < 7 and t0 > 7 - 3 * 0.02, $t0

proc rowsMatchSingle(T: typedesc, phase: IirPhase, tol: float64) =
  const rows = 19
  var g = cmpGather[T](cmpX = 0, offsets = newSeq[float64](rows), 2001, dt)
  for r in 0..<rows:
    var row = clean()
    row.addDc(float64(r) - 9)
    row.addDrift(dt, slopePerS = float64(r mod 5))
    for j in 0..<2001: g.data[r, j] = T(row[j])
  for i, n in [2001, 1500, 700, 2, 1, 0, 1999]: g.setLive(3 * i, n)
  g.keys[4].status = tsDead
  let meta = initChunkMeta(g.rows, dtMs = dt, failed = [7])
  let plan = initIirPlan(T, butterBandPass(3, 70, 24), dt, 2001, phase)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  let before = g.data.clone()
  plan.applyFilter(g.data, meta, ws)
  for r in 0..<rows:
    let n = meta.liveSamples(r, 2001)
    if not meta.isLive(r):
      for j in 0..<2001: doAssert g.data[r, j] == before[r, j]
      continue
    var want = newSeq[T](n)
    for j in 0..<n: want[j] = before[r, j]
    plan.applyFilter(want, ws)
    var e = 0.0
    for j in 0..<n: e = max(e, abs(float64(g.data[r, j]) - float64(want[j])))
    doAssert e <= tol, $T & " " & $phase & " row " & $r & " (" & $n & "): " & $e
    for j in n..<2001: doAssert g.data[r, j] == before[r, j]
  let a = allocsIn:
    plan.applyFilter(g.data, meta, ws)
  doAssert a == 0, $a & " allocations"

block rows_of_varying_live_length:
  for phase in [ipZero, ipMinimum]:
    rowsMatchSingle(float64, phase, 1e-12)
    rowsMatchSingle(float32, phase, 1e-6)

block nan_and_shapes:
  var g = cmpGather[float32](cmpX = 0, offsets = newSeq[float64](3), 400, dt)
  for r in 0..<3:
    for j in 0..<400: g.data[r, j] = float32(sin(float64(j + r)))
  g.data[1, 200] = NaN
  let meta = initChunkMeta(g.rows, dtMs = dt)
  let plan = initIirPlan(float32, butterLowCut(4), dt, 400)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  plan.applyFilter(g.data, meta, ws)
  doAssert g.data[1, 0].isNaN
  for j in 0..<400: doAssert not g.data[0, j].isNaN and not g.data[2, j].isNaN
  let small = initIirPlan(float32, butterLowCut(4), dt, 300)
  ws.reserve(small.workspaceSize)
  doAssertRaises ValueError: small.applyFilter(g.data, meta, ws)
  var long = newSeq[float32](301)
  doAssertRaises AssertionDefect: small.applyFilter(long, ws)
  var p0: IirPlan[float32]
  doAssertRaises AssertionDefect: p0.applyFilter(long, ws)

block trace_ends_match_a_long_context:
  # As for the FFT filter: a zero-phase low-cut or band-pass gives at the
  # ends of a trace what it gives there inside a longer one (pads of zeros
  # and passes from rest; the reflected pads with a steady-state start gave
  # 48 % and 50 % here, now 7 % and 3 %).
  for (spec, limit) in [(butterBandPass(4, 60, 24), 0.15), (butterLowCut(3, 24), 0.08)]:
    let n = 2001
    let L = initIirPlan(float64, spec, dt, n).requiredMargin
    let ctx = 6 * L
    var err, sig = 0.0 # over the first and last 50 samples of 20 traces
    for seed in 1'i64..20:
      var long = zeroSignal[float64](n + 2 * ctx)
      long.addNoise(1, seed = seed)
      let truth = filtered(long, initIirPlan(float64, spec, dt, long.len))
      let got = filtered(long[ctx ..< ctx + n], initIirPlan(float64, spec, dt, n))
      for i in 0..<50:
        for j in [i, n - 1 - i]:
          err += (got[j] - truth[ctx + j]) ^ 2
          sig += truth[ctx + j] ^ 2
    doAssert sqrt(err / sig) < limit, $sqrt(err / sig) & " (limit " & $limit & ")"

block no_subnormal_state:
  # After the input stops, a section's state decays for ever; it must reach
  # exactly 0, not run on in subnormal numbers (slow on every CPU, by
  # microcode on some): a spike, then silence, at 0.25 ms through a 100 Hz
  # high-cut (subnormal from sample 6338 on for good, before the flush).
  for phase in [ipZero, ipMinimum]:
    var x = zeroSignal[float64](40_000)
    x.addSpike(10)
    let y = filtered(x, initIirPlan(float64, butterHighCut(100, 24), 0.25, x.len, phase,
      trNone))
    var subnormal = 0
    for v in y:
      if v.classify == fcSubnormal: inc subnormal
    doAssert subnormal < 600, $phase & ": " & $subnormal & " subnormal samples"
    doAssert y[30_000] == 0 and y[^1] == 0, $phase & ": the tail is exactly zero"
  # Rows in the lane kernel too (a block of them).
  var g = cmpGather[float32](cmpX = 0, offsets = newSeq[float64](30), 40_000, 0.25)
  for r in 0..<30: g.data[r, 10 + r] = 1
  let meta = initChunkMeta(g.rows, dtMs = 0.25)
  let plan = initIirPlan(float32, butterHighCut(100, 24), 0.25, 40_000, ipMinimum, trNone)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  plan.applyFilter(g.data, meta, ws)
  for r in 0..<30: doAssert g.data[r, 39_999] == 0

block zoom_window_with_margin:
  # A window decoded with requiredMargin (before it, for minimum phase; both
  # sides for zero phase) matches the whole trace filtered, over the window.
  const total = 4000
  for s in [butterLowCut(2, 24), butterLowCut(2, 48), butterBandPass(4, 60, 48)]:
    for phase in [ipZero, ipMinimum]:
      var x = zeroSignal[float64](total)
      x.addNoise(1, seed = 3)
      x.addRicker(dt, 3000, 25, amp = 5)
      x.addDrift(dt, 4, 1)
      let whole = filtered(x, initIirPlan(float64, s, dt, total, phase))
      let (a, b) = (1500, 2300)
      let plan = initIirPlan(float64, s, dt, total, phase)
      let m = plan.requiredMargin
      let lo = max(0, a - m)
      let hi = if phase == ipZero: min(total - 1, b + m) else: b
      let win = filtered(x[lo .. hi], plan)
      var e = 0.0
      for j in a..b: e = max(e, abs(win[j - lo] - whole[j]))
      let rel = e / maxAbs(whole.toOpenArray(a, b))
      doAssert rel < 0.01, $phase & " m = " & $m & ": " & $rel
