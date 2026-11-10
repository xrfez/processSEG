## Filter responses (filter/design): passband ripple, stopband level and the
## −6 dB corner points of the cos² trapezoid, Butterworth magnitude and notch;
## validation; and the impulse length (99.9 % energy) against the ramp width
## (the bin spacing is checked by the plan, tfilterapply). The −3 dB corner of a single (minimum-phase) pass is
## tested in tiir.

import std/[assertions, math]
import processSEG

func db(g: float64): float64 = 20 * log10(g)
func near(a, b, tol: float64): bool = abs(a - b) <= tol

proc sweep(spec: FilterSpec, lo, hi: float64, steps = 2000): (float64, float64) =
  ## The minimum and maximum gain over lo .. hi.
  result = (Inf, -Inf)
  for i in 0..steps:
    let g = spec.response(lo + (hi - lo) * float64(i) / float64(steps))
    result = (min(result[0], g), max(result[1], g))

block trapezoid:
  let bp = bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80))
  let (pmin, pmax) = bp.sweep(8, 60)
  doAssert pmin == 1 and pmax == 1, "no passband ripple"
  doAssert bp.sweep(0, 4)[1] == 0 and bp.sweep(80, 250)[1] == 0, "stopband is 0"
  doAssert near(db(bp.response(6)), db(0.5), 1e-9), "−6 dB mid-ramp (low)"
  doAssert near(db(bp.response(70)), db(0.5), 1e-9), "−6 dB mid-ramp (high)"
  # Monotonic ramps.
  var prev = 0.0
  for i in 0..400:
    let g = bp.response(4 + 4 * float64(i) / 400)
    doAssert g >= prev
    prev = g
  let lc = lowCut(0.5, 1)
  doAssert lc.response(0) == 0 and near(lc.response(0.75), 0.5, 1e-15) and lc.response(400) == 1
  let hc = highCut(60, 80)
  doAssert hc.response(0) == 1 and near(hc.response(70), 0.5, 1e-15) and hc.response(80) == 0
  doAssert lowCut(0, 2).response(0) == 0 and near(lowCut(0, 2).response(1), 0.5, 1e-15)

block butterworth:
  # Effective slope 24 dB/octave: |H|² of 2 poles, −6 dB at the corner.
  let hc = butterHighCut(50, slopeDbPerOctave = 24)
  doAssert near(db(hc.response(50)), db(0.5), 1e-9)
  # Maximally flat: below fc/4, within 20·log10(1 + 4^−4) = 0.034 dB (k = 4).
  doAssert db(hc.sweep(0, 12.5)[0]) >= -0.0339
  # Stopband: the asymptotic slope is 24 dB/octave (6 dB per unit of k).
  let a1 = db(hc.response(400))
  let a2 = db(hc.response(800))
  doAssert near(a1 - a2, 24.08, 0.01), $(a1 - a2)
  let lc = butterLowCut(4, slopeDbPerOctave = 36)
  doAssert lc.response(0) == 0 and near(db(lc.response(4)), db(0.5), 1e-9)
  doAssert near(db(lc.response(1)) - db(lc.response(0.5)), 36.12, 0.01)
  let bp = butterBandPass(5, 80, 24, 48)
  doAssert near(db(bp.response(5)), -6.02, 0.05) and near(db(bp.response(80)), -6.02, 0.05)
  doAssert bp.response(20) > 0.99

block notch_and_zero_dc:
  let s = bandPass(Corners(f1: 2, f2: 4, f3: 100, f4: 120)).withNotch(50, 4)
  doAssert s.response(50) == 0 and near(db(s.response(49)), db(0.5), 1e-9)
  doAssert s.response(48) == 1 and s.response(52) == 1
  doAssert s.notches.len == 1
  let z = highCut(60, 80).withZeroDc
  var h = newSeq[float64](65)
  z.responseInto(dtMs = 2, nfft = 128, h)
  doAssert h[0] == 0 and h[1] == 1, "bin 0 only"
  doAssert near(h[64], z.response(250), 0) and h[64] == 0
  doAssert near(h[18], z.response(18 * 1000 / 256), 0)

block validation:
  for c in [Corners(f1: 8, f2: 4, f3: 60, f4: 80), Corners(f1: 4, f2: 4, f3: 60, f4: 80),
      Corners(f1: 4, f2: 8, f3: 80, f4: 80), Corners(f1: 4, f2: 70, f3: 60, f4: 80),
      Corners(f1: NaN, f2: 8, f3: 60, f4: 80), Corners(f1: -1, f2: 8, f3: 60, f4: 80),
      Corners(f1: 1, f2: 0, f3: 60, f4: 80), Corners(f1: 4, f2: 8, f3: Inf, f4: 80)]:
    doAssertRaises ValueError: discard bandPass(c)
  for (lo, hi, s) in [(5.0, 5.0, 24.0), (10.0, 5.0, 24.0), (5.0, 50.0, 0.0),
      (5.0, 50.0, -6.0), (5.0, 50.0, Inf), (-1.0, 50.0, 24.0), (NaN, 50.0, 24.0)]:
    doAssertRaises ValueError: discard butterBandPass(lo, hi, s)
  doAssertRaises ValueError: discard butterLowCut(0)
  doAssertRaises ValueError: discard butterHighCut(Inf)
  doAssertRaises ValueError: discard lowCut(1, 2).withNotch(0, 1)
  doAssertRaises ValueError: discard lowCut(1, 2).withNotch(50, 0)
  # Against Nyquist (250 Hz at 2 ms).
  bandPass(Corners(f1: 4, f2: 8, f3: 200, f4: 250)).validate(2)
  doAssertRaises ValueError: bandPass(Corners(f1: 4, f2: 8, f3: 200, f4: 251)).validate(2)
  doAssertRaises ValueError: lowCut(200, 250).validate(2)
  doAssertRaises ValueError: butterHighCut(250).validate(2)
  doAssertRaises ValueError: lowCut(1, 2).withNotch(250, 2).validate(2)
  doAssertRaises ValueError: lowCut(1, 2).validate(0)
  var bad = lowCut(1, 2)
  bad.corners.f1 = 3 # built by hand
  doAssertRaises ValueError: bad.validate(2)

block impulse_length:
  # A cos² ramp of width w gives L ≈ c/(w·dt): halving the ramp doubles L.
  let a = lowCut(1, 2).impulseLength(dtMs = 2)
  let b = lowCut(1.5, 2).impulseLength(dtMs = 2)
  doAssert a > 0 and near(float64(b) / float64(a), 2, 0.25), $a & " " & $b
  # About 1/(w·dt) samples: 1 Hz at 2 ms is 500 samples.
  doAssert a > 100 and a < 1500, $a
  # The same filter at half the interval has twice the samples.
  let c = lowCut(1, 2).impulseLength(dtMs = 1)
  doAssert near(float64(c) / float64(a), 2, 0.05), $c & " " & $a
  # Flat responses have no length; zeroDc does not count.
  doAssert bandPass(Corners(f1: 0, f2: 0, f3: Inf, f4: Inf)).impulseLength(2) == 0
  doAssert lowCut(1, 2).withZeroDc.impulseLength(2) == a
  # Wider ramps, shorter filters; Butterworth and notches too.
  doAssert highCut(60, 80).impulseLength(2) < a div 10
  let bw = butterLowCut(2, 24).impulseLength(2)
  doAssert bw > 0 and bw < 2000, $bw
  doAssert butterLowCut(2, 48).impulseLength(2) > bw, "steeper is longer"
  let nt = highCut(100, 120).withNotch(50, 1).impulseLength(2)
  doAssert nt > highCut(100, 120).impulseLength(2)
  # Too narrow for a sane grid.
  doAssertRaises ValueError: discard lowCut(1, 1.00001).impulseLength(0.01)
