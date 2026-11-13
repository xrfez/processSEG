## Sweep-end tapers (task 10.1): each shape against its closed form at the
## end points, the midpoint and the derivative, lengths in ms, the overlap
## refusal, and float32/float64.

import std/[assertions, math]
import processSEG

func closeTo(a, b: float64, tol = 1e-12): bool = abs(a - b) <= tol

block closed_forms:
  # w(0), w(0.5), w(1) and w'(1) (zero slope at full amplitude, except the
  # linear, cosine and power ramps).
  let half = [
    (tpLinear, 0.0, 0.5), (tpCosine, 0.0, sin(PI / 4)), (tpCos2, 0.0, 0.5),
    (tpHamming, 0.08, 0.54), (tpBlackman, 0.0, 0.42 - 0.08),
    (tpBlackmanHarris, 0.35875 - 0.48829 + 0.14128 - 0.01168, 0.35875 - 0.14128)]
  for (shape, w0, wHalf) in half:
    doAssert closeTo(taperWeight(shape, 0), w0), $shape
    doAssert closeTo(taperWeight(shape, 0.5), wHalf), $shape
    doAssert closeTo(taperWeight(shape, 1), 1), $shape
  # Kaiser: I0(β·√(1 − (1 − x)²)) / I0(β); I0(6) = 67.23440697647797.
  doAssert closeTo(taperWeight(tpKaiser, 0, 6), 1 / 67.23440697647797, 1e-13)
  doAssert closeTo(taperWeight(tpKaiser, 1, 6), 1)
  # Gaussian with σ = 0.5: exp(−½·(1/0.5)²) = exp(−2) at x = 0.
  doAssert closeTo(taperWeight(tpGaussian, 0, 0.5), exp(-2.0))
  doAssert closeTo(taperWeight(tpGaussian, 0.5, 0.5), exp(-0.5))
  doAssert closeTo(taperWeight(tpPower, 0.5, 3), 0.125)
  doAssert closeTo(taperWeight(tpPower, 0.5), 0.25) # default p = 2
  # Derivatives at x = 1 (central difference just inside, one-sided).
  let h = 1e-6
  for shape in [tpCos2, tpHamming, tpBlackman, tpBlackmanHarris, tpKaiser, tpGaussian]:
    let d = (taperWeight(shape, 1) - taperWeight(shape, 1 - h)) / h
    doAssert abs(d) < 1e-4, $shape & " slope at the top " & $d
  doAssert closeTo((taperWeight(tpLinear, 1) - taperWeight(tpLinear, 1 - h)) / h, 1, 1e-6)
  doAssert closeTo((taperWeight(tpCosine, h) - taperWeight(tpCosine, 0)) / h, PI / 2, 1e-5)
  # Monotonic rise for every shape.
  for shape in TaperShape:
    var prev = -1.0
    for i in 0 .. 100:
      let w = taperWeight(shape, i / 100)
      doAssert w >= prev - 1e-15, $shape
      prev = w

block apply_lengths_in_ms:
  var x = newSeq[float64](101) # 0 .. 200 ms at 2 ms
  for v in x.mitems: v = 1
  x.applyTaper(2, initSweepTaper(20, 40, tpLinear))
  doAssert x[0] == 0 and closeTo(x[5], 0.5) and x[10] == 1
  doAssert x[100] == 0 and closeTo(x[90], 0.5) and x[80] == 1
  # A taper that is not a whole number of samples: 5 ms at 2 ms.
  var y = newSeq[float32](11)
  for v in y.mitems: v = 1
  y.applyTaper(2, initSweepTaper(5, NaN, tpCos2))
  doAssert y[0] == 0 and closeTo(y[1], 0.5 - 0.5 * cos(PI * 0.4), 1e-7)
  doAssert closeTo(y[2], 0.5 - 0.5 * cos(PI * 0.8), 1e-7) and y[3] == 1 and y[10] == 1

block zero_and_nan_lengths_are_no_ops:
  var x = @[1.0, 2.0, 3.0, 4.0]
  x.applyTaper(1, initSweepTaper(0, NaN))
  doAssert x == @[1.0, 2.0, 3.0, 4.0]
  x.applyTaper(1, noTaper())
  doAssert x == @[1.0, 2.0, 3.0, 4.0]

block overlap_is_refused_meeting_is_not:
  var x = newSeq[float64](11) # 10 ms at 1 ms
  for v in x.mitems: v = 1
  doAssertRaises(ValueError): x.applyTaper(1, initSweepTaper(6, 5))
  x.applyTaper(1, initSweepTaper(5, 5, tpLinear))
  doAssert x[5] == 1 and x[0] == 0 and x[10] == 0 and closeTo(x[4], 0.8)

block bad_arguments:
  doAssertRaises(ValueError): discard initSweepTaper(-1, 0)
  doAssertRaises(ValueError): discard initSweepTaper(Inf, 0)
  doAssertRaises(ValueError): discard initSweepTaper(1, 1, tpGaussian, 0)
  doAssertRaises(ValueError): discard initSweepTaper(1, 1, tpKaiser, -1)
  doAssertRaises(ValueError): discard initSweepTaper(1, 1, tpPower, 0)
  var x = @[1.0, 1.0]
  doAssertRaises(ValueError): x.applyTaper(0, noTaper())

block segy_codes:
  doAssert tpLinear.segyTaperType == 1 and tpCos2.segyTaperType == 2
  for s in [tpCosine, tpHamming, tpBlackman, tpBlackmanHarris, tpKaiser, tpGaussian, tpPower]:
    doAssert s.segyTaperType == 3

block a_taper_built_as_an_object_takes_its_shape_default:
  # `param` is NaN unless given, and NaN is the shape's default wherever a
  # taper is used: as a plain zero it made a Kaiser taper of β = 0, which
  # tapers nothing, and a Gaussian or power one invalid.
  for shape in [tpKaiser, tpGaussian, tpPower]:
    let built = SweepTaper(startMs: 4, endMs: 4, shape: shape)
    doAssert built.param.isNaN
    built.validate
    var x, y = newSeq[float64](21)
    for j in 0 ..< x.len:
      x[j] = 1
      y[j] = 1
    x.applyTaper(1, built)
    y.applyTaper(1, initSweepTaper(4, 4, shape))
    doAssert x == y and x[1] < 0.9 and x[10] == 1, $shape & ": " & $x[0 .. 4]
  doAssert SweepTaper().param.isNaN and noTaper().param.isNaN
