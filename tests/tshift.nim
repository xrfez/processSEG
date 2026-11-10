## Fractional shifts (core/shift): integer shifts exact, in place equal to
## a copy-based reference for every direction and fraction (the in-place
## read order), polynomials reproduced, zeros outside, NaN and large
## shifts, float32, and no allocation.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

func reference(x: openArray[float64], shift: float64, interp: ShiftInterp): seq[float64] =
  ## The shift computed from an untouched copy.
  let n = x.len
  template at(i: int): float64 = (if i >= 0 and i < n: x[i] else: 0.0)
  result = newSeq[float64](n)
  let k = int(floor(shift))
  let f = shift - float64(k)
  for j in 0..<n:
    let m = j + k
    case interp
    of siLinear: result[j] = (1 - f) * at(m) + f * at(m + 1)
    of siCubic:
      let w = [-0.5 * f^3 + f^2 - 0.5 * f, 1.5 * f^3 - 2.5 * f^2 + 1,
               -1.5 * f^3 + 2 * f^2 + 0.5 * f, 0.5 * f^3 - 0.5 * f^2]
      result[j] = w[0] * at(m - 1) + w[1] * at(m) + w[2] * at(m + 1) + w[3] *
          at(m + 2)

block in_place_equals_reference:
  var x0 = zeroSignal[float64](64)
  x0.addNoise(1, seed = 7)
  for interp in [siLinear, siCubic]:
    for shift in [0.0, 0.25, 0.5, 0.999, 1, 1.5, 3.75, 62.5, -0.001, -0.25,
                  -0.5, -1, -1.5, -3.25, -63.5]:
      var x = x0
      x.shiftSamples(shift, interp)
      let r = reference(x0, shift, interp)
      for j in 0..<x.len:
        doAssert abs(x[j] - r[j]) <= 1e-12, $interp & " " & $shift & " " & $j

block integer_shifts_exact:
  var x0 = zeroSignal[float32](20)
  for j in 0..<20: x0[j] = float32(j + 1)
  for interp in [siLinear, siCubic]:
    var x = x0
    x.shiftSamples(3, interp)
    for j in 0..<17: doAssert x[j] == x0[j + 3]
    for j in 17..<20: doAssert x[j] == 0
    x = x0
    x.shiftSamples(-2, interp)
    for j in 0..<2: doAssert x[j] == 0
    for j in 2..<20: doAssert x[j] == x0[j - 2]

block polynomials:
  # Linear interpolation reproduces lines, the cubic quadratics (inside).
  var line = newSeq[float64](30)
  var quad = newSeq[float64](30)
  for j in 0..<30:
    line[j] = 2 + 0.5 * float64(j)
    quad[j] = 1 - 0.3 * float64(j) + 0.02 * float64(j) ^ 2
  for s in [0.3, -0.7, 2.6]:
    var a = line
    a.shiftSamples(s)
    var b = quad
    b.shiftSamples(s, siCubic)
    for j in 4..<25:
      let t = float64(j) + s
      doAssert abs(a[j] - (2 + 0.5 * t)) <= 1e-12
      doAssert abs(b[j] - (1 - 0.3 * t + 0.02 * t ^ 2)) <= 1e-12

block band_limited_round_trip:
  # A low-frequency sinusoid (10 Hz at 2 ms: 0.04 Nyquist) shifted by 2.5
  # samples and back: the cubic keeps it within 1e-3, linear within 2 %.
  for (interp, tol) in [(siCubic, 1e-3), (siLinear, 2e-2)]:
    var x0 = zeroSignal[float64](500)
    x0.addSinusoid(2, 10, 1)
    var x = x0
    x.shiftSamples(2.5, interp)
    x.shiftSamples(-2.5, interp)
    for j in 10..<490: doAssert abs(x[j] - x0[j]) <= tol, $interp

block edges:
  var x = @[1.0, 2, 3]
  x.shiftSamples(NaN)
  doAssert x == @[1.0, 2, 3]
  x.shiftSamples(Inf)
  doAssert x == @[1.0, 2, 3]
  x.shiftSamples(3)
  doAssert x == @[0.0, 0, 0]
  x = @[1.0, 2, 3]
  x.shiftSamples(-5, siCubic)
  doAssert x == @[0.0, 0, 0]
  var e: seq[float64]
  e.shiftSamples(1.5)
  var y = zeroSignal[float32](100)
  doAssert allocsIn(y.shiftSamples(1.5, siCubic)) == 0
