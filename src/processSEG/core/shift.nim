## Fractional time shifts of one row, in place: `x[j] ← x(j + shift)`, by
## linear or 4-point cubic (Catmull-Rom) interpolation, with zeros outside the
## row. Used by LMO apply and remove (`velocity/moveout`) and, later, statics.
##
## ```nim
## shiftSamples(row, 2.5)               # events move 2.5 samples earlier
## shiftSamples(row, -2.5, siCubic)     # and back
## ```
##
## Interpolation smooths. At a half-sample shift, the worst case, linear
## interpolation has the gain `|cos(πf·dt)|` (−3 dB at half Nyquist), the
## cubic −3 dB near 0.65 Nyquist. Integer shifts are exact with both.
## A shift there and back loses the samples moved out of the row.

import std/math
import parseSEG

type
  ShiftInterp* = enum
    ## How a fractional shift interpolates between samples.
    siLinear ## 2 points
    siCubic  ## 4 points, Catmull-Rom (Keys, a = −0.5)

func cubicWeights(f: float64): array[4, float64] {.inline.} =
  ## Catmull-Rom weights of samples −1, 0, 1, 2 at fraction `f` in [0, 1).
  let f2 = f * f
  let f3 = f2 * f
  [-0.5 * f3 + f2 - 0.5 * f,
   1.5 * f3 - 2.5 * f2 + 1,
   -1.5 * f3 + 2 * f2 + 0.5 * f,
   0.5 * f3 - 0.5 * f2]

proc shiftSamples*[T: SampleType](x: var openArray[T], shift: float64,
    interp = siLinear) {.raises: [].} =
  ## Shifts `x` in place so that `x[j]` becomes the old `x` at `j + shift`
  ## (positive: earlier; negative: later), zero where that falls outside
  ## the row. A NaN or infinite shift leaves `x` untouched; one beyond the
  ## interpolation's reach zeroes it. Allocates nothing.
  let n = x.len
  if n == 0 or shift.classify in {fcNan, fcInf, fcNegInf}: return
  # The cubic reaches one sample further than linear interpolation.
  if abs(shift) >= float64(n + ord(interp == siCubic)):
    for v in x.mitems: v = T(0)
    return
  let k = int(floor(shift))
  let f = shift - float64(k)
  # Reading ahead of the write position (shift ≥ 0, ascending) or behind it
  # (shift < 0, descending) leaves the samples read unwritten, except the one
  # next to the write position, which the cubic reads: it is kept in `saved`.
  template sample(m, j: int, saved: float64, savedAt: int): float64 =
    if m < 0 or m >= n: 0.0
    elif m == savedAt: saved
    else: float64(x[m])
  template body(j: int, saved: var float64, savedAt: int) =
    let m = j + k
    var y: float64
    case interp
    of siLinear:
      y = (1 - f) * sample(m, j, saved, savedAt) +
        (if f > 0: f * sample(m + 1, j, saved, savedAt) else: 0.0)
    of siCubic:
      if f == 0:
        y = sample(m, j, saved, savedAt)
      else:
        let w = cubicWeights(f)
        y = w[0] * sample(m - 1, j, saved, savedAt) + w[1] * sample(m, j,
            saved, savedAt) + w[2] * sample(m + 1, j, saved, savedAt) +
            w[3] * sample(m + 2, j, saved, savedAt)
    saved = float64(x[j])
    x[j] = T(y)
  {.push boundChecks: off.}
  var saved = 0.0
  if shift >= 0:
    for j in 0..<n:
      body(j, saved, j - 1)
  else:
    for j in countdown(n - 1, 0):
      body(j, saved, j + 1)
  {.pop.}
