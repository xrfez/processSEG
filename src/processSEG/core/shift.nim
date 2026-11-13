## Fractional time shifts of one row, in place: `x[j] ← x(j + shift)`, by
## linear or 4-point cubic (Catmull-Rom) interpolation, with zeros outside the
## row. Used by LMO apply and remove (`velocity/moveout`) and by statics
## (`statics/elevation`).
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

const blockLen = 256 # samples interpolated at a time, through a stack buffer

func cubicWeights(f: float64): array[4, float64] {.inline.} =
  ## Catmull-Rom weights of samples −1, 0, 1, 2 at fraction `f` in [0, 1).
  let f2 = f * f
  let f3 = f2 * f
  [-0.5 * f3 + f2 - 0.5 * f,
   1.5 * f3 - 2.5 * f2 + 1,
   -1.5 * f3 + 2 * f2 + 0.5 * f,
   0.5 * f3 - 0.5 * f2]

# The interior kernels: every tap inside the row, no branch, one output per
# iteration, into a buffer the row cannot overlap: they vectorize. The sums
# run in float64 whatever `T`, as the edge samples' do, so a float32 row
# gets the bits it got from the scalar loop these replace.

proc lerpInto[T](dst {.noalias.}, src {.noalias.}: ptr UncheckedArray[T],
    len: int, w0, w1: float64) {.noinline, raises: [].} =
  ## `dst[i] = w0·src[i] + w1·src[i + 1]` for `i < len`.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< len:
    dst[i] = T(w0 * float64(src[i]) + w1 * float64(src[i + 1]))
  {.pop.}

proc cubicInto[T](dst {.noalias.}, src {.noalias.}: ptr UncheckedArray[T],
    len: int, w: array[4, float64]) {.noinline, raises: [].} =
  ## `dst[i] = Σ w[t]·src[i + t]`, `t` in 0 .. 3, for `i < len`.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< len:
    dst[i] = T(w[0] * float64(src[i]) + w[1] * float64(src[i + 1]) +
      w[2] * float64(src[i + 2]) + w[3] * float64(src[i + 3]))
  {.pop.}

proc shiftSamples*[T: SampleType](x: var openArray[T], shift: float64,
    interp = siLinear) {.raises: [].} =
  ## Shifts `x` in place so that `x[j]` becomes the old `x` at `j + shift`
  ## (positive: earlier; negative: later), zero where that falls outside
  ## the row. A NaN or infinite shift leaves `x` untouched; one beyond the
  ## interpolation's reach zeroes it. A whole number of samples is a move:
  ## exact, whatever the samples (NaN and infinities stay where they are in
  ## the trace). Allocates nothing.
  let n = x.len
  if n == 0 or shift.classify in {fcNan, fcInf, fcNegInf}: return
  # The cubic reaches one sample further than linear interpolation.
  if abs(shift) >= float64(n + ord(interp == siCubic)):
    for v in x.mitems: v = T(0)
    return
  let k = int(floor(shift))
  let f = shift - float64(k)
  let p = cast[ptr UncheckedArray[T]](addr x[0])
  {.push boundChecks: off.}
  if f == 0:
    if k > 0:
      moveMem(p, addr p[k], (n - k) * sizeof(T))
      zeroMem(addr p[n - k], k * sizeof(T))
    elif k < 0:
      moveMem(addr p[-k], p, (n + k) * sizeof(T))
      zeroMem(p, -k * sizeof(T))
    return
  # Output j reads the taps j + k + a .. j + k + b.
  let (a, b) = if interp == siLinear: (0, 1) else: (-1, 2)
  let w = if interp == siLinear: [1 - f, f, 0.0, 0.0] else: cubicWeights(f)
  # `some`: the outputs with a tap in the row (zero outside them); `all`,
  # the interior: every tap in the row. Between them at most b − a partial
  # samples at each end.
  let someLo = clamp(-k - b, 0, n)
  let someHi = clamp(n - k - a, someLo, n)
  let allLo = clamp(-k - a, someLo, someHi)
  let allHi = clamp(n - k - b, allLo, someHi)
  # The partial samples first, from the row as it is: the interior loop
  # overwrites what they read.
  var edge {.noinit.}: array[6, T]
  var e = 0
  template partial(j: int) =
    var y = 0.0
    for t in 0 .. b - a:
      let m = j + k + a + t
      if m >= 0 and m < n: y += w[t] * float64(p[m])
    edge[e] = T(y)
    inc e
  for j in someLo ..< allLo: partial(j)
  for j in allHi ..< someHi: partial(j)
  # The interior, a block at a time into `tmp`, then back into the row.
  # Reading ahead of the write position (k ≥ 0, ascending) or behind it
  # (k < 0, descending) leaves the samples read unwritten, except the one
  # next to the block, which the cubic reads at a shift below one sample:
  # each block's last output is held back (`pend`) until the next block
  # has been computed.
  var tmp {.noinit.}: array[blockLen, T]
  let t = cast[ptr UncheckedArray[T]](addr tmp[0])
  var pendAt = -1
  var pend: T
  template interpolate(j0, len: int) =
    let src = cast[ptr UncheckedArray[T]](addr p[j0 + k + a])
    if interp == siLinear: lerpInto(t, src, len, w[0], w[1])
    else: cubicInto(t, src, len, w)
    if pendAt >= 0: p[pendAt] = pend
  if k >= 0:
    var j0 = allLo
    while j0 < allHi:
      let len = min(blockLen, allHi - j0)
      interpolate(j0, len)
      copyMem(addr p[j0], t, (len - 1) * sizeof(T))
      pendAt = j0 + len - 1
      pend = t[len - 1]
      j0 += len
  else:
    var j1 = allHi
    while j1 > allLo:
      let len = min(blockLen, j1 - allLo)
      let j0 = j1 - len
      interpolate(j0, len)
      copyMem(addr p[j0 + 1], addr t[1], (len - 1) * sizeof(T))
      pendAt = j0
      pend = t[0]
      j1 = j0
  if pendAt >= 0: p[pendAt] = pend
  e = 0
  for j in someLo ..< allLo:
    p[j] = edge[e]
    inc e
  for j in allHi ..< someHi:
    p[j] = edge[e]
    inc e
  if someLo > 0: zeroMem(p, someLo * sizeof(T))
  if someHi < n: zeroMem(addr p[someHi], (n - someHi) * sizeof(T))
  {.pop.}
