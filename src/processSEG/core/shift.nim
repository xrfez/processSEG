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
## A viewer's length-keeping shift by whole samples is `timeShift`
## (`core/timeshift`), not this interpolator.
##
## A stored trace placed later, longer or shorter than it was, is not a
## mode of `shiftSamples`: that one stays in place and keeps the length.
## `placedWhole` moves by whole samples and `placedSinc` by a
## Kaiser-windowed sinc; both return a new trace. A positive amount places
## the stored samples later (zeros before them), the opposite sign from
## `shiftSamples`.

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

const SincHalf = 24 # taps each side of the Kaiser-windowed sinc
const SincTaps = 2 * SincHalf + 1

# Interior of `placedSinc`: one tap across a block of outputs, so output i
# is still tap 0, then 1, then 2, ... . The multiply-add is the scalar
# loop's expression (`acc + w * x`).
proc sincBlock[T](dst {.noalias.}: ptr UncheckedArray[T],
    src {.noalias.}: ptr UncheckedArray[T], len, tap0: int,
    w: ptr UncheckedArray[float64]) {.noinline, raises: [].} =
  ## `len` <= `blockLen`. `src[tap0 - t + i]` is in range for every tap.
  var acc {.noinit.}: array[blockLen, float64]
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< len:
    acc[i] = 0.0
  for t in 0 ..< SincTaps:
    let weight = w[t]
    let s = cast[ptr UncheckedArray[T]](addr src[tap0 - t])
    for i in 0 ..< len:
      acc[i] = acc[i] + weight * float64(s[i])
  for i in 0 ..< len:
    dst[i] = T(acc[i])
  {.pop.}

# Thirty terms of (x/(2k))^2. Not `vibroseis/kaiser.besselI0`: that series
# multiplies by (x*x/4)/k^2 and stops early, and the two differ (1 ulp at
# beta*sqrt(1-r*r) for r = 0.00008, and in the window weights) on the
# arguments used here (beta 8, r in [0, 1)).
func besselI0(x: float64): float64 =
  var term = 1.0
  result = 1.0
  for k in 1 .. 30:
    term *= (x / (2 * float64(k))) ^ 2
    result += term

func placedWhole*[T: SampleType](stored: openArray[T],
    wholeSamples: int): seq[T] {.raises: [ValueError].} =
  ## `stored` starting `wholeSamples` after the first sample (zeros before
  ## it); a negative number drops that many of its first samples. Longer by
  ## a positive shift, shorter by a negative one, empty when the drop covers
  ## `stored`. A positive shift whose length is past the range of a sample
  ## count raises `ValueError`.
  if wholeSamples >= 0:
    if wholeSamples > high(int) - stored.len:
      raise newException(ValueError,
        "placedWhole: shift " & $wholeSamples & " is past the range of a sample count")
    result = newSeq[T](wholeSamples + stored.len)
    for i, v in stored: result[wholeSamples + i] = v
  else:
    # Not negated first: -low(int) overflows.
    if wholeSamples <= -stored.len: return newSeq[T]()
    let drop = -wholeSamples
    result = newSeq[T](stored.len - drop)
    for i in 0 ..< result.len: result[i] = stored[drop + i]

func placedSinc*[T: SampleType](stored: openArray[T], shiftMs,
    dtMs: float64): seq[T] {.raises: [ValueError].} =
  ## `stored` moved later by `shiftMs`, a fraction of a sample included.
  ## Each output sample is a Kaiser-windowed sinc (24 taps each side, beta 8)
  ## of the stored samples around the time it lies at, so 100.3 ms at 1 ms
  ## lands at 100.3 ms, not at 100. Samples outside `stored` are zero, so
  ## nothing wraps. Longer by the shift, shorter when it is negative.
  ## A `dtMs` that is not positive, an empty `stored`, a NaN or infinite
  ## `shiftMs`, or a shift whose quotient by `dtMs` is beyond the range of a
  ## sample count returns a copy of `stored`. A fraction below 1e-9 above a
  ## whole number of samples is `placedWhole` at that whole number, which
  ## raises `ValueError` only for a whole number past the range of a count.
  if not (dtMs > 0) or stored.len == 0 or shiftMs.isNaN: return @stored
  let d = shiftMs / dtMs
  # Also false for NaN (every comparison with NaN is false).
  if not (abs(d) < 9223372036854775808.0): return @stored
  let whole = int(floor(d))
  let frac = d - float64(whole)
  if frac < 1e-9: return placedWhole(stored, whole)
  let n = max(0, stored.len + int(ceil(d)))
  result = newSeq[T](n)
  let beta = 8.0
  let norm = besselI0(beta)
  var taps = newSeq[float64](2 * SincHalf + 1)
  for i in 0 .. 2 * SincHalf:
    let m = i - SincHalf
    let x = float64(m) - frac
    let r = abs(x) / float64(SincHalf)
    if r < 1:
      let w = besselI0(beta * sqrt(1 - r * r)) / norm
      taps[i] = (if abs(x) < 1e-12: 1.0 else: sin(PI * x) / (PI * x)) * w
  # k = j - whole - (i - SincHalf) runs from j - whole + SincHalf down
  # to j - whole - SincHalf. Inside [lo, hi) every tap is in `stored`.
  # Those outputs are summed a block at a time, one tap across the block,
  # so each sample's products still accumulate in tap order. The ends keep
  # the range test.
  let lo = min(n, max(0, whole + SincHalf))
  let hi = min(n, stored.len + whole - SincHalf)
  template addGuarded(j: int) =
    var sum = 0.0
    for i in 0 .. 2 * SincHalf:
      let k = j - whole - (i - SincHalf)
      if k >= 0 and k < stored.len: sum += float64(stored[k]) * taps[i]
    result[j] = T(sum)
  for j in 0 ..< lo: addGuarded(j)
  if hi > lo:
    let sp = cast[ptr UncheckedArray[T]](addr stored[0])
    let rp = cast[ptr UncheckedArray[T]](addr result[0])
    let wp = cast[ptr UncheckedArray[float64]](addr taps[0])
    var j0 = lo
    while j0 < hi:
      let len = min(blockLen, hi - j0)
      # Same index expression as the scalar loop, wrapping the same way.
      {.push overflowChecks: off.}
      let tap0 = j0 - whole + SincHalf
      {.pop.}
      sincBlock(cast[ptr UncheckedArray[T]](addr rp[j0]), sp, len, tap0, wp)
      j0 += len
  for j in max(hi, lo) ..< n: addGuarded(j)
