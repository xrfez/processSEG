## A time shift by a whole number of samples. Values move, the end they
## leave is clipped, and the vacated end is filled with zeros. The trace
## keeps its length. Either sign. Sub-sample shifts are out of scope.
##
## ```nim
## timeShiftInto(src, dst, shift = 3)          # 3 samples earlier
## timeShift(row, shift = -2)                  # 2 samples later, in place
## timeShift(row, wholeSamples(-6, dtMs = 2))  # 3 samples later
## ```
##
## **Sign.** As `shiftSamples`: a positive shift moves events earlier
## (`dst[j]` is the old sample at `j + shift`). A negative shift moves
## them later. Milliseconds snap with `wholeSamples`: round to nearest,
## ties away from zero.
##
## A shift of `len` samples or more zeroes the trace. An empty trace stays
## empty. NaN and infinities move with the samples that hold them. Nothing
## allocates.
##
## `shiftSamples` is the other shift: linear or cubic, used by statics and
## LMO. This one is the viewer's trace shift.

import std/math
import parseSEG

proc moveInto[T](src: openArray[T], dst: var openArray[T], k: int) {.raises: [].} =
  # `dst[j] = src[j + k]`, zero outside. `|k| < dst.len`, `dst.len > 0`,
  # `src.len == dst.len`. `src` and `dst` may be the same buffer; a
  # positive `k` moves toward the start.
  let n = dst.len
  let p = cast[ptr UncheckedArray[T]](addr dst[0])
  if k == 0:
    if cast[pointer](addr src[0]) != cast[pointer](p):
      copyMem(p, addr src[0], n * sizeof(T))
    return
  if cast[pointer](addr src[0]) == cast[pointer](p):
    if k > 0:
      moveMem(p, addr p[k], (n - k) * sizeof(T))
      zeroMem(addr p[n - k], k * sizeof(T))
    else:
      moveMem(addr p[-k], p, (n + k) * sizeof(T))
      zeroMem(p, -k * sizeof(T))
    return
  if k > 0:
    copyMem(p, addr src[k], (n - k) * sizeof(T))
    zeroMem(addr p[n - k], k * sizeof(T))
  else:
    copyMem(addr p[-k], addr src[0], (n + k) * sizeof(T))
    zeroMem(p, -k * sizeof(T))

proc timeShiftInto*[T: SampleType](src: openArray[T], dst: var openArray[T],
    shift: int) {.raises: [].} =
  ## `src` shifted by `shift` samples into `dst` (module documentation).
  ## Both have the same length; they may be the same buffer. A `|shift|`
  ## of `len` or more zeroes `dst`. Nothing allocates.
  doAssert dst.len == src.len, "dst needs src.len values"
  let n = dst.len
  if n == 0: return
  if shift >= n or shift <= -n:
    zeroMem(addr dst[0], n * sizeof(T))
    return
  moveInto(src, dst, shift)

proc timeShift*[T: SampleType](x: var openArray[T], shift: int) {.
    inline, raises: [].} =
  ## `timeShiftInto` in place.
  timeShiftInto(x, x, shift)

func wholeSamples*(ms, dtMs: float64): int {.raises: [].} =
  ## Whole samples in `ms` at a sample interval `dtMs`: `ms / dtMs` rounded
  ## to nearest, ties away from zero (`1.5` is `2`, `-1.5` is `-2`, not
  ## half-to-even). A non-finite `ms`, or a `dtMs` that is not finite and
  ## positive, is 0. A quotient past ±2⁵³ saturates, so the result always
  ## fits an `int`.
  if ms.classify in {fcNan, fcInf, fcNegInf}: return 0
  if dtMs.classify != fcNormal or dtMs <= 0: return 0
  let s = ms / dtMs
  if s > 9007199254740992.0: return high(int) # past 2^53
  if s < -9007199254740992.0: return low(int)
  int(round(s))
