## A whole-sample shift: values move, the vacated end is zero, the other
## end is clipped. Exact against a hand-written move, both signs, NaN,
## in place, and the millisecond snap.
##
## Fail-first: `equals_hand_move` failed while `timeShiftInto` passed
## `-shift` to the move, so the sign was reversed (float64, n = 2,
## shift = -1). `ms_to_samples` failed while `wholeSamples` truncated
## instead of rounding (`wholeSamples(0.5, 1.0)` was 0).

import std/[assertions, math]
import processSEG
import utils/allocs

proc main*() =
  proc moved[T](src: openArray[T], shift: int): seq[T] =
    ## The shift by hand: `dst[j] = src[j + shift]`, else 0. Not the library.
    result = newSeq[T](src.len)
    for j in 0 ..< src.len:
      let m = j + shift
      if m >= 0 and m < src.len: result[j] = src[m]

  proc same[T](a, b: openArray[T]): bool =
    if a.len != b.len: return false
    for i in 0 ..< a.len:
      if a[i].isNaN:
        if not b[i].isNaN: return false
      elif a[i] != b[i]: return false
    true

  proc check[T]() =
    const lengths = [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 100, 127, 128,
      255, 256, 257, 513, 1000]
    for n in lengths:
      for specials in [false, true]:
        var src = newSeq[T](n)
        for j in 0 ..< n: src[j] = T(j + 1) * T(0.25)
        if specials and n > 0:
          src[0] = T(NaN)
          if n > 2: src[n div 2] = T(Inf)
          if n > 1: src[n - 1] = T(NegInf)
        let saved = src
        for shift in -n - 3 .. n + 3:
          var dst = newSeq[T](n)
          timeShiftInto(src, dst, shift)
          let want = moved(src, shift)
          doAssert same(dst, want), $T & " n " & $n & " shift " & $shift &
            " specials " & $specials
          doAssert same(src, saved), "src changed"
          var inplace = src
          timeShift(inplace, shift)
          doAssert same(inplace, want), "in place " & $T & " n " & $n &
            " shift " & $shift
          doAssert same(src, saved), "src changed by the copy"
        if n > 0:
          for shift in [n * 2, -n * 2, n, -n]:
            var dst = newSeq[T](n)
            timeShiftInto(src, dst, shift)
            doAssert same(dst, moved(src, shift)), "large " & $shift

  block equals_hand_move:
    check[float64]()
    check[float32]()

  block ms_to_samples:
    # Ties away from zero, not half-to-even: 2.5 is 3, not 2.
    doAssert wholeSamples(0.5, 1.0) == 1
    doAssert wholeSamples(-0.5, 1.0) == -1
    doAssert wholeSamples(1.5, 1.0) == 2
    doAssert wholeSamples(-1.5, 1.0) == -2
    doAssert wholeSamples(2.5, 1.0) == 3
    doAssert wholeSamples(-2.5, 1.0) == -3
    doAssert wholeSamples(3.5, 1.0) == 4
    doAssert wholeSamples(-3.5, 1.0) == -4
    doAssert wholeSamples(1.4, 1.0) == 1
    doAssert wholeSamples(1.6, 1.0) == 2
    doAssert wholeSamples(-1.4, 1.0) == -1
    doAssert wholeSamples(-1.6, 1.0) == -2
    doAssert wholeSamples(5.0, 2.0) == 3
    doAssert wholeSamples(-5.0, 2.0) == -3
    doAssert wholeSamples(6.0, 2.0) == 3
    doAssert wholeSamples(-6.0, 2.0) == -3
    doAssert wholeSamples(7.0, 2.0) == 4
    doAssert wholeSamples(0.0, 2.0) == 0
    doAssert wholeSamples(4.0, 4.0) == 1
    doAssert wholeSamples(-4.0, 4.0) == -1
    doAssert wholeSamples(NaN, 2.0) == 0
    doAssert wholeSamples(Inf, 2.0) == 0
    doAssert wholeSamples(NegInf, 2.0) == 0
    doAssert wholeSamples(4.0, 0.0) == 0
    doAssert wholeSamples(4.0, -2.0) == 0
    doAssert wholeSamples(4.0, NaN) == 0
    doAssert wholeSamples(4.0, Inf) == 0
    doAssert wholeSamples(1.0, 1e-320) == 0 # subnormal dtMs
    doAssert wholeSamples(1e20, 1e-10) == high(int)
    doAssert wholeSamples(-1e20, 1e-10) == low(int)
    # Exactly ±2^53 is not saturated: it is the whole sample it names.
    doAssert wholeSamples(9007199254740992.0, 1.0) == 9007199254740992
    doAssert wholeSamples(-9007199254740992.0, 1.0) == -9007199254740992
    var row = newSeq[float64](16)
    for j in 0 ..< row.len: row[j] = float64(j + 1)
    row[4] = NaN
    let k = wholeSamples(5.0, 2.0)
    doAssert k == 3
    var got = row
    got.timeShift(k)
    doAssert same(got, moved(row, k))
    doAssert got[1].isNaN

  block allocates_nothing:
    var x = newSeq[float64](128)
    var y = newSeq[float64](128)
    for j in 0 ..< x.len: x[j] = float64(j)
    x[3] = NaN
    doAssert allocsIn(timeShiftInto(x, y, 5)) == 0
    doAssert allocsIn(timeShift(x, -5)) == 0
    doAssert allocsIn(timeShift(x, 1000)) == 0
    doAssert allocsIn(timeShift(x, 0)) == 0
    var e: seq[float64]
    doAssert allocsIn(timeShift(e, 1)) == 0
    var one = newSeq[float32](1)
    one[0] = NaN
    doAssert allocsIn(timeShift(one, 0)) == 0
    doAssert one[0].isNaN
    doAssert allocsIn(timeShift(one, -1)) == 0
    doAssert one[0] == 0
    let snap = allocsIn:
      discard wholeSamples(2.5, 1.0)
    doAssert snap == 0

when isMainModule: main()
