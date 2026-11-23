## Length-changing placement (`placedWhole`, `placedSinc`): equality with
## `==` against a copy of the original arithmetic, over a seeded random set
## and the edges named in the proc docs (zeros before a positive whole
## shift, a drop for a negative one, a non-positive interval, an empty
## trace, a NaN shift, a fraction below 1e-9, 100.3 ms not snapped to 100,
## zeros outside the stored length, longer when positive and shorter when
## negative).
##
## Break check, restored: `placedWhole`'s store `result[wholeSamples + i] = v`
## written as zero failed the positive-shift equality; `placedSinc`'s
## `sum += float64(stored[k]) * taps[i]` with the multiply removed failed
## the fractional equality.

import std/[assertions, math, random]
import processSEG

const RefSincHalf = 24

func refBessel(x: float64): float64 =
  var term = 1.0
  result = 1.0
  for k in 1 .. 30:
    term *= (x / (2 * float64(k))) ^ 2
    result += term

func refPlaced(stored: openArray[float32], wholeSamples: int): seq[float32] =
  if wholeSamples >= 0:
    result = newSeq[float32](wholeSamples + stored.len)
    for i, v in stored: result[wholeSamples + i] = v
  else:
    let drop = min(-wholeSamples, stored.len)
    result = newSeq[float32](stored.len - drop)
    for i in 0 ..< result.len: result[i] = stored[drop + i]

func refPlacedBy(stored: openArray[float32], shiftMs, dtMs: float64): seq[float32] =
  if not (dtMs > 0) or stored.len == 0 or shiftMs.isNaN: return @stored
  let d = shiftMs / dtMs
  let whole = int(floor(d))
  let frac = d - float64(whole)
  if frac < 1e-9: return refPlaced(stored, whole)
  let n = max(0, stored.len + int(ceil(d)))
  result = newSeq[float32](n)
  let beta = 8.0
  let norm = refBessel(beta)
  var taps = newSeq[float64](2 * RefSincHalf + 1)
  for i in 0 .. 2 * RefSincHalf:
    let m = i - RefSincHalf
    let x = float64(m) - frac
    let r = abs(x) / float64(RefSincHalf)
    if r < 1:
      let w = refBessel(beta * sqrt(1 - r * r)) / norm
      taps[i] = (if abs(x) < 1e-12: 1.0 else: sin(PI * x) / (PI * x)) * w
  for j in 0 ..< n:
    var sum = 0.0
    for i in 0 .. 2 * RefSincHalf:
      let k = j - whole - (i - RefSincHalf)
      if k >= 0 and k < stored.len: sum += float64(stored[k]) * taps[i]
    result[j] = float32(sum)

proc main*() =
  block whole_zeros_before:
    let stored = @[1.0'f32, 2, 3]
    let y = placedWhole(stored, 2)
    doAssert y == refPlaced(stored, 2)
    doAssert y == @[0.0'f32, 0, 1, 2, 3]
    doAssert placedWhole(stored, 0) == stored
    doAssert placedWhole(newSeq[float32](), 4) == newSeq[float32](4)

  block whole_drops_first:
    let stored = @[1.0'f32, 2, 3, 4]
    doAssert placedWhole(stored, -1) == @[2.0'f32, 3, 4]
    doAssert placedWhole(stored, -1) == refPlaced(stored, -1)
    doAssert placedWhole(stored, -4).len == 0
    doAssert placedWhole(stored, -9).len == 0
    doAssert placedWhole(stored, -9) == refPlaced(stored, -9)
    doAssert placedWhole(newSeq[float32](), -2).len == 0

  block sinc_copies_when_it_cannot_place:
    let stored = @[1.0'f32, -2, 3.5]
    for dt in [0.0, -1.0, NaN]:
      doAssert placedSinc(stored, 1.5, dt) == stored
      doAssert placedSinc(stored, 1.5, dt) == refPlacedBy(stored, 1.5, dt)
    doAssert placedSinc(stored, NaN, 2.0) == stored
    doAssert placedSinc(newSeq[float32](), 3.0, 1.0).len == 0
    doAssert placedSinc(stored, 0.0, 2.0) == stored

  block fraction_below_1e9_is_whole:
    let stored = @[1.0'f32, 2, 3, 4, 5]
    let shift = 5.0 + 1e-10
    doAssert placedSinc(stored, shift, 1.0) == placedWhole(stored, 5)
    doAssert placedSinc(stored, shift, 1.0) == refPlacedBy(stored, shift, 1.0)
    let neg = -2.0 + 1e-12
    doAssert placedSinc(stored, neg, 1.0) == placedWhole(stored, -2)
    doAssert placedSinc(stored, neg, 1.0) == refPlacedBy(stored, neg, 1.0)

  block fractional_not_snapped_to_a_sample:
    # 100.3 ms at 1 ms is not the whole-sample placement at 100.
    let impulse = @[1.0'f32, 0, 0, 0]
    let atFrac = placedSinc(impulse, 100.3, 1.0)
    doAssert atFrac == refPlacedBy(impulse, 100.3, 1.0)
    doAssert atFrac != placedWhole(impulse, 100)
    doAssert atFrac.len == impulse.len + 101 # ceil(100.3)
    let s = @[1.0'f32, 2, 3, 4]
    let frac = placedSinc(s, 0.3, 1.0)
    doAssert frac == refPlacedBy(s, 0.3, 1.0)
    doAssert frac.len == s.len + 1
    doAssert frac != placedWhole(s, 1)

  block longer_or_shorter:
    var stored = newSeq[float32](20)
    for j in 0 ..< stored.len: stored[j] = float32(j) * 0.1'f32
    let later = placedSinc(stored, 2.2, 1.0)
    doAssert later.len == 23 # ceil(2.2) = 3
    doAssert later == refPlacedBy(stored, 2.2, 1.0)
    let earlier = placedSinc(stored, -3.4, 1.0)
    doAssert earlier.len == 17 # ceil(-3.4) = -3
    doAssert earlier == refPlacedBy(stored, -3.4, 1.0)
    doAssert placedSinc(stored, -25.2, 1.0).len == 0

  block zeros_outside_no_wrap:
    # A short trace: a wrap would copy an end sample into the other end.
    # The reference reads zero outside the stored length.
    let stored = @[1.0'f32, 0, 0]
    let y = placedSinc(stored, 0.5, 1.0)
    doAssert y == refPlacedBy(stored, 0.5, 1.0)
    doAssert y.len == 4
    doAssert y[0] != stored[^1] or stored[^1] == 0

  block random_matches_reference:
    var rng = initRand(11)
    for it in 0 ..< 400:
      let n =
        case it mod 5
        of 0: rng.rand(0 .. 8)
        of 1: [23, 24, 25, 47, 48, 49, 64][rng.rand(6)]
        else: rng.rand(0 .. 200)
      var stored = newSeq[float32](n)
      for v in stored.mitems: v = float32(rng.rand(-2.0 .. 2.0))
      let whole = rng.rand(-n - 4 .. n + 6)
      doAssert placedWhole(stored, whole) == refPlaced(stored, whole),
        "whole n " & $n & " shift " & $whole
      let dt = [0.5, 1.0, 2.0, 4.0, 0.0, -1.0][rng.rand(5)]
      let shift =
        case it mod 4
        of 0: float64(rng.rand(-n - 2 .. n + 2))
        of 1: float64(rng.rand(-30 .. 40)) + 0.3
        of 2: NaN
        else: rng.rand(-40.0 .. 80.0)
      doAssert placedSinc(stored, shift, dt) == refPlacedBy(stored, shift, dt),
        "sinc n " & $n & " shift " & $shift & " dt " & $dt

  block long_trace_matches_reference:
    var rng = initRand(19)
    var stored = newSeq[float32](4000)
    for v in stored.mitems: v = float32(rng.rand(-1.0 .. 1.0))
    doAssert placedSinc(stored, 100.3, 2.0) == refPlacedBy(stored, 100.3, 2.0)
    doAssert placedSinc(stored, -12.7, 1.0) == refPlacedBy(stored, -12.7, 1.0)
    doAssert placedWhole(stored, 30) == refPlaced(stored, 30)
    doAssert placedWhole(stored, -7) == refPlaced(stored, -7)

  block float64_whole_placement:
    let stored = @[1.0, -2.5, 4.0]
    doAssert placedWhole(stored, 1) == @[0.0, 1.0, -2.5, 4.0]
    doAssert placedWhole(stored, -2) == @[4.0]
    doAssert placedSinc(stored, 2.0, 1.0) == placedWhole(stored, 2)

  block unusable_shift_returns_stored:
    # Break check: without the isFinite guard in placedSinc, Inf, -Inf, 1e20
    # and (Inf, Inf) raise OverflowDefect at int(floor(d)).
    let stored = @[1'f32, -2, 3.5, 0.25]
    doAssert placedSinc(stored, Inf, 1.0) == stored
    doAssert placedSinc(stored, -Inf, 1.0) == stored
    doAssert placedSinc(stored, 1e20, 1.0) == stored
    doAssert placedSinc(stored, Inf, Inf) == stored

  block whole_shift_range:
    # Break check: without the range test, placedWhole(@[1'f32], low(int))
    # raises OverflowDefect when it negates low(int), and
    # placedWhole(@[1'f32], high(int)) raises OverflowDefect in newSeq.
    doAssert placedWhole(@[1'f32], low(int)) == newSeq[float32]()
    doAssertRaises ValueError:
      discard placedWhole(@[1'f32], high(int))

when isMainModule: main()
