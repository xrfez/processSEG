## Fractional shifts (core/shift): integer shifts exact, in place equal to
## a copy-based reference for every direction and fraction (the in-place
## read order), polynomials reproduced, zeros outside, NaN and large
## shifts, float32, and no allocation. `placedSinc` matches `oldPlacedSinc`
## (the per-sample sum in tap order, built the same way) on every bit.
##
## Break check, restored: `sincBlock`'s `acc[i] = acc[i] + weight *
## float64(s[i])` with the multiply removed failed that equality.

import std/[assertions, math, random]
import processSEG
import utils/[allocs, synth]

const OldSincHalf = 24

func oldBessel(x: float64): float64 =
  var term = 1.0
  result = 1.0
  for k in 1 .. 30:
    term *= (x / (2 * float64(k))) ^ 2
    result += term

func oldPlacedSinc[T](stored: openArray[T], shiftMs, dtMs: float64): seq[T] =
  ## Copy of the previous per-sample sum: 49 taps, tap 0 then 1 then 2,
  ## `sum += sample * tap`, zero outside the trace. The range test stays on
  ## every sample. Both this and the previous release sum in tap order; an
  ## optimised build contracts `sum + sample * tap` to a fused multiply-add
  ## here and in `placedSinc`, where the previous float64 interior multiplied
  ## and added separately, so float64 interiors differ from that release by a
  ## few units in the last place. float32 results are the same bits.
  if not (dtMs > 0) or stored.len == 0 or shiftMs.isNaN: return @stored
  let d = shiftMs / dtMs
  if not (abs(d) < 9223372036854775808.0): return @stored
  let whole = int(floor(d))
  let frac = d - float64(whole)
  if frac < 1e-9: return placedWhole(stored, whole)
  let n = max(0, stored.len + int(ceil(d)))
  result = newSeq[T](n)
  let beta = 8.0
  let norm = oldBessel(beta)
  var taps = newSeq[float64](2 * OldSincHalf + 1)
  for i in 0 .. 2 * OldSincHalf:
    let m = i - OldSincHalf
    let x = float64(m) - frac
    let r = abs(x) / float64(OldSincHalf)
    if r < 1:
      let w = oldBessel(beta * sqrt(1 - r * r)) / norm
      taps[i] = (if abs(x) < 1e-12: 1.0 else: sin(PI * x) / (PI * x)) * w
  for j in 0 ..< n:
    var sum = 0.0
    for i in 0 .. 2 * OldSincHalf:
      let k = j - whole - (i - OldSincHalf)
      if k >= 0 and k < stored.len: sum += float64(stored[k]) * taps[i]
    result[j] = T(sum)

proc bitsEqual[T: float32 | float64](a, b: T): bool =
  when T is float32: cast[uint32](a) == cast[uint32](b)
  else: cast[uint64](a) == cast[uint64](b)

proc main*() =
  func reference[T](x: openArray[T], shift: float64, interp: ShiftInterp): seq[float64] =
    ## The shift computed from an untouched copy.
    let n = x.len
    template at(i: int): float64 = (if i >= 0 and i < n: float64(x[i]) else: 0.0)
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

  block random_shifts_equal_reference:
    # Rows of every length around the 256-sample blocks of the interior loop,
    # whole, small, half-sample and arbitrary shifts both ways, both
    # interpolations: in place equals the copy-based reference. A float32 row
    # is interpolated in float64 as the reference is, so it equals the
    # reference rounded to float32 (one ulp is left for a product folded
    # into a multiply-add).
    proc check[T](tol: float64) =
      var rng = initRand(5)
      for it in 0..<6000:
        let n =
          case it mod 4
          of 0: rng.rand(1..12)
          of 1: [255, 256, 257, 258, 511, 512, 513, 514, 768, 769][rng.rand(9)]
          else: rng.rand(1..700)
        var x0 = newSeq[T](n)
        for v in x0.mitems: v = T(rng.gauss(0, 1))
        let shift =
          case it mod 5
          of 0: float64(rng.rand(-n - 2 .. n + 2))
          of 1: rng.rand(-3.0 .. 3.0)
          of 2: float64(rng.rand(-n - 2 .. n + 2)) + [0.5, 0.25, 0.999, 0.001][rng.rand(3)]
          else: rng.rand(-float64(n + 2) .. float64(n + 2))
        for interp in [siLinear, siCubic]:
          var x = x0
          x.shiftSamples(shift, interp)
          let r = reference(x0, shift, interp)
          for j in 0..<n:
            doAssert abs(float64(x[j]) - float64(T(r[j]))) <= tol * max(1, abs(r[j])),
              $T & " " & $interp & " n " & $n & " shift " & $shift & " at " & $j
    check[float64](1e-12)
    check[float32](1.2e-7)

  block whole_shifts_move_any_sample:
    # A whole number of samples is a move: a NaN or an infinity stays one
    # sample of the trace, and its neighbours stay exact.
    for interp in [siLinear, siCubic]:
      for k in [-5, -1, 1, 4]:
        var x = newSeq[float32](20)
        for j in 0..<20: x[j] = float32(j) + 0.5
        x[7] = NaN
        x[12] = Inf
        let x0 = x
        x.shiftSamples(float64(k), interp)
        for j in 0..<20:
          let m = j + k
          if m < 0 or m >= 20: doAssert x[j] == 0
          elif m == 7: doAssert x[j].isNaN
          else: doAssert x[j] == x0[m], $interp & " " & $k & " " & $j

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
    # The cubic reaches one sample past linear: a shift of n still reads x[0].
    for interp in [siLinear, siCubic]:
      for sh in [-3.5, -3.0, -2.5, 2.5, 3.0, 3.5]:
        var z = @[1.0, 2, 3]
        z.shiftSamples(sh, interp)
        let r = reference(@[1.0, 2, 3], sh, interp)
        for j in 0..<3: doAssert abs(z[j] - r[j]) <= 1e-12, $interp & " " & $sh
    var e: seq[float64]
    e.shiftSamples(1.5)
    var y = zeroSignal[float32](100)
    doAssert allocsIn(y.shiftSamples(1.5, siCubic)) == 0

  block placed_sinc_matches_old_bits:
    proc check[T: float32 | float64](n: int, shift, dt: float64, nanAt = -1) =
      var stored = newSeq[T](n)
      for i in 0 ..< n:
        stored[i] = T(float64((i * 17 + 3) mod 97) * 0.01 - 0.2)
      if nanAt >= 0 and nanAt < n: stored[nanAt] = T(NaN)
      let got = placedSinc(stored, shift, dt)
      let old = oldPlacedSinc(stored, shift, dt)
      doAssert got.len == old.len, $T & " n " & $n & " shift " & $shift
      for j in 0 ..< got.len:
        doAssert bitsEqual(got[j], old[j]), $T & " n " & $n & " shift " & $shift &
            " at " & $j
    proc run[T: float32 | float64]() =
      for n in [1, 2, 48, 49, 50, 1000]:
        for shift in [0.0, 0.5, -0.5, 0.25, 123.75, -3.1, float64(n) + 2.7,
            -float64(n) - 2.7]:
          check[T](n, shift, 1.0)
        if n > 1: check[T](n, 0.3, 1.0, nanAt = n div 2)
    run[float32]()
    run[float64]()

when isMainModule: main()
