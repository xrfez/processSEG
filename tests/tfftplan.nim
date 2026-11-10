## FFT plans (fft/plan): fast lengths, a naive-DFT match on every 2·3·5 length
## ≤ 4096 in float32 and float64, round trips, the batched lanes path and no
## allocation in a steady state.

import std/[assertions, math, random]
import processSEG
import utils/allocs

func refDft(x: openArray[float64]): (seq[float64], seq[float64]) =
  ## Bins 0 .. n div 2 of the DFT, in float64 with an exact twiddle table.
  let n = x.len
  var c, s = newSeq[float64](n)
  for i in 0..<n:
    c[i] = cos(2 * PI * float64(i) / float64(n))
    s[i] = -sin(2 * PI * float64(i) / float64(n))
  let h = n div 2 + 1
  var re, im = newSeq[float64](h)
  for k in 0..<h:
    var ar, ai = 0.0
    var idx = 0
    for j in 0..<n:
      ar += x[j] * c[idx]
      ai += x[j] * s[idx]
      idx += k
      if idx >= n: idx -= n
    re[k] = ar
    im[k] = ai
  (re, im)

proc fastLens(limit: int): seq[int] =
  for n in 1..limit:
    if isFastLen(n): result.add n

proc signal(rng: var Rand, n: int): seq[float64] =
  result = newSeq[float64](n)
  for v in result.mitems: v = float64(float32(rng.rand(-1.0..1.0)))

proc relErr[T](re, im: openArray[T], rr, ri: openArray[float64]): float64 =
  var e, top = 0.0
  for k in 0..<rr.len:
    e = max(e, hypot(float64(re[k]) - rr[k], float64(im[k]) - ri[k]))
    top = max(top, hypot(rr[k], ri[k]))
  e / max(top, 1e-300)

block fast_lengths:
  doAssert isFastLen(1) and isFastLen(2) and isFastLen(2025) and isFastLen(4096)
  doAssert not isFastLen(0) and not isFastLen(7) and not isFastLen(2001)
  doAssert nextFastLen(2001) == 2048 and nextFastLen(2001, even = false) == 2025
  doAssert nextFastLen(0) == 2 and nextFastLen(0, even = false) == 1
  doAssert nextFastLen(7) == 8 and nextFastLen(11) == 12 and nextFastLen(13) == 16
  doAssert nextFastLen(14, even = false) == 15
  doAssert nextFastLen(4097) == 4320
  for n in 1..5000:
    let e = nextFastLen(n)
    doAssert isFastLen(e) and e mod 2 == 0 and e >= n
    for k in n..<e: doAssert not (isFastLen(k) and k mod 2 == 0)
  doAssertRaises ValueError:
    discard initFftPlan(float32, 7)
  doAssertRaises ValueError:
    discard initFftPlan(float64, 0)

proc checkType(T: typedesc, tol: float64) =
  var rng = initRand(17)
  var ws = initWorkspace()
  var worst = 0.0
  var worstRt = 0.0
  for n in fastLens(4096):
    let p = initFftPlan(T, n)
    doAssert p.len == n and p.bins == n div 2 + 1
    ws.reserve(p.workspaceSize(batched = false))
    let x64 = signal(rng, n)
    var x = newSeq[T](n)
    for i in 0..<n: x[i] = T(x64[i])
    let (rr, ri) = refDft(x64)
    var re, im = newSeq[T](p.bins)
    p.forward(x, re, im, ws)
    let e = relErr(re, im, rr, ri)
    doAssert e <= tol, $T & " n = " & $n & ": error " & $e
    worst = max(worst, e)
    var back = newSeq[T](n)
    p.inverse(re, im, back, ws)
    var rt = 0.0
    for i in 0..<n: rt = max(rt, abs(float64(back[i]) - x64[i]))
    doAssert rt <= tol, $T & " n = " & $n & ": round trip error " & $rt
    worstRt = max(worstRt, rt)
  echo "  ", $T, ": worst DFT error ", worst, ", round trip ", worstRt

block naive_dft_f32:
  checkType(float32, 1e-5)

block naive_dft_f64:
  checkType(float64, 1e-12)

block known_spectra:
  var ws = initWorkspace()
  let p = initFftPlan(float64, 16)
  ws.reserve(p.workspaceSize)
  var x = newSeq[float64](16)
  var re, im = newSeq[float64](p.bins)
  x[0] = 1 # an impulse is flat
  p.forward(x, re, im, ws)
  for k in 0..<p.bins:
    doAssert abs(re[k] - 1) < 1e-15 and abs(im[k]) < 1e-15
  for j in 0..<16: x[j] = cos(2 * PI * 3 * float64(j) / 16) # bin 3, n/2 each
  p.forward(x, re, im, ws)
  for k in 0..<p.bins:
    let want = if k == 3: 8.0 else: 0.0
    doAssert abs(re[k] - want) < 1e-13 and abs(im[k]) < 1e-13

block inverse_ignores_dc_and_nyquist_imag:
  var ws = initWorkspace()
  for n in [12, 15]:
    let p = initFftPlan(float64, n)
    ws.reserve(p.workspaceSize)
    var re, im = newSeq[float64](p.bins)
    re[0] = 1
    im[0] = 5
    im[^1] = 7 # Nyquist for n = 12; an ordinary bin for n = 15
    var x = newSeq[float64](n)
    p.inverse(re, im, x, ws)
    var back, bim = newSeq[float64](p.bins)
    p.forward(x, back, bim, ws)
    doAssert abs(back[0] - 1) < 1e-14 and abs(bim[0]) < 1e-14
    if n == 12: doAssert abs(bim[^1]) < 1e-14
    else: doAssert abs(bim[^1] - 7) < 1e-13

proc lanesMatchSingle(T: typedesc, rows: int, tol: float64) =
  ## The lanes path against the one-trace path, with different data in every
  ## lane (`rows` of them; the rest of the batch is zero), and back.
  var rng = initRand(5 + rows)
  var ws = initWorkspace()
  for n in [1, 2, 3, 5, 8, 30, 125, 2025, 2048]:
    let p = initFftPlan(T, n)
    ws.reserve(p.workspaceSize)
    var t = newSeq[T](rows * n)
    for v in t.mitems: v = T(rng.rand(-1.0..1.0))
    let b = rowBlock(t, rows, n)
    var x = newSeq[T](n * fftLanes)
    gatherLanes(b, 0, x)
    var re, im = newSeq[T](p.bins * fftLanes)
    p.forwardLanes(x, re, im, ws)
    var r1, i1 = newSeq[T](p.bins)
    let lim = tol * sqrt(float64(n))
    for l in 0..<fftLanes:
      if l < rows:
        p.forward(b.row(l), r1, i1, ws)
      else:
        for k in 0..<p.bins: (r1[k] = 0; i1[k] = 0)
      for k in 0..<p.bins:
        doAssert abs(float64(re[k * fftLanes + l] - r1[k])) <= lim and
          abs(float64(im[k * fftLanes + l] - i1[k])) <= lim,
          $T & " n = " & $n & " lane " & $l & " bin " & $k
    var y = newSeq[T](n * fftLanes)
    p.inverseLanes(re, im, y, ws)
    var res = newSeq[T](rows * n)
    scatterLanes(y, rowBlock(res, rows, n), 0)
    for i in 0..<t.len:
      doAssert abs(float64(res[i] - t[i])) < 10 * tol, $T & " n = " & $n &
        " round trip"

block lanes_match_single:
  lanesMatchSingle(float32, fftLanes, 1e-6) # a full batch
  lanesMatchSingle(float32, fftLanes - 3, 1e-6) # a partial one
  lanesMatchSingle(float64, fftLanes, 1e-14)
  lanesMatchSingle(float64, 1, 1e-14)

block nan_stays_in_its_lane:
  for n in [60, 64, 45]:
    let p = initFftPlan(float32, n)
    var ws = initWorkspace()
    ws.reserve(p.workspaceSize)
    var x = newSeq[float32](n * fftLanes)
    for i in 0..<x.len: x[i] = float32(i mod 7)
    x[5 * fftLanes + 3] = NaN
    var re, im = newSeq[float32](p.bins * fftLanes)
    p.forwardLanes(x, re, im, ws)
    p.inverseLanes(re, im, x, ws)
    for j in 0..<n:
      for l in 0..<fftLanes:
        let v = x[j * fftLanes + l]
        if l == 3: doAssert v.isNaN
        else: doAssert abs(v - float32((j * fftLanes + l) mod 7)) < 1e-4

block gather_pads_and_clips:
  var t = @[1'f32, 2, 3, 4, 5, 6]
  let b = rowBlock(t, 2, 3)
  var x = newSeq[float32](4 * fftLanes) # n = 4 > cols: zero after
  for v in x.mitems: v = 99
  gatherLanes(b, 1, x)
  doAssert x[0] == 4 and x[fftLanes] == 5 and x[2 * fftLanes] == 6 and
    x[3 * fftLanes] == 0 and x[1] == 0 and x[3 * fftLanes + 7] == 0
  var y = newSeq[float32](2 * fftLanes) # n = 2 < cols: only 2 samples move
  y[0] = -1
  y[fftLanes] = -2
  scatterLanes(y, b, 1)
  doAssert t == @[1'f32, 2, 3, -1, -2, 6]

block no_allocation_steady_state:
  let p = initFftPlan(float32, 2025)
  var ws = initWorkspace()
  ws.reserve(p.workspaceSize)
  var x = newSeq[float32](p.len * fftLanes)
  var re, im = newSeq[float32](p.bins * fftLanes)
  x[3] = 1
  let n = allocsIn:
    for _ in 0..<3:
      p.forwardLanes(x, re, im, ws)
      p.inverseLanes(re, im, x, ws)
      p.forward(x.toOpenArray(0, p.len - 1), re.toOpenArray(0, p.bins - 1),
        im.toOpenArray(0, p.bins - 1), ws)
  doAssert n == 0, $n & " allocations"

block unreserved_workspace:
  when compileOption("boundChecks"):
    let p = initFftPlan(float64, 64)
    var ws = initWorkspace()
    var x = newSeq[float64](64)
    var re, im = newSeq[float64](p.bins)
    doAssertRaises IndexDefect:
      p.forward(x, re, im, ws)

block wrong_lengths_and_default_plans:
  # Always checked, in every build mode.
  let p = initFftPlan(float64, 64)
  var ws = initWorkspace()
  ws.reserve(p.workspaceSize(batched = false))
  doAssert ws.fits(p.workspaceSize(batched = false)) and
    not ws.fits(p.workspaceSize)
  var x = newSeq[float64](64)
  var re, im = newSeq[float64](p.bins)
  p.forward(x, re, im, ws) # one lane is enough for the one-trace path
  doAssertRaises AssertionDefect:
    p.forward(x, re.toOpenArray(0, 3), im, ws)
  doAssertRaises AssertionDefect:
    p.inverse(re, im, x.toOpenArray(0, 62), ws)
  doAssertRaises AssertionDefect:
    p.forwardLanes(x, re, im, ws)
  var none: FftPlan[float64]
  var empty: seq[float64]
  var one = newSeq[float64](1)
  doAssertRaises AssertionDefect:
    none.forward(empty, one, one, ws)
  var lanes = newSeq[float64](3)
  doAssertRaises AssertionDefect:
    gatherLanes(rowBlock(x, 8, 8), 0, lanes)
  doAssertRaises AssertionDefect:
    gatherLanes(rowBlock(x, 8, 8), -1, x)
