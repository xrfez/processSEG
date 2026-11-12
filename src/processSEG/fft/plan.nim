## Real FFT plans (`FftPlan`, decision D2): pure Nim, mixed radix 2/3/4/5.
##
## A plan transforms real sequences of one length `n`, which must be
## 2·3·5-smooth (`nextFastLen` rounds a length up to one). Twiddles are
## computed in float64 when the plan is built and stored in the plan, so
## nothing is global (D11) and a plan can be shared by several threads (each
## with its own `Workspace`).
##
## ```nim
## let plan = initFftPlan(float32, nextFastLen(2001))   # n = 2048
## var ws = initWorkspace()
## ws.reserve(plan.workspaceSize)                      # covers both paths below
## var re, im = newSeq[float32](plan.bins)             # n div 2 + 1 bins
## plan.forward(trace, re, im, ws)                     # one trace
## plan.inverse(re, im, trace, ws)                     # back, scaled by 1/n
## ```
##
## **Conventions.** `forward` computes `X[k] = Σ x[j]·exp(−2πi·jk/n)` for
## `k = 0 .. n div 2` (the other half is the complex conjugate), unscaled.
## `inverse` takes those bins and returns `x`, scaled by `1/n`, so a round
## trip is the identity. The imaginary parts of bin 0 and, for an even `n`,
## bin `n div 2` are taken as 0 by `inverse`. Bin `k` is at `k / (n·dt)` Hz.
##
## **Batches across traces.** The batched path transforms `fftLanes` traces
## at once, held interleaved: sample `j` of lane `l` is element
## `j·fftLanes + l` of the lane buffer, and bin `k` of lane `l` is element
## `k·fftLanes + l` of the spectrum buffers. Every inner loop then runs over
## contiguous lanes and compiles to packed SIMD. `gatherLanes` and
## `scatterLanes` move rows of a `RowBlock` in and out of that layout.
##
## The transforms do not allocate or raise. Buffers of the wrong length, and a
## plan not made by `initFftPlan`, are an `AssertionDefect` in every build
## mode (the checks cost nothing next to a transform), and so is a workspace
## reserved too small (an `IndexDefect`). `initFftPlan` raises `ValueError`.

import std/math
import parseSEG
import ../core/[rows, workspace]

const fftLanes* = 8
  ## Traces per batch in the batched path (`forwardLanes`, `inverseLanes`).
  ## Eight float32 lanes fill one AVX2 register.

const
  waySpan = 65536  # bytes of one way of an L2 cache (512 KiB, 8 ways)
  wayPhase = 17472 # 4 pages and 17 cache lines: 1088 bytes into an L1 way

func workStride(elems: int, T: typedesc): int {.inline.} =
  ## The distance, in elements, between the starts of the four work arrays
  ## of a transform, each `elems` long: the least whole number of cache
  ## lines that holds an array and is `wayPhase` more than a multiple of
  ## `waySpan`.
  ##
  ## Laid end to end, the arrays are a whole number of pages apart whenever
  ## `elems·sizeof(T)` is a multiple of 4096 (every power-of-two length),
  ## and so are the `r` input streams of a radix-`r` pass inside each array:
  ## all the streams of a pass then compete for the same cache sets (L1: 8
  ## ways of 4 KiB; L2: 8 ways of 64 KiB for long transforms) and evict each
  ## other on every butterfly. With this stride the arrays sit a quarter of a
  ## way apart in L2 and 17 lines apart in L1, whatever the length. The gaps
  ## are never touched: they cost address space, not memory. Measured, 8
  ## lanes of float32: n = 2048 1.7× faster, 65536 4.5×, 131072 15×
  ## (bench/README).
  let base = (elems * sizeof(T) + 63) div 64 * 64
  (base + (wayPhase - base mod waySpan + waySpan) mod waySpan) div sizeof(T)

type
  FftPlan*[T: SampleType] = object
    ## A real FFT of one length; see the module documentation.
    n: int                   # real length
    m: int                   # complex length: n div 2 (even n) or n (odd n)
    radices: seq[int]        # radix of each Stockham stage, first stage first
    twRe, twIm: seq[T]       # stage twiddles, stage after stage
    splitRe, splitIm: seq[T] # exp(−2πi·k/n), k = 0 .. m (even n only)

  Buf[T] = ptr UncheckedArray[T]

# ------------------------------------------------------------------ lengths --

func isFastLen*(n: int): bool {.raises: [].} =
  ## `n ≥ 1` has no prime factor other than 2, 3 and 5.
  if n < 1: return false
  var k = n
  for p in [2, 3, 5]:
    while k mod p == 0: k = k div p
  k == 1

func nextFastLen*(n: int, even = true): int {.raises: [].} =
  ## The smallest 2·3·5-smooth length `≥ max(n, 1)`, and even unless `even`
  ## is false. An even length runs a complex FFT of half the length, so it is
  ## about twice as fast as an odd one near it (2048 against 2025 for 2001).
  result = max(n, 1)
  while not isFastLen(result) or (even and result mod 2 == 1): inc result

func factorize(n: int): seq[int] =
  ## Stage radices of `n`: 4s first, then a 2, 3s and 5s.
  var k = n
  while k mod 4 == 0:
    result.add 4
    k = k div 4
  if k mod 2 == 0:
    result.add 2
    k = k div 2
  for p in [3, 5]:
    while k mod p == 0:
      result.add p
      k = k div p

# --------------------------------------------------------------------- plan --

proc initFftPlan*(T: typedesc[SampleType], n: int): FftPlan[T] =
  ## A plan for real sequences of length `n`. Raises `ValueError` unless `n`
  ## is 2·3·5-smooth (`isFastLen`); use `nextFastLen` to pad up to one.
  if not isFastLen(n):
    raise newException(ValueError, "an FFT length must be ≥ 1 with prime " &
      "factors 2, 3 and 5 only, got " & $n & " (try nextFastLen(" & $n &
      ") = " &
      $nextFastLen(n) & ")")
  result.n = n
  result.m = if n mod 2 == 0: n div 2 else: n
  result.radices = factorize(result.m)
  var cur = result.m
  for r in result.radices:
    let sub = cur div r
    for p in 0..<sub:
      for k in 1..<r:
        let a = -2 * PI * float64((p * k) mod cur) / float64(cur)
        result.twRe.add T(cos(a))
        result.twIm.add T(sin(a))
    cur = sub
  if n mod 2 == 0:
    for k in 0..result.m:
      let a = -2 * PI * float64(k) / float64(n)
      result.splitRe.add T(cos(a))
      result.splitIm.add T(sin(a))

func len*[T](p: FftPlan[T]): int {.inline, raises: [].} =
  ## The real length `n`.
  p.n

func bins*[T](p: FftPlan[T]): int {.inline, raises: [].} =
  ## The number of spectrum bins, `n div 2 + 1`.
  p.n div 2 + 1

func workspaceSize*[T](p: FftPlan[T], batched = true): WorkspaceSize {.raises: [].} =
  ## The scratch one transform needs: two complex work buffers, of
  ## `fftLanes` lanes for the batched path (`forwardLanes`, `inverseLanes`;
  ## this also covers the one-trace path) or of one lane (`batched = false`,
  ## enough for `forward` and `inverse` only), plus the gaps that keep their
  ## four arrays out of each other's cache sets.
  let each = p.m * (if batched: fftLanes else: 1)
  let elems = 3 * workStride(each, T) + each
  when T is float32: WorkspaceSize(f32: elems)
  else: WorkspaceSize(f64: elems)

# ---------------------------------------------------------- stockham passes --
#
# One stage of a decimation-in-frequency Stockham FFT on split complex data:
# with `sub = cur / r`, for p < sub, q < s and k < r,
#   y[q + s·(r·p + k)] = (Σ_j x[q + s·(p + j·sub)]·ω_r^{jk}) · ω_cur^{pk},
# where each complex element is `lanes` contiguous values. The inner loop runs
# over the `w = s·lanes` contiguous values of one (p, j) block.

{.push boundChecks: off, overflowChecks: off, infChecks: off, nanChecks: off.}

# Each butterfly kernel gets every input and output stream as its own
# `noalias` (C `restrict`) pointer. With one pointer per array, GCC cannot
# tell that `y[b + w + i]` and `y[b + 2w + i]` never meet, and the radix-4/5
# loops need more runtime alias checks than it will emit: they stay scalar.

template `+@`[T](b: Buf[T], o: int): Buf[T] =
  cast[Buf[T]](addr b[o])

template mulTw(yr, yi, wr, wi: untyped) =
  let t = yr * wr - yi * wi
  yi = yr * wi + yi * wr
  yr = t

template lanesLoop(w: int, L: static int, i, body: untyped) =
  ## Runs `body` for `i in 0 ..< w`, where `w` is a multiple of the lane
  ## count `L`. For `L > 1` the loop is written as runs of exactly `L`
  ## values (one SIMD register of float32 lanes for `fftLanes`), so the
  ## compiler emits no remainder loop and no trip-count test per butterfly:
  ## the first stages, whose `w` is `L` or `4·L`, are then straight-line
  ## packed code (17 % of an n = 2048 transform, measured).
  when L == 1:
    for i in 0..<w: body
  else:
    var first = 0
    while first < w:
      for lane in 0..<L:
        let i {.inject.} = first + lane
        body
      first += L

proc bfly2[T; L: static int](x0r {.noalias.}, x0i {.noalias.}, x1r {.noalias.},
    x1i {.noalias.}, y0r {.noalias.}, y0i {.noalias.}, y1r {.noalias.},
    y1i {.noalias.}: Buf[T], wr, wi: T, w: int) {.inline, raises: [].} =
  lanesLoop(w, L, i):
    let
      ur = x0r[i]
      ui = x0i[i]
      vr = x1r[i]
      vi = x1i[i]
    y0r[i] = ur + vr
    y0i[i] = ui + vi
    var dr = ur - vr
    var di = ui - vi
    mulTw(dr, di, wr, wi)
    y1r[i] = dr
    y1i[i] = di

proc bfly3[T; L: static int](x0r {.noalias.}, x0i {.noalias.}, x1r {.noalias.},
    x1i {.noalias.}, x2r {.noalias.}, x2i {.noalias.}, y0r {.noalias.},
    y0i {.noalias.}, y1r {.noalias.}, y1i {.noalias.}, y2r {.noalias.},
    y2i {.noalias.}: Buf[T], w1r, w1i, w2r, w2i: T, w: int) {.inline,
    raises: [].} =
  const
    c = T(-0.5)
    s = T(0.8660254037844386467637231707529362) # √3/2
  lanesLoop(w, L, i):
    let
      ar = x0r[i]
      ai = x0i[i]
      t1r = x1r[i] + x2r[i]
      t1i = x1i[i] + x2i[i]
      t2r = x1r[i] - x2r[i]
      t2i = x1i[i] - x2i[i]
      m1r = ar + c * t1r
      m1i = ai + c * t1i
      m2r = s * t2i # −i·s·t2
      m2i = -s * t2r
    y0r[i] = ar + t1r
    y0i[i] = ai + t1i
    var
      u1r = m1r + m2r
      u1i = m1i + m2i
      u2r = m1r - m2r
      u2i = m1i - m2i
    mulTw(u1r, u1i, w1r, w1i)
    mulTw(u2r, u2i, w2r, w2i)
    y1r[i] = u1r
    y1i[i] = u1i
    y2r[i] = u2r
    y2i[i] = u2i

proc bfly4[T; L: static int](x0r {.noalias.}, x0i {.noalias.}, x1r {.noalias.},
    x1i {.noalias.}, x2r {.noalias.}, x2i {.noalias.}, x3r {.noalias.},
    x3i {.noalias.}, y0r {.noalias.}, y0i {.noalias.}, y1r {.noalias.},
    y1i {.noalias.}, y2r {.noalias.}, y2i {.noalias.}, y3r {.noalias.},
    y3i {.noalias.}: Buf[T], w1r, w1i, w2r, w2i, w3r, w3i: T,
    w: int) {.inline, raises: [].} =
  lanesLoop(w, L, i):
    let
      t0r = x0r[i] + x2r[i]
      t0i = x0i[i] + x2i[i]
      t1r = x0r[i] - x2r[i]
      t1i = x0i[i] - x2i[i]
      t2r = x1r[i] + x3r[i]
      t2i = x1i[i] + x3i[i]
      t3r = x1i[i] - x3i[i] # (x1 − x3)·(−i)
      t3i = x3r[i] - x1r[i]
    y0r[i] = t0r + t2r
    y0i[i] = t0i + t2i
    var
      u1r = t1r + t3r
      u1i = t1i + t3i
      u2r = t0r - t2r
      u2i = t0i - t2i
      u3r = t1r - t3r
      u3i = t1i - t3i
    mulTw(u1r, u1i, w1r, w1i)
    mulTw(u2r, u2i, w2r, w2i)
    mulTw(u3r, u3i, w3r, w3i)
    y1r[i] = u1r
    y1i[i] = u1i
    y2r[i] = u2r
    y2i[i] = u2i
    y3r[i] = u3r
    y3i[i] = u3i

proc bfly5[T; L: static int](x0r {.noalias.}, x0i {.noalias.}, x1r {.noalias.},
    x1i {.noalias.}, x2r {.noalias.}, x2i {.noalias.}, x3r {.noalias.},
    x3i {.noalias.}, x4r {.noalias.}, x4i {.noalias.}, y0r {.noalias.},
    y0i {.noalias.}, y1r {.noalias.}, y1i {.noalias.}, y2r {.noalias.},
    y2i {.noalias.}, y3r {.noalias.}, y3i {.noalias.}, y4r {.noalias.},
    y4i {.noalias.}: Buf[T], w1r, w1i, w2r, w2i, w3r, w3i, w4r, w4i: T,
    w: int) {.inline, raises: [].} =
  const
    c1 = T(0.3090169943749474241022934171828191)  # cos 2π/5
    c2 = T(-0.8090169943749474241022934171828191) # cos 4π/5
    s1 = T(0.9510565162951535721164393333793821)  # sin 2π/5
    s2 = T(0.5877852522924731291687059546390728)  # sin 4π/5
  lanesLoop(w, L, i):
    let
      ar = x0r[i]
      ai = x0i[i]
      t1r = x1r[i] + x4r[i]
      t1i = x1i[i] + x4i[i]
      t2r = x2r[i] + x3r[i]
      t2i = x2i[i] + x3i[i]
      t3r = x1r[i] - x4r[i]
      t3i = x1i[i] - x4i[i]
      t4r = x2r[i] - x3r[i]
      t4i = x2i[i] - x3i[i]
      b1r = ar + c1 * t1r + c2 * t2r
      b1i = ai + c1 * t1i + c2 * t2i
      b2r = ar + c2 * t1r + c1 * t2r
      b2i = ai + c2 * t1i + c1 * t2i
      d1r = s1 * t3r + s2 * t4r
      d1i = s1 * t3i + s2 * t4i
      d2r = s2 * t3r - s1 * t4r
      d2i = s2 * t3i - s1 * t4i
    y0r[i] = ar + t1r + t2r
    y0i[i] = ai + t1i + t2i
    # y1 = b1 − i·d1, y4 = b1 + i·d1, y2 = b2 − i·d2, y3 = b2 + i·d2
    var
      u1r = b1r + d1i
      u1i = b1i - d1r
      u4r = b1r - d1i
      u4i = b1i + d1r
      u2r = b2r + d2i
      u2i = b2i - d2r
      u3r = b2r - d2i
      u3i = b2i + d2r
    mulTw(u1r, u1i, w1r, w1i)
    mulTw(u2r, u2i, w2r, w2i)
    mulTw(u3r, u3i, w3r, w3i)
    mulTw(u4r, u4i, w4r, w4i)
    y1r[i] = u1r
    y1i[i] = u1i
    y2r[i] = u2r
    y2i[i] = u2i
    y3r[i] = u3r
    y3i[i] = u3i
    y4r[i] = u4r
    y4i[i] = u4i

proc pass2[T; L: static int](xr, xi, yr, yi, tr, ti: Buf[T], sub,
    w: int) {.raises: [].} =
  for p in 0..<sub:
    let a = p * w
    let b = 2 * p * w
    let o = sub * w
    bfly2[T, L](xr +@ a, xi +@ a, xr +@ (a + o), xi +@ (a + o),
      yr +@ b, yi +@ b, yr +@ (b + w), yi +@ (b + w), tr[p], ti[p], w)

proc pass3[T; L: static int](xr, xi, yr, yi, tr, ti: Buf[T], sub,
    w: int) {.raises: [].} =
  for p in 0..<sub:
    let a = p * w
    let b = 3 * p * w
    let o = sub * w
    let t = 2 * p
    bfly3[T, L](xr +@ a, xi +@ a, xr +@ (a + o), xi +@ (a + o),
      xr +@ (a + 2 * o), xi +@ (a + 2 * o),
      yr +@ b, yi +@ b, yr +@ (b + w), yi +@ (b + w),
      yr +@ (b + 2 * w), yi +@ (b + 2 * w),
      tr[t], ti[t], tr[t + 1], ti[t + 1], w)

proc pass4[T; L: static int](xr, xi, yr, yi, tr, ti: Buf[T], sub,
    w: int) {.raises: [].} =
  for p in 0..<sub:
    let a = p * w
    let b = 4 * p * w
    let o = sub * w
    let t = 3 * p
    bfly4[T, L](xr +@ a, xi +@ a, xr +@ (a + o), xi +@ (a + o),
      xr +@ (a + 2 * o), xi +@ (a + 2 * o), xr +@ (a + 3 * o), xi +@ (a + 3 * o),
      yr +@ b, yi +@ b, yr +@ (b + w), yi +@ (b + w),
      yr +@ (b + 2 * w), yi +@ (b + 2 * w), yr +@ (b + 3 * w), yi +@ (b + 3 * w),
      tr[t], ti[t], tr[t + 1], ti[t + 1], tr[t + 2], ti[t + 2], w)

proc pass5[T; L: static int](xr, xi, yr, yi, tr, ti: Buf[T], sub,
    w: int) {.raises: [].} =
  for p in 0..<sub:
    let a = p * w
    let b = 5 * p * w
    let o = sub * w
    let t = 4 * p
    bfly5[T, L](xr +@ a, xi +@ a, xr +@ (a + o), xi +@ (a + o),
      xr +@ (a + 2 * o), xi +@ (a + 2 * o), xr +@ (a + 3 * o), xi +@ (a + 3 * o),
      xr +@ (a + 4 * o), xi +@ (a + 4 * o),
      yr +@ b, yi +@ b, yr +@ (b + w), yi +@ (b + w),
      yr +@ (b + 2 * w), yi +@ (b + 2 * w), yr +@ (b + 3 * w), yi +@ (b + 3 * w),
      yr +@ (b + 4 * w), yi +@ (b + 4 * w),
      tr[t], ti[t], tr[t + 1], ti[t + 1], tr[t + 2], ti[t + 2], tr[t + 3],
      ti[t + 3], w)

{.pop.}

proc complexForward[T; L: static int](p: FftPlan[T], ar, ai, br,
    bi: Buf[T]): bool {.raises: [].} =
  ## The forward complex FFT of length `m` of (ar, ai), with (br, bi) as the
  ## second work buffer. Returns true when the result ended in (br, bi).
  var
    xr = ar
    xi = ai
    yr = br
    yi = bi
    cur = p.m
    s = 1
    at = 0
  let tr = cast[Buf[T]](if p.twRe.len > 0: addr p.twRe[0] else: nil)
  let ti = cast[Buf[T]](if p.twIm.len > 0: addr p.twIm[0] else: nil)
  for r in p.radices:
    let sub = cur div r
    let w = s * L
    let twr = cast[Buf[T]](if tr == nil: nil else: addr tr[at])
    let twi = cast[Buf[T]](if ti == nil: nil else: addr ti[at])
    case r
    of 4: pass4[T, L](xr, xi, yr, yi, twr, twi, sub, w)
    of 2: pass2[T, L](xr, xi, yr, yi, twr, twi, sub, w)
    of 3: pass3[T, L](xr, xi, yr, yi, twr, twi, sub, w)
    else: pass5[T, L](xr, xi, yr, yi, twr, twi, sub, w)
    at += sub * (r - 1)
    cur = sub
    s *= r
    swap(xr, yr)
    swap(xi, yi)
  xr == br

# ------------------------------------------------------------- real wrappers --
#
# The lane count `L` is static, so each per-bin loop below is a fixed run of
# `L` contiguous values (one AVX2 register of float32 for `fftLanes`), and
# every kernel takes its streams as `noalias` pointers, as the passes do.

{.push boundChecks: off, overflowChecks: off, infChecks: off, nanChecks: off.}

proc copyLanes[T; L: static int](d {.noalias.}, s {.noalias.}: Buf[T]) {.inline,
    raises: [].} =
  for l in 0..<L: d[l] = s[l]

proc packLanes[T; L: static int](zr {.noalias.}, zi {.noalias.},
    x0 {.noalias.}, x1 {.noalias.}: Buf[T]) {.inline, raises: [].} =
  for l in 0..<L:
    zr[l] = x0[l]
    zi[l] = x1[l]

proc splitForward[T; L: static int](ar {.noalias.}, ai {.noalias.},
    br {.noalias.}, bi {.noalias.}, xr {.noalias.}, xi {.noalias.}: Buf[T],
    wr, wi: T) {.inline, raises: [].} =
  ## X[k] = E + ω^k·O, E = (Z[k] + Z*[m−k])/2, O = (Z[k] − Z*[m−k])/2i, with
  ## Z[k] = (ar, ai) and Z[m−k] = (br, bi).
  for l in 0..<L:
    let
      er = T(0.5) * (ar[l] + br[l])
      ei = T(0.5) * (ai[l] - bi[l])
      orr = T(0.5) * (ai[l] + bi[l])
      oi = T(0.5) * (br[l] - ar[l])
    xr[l] = er + wr * orr - wi * oi
    xi[l] = ei + wr * oi + wi * orr

proc splitInverse[T; L: static int](ar {.noalias.}, ai {.noalias.},
    br {.noalias.}, bi {.noalias.}, zr {.noalias.}, zi {.noalias.}: Buf[T],
    wr, wi, h: T) {.inline, raises: [].} =
  ## Z[k] = E + i·O, E = (X[k] + X*[m−k])/2, O = ω^−k·(X[k] − X*[m−k])/2 (the
  ## halves and 1/m folded into `h`), with X[k] = (ar, ai), X[m−k] = (br, bi)
  ## and ω^−k = (wr, wi). Writes swap(Z): `zr` gets Im Z and `zi` Re Z.
  for l in 0..<L:
    let
      er = ar[l] + br[l]
      ei = ai[l] - bi[l]
      dr = ar[l] - br[l]
      di = ai[l] + bi[l]
      orr = dr * wr - di * wi
      oi = dr * wi + di * wr
    zi[l] = (er - oi) * h
    zr[l] = (ei + orr) * h

proc forwardImpl[T; L: static int](p: FftPlan[T], x: Buf[T], re, im: Buf[T],
    ws: var Workspace) {.raises: [].} =
  ## Real forward transform of `L` interleaved sequences.
  let mL = p.m * L
  let stride = workStride(mL, T)
  let w = ws.scratch(T, 3 * stride + mL)
  let
    ar = w
    ai = w +@ stride
    br = w +@ (2 * stride)
    bi = w +@ (3 * stride)
    m = p.m
  if p.n mod 2 == 1:
    for i in 0..<mL:
      ar[i] = x[i]
      ai[i] = T(0)
  else:
    for k in 0..<m: # z[k] = x[2k] + i·x[2k+1]
      packLanes[T, L](ar +@ (k * L), ai +@ (k * L), x +@ (2 * k * L),
        x +@ ((2 * k + 1) * L))
  let inB = complexForward[T, L](p, ar, ai, br, bi)
  let zr = if inB: br else: ar
  let zi = if inB: bi else: ai
  if p.n mod 2 == 1:
    copyLanes[T, L](re, zr) # bin 0; the rest below
    copyLanes[T, L](im, zi)
    for k in 1..<p.bins:
      copyLanes[T, L](re +@ (k * L), zr +@ (k * L))
      copyLanes[T, L](im +@ (k * L), zi +@ (k * L))
    return
  for l in 0..<L:
    let r0 = zr[l]
    let i0 = zi[l]
    re[l] = r0 + i0
    im[l] = T(0)
    re[m * L + l] = r0 - i0
    im[m * L + l] = T(0)
  let sr = cast[Buf[T]](addr p.splitRe[0])
  let si = cast[Buf[T]](addr p.splitIm[0])
  for k in 1..<m:
    splitForward[T, L](zr +@ (k * L), zi +@ (k * L), zr +@ ((m - k) * L),
      zi +@ ((m - k) * L), re +@ (k * L), im +@ (k * L), sr[k], si[k])

proc inverseImpl[T; L: static int](p: FftPlan[T], re, im: Buf[T], x: Buf[T],
    ws: var Workspace) {.raises: [].} =
  ## Real inverse transform (scaled by 1/n) of `L` interleaved spectra. The
  ## inverse complex FFT is the forward one on swapped real and imaginary
  ## parts: ifft(Z) = swap(fft(swap(Z))).
  let mL = p.m * L
  let stride = workStride(mL, T)
  let w = ws.scratch(T, 3 * stride + mL)
  let
    ar = w
    ai = w +@ stride
    br = w +@ (2 * stride)
    bi = w +@ (3 * stride)
    scale = T(1) / T(p.n)
    m = p.m
  # ai receives Re Z and ar Im Z, i.e. (ar, ai) is swap(Z).
  if p.n mod 2 == 1:
    for l in 0..<L:
      ai[l] = re[l] * scale
      ar[l] = T(0)
    for k in 1..<p.bins:
      for l in 0..<L:
        let vr = re[k * L + l] * scale
        let vi = im[k * L + l] * scale
        ai[k * L + l] = vr
        ar[k * L + l] = vi
        ai[(m - k) * L + l] = vr
        ar[(m - k) * L + l] = -vi
  else:
    # 1/n = the 1/2 of E and O times the 1/m of the inverse (n = 2m).
    for l in 0..<L:
      let x0 = re[l]
      let xm = re[m * L + l]
      ai[l] = (x0 + xm) * scale # Re Z0 = E
      ar[l] = (x0 - xm) * scale # Im Z0 = O
    let sr = cast[Buf[T]](addr p.splitRe[0])
    let si = cast[Buf[T]](addr p.splitIm[0])
    for k in 1..<m:
      splitInverse[T, L](re +@ (k * L), im +@ (k * L), re +@ ((m - k) * L),
        im +@ ((m - k) * L), ar +@ (k * L), ai +@ (k * L), sr[k], -si[k], scale)
  # (ar, ai) holds swap(Z) = Im Z + i·Re Z; z = swap(fft(swap(Z))).
  let inB = complexForward[T, L](p, ar, ai, br, bi)
  let zr = if inB: bi else: ai
  let zi = if inB: br else: ar
  if p.n mod 2 == 1:
    for i in 0..<mL:
      x[i] = zr[i]
  else:
    for k in 0..<m:
      packLanes[T, L](x +@ (2 * k * L), x +@ ((2 * k + 1) * L), zr +@ (k * L),
        zi +@ (k * L))

{.pop.}

proc badLen(what: string, got, want: int) {.noinline, noreturn.} =
  raise newException(AssertionDefect, what & " has " & $got &
      " values, the plan needs " & $want)

proc noPlan() {.noinline, noreturn.} =
  raise newException(AssertionDefect, "an FftPlan must come from initFftPlan")

template checkLen(what: string, a: openArray, want: int) =
  ## Always on: a wrong length would write past the end of a buffer.
  if a.len != want: badLen(what, a.len, want)

template checkPlan(p: FftPlan) =
  if p.n < 1: noPlan()

template bufOf[T](a: openArray[T]): Buf[T] =
  cast[Buf[T]](if a.len == 0: nil else: addr a[0])

# ------------------------------------------------------------- public forms --

proc forward*[T: SampleType](p: FftPlan[T], x: openArray[T],
    re, im: var openArray[T], ws: var Workspace) {.raises: [].} =
  ## The spectrum of one trace: `x.len == p.len`, `re.len == im.len ==
  ## p.bins`. `ws` must hold `p.workspaceSize(batched = false)`.
  checkPlan(p)
  checkLen("x", x, p.n)
  checkLen("re", re, p.bins)
  checkLen("im", im, p.bins)
  forwardImpl[T, 1](p, bufOf(x), bufOf(re), bufOf(im), ws)

proc inverse*[T: SampleType](p: FftPlan[T], re, im: openArray[T],
    x: var openArray[T], ws: var Workspace) {.raises: [].} =
  ## The trace of a spectrum from `forward`, scaled by `1/n`; lengths as for
  ## `forward`. `re` and `im` are not changed.
  checkPlan(p)
  checkLen("x", x, p.n)
  checkLen("re", re, p.bins)
  checkLen("im", im, p.bins)
  inverseImpl[T, 1](p, bufOf(re), bufOf(im), bufOf(x), ws)

proc forwardLanes*[T: SampleType](p: FftPlan[T], x: openArray[T],
    re, im: var openArray[T], ws: var Workspace) {.raises: [].} =
  ## The spectra of `fftLanes` interleaved traces (module documentation):
  ## `x.len == p.len · fftLanes`, `re.len == im.len == p.bins · fftLanes`.
  ## `ws` must hold `p.workspaceSize`.
  checkPlan(p)
  checkLen("x", x, p.n * fftLanes)
  checkLen("re", re, p.bins * fftLanes)
  checkLen("im", im, p.bins * fftLanes)
  forwardImpl[T, fftLanes](p, bufOf(x), bufOf(re), bufOf(im), ws)

proc inverseLanes*[T: SampleType](p: FftPlan[T], re, im: openArray[T],
    x: var openArray[T], ws: var Workspace) {.raises: [].} =
  ## The traces of `fftLanes` interleaved spectra, scaled by `1/n`; lengths
  ## as for `forwardLanes`.
  checkPlan(p)
  checkLen("x", x, p.n * fftLanes)
  checkLen("re", re, p.bins * fftLanes)
  checkLen("im", im, p.bins * fftLanes)
  inverseImpl[T, fftLanes](p, bufOf(re), bufOf(im), bufOf(x), ws)

proc gatherLanes*[T: SampleType](b: RowBlock[T], first: int,
    x: var openArray[T]) {.raises: [].} =
  ## Interleaves rows `first ..< first + fftLanes` of `b` into the lane
  ## buffer `x` (`x.len = n · fftLanes`): each row's first `min(n, b.cols)`
  ## samples, zero after them, and zero lanes for rows past `b.len`. An
  ## `AssertionDefect` for a negative `first` or an `x.len` that is not a
  ## multiple of `fftLanes`.
  doAssert first >= 0 and x.len mod fftLanes == 0, "bad lane buffer or first row"
  let n = x.len div fftLanes
  let c = min(n, b.cols)
  for l in 0..<fftLanes:
    let r = first + l
    if r < b.len:
      let src = b.rowPtr(r)
      {.push boundChecks: off.}
      for j in 0..<c: x[j * fftLanes + l] = src[j]
      for j in c..<n: x[j * fftLanes + l] = T(0)
      {.pop.}
    else:
      for j in 0..<n: x[j * fftLanes + l] = T(0)

proc scatterLanes*[T: SampleType](x: openArray[T], b: RowBlock[T],
    first: int) {.raises: [].} =
  ## The reverse of `gatherLanes`: writes lane `l` of `x` to row
  ## `first + l` of `b` (rows past `b.len` are skipped), its first
  ## `min(n, b.cols)` samples. Checks as for `gatherLanes`.
  doAssert first >= 0 and x.len mod fftLanes == 0, "bad lane buffer or first row"
  let n = x.len div fftLanes
  let c = min(n, b.cols)
  for l in 0..<fftLanes:
    let r = first + l
    if r < b.len:
      let dst = b.rowPtr(r)
      {.push boundChecks: off.}
      for j in 0..<c: dst[j] = x[j * fftLanes + l]
      {.pop.}
