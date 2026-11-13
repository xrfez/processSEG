## Robust geometry QC of one shot from its first breaks: a refraction
## model fitted to the confident picks against the header offsets, residuals
## against that model and against the neighbouring traces, the symmetry of
## the two sides of a split spread, and cross-correlation with the
## neighbouring traces. Traces whose geometry is suspect are flagged.
##
## ```nim
## # One shot's rows (e.g. meta.sub(first .. last)) and their picks (qc/firstbreak).
## var fit: ShotFit
## fitShot(samples, meta, picks, ShotFitParams(), fit)  # or fitShot(meta, picks, ...)
## for row, bad in fit.suspect:
##   if bad: echo row, " ", fit.localMs[row], " ", fit.lagMs[row]
## echo fit.asymmetryMs                                 # a misplaced source
## ```
##
## **Model** (`model`), times against the header `|offset|`:
##
## - `rmTwoSegment`: a direct (or first refractor) segment and a refractor
##   segment meeting at a crossover offset, `t = a + x/v1` up to `xc`,
##   `a + xc/v1 + (x − xc)/v2` beyond; fitted by least absolute deviations
##   (iteratively reweighted), the crossover by a search over the picks'
##   offsets (at least 3 picks on each segment; up to 64 quantiles of them
##   on long spreads) refined by golden section, the second segment faster
##   than the first; a single segment when that fits as well (to a part in
##   10⁶ of the misfit, or to 1e-9 ms a pick); no model when all used picks
##   share one offset;
## - `rmOffsetBinned`: the median pick time in offset bins of `binWidth`,
##   linear between the bins' median offsets and extended linearly beyond
##   the end bins: for refractors no two lines describe. Bins should be
##   narrower than the curve's bends (the crossover), or the traces there
##   get flagged.
##
## Only picks of `confidence ≥ minConfidence` on live rows with a known
## offset count. The spread is taken as one line (2D, or one receiver line
## of a 3D shot: pass each line's rows separately, since a patch of lines
## interleaves along any one axis).
##
## **Residuals.** `residualMs` is the pick minus the model. Along the
## spread (traces ordered by `position`: when every live row has
## coordinates, the receiver's distance from the source along the
## receivers' principal axis, oriented toward +x (+y for a north–south
## line); else the signed offset when the offsets are signed; else the
## channel number; else the row), a
## refractor whose depth or velocity varies bends the residuals smoothly,
## while a trace whose header places it at another station jumps.
## `localMs` is the residual minus the median residual of the nearest
## `localTraces` traces with a residual on each side (itself excluded; at
## least 2 in all, else NaN), which removes the smooth part. A trace is suspect when `|localMs|` exceeds
## `max(maxResidualMs, madFactor · 1.4826 · MAD)` of the shot's local
## residuals.
##
## **Cross-correlation** (the form with samples, `correlate`): each trace's
## window of `windowMs` from `windowMs / 4` before its two-segment model
## time (whichever `model` the residuals use: a few bad picks do not move
## an L1 fit, while they can bump a bin's median) is
## correlated with its neighbours' (in spread order), over lags up to
## `maxLagMs`. Windows at the model times align the break when the
## geometry is right; `lagMs` is the lag to the neighbour with the smaller
## `|lag|` (the better correlation breaks a tie), `coherence` that pair's
## correlation coefficient. The lag is the neighbour's arrival minus the
## trace's, each counted from its own model time: negative for a trace
## that arrives late against its neighbour (its header puts it nearer the
## source than it is), positive for one that arrives early. A trace is suspect also when it is the odd one out: its
## lags to both neighbours exceed `lagToleranceMs` while the two
## neighbours, correlated with each other, agree within it with a
## correlation coefficient of at least `minCoherence`. (The neighbour
## of a bad trace agrees with its other neighbour; where the model is off,
## as near the source without confident direct-wave picks, the lags are
## large but the neighbours disagree too, and nothing is flagged.) The end
## traces of the spread have one neighbour and are not checked this way.
## This needs no picks at the trace itself, so it also checks traces whose
## pick was not confident.
##
## **Symmetry.** `asymmetryMs` is the median residual of the negative side
## of the spread minus that of the positive side (NaN for one side). A
## source whose **header** position is `d` along the line too far toward
## the positive side makes the positive side's picks late against the
## model and the negative side's early, by `d / v` each: `asymmetryMs` is
## about `−2d / v` (positive when the header is too far toward the negative
## side).
## A refractor that varies laterally also makes the sides differ by a few
## ms, so compare shots rather than trust one value. The source's position
## is not checked further.
##
## **What a swap must exceed.** A trace whose header is `Δx` along the line
## from its true station is off the model by about `Δx / v` (`v` the
## velocity of the arrival there), and off by `Δx / v` from its neighbours'
## lags. It is flagged when that exceeds `maxResidualMs` (12 ms) or
## `lagToleranceMs` (8 ms): with 25 m groups, an adjacent-channel swap on a
## 3000 m/s refractor (8 ms) is below both and is missed, while swaps of
## two stations or more, or on slower arrivals, are found. Lower the
## thresholds for clean data and short groups.
##
## `fitShot` resizes the `ShotFit` seqs to the shot's rows and reuses them
## and its scratch: after the first shot of a size, a call allocates a few
## times (its three sorts), whatever the number of rows.

import std/[algorithm, math]
import parseSEG
import ../core/[rows, types]
import ./firstbreak

type
  RefractionModel* = enum
    ## The model of the first-break times (see the module documentation).
    rmTwoSegment   ## Two lines in |offset| meeting at a crossover; L1 fit.
    rmOffsetBinned ## Median time per offset bin, linear between bins.

  ShotFitParams* = object
    ## What `fitShot` checks; times in ms, distances in offset units.
    model*: RefractionModel = rmTwoSegment
    minConfidence*: float64 = 0.5 ## Picks below it are not used.
    binWidth*: float64 = 100      ## `rmOffsetBinned`: the offset bin width.
    localTraces*: int = 3         ## Neighbours on each side for the local median.
    maxResidualMs*: float64 = 12  ## The largest `|localMs|` that does not flag a trace.
    madFactor*: float64 = 4       ## Robust threshold, in MADs of `localMs`.
    correlate*: bool = true       ## Cross-correlate neighbours (the form with samples).
    windowMs*: float64 = 80       ## Correlation window.
    maxLagMs*: float64 = 40       ## Largest lag searched.
    lagToleranceMs*: float64 = 8  ## The smallest neighbour lag that flags a trace.
    minCoherence*: float64 = 0.8  ## The neighbours' correlation that makes them agree.

  ShotFit* = object
    ## One shot's fit and findings; per-trace seqs are indexed by row.
    model*: RefractionModel
    interceptMs*: float64              ## The two-segment fit's `a` (fitted with either model).
    v1*, v2*: float64                  ## Its segment velocities (offset units/s); `v2` NaN: one segment.
    crossover*: float64                ## Its `xc` (Inf: one segment).
    used*: int                         ## Picks the fit used.
    asymmetryMs*: float64              ## Negative-side minus positive-side median residual.
    threshold*: float64                ## The `|localMs|` above which a trace is suspect.
    position*: seq[float64]            ## Signed position along the spread; NaN unknown.
    modelMs*: seq[float64]             ## The model time at the header offset; NaN unknown.
    residualMs*: seq[float64]          ## Pick − model; NaN without a used pick.
    localMs*: seq[float64]             ## Residual − neighbours' median; NaN without one.
    lagMs*: seq[float64]               ## Best neighbour lag (sign: module docs); NaN without.
    coherence*: seq[float64]           ## Its correlation coefficient; NaN without.
    suspect*: seq[bool]                ## Geometry suspect.
    # Scratch, kept for reuse.
    order, idx: seq[int]
    xs, ts, ws, tmp, sideA, sideB: seq[float64]
    absOffset, segmentMs: seq[float64] ## |offset| per row; the two-segment model.
    bins: seq[(float64, float64)]
    keyed: seq[(float64, int)]

proc badParams(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc checkParams(p: ShotFitParams) =
  if not (p.minConfidence >= 0 and p.minConfidence <= 1):
    badParams("minConfidence must be in 0 .. 1, got " & $p.minConfidence)
  if not (p.binWidth > 0 and p.binWidth < Inf):
    badParams("binWidth must be finite and positive, got " & $p.binWidth)
  if p.localTraces < 1: badParams("localTraces must be ≥ 1, got " & $p.localTraces)
  if not (p.maxResidualMs >= 0 and p.maxResidualMs < Inf and p.madFactor >= 0 and
      p.madFactor < Inf):
    badParams("maxResidualMs and madFactor must be finite and ≥ 0")
  if not (p.windowMs > 0 and p.windowMs < Inf and p.maxLagMs >= 0 and p.maxLagMs < Inf and
      p.lagToleranceMs >= 0 and p.lagToleranceMs < Inf):
    badParams("correlation windows must be finite (windowMs > 0, the others ≥ 0)")
  if not (p.minCoherence >= -1 and p.minCoherence <= 1):
    badParams("minCoherence must be in -1 .. 1, got " & $p.minCoherence)

func select(x: var openArray[float64], k: int): float64 =
  # The k-th smallest of x (from 0), reordering x so that nothing before k
  # is larger and nothing after it smaller: a quickselect, median of three
  # pivots. No allocation (a sort has its buffer).
  var lo = 0
  var hi = x.high
  while lo < hi:
    let mid = (lo + hi) shr 1
    if x[mid] < x[lo]: swap(x[mid], x[lo])
    if x[hi] < x[lo]: swap(x[hi], x[lo])
    if x[hi] < x[mid]: swap(x[hi], x[mid])
    let pivot = x[mid]
    var i = lo
    var j = hi
    while i <= j:
      while x[i] < pivot: inc i
      while pivot < x[j]: dec j
      if i <= j:
        swap(x[i], x[j])
        inc i
        dec j
    if k <= j: hi = j
    elif k >= i: lo = i
    else: return x[k]
  x[k]

func median(x: var openArray[float64]): float64 =
  # The median of x (reordered); NaN when empty. The same value as from a
  # sort: the middle one, or the mean of the two middle ones.
  if x.len == 0: return NaN
  let m = x.len div 2
  let upper = select(x, m)
  if x.len mod 2 == 1: upper
  else:
    var lower = x[0]
    for i in 1 ..< m: lower = max(lower, x[i])
    (lower + upper) / 2

# ------------------------------------------------------------- positions --

proc positionsInto(meta: ChunkMeta, pos: var seq[float64]) =
  # Signed position of each row along the spread: (rcv − src)·u with u the
  # receivers' principal direction when every live row has coordinates,
  # else the signed offset.
  let n = meta.len
  var cx, cy, k = 0.0
  var coords = true
  for r in 0 ..< n:
    if not meta.isLive(r): continue
    let q = meta.keys(r)
    if not (q.hasSourceXY and q.hasReceiverXY):
      coords = false
      break
    cx += q.rcvX
    cy += q.rcvY
    k += 1
  if coords and k >= 2:
    cx /= k
    cy /= k
    var sxx, sxy, syy = 0.0
    for r in 0 ..< n:
      if not meta.isLive(r): continue
      let q = meta.keys(r)
      let dx = q.rcvX - cx
      let dy = q.rcvY - cy
      sxx += dx * dx
      sxy += dx * dy
      syy += dy * dy
    let theta = 0.5 * arctan2(2 * sxy, sxx - syy) # principal axis
    var ux = cos(theta)
    var uy = sin(theta)
    if ux < -1e-12 or (abs(ux) <= 1e-12 and uy < 0): # toward +x (else +y)
      ux = -ux
      uy = -uy
    for r in 0 ..< n:
      let q = meta.keys(r)
      pos[r] = if meta.isLive(r): (q.rcvX - q.srcX) * ux + (q.rcvY - q.srcY) * uy else: NaN
    return
  # No coordinates: the signed offset when the offsets are signed (some
  # negative) and all known; else the channel number when every live row
  # has one; else the row. Unsigned offsets would interleave the two sides
  # of a split spread.
  var signed, offsets, channels = true
  var anyNeg = false
  for r in 0 ..< n:
    if not meta.isLive(r): continue
    let q = meta.keys(r)
    if q.offset.isNaN: offsets = false
    elif q.offset < 0: anyNeg = true
    if q.channel.isNaN: channels = false
  signed = offsets and anyNeg
  for r in 0 ..< n:
    let q = meta.keys(r)
    pos[r] = if not meta.isLive(r): NaN
             elif signed: q.offset
             elif channels: q.channel
             else: float64(r)

proc sortByKey(idx: var seq[int], key: openArray[float64],
    keyed: var seq[(float64, int)]) {.raises: [].}

# ----------------------------------------------------------------- models --

proc solve3(m: var array[3, array[4, float64]]): bool =
  # Gaussian elimination with partial pivoting of a 3×3 system (column 3
  # the right-hand side); the solution in column 3. False if singular.
  for c in 0..2:
    var piv = c
    for r in c + 1 .. 2:
      if abs(m[r][c]) > abs(m[piv][c]): piv = r
    if abs(m[piv][c]) < 1e-300: return false
    swap(m[c], m[piv])
    for r in 0..2:
      if r != c:
        let f = m[r][c] / m[c][c]
        for k in c..3: m[r][k] -= f * m[c][k]
  for c in 0..2: m[c][3] /= m[c][c]
  true

proc l1Fit(xs, ts: openArray[float64], ws: var seq[float64], xc: float64,
    coef: var array[3, float64]): float64 =
  # Least absolute deviations of t = c0 + c1·min(x, xc) + c2·max(x − xc, 0)
  # by iteratively reweighted least squares; the sum of |residuals|, Inf if
  # singular. With xc = Inf, c2 is fixed at 0.
  let single = xc == Inf
  for i in 0 ..< ws.len: ws[i] = 1
  result = Inf
  for iter in 0 ..< 30:
    # The normal equations, in local sums (symmetric: six of nine terms),
    # so the loop vectorizes.
    var s00, s01, s02, s11, s12, s22, r0, r1, r2 = 0.0
    for i in 0 ..< xs.len:
      let w = ws[i]
      let b1 = min(xs[i], xc)
      let b2 = if single: 0.0 else: max(xs[i] - xc, 0.0)
      let wt = w * ts[i]
      s00 += w
      s01 += w * b1
      s02 += w * b2
      s11 += w * b1 * b1
      s12 += w * b1 * b2
      s22 += w * b2 * b2
      r0 += wt
      r1 += wt * b1
      r2 += wt * b2
    var m = [[s00, s01, s02, r0], [s01, s11, s12, r1], [s02, s12, s22, r2]]
    if single: # c2 = 0: an identity row keeps the system regular
      m[2] = [0.0, 0.0, 1.0, 0.0]
    if not solve3(m): return Inf
    for r in 0..2: coef[r] = m[r][3]
    var s = 0.0
    for i in 0 ..< xs.len:
      let b1 = min(xs[i], xc)
      let b2 = if single: 0.0 else: max(xs[i] - xc, 0.0)
      let e = abs(ts[i] - (coef[0] + coef[1] * b1 + coef[2] * b2))
      s += e
      ws[i] = 1 / max(e, 0.05) # 0.05 ms: below pick precision
    if abs(result - s) <= 1e-9 * max(s, 1): return s
    result = s

proc fitTwoSegment(fit: var ShotFit) =
  let n = fit.xs.len
  fit.ws.setLen n
  var best = Inf
  var bestC: array[3, float64]
  var bestXc = Inf
  var c: array[3, float64]
  # One segment, then crossovers at the picks' offsets (3 picks or more on
  # each segment; at most 64 quantiles of them on a long spread), then a
  # golden-section refinement around the best.
  fit.tmp.setLen n
  for i in 0 ..< n: fit.tmp[i] = fit.xs[i]
  fit.tmp.sort()
  if fit.tmp[n - 1] - fit.tmp[0] <= 1e-9 * max(1.0, fit.tmp[n - 1]):
    return # one offset: no slope can be fitted (the model stays NaN)
  best = l1Fit(fit.xs, fit.ts, fit.ws, Inf, c)
  bestC = c
  # A second segment must be faster than the first (a smaller slowness).
  template valid(c: array[3, float64]): bool = c[1] > 0 and c[2] > 0 and c[2] < c[1]
  var bestK = -1
  var last = NaN
  let stride = max(1, (n - 5) div 64)
  for k in countup(2, n - 4, stride):
    let xc = fit.tmp[k]
    if xc == last or xc <= fit.tmp[0] or xc >= fit.tmp[n - 1]: continue
    last = xc
    let s = l1Fit(fit.xs, fit.ts, fit.ws, xc, c)
    # A second segment must lower the misfit by a part in 10⁶ and by more
    # than rounding leaves of an exact fit (1e-9 ms a pick): picks on one
    # line have a misfit near 1e-13, and any split "improves" on that.
    if s < best * (1 - 1e-6) and best - s > 1e-9 * float64(n) and valid(c):
      best = s
      bestC = c
      bestXc = xc
      bestK = k
  if bestK >= 0:
    # Within the candidates' range, so each segment keeps 3 picks or more.
    var a = fit.tmp[max(bestK - stride, 2)]
    var b = fit.tmp[min(bestK + stride, n - 4)]
    const g = 0.6180339887498949
    # Golden section: each step keeps one interior point and its fit.
    var c1, c2: array[3, float64]
    var x1 = b - g * (b - a)
    var x2 = a + g * (b - a)
    var s1 = l1Fit(fit.xs, fit.ts, fit.ws, x1, c1)
    var s2 = l1Fit(fit.xs, fit.ts, fit.ws, x2, c2)
    template consider(s: float64, c: array[3, float64], x: float64) =
      if s < best and valid(c):
        best = s
        bestC = c
        bestXc = x
    consider(s1, c1, x1)
    consider(s2, c2, x2)
    for iter in 0 ..< 40:
      if b - a <= 1e-6 * max(1.0, b): break
      if s1 <= s2:
        b = x2
        x2 = x1
        s2 = s1
        c2 = c1
        x1 = b - g * (b - a)
        s1 = l1Fit(fit.xs, fit.ts, fit.ws, x1, c1)
        consider(s1, c1, x1)
      else:
        a = x1
        x1 = x2
        s1 = s2
        c1 = c2
        x2 = a + g * (b - a)
        s2 = l1Fit(fit.xs, fit.ts, fit.ws, x2, c2)
        consider(s2, c2, x2)
  fit.interceptMs = bestC[0]
  fit.v1 = if bestC[1] > 0: 1000 / bestC[1] else: NaN
  fit.crossover = bestXc
  fit.v2 = if bestXc == Inf: NaN elif bestC[2] > 0: 1000 / bestC[2] else: NaN
  # Model time of every row from the coefficients.
  for r in 0 ..< fit.absOffset.len:
    let x = fit.absOffset[r]
    fit.segmentMs[r] = if x.isNaN: NaN
      else: bestC[0] + bestC[1] * min(x, bestXc) +
        (if bestXc == Inf: 0.0 else: bestC[2] * max(x - bestXc, 0.0))

proc fitBinned(fit: var ShotFit, width: float64) =
  # Median offset and time per bin, ascending; linear between and beyond.
  # All in the fit's own scratch: nothing is allocated per bin or per row.
  fit.idx.setLen fit.xs.len
  for i in 0 ..< fit.xs.len: fit.idx[i] = i
  sortByKey(fit.idx, fit.xs, fit.keyed)
  fit.bins.setLen 0
  var i = 0
  while i < fit.idx.len:
    let b = floor(fit.xs[fit.idx[i]] / width)
    fit.sideA.setLen 0
    fit.sideB.setLen 0
    while i < fit.idx.len and floor(fit.xs[fit.idx[i]] / width) == b:
      fit.sideA.add fit.xs[fit.idx[i]]
      fit.sideB.add fit.ts[fit.idx[i]]
      inc i
    fit.bins.add (median(fit.sideA), median(fit.sideB))
  template bs: untyped = fit.bins # in place: a `let` copied the bins for every row
  for r in 0 ..< fit.absOffset.len:
    let x = fit.absOffset[r]
    if not x.isNaN:
      var t: float64
      if bs.len == 1: t = bs[0][1]
      else:
        # The segment holding x; the end segments extend beyond the bins.
        var k = 0
        while k + 2 < bs.len and bs[k + 1][0] < x: inc k
        let f = (x - bs[k][0]) / (bs[k + 1][0] - bs[k][0])
        t = bs[k][1] + f * (bs[k + 1][1] - bs[k][1])
      fit.modelMs[r] = t

# ------------------------------------------------------------ correlation --

proc lagOf[T](a, b: ptr UncheckedArray[T], na, nb, sa, sb, m, maxLag: int): (float64, float64) =
  # The lag (samples) maximizing the normalized correlation of a[sa ..< sa+m]
  # with b[sb + lag ..< sb + lag + m], |lag| ≤ maxLag, windows inside the
  # traces; (NaN, NaN) when no lag fits or a window has no energy.
  var bestLag = NaN
  var best = -Inf
  for lag in -maxLag .. maxLag:
    let s2 = sb + lag
    if sa < 0 or s2 < 0 or sa + m > na or s2 + m > nb: continue
    var xy, xx, yy = 0.0
    for k in 0 ..< m:
      let x = float64(a[sa + k])
      let y = float64(b[s2 + k])
      xy += x * y
      xx += x * x
      yy += y * y
    if xx <= 0 or yy <= 0: continue
    let c = xy / sqrt(xx * yy)
    if c > best:
      best = c
      bestLag = float64(lag)
  if bestLag.isNaN: (NaN, NaN) else: (bestLag, best)

proc correlateInto[T](fit: var ShotFit, b: RowBlock[T], meta: ChunkMeta,
    p: ShotFitParams) =
  let dt = meta.dtMs
  # Rounded and clamped as floats (a long window does not fit an int); no
  # window or lag can usefully exceed the block.
  let m = int(clamp(round(p.windowMs / dt), 2.0, float64(max(2, b.cols))))
  let maxLag = int(min(round(p.maxLagMs / dt), float64(b.cols)))
  template ord: untyped = fit.order
  template startCol(r: int): int =
    # The window's first column: windowMs / 4 before the model time.
    let t0 = meta.keys(r).t0Ms
    let c = (fit.segmentMs[r] - p.windowMs / 4 - (if t0.isNaN: 0.0 else: t0)) / dt -
      float64(meta.firstSample)
    if c.isNaN or abs(c) > 1e9: -1 else: int(round(c))
  template pairLag(r, q: int): (float64, float64) =
    let sa = startCol(r)
    let sb = startCol(q)
    if sa < 0 or sb < 0: (NaN, NaN)
    else: lagOf(b.rowPtr(r), b.rowPtr(q), meta.liveSamples(r, b.cols),
      meta.liveSamples(q, b.cols), sa, sb, m, maxLag)
  let tol = p.lagToleranceMs / dt
  for k in 0 ..< ord.len:
    let r = ord[k]
    var lags: array[2, float64] = [NaN, NaN]
    for side, j in [k - 1, k + 1]:
      if j < 0 or j >= ord.len: continue
      let (lag, c) = pairLag(r, ord[j])
      lags[side] = lag
      if lag.isNaN: continue
      if fit.lagMs[r].isNaN or abs(lag) * dt < abs(fit.lagMs[r]) or
          (abs(lag) * dt == abs(fit.lagMs[r]) and c > fit.coherence[r]):
        fit.lagMs[r] = lag * dt
        fit.coherence[r] = c
    # Odd one out: both neighbours disagree with it and agree with each other.
    if abs(lags[0]) > tol and abs(lags[1]) > tol:
      let (between, c) = pairLag(ord[k - 1], ord[k + 1])
      if abs(between) <= tol and c >= p.minCoherence: fit.suspect[r] = true

# -------------------------------------------------------------------- fit --

proc sortByKey(idx: var seq[int], key: openArray[float64],
    keyed: var seq[(float64, int)]) {.raises: [].} =
  # Sorts idx by key[idx[i]], stable (ties keep their order), with `keyed`
  # as its scratch.
  keyed.setLen idx.len
  for i, v in idx: keyed[i] = (key[v], v)
  keyed.sort() # by key, then index: a stable order (no closure: raises nothing)
  for i in 0 ..< idx.len: idx[i] = keyed[i][1]

proc fitPicks(meta: ChunkMeta, picks: openArray[FirstBreakPick], p: ShotFitParams,
    fit: var ShotFit) =
  p.checkParams
  if picks.len != meta.len:
    badParams($picks.len & " picks for " & $meta.len & " rows")
  let n = meta.len
  fit.model = p.model
  fit.interceptMs = NaN
  fit.v1 = NaN
  fit.v2 = NaN
  fit.crossover = NaN
  fit.asymmetryMs = NaN
  fit.threshold = NaN
  for s in [addr fit.position, addr fit.modelMs, addr fit.residualMs, addr fit.localMs,
      addr fit.lagMs, addr fit.coherence, addr fit.absOffset, addr fit.segmentMs]:
    s[].setLen n
    for i in 0 ..< n: s[][i] = NaN
  fit.suspect.setLen n
  for i in 0 ..< n: fit.suspect[i] = false
  positionsInto(meta, fit.position)
  # Spread order: live rows with a position.
  fit.order.setLen 0
  for r in 0 ..< n:
    if not fit.position[r].isNaN: fit.order.add r
  sortByKey(fit.order, fit.position, fit.keyed)
  # The confident picks; |offset| of every live row with one, in modelMs.
  fit.xs.setLen 0
  fit.ts.setLen 0
  for r in 0 ..< n:
    if not meta.isLive(r): continue
    let x = abs(meta.keys(r).offset)
    if x.isNaN: continue
    fit.absOffset[r] = x
    let q = picks[r]
    if not q.timeMs.isNaN and q.confidence >= p.minConfidence:
      fit.xs.add x
      fit.ts.add q.timeMs
  fit.used = fit.xs.len
  if fit.used < 3:
    return
  # The two-segment fit always (the correlation windows sit on it), the
  # binned model for the residuals when asked for.
  fit.fitTwoSegment
  case p.model
  of rmTwoSegment:
    for r in 0 ..< n: fit.modelMs[r] = fit.segmentMs[r]
  of rmOffsetBinned: fit.fitBinned(p.binWidth)
  for r in 0 ..< n:
    let q = picks[r]
    if meta.isLive(r) and not q.timeMs.isNaN and q.confidence >= p.minConfidence:
      fit.residualMs[r] = q.timeMs - fit.modelMs[r]
  # Local residuals along the spread: minus the neighbours' median.
  template ord: untyped = fit.order
  for k in 0 ..< ord.len:
    let r = ord[k]
    if fit.residualMs[r].isNaN: continue
    # The nearest localTraces residuals on each side (traces without one
    # are skipped, so a gap does not leave a bad neighbour alone).
    fit.tmp.setLen 0
    for dir in [-1, 1]:
      var j = k + dir
      var got = 0
      while j >= 0 and j < ord.len and got < p.localTraces:
        let v = fit.residualMs[ord[j]]
        if not v.isNaN:
          fit.tmp.add v
          inc got
        j += dir
    if fit.tmp.len >= 2: fit.localMs[r] = fit.residualMs[r] - median(fit.tmp)
  # Threshold from the MAD of the local residuals.
  fit.tmp.setLen 0
  for v in fit.localMs:
    if not v.isNaN: fit.tmp.add v
  let mid = median(fit.tmp)
  for v in fit.tmp.mitems: v = abs(v - mid)
  let mad = median(fit.tmp)
  fit.threshold = max(p.maxResidualMs, p.madFactor * 1.4826 * (if mad.isNaN: 0.0 else: mad))
  for r in 0 ..< n:
    if abs(fit.localMs[r]) > fit.threshold: fit.suspect[r] = true
  # Symmetry: negative side minus positive side.
  fit.sideA.setLen 0
  fit.sideB.setLen 0
  for r in 0 ..< n:
    let x = fit.position[r]
    if not (fit.residualMs[r].isNaN or x.isNaN or x == 0):
      if x < 0: fit.sideA.add fit.residualMs[r] else: fit.sideB.add fit.residualMs[r]
  if fit.sideA.len > 0 and fit.sideB.len > 0:
    fit.asymmetryMs = median(fit.sideA) - median(fit.sideB)

proc fitShot*(meta: ChunkMeta, picks: openArray[FirstBreakPick], params: ShotFitParams,
    fit: var ShotFit) =
  ## Fits one shot's rows (`meta`, with `picks[row]` from `qc/firstbreak`)
  ## and flags suspect traces from the residuals (no cross-correlation).
  ## Fewer than 3 usable picks give NaN models and residuals and no flags.
  ## Raises `ValueError` for bad `params` or `picks.len != meta.len`.
  fitPicks(meta, picks, params, fit)

proc fitShot*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    picks: openArray[FirstBreakPick], params: ShotFitParams, fit: var ShotFit) =
  ## `fitShot` with the shot's samples `b`: also cross-correlates each
  ## trace with its neighbours (when `params.correlate`) and flags the odd
  ## ones out (module documentation). Raises `ValueError`
  ## also when `b` does not match `meta`.
  meta.checkShape(b)
  fitPicks(meta, picks, params, fit)
  if params.correlate and fit.used >= 3:
    fit.correlateInto(b, meta, params)

proc fitShot*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    picks: openArray[FirstBreakPick], params: ShotFitParams, fit: var ShotFit) =
  ## `fitShot` on a row-major `[rows, samples]` tensor.
  fitShot(rowBlock(samples), meta, picks, params, fit)
