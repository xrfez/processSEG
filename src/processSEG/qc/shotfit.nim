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
##   offsets (at least 3 picks on each segment) refined by golden section;
##   a single segment when that fits as well;
## - `rmOffsetBinned`: the median pick time in offset bins of `binWidth`,
##   linear between the bins' median offsets and extended linearly beyond
##   the end bins: for refractors no two lines describe. Bins should be
##   narrower than the curve's bends (the crossover), or the traces there
##   get flagged.
##
## Only picks of `confidence ≥ minConfidence` on live rows with a known
## offset count.
##
## **Residuals.** `residualMs` is the pick minus the model. Along the
## spread (traces ordered by their position: signed along the receivers'
## line when the coordinates are known, else by the signed offset), a
## refractor whose depth or velocity varies bends the residuals smoothly,
## while a trace whose header places it at another station jumps.
## `localMs` is the residual minus the median residual of the nearest
## `localTraces` traces with a residual on each side (itself excluded; at
## least 2 in all, else NaN), which removes the smooth part. A trace is suspect when `|localMs|` exceeds
## `max(residualMs, madFactor · 1.4826 · MAD)` of the shot's local
## residuals.
##
## **Cross-correlation** (the form with samples, `correlate`): each trace's
## window of `windowMs` from `windowMs / 4` before its two-segment model
## time (whichever `model` the residuals use: a few bad picks do not move
## an L1 fit, while they can bump a bin's median) is
## correlated with its neighbours' (in spread order), over lags up to
## `maxLagMs`. Windows at the model times align the break when the
## geometry is right; `lagMs` is the lag of the best-correlated neighbour
## (the smaller `|lag|` of the two), `coherence` its correlation
## coefficient. A trace is suspect also when it is the odd one out: its
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
## of the spread minus that of the positive side (NaN for one side): a
## source placed `d` along the line too far toward the positive side makes
## that side early and the other late by `d / v` each, so about `2d / v`.
## A refractor that varies laterally also makes the sides differ by a few
## ms, so compare shots rather than trust one value. The source's position
## is not checked further.
##
## `fitShot` resizes the `ShotFit` seqs to the shot's rows and reuses them
## (QC is not a hot path: it allocates a little scratch per call).

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
    residualMs*: float64 = 12     ## The smallest `|localMs|` that flags a trace.
    madFactor*: float64 = 4       ## Robust threshold, in MADs of `localMs`.
    correlate*: bool = true       ## Cross-correlate neighbours (the form with samples).
    windowMs*: float64 = 80       ## Correlation window.
    maxLagMs*: float64 = 40       ## Largest lag searched.
    lagToleranceMs*: float64 = 8  ## The smallest neighbour lag that flags a trace.
    minCoherence*: float64 = 0.8  ## The neighbours' correlation that makes them agree.

  ShotFit* = object
    ## One shot's fit and findings; per-trace seqs are indexed by row.
    model*: RefractionModel
    interceptMs*: float64 ## The two-segment fit's `a` (fitted with either model).
    v1*, v2*: float64     ## Its segment velocities (offset units/s); `v2` NaN: one segment.
    crossover*: float64   ## Its `xc` (Inf: one segment).
    used*: int            ## Picks the fit used.
    asymmetryMs*: float64 ## Negative-side minus positive-side median residual.
    threshold*: float64   ## The `|localMs|` above which a trace is suspect.
    position*: seq[float64]   ## Signed position along the spread; NaN unknown.
    modelMs*: seq[float64]    ## The model time at the header offset; NaN unknown.
    residualMs*: seq[float64] ## Pick − model; NaN without a used pick.
    localMs*: seq[float64]    ## Residual − neighbours' median; NaN without one.
    lagMs*: seq[float64]      ## Best neighbour lag; NaN without correlation.
    coherence*: seq[float64]  ## Its correlation coefficient; NaN without.
    suspect*: seq[bool]       ## Geometry suspect.
    # Scratch, kept for reuse.
    order: seq[int]
    xs, ts, ws, tmp: seq[float64]
    absOffset, segmentMs: seq[float64] ## |offset| per row; the two-segment model.
    bins: seq[(float64, float64)]

proc badParams(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc checkParams(p: ShotFitParams) =
  if not (p.minConfidence >= 0 and p.minConfidence <= 1):
    badParams("minConfidence must be in 0 .. 1, got " & $p.minConfidence)
  if not (p.binWidth > 0 and p.binWidth < Inf):
    badParams("binWidth must be finite and positive, got " & $p.binWidth)
  if p.localTraces < 1: badParams("localTraces must be ≥ 1, got " & $p.localTraces)
  if not (p.residualMs >= 0 and p.residualMs < Inf and p.madFactor >= 0 and p.madFactor < Inf):
    badParams("residualMs and madFactor must be finite and ≥ 0")
  if not (p.windowMs > 0 and p.windowMs < Inf and p.maxLagMs >= 0 and p.maxLagMs < Inf and
      p.lagToleranceMs >= 0 and p.lagToleranceMs < Inf):
    badParams("correlation windows must be finite (windowMs > 0, the others ≥ 0)")
  if not (p.minCoherence >= -1 and p.minCoherence <= 1):
    badParams("minCoherence must be in -1 .. 1, got " & $p.minCoherence)

func median(x: var seq[float64]): float64 =
  # The median of x (sorted in place); NaN when empty.
  if x.len == 0: return NaN
  x.sort()
  let m = x.len div 2
  if x.len mod 2 == 1: x[m] else: (x[m - 1] + x[m]) / 2

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
    let ux = cos(theta)
    let uy = sin(theta)
    for r in 0 ..< n:
      let q = meta.keys(r)
      pos[r] = if meta.isLive(r): (q.rcvX - q.srcX) * ux + (q.rcvY - q.srcY) * uy else: NaN
  else:
    for r in 0 ..< n:
      pos[r] = if meta.isLive(r): meta.keys(r).offset else: NaN

proc sortByKey(idx: var seq[int], key: openArray[float64])

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
    var m: array[3, array[4, float64]]
    for i in 0 ..< xs.len:
      let b = [1.0, min(xs[i], xc), (if single: 0.0 else: max(xs[i] - xc, 0.0))]
      for r in 0..2:
        for c in 0..2: m[r][c] += ws[i] * b[r] * b[c]
        m[r][3] += ws[i] * b[r] * ts[i]
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
  # each segment), then a golden-section refinement around the best.
  best = l1Fit(fit.xs, fit.ts, fit.ws, Inf, c)
  bestC = c
  fit.tmp.setLen n
  for i in 0 ..< n: fit.tmp[i] = fit.xs[i]
  fit.tmp.sort()
  template valid(c: array[3, float64]): bool = c[1] > 0 and c[2] > 0
  var bestK = -1
  var last = NaN
  for k in 2 .. n - 4:
    let xc = fit.tmp[k]
    if xc == last or xc <= fit.tmp[0] or xc >= fit.tmp[n - 1]: continue
    last = xc
    let s = l1Fit(fit.xs, fit.ts, fit.ws, xc, c)
    if s < best * (1 - 1e-6) and valid(c):
      best = s
      bestC = c
      bestXc = xc
      bestK = k
  if bestK >= 0:
    var a = fit.tmp[max(bestK - 1, 0)]
    var b = fit.tmp[min(bestK + 1, n - 1)]
    const g = 0.6180339887498949
    for iter in 0 ..< 40:
      let x1 = b - g * (b - a)
      let x2 = a + g * (b - a)
      var c1, c2: array[3, float64]
      let s1 = l1Fit(fit.xs, fit.ts, fit.ws, x1, c1)
      let s2 = l1Fit(fit.xs, fit.ts, fit.ws, x2, c2)
      if s1 < best and valid(c1):
        best = s1
        bestC = c1
        bestXc = x1
      if s2 < best and valid(c2):
        best = s2
        bestC = c2
        bestXc = x2
      if s1 <= s2: b = x2 else: a = x1
      if b - a <= 1e-6 * max(1.0, b): break
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
  var idx = newSeq[int](fit.xs.len)
  for i in 0 ..< fit.xs.len: idx[i] = i
  sortByKey(idx, fit.xs)
  fit.bins.setLen 0
  var i = 0
  var bx, bt: seq[float64]
  while i < idx.len:
    let b = floor(fit.xs[idx[i]] / width)
    bx.setLen 0
    bt.setLen 0
    while i < idx.len and floor(fit.xs[idx[i]] / width) == b:
      bx.add fit.xs[idx[i]]
      bt.add fit.ts[idx[i]]
      inc i
    fit.bins.add (median(bx), median(bt))
  for r in 0 ..< fit.absOffset.len:
    let x = fit.absOffset[r]
    if x.isNaN: continue
    let bs = fit.bins
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
  let m = max(2, int(round(p.windowMs / dt)))
  let maxLag = int(round(p.maxLagMs / dt))
  let ord = fit.order
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

proc sortByKey(idx: var seq[int], key: openArray[float64]) =
  # Insertion sort of idx by key[idx[i]] (stable; a shot is small).
  for i in 1 ..< idx.len:
    let v = idx[i]
    var j = i - 1
    while j >= 0 and key[idx[j]] > key[v]:
      idx[j + 1] = idx[j]
      dec j
    idx[j + 1] = v

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
  sortByKey(fit.order, fit.position)
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
  let ord = fit.order
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
  fit.threshold = max(p.residualMs, p.madFactor * 1.4826 * (if mad.isNaN: 0.0 else: mad))
  for r in 0 ..< n:
    if abs(fit.localMs[r]) > fit.threshold: fit.suspect[r] = true
  # Symmetry: negative side minus positive side.
  var neg, pos: seq[float64]
  for r in 0 ..< n:
    let x = fit.position[r]
    if fit.residualMs[r].isNaN or x.isNaN or x == 0: continue
    if x < 0: neg.add fit.residualMs[r] else: pos.add fit.residualMs[r]
  if neg.len > 0 and pos.len > 0: fit.asymmetryMs = median(neg) - median(pos)

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
