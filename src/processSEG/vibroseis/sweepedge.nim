## The top frequency of a reference sweep, read from its own samples, with the reading's
## uncertainty: the sweep's frequency law near its top, taken to where the sweep ends. The
## shape of the ramp at the end is not assumed: its fall is fitted against a few shapes (powers
## of the time left, the cosine squared) and the data weigh them; where none of them is
## trusted the reading says so.
##
## - The envelope is the analytic one (processSEG's `HilbertPlan`) of the record less its
##   mean, squared and averaged over a window. The noise's mean square is estimated from the
##   scatter of the squared envelope about its trend over the sweep's plateau (not from a quiet
##   stretch, which a reference stored at the sweep's length lacks) and taken off.
## - The plateau is taken from the reference's measured flat part (`SweepExtent`) and extended
##   while the envelope stays above half its level.
## - The law is fitted over the last part of the plateau (f or ln f against time, a line or a
##   parabola, by Akaike's criterion, outliers dropped), so a sweep with a boost in dB per
##   octave is followed without being named.
## - The fall of the envelope is followed down to the noise's reach (3 % of the amplitude at the
##   least); it ends where it stops steepening (a ring-down or leakage after the ramp leaves it
##   as a gentle slope). A ramp that ends on a step, or on none, ends at the last point the
##   envelope reaches.
## - The uncertainty is three standard errors of the end frequency (on 5 000 synthetic sweeps,
##   every taper shape, four laws, noise and tails, the error lay within it in 77 to 100 % of
##   each family, the step-ended ramps the lowest): the law's error at the end, and the end
##   time's from the fit of the fall, the spread between the shapes that fit about as well, a
##   share of the stretch the fall is extended over and the window's blur, times the sweep's
##   rate. A reading is marked uncertain when it exceeds `UncertainFraction` of the frequency.
##   Where the fall could not be fitted the error is at least what separates the plateau's edge
##   from the last point the envelope reaches.

import std/[math, algorithm]
import ../core/workspace
import ../fft/hilbert
import sweeptrack
from ../qc/breakfit import median

const
  UncertainFraction* = 0.006
    ## an uncertainty (three standard errors) above this fraction of the top frequency marks
    ## the reading uncertain
  FallHigh = 0.5 # the fall is fitted from this fraction of the plateau's amplitude ...
  FallLow = 0.03 # ... to this one at the least (-30 dB), or the noise's reach, or the knee
  SlopeKnee = 0.35 # the fall ends where its steepness (in ln A) drops under this share of
                     # the steepest so far: a tail leaves the ramp there, as a gentle slope
  Shapes = [(false, 1.0), (false, 1.25), (false, 1.5), (false, 1.75), (false, 2.0),
    (false, 2.5), (false, 3.0), (false, 4.0), (true, 1.0), (true, 2.0)]
    ## the shapes of the end the fall is tried against, each making a straight line in time:
    ## a power of the time left, A ~ x^q (false, q), or a quarter sine to the power p,
    ## A ~ sin^p(pi x / 2) (true, p: p = 2 is the cosine squared ramp, the commonest one)
  Coverage = 3.0 # the stated uncertainty is this many standard errors
  ModelShare = 0.15 # of the extrapolated stretch, counted as error
  MisfitFloor = 0.001 # a fit's misfit below this share of its range tells no shapes apart
  OwnMisfit = 0.25 # shapes are told apart by more than this share of the best one's own misfit
  ExtrapolationShare = 0.1 # of the law's change beyond its fitted points, counted as error
  Overlap = 5.0 # track points counted as one in this many

type
  TopReason* = enum
    ## Why a reading is uncertain, or why there is none. The sentence for a
    ## value is its comment.
    trNone ## No sentence: the reading is sure enough.
    trNoSweep ## the reference has no sweep to read
    trTaperShort ## the end taper of the sweep is too short or too noisy to be fitted
    trNoTop ## the sweep's frequency law gives no top
    trNotSure ## the end of the sweep is not sure enough

  TopReading* = object
    ## `hz` NaN: no reading. `endMs` is where the sweep is taken to end (its top-frequency end).
    hz*: float64
    uncertaintyHz*: float64 = Inf
    uncertain*: bool
    endMs*: float64 = NaN
    lastMs*: float64 = NaN  ## the last point of the fall that was fitted
    rateHzS*: float64 = NaN ## the sweep's rate at its end, Hz per second
    why*: TopReason         ## `trNone` when the reading is sure enough

proc badParams(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

type Poly = object
  ## a polynomial in u = (t - t0) / h fitted by least squares
  c: array[3, float64]
  inv: array[9, float64] # (XtX)^-1, row-major, of the degree + 1 first columns
  deg: int
  t0, h, s: float64      # the residuals' standard deviation `s`
  ok: bool

func powers(u: float64): array[3, float64] = [1.0, u, u * u]

func fitPoly(ts, ys: openArray[float64]; deg: int): Poly =
  ## least squares of degree `deg` (1 or 2) with the outliers (beyond 3.5 median deviations)
  ## dropped, three rounds
  result = Poly(deg: deg, s: Inf)
  let m = deg + 1
  if ts.len < m + 2: return
  result.t0 = 0.5 * (ts[0] + ts[^1])
  result.h = max(1e-9, 0.5 * abs(ts[^1] - ts[0]))
  var keep = newSeq[bool](ts.len)
  for i in 0 ..< ts.len: keep[i] = true
  for round in 0 .. 3:
    var a: array[9, float64] # XtX
    var b: array[3, float64] # Xty
    var count = 0
    for i in 0 ..< ts.len:
      if keep[i]:
        inc count
        let p = powers((ts[i] - result.t0) / result.h)
        for r in 0 ..< m:
          b[r] += p[r] * ys[i]
          for c in 0 ..< m: a[r * m + c] += p[r] * p[c]
    if count < m + 2: return Poly(deg: deg, s: Inf)
    # Gauss-Jordan on [a | 1]
    var inv: array[9, float64]
    for r in 0 ..< m: inv[r * m + r] = 1.0
    for col in 0 ..< m:
      var piv = col
      for r in col + 1 ..< m:
        if abs(a[r * m + col]) > abs(a[piv * m + col]): piv = r
      if abs(a[piv * m + col]) < 1e-12: return Poly(deg: deg, s: Inf)
      if piv != col:
        for c in 0 ..< m:
          swap(a[piv * m + c], a[col * m + c])
          swap(inv[piv * m + c], inv[col * m + c])
      let d = a[col * m + col]
      for c in 0 ..< m:
        a[col * m + c] /= d
        inv[col * m + c] /= d
      for r in 0 ..< m:
        if r != col:
          let f = a[r * m + col]
          for c in 0 ..< m:
            a[r * m + c] -= f * a[col * m + c]
            inv[r * m + c] -= f * inv[col * m + c]
    result.inv = inv
    result.c = [0.0, 0.0, 0.0]
    for r in 0 ..< m:
      for c in 0 ..< m: result.c[r] += inv[r * m + c] * b[c]
    var res = newSeq[float64](ts.len)
    var ss = 0.0
    for i in 0 ..< ts.len:
      let p = powers((ts[i] - result.t0) / result.h)
      var f = 0.0
      for r in 0 ..< m: f += result.c[r] * p[r]
      res[i] = abs(ys[i] - f)
      if keep[i]: ss += res[i] * res[i]
    result.s = sqrt(ss / max(1.0, float64(count - m)))
    result.ok = true
    if round < 3:
      let lim = max(3.5 * 1.4826 * median(res), 1e-9)
      for i in 0 ..< ts.len: keep[i] = res[i] <= lim

func at(p: Poly; t: float64): float64 =
  let u = powers((t - p.t0) / p.h)
  for r in 0 .. p.deg: result += p.c[r] * u[r]

func slope(p: Poly; t: float64): float64 =
  ## d/dt
  let u = (t - p.t0) / p.h
  (p.c[1] + (if p.deg == 2: 2 * p.c[2] * u else: 0.0)) / p.h

func variance(p: Poly; t: float64): float64 =
  ## the prediction's variance at `t`, the points counted as one in `Overlap`
  let x = powers((t - p.t0) / p.h)
  let m = p.deg + 1
  var q = 0.0
  for r in 0 ..< m:
    for c in 0 ..< m: q += x[r] * p.inv[r * m + c] * x[c]
  p.s * p.s * q * Overlap

type Law = object
  ## the frequency law near the top: a polynomial in f or in ln f
  p: Poly
  logged: bool

func hz(l: Law; t: float64): float64 =
  if l.logged: exp(l.p.at(t)) else: l.p.at(t)

func rate(l: Law; t: float64): float64 =
  abs(if l.logged: l.hz(t) * l.p.slope(t) else: l.p.slope(t))

func sd(l: Law; t: float64): float64 =
  ## the standard error of the frequency at `t`
  let v = sqrt(l.p.variance(t))
  if l.logged: v * l.hz(t) else: v

func chooseLaw(ts, fs: openArray[float64]): Law =
  ## the best of a line and a parabola in f and in ln f by Akaike's criterion over the
  ## residuals in Hz (the worst tenth left out)
  var lf = newSeq[float64](fs.len)
  for i in 0 ..< fs.len: lf[i] = ln(fs[i])
  var best = Inf
  for logged in [false, true]:
    for deg in 1 .. 2:
      let p = fitPoly(ts, (if logged: lf else: @fs), deg)
      if p.ok:
        let cand = Law(p: p, logged: logged)
        var r2: seq[float64] = @[]
        for i in 0 ..< ts.len: r2.add (cand.hz(ts[i]) - fs[i]) ^ 2
        r2.sort()
        let keep = max(3, r2.len * 9 div 10)
        var rss = 0.0
        for i in 0 ..< keep: rss += r2[i]
        let aic = float64(keep) * ln(max(rss / float64(keep), 1e-30)) + 2.0 * float64(deg + 1)
        if aic < best:
          best = aic
          result = cand

func meanOf(cum: openArray[float64]; a, b: int): float64 =
  ## the mean of the values in `a ..< b` from their running sums, the range cut to the record
  let (lo, hi) = (max(0, a), min(cum.len - 1, b))
  if hi <= lo: 0.0 else: (cum[hi] - cum[lo]) / float64(hi - lo)

func notBefore*(tEnd, tLast: float64; dir: int): float64 =
  ## `tEnd` kept at or past `tLast`, the last point of the fall that was fitted, in the sweep's
  ## direction (`dir` 1: later in time, -1: earlier): the sweep does not end before its own
  ## envelope does.
  if dir > 0: max(tEnd, tLast) else: min(tEnd, tLast)

type Weighed* = object
  ## the shapes' roots weighed by their fits
  tEnd*: float64   ## the weighted root, kept at or past the last point fitted
  seSq*: float64   ## the weighted squared standard error of the roots that count
  spread*: float64 ## the standard deviation of the roots between the shapes
  best*: int       ## the shape that fits best

func weighShapes*(roots, sses, floors, rootSe: openArray[float64]; n: int;
    kCorr, tLast: float64; dir: int): Weighed =
  ## The shapes weighted by how well each fits (a Gaussian likelihood: the misfit above the
  ## best one's over twice its floor, which includes the noise's share and a share of the
  ## range; the points overlap, so only one in `kCorr` of the `n` counts). The end is never
  ## short of the last point fitted. Raises `ValueError` when the arrays differ in length.
  if roots.len != sses.len or roots.len != floors.len or roots.len != rootSe.len:
    badParams("weighShapes: " & $roots.len & " roots, " & $sses.len & " sses, " &
      $floors.len & " floors, " & $rootSe.len & " rootSe")
  var sw, sr, sr2, se2 = 0.0
  var bestSse = Inf
  for v in sses: bestSse = min(bestSse, v)
  for qi in 0 ..< roots.len:
    if sses[qi] < sses[result.best]: result.best = qi
    if not roots[qi].isNaN:
      let fl = max(floors[qi], OwnMisfit * bestSse / float64(n))
      let w = exp(-(sses[qi] - bestSse) / (2.0 * fl * kCorr))
      sw += w
      sr += w * roots[qi]
      sr2 += w * roots[qi] * roots[qi]
      if w > 1e-3: se2 += w * rootSe[qi] ^ 2
  let mean = sr / sw
  result.spread = sqrt(max(0.0, sr2 / sw - mean * mean))
  result.seSq = se2 / sw
  # never short of the last point fitted
  result.tEnd = notBefore(mean, tLast, dir)

func shapeAmp(sine: bool; q, y: float64): float64 =
  ## the amplitude a shape gives at the straight line's value `y` (0 at the end, 1 at one)
  let v = clamp(y, 0.0, 1.0)
  if sine: pow(sin(PI / 2 * v), q) else: pow(v, q)

proc readTop*(x: openArray[float32]; dtMs: float64; track: SweepTrack;
    extent: SweepExtent): TopReading =
  ## The reference's top frequency: the end of the sweep where its frequency is highest.
  ## `x` is the reference's samples (the first at 0 ms), `track` its frequency track and
  ## `extent` its measured sweep and flat part (where the plateau is; no shape is assumed).
  ## Raises `ValueError` when `track.timeMs` and `track.referenceHz` differ in length.
  if track.timeMs.len != track.referenceHz.len:
    badParams("readTop: " & $track.timeMs.len & " timeMs, " &
      $track.referenceHz.len & " referenceHz")
  result = TopReading(hz: NaN, why: trNoSweep)
  let n = x.len
  if n < 64 or not (dtMs > 0): return
  var ts, fs: seq[float64] = @[]
  for m in 0 ..< track.len:
    let f = track.referenceHz[m]
    if not f.isNaN and f > 0:
      ts.add track.timeMs[m] / 1000
      fs.add f
  if ts.len < 20: return
  let (e0, e1) = (extent.flatStartMs, extent.flatEndMs)
  if e0.isNaN or e1.isNaN: return
  # which end is the top: where the frequency is highest
  let tenth = max(5, ts.len div 10)
  let (fa, fb) = (median(fs[0 ..< tenth]), median(fs[^tenth .. ^1]))
  let dir = if fb > fa: 1 else: -1
  # the envelope: the analytic one of the record less its mean, squared, as running sums
  var mean = 0.0
  for v in x: mean += float64(v)
  mean /= float64(n)
  var xs = newSeq[float64](n)
  for j in 0 ..< n: xs[j] = float64(x[j]) - mean
  var env = newSeq[float64](n)
  block:
    let hp = initHilbertPlan(float64, n)
    var ws = initWorkspace()
    ws.reserve(hp.workspaceSize)
    hp.envelopeInto(xs, env, ws)
  var cum = newSeq[float64](n + 1)
  for j in 0 ..< n: cum[j + 1] = cum[j] + env[j] * env[j]
  let (jIn, jOut) = (clamp(int(round((if dir > 0: e0 else: e1) / dtMs)), 0, n - 1),
    clamp(int(round((if dir > 0: e1 else: e0) / dtMs)), 0, n - 1))
  let inside = (jOut - jIn) * dir
  let skip = min(int(round(200.0 / dtMs)), inside div 8)
  let reach = min(inside, int(round(800.0 / dtMs)))
  if inside < 100 or reach - skip < 50: return
  # the plateau's level and the noise in it: the squared envelope of a sweep in noise has the
  # mean m = A^2 + N and the variance 2 N A^2 + N^2 (N: the noise's share), so N follows from
  # the scatter about the trend
  # (the stretch is cut where the frequency falls under half the top: the envelope of a few low
  # cycles ripples, which is no noise)
  var rFar = jOut - dir * reach
  let top0 = max(fa, fb)
  var jf = jOut - dir * skip
  while (jf - dir * 10 - rFar) * dir > 0 and jf - dir * 10 >= 0 and jf - dir * 10 < n:
    let i = clamp(lowerBound(ts, float64(jf - dir * 10) * dtMs / 1000), 0, ts.len - 1)
    if fs[i] < 0.5 * top0: break
    jf -= dir * 10
  if abs(jf - (jOut - dir * skip)) >= 50: rFar = jf
  let (rA, rB) = (rFar, jOut - dir * skip)
  let (rLo, rHi) = (min(rA, rB), max(rA, rB))
  let m = meanOf(cum, rLo, rHi + 1)
  if not (m > 0): return
  let wt = max(5, int(round(100.0 / dtMs))) div 2 * 2 + 1
  func scatter(k: int): float64 =
    ## the variance of the squared envelope's `k`-sample means about their 100 ms mean
    for j in rLo .. rHi:
      var s = 0.0
      let (a, b) = (j - k div 2, j + k div 2 + 1)
      s = meanOf(cum, a, b) - meanOf(cum, j - wt div 2, j + wt div 2 + 1)
      result += s * s
    result /= float64(rHi - rLo + 1)
  let v = scatter(1) * float64(wt) / float64(wt - 1)
  var noise = m - sqrt(max(0.0, m * m - v))
  noise = min(noise, 0.9 * m)
  let level = m - noise
  # the correlation length of the squared envelope's noise, in samples
  let kb = max(3, int(round(10.0 / dtMs)))
  let rho = if v > 0: clamp(float64(kb) * scatter(kb) / v, 1.0, float64(kb)) else: 1.0
  # the window: wide enough that the lowest level the fall is fitted to stands 3 scatters above
  # the noise
  func reachOf(k: int): float64 =
    ## the lowest amplitude (of the plateau's) the noise leaves readable with `k` samples
    let neff = float64(k) / rho
    sqrt(9.0 * noise * (1.0 + sqrt(1.0 + neff / 9.0)) / neff / level)
  let kMax = max(3, int(round(400.0 / dtMs)))
  var k = max(3, int(round(6.0 / dtMs)))
  while k < kMax and reachOf(k) > 0.12: k = max(k + 1, int(float64(k) * 1.25))
  let lowest = max(FallLow, reachOf(k))
  let highest = clamp(4.0 * lowest, 0.3, FallHigh)
  var pe = newSeq[float64](n) # the squared amplitude as a share of the plateau's
  for j in 0 ..< n:
    pe[j] = max(0.0, meanOf(cum, j - k div 2, j + k div 2 + 1) - noise) / level
  # the plateau's top-side edge: from a point well inside it, to where the envelope falls
  # below half the plateau's level
  var jb = jOut - dir * skip
  while jb + dir >= 0 and jb + dir < n and pe[jb + dir] >= 0.5: jb += dir
  let tEdge = float64(jb) * dtMs / 1000
  # the law near the top: the last part of the plateau
  let plateauS = float64(abs(jb - jIn)) * dtMs / 1000
  let lawSpan = min(0.9 * plateauS, max(1.0, 0.3 * plateauS))
  var lt, lf: seq[float64] = @[]
  for i in 0 ..< ts.len:
    if (ts[i] - tEdge) * float64(dir) <= 0 and abs(tEdge - ts[i]) <= lawSpan:
      lt.add ts[i]
      lf.add fs[i]
  if lt.len < 20: return
  let law = chooseLaw(lt, lf)
  if not law.p.ok: return
  # the fall: from the plateau's edge outward while the envelope stays above `lowest`
  let width = float64(k) * dtMs / 1000
  var tAll, aAll: seq[float64] = @[]
  var j = jb
  var below = 0 # samples in a row under the lowest level
  var scanLast = tEdge
  while j + dir >= 0 and j + dir < n and below < max(3, k):
    j += dir
    let a = sqrt(pe[j])
    if a < lowest:
      inc below
    else:
      below = 0
      scanLast = float64(j) * dtMs / 1000
    tAll.add float64(j) * dtMs / 1000
    aAll.add max(a, 1e-9)
  var tt, aa: seq[float64] = @[]
  for i in 0 ..< tAll.len:
    if (tAll[i] - scanLast) * float64(dir) <= 0 and aAll[i] >= lowest and aAll[i] <= highest:
      tt.add tAll[i]
      aa.add aAll[i]
  # a step: the envelope is down to the fit's levels within two windows of the plateau's edge,
  # so there is no ramp to fit
  let kneeSpan = max(3, k)
  var step = false
  if tt.len > 0 and abs(tt[0] - tEdge) < 2.0 * width:
    tt.setLen 0
    aa.setLen 0
    step = true
  # the knee: a taper falls ever more steeply (in ln A) towards its end, a tail or a floor
  # leaves it as a gentle slope; the fall is kept to where its steepness (over `kneeSpan`
  # samples) stays above `SlopeKnee` of the steepest so far
  if tt.len > 2 * kneeSpan + 3:
    var steepest = 0.0
    var cut = tt.len
    for i in kneeSpan ..< tt.len - kneeSpan:
      let sl = abs(ln(aa[i - kneeSpan]) - ln(aa[i + kneeSpan])) /
        (2 * float64(kneeSpan) * dtMs / 1000)
      steepest = max(steepest, sl)
      if steepest > 0 and sl < SlopeKnee * steepest:
        cut = i
        break
    tt.setLen cut
    aa.setLen cut
  let tLast = if tt.len > 0: tt[^1] else: tEdge
  # each shape: the root of its straight line in time, its misfit and the noise's share of it
  var roots, sses, floors, spans, lineA, lineB, rootSe: seq[float64] = @[]
  let kCorr = float64(max(k, 2 * kneeSpan)) # the points are one in this many independent
  let nFit = tt.len
  var bestSse = Inf
  let relNoise = noise / level
  if nFit >= 4:
    let kEff = float64(max(k, 3)) * 1.5
    for (sine, q) in Shapes:
      var ys: seq[float64] = @[]
      var share = 0.0
      for a in aa:
        ys.add (if sine: arcsin(min(1.0, pow(a, 1.0 / q))) * 2.0 / PI else: pow(a, 1.0 / q))
        # scatter of the mean square (relative to the plateau's) and of the amplitude
        let sig = sqrt(2.0 / kEff * (relNoise * relNoise + 2.0 * relNoise * a * a)) +
          0.01 * a * a
        let da = sig / (2.0 * a)
        let sl = pow(a, 1.0 / q - 1.0) / q
        share += (da * (if sine: 2.0 / PI * sl / sqrt(max(1e-9, 1.0 - pow(a, 2.0 / q)))
          else: sl)) ^ 2
      let p = fitPoly(tt, ys, 1)
      if not p.ok or p.c[1] / p.h * float64(dir) > 0:
        roots.add NaN
        sses.add Inf
        floors.add 0.0
        spans.add 1.0
        lineA.add 0.0
        lineB.add 0.0
        rootSe.add Inf
      else:
        # y = c0 + c1 u, u = (t - t0) / h: the root is where y = 0
        let root = p.t0 - p.c[0] * p.h / p.c[1]
        let span = max(1e-9, abs(ys[0] - ys[^1]))
        roots.add root
        sses.add p.s * p.s * float64(tt.len - 2)
        spans.add span
        floors.add max(share / float64(tt.len), (MisfitFloor * span) ^ 2)
        lineA.add p.c[0] - p.c[1] * p.t0 / p.h
        lineB.add p.c[1] / p.h
        bestSse = min(bestSse, sses[^1])
        # the scatter is the larger of the fit's own, which a handful of points understates,
        # the noise's expected share and the misfit floor; the points are one in `kCorr`
        let sy = max(p.s, sqrt(floors[^1]))
        let neff = max(2.0, float64(tt.len) / kCorr)
        rootSe.add sy / abs(p.c[1] / p.h) *
          sqrt((1.0 + 3.0 * (root - p.t0) ^ 2 / (p.h * p.h)) / neff)
  var tEnd = NaN
  var sigT = Inf
  var why = trNone
  if nFit >= 4 and bestSse < Inf:
    let weighed = weighShapes(roots, sses, floors, rootSe, tt.len, kCorr, tLast, dir)
    let (bestAt, se2, spread) = (weighed.best, weighed.seSq, weighed.spread)
    tEnd = weighed.tEnd
    sigT = sqrt(se2 + spread ^ 2 + (ModelShare * abs(tEnd - tLast)) ^ 2 +
      (0.25 * width) ^ 2)
    # a ramp that ends on a floor, then a step: the best shape still has the envelope above the
    # lowest level two windows past the last point reached, the measured one is below it
    let (sine, q) = Shapes[bestAt]
    let probe = scanLast + float64(dir) * 2.0 * width
    let model = shapeAmp(sine, q, lineA[bestAt] + lineB[bestAt] * probe)
    let measured = sqrt(pe[clamp(int(round(probe * 1000.0 / dtMs)), 0, n - 1)])
    if model > 0.7 * lowest and measured < 0.5 * model and
        (tEnd - scanLast) * float64(dir) > 2.0 * width:
      let was = tEnd
      tEnd = scanLast
      sigT = max(sigT, 0.5 * abs(was - scanLast))
  else:
    # the fall could not be fitted: a step or no ramp ends at the plateau's edge; a fall the
    # envelope still follows beyond it ends no earlier than its last point, and the error is at
    # least what lies between
    tEnd = if step or abs(scanLast - tEdge) <= 2.5 * width: tEdge else: scanLast
    sigT = max(0.5 * width, abs(scanLast - tEdge))
    if not (step or abs(scanLast - tEdge) <= 2.5 * width):
      why = trTaperShort
  let est = law.hz(tEnd)
  if not (est > 0 and est < Inf):
    result.why = trNoTop
    return
  # the law taken beyond the points it was fitted to: a tenth of the change over that stretch,
  # times its length over the fitted one, is counted as error (a law of another form than the
  # parabola, or a ramp longer than the plateau's end part)
  let reachOut = min(2.0, abs(tEnd - tEdge) / max(1e-3, lt[^1] - lt[0]))
  let extra = ExtrapolationShare * abs(law.hz(tEnd) - law.hz(tEdge)) * reachOut
  let u = Coverage * sqrt(law.sd(tEnd) ^ 2 + extra ^ 2 + (law.rate(tEnd) * sigT) ^ 2)
  result = TopReading(hz: est, uncertaintyHz: u, endMs: tEnd * 1000, lastMs: tLast * 1000,
    rateHzS: law.rate(tEnd))
  if why != trNone or u > UncertainFraction * est:
    result.uncertain = true
    result.why = if why != trNone: why else: trNotSure
