## Robust start and trimmed model of first-break picks.
##
## ```nim
## proc fit(use: openArray[bool]): seq[float64] =
##   let f = lineFit(offset, picks, use)
##   if not f.ok: return
##   result = newSeq[float64](offset.len)
##   for i, x in offset: result[i] = f.interceptMs + f.msPerUnit * x
## let t = trimmedModel(offset, picks, confidence, fit)
## ```
##
## **Start.** `medianLineStart` ranks the confident picks by their distance
## from the straight line of least median of squares: the best of `StartLines`
## lines through pairs of picks, the pairs chosen by a fixed generator. That
## line is not pulled by up to half of the picks being noise, which a fit of
## all of them is.
##
## **Trim.** `trimmedModel` starts from the closer half of those picks (at
## least `MinTrimmedPicks`). Then, up to `TrimRounds` times, it fits the
## caller's model and keeps the confident picks within
## `max(TrimFloorMs, TrimSigmas * sigma)` of it, until the set no longer
## changes. Picks dropped may come back when the model moves to them. Not
## trusted when the fit gives nothing or fewer than `MinTrimmedPicks` picks
## are left. A start that lies on a second segment drops the first as gross:
## it is tried again from the picks kept and the nearest dropped ones, and
## kept when that trim holds at least `MinSecondPicks` more picks as tightly
## (`RestartSigmaShare`, `RestartSigmaFloorMs`).
##
## **A second segment.** `keepSecond` is whether it stays: at least
## `MinSecondPicks` trimmed picks beyond the crossover, and a robust sigma
## below `SecondGain` of the single line's, by at least `SecondGainMs`.
##
## Times are milliseconds. A pick counts from confidence `ConfidentPick`.
## An empty input is NaN or an empty result, never an error. `offset`,
## `pickMs` and `confidence` are parallel, as are `x`, `t` and `use`:
## different lengths raise `ValueError`.

import std/[algorithm, math]

const
  ConfidentPick* = 0.5
    ## The confidence from which a pick counts (`ShotFitParams.minConfidence`).
  MadToSigma* = 1.4826
    ## A MAD scaled to a normal distribution's standard deviation.
  TrimSigmas* = 3.0
    ## Picks farther than this many sigmas from the model are left out of it.
  TrimFloorMs* = 12.0
    ## The trim is never closer than this (ms).
  TrimRounds* = 5
    ## The most fit-and-drop rounds.
  MinTrimmedPicks* = 10
    ## The fewest picks left that make a model to trust.
  StartLines* = 300
    ## The pairs of picks the start's robust line is chosen from.
  MinSecondPicks* = 15
    ## The fewest trimmed picks beyond the crossover that keep a second
    ## segment, and the fewest dropped picks that start another trim.
  SecondGain* = 0.9
    ## The second segment must bring the robust sigma below this share of the
    ## single line's.
  SecondGainMs* = 0.5
    ## And by at least this many ms: on picks that all lie on the line within
    ## a sample, a second segment of the same velocity is no finding.
  RestartSigmaShare* = 1.1
    ## A trim restarted from the picks the first one dropped replaces it when
    ## its robust sigma is at most this share of the first's.
  RestartSigmaFloorMs* = 2.0
    ## The floor of the first trim's sigma in that comparison (ms).

type
  ModelFn* = proc (use: openArray[bool]): seq[float64] {.closure.}
    ## The model's time (ms) at every trace, fitted to the picks `use` marks;
    ## empty when it cannot be fitted.

  Trimmed* = object
    ## What `trimmedModel` found.
    trusted*: bool       ## At least `MinTrimmedPicks` picks were left, and a model was fitted.
    model*: seq[float64] ## The last fit's times; empty unless `trusted`.
    kept*: seq[bool]     ## The picks the last fit used.
    used*: int           ## How many.
    sigmaMs*: float64    ## The robust sigma of the kept picks' residuals; NaN for none.
    rounds*: int         ## Fits made.

func median*(xs: openArray[float64]): float64 =
  ## The median of `xs`; NaN when empty. The middle value, or the mean of the
  ## two middle values.
  if xs.len == 0: return NaN
  var s = @xs
  s.sort()
  let h = s.len div 2
  if s.len mod 2 == 1: s[h] else: (s[h - 1] + s[h]) / 2

func robustSigma*(xs: openArray[float64]): float64 =
  ## `MadToSigma` times the median absolute deviation about the median; NaN
  ## when empty.
  if xs.len == 0: return NaN
  let m = median(xs)
  var d = newSeq[float64](xs.len)
  for i, x in xs: d[i] = abs(x - m)
  MadToSigma * median(d)

proc badParams(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func lineFit*(x, t: openArray[float64]; use: openArray[bool]):
    tuple[interceptMs, msPerUnit: float64; ok: bool] =
  ## The straight line `t = a + s x` through the points `use` marks, by least
  ## absolute deviations (iteratively reweighted least squares, 40 rounds).
  ## `ok` is false for fewer than two points, all at one `x`, or a line that
  ## does not rise. Raises `ValueError` when `x`, `t` and `use` differ in length.
  if x.len != t.len or x.len != use.len:
    badParams("lineFit: " & $x.len & " x, " & $t.len & " t, " & $use.len & " use")
  result = (NaN, NaN, false)
  var xs, ts: seq[float64]
  for i in 0 ..< x.len:
    if use[i] and not x[i].isNaN and not t[i].isNaN:
      xs.add x[i]
      ts.add t[i]
  if xs.len < 2 or xs.min == xs.max: return
  var w = newSeq[float64](xs.len)
  for i in 0 ..< w.len: w[i] = 1.0
  var (a, s) = (0.0, 0.0)
  for round in 0 ..< 40:
    var (sw, sx, st, sxx, sxt) = (0.0, 0.0, 0.0, 0.0, 0.0)
    for i in 0 ..< xs.len:
      sw += w[i]
      sx += w[i] * xs[i]
      st += w[i] * ts[i]
      sxx += w[i] * xs[i] * xs[i]
      sxt += w[i] * xs[i] * ts[i]
    let den = sw * sxx - sx * sx
    if abs(den) < 1e-12 * sw * sxx: return
    s = (sw * sxt - sx * st) / den
    a = (st - s * sx) / sw
    for i in 0 ..< xs.len: w[i] = 1.0 / max(abs(ts[i] - a - s * xs[i]), 0.25)
  if s > 0: result = (a, s, true)

func selectNth(a: var openArray[float64]; k: int) =
  # Puts the k-th smallest of a (from 0) at a[k], the smaller ones before it
  # and the larger after (quickselect; a holds no NaN).
  var (lo, hi) = (0, a.high)
  while lo < hi:
    let mid = lo + (hi - lo) div 2
    # the median of three as the pivot
    if a[mid] < a[lo]: swap(a[mid], a[lo])
    if a[hi] < a[lo]: swap(a[hi], a[lo])
    if a[hi] < a[mid]: swap(a[hi], a[mid])
    let pivot = a[mid]
    var (i, j) = (lo, hi)
    while i <= j:
      while a[i] < pivot: inc i
      while pivot < a[j]: dec j
      if i <= j:
        swap(a[i], a[j])
        inc i
        dec j
    if k <= j: hi = j
    elif k >= i: lo = i
    else: return

func medianIfCloser*(dev: var openArray[float64]; best: float64): float64 =
  ## The median of `dev` when it can lie below `best`, else +Inf. `dev` is
  ## rearranged when that median is selected. A NaN in `dev` is taken by a
  ## full sort, whose order with a NaN in it is whatever the sort makes of it.
  ##
  ## At least half the distances must lie below `best` before the median is
  ## selected, one more when the count is odd: fewer, and the median cannot
  ## be below `best`. For an even count the bound is half, not one more,
  ## because the median is the mean of the two middle values.
  result = Inf
  if dev.len == 0: return
  let half = dev.len div 2
  let need = if dev.len mod 2 == 1: half + 1 else: half # distances below `best` a line needs
  var below = 0
  var notNumber = false
  for d in dev:
    below += int(d < best)
    notNumber = notNumber or d.isNaN
  if notNumber:
    result = median(dev)
  elif below >= need:
    selectNth(dev, half)
    result = dev[half]
    if dev.len mod 2 == 0:
      var lower = dev[0]
      for k in 1 ..< half: lower = max(lower, dev[k])
      result = (lower + result) / 2

func medianLineStart*(offs, pickMs: openArray[float64]; confident: openArray[bool]):
    seq[(float64, int)] =
  ## The confident picks with their distance from the straight line of least
  ## median of squares (the best of `StartLines` lines through pairs of picks,
  ## the pairs chosen by a fixed generator), sorted nearest first. That line
  ## is not pulled by up to half of the picks being noise, which a fit of all
  ## of them is. Raises `ValueError` when `offs`, `pickMs` and `confident`
  ## differ in length.
  ##
  ## A line's median distance is looked for only when it can beat the best so
  ## far (`medianIfCloser`): at least half of the distances must lie below it,
  ## which one count over them shows (a random pair of picks mostly gives a
  ## line that fails that), and then the median is selected, not sorted. The
  ## line chosen is the one a full sort of every line's distances chooses.
  if offs.len != pickMs.len or offs.len != confident.len:
    badParams("medianLineStart: " & $offs.len & " offs, " & $pickMs.len &
      " pickMs, " & $confident.len & " confident")
  var ids: seq[int]
  for i in 0 ..< pickMs.len:
    if confident[i] and not offs[i].isNaN: ids.add i
  if ids.len < 2: return
  var state = 88172645463325252'u64
  func draw(state: var uint64; n: int): int =
    state = state * 6364136223846793005'u64 + 1442695040888963407'u64
    int((state shr 33) mod uint64(n))
  var best = Inf
  var (bestA, bestS) = (0.0, 0.0)
  var dev = newSeq[float64](ids.len)
  for line in 0 ..< StartLines:
    let (i, j) = (ids[draw(state, ids.len)], ids[draw(state, ids.len)])
    if offs[i] != offs[j]:
      let s = (pickMs[j] - pickMs[i]) / (offs[j] - offs[i])
      let a = pickMs[i] - s * offs[i]
      for k, id in ids: dev[k] = abs(pickMs[id] - a - s * offs[id])
      let m = medianIfCloser(dev, best)
      if m < best: (best, bestA, bestS) = (m, a, s)
  if best == Inf: return
  for id in ids: result.add (abs(pickMs[id] - bestA - bestS * offs[id]), id)
  result.sort()

proc trimmedModel*(offsetAbs, pickMs, confidence: openArray[float64]; fit: ModelFn): Trimmed =
  ## A model that noise picks do not set. It starts from the closer half of
  ## the confident picks to a robust straight line (`medianLineStart`); then,
  ## up to `TrimRounds` times, `fit` on the picks kept, and the confident
  ## picks within `max(TrimFloorMs, TrimSigmas * sigma)` of that model kept
  ## for the next round (sigma: the robust sigma of the picks it was fitted
  ## to), until the set no longer changes. Picks dropped may come back when
  ## the model moves to them. Not `trusted` when `fit` gives nothing or fewer
  ## than `MinTrimmedPicks` picks are left. A trim whose start lies on the
  ## record's second segment drops the first as gross: it is made again from
  ## the picks kept and the nearest dropped ones (the end of this proc).
  ## Raises `ValueError` when `offsetAbs`, `pickMs` and `confidence` differ
  ## in length.
  if offsetAbs.len != pickMs.len or offsetAbs.len != confidence.len:
    badParams("trimmedModel: " & $offsetAbs.len & " offsetAbs, " & $pickMs.len &
      " pickMs, " & $confidence.len & " confidence")
  let n = pickMs.len
  result = Trimmed(kept: newSeq[bool](n), sigmaMs: NaN)
  var confident = newSeq[bool](n)
  var use = newSeq[bool](n)
  var count = 0
  for i in 0 ..< n:
    confident[i] = not pickMs[i].isNaN and confidence[i] >= ConfidentPick
    use[i] = confident[i]
    if use[i]: inc count
  result.kept = use
  result.used = count
  if count < MinTrimmedPicks: return
  let near = medianLineStart(offsetAbs, pickMs, confident)
  if near.len < MinTrimmedPicks: return
  let keep = max(MinTrimmedPicks, (near.len + 1) div 2)
  var start0 = newSeq[bool](n)
  for k in 0 ..< keep: start0[near[k][1]] = true
  let pmS = @pickMs
  proc fromStart(start: seq[bool]; startCount: int): Trimmed =
    result = Trimmed(kept: start, used: startCount, sigmaMs: NaN)
    var use = start
    var count = startCount
    var model: seq[float64] = @[]
    for round in 1 .. TrimRounds + 1:
      model = fit(use)
      inc result.rounds
      if model.len != n:
        result.used = count
        return
      var res: seq[float64] = @[]
      for i in 0 ..< n:
        if use[i] and not model[i].isNaN: res.add pmS[i] - model[i]
      let sigma = robustSigma(res)
      let cut = max(TrimFloorMs, TrimSigmas * (if sigma.isNaN: 0.0 else: sigma))
      var next = newSeq[bool](n)
      var nextCount = 0
      for i in 0 ..< n:
        next[i] = confident[i] and not model[i].isNaN and abs(pmS[i] - model[i]) <= cut
        if next[i]: inc nextCount
      if next == use or round > TrimRounds:
        return Trimmed(trusted: count >= MinTrimmedPicks, model: model, kept: use, used: count,
          sigmaMs: sigma, rounds: result.rounds)
      if nextCount < MinTrimmedPicks:
        result.used = nextCount
        return
      use = next
      count = nextCount
      result.kept = use
      result.used = count
  let first = fromStart(start0, keep)
  result = first
  # A strong bend with most picks on the second segment: the start (the closer half to a
  # straight line) lies on that segment, and the picks on the first are dropped as gross. Start
  # again from what was kept and the nearest of the dropped picks to their own line, and keep
  # that trim when it holds at least `MinSecondPicks` more picks as tightly.
  if not first.trusted: return
  var rest = newSeq[bool](n)
  var nRest = 0
  for i in 0 ..< n:
    rest[i] = confident[i] and not first.kept[i]
    if rest[i]: inc nRest
  if nRest < MinSecondPicks: return
  let near2 = medianLineStart(offsetAbs, pickMs, rest)
  let k2 = max(MinSecondPicks, (near2.len + 1) div 2)
  if near2.len < k2: return
  var start2 = first.kept
  var c2 = first.used
  for k in 0 ..< k2:
    if not start2[near2[k][1]]:
      start2[near2[k][1]] = true
      inc c2
  let second = fromStart(start2, c2)
  if second.trusted and second.used >= first.used + MinSecondPicks and
      second.sigmaMs <= RestartSigmaShare * max(first.sigmaMs, RestartSigmaFloorMs):
    result = second
  else:
    discard fit(first.kept) # the caller's view of the last fit is the first's again

func keepSecond*(beyond: int; sigmaOne, sigmaTwo: float64): bool =
  ## Whether a second segment stays: at least `MinSecondPicks` trimmed picks
  ## lie beyond the crossover and it brings the robust sigma below
  ## `SecondGain` of the single line's, and by at least `SecondGainMs`.
  beyond >= MinSecondPicks and not sigmaOne.isNaN and not sigmaTwo.isNaN and
    sigmaTwo < SecondGain * sigmaOne and sigmaOne - sigmaTwo >= SecondGainMs
