## Robust start and trimmed model of first-break picks (qc/breakfit).
##
## The slow references are the full-sort start and the trim that calls it.
## Equality is `==` on float64 (NaN only where both sides are NaN).
##
## Seen fail: `selectNth` swapped the pivot test (`a[i] < pivot` became
## `a[i] <= pivot`). `median_if_closer_contract` then died in `selectNth`
## (`index 4 not in 0 .. 3`) on four tied distances.
##
## Seen fail: `lineFit` left every weight at 1, or ran 1 reweighting round
## instead of 40. `line_fit_reweights_outliers` then missed the
## least-absolute-deviation line (intercept off by about 11 ms).
##
## Seen fail: `trimmedModel`'s restart sigma test replaced by `false`.
## `restart_keeps_the_bend` then kept the first trim (used 62 of 80, the 18
## picks before the crossover dropped).
##
## Seen fail: unequal parallel arrays had no length check.
## `unequal_lengths` died with an IndexDefect (`index out of bounds`) in
## `lineFit`, `medianLineStart` and `trimmedModel`.
##
## Seen fail: `trimmedModel` left `used` as the fitted count when the next
## set fell under `MinTrimmedPicks`. `trim_rejects_a_short_next` then saw
## used=10, not the rejected 9.
##
## Seen fail: `keepSecond` used `sigmaTwo <=` instead of `<`.
## `keep_second_bounds` then accepted sigmaTwo == SecondGain * sigmaOne
## (5.0 and 4.5). `odd_half_keeps_one_more` died when the start kept
## `near.len div 2` (10) instead of `(near.len + 1) div 2` (11).

import std/[algorithm, assertions, math, random]
import processSEG

# --------------------------------------------------------- slow references --
# Full sort of every trial line's distances, instead of the early-out select.

func medianLineStartReference(offs, pickMs: openArray[float64]; confident: openArray[bool]):
    seq[(float64, int)] =
  ## The confident picks with their distance from the straight line of least median of squares
  ## (the best of `StartLines` lines through pairs of picks, the pairs chosen by a fixed
  ## generator), sorted nearest first. That line is not pulled by up to half of the picks being
  ## noise, which a fit of all of them is.
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
      let m = median(dev)
      if m < best: (best, bestA, bestS) = (m, a, s)
  if best == Inf: return
  for id in ids: result.add (abs(pickMs[id] - bestA - bestS * offs[id]), id)
  result.sort()

proc trimmedModelReference(offsetAbs, pickMs, confidence: openArray[float64];
    fit: ModelFn): Trimmed =
  ## A model that noise picks do not set. It starts from the closer half of the confident picks
  ## to a robust straight line (`medianLineStart`); then, up to `TrimRounds` times, `fit` on the
  ## picks kept, and the confident picks within `max(TrimFloorMs, TrimSigmas * sigma)` of that
  ## model kept for the next round (sigma: the robust sigma of the picks it was fitted to),
  ## until the set no longer changes. Picks dropped may come back when the model moves to them.
  ## Not `trusted` when `fit` gives nothing or fewer than `MinTrimmedPicks` picks are left.
  ## A trim whose start lies on the record's second segment drops the first as gross: it is made
  ## again from the picks kept and the nearest dropped ones (the end of this proc).
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
  let near = medianLineStartReference(offsetAbs, pickMs, confident)
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
  if not first.trusted: return
  var rest = newSeq[bool](n)
  var nRest = 0
  for i in 0 ..< n:
    rest[i] = confident[i] and not first.kept[i]
    if rest[i]: inc nRest
  if nRest < MinSecondPicks: return
  let near2 = medianLineStartReference(offsetAbs, pickMs, rest)
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
    discard fit(first.kept)

# ------------------------------------------------------- independent median --

func medianSorted(xs: openArray[float64]): float64 =
  ## Insertion-sort median: the middle value, or the mean of the two middle.
  if xs.len == 0: return NaN
  var s = newSeq[float64](xs.len)
  for i, x in xs: s[i] = x
  for i in 1 ..< s.len:
    let v = s[i]
    var j = i
    while j > 0 and s[j - 1] > v:
      s[j] = s[j - 1]
      dec j
    s[j] = v
  let h = s.len div 2
  if s.len mod 2 == 1: s[h] else: (s[h - 1] + s[h]) / 2

func ladLine(x, t: openArray[float64]): tuple[a, s: float64] =
  ## Least absolute deviations among rising lines through two of the points:
  ## the line with the least sum of absolute residuals.
  result = (NaN, NaN)
  var best = Inf
  for i in 0 ..< x.len:
    for j in i + 1 ..< x.len:
      if x[i] != x[j]:
        let s = (t[j] - t[i]) / (x[j] - x[i])
        if s > 0:
          let a = t[i] - s * x[i]
          var cost = 0.0
          for k in 0 ..< x.len: cost += abs(t[k] - a - s * x[k])
          if cost < best:
            best = cost
            result = (a, s)

func closerSorted(dev: openArray[float64]; best: float64): float64 =
  ## `medianIfCloser` from its contract, the median taken by a full sort.
  result = Inf
  if dev.len == 0: return
  let half = dev.len div 2
  let need = if dev.len mod 2 == 1: half + 1 else: half
  var below = 0
  var notNumber = false
  for d in dev:
    below += int(d < best)
    notNumber = notNumber or d.isNaN
  if notNumber or below >= need:
    result = medianSorted(dev)

type Watch = ref object
  last: seq[bool]
  calls: int

proc watched(off, pick: seq[float64]; w: Watch): ModelFn =
  result = proc (use: openArray[bool]): seq[float64] =
    inc w.calls
    w.last = @use
    let f = lineFit(off, pick, use)
    if not f.ok: return
    result = newSeq[float64](off.len)
    for i, x in off: result[i] = f.interceptMs + f.msPerUnit * x

proc sameFloat(a, b: float64): bool =
  a == b or (a.isNaN and b.isNaN)

proc sameModel(a, b: seq[float64]): bool =
  if a.len != b.len: return false
  for i in 0 ..< a.len:
    if not sameFloat(a[i], b[i]): return false
  true

proc sameTrimmed(a, b: Trimmed): bool =
  a.trusted == b.trusted and a.kept == b.kept and a.used == b.used and
    a.rounds == b.rounds and sameFloat(a.sigmaMs, b.sigmaMs) and sameModel(a.model, b.model)

proc picks(n: int; seed: int; kind: string): tuple[off, pick, conf: seq[float64]] =
  var rng = initRand(seed)
  result.off = newSeq[float64](n)
  result.pick = newSeq[float64](n)
  result.conf = newSeq[float64](n)
  for i in 0 ..< n:
    result.off[i] = 25.0 * float64(i + 1)
    let noise = float64(rng.rand(1000) - 500) / 100.0 # -5.00 .. 4.99 ms
    result.pick[i] = 20.0 + result.off[i] / 3.0 + noise
    result.conf[i] = 0.9
  if kind == "ties" and n > 0:
    var i = 0
    while i < n:
      let t = result.pick[i]
      for j in i ..< min(i + 3, n): result.pick[j] = t
      i += 3
  elif kind == "unconfident":
    for i in 0 ..< n: result.conf[i] = 0.1
  elif kind == "out5" or kind == "out40":
    let frac = if kind == "out5": 0.05 else: 0.40
    let nOut = int(float64(n) * frac)
    for i in 0 ..< nOut:
      result.pick[(i * 17) mod n] += 500.0

proc bend(n: int): tuple[off, pick, conf: seq[float64]] =
  ## A direct wave, then a faster refractor: most picks sit on the second.
  result.off = newSeq[float64](n)
  result.pick = newSeq[float64](n)
  result.conf = newSeq[float64](n)
  let xc = 25.0 * float64(n) * 0.25
  for i in 0 ..< n:
    let x = 25.0 * float64(i + 1)
    result.off[i] = x
    result.pick[i] = if x <= xc: x / 1.5 else: xc / 1.5 + (x - xc) / 3.0
    result.conf[i] = 0.95

proc main*() =
  block median_empty_and_ties:
    doAssert median(newSeq[float64]()).isNaN
    doAssert median(@[3.0]) == 3.0
    doAssert median(@[1.0, 3.0]) == 2.0
    doAssert median(@[1.0, 2.0, 100.0]) == 2.0
    doAssert median(@[4.0, 1.0, 3.0, 2.0]) == 2.5
    doAssert median(@[5.0, 5.0, 5.0, 5.0]) == 5.0
    doAssert median(@[1.0, 3.0, 5.0, 7.0]) == 4.0
    doAssert median(@[7.0, 1.0, 7.0]) == 7.0

  block median_matches_sort:
    var rng = initRand(11)
    for n in [0, 1, 2, 9, 10, 11, 300]:
      var xs = newSeq[float64](n)
      for i in 0 ..< n: xs[i] = float64(rng.rand(50) - 25)
      doAssert sameFloat(median(xs), medianSorted(xs)), "n=" & $n
      if n > 0: doAssert median(xs) == medianSorted(xs), "n=" & $n
      if n >= 4:
        for i in countup(0, n - 1, 2): xs[i] = xs[0]
        doAssert median(xs) == medianSorted(xs), "ties n=" & $n

  block robust_sigma:
    doAssert robustSigma(newSeq[float64]()).isNaN
    doAssert robustSigma(@[2.0, 2.0, 2.0]) == 0.0
    let xs = @[1.0, 2.0, 4.0, 8.0, 16.0]
    let m = medianSorted(xs)
    var d = newSeq[float64](xs.len)
    for i, x in xs: d[i] = abs(x - m)
    doAssert robustSigma(xs) == MadToSigma * medianSorted(d)

  block line_fit_edges:
    var none = lineFit(newSeq[float64](), newSeq[float64](), newSeq[bool]())
    doAssert not none.ok and none.interceptMs.isNaN and none.msPerUnit.isNaN
    none = lineFit(@[1.0], @[2.0], @[true])
    doAssert not none.ok
    none = lineFit(@[1.0, 1.0], @[2.0, 9.0], @[true, true])
    doAssert not none.ok
    none = lineFit(@[0.0, 10.0], @[5.0, 1.0], @[true, true])
    doAssert not none.ok # does not rise
    none = lineFit(@[0.0, 10.0], @[4.0, 4.0], @[true, true])
    doAssert not none.ok # horizontal
    let two = lineFit(@[0.0, 100.0], @[10.0, 40.0], @[true, true])
    doAssert two.ok
    doAssert two.interceptMs == 10.0 and two.msPerUnit == 0.3
    let three = lineFit(@[0.0, 50.0, 100.0], @[10.0, 25.0, 40.0], @[true, true, true])
    doAssert three.ok and three.interceptMs == 10.0 and three.msPerUnit == 0.3
    let skipped = lineFit(@[0.0, 50.0, NaN], @[10.0, 20.0, 99.0], @[true, true, true])
    doAssert skipped.ok and skipped.interceptMs == 10.0 and skipped.msPerUnit == 0.2
    let marked = lineFit(@[0.0, 10.0, 100.0], @[10.0, 999.0, 40.0], @[true, false, true])
    doAssert marked.ok and marked.interceptMs == 10.0 and marked.msPerUnit == 0.3

  block line_fit_reweights_outliers:
    ## 18 picks near `t = 15 + 0.25 x`, five of them 60 ms high. Within 0.05 ms
    ## of intercept and 0.002 ms per unit of the least-absolute-deviation line.
    let noise = [0.4, -1.1, 0.8, -0.3, 1.5, -0.7, 0.2, -1.4, 0.9, -0.5, 1.2,
      -0.8, 0.1, 1.1, -1.3, 0.6, -0.2, 0.7]
    var x = newSeq[float64](noise.len)
    var t = newSeq[float64](noise.len)
    var use = newSeq[bool](noise.len)
    for i, n in noise:
      x[i] = 12.0 * float64(i)
      t[i] = 15.0 + 0.25 * x[i] + n
      use[i] = true
    for k in [2, 7, 9, 13, 16]: t[k] += 60.0
    let fit = lineFit(x, t, use)
    let lad = ladLine(x, t)
    doAssert fit.ok
    doAssert abs(fit.interceptMs - lad.a) < 0.05
    doAssert abs(fit.msPerUnit - lad.s) < 0.002

  block unequal_lengths:
    doAssertRaises ValueError:
      discard lineFit(@[1.0, 2.0], @[3.0, 4.0], @[true])
    doAssertRaises ValueError:
      discard medianLineStart(@[1.0, 2.0], @[3.0, 4.0], @[true])
    proc noFit(use: openArray[bool]): seq[float64] = @[]
    var pick = newSeq[float64](MinTrimmedPicks + 2)
    var conf = newSeq[float64](pick.len)
    for i in 0 ..< pick.len:
      pick[i] = 10.0 + float64(i)
      conf[i] = 1.0
    doAssertRaises ValueError:
      discard trimmedModel(@[1.0], pick, conf, noFit) # offset shorter than the picks

  block keep_second_bounds:
    doAssert not keepSecond(MinSecondPicks - 1, 2.0, 1.0)
    doAssert not keepSecond(MinSecondPicks, NaN, 1.0)
    doAssert not keepSecond(MinSecondPicks, 2.0, NaN)
    doAssert not keepSecond(MinSecondPicks, 2.0, SecondGain * 2.0) # not strictly below
    doAssert not keepSecond(MinSecondPicks, 2.0, 1.6) # share ok, short of SecondGainMs
    doAssert keepSecond(MinSecondPicks, 2.0, 1.5) # 1.5 < 1.8 and 2.0 - 1.5 == 0.5
    # 4.5 == SecondGain * 5, and 5 - 4.5 == SecondGainMs: `<` rejects, `<=` would not
    doAssert not keepSecond(MinSecondPicks, 5.0, SecondGain * 5.0)
    doAssert SecondGainMs == 0.5

  block median_if_closer_contract:
    var empty: seq[float64] = @[]
    doAssert medianIfCloser(empty, 1.0) == Inf
    var oddShort = @[10.0, 20.0, 30.0]
    doAssert medianIfCloser(oddShort, 15.0) == Inf # one below, an odd count needs two
    var evenShort = @[10.0, 20.0, 30.0, 40.0]
    doAssert medianIfCloser(evenShort, 15.0) == Inf # one below, an even count needs half
    var even = @[5.0, 1.0, 9.0, 3.0]
    doAssert medianIfCloser(even, 100.0) == 4.0 # mean of 3 and 5
    var ties = @[2.0, 2.0, 2.0, 2.0]
    doAssert medianIfCloser(ties, 3.0) == 2.0
    var atBest = @[5.0, 5.0, 5.0, 5.0]
    doAssert medianIfCloser(atBest, 5.0) == Inf # none strictly below
    var withNan = @[1.0, NaN, 3.0]
    let got = medianIfCloser(withNan, 100.0)
    let bySort = median(@[1.0, NaN, 3.0])
    doAssert sameFloat(got, bySort)
    var rng = initRand(19)
    for n in [1, 2, 9, 10, 11, 64]:
      var dev = newSeq[float64](n)
      for i in 0 ..< n: dev[i] = float64(rng.rand(40))
      if n >= 4:
        dev[1] = dev[0]
        dev[2] = dev[0]
      let copy = dev
      let best = float64(rng.rand(30))
      doAssert medianIfCloser(dev, best) == closerSorted(copy, best), "n=" & $n

  block start_edges:
    doAssert medianLineStart(newSeq[float64](), newSeq[float64](), newSeq[bool]()).len == 0
    doAssert medianLineStart(@[1.0], @[2.0], @[true]).len == 0
    doAssert medianLineStart(@[1.0, 2.0], @[3.0, 4.0], @[false, true]).len == 0
    doAssert medianLineStart(@[NaN, 2.0], @[3.0, 4.0], @[true, true]).len == 0
    doAssert medianLineStart(@[1.0, 1.0], @[3.0, 4.0], @[true, true]).len == 0
    let onLine = medianLineStart(@[0.0, 10.0], @[5.0, 15.0], @[true, true])
    doAssert onLine.len == 2
    doAssert onLine[0][0] == 0.0 and onLine[1][0] == 0.0
    doAssert onLine == medianLineStartReference(@[0.0, 10.0], @[5.0, 15.0], @[true, true])

  block not_pulled_by_half:
    ## Up to half the picks can be noise without setting the line.
    const n = 80
    var off = newSeq[float64](n)
    var pick = newSeq[float64](n)
    var conf = newSeq[bool](n)
    for i in 0 ..< n:
      off[i] = 10.0 * float64(i + 1)
      pick[i] = 30.0 + off[i] / 4.0
      conf[i] = true
    for i in countup(0, 58, 2): pick[i] += 800.0 # 30 of 80, under half, 800 ms off
    let near = medianLineStart(off, pick, conf)
    doAssert near == medianLineStartReference(off, pick, conf)
    let keep = (near.len + 1) div 2
    doAssert keep >= 2
    doAssert near[keep - 1][0] < 1.0 # the closer half still lies on the line

  block trim_edges:
    proc noFit(use: openArray[bool]): seq[float64] = @[]
    proc shortFit(use: openArray[bool]): seq[float64] = @[0.0]
    var off = newSeq[float64](12)
    var pick = newSeq[float64](12)
    var conf = newSeq[float64](12)
    for i in 0 ..< 12:
      off[i] = float64(i + 1) * 20
      pick[i] = 10.0 + off[i] / 2.0
      conf[i] = 1.0
    let few = trimmedModel(off[0 ..< 9], pick[0 ..< 9], conf[0 ..< 9], noFit)
    doAssert not few.trusted and few.model.len == 0 and few.sigmaMs.isNaN
    doAssert few.used == 9 and few.rounds == 0
    for c in conf.mitems: c = ConfidentPick - 0.1
    let low = trimmedModel(off, pick, conf, noFit)
    doAssert not low.trusted and low.used == 0
    for c in conf.mitems: c = ConfidentPick
    let emptyFit = trimmedModel(off, pick, conf, noFit)
    doAssert not emptyFit.trusted and emptyFit.rounds == 1
    let badLen = trimmedModel(off, pick, conf, shortFit)
    doAssert not badLen.trusted and badLen.rounds == 1
    var sameX = off
    for x in sameX.mitems: x = 4.0
    let flat = trimmedModel(sameX, pick, conf, watched(sameX, pick, Watch()))
    doAssert not flat.trusted # no rising line to start from
    var nanPick = pick
    nanPick[0] = NaN
    var nanOff = off
    nanOff[1] = NaN
    let holes = trimmedModel(nanOff, nanPick, conf, watched(nanOff, nanPick, Watch()))
    doAssert holes.trusted
    doAssert not holes.kept[0] and not holes.kept[1]

  block trim_rejects_a_short_next:
    ## 12 confident picks; the fit matches the 9 on the line, not the three
    ## outliers. The next set is under `MinTrimmedPicks`, so `used` is that
    ## rejected count and `kept` is still the 10 just fitted.
    const n = 12
    var off = newSeq[float64](n)
    var pick = newSeq[float64](n)
    var conf = newSeq[float64](n)
    for i in 0 ..< n:
      off[i] = 20.0 * float64(i + 1)
      pick[i] = 10.0 + off[i] / 2.0
      conf[i] = 1.0
    pick[9] += 400.0
    pick[10] += 500.0
    pick[11] += 600.0
    proc onLine(use: openArray[bool]): seq[float64] =
      result = newSeq[float64](n)
      for i in 0 ..< n: result[i] = 10.0 + off[i] / 2.0
    let t = trimmedModel(off, pick, conf, onLine)
    var trues = 0
    for k in t.kept:
      if k: inc trues
    doAssert not t.trusted
    doAssert t.used == 9
    doAssert trues == 10
    doAssert t.model.len == 0

  block start_and_trim_match_reference:
    for n in [0, 1, 2, 9, 10, 11, 300, 4000]:
      for kind in ["base", "ties", "unconfident", "out5", "out40"]:
        let d = picks(n, 1000 + n * 10 + kind.len, kind)
        var confident = newSeq[bool](n)
        for i in 0 ..< n:
          confident[i] = not d.pick[i].isNaN and d.conf[i] >= ConfidentPick and
            not d.off[i].isNaN
        let got = medianLineStart(d.off, d.pick, confident)
        let want = medianLineStartReference(d.off, d.pick, confident)
        doAssert got == want, "start n=" & $n & " " & kind
        let wa = Watch()
        let wb = Watch()
        let a = trimmedModel(d.off, d.pick, d.conf, watched(d.off, d.pick, wa))
        let b = trimmedModelReference(d.off, d.pick, d.conf, watched(d.off, d.pick, wb))
        doAssert sameTrimmed(a, b), "trim n=" & $n & " " & kind
        doAssert wa.calls == wb.calls, "calls n=" & $n & " " & kind
        doAssert wa.last == wb.last, "last fit n=" & $n & " " & kind
        if not a.sigmaMs.isNaN:
          doAssert a.sigmaMs == b.sigmaMs
          doAssert a.model == b.model
        if kind == "unconfident":
          doAssert not a.trusted and a.used == 0
        if kind == "out40" and n >= 300:
          doAssert a.trusted
          var far = 0
          for i, k in a.kept:
            if k and d.pick[i] > 20.0 + d.off[i] / 3.0 + 100: inc far
          doAssert far == 0 # the 500 ms outliers are not kept

    for n in [60, 300]:
      let d = bend(n)
      var confident = newSeq[bool](n)
      for i in 0 ..< n: confident[i] = true
      doAssert medianLineStart(d.off, d.pick, confident) ==
        medianLineStartReference(d.off, d.pick, confident)
      let wa = Watch()
      let wb = Watch()
      let a = trimmedModel(d.off, d.pick, d.conf, watched(d.off, d.pick, wa))
      let b = trimmedModelReference(d.off, d.pick, d.conf, watched(d.off, d.pick, wb))
      doAssert sameTrimmed(a, b), "bend n=" & $n
      doAssert wa.calls == wb.calls and wa.last == wb.last
      if not a.sigmaMs.isNaN: doAssert a.sigmaMs == b.sigmaMs and a.model == b.model

  block odd_half_keeps_one_more:
    ## 21 confident picks: `(near.len + 1) div 2` is 11, `near.len div 2` is 10.
    ## The empty fit returns at once, so `used` is that start count.
    const n = 21
    var off = newSeq[float64](n)
    var pick = newSeq[float64](n)
    var conf = newSeq[float64](n)
    for i in 0 ..< n:
      off[i] = float64(i + 1) * 10
      pick[i] = 5.0 + off[i] / 4.0
      conf[i] = 1.0
    proc emptyFit(use: openArray[bool]): seq[float64] = @[]
    let t = trimmedModel(off, pick, conf, emptyFit)
    doAssert not t.trusted
    doAssert t.used == (n + 1) div 2

  block restart_keeps_the_bend:
    ## Most picks sit on the faster segment, so the start drops the first.
    ## The model is that segment's line until `MinSecondPicks` of the first
    ## are marked, then both segments. The restart must win: every pick kept,
    ## including those more than 40 units before the crossover.
    const n = 80
    let d = bend(n)
    let xc = 25.0 * float64(n) * 0.25
    proc segmentFit(use: openArray[bool]): seq[float64] =
      var onFirst = 0
      for i, u in use:
        if u and d.off[i] <= xc: inc onFirst
      result = newSeq[float64](n)
      for i, x in d.off:
        if onFirst >= MinSecondPicks:
          result[i] = if x <= xc: x / 1.5 else: xc / 1.5 + (x - xc) / 3.0
        else:
          result[i] = xc / 1.5 + (x - xc) / 3.0
    let t = trimmedModel(d.off, d.pick, d.conf, segmentFit)
    doAssert t.trusted and t.used == n
    doAssert t.sigmaMs <= RestartSigmaShare * RestartSigmaFloorMs
    var far = 0
    for i, x in d.off:
      if x < xc - 40.0:
        inc far
        doAssert t.kept[i]
    doAssert far >= MinSecondPicks

when isMainModule: main()
