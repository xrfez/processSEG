## Robust geometry fit per shot (qc/shotfit): a split spread over a
## refractor whose delay time varies along the line; receivers with
## swapped headers are detected by the residuals and the cross-correlation,
## without flagging the good traces; a misplaced source shows as an
## asymmetry; the two-segment fit recovers the velocities.

import std/[assertions, math]
import processSEG
import parseSEG
import utils/synth

const
  dt = 2.0
  cols = 750
  v0 = 1000.0 # direct wave
  v1 = 3000.0 # refractor
  perSide = 48
  group = 25.0

proc delayMs(x: float64): float64 =
  ## The refractor's intercept time under receiver x: it varies along the line.
  80 + 15 * sin(2 * PI * x / 1600)

proc breakMs(src, rcv: float64): float64 =
  let d = abs(rcv - src)
  min(1000 * d / v0, 1000 * d / v1 + delayMs(rcv))

type Shot = object
  g: SynthGather[float32]
  truth: seq[float64] # true first-break times

proc makeShot(srcX: float64, sigma = 0.02, seed = 1'i64,
    offsets = splitSpread(perSide, group, group)): Shot =
  let rel = offsets
  result.g = shotGather[float32](srcX, rel, cols, dt)
  for r in 0 ..< result.g.len:
    let t = breakMs(srcX, result.g.keys[r].rcvX)
    result.truth.add t
    var row = newSeq[float32](cols)
    row.addOnset(dt, t, 30)
    row.addNoise(sigma, seed + r)
    for j in 0 ..< cols: result.g.data[r, j] = row[j]

proc swapHeaders(g: var SynthGather[float32], a, b: int) =
  ## The recorded traces stay; their geometry headers trade places.
  let ka = g.keys[a]
  g.keys[a] = g.keys[b]
  g.keys[b] = ka
  swap(g.keys[a].channel, g.keys[b].channel)

proc pickAll(g: SynthGather[float32]): seq[FirstBreakPick] =
  let meta = initChunkMeta(g.rows, dt)
  let plan = initFirstBreakPlan(float32, FirstBreakParams(pickMethod: pmCoppens), dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  result = newSeq[FirstBreakPick](g.len)
  plan.pickInto(g.data, meta, ws, result)

proc suspects(f: ShotFit): seq[int] =
  for r, s in f.suspect:
    if s: result.add r

block two_segment_fit_on_exact_picks:
  # Exact picks of a flat refractor: the fit recovers v0, v1 and the intercept.
  var s = makeShot(5000)
  var picks = newSeq[FirstBreakPick](s.g.len)
  for r in 0 ..< s.g.len:
    let d = abs(s.g.keys[r].offset)
    picks[r] = FirstBreakPick(timeMs: min(1000 * d / v0, 1000 * d / v1 + 80), col: 0,
      snrDb: 30, confidence: 1)
  let meta = initChunkMeta(s.g.rows, dt)
  var fit: ShotFit
  fitShot(meta, picks, ShotFitParams(), fit)
  doAssert fit.used == s.g.len
  doAssert abs(fit.v1 - v0) < 1 and abs(fit.v2 - v1) < 1, $fit.v1 & " " & $fit.v2
  doAssert abs(fit.interceptMs) < 0.01
  doAssert abs(fit.crossover - 120) <= group, $fit.crossover
  for r in 0 ..< s.g.len: doAssert abs(fit.residualMs[r]) < 0.5
  doAssert suspects(fit).len == 0 and abs(fit.asymmetryMs) < 0.01

block swapped_receivers_detected:
  # Two pairs of receivers with swapped headers, over the varying refractor.
  for model in RefractionModel:
    # Bins narrower than the bend at the 120 m crossover.
    let params = ShotFitParams(model: model, binWidth: 50)
    var s = makeShot(5000)
    let picks = pickAll(s.g) # the picks follow the recorded traces
    for r in 0 ..< s.g.len:
      doAssert abs(picks[r].timeMs - s.truth[r]) <= 3 * dt, $r & ": " & $picks[r] & " " & $s.truth[r]
    s.g.swapHeaders(20, 27)
    s.g.swapHeaders(60, 64)
    let meta = initChunkMeta(s.g.rows, dt)
    var fit: ShotFit
    # Residuals only.
    fitShot(meta, picks, params, fit)
    doAssert suspects(fit) == @[20, 27, 60, 64], $model & ": " & $suspects(fit)
    # With cross-correlation too.
    fitShot(s.g.data, meta, picks, params, fit)
    doAssert suspects(fit) == @[20, 27, 60, 64], $model & ": " & $suspects(fit)
    for r in [20, 27, 60, 64]: doAssert abs(fit.lagMs[r]) > 8, $fit.lagMs[r]
    for r in 0 ..< s.g.len:
      if r notin [20, 27, 60, 64]: doAssert abs(fit.lagMs[r]) <= 4, $r & ": " & $fit.lagMs[r]
    doAssert fit.coherence[30] > 0.9
    # The unswapped shot: nothing flagged, whatever the refractor does.
    var clean = makeShot(5000)
    let cmeta = initChunkMeta(clean.g.rows, dt)
    fitShot(clean.g.data, cmeta, pickAll(clean.g), params, fit)
    doAssert suspects(fit).len == 0, $model & ": " & $suspects(fit)

block correlation_without_picks:
  # Low-confidence picks are not used, but the lags still flag a swap (the
  # model comes from the confident picks of the other traces).
  var s = makeShot(5000)
  var picks = pickAll(s.g)
  s.g.swapHeaders(30, 36)
  picks[30].confidence = 0.1
  picks[36].confidence = 0.1
  let meta = initChunkMeta(s.g.rows, dt)
  var fit: ShotFit
  fitShot(s.g.data, meta, picks, ShotFitParams(), fit)
  doAssert fit.used == s.g.len - 2
  doAssert fit.residualMs[30].isNaN and fit.localMs[30].isNaN
  doAssert suspects(fit) == @[30, 36], $suspects(fit)

block the_sign_of_a_lag:
  # A trace whose data come 10 ms late against its neighbours' (its header
  # as it was) has a negative lag; one that comes early, a positive one.
  for shift in [5, -5]: # samples
    var s = makeShot(5000)
    let picks = pickAll(s.g)
    var row = newSeq[float32](cols)
    for j in 0 ..< cols:
      if j - shift in 0 ..< cols: row[j] = s.g.data[60, j - shift]
    for j in 0 ..< cols: s.g.data[60, j] = row[j]
    let meta = initChunkMeta(s.g.rows, dt)
    var fit: ShotFit
    fitShot(s.g.data, meta, picks, ShotFitParams(), fit)
    doAssert abs(fit.lagMs[60] + float64(shift) * dt) <= dt, $shift & ": " & $fit.lagMs[60]
    doAssert abs(fit.lagMs[40]) <= dt

block noisy_picks:
  # Noisier traces and a few wild picks: the wild ones are flagged by the
  # residuals (a bad pick looks like bad geometry, which is why the lags
  # are a second check), the swaps are found, nothing else.
  var s = makeShot(5000, sigma = 0.1, seed = 7)
  var picks = pickAll(s.g)
  picks[10].timeMs += 60 # confident but wrong
  picks[80].timeMs -= 40
  picks[10].confidence = 0.9
  picks[80].confidence = 0.9
  s.g.swapHeaders(20, 27)
  let meta = initChunkMeta(s.g.rows, dt)
  var fit: ShotFit
  fitShot(meta, picks, ShotFitParams(), fit)
  doAssert suspects(fit) == @[10, 20, 27, 80], $suspects(fit)
  # The lags do not depend on the picks: with them, the wild picks' traces
  # are still suspect from their residuals, the swaps from both.
  fitShot(s.g.data, meta, picks, ShotFitParams(), fit)
  doAssert suspects(fit) == @[10, 20, 27, 80], $suspects(fit)
  doAssert abs(fit.lagMs[10]) <= 4 and abs(fit.lagMs[80]) <= 4

block misplaced_source:
  # The source header 60 m along the line from where the shot was fired:
  # one side early, the other late, by about 60 / v each.
  var s = makeShot(5000)
  let picks = pickAll(s.g)
  for k in s.g.keys.mitems:
    k.srcX += 60
    k.offset = k.rcvX - k.srcX
  let meta = initChunkMeta(s.g.rows, dt)
  var fit: ShotFit
  fitShot(meta, picks, ShotFitParams(), fit)
  # About 2·60/v1 = 40 ms (more from the near traces on the direct wave).
  doAssert abs(fit.asymmetryMs - (-2 * 1000 * 60 / v1)) < 12, $fit.asymmetryMs
  var ok: ShotFit
  var c = makeShot(5000)
  let cm = initChunkMeta(c.g.rows, dt)
  fitShot(cm, pickAll(c.g), ShotFitParams(), ok)
  # The refractor's delay differs under the two sides: a few ms.
  doAssert abs(ok.asymmetryMs) < 8, $ok.asymmetryMs

block positions_without_coordinates:
  # Without coordinates the spread is ordered by the signed offset.
  var s = makeShot(5000)
  let picks = pickAll(s.g)
  s.g.swapHeaders(20, 27)
  for k in s.g.keys.mitems:
    k.srcX = NaN
    k.rcvX = NaN
  let meta = initChunkMeta(s.g.rows, dt)
  var fit: ShotFit
  fitShot(meta, picks, ShotFitParams(), fit)
  doAssert fit.position[0] == s.g.keys[0].offset
  doAssert suspects(fit) == @[20, 27], $suspects(fit)

block unsigned_offsets_without_coordinates:
  # Unsigned offsets would interleave the two sides: the channel order is
  # used (review 8.4 #1), and the swap is still the only finding.
  var s = makeShot(5000)
  let picks = pickAll(s.g)
  s.g.swapHeaders(20, 27)
  s.g.unsignOffsets
  for k in s.g.keys.mitems:
    k.srcX = NaN
    k.rcvX = NaN
  let meta = initChunkMeta(s.g.rows, dt)
  var fit: ShotFit
  fitShot(s.g.data, meta, picks, ShotFitParams(), fit)
  doAssert fit.position[5] == s.g.keys[5].channel
  doAssert suspects(fit) == @[20, 27], $suspects(fit)
  # No channels either: the row order.
  for k in s.g.keys.mitems: k.channel = NaN
  fitShot(s.g.data, meta, picks, ShotFitParams(), fit)
  doAssert fit.position[5] == 5 and suspects(fit) == @[20, 27], $suspects(fit)

block degenerate_fits:
  # Every pick at one offset: no velocity can be fitted (review 8.4 #6).
  var keys = newSeq[TraceKeys](4)
  for i, k in keys.mpairs:
    k = stationKeys(0, 100, 0, 0, dt)
    k.channel = float64(i + 1)
  var counts = @[10, 10, 10, 10]
  let meta = initChunkMeta(rowsOf(keys, counts), dt)
  var picks = newSeq[FirstBreakPick](4)
  for p in picks.mitems: p = FirstBreakPick(timeMs: 100, col: 50, snrDb: 30, confidence: 1)
  var fit: ShotFit
  fitShot(meta, picks, ShotFitParams(), fit)
  doAssert fit.v1.isNaN and fit.modelMs[0].isNaN and suspects(fit).len == 0
  # Picks that one line fits to rounding are one segment. A second one that
  # lowers the misfit from 1e-13 to 1e-14 ms describes nothing (flat picks
  # came back as two segments of 2e19 and 8e19 units/s meeting at 425).
  var line = makeShot(5000, offsets = endOn(24, 25, 25))
  let lmeta = initChunkMeta(line.g.rows, dt)
  var lp = newSeq[FirstBreakPick](line.g.len)
  for slowness in [0.0, 0.5]: # flat; 2000 units/s
    for r in 0 ..< line.g.len:
      lp[r] = FirstBreakPick(timeMs: 100 + slowness * abs(line.g.keys[r].offset), col: 0,
        snrDb: 30, confidence: 1)
    fitShot(lmeta, lp, ShotFitParams(), fit)
    doAssert fit.used == 24 and fit.crossover == Inf and fit.v2.isNaN, $slowness & ": " &
      $fit.crossover & " " & $fit.v2
    doAssert abs(fit.interceptMs - 100) < 1e-6 and suspects(fit).len == 0
    for r in 0 ..< line.g.len: doAssert abs(fit.residualMs[r]) < 1e-6
    if slowness > 0: doAssert abs(fit.v1 - 2000) < 1e-3, $fit.v1
    else: doAssert fit.v1.isNaN or fit.v1 > 1e9, $fit.v1 # no slope: no velocity
  # A long spread (2000 traces) fits quickly: crossover candidates are capped.
  var big = makeShot(50_000, offsets = splitSpread(1000, 5, 5))
  var bp = newSeq[FirstBreakPick](big.g.len)
  for r in 0 ..< big.g.len:
    bp[r] = FirstBreakPick(timeMs: big.truth[r], col: 0, snrDb: 30, confidence: 1)
  let bmeta = initChunkMeta(big.g.rows, dt)
  fitShot(bmeta, bp, ShotFitParams(), fit)
  doAssert abs(fit.v2 - v1) < 100 and suspects(fit).len == 0, $fit.v2 & " " & $suspects(fit)

block too_few_picks_and_errors:
  var s = makeShot(5000)
  var picks = newSeq[FirstBreakPick](s.g.len)
  for p in picks.mitems: p = noPick()
  let meta = initChunkMeta(s.g.rows, dt)
  var fit: ShotFit
  fitShot(s.g.data, meta, picks, ShotFitParams(), fit)
  doAssert fit.used == 0 and fit.modelMs[0].isNaN and suspects(fit).len == 0
  doAssertRaises(ValueError): fitShot(meta, picks[0 .. 5], ShotFitParams(), fit)
  doAssertRaises(ValueError): fitShot(meta, picks, ShotFitParams(minConfidence: 2), fit)
  doAssertRaises(ValueError): fitShot(meta, picks, ShotFitParams(localTraces: 0), fit)
  doAssertRaises(ValueError): fitShot(meta, picks, ShotFitParams(binWidth: 0), fit)
  doAssertRaises(ValueError): fitShot(meta, picks, ShotFitParams(windowMs: 0), fit)
