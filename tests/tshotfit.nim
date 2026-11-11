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

proc makeShot(srcX: float64, sigma = 0.02, seed = 1'i64): Shot =
  let rel = splitSpread(perSide, group, group)
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
