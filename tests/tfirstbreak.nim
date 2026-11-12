## First-break picking (qc/firstbreak): STA/LTA, energy ratio and modified
## Coppens on noisy synthetic shots, picks within a tolerance, confidence
## against SNR, the LMO-guided and fixed search windows, the band-pass, a
## window block, no-pick cases and errors; the example SEG-D record
## (skipped when absent).

import std/[algorithm, assertions, math, os]
import processSEG
import parseSEG
import utils/[allocs, synth]

const
  dt = 2.0
  cols = 600
  vel = 2000.0
  t0 = 20.0

proc onsetMs(offset: float64): float64 = t0 + 1000 * abs(offset) / vel

proc shot(sigma: float64, seed: int64, offsets = splitSpread(24, 25, 25)): SynthGather[float32] =
  ## A shot whose first break is linear in |offset|, with a later, stronger
  ## reflection-like event, plus white noise.
  result = shotGather[float32](1000, offsets, cols, dt)
  for r in 0 ..< result.len:
    var row = newSeq[float32](cols)
    row.addOnset(dt, onsetMs(result.keys[r].offset), 30)
    row.addRicker(dt, onsetMs(result.keys[r].offset) + 250, 25, amp = 2)
    row.addNoise(sigma, seed + r)
    for j in 0 ..< cols: result.data[r, j] = row[j]

proc picks(g: SynthGather[float32], params: FirstBreakParams): seq[FirstBreakPick] =
  let meta = initChunkMeta(g.rows, dt)
  let plan = initFirstBreakPlan(float32, params, dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  result = newSeq[FirstBreakPick](g.len)
  plan.pickInto(g.data, meta, ws, result)

proc errorsMs(g: SynthGather[float32], p: openArray[FirstBreakPick]): seq[float64] =
  for r in 0 ..< g.len: result.add abs(p[r].timeMs - onsetMs(g.keys[r].offset))

proc meanOf(x: openArray[float64]): float64 =
  for v in x: result += v
  result /= float64(x.len)

let guided = FirstBreakParams(guideVelocity: vel, guideT0Ms: t0, beforeMs: 60,
  afterMs: 80)

block picks_within_a_tolerance:
  # High SNR: every method picks every trace within 2 samples, confidently.
  var g = shot(0.02, 1)
  # STA/LTA and the energy ratio pick the energy arrival, a consistent
  # fraction of a period after the onset: the bias is bounded and the
  # spread small.
  for m in PickMethod:
    var p = guided
    p.pickMethod = m
    let ps = picks(g, p)
    var lag: seq[float64]
    for r in 0 ..< g.len: lag.add ps[r].timeMs - onsetMs(g.keys[r].offset)
    let bias = meanOf(lag)
    var spread = 0.0
    for x in lag: spread = max(spread, abs(x - bias))
    let tol = if m == pmCoppens: 2 * dt else: 5 * dt
    doAssert abs(bias) + spread <= tol, $m & ": " & $lag
    doAssert spread <= 1.5 * dt, $m & ": " & $lag
    for q in ps: doAssert q.confidence > 0.9 and q.snrDb > 20, $m & ": " & $q
  # Unguided (the whole trace): Coppens' cumulative energy favours the
  # first arrival, though the later event is twice as strong (the energy
  # ratio, weighted by |x|, takes the stronger event: guide it).
  let e = errorsMs(g, picks(g, FirstBreakParams(pickMethod: pmCoppens)))
  doAssert max(e) <= 2 * dt, $max(e)
  # Noisier (SNR about 10 dB over the first break's peak), over 3 noise
  # draws: 85 % of the picks within 3 samples of the method's bias (which a
  # filter's delay adds to), with a 60 Hz minimum-phase high cut, which
  # helps every method. The energy ratio needs the longer window.
  for m in PickMethod:
    var counts: array[2, int]
    for f, hc in [60.0, Inf]:
      var bp = guided
      bp.pickMethod = m
      bp.highCutHz = hc
      bp.windowMs = if m == pmEnergyRatio: 40 else: 20
      for seed in [2'i64, 20, 200]:
        let n = shot(0.3, seed)
        let ps = picks(n, bp)
        var lag: seq[float64]
        for r in 0 ..< n.len: lag.add ps[r].timeMs - onsetMs(n.keys[r].offset)
        let med = sorted(lag)[lag.len div 2]
        doAssert abs(med) <= 8 * dt, $m & ": " & $lag
        var good = 0
        for x in lag:
          if abs(x - med) <= 3 * dt: inc good
        if hc < Inf: doAssert good >= lag.len * 85 div 100, $m & ": " & $lag
        counts[f] += good
    doAssert counts[0] > counts[1], $m & ": " & $counts

block confidence_against_snr:
  # Mean confidence falls with the SNR; noise alone is not confident.
  var last = 2.0
  for sigma in [0.02, 0.3, 1.0]:
    let c = meanOf(block:
      var cs: seq[float64]
      for q in picks(shot(sigma, 10), guided): cs.add q.confidence
      cs)
    doAssert c < last, $sigma & ": " & $c
    last = c
  var noise = shotGather[float32](1000, splitSpread(24, 25, 25), cols, dt)
  for r in 0 ..< noise.len:
    var row = newSeq[float32](cols)
    row.addNoise(1.0, 100 + r)
    for j in 0 ..< cols: noise.data[r, j] = row[j]
  for m in PickMethod:
    var p = guided
    p.pickMethod = m
    var cs: seq[float64]
    for q in picks(noise, p):
      cs.add q.confidence
      doAssert q.snrDb.isNaN or abs(q.snrDb) < 8, $q
    doAssert meanOf(cs) < 0.1, $m & ": " & $meanOf(cs)
  doAssert last < 0.5 # sigma 1: the break is buried

block guided_and_fixed_windows:
  # A burst before the first break is picked without a guide, avoided with
  # one. The burst comes a window after the trace's start (Coppens picks
  # nothing before that) and the nearest first break 100 ms later.
  var g = shot(0.02, 3, splitSpread(24, 25, 200))
  for r in 0 ..< g.len:
    var row = newSeq[float32](cols)
    row.addOnset(dt, 24, 40, amp = 3, decayMs = 5)
    for j in 0 ..< cols: g.data[r, j] += row[j]
  let free = picks(g, FirstBreakParams(pickMethod: pmCoppens))
  var early = 0
  for r in 0 ..< g.len:
    if free[r].timeMs < 32: inc early
  doAssert early > g.len div 2
  doAssert max(errorsMs(g, picks(g, guided))) <= 2 * dt
  # A fixed window that starts after the burst works too.
  doAssert max(errorsMs(g, picks(g, FirstBreakParams(pickMethod: pmCoppens,
    searchStartMs: 60)))) <= 2 * dt
  # A window after every first break picks inside it.
  for q in picks(g, FirstBreakParams(searchStartMs: 600, searchEndMs: 700)):
    doAssert q.timeMs >= 600 and q.timeMs <= 700

block no_picks:
  var g = shot(0.02, 4)
  g.keys[0].status = tsDead
  g.keys[1].offset = NaN
  for j in 0 ..< cols: g.data[2, j] = 0
  g.keys[3].t0Ms = NaN
  let ps = picks(g, guided)
  for r in 0..3:
    doAssert ps[r].timeMs.isNaN and ps[r].col.isNaN and ps[r].confidence == 0, $r
  # Unguided, a NaN offset is not needed; a NaN t0 gives a column, no time.
  let pf = picks(g, FirstBreakParams(pickMethod: pmCoppens))
  doAssert not pf[1].timeMs.isNaN and pf[2].timeMs.isNaN
  doAssert pf[3].timeMs.isNaN and abs(pf[3].col * dt - onsetMs(g.keys[3].offset)) <= 2 * dt
  # The same in a window (firstSample 10): the column is the block's.
  let w = g.window(10, 400)
  let wmeta = initChunkMeta(w.rows, dt, firstSample = w.firstSample)
  let wplan = initFirstBreakPlan(float32, FirstBreakParams(pickMethod: pmCoppens), dt, cols)
  var wws = initWorkspace()
  wws.reserve(wplan.workspaceSize)
  var wp = newSeq[FirstBreakPick](w.len)
  wplan.pickInto(w.data, wmeta, wws, wp)
  doAssert wp[3].timeMs.isNaN and wp[3].col == pf[3].col - 10
  # A search window outside the trace.
  for q in picks(shot(0.02, 5), FirstBreakParams(searchStartMs: 5000)):
    doAssert q.timeMs.isNaN

block window_block_and_single_trace:
  # A window of the gather (firstSample 10, a shorter live length) picks the
  # same times; the single-trace form agrees with the block form.
  var g = shot(0.02, 6)
  let full = picks(g, guided)
  let w = g.window(10, 400)
  let meta = initChunkMeta(w.rows, dt, firstSample = w.firstSample)
  let plan = initFirstBreakPlan(float32, guided, dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var ps = newSeq[FirstBreakPick](w.len)
  let before = w.data.clone()
  plan.pickInto(w.data, meta, ws, ps)
  doAssert w.data == before # the samples are not changed
  for r in 0 ..< g.len:
    doAssert abs(ps[r].timeMs - full[r].timeMs) <= dt
    doAssert ps[r].col == (ps[r].timeMs - 10 * dt) / dt
  var row = newSeq[float32](cols)
  for j in 0 ..< cols: row[j] = g.data[5, j]
  let one = plan.pick(row, ws, g.keys[5].offset, 0.0)
  doAssert one == full[5]
  # No allocation once the workspace is reserved.
  doAssert allocsIn(plan.pickInto(w.data, meta, ws, ps)) == 0

block errors:
  doAssertRaises(ValueError): discard initFirstBreakPlan(float32, FirstBreakParams(), 0, cols)
  doAssertRaises(ValueError): discard initFirstBreakPlan(float32, FirstBreakParams(), dt, 0)
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(windowMs: 0), dt, cols)
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(ltaMs: 10), dt, cols)
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(lowDb: 20, highDb: 20), dt, cols)
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(searchStartMs: 100,
      searchEndMs: 50), dt, cols)
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(guideVelocity: -1), dt, cols)
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(stabilization: NaN), dt, cols)
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(lowCutHz: 5, highCutHz: 300), dt,
      cols) # beyond Nyquist
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(lowCutHz: 60, highCutHz: 10), dt, cols)
  doAssertRaises(ValueError):
    discard initFirstBreakPlan(float32, FirstBreakParams(lowCutHz: 5, slopeDbPerOctave: 10),
      dt, cols)
  var g = shot(0.02, 7)
  let meta = initChunkMeta(g.rows, dt)
  let plan = initFirstBreakPlan(float32, guided, dt, cols - 1)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var ps = newSeq[FirstBreakPick](g.len)
  doAssertRaises(ValueError): plan.pickInto(g.data, meta, ws, ps) # too many columns
  let plan4 = initFirstBreakPlan(float32, guided, 4, cols)
  doAssertRaises(ValueError): plan4.pickInto(g.data, meta, ws, ps) # other interval
  var short = newSeq[FirstBreakPick](1)
  let plan2 = initFirstBreakPlan(float32, guided, dt, cols)
  doAssertRaises(ValueError): plan2.pickInto(g.data, meta, ws, short)

block float64_traces:
  # float64 samples (the trace copy and the energy arrays share the float64
  # scratch) pick as float32 ones, with a filter.
  var g = shot(0.02, 8)
  var p = guided
  p.highCutHz = 60
  let p32 = picks(g, p)
  var d = newTensor[float64](g.len, cols)
  for r in 0 ..< g.len:
    for j in 0 ..< cols: d[r, j] = float64(g.data[r, j])
  let meta = initChunkMeta(g.rows, dt)
  let plan = initFirstBreakPlan(float64, p, dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var p64 = newSeq[FirstBreakPick](g.len)
  plan.pickInto(d, meta, ws, p64)
  for r in 0 ..< g.len:
    doAssert p64[r].timeMs == p32[r].timeMs
    doAssert abs(p64[r].snrDb - p32[r].snrDb) < 1e-3

block batched_filter_matches_single_trace:
  # pickInto band-passes fftLanes rows at a time (the IIR filter's lanes);
  # `pick` filters one trace. Same picks, across lane groups, with short,
  # dead and failed rows (they shift which lane a row lands in).
  var g = shot(0.3, 7, splitSpread(30, 25, 25))
  g.setLive(3, 200)
  g.setLive(11, 1)
  g.keys[9].status = tsDead
  let failed = @[17]
  let meta = initChunkMeta(g.rows, dt, failed = failed)
  for m in PickMethod:
    var p = guided
    p.pickMethod = m
    p.lowCutHz = 8
    p.highCutHz = 70
    let plan = initFirstBreakPlan(float32, p, dt, cols)
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var got = newSeq[FirstBreakPick](g.len)
    plan.pickInto(g.data, meta, ws, got)
    for r in 0 ..< g.len:
      if not meta.isLive(r):
        doAssert got[r].col.isNaN
        continue
      var x = newSeq[float32](g.liveCols(r))
      for j in 0 ..< x.len: x[j] = g.data[r, j]
      let want = plan.pick(x, ws, g.keys[r].offset, g.keys[r].t0Ms)
      doAssert got[r].col == want.col or (got[r].col.isNaN and want.col.isNaN),
        $m & " row " & $r & ": " & $got[r] & " vs " & $want
      doAssert abs(got[r].confidence - want.confidence) < 1e-4

block filtered_batch_allocates_nothing_and_infinite_t0:
  # The batched band-pass of pickInto works in the reserved workspace.
  var g = shot(0.1, 3)
  let meta = initChunkMeta(g.rows, dt)
  var p = guided
  p.lowCutHz = 8
  p.highCutHz = 70
  let plan = initFirstBreakPlan(float32, p, dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var picks = newSeq[FirstBreakPick](g.len)
  plan.pickInto(g.data, meta, ws, picks) # warm
  doAssert allocsIn(plan.pickInto(g.data, meta, ws, picks)) == 0
  # An infinite t0 is no time axis (as NaN): no time window can apply, so a
  # guided pick is none, and an open window gives a column without a time.
  var x = newSeq[float32](cols)
  for j in 0 ..< cols: x[j] = g.data[4, j]
  for t0 in [Inf, -Inf]:
    doAssert plan.pick(x, ws, g.keys[4].offset, t0).col.isNaN
    let open = initFirstBreakPlan(float32, FirstBreakParams(), dt, cols)
    ws.reserve(open.workspaceSize)
    let q = open.pick(x, ws, NaN, t0)
    doAssert not q.col.isNaN and q.timeMs.isNaN, $q

const shotFile = currentSourcePath().parentDir.parentDir / "exampleFiles" /
  "00000151_12900_11300.sgd"

block coppens_does_not_pick_its_own_warm_up:
  # Unguided, in noise a tenth of the onset: over the first window the
  # ratio's numerator is cut by the trace's start and rises from 0 with it,
  # which is no arrival.
  const rows = 200
  let w = int(FirstBreakParams().windowMs / dt)
  var g = shotGather[float32](0, endOn(rows, 25, 25), cols, dt)
  for r in 0 ..< rows:
    var row = newSeq[float32](cols)
    row.addOnset(dt, 300 + 0.37 * float64(r), 30)
    row.addNoise(0.1, 700 + r)
    for j in 0 ..< cols: g.data[r, j] = row[j]
  var good = 0
  for r, q in picks(g, FirstBreakParams(pickMethod: pmCoppens)):
    doAssert q.col >= float64(w), "row " & $r & ": " & $q
    if abs(q.timeMs - (300 + 0.37 * float64(r))) <= 3 * dt: inc good
  doAssert good >= rows * 95 div 100, $good

block coppens_on_the_example_record:
  if not fileExists(shotFile):
    echo "  skipped: the example SEG-D record is not present"
  else:
    var f = openSeismic(shotFile, keys = true)
    defer: f.close()
    let data = f.readTraces(0 ..< f.len, float32)
    let meta = chunkMeta(f, 0 ..< f.len)
    let plan = initFirstBreakPlan(float32, FirstBreakParams(pickMethod: pmCoppens),
      meta.dtMs, data.shape[1])
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var ps = newSeq[FirstBreakPick](meta.len)
    plan.pickInto(data, meta, ws, ps)
    let w = int(FirstBreakParams().windowMs / meta.dtMs)
    var early, picked = 0
    for q in ps:
      if not q.col.isNaN:
        inc picked
        if q.col < float64(w): inc early
    echo "  example record: ", picked, " Coppens picks, ", early, " in the first window (",
      w, " samples)"
    doAssert picked > 3900 and early == 0 # 3210 of 3954 picked the warm-up
