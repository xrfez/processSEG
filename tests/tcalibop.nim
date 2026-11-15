## The calibration in a pipeline: `calibrateOp` and
## `uncalibrateOp`; gain, statistics, a per-panel normalization and AGC after
## it see physical values; the order rule; `amplitudeUnits`; editing the op;
## the pool with a file-scale column; no allocation per apply.

import std/[assertions, math, strutils]
import parseSEG, processSEG
import utils/allocs

const
  rows = 24
  cols = 300
  dt = 2.0

func closeTo(a, b: float64, rel = 1e-6): bool = abs(a - b) <= rel * max(abs(a), abs(b))

var keys = newSeq[TraceKeys](rows)
var counts = newSeq[int](rows)
var scales = newSeq[FileScale](rows)
for r in 0 ..< rows:
  keys[r] = initTraceKeys()
  keys[r].status = tsLive
  keys[r].sensorType = (if r mod 4 == 0: 1'u8 else: 2'u8) # a hydrophone every fourth row
  keys[r].t0Ms = 0
  counts[r] = cols
  # The file gives a descale that differs per row.
  scales[r] = FileScale(descale: 1e-6 * float64(1 + r mod 5), sensitivity: NaN,
    unit: auRecorded)
keys[5].status = tsAuxiliary
keys[9].status = tsDead
keys[13].sensorType = 7 # an accelerometer without a rule: NaN
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = dt, scales = scales)

let cal = initCalibration([
  CalibrationRule(match: RuleMatch(sensorType: 1), sensitivity: sensitivity(20, suVPerBar)),
  CalibrationRule(match: RuleMatch(sensorType: 2), sensitivity: sensitivity(22.5, suVPerMPerS))])

proc rawPanel(): Tensor[float32] =
  result = newTensor[float32](rows, cols)
  for r in 0 ..< rows:
    for j in 0 ..< cols:
      result[r, j] = float32(1000 * sin(0.05 * float64(j * (r + 1))) + 300)

block op_equals_kernel_then_ops:
  # calibrate → demean → per-trace stats, against the kernel and the ops by hand.
  let raw = rawPanel()
  var flow = initPipeline([calibrateOp(cal), demeanOp()])
  let plan = flow.compile(float32, dt, cols)
  doAssert plan.amplitudeUnits == paCalibrated
  var ws = initWorkspace()
  ws.reserve(plan)
  var processed = newTensor[float32](rows, cols)
  plan.apply(raw, processed, meta, ws)
  var hand = raw.clone
  let cp = initCalibrationPlan(cal)
  let s = cp.calibrate(hand, meta)
  doAssert s.unresolved == 1 and s.units[auVolt] == 1 # the accelerometer; the aux row
  hand.demean(meta)
  for r in 0 ..< rows:
    for j in 0 ..< cols:
      let a = processed[r, j]
      let b = hand[r, j]
      doAssert (a.isNaN and b.isNaN) or a == b, $r & " " & $j
  doAssert processed[13, 0].isNaN
  # The auxiliary row is in volts, from the file's descale (row 5: 1e-6 V/count),
  # and not demeaned (not live).
  doAssert closeTo(processed[5, 7], raw[5, 7] * 1e-6, 1e-6)
  # Statistics on the output are physical: m/s around 1e-4.
  var stats = newSeq[TraceStats](rows)
  traceStatsInto(processed, meta, stats)
  doAssert stats[1].rms > 1e-5 and stats[1].rms < 1e-3
  doAssert stats[13].rms.isNaN # the NaN row

block panel_normalization_sees_physical_values:
  # Every row records the same counts, but the factors differ per row: after
  # calibration the largest |x| of the panel is in the row with the largest
  # factor, and a per-panel normalization to max 1 is set by it.
  var raw = newTensor[float32](rows, cols)
  for r in 0 ..< rows:
    for j in 0 ..< cols: raw[r, j] = float32(1000 * sin(0.1 * float64(j)))
  var flow = initPipeline([calibrateOp(cal), normalizeOp(normMax, perPanel)])
  let plan = flow.compile(float32, dt, cols)
  doAssert plan.amplitudeUnits == paRelative
  var ws = initWorkspace()
  ws.reserve(plan)
  var processed = newTensor[float32](rows, cols)
  plan.apply(raw, processed, meta, ws)
  var rc = newSeq[RowCalibration](rows)
  discard initCalibrationPlan(cal).resolveInto(meta, rc)
  var best = 0.0
  var bestRow = -1
  for r in 0 ..< rows:
    if meta.isLive(r) and rc[r].status == csConverted and abs(rc[r].factor) > best:
      (best, bestRow) = (abs(rc[r].factor), r)
  var peak = 0.0
  for j in 0 ..< cols: peak = max(peak, abs(float64(processed[bestRow, j])))
  doAssert closeTo(peak, 1, 1e-5), $peak
  # Another row is smaller by its factor's ratio.
  var other = 0.0
  for j in 0 ..< cols: other = max(other, abs(float64(processed[2, j])))
  doAssert closeTo(other, rc[2].factor / best, 1e-5)

block agc_after_calibration:
  let raw = rawPanel()
  var flow = initPipeline([calibrateOp(cal), agcOp(windowMs = 100)])
  let plan = flow.compile(float64, dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan)
  var a = newTensor[float64](rows, cols)
  for r in 0 ..< rows:
    for j in 0 ..< cols: a[r, j] = float64(raw[r, j])
  var b = a.clone
  plan.apply(a, meta, ws)
  discard initCalibrationPlan(cal).calibrate(b, meta)
  let agc = initAgcPlan(100, dt, cols)
  ws.reserve(agc.workspaceSize)
  agc.agc(b, meta, ws)
  for r in 0 ..< rows:
    for j in 0 ..< cols:
      doAssert (a[r, j].isNaN and b[r, j].isNaN) or a[r, j] == b[r, j]

block order_rule:
  for bad in [@[gainDbOp(6), calibrateOp(cal)], @[scaleOp(2), calibrateOp(cal)],
      @[clipOp(-1, 1), calibrateOp(cal)], @[agcOp(100), calibrateOp(cal)],
      @[normalizeOp(normRms), calibrateOp(cal)], @[gainCurveOp(powerGain(2)), calibrateOp(cal)],
      @[calibrateOp(cal), agcOp(100), uncalibrateOp(cal)]]:
    try:
      discard initPipeline(bad).compile(float32, dt, cols)
      doAssert false, "compiled: " & $bad.len
    except ValueError as e:
      doAssert "must come before every op that changes amplitudes" in e.msg, e.msg
      doAssert e.msg.startsWith("op " & $(bad.len - 1)), e.msg
  try:
    discard initPipeline([calibrateOp(cal), demeanOp(), calibrateOp(cal)]).compile(float32, dt, cols)
    doAssert false
  except ValueError as e:
    doAssert e.msg.startsWith("op 2 (opCalibrate): one per pipeline: op 0"), e.msg
  # A bad rule is reported with the op.
  let dup = initCalibration([CalibrationRule(descale: descale(1, duVPerCount)),
    CalibrationRule(sensitivity: sensitivity(1, suVPerPa))])
  try:
    discard initPipeline([calibrateOp(dup)]).compile(float32, dt, cols)
    doAssert false
  except ValueError as e:
    doAssert e.msg.startsWith("op 0 (opCalibrate): rule 1:"), e.msg
  # Allowed: linear, scale-free ops before it; an uncalibration after it.
  for good in [@[demeanOp(), detrendOp(), polarityOp(), calibrateOp(cal)],
      @[bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)), calibrateOp(cal), uncalibrateOp(cal)],
      @[muteTopOp(timeMute(100)), calibrateOp(cal)]]:
    discard initPipeline(good).compile(float32, dt, cols)

block amplitude_units:
  template units(ops: openArray[Op]): PlanAmplitude =
    initPipeline(ops).compile(float32, dt, cols).amplitudeUnits
  doAssert units([]) == paRecorded
  doAssert units([demeanOp(), agcOp(100)]) == paRecorded
  doAssert units([calibrateOp(cal)]) == paCalibrated
  doAssert units([calibrateOp(cal), detrendOp(), bandPassOp(Corners(f1: 4, f2: 8, f3: 60,
    f4: 80))]) == paCalibrated
  doAssert units([calibrateOp(cal), gainDbOp(3)]) == paRelative
  doAssert units([calibrateOp(cal), demeanOp(), uncalibrateOp(cal)]) == paRecorded
  doAssert units([uncalibrateOp(cal), calibrateOp(cal)]) == paCalibrated

block edit_in_place:
  var flow = initPipeline([calibrateOp(cal)])
  doAssert flow[0].kind == opCalibrate and flow[0].calibration.rules.len == 2
  flow[0].calibration.rules[1].sensitivity = sensitivity(45, suVPerMPerS) # twice the 22.5
  flow[0].calibration.target = ctPhysical
  let a = initPipeline([calibrateOp(cal)]).compile(float64, dt, cols)
  let b = flow.compile(float64, dt, cols)
  doAssert b.generation > a.generation
  var ws = initWorkspace()
  var x = newTensor[float64](rows, cols)
  var y = newTensor[float64](rows, cols)
  for r in 0 ..< rows:
    for j in 0 ..< cols:
      x[r, j] = 100
      y[r, j] = 100
  a.apply(x, meta, ws)
  b.apply(y, meta, ws)
  doAssert closeTo(x[1, 0], 2 * y[1, 0], 1e-12) and x[0, 0] == y[0, 0] # the hydrophone unchanged
  doAssertRaises(ValueError): discard flow[0].agc
  doAssertRaises(ValueError): discard demeanOp().calibration
  let u = uncalibrateOp(cal)
  doAssert u.kind == opUncalibrate and u.calibration.rules.len == 2

block uncalibrate_in_a_flow:
  # calibrate → band-pass → uncalibrate: counts again, filtered.
  let raw = rawPanel()
  let band = bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80))
  let plan = initPipeline([calibrateOp(cal), band, uncalibrateOp(cal)]).compile(float64, dt, cols)
  let only = initPipeline([band]).compile(float64, dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan)
  var a = newTensor[float64](rows, cols)
  for r in 0 ..< rows:
    for j in 0 ..< cols: a[r, j] = float64(raw[r, j])
  var b = a.clone
  plan.apply(a, meta, ws)
  only.apply(b, meta, ws)
  for r in [1, 2, 3, 4]:
    for j in 0 ..< cols: doAssert abs(a[r, j] - b[r, j]) <= 1e-9 * 1000

block pool_with_file_scales:
  # The pool splits the rows; each worker's meta must keep its rows' scales.
  # Rows 8000 samples wide (live for 300) make runs of 8 rows (256 KB), so
  # the workers get rows 8 .. 15 and 16 .. 23, whose file descales differ.
  const wide = 8000
  var raw = newTensor[float32](rows, wide)
  let narrow = rawPanel()
  for r in 0 ..< rows:
    for j in 0 ..< cols: raw[r, j] = narrow[r, j]
  let plan = initPipeline([calibrateOp(cal), demeanOp()]).compile(float32, dt, wide)
  var ws = initWorkspace()
  ws.reserve(plan)
  var one = newTensor[float32](rows, wide)
  plan.apply(raw, one, meta, ws)
  var pool = newProcessPool(workers = 4)
  var many = newTensor[float32](rows, wide)
  pool.apply(plan, raw, many, meta)
  # The reference: the kernels on the whole block, with no sub-metas.
  var hand = raw.clone
  discard initCalibrationPlan(cal).calibrate(hand, meta)
  hand.demean(meta)
  for r in 0 ..< rows:
    for j in 0 ..< wide:
      doAssert (one[r, j].isNaN and many[r, j].isNaN) or one[r, j] == many[r, j], $r & " " & $j
      doAssert (one[r, j].isNaN and hand[r, j].isNaN) or one[r, j] == hand[r, j], $r & " " & $j

block no_allocation_per_apply:
  let raw = rawPanel()
  let plan = initPipeline([calibrateOp(cal), demeanOp(),
    bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80))]).compile(float32, dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan)
  var dst = newTensor[float32](rows, cols)
  plan.apply(raw, dst, meta, ws)
  let n = allocsIn:
    for i in 0 ..< 3: plan.apply(raw, dst, meta, ws)
  doAssert n == 0, $n & " allocations"
