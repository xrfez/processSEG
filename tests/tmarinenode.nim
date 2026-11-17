## The case calibration was built for: marine nodes whose
## recorder writes counts, each node a hydrophone and three geophones, the
## sensitivities applied later from a table by component. End to end on a
## synthetic SEG-D record: what the file says, prefetch, the file's descale
## column, a pipeline with `calibrateOp`, the pool; hydrophones in Pa,
## geophones in m/s, to the last bit of the factor. And a caller who does not
## calibrate sees no change.

import std/[assertions, math]
import parseSEG, processSEG
import utils/[allocs, calibfiles]

proc main*() =
  func closeTo(a, b: float64, rel = 1e-12): bool = abs(a - b) <= rel * max(abs(a), abs(b))

  const units = 40 # a chunk per component, split across the pool's workers

  let
    # The table a user enters from the sensor spec sheets, by component.
    hydrophone = sensitivity(-192, suDbReVPerMicroPa) # ≈ 25 V/bar
    vertical = sensitivity(28.8, suVPerMPerS)
    inline = sensitivity(28.5, suVPerMPerS)
    crossline = sensitivity(0.72, suVPerInPerS)       # another unit, the same quantity
    table = initCalibration([
      CalibrationRule(match: RuleMatch(component: 1), sensitivity: hydrophone),
      CalibrationRule(match: RuleMatch(component: 2), sensitivity: vertical),
      CalibrationRule(match: RuleMatch(component: 3), sensitivity: inline),
      CalibrationRule(match: RuleMatch(component: 4), sensitivity: crossline)])
    bySensor = [hydrophone, vertical, inline, crossline]

  writeMarineNodes(units = units)
  var f = openSeismic(marinePath(), keys = true)
  doAssert f.len == 4 * units

  block what_the_file_says:
    # Four groups, one per component's channel set: a descale, no sensitivity.
    let rep = f.scaleReport()
    doAssert rep.len == 4
    for c in 0 ..< 4:
      doAssert rep[c].count == units and rep[c].channelSet == c
      doAssert rep[c].scale.sensitivity.isNaN
      doAssert closeTo(rep[c].scale.descale, float64(float32(marineDsm[c])) * 1e-3)
    for t in 0 ..< f.len:
      let k = f.traceKeys(t)
      doAssert int(k.component) == 1 + t div units and int(k.sensorType) == 1 + t div units

  block end_to_end:
    let plan = initPipeline([calibrateOp(table)]).compile(float64, dtMs = 1, samples = 100)
    doAssert plan.amplitudeUnits == paCalibrated
    var ws = initWorkspace()
    ws.reserve(plan)
    var pool = newProcessPool(workers = 3)
    let cal = initCalibrationPlan(table)
    var p = newPrefetcher[float64](f, 0 ..< f.len, chunkTraces = 64, uniform = true)
    var c: PrefetchChunk[float64]
    var scales: seq[FileScale]
    var rows: seq[RowCalibration]
    var seen, chunks = 0
    while p.next(c):
      inc chunks
      scales.setLen(c.rows.len)
      f.fileScalesInto(c.rows, scales)
      let meta = chunkMeta(c, scales = scales)
      rows.setLen(meta.len)
      let s = cal.resolveInto(meta, rows)
      doAssert s.converted == meta.len and s.unresolved == 0 and s.conflicts == 0 and
        s.mismatches == 0, $s
      var processed = newTensor[float64](meta.len, c.samples.shape[1])
      var pooled = newTensor[float64](meta.len, c.samples.shape[1])
      plan.apply(c.samples, processed, meta, ws)
      pool.apply(plan, c.samples, pooled, meta)
      for r in 0 ..< meta.len:
        let t = c.rows.traceOf(r)
        let comp = t div units
        let d = float64(float32(marineDsm[comp])) * 1e-3             # V per count
        let want = rowValue(t) * d / bySensor[comp].voltsPerUnit
        doAssert rows[r].unit == (if comp == 0: auPressure else: auVelocity)
        doAssert rows[r].descaleFrom == fsFile and rows[r].sensitivityFrom == fsRule
        doAssert rows[r].sensitivityRule == comp
        for j in 0 ..< 100:
          doAssert closeTo(processed[r, j], want), $t & ": " & $processed[r, j] & " " & $want
          doAssert pooled[r, j] == processed[r, j]
        inc seen
    p.close()
    doAssert seen == f.len and chunks == 4 # uniform chunks: one per channel set

  block a_hydrophone_given_a_geophone_figure:
    # A slip in the table (the vertical geophone's figure on component 1) is
    # converted as entered, and flagged: the sensor type says pressure.
    var wrong = table
    wrong.rules[0].sensitivity = vertical
    let meta = chunkMeta(f, 0 ..< f.len)
    var rows = newSeq[RowCalibration](f.len)
    let s = initCalibrationPlan(wrong).resolveInto(meta, rows)
    doAssert s.mismatches == units and rows[0].mismatch and not rows[units].mismatch

  block no_change_without_calibration:
    # A caller who does not calibrate: the same flow gives the same bits with
    # or without a file-scale column in the meta, the plan has no calibration
    # step, its output stays "recorded", and applying it allocates nothing.
    let flow = initPipeline([demeanOp(), bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
      agcOp(windowMs = 50)])
    let plan = flow.compile(float32, dtMs = 1, samples = 100)
    doAssert plan.len == 3 and plan.amplitudeUnits == paRecorded
    var raw = f.readTraces(0 ..< f.len, float32)
    for r in 0 ..< f.len:
      for j in 0 ..< 100: raw[r, j] = raw[r, j] + float32(50 * sin(0.3 * float64(j * (r + 1))))
    var scales = newSeq[FileScale](f.len)
    f.fileScalesInto(0 ..< f.len, scales)
    let plain = chunkMeta(f, 0 ..< f.len)
    let withScales = chunkMeta(f, 0 ..< f.len, scales = scales)
    var ws = initWorkspace()
    ws.reserve(plan)
    var a = newTensor[float32](f.len, 100)
    var b = newTensor[float32](f.len, 100)
    plan.apply(raw, a, plain, ws)
    plan.apply(raw, b, withScales, ws)
    for r in 0 ..< f.len:
      for j in 0 ..< 100: doAssert a[r, j] == b[r, j]
    let n = allocsIn:
      plan.apply(raw, a, plain, ws)
      plan.apply(raw, b, withScales, ws)
    doAssert n == 0

  f.close()

when isMainModule: main()
