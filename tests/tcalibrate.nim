## Calibration (task 14.2): the precedence ladder of the rules, independently
## per factor; the file's factors in each `fileFactors` order; conflicts and
## sensor-type mismatches; NaN for unresolved live rows; dead, auxiliary and
## failed rows; the gain; volts; the inverse; already-physical rows; refusals;
## the per-row kernels; the comparison of two tables; no allocation.

import std/[assertions, math, options]
import parseSEG, processSEG
import utils/[allocs, calibfiles]

func closeTo(a, b: float64, rel = 1e-12): bool = abs(a - b) <= rel * max(abs(a), abs(b))

const cols = 16

proc block64(rows: int, value = 1000.0): Tensor[float64] =
  result = newTensor[float64](rows, cols)
  for r in 0 ..< rows:
    for c in 0 ..< cols: result[r, c] = value

let
  d1 = descale(1, duMicroVPerCount) # 1e-6 V/count
  d2 = descale(2, duMicroVPerCount)
  geo = sensitivity(20, suVPerMPerS)
  hyd = sensitivity(20, suVPerBar)  # 2e-4 V/Pa

proc oneRow(k: TraceKeys, counts: var seq[int], keys: var seq[TraceKeys]): ChunkMeta =
  keys = @[k]
  counts = @[cols]
  initChunkMeta(rowsOf(keys, counts), dtMs = 2)

proc stationKeys(sensor = 2'u8): TraceKeys =
  result = initTraceKeys()
  result.status = tsLive
  result.sensorType = sensor
  result.component = sensor
  result.channel = 3
  (result.rcvLine, result.rcvPoint, result.rcvIndex) = (10.0, 201.0, 1'i16)

block precedence_ladder:
  # Each rank beats the next: station + channel > station > channel >
  # component > sensor type > the catch-all. Each rule gives a different
  # sensitivity; remove the winner and the next one takes over.
  var keys: seq[TraceKeys]
  var counts: seq[int]
  let meta = oneRow(stationKeys(), counts, keys)
  let sens = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
  let matches = [
    RuleMatch(station: some stationKey(10, 201, 0), channel: 3),                 # station + channel
    RuleMatch(station: some stationKey(10, 201, 1)),                             # station (exact index)
    RuleMatch(channel: 3),
    RuleMatch(component: 2),
    RuleMatch(sensorType: 2),
    RuleMatch()]                                                                 # catch-all
  for first in 0 ..< matches.len:
    var rules: seq[CalibrationRule]
    for i in first ..< matches.len:
      rules.add CalibrationRule(match: matches[i], sensitivity: sensitivity(sens[i], suVPerMPerS))
    rules.add CalibrationRule(match: RuleMatch(sensorType: 99), descale: d1) # matches nothing
    let plan = initCalibrationPlan(initCalibration(rules))
    var rc = newSeq[RowCalibration](1)
    let s = plan.resolveInto(meta, rc)
    # No descale anywhere: unresolved, but the sensitivity is still chosen.
    doAssert rc[0].status == csUnresolved and s.unresolved == 1 and s.noDescale == 1
    doAssert rc[0].sensitivityRule == 0, $first & ": " & $rc[0].sensitivityRule
    doAssert rc[0].sensitivityFrom == fsRule
  # Independent factors: the descale from the catch-all, the sensitivity
  # from the station rule, the gain from the sensor-type rule.
  let plan = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(station: some stationKey(10, 201, 0)), sensitivity: geo),
    CalibrationRule(match: RuleMatch(sensorType: 2), gainDb: 20),
    CalibrationRule(descale: d1)]))
  var rc = newSeq[RowCalibration](1)
  let s = plan.resolveInto(meta, rc)
  doAssert s.converted == 1 and s.units[auVelocity] == 1
  doAssert (rc[0].descaleRule, rc[0].sensitivityRule, rc[0].gainRule) == (2, 0, 1)
  doAssert closeTo(rc[0].factor, 1e-6 / (10 * 20)) and rc[0].unit == auVelocity
  # A trace without a receiver index matches a station rule for index 1.
  keys[0].rcvIndex = 0
  let exact = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(station: some stationKey(10, 201, 1)), descale: d2,
      sensitivity: geo)]))
  discard exact.resolveInto(meta, rc)
  doAssert rc[0].status == csConverted and closeTo(rc[0].factor, 2e-6 / 20)
  # ... and not one for index 2.
  let other = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(station: some stationKey(10, 201, 2)), descale: d2,
      sensitivity: geo)]))
  discard other.resolveInto(meta, rc)
  doAssert rc[0].status == csUnresolved

block channel_set_rank:
  # A file view gives channel sets: on writeRev2's record, a channel-set rule
  # beats a sensor-type rule and loses to a channel rule.
  writeRev2()
  var f = openSeismic(rev2Path(), keys = true)
  let meta = initChunkMeta(f.rowsOf(0 ..< f.len), dtMs = 2)
  doAssert meta.rows.channelSetIndex(1) == 1 and meta.keys(1).sensorType == 2
  let plan = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(sensorType: 2), descale: d1, sensitivity: geo),
    CalibrationRule(match: RuleMatch(channelSet: 1), descale: d2),
    CalibrationRule(match: RuleMatch(channel: 3), sensitivity: sensitivity(40, suVPerMPerS)),
    CalibrationRule(match: RuleMatch(sensorType: 1), descale: d1, sensitivity: hyd)]))
  var rc = newSeq[RowCalibration](f.len)
  let s = plan.resolveInto(meta, rc)
  doAssert rc[0].status == csKept # auxiliary, no descale: kept
  for r in 1 .. 3: # the geophones of channel set 1: its descale
    doAssert rc[r].descaleRule == 1 and rc[r].unit == auVelocity
  doAssert closeTo(rc[1].factor, 2e-6 / 20) and closeTo(rc[3].factor, 2e-6 / 40)
  for r in 4 .. 5: # the hydrophones
    doAssert rc[r].unit == auPressure and closeTo(rc[r].factor, 1e-6 / 2e-4)
  doAssert s.converted == 5 and s.kept == 1 and s.units[auPressure] == 2
  f.close()

block file_factors:
  # A FileScale column: the file gives 3e-6 V/count and 25 V/(m/s).
  var keys = @[stationKeys(), stationKeys()]
  let counts = @[cols, cols]
  var scales = @[FileScale(descale: 3e-6, sensitivity: 25, unit: auVelocity),
    FileScale(descale: 3e-6, sensitivity: 25, unit: auVelocity)]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2, scales = scales)
  doAssert meta.hasFileScales and meta.fileScale(1).descale == 3e-6
  var rc = newSeq[RowCalibration](2)
  proc resolved(cal: Calibration): RowCalibration =
    discard initCalibrationPlan(cal).resolveInto(meta, rc)
    rc[0]
  # Fallback: a specific rule beats the file (a conflict); the file beats the catch-all.
  var r = resolved(initCalibration([CalibrationRule(match: RuleMatch(sensorType: 2),
    descale: d1), CalibrationRule(sensitivity: geo)]))
  doAssert r.descaleFrom == fsRule and r.sensitivityFrom == fsFile and r.conflict
  doAssert closeTo(r.factor, 1e-6 / 25)
  # First: the file wins both.
  r = resolved(initCalibration([CalibrationRule(match: RuleMatch(sensorType: 2), descale: d1),
    CalibrationRule(sensitivity: geo)], fileFactors = ffFirst))
  doAssert r.descaleFrom == fsFile and r.sensitivityFrom == fsFile and r.conflict
  doAssert closeTo(r.factor, 3e-6 / 25)
  # Ignore: the rules only, no conflict.
  r = resolved(initCalibration([CalibrationRule(match: RuleMatch(sensorType: 2), descale: d1),
    CalibrationRule(sensitivity: geo)], fileFactors = ffIgnore))
  doAssert r.descaleFrom == fsRule and r.sensitivityFrom == fsRule and not r.conflict
  doAssert closeTo(r.factor, 1e-6 / 20)
  # Rules that agree with the file within 0.1 %: no conflict.
  r = resolved(initCalibration([CalibrationRule(descale: descale(3.002, duMicroVPerCount),
    sensitivity: sensitivity(25, suVPerMPerS))]))
  doAssert not r.conflict and r.descaleFrom == fsFile
  r = resolved(initCalibration([CalibrationRule(descale: descale(3.01, duMicroVPerCount))]))
  doAssert r.conflict
  # The file alone.
  r = resolved(initCalibration())
  doAssert r.status == csConverted and closeTo(r.factor, 3e-6 / 25) and r.unit == auVelocity
  # The file's sensitivity in another unit than the rule's: a conflict.
  scales[0] = FileScale(descale: 3e-6, sensitivity: 25, unit: auPressure)
  r = resolved(initCalibration([CalibrationRule(sensitivity: geo)]))
  doAssert r.conflict and r.unit == auPressure # the file beats the catch-all
  # The same meta without the column: the rules only.
  let bare = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
  doAssert not bare.hasFileScales
  discard initCalibrationPlan(initCalibration()).resolveInto(bare, rc)
  doAssert rc[0].status == csUnresolved

block already_physical:
  var keys = @[stationKeys(1)]
  let counts = @[cols]
  let scales = @[FileScale(descale: 0.125, sensitivity: NaN, unit: auPressure, physical: true)]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2, scales = scales)
  var rc = newSeq[RowCalibration](1)
  discard initCalibrationPlan(initCalibration()).resolveInto(meta, rc)
  doAssert rc[0].factor == 0.125 and rc[0].unit == auPressure and not rc[0].conflict
  let withRule = initCalibration([CalibrationRule(descale: d1, sensitivity: hyd)])
  discard initCalibrationPlan(withRule).resolveInto(meta, rc)
  doAssert rc[0].factor == 0.125 and rc[0].conflict # the file says what the samples are
  var ignore = withRule
  ignore.fileFactors = ffIgnore
  discard initCalibrationPlan(ignore).resolveInto(meta, rc)
  doAssert closeTo(rc[0].factor, 1e-6 / 2e-4) and not rc[0].conflict
  var volts = initCalibration(target = ctVolts)
  discard initCalibrationPlan(volts).resolveInto(meta, rc)
  doAssert rc[0].status == csUnresolved # physical samples have no volts

block rows_by_status:
  # live (resolved), live (no sensitivity: NaN), dead (resolved), dead (no
  # sensitivity: kept), auxiliary (volts), failed (untouched).
  var keys = newSeq[TraceKeys](6)
  for i in 0 ..< 6:
    keys[i] = initTraceKeys()
    keys[i].sensorType = 2
  keys[1].sensorType = 7 # an accelerometer: no rule gives it a sensitivity
  keys[2].status = tsDead
  keys[3].status = tsDead
  keys[3].sensorType = 7
  keys[4].status = tsAuxiliary
  keys[4].sensorType = 0
  let counts = @[cols, cols, 10, cols, cols, cols] # row 2: 10 live samples
  let failed = @[5]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2, failed = failed)
  let plan = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(sensorType: 2), sensitivity: geo),
    CalibrationRule(descale: d1)]))
  var x = block64(6)
  let s = plan.calibrate(x, meta)
  let k = 1e-6 / 20 * 1000
  for c in 0 ..< cols:
    doAssert closeTo(x[0, c], k)
    doAssert x[1, c].isNaN
    doAssert x[2, c] == (if c < 10: k else: 1000.0) # past the live length: untouched
    doAssert x[3, c] == 1000 # dead, unresolved: kept
    doAssert closeTo(x[4, c], 1e-6 * 1000) # auxiliary: volts
    doAssert x[5, c] == 1000 # failed
  doAssert s.converted == 3 and s.unresolved == 1 and s.kept == 2 and s.noSensitivity == 1
  doAssert s.units[auVelocity] == 2 and s.units[auVolt] == 1 and s.units[auRecorded] == 2
  var rc = newSeq[RowCalibration](6)
  doAssert plan.resolveInto(meta, rc) == s
  doAssert rc[4].unit == auVolt and rc[4].sensitivityFrom == fsNone
  doAssert rc[5].status == csKept and rc[5].factor == 1 and rc[1].factor.isNaN
  # A catch-all sensitivity is never given to an auxiliary row.
  var cal = initCalibration([CalibrationRule(descale: d1, sensitivity: geo)])
  discard initCalibrationPlan(cal).resolveInto(meta, rc)
  doAssert rc[4].unit == auVolt and rc[4].sensitivityFrom == fsNone and
    rc[4].sensitivityRule == -1 and rc[4].factor == 1e-6
  # Auxiliary rows kept when asked.
  cal = initCalibration([CalibrationRule(descale: d1, sensitivity: geo)],
    scaleAuxiliary = false)
  discard initCalibrationPlan(cal).resolveInto(meta, rc)
  doAssert rc[4].status == csKept
  # Volts: the sensitivity is not needed, nor resolved.
  cal = initCalibration([CalibrationRule(descale: d1)], target = ctVolts)
  let vs = initCalibrationPlan(cal).resolveInto(meta, rc)
  doAssert vs.converted == 5 and vs.units[auVolt] == 5 and rc[1].factor == 1e-6
  doAssert rc[0].sensitivityFrom == fsNone

block gain_and_mismatch:
  var keys: seq[TraceKeys]
  var counts: seq[int]
  let meta = oneRow(stationKeys(1), counts, keys) # a hydrophone
  var rc = newSeq[RowCalibration](1)
  discard initCalibrationPlan(initCalibration([CalibrationRule(descale: d1, sensitivity: hyd,
    gainDb: 12)])).resolveInto(meta, rc)
  doAssert closeTo(rc[0].factor, 1e-6 / (pow(10.0, 0.6) * 2e-4)) and not rc[0].mismatch
  # A velocity sensitivity on a hydrophone: converted, and flagged.
  let s = initCalibrationPlan(initCalibration([CalibrationRule(descale: d1,
    sensitivity: geo)])).resolveInto(meta, rc)
  doAssert rc[0].mismatch and rc[0].unit == auVelocity and s.mismatches == 1
  # A velocity sensitivity where the file's headers name pascals: flagged too.
  var k2 = @[initTraceKeys()]
  k2[0].status = tsLive
  let c2 = @[cols]
  let fileUnit = @[FileScale(descale: 1e-6, sensitivity: NaN, unit: auPressure)]
  let m2 = initChunkMeta(rowsOf(k2, c2), dtMs = 2, scales = fileUnit)
  discard initCalibrationPlan(initCalibration([CalibrationRule(sensitivity: geo)])).resolveInto(m2, rc)
  doAssert rc[0].mismatch and rc[0].status == csConverted
  discard initCalibrationPlan(initCalibration([CalibrationRule(sensitivity: hyd)])).resolveInto(m2, rc)
  doAssert not rc[0].mismatch
  # A negative gain (attenuation the descale does not hold) multiplies.
  discard initCalibrationPlan(initCalibration([CalibrationRule(descale: d1, sensitivity: hyd,
    gainDb: -20)])).resolveInto(meta, rc)
  doAssert closeTo(rc[0].factor, 1e-6 * 10 / 2e-4)

block inverse_round_trip:
  var keys = @[stationKeys(), stationKeys(1), stationKeys(7)]
  let counts = @[cols, cols, cols]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
  let plan = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(sensorType: 1), sensitivity: hyd),
    CalibrationRule(match: RuleMatch(sensorType: 2), sensitivity: geo,
      descale: descale(0.3, duMicroVPerCount)),
    CalibrationRule(descale: d1)]))
  var x64 = newTensor[float64](3, cols)
  var x32 = newTensor[float32](3, cols)
  for r in 0 ..< 3:
    for c in 0 ..< cols:
      x64[r, c] = float64(c * 37 - 200) + 0.5
      x32[r, c] = float32(x64[r, c])
  let raw64 = x64.clone
  let raw32 = x32.clone
  plan.calibrate(x64, meta)
  plan.calibrate(x32, meta)
  doAssert x64[2, 0].isNaN # the accelerometer: no sensitivity
  plan.uncalibrate(x64, meta)
  plan.uncalibrate(x32, meta)
  for r in 0 .. 1:
    for c in 0 ..< cols:
      doAssert closeTo(x64[r, c], raw64[r, c], 1e-12)
      doAssert closeTo(x32[r, c], raw32[r, c], 1e-6)
  # Into: the source is kept.
  var dst = newTensor[float64](3, cols)
  plan.calibrateInto(raw64, dst, meta)
  doAssert raw64[0, 3] == float64(3 * 37 - 200) + 0.5
  doAssert closeTo(dst[0, 3], raw64[0, 3] * 0.3e-6 / 20)
  var back = newTensor[float64](3, cols)
  plan.uncalibrateInto(dst, back, meta)
  doAssert closeTo(back[1, 5], raw64[1, 5], 1e-12)

block per_row_kernels:
  var keys = @[stationKeys(), stationKeys(), stationKeys()]
  keys[1].status = tsAuxiliary
  let counts = @[cols, 4, cols]
  let failed = @[2]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2, failed = failed)
  var x = block64(3)
  x.scaleRows(meta, [0.5, 3.0, 7.0])
  for c in 0 ..< cols:
    doAssert x[0, c] == 500
    doAssert x[1, c] == (if c < 4: 3000.0 else: 1000.0) # any status, up to its live length
    doAssert x[2, c] == 1000 # failed
  let src = block64(3)
  var dst = newTensor[float64](3, cols)
  scaleRowsInto(src, dst, meta, [NaN, 2.0, 7.0])
  for c in 0 ..< cols:
    doAssert dst[0, c].isNaN and src[0, c] == 1000
    doAssert dst[1, c] == (if c < 4: 2000.0 else: 1000.0) # the rest copied
    doAssert dst[2, c] == 1000
  doAssertRaises(ValueError): x.scaleRows(meta, [1.0, 1.0])
  doAssertRaises(ValueError): x.scaleRows(meta, [1.0, Inf, 1.0])
  var narrow = newTensor[float64](3, cols - 1)
  doAssertRaises(ValueError): scaleRowsInto(src, narrow, meta, [1.0, 1.0, 1.0])
  # `scaleRows` with `resolveInto`'s factors is `calibrate`.
  let plan = initCalibrationPlan(initCalibration([CalibrationRule(descale: d1,
    sensitivity: geo)]))
  var rc = newSeq[RowCalibration](3)
  discard plan.resolveInto(meta, rc)
  var a = block64(3)
  var b = block64(3)
  plan.calibrate(a, meta)
  var factors = newSeq[float64](3)
  for i in 0 ..< 3: factors[i] = rc[i].factor
  b.scaleRows(meta, factors)
  for r in 0 ..< 3:
    for c in 0 ..< cols: doAssert a[r, c] == b[r, c]

block compare_two_tables:
  # The damped reading of a sheet against an open-circuit reading of the
  # same sheet: 20 V/(m/s) as wired, or 20 open circuit into 400 Ω / 1600 Ω
  # (16 V/(m/s)): data comes out 20·log10(20/16) = 1.94 dB louder under the
  # second.
  var keys = @[stationKeys(), stationKeys(1)]
  let counts = @[cols, cols]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
  let damped = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(sensorType: 2), sensitivity: geo),
    CalibrationRule(match: RuleMatch(sensorType: 1), sensitivity: hyd),
    CalibrationRule(descale: d1)]))
  let open = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(sensorType: 2), sensitivity: openCircuitSensitivity(20,
      suVPerMPerS, coilOhms = 400, shuntOhms = 1600)),
    CalibrationRule(descale: d1)]))
  var db = newSeq[float64](2)
  let rr = ratioDbInto(open, damped, meta, db)
  doAssert closeTo(db[0], 20 * log10(20 / 16)) and db[1].isNaN # no hydrophone rule in `open`
  doAssert rr.rows == 1 and rr.minDb == db[0] and rr.maxDb == db[0]
  var three = newSeq[float64](3)
  doAssertRaises(ValueError): discard ratioDbInto(open, damped, meta, three)

block refusals:
  doAssertRaises(ValueError): # no factor
    discard initCalibrationPlan(initCalibration([CalibrationRule(match: RuleMatch(sensorType: 2))]))
  doAssertRaises(ValueError): # two rules for the same rows
    discard initCalibrationPlan(initCalibration([
      CalibrationRule(match: RuleMatch(sensorType: 2), descale: d1),
      CalibrationRule(match: RuleMatch(sensorType: 2), sensitivity: geo)]))
  doAssertRaises(ValueError): # two catch-alls
    discard initCalibrationPlan(initCalibration([CalibrationRule(descale: d1),
      CalibrationRule(sensitivity: geo)]))
  for bad in [RuleMatch(sensorType: 256), RuleMatch(component: -2), RuleMatch(channelSet: -2),
      RuleMatch(channel: Inf), RuleMatch(station: some stationKey(NaN, 1, 0))]:
    doAssertRaises(ValueError):
      discard initCalibrationPlan(initCalibration([CalibrationRule(match: bad, descale: d1)]))
  doAssertRaises(ValueError):
    discard initCalibrationPlan(initCalibration([CalibrationRule(descale: d1, gainDb: Inf)]))
  # Different criteria sets that could match the same row are not a duplicate.
  discard initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(sensorType: 2), descale: d1),
    CalibrationRule(match: RuleMatch(component: 2), descale: d1)]))
  # A shape mismatch.
  var keys = @[stationKeys()]
  let counts = @[cols]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
  var x = block64(2)
  let plan = initCalibrationPlan(initCalibration([CalibrationRule(descale: d1)]))
  doAssertRaises(ValueError): plan.calibrate(x, meta)
  var rc = newSeq[RowCalibration](2)
  doAssertRaises(ValueError): discard plan.resolveInto(meta, rc)
  var scales = @[initFileScale(), initFileScale()]
  doAssertRaises(ValueError): discard initChunkMeta(rowsOf(keys, counts), dtMs = 2,
    scales = scales)

block sub_meta_and_printing:
  # `sub` keeps the column aligned with its rows.
  var keys = @[stationKeys(), stationKeys(), stationKeys()]
  let counts = @[cols, cols, cols]
  let scales = @[FileScale(descale: 1e-6), FileScale(descale: 2e-6), FileScale(descale: 3e-6)]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2, scales = scales)
  let s = meta.sub(1 .. 2)
  doAssert s.hasFileScales and s.fileScale(0).descale == 2e-6 and s.fileScale(1).descale == 3e-6
  doAssert not meta.sub(3 .. 2).hasFileScales
  let cal = initCalibration([CalibrationRule(match: RuleMatch(sensorType: 1), sensitivity: hyd),
    CalibrationRule(descale: d1, gainDb: 6)])
  doAssert $cal.rules[0] == "sensor type 1: sensitivity 20 V/bar (0.0002 V/Pa)", $cal.rules[0]
  doAssert $cal.rules[1] == "any row: descale 1 µV/count, gain 6.0 dB", $cal.rules[1]
  doAssert $RuleMatch(station: some stationKey(10, 201, 2), channel: 3) ==
    "station 10.0/201.0.2, channel 3.0"
  doAssert $CalibrationSummary(converted: 2, units: [auRecorded: 0, auVolt: 0,
    auVelocity: 2, auAcceleration: 0, auPressure: 0]) ==
    "2 converted (2 m/s), 0 unresolved, 0 kept"

block no_allocation:
  var keys = newSeq[TraceKeys](64)
  for i in 0 ..< 64:
    keys[i] = stationKeys(uint8(1 + i mod 2))
    keys[i].rcvPoint = float64(i)
  let counts = newSeq[int](64)
  var cs = counts
  for c in cs.mitems: c = cols
  let meta = initChunkMeta(rowsOf(keys, cs), dtMs = 2)
  let plan = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(station: some stationKey(10, 5, 0)), sensitivity: geo),
    CalibrationRule(match: RuleMatch(sensorType: 1), sensitivity: hyd),
    CalibrationRule(match: RuleMatch(sensorType: 2), sensitivity: geo),
    CalibrationRule(descale: d1)]))
  var x = newTensor[float32](64, cols)
  var y = newTensor[float32](64, cols)
  var rc = newSeq[RowCalibration](64)
  plan.calibrate(x, meta)
  let n = allocsIn:
    for i in 0 ..< 3:
      plan.calibrate(x, meta)
      plan.calibrateInto(x, y, meta)
      discard plan.resolveInto(meta, rc)
  doAssert n == 0, $n & " allocations"
