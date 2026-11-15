## A file's own scale headers (task 14.4): `fileScalesInto` and `scaleReport`
## on synthetic SEG-D Rev 2 and Rev 3, SEG-Y and SEG-2 files written with
## parseSEG; a prefetch chunk calibrated from its file's factors; SEG-Y
## written with `segyUnitOverrides` read back as physical; the cost of a
## Rev 3 column.

import std/[assertions, math, monotimes, options, os, strutils, times]
import parseSEG, processSEG
import utils/calibfiles

func closeTo(a, b: float64, rel = 1e-6): bool = abs(a - b) <= rel * max(abs(a), abs(b))

proc scalesOf(f: SeismicFile): seq[FileScale] =
  result = newSeq[FileScale](f.len)
  f.fileScalesInto(0 ..< f.len, result)

block segd_rev2:
  writeRev2()
  var f = openSeismic(rev2Path(), keys = true)
  doAssert f.len == 6
  let s = scalesOf(f)
  doAssert closeTo(s[0].descale, 1e-3) # MP 0: 1 mV
  for t in 1 .. 3: doAssert closeTo(s[t].descale, pow(2.0, -10) * 1e-3)
  for t in 4 .. 5: doAssert closeTo(s[t].descale, pow(2.0, 3.5) * 1e-3)
  for t in 0 .. 5: doAssert s[t].sensitivity.isNaN and s[t].unit == auRecorded and not s[t].physical
  let rep = f.scaleReport()
  doAssert rep.len == 3, $rep.len
  doAssert rep[1].first == 1 and rep[1].count == 3 and rep[1].channelSet == 1
  doAssert ("cs.mpFactor", -10.0) in rep[1].fields and ("ext1.sensorType", 2.0) in rep[1].fields
  doAssert ("gh1.formatCode", 8058.0) in rep[1].fields
  let line = $rep[1]
  doAssert line.startsWith("3 traces from 1, channel set 1: gh1.formatCode = 8058"), line
  doAssert "cs.mpFactor = -10" in line and "→ 9.76563e-07 V/count, no sensitivity" in line, line
  f.close()

block segd_rev3:
  writeRev3()
  var f = openSeismic(rev3Path(), keys = true)
  doAssert f.len == 5
  let s = scalesOf(f)
  doAssert s[0].descale.isNaN # DSM 0: unknown, not 1
  for t in 1 .. 2:
    doAssert closeTo(s[t].descale, 1.5e-7) # 1.5e-4 mV
    doAssert closeTo(s[t].sensitivity, 20) and s[t].unit == auVelocity # 20 mV per mm/s
  doAssert closeTo(s[3].descale, 2e-7) and closeTo(s[3].sensitivity, 2e-4)
  doAssert s[3].unit == auPressure # the channel set's unit
  doAssert closeTo(s[4].descale, 2e-7) and s[4].sensitivity.isNaN # STY 0
  doAssert s[4].unit == auPressure and s[0].unit == auRecorded # the unit without a sensitivity
  doAssert $s[4] == "2e-07 V/count, no sensitivity (the headers name Pa)", $s[4]
  let rep = f.scaleReport()
  doAssert rep.len == 4, $rep.len
  doAssert rep[1].count == 2 and ("sensorInfo.sensorSensitivity", 20.0) in rep[1].fields
  doAssert ("ext1.physicalUnit", 3.0) in rep[1].fields
  doAssert ("cs.physicalUnit", 18.0) in rep[2].fields and rep[2].first == 3
  doAssert ("sensorInfo.sensorSensitivity", 0.0) in rep[3].fields
  # Calibrated from the file alone, through a prefetch chunk and its meta.
  var p = newPrefetcher[float64](f, 0 ..< f.len, chunkTraces = 8, uniform = false)
  var c: PrefetchChunk[float64]
  var scales: seq[FileScale]
  var seen = 0
  let plan = initCalibrationPlan(initCalibration())
  while p.next(c):
    scales.setLen(c.rows.len)
    f.fileScalesInto(c.rows, scales)
    let meta = chunkMeta(c, dtMs = 1, scales = scales)
    var rc = newSeq[RowCalibration](meta.len)
    let sum = plan.resolveInto(meta, rc)
    doAssert sum.converted == 3 and sum.unresolved == 1 and sum.kept == 1, $sum
    plan.calibrate(c.samples, meta)
    for r in 0 ..< meta.len:
      let t = c.rows.traceOf(r)
      let v = c.samples[r, 10]
      case t
      of 0: doAssert v == rowValue(0) # auxiliary, no descale: kept
      of 1, 2: doAssert closeTo(v, rowValue(t) * 1.5e-7 / 20)
      of 3: doAssert closeTo(v, rowValue(t) * 2e-7 / 2e-4)
      else: doAssert v.isNaN # no sensitivity anywhere
      inc seen
  p.close()
  doAssert seen == 5
  # A rule fills the hydrophone without one; where the file has one, a
  # different rule value is a conflict, and the rule wins (fallback order).
  let meta = chunkMeta(f, 0 ..< f.len, dtMs = 1, scales = s)
  let rules = initCalibrationPlan(initCalibration([CalibrationRule(match: RuleMatch(sensorType: 1),
    sensitivity: sensitivity(25, suVPerBar))]))
  var rc = newSeq[RowCalibration](f.len)
  let sum = rules.resolveInto(meta, rc)
  doAssert rc[4].status == csConverted and rc[4].sensitivityFrom == fsRule and not rc[4].conflict
  doAssert rc[3].sensitivityFrom == fsRule and rc[3].conflict and sum.conflicts == 1
  doAssert rc[1].sensitivityFrom == fsFile and rc[1].descaleFrom == fsFile
  f.close()

block segy:
  writeSegy()
  var f = openSeismic(segyPath(), keys = true)
  doAssert f.len == segyCases.len
  let s = scalesOf(f)
  doAssert s[0].descale.isNaN and s[0].sensitivity.isNaN # nothing set: unknown
  doAssert s[1].descale == pow(2.0, -20) and s[1].sensitivity.isNaN
  doAssert closeTo(s[2].descale, pow(2.0, -10) * 1e-3)
  doAssert s[3].descale == pow(2.0, -10) and closeTo(s[3].sensitivity, 0.02)
  doAssert s[3].unit == auVelocity and not s[3].physical
  doAssert s[4].physical and s[4].unit == auPressure and s[4].descale == 0.125
  doAssert s[5].descale == 1 and closeTo(s[5].sensitivity, 500) and s[5].unit == auPressure
  doAssert s[6].descale.isNaN and s[6].sensitivity.isNaN and not s[6].physical # C, no units
  doAssert s[7].physical and s[7].unit == auVelocity and closeTo(s[7].descale, 0.5)
  let rep = f.scaleReport()
  doAssert rep.len == segyCases.len
  doAssert ("trace.transductionMantissa", 5.0) in rep[3].fields
  doAssert ("trace.instrumentGainConstant", 0.0) in rep[3].fields
  doAssert $rep[4] == "1 trace from 4: trace.traceWeightingFactor = 3, trace.traceValueUnit = 1, " &
    "trace.transductionMantissa = 0, trace.transductionExponent = 0, " &
    "trace.transductionUnits = 0, trace.gainType = 0, trace.instrumentGainConstant = 0, " &
    "trace.instrumentInitialGain = 0 → 0.125 Pa/count (physical)", $rep[4]
  let part = f.scaleReport(2 .. 3)
  doAssert part.len == 2 and part[0].first == 2
  f.close()

block seg2:
  writeSeg2()
  var f = openSeismic(seg2Path(), keys = true)
  let s = scalesOf(f)
  doAssert closeTo(s[0].descale, 0.25e-3) and s[1].descale.isNaN and closeTo(s[2].descale, 1e-6)
  let rep = f.scaleReport()
  doAssert rep.len == 3 and ("trace.STACK", 2.0) in rep[0].fields
  doAssert rep[1].fields[0][1].isNaN # no DESCALING_FACTOR
  f.close()

block write_physical_and_read_back:
  # m/s samples written with their unit: read back as physical, so the
  # calibration of the copy leaves them as they are, whatever a rule says.
  let path = segyPath().replace(".sgy", "_out.sgy")
  var w = openSeismicWriter(path, SegyOutputOptions())
  var k = initTraceKeys()
  k.intervalUs = 2000
  k.t0Ms = 0
  k.status = tsLive
  k.sensorType = 2
  let v = @[1.5e-6'f32, -2e-6, 3e-7, 0]
  for u in [auVelocity, auPressure, auVolt]:
    w.writeTrace(k, v, segyUnitOverrides(u))
  doAssert w.close().len == 0
  var f = openSeismic(path, keys = true)
  let s = scalesOf(f)
  doAssert s[0].physical and s[0].unit == auVelocity and s[0].descale == 1
  doAssert s[1].physical and s[1].unit == auPressure
  doAssert not s[2].physical and s[2].descale == 1 and s[2].sensitivity.isNaN # volts
  var x = f.readTraces(0 ..< f.len, float32)
  let meta = chunkMeta(f, 0 ..< f.len, scales = s)
  let plan = initCalibrationPlan(initCalibration([CalibrationRule(
    descale: descale(1, duMicroVPerCount), sensitivity: sensitivity(20, suVPerMPerS))]))
  var rc = newSeq[RowCalibration](f.len)
  discard plan.resolveInto(meta, rc)
  doAssert rc[0].factor == 1 and rc[0].unit == auVelocity and rc[0].conflict
  # The volts row: the file's descale beats the catch-all rule (a conflict),
  # whose sensitivity then applies.
  doAssert rc[2].unit == auVelocity and rc[2].descaleFrom == fsFile and rc[2].conflict
  plan.calibrate(x, meta)
  doAssert x[0, 0] == 1.5e-6'f32 and x[1, 1] == -2e-6'f32
  doAssert segyUnitOverrides(auRecorded)[0].value.toFloat == 0
  f.close()

block errors:
  var f = openSeismic(rev2Path(), keys = true)
  var s = newSeq[FileScale](2)
  doAssertRaises(ValueError): f.fileScalesInto(0 ..< 3, s)
  doAssertRaises(ValueError): f.fileScalesInto([0, 99], s)
  doAssertRaises(ValueError): discard f.scaleReport(0 .. 99)
  let empty = f.scaleReport(3 ..< 3)
  doAssert empty.len == 0
  var none: seq[FileScale]
  f.fileScalesInto(3 ..< 3, none)
  f.close()

block example_files:
  # The SEG-D Rev 3.0 shot of the exampleFiles (skipped when absent): its
  # headers give 1000 mV per count (DSM) and name m/s, without Sensor Info.
  let shot = currentSourcePath().parentDir / ".." / "exampleFiles" / "00000151_12900_11300.sgd"
  if fileExists(shot):
    var f = openSeismic(shot, keys = true)
    let rep = f.scaleReport()
    doAssert rep.len == 4 and rep[3].count == 3954 and rep[3].channelSet == 3
    doAssert ("cs.descaleMultiplier", 1000.0) in rep[3].fields
    doAssert ("ext1.physicalUnit", 4.0) in rep[3].fields
    for (name, v) in rep[3].fields:
      if name == "sensorInfo.sensorSensitivity": doAssert v.isNaN # no Sensor Info block
    var s = newSeq[FileScale](f.len)
    let t0 = getMonoTime()
    f.fileScalesInto(0 ..< f.len, s)
    let ms = float64(inMicroseconds(getMonoTime() - t0)) / 1000
    doAssert s[3].descale == 1 and s[3].sensitivity.isNaN and s[3].unit == auVelocity
    let meta = chunkMeta(f, 0 ..< f.len, scales = s)
    var rc = newSeq[RowCalibration](f.len)
    let geo = initCalibrationPlan(initCalibration([CalibrationRule(
      sensitivity: sensitivity(22.5, suVPerMPerS))]))
    let sg = geo.resolveInto(meta, rc)
    doAssert sg.units[auVelocity] == 3954 and sg.units[auVolt] == 3 and sg.mismatches == 0, $sg
    let hyd = initCalibrationPlan(initCalibration([CalibrationRule(
      sensitivity: sensitivity(20, suVPerBar))]))
    let sh = hyd.resolveInto(meta, rc)
    doAssert sh.mismatches == 3954, $sh # the headers name m/s
    echo "  example shot: ", sg, "; fileScalesInto ", formatFloat(ms, ffDecimal, 1), " ms for ",
      f.len, " traces"
    f.close()
  let sweep = currentSourcePath().parentDir / ".." / "exampleFiles" / "Sweep0.sgy"
  if fileExists(sweep):
    # A transduction mantissa of 1024 with no units, no value unit, N = 0: unknown.
    var f = openSeismic(sweep, keys = true)
    var s = newSeq[FileScale](f.len)
    f.fileScalesInto(0 ..< f.len, s)
    doAssert s[0].descale.isNaN and s[0].sensitivity.isNaN
    f.close()

block cost_of_a_rev3_column:
  # The Sensor Info block is read through parseSEG's header listing: measure
  # it on a record of 2000 geophones.
  let path = rev3Path().replace(".sgd", "_big.sgd")
  writeRev3(path, geophones = 2000)
  var f = openSeismic(path, keys = true)
  var s = newSeq[FileScale](f.len)
  let t0 = getMonoTime()
  f.fileScalesInto(0 ..< f.len, s)
  let us = float64(inMicroseconds(getMonoTime() - t0)) / float64(f.len)
  doAssert closeTo(s[1000].sensitivity, 20)
  echo "  fileScalesInto, SEG-D Rev 3 with Sensor Info: ", formatFloat(us, ffDecimal, 2),
    " µs per trace (", f.len, " traces)"
  f.close()
