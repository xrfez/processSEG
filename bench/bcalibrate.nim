## Calibration cost on a viewer panel, 2000 traces × 2001 samples
## of float32, one thread: resolving every row's factor (`resolveInto`), and
## converting the panel in place and `src → dst`, with a table of four rules
## (sensor types, one per-station rule) and a file-scale column.

import std/[options, strutils]
import benchutil
import parseSEG, processSEG

const
  traces = 2000
  ns = 2001

proc ms(r: BenchResult): string = formatFloat(r.medianNs / 1e6, ffDecimal, 3) & " ms per panel"

proc main() =
  printHeader "calibration, " & $traces & " traces × " & $ns & " samples f32, 1 thread"
  var keys = newSeq[TraceKeys](traces)
  var counts = newSeq[int](traces)
  var scales = newSeq[FileScale](traces)
  for r in 0 ..< traces:
    keys[r] = initTraceKeys()
    keys[r].status = tsLive
    keys[r].sensorType = uint8(1 + r mod 4) # a four-component unit per station
    keys[r].channel = float64(1 + r mod 4)
    (keys[r].rcvLine, keys[r].rcvPoint, keys[r].rcvIndex) = (10.0, float64(r div 4), 1'i16)
    counts[r] = ns
    scales[r] = FileScale(descale: 2.5e-7, sensitivity: NaN, unit: auRecorded)
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2, scales = scales)
  let plan = initCalibrationPlan(initCalibration([
    CalibrationRule(match: RuleMatch(sensorType: 1), sensitivity: sensitivity(20, suVPerBar)),
    CalibrationRule(match: RuleMatch(sensorType: 2), sensitivity: sensitivity(22.5, suVPerMPerS)),
    CalibrationRule(match: RuleMatch(component: 3), sensitivity: sensitivity(21, suVPerMPerS)),
    CalibrationRule(match: RuleMatch(station: some stationKey(10, 100, 0), channel: 4),
      sensitivity: sensitivity(21.7, suVPerMPerS))]))
  var x = newTensor[float32](traces, ns)
  for r in 0 ..< traces:
    for j in 0 ..< ns: x[r, j] = float32((r + j) mod 97)
  var y = newTensor[float32](traces, ns)
  var rows = newSeq[RowCalibration](traces)
  let bytes = traces * ns * 4
  var sink = 0
  let a = bench("resolveInto (no samples)", 0):
    sink += plan.resolveInto(meta, rows).converted
  a.report
  echo "    ", formatFloat(a.medianNs / 1e3, ffDecimal, 1), " µs per panel, ",
    formatFloat(a.medianNs / float64(traces), ffDecimal, 1), " ns per row"
  # In place, then back (the values must not drift into subnormals).
  let b = bench("calibrate + uncalibrate in place", 4 * bytes):
    sink += plan.calibrate(x, meta).converted
    sink += plan.uncalibrate(x, meta).converted
  b.report
  echo "    ", formatFloat(b.medianNs / 2e6, ffDecimal, 3), " ms per pass"
  let c = bench("calibrateInto (src → dst)", 2 * bytes):
    sink += plan.calibrateInto(x, y, meta).converted
  c.report
  echo "    ", c.ms
  let flow = initPipeline([calibrateOp(initCalibration([CalibrationRule(
    sensitivity: sensitivity(22.5, suVPerMPerS))])), demeanOp()]).compile(float32, 2, ns)
  var ws = initWorkspace()
  ws.reserve(flow)
  let d = bench("pipeline calibrate + demean, src → dst", 2 * bytes):
    flow.apply(x, y, meta, ws)
  d.report
  echo "    ", d.ms
  doAssert sink > 0

main()
