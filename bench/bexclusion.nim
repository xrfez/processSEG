## The exclusion area's cost against the first-break pick it follows: one
## shot of 4000 traces × 4000 samples (4 receiver lines of 1000 stations) is
## picked, fitted and judged. `noiseQc` takes the picks and the fit already
## made (a viewer has them from its overlay), so its cost is the area, the
## statistics and the neighbour finding alone.

import std/[math, random, strutils]
import benchutil
import processSEG
import parseSEG

const
  dt = 2.0
  lines = 4
  stations = 1000
  traces = lines * stations
  ns = 4000

proc ms(r: BenchResult): string = formatFloat(r.medianNs / 1e6, ffDecimal, 3) & " ms"

proc main() =
  var keys = newSeq[TraceKeys](traces)
  var counts = newSeq[int](traces)
  for j in 0 ..< lines:
    for i in 0 ..< stations:
      let r = j * stations + i
      var k = initTraceKeys()
      k.status = tsLive
      k.t0Ms = 0
      k.srcX = 12500
      k.srcY = -300
      k.rcvX = 25.0 * float64(i)
      k.rcvY = 400.0 * float64(j)
      k.rcvLine = float64(1000 + j)
      k.rcvPoint = float64(i + 1)
      k.offset = hypot(k.rcvX - k.srcX, k.rcvY - k.srcY)
      keys[r] = k
      counts[r] = ns
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = dt)
  var x = zeros[float32](traces, ns)
  var rng = initRand(1)
  for r in 0 ..< traces:
    let d = keys[r].offset
    let level = 0.1 + 40 * exp(-((d / 1500) ^ 2))
    var row = newSeq[float32](ns)
    row.addNoise(0.1, 100 + r)
    row.addRicker(dt, 30 + d / 6.0, 30, level)
    for j in 0 ..< ns: x[r, j] = row[j]
  var ws = initWorkspace()
  let plan = initFirstBreakPlan(float32, FirstBreakParams(), dt, ns)
  ws.reserve(plan.workspaceSize + traceAmplitudeWorkspace(ns) + noiseWorkspace(traces))
  var picks = newSeq[FirstBreakPick](traces)
  let pick = bench("pickInto 4000 x 4000 f32", traces * ns * 4):
    plan.pickInto(x, meta, picks, ws)
  pick.report
  echo "    ", pick.ms
  var amps = newSeq[TraceAmplitude](traces)
  traceAmplitudeInto(x, meta, AmplitudeParams(), amps, ws)
  var fit: ShotFit
  fitShot(meta, picks, ShotFitParams(correlate: false), fit)
  var roles = newSeq[TraceRole](traces)
  meta.traceRolesInto(gdCommonSource, [], roles)
  var res: NoiseResult
  for params in [ExclusionParams(), ExclusionParams(choice: acSteps)]:
    let q = bench("noiseQc " & $params.choice & " 4000 traces", 0):
      noiseQc(meta, amps, picks, fit, roles, AmplitudeParams(), params, gdCommonSource, res, ws)
    q.report
    echo "    ", q.ms, " (", formatFloat(100 * q.medianNs / pick.medianNs, ffDecimal, 1),
      " % of pickInto; rule ", res.rule, ", ", res.excluded, " traces not judged)"
  let rt = bench("traceRolesInto 4000 traces", 0):
    meta.traceRolesInto(gdCommonSource, [], roles)
  rt.report
  echo "    ", rt.ms

main()
