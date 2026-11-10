## Gain throughput (Phase 4) on a viewer panel, 2000 traces × 2001 samples of
## float32 at 2 ms, one thread: per-trace and panel statistics (with and
## without the percentile histogram), AGC at three window lengths, a panel
## normalization and a time-variant gain curve.

import std/strutils
import benchutil
import processSEG
import ../tests/utils/synth

const
  traces = 2000
  ns = 2001
  dt = 2.0

proc panel(): SynthGather[float32] =
  result = cmpGather[float32](cmpX = 0, offsets = newSeq[float64](traces), ns, dt)
  for r in 0..<traces:
    var x = newSeq[float32](ns)
    x.addNoise(1, seed = r)
    x.addRicker(dt, 1000 + float64(r), 25, amp = 4)
    for j in 0..<ns: result.data[r, j] = x[j]

proc ms(r: BenchResult): string = formatFloat(r.medianNs / 1e6, ffDecimal, 3) & " ms per panel"

proc main() =
  printHeader "gain, " & $traces & " traces × " & $ns & " samples f32 at 2 ms, 1 thread"
  var g = panel()
  let meta = initChunkMeta(g.rows, dtMs = dt)
  var ws = initWorkspace()
  ws.reserve(panelStatsWorkspace)
  var stats = newSeq[TraceStats](traces)
  let bytes = traces * ns * 4
  var sink: float64
  let a = bench("traceStatsInto", bytes):
    traceStatsInto(g.data, meta, stats)
  a.report
  echo "    ", a.ms
  let b = bench("panelStats, no clip", bytes):
    sink += panelStats(g.data, meta, ws, clipPercentile = NaN).rms
  b.report
  echo "    ", b.ms
  let c = bench("panelStats, p99 clip", bytes):
    sink += panelStats(g.data, meta, ws, clipPercentile = 99).clip
  c.report
  echo "    ", c.ms
  # AGC: the cost must not depend on the window (O(n)).
  for windowMs in [100.0, 500.0, 2000.0]:
    let plan = initAgcPlan(windowMs, dtMs = dt, samples = ns)
    ws.reserve(plan.workspaceSize)
    let r = bench("agc rms " & $int(windowMs) & " ms", 2 * bytes):
      g.data.agc(meta, plan, ws)
    r.report
    echo "    ", r.ms
  let n99 = initNormalizePlan(normPercentile, samples = ns, scope = perPanel)
  ws.reserve(n99.workspaceSize)
  let n = bench("normalize panel p99", 2 * bytes):
    g.data.normalize(meta, n99, ws)
  n.report
  echo "    ", n.ms
  let curve = initGainCurvePlan(float32, knotGain([GainKnot(tMs: 0, dB: 0),
    GainKnot(tMs: 4000, dB: 24)]), dtMs = dt, samples = ns)
  let tv = bench("gain curve, t0 on the grid", 2 * bytes):
    g.data.applyGainCurve(meta, curve)
  tv.report
  echo "    ", tv.ms
  doAssert sink > 0 and stats[0].rms > 0

main()
