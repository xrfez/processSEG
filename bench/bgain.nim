## Gain throughput (Phase 4) on a viewer panel, 2000 traces × 2001 samples of
## float32 at 2 ms, one thread: per-trace and panel statistics (with and
## without the percentile histogram).

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
  doAssert sink > 0 and stats[0].rms > 0

main()
