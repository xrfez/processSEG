## Amplitude statistics (gain/stats): per-trace RMS and max, exact and
## histogram percentiles against a sort-based reference, liveness, NaN rows,
## and no allocation.

import std/[algorithm, assertions, math, random]
import processSEG
import utils/[allocs, synth]

const binRel = 1.0 / 256 # the histogram bin width, relative

func near(a, b: float64, rel: float64): bool =
  abs(a - b) <= rel * max(abs(a), abs(b))

proc sortPercentile(values: seq[float64], p: float64): float64 =
  ## numpy's default (linear) percentile of `values`.
  var s = values
  s.sort()
  let h = float64(s.len - 1) * p / 100
  let k = int(floor(h))
  if k + 1 >= s.len: return s[^1]
  s[k] + (h - float64(k)) * (s[k + 1] - s[k])

block trace_stats_exact:
  let x = @[3'f32, -4, 0, 0]
  let s = traceStats(x)
  doAssert s.rms == sqrt(25.0 / 4) and s.maxAbs == 4
  let e = traceStats(newSeq[float64]())
  doAssert e.rms == 0 and e.maxAbs == 0
  # A length that is not a multiple of the 8 accumulators.
  var y = newSeq[float64](37)
  for i in 0..<y.len: y[i] = float64(i) - 20
  var sq = 0.0
  for v in y: sq += v * v
  doAssert traceStats(y).rms == sqrt(sq / 37) and traceStats(y).maxAbs == 20

block trace_stats_non_finite:
  var x = @[1'f32, 2, NaN, 3, 4, 5, 6, 7, 8, 9]
  doAssert traceStats(x).rms.isNaN and traceStats(x).maxAbs.isNaN
  x[2] = -Inf
  doAssert traceStats(x).rms == Inf and traceStats(x).maxAbs == Inf
  x[3] = NaN
  doAssert traceStats(x).maxAbs.isNaN, "NaN wins over Inf"

block percentile_abs_exact:
  var r = initRand(7)
  var ws = initWorkspace()
  for n in [1, 2, 3, 10, 101, 2001]:
    var x = newSeq[float32](n)
    var ref64 = newSeq[float64](n)
    for i in 0..<n:
      x[i] = float32(r.gauss() * 100)
      ref64[i] = abs(float64(x[i]))
    ws.reserve(percentileWorkspace(n))
    for p in [0.0, 1, 25, 50, 90, 99, 99.9, 100]:
      let got = percentileAbs(x, p, ws)
      let want = sortPercentile(ref64, p)
      doAssert near(got, want, 1e-12), $n & " " & $p & ": " & $got & " vs " & $want
  # Many ties.
  var t = newSeq[float64](500)
  for i in 0..<t.len: t[i] = float64(i mod 3)
  var tr = t
  ws.reserve(percentileWorkspace(t.len))
  for p in [10.0, 50, 66.7, 95]:
    doAssert percentileAbs(t, p, ws) == sortPercentile(tr, p), $p
  doAssert percentileAbs(newSeq[float32](), 50, ws) == 0
  doAssert percentileAbs(@[1.0, NaN], 50, ws).isNaN
  doAssertRaises ValueError: discard percentileAbs(t, 101, ws)
  doAssertRaises ValueError: discard percentileAbs(t, NaN, ws)

proc panelGather(rows, cols: int, seed: int64): SynthGather[float32] =
  result = cmpGather[float32](0, newSeq[float64](rows), cols, dtMs = 2)
  for r in 0..<rows:
    var x = newSeq[float32](cols)
    x.addNoise(1, seed = seed + r)
    x.addSpike(cols div 2, amp = 50) # a few large values, as real data has
    let decay = 10.0 ^ (-3.0 * float64(r) / float64(rows)) # 60 dB across the panel
    for j in 0..<cols: result.data[r, j] = float32(float64(x[j]) * decay)

block panel_stats_against_sort:
  var g = panelGather(rows = 64, cols = 501, seed = 1)
  g.setLive(3, 200)
  g.keys[5].status = tsDead
  g.keys[6].status = tsAuxiliary
  let failed = @[9]
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  var used: seq[float64]
  var sq = 0.0
  var rowsUsed = 0
  for r in 0..<g.len:
    if not meta.isLive(r): continue
    inc rowsUsed
    for j in 0..<meta.liveSamples(r, g.cols):
      let v = float64(g.data[r, j])
      used.add abs(v)
      sq += v * v
  var ws = initWorkspace()
  ws.reserve(panelStatsWorkspace)
  for p in [0.0, 1, 50, 90, 99, 99.5, 99.9, 100]:
    let s = panelStats(g.data, meta, ws, clipPercentile = p)
    doAssert s.rows == rowsUsed and s.samples == used.len
    doAssert near(s.rms, sqrt(sq / float64(used.len)), 1e-12)
    doAssert s.maxAbs == max(used)
    let want = sortPercentile(used, p)
    doAssert near(s.clip, want, binRel), $p & ": " & $s.clip & " vs " & $want
  doAssert panelStats(g.data, meta, ws, clipPercentile = 100).clip == max(used)
  let none = panelStats(g.data, meta, ws, clipPercentile = NaN)
  doAssert none.clip.isNaN and none.rows == rowsUsed
  doAssertRaises ValueError: discard panelStats(g.data, meta, ws, clipPercentile = -1)

block panel_stats_wide_dynamic_range:
  # 120 dB between rows: the relative error of the clip stays under 2⁻⁸.
  var g = cmpGather[float64](0, newSeq[float64](16), 300, dtMs = 2)
  var all: seq[float64]
  var r = initRand(3)
  for row in 0..<16:
    let level = 10.0 ^ (-6.0 * float64(row) / 15)
    for j in 0..<300:
      let v = r.gauss() * level
      g.data[row, j] = v
      all.add abs(v)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var ws = initWorkspace()
  ws.reserve(panelStatsWorkspace)
  for p in [1.0, 10, 30, 50, 80, 99]:
    let s = panelStats(g.data, meta, ws, clipPercentile = p)
    doAssert near(s.clip, sortPercentile(all, p), binRel), $p

block panel_rows_and_nan_rows:
  var g = panelGather(rows = 10, cols = 100, seed = 4)
  g.data[2, 17] = NaN
  g.data[7, 3] = Inf
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var ws = initWorkspace()
  ws.reserve(panelStatsWorkspace)
  let s = panelStats(g.data, meta, ws)
  doAssert s.rows == 8 and s.samples == 800, "the NaN and Inf rows are left out"
  doAssert s.rms.classify == fcNormal and s.clip.classify == fcNormal
  let sub = panelStats(g.data, meta, ws, rows = 3 .. 5)
  doAssert sub.rows == 3
  var used: seq[float64]
  for row in 3..5:
    for j in 0..<100: used.add abs(float64(g.data[row, j]))
  doAssert near(sub.clip, sortPercentile(used, 99), binRel)
  doAssert panelStats(g.data, meta, ws, rows = 20 .. 30).rows == 0
  var stats = newSeq[TraceStats](10)
  traceStatsInto(g.data, meta, stats)
  doAssert stats[2].rms.isNaN and stats[7].maxAbs == Inf
  doAssert stats[0].maxAbs == traceStats(g.data.toFlatSeq[0 ..< 100]).maxAbs
  var wrong = newSeq[TraceStats](3)
  doAssertRaises ValueError: traceStatsInto(g.data, meta, wrong)

block panel_percentile_among_zeros:
  # A muted panel with one spike: the p99 of |x| is 0, as a sort gives (bin 0
  # is not interpolated into a subnormal, which 1/clip would blow up).
  var g = cmpGather[float32](0, newSeq[float64](10), 100, dtMs = 2)
  g.data[4, 30] = 5
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var ws = initWorkspace()
  ws.reserve(panelStatsWorkspace)
  doAssert panelStats(g.data, meta, ws, clipPercentile = 99).clip == 0
  doAssert panelStats(g.data, meta, ws, clipPercentile = 99.95).clip > 0
  doAssert panelStats(g.data, meta, ws, clipPercentile = 100).clip == 5

block trace_stats_liveness:
  var g = panelGather(rows = 4, cols = 50, seed = 5)
  g.keys[1].status = tsDead
  g.setLive(2, 0)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var stats = newSeq[TraceStats](4)
  traceStatsInto(g.data, meta, stats)
  doAssert stats[1].rms.isNaN and stats[1].maxAbs.isNaN, "dead rows have none"
  doAssert stats[2] == TraceStats(), "an empty live row is 0"
  doAssert stats[0].rms > 0

block no_allocation:
  var g = panelGather(rows = 32, cols = 400, seed = 6)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var ws = initWorkspace()
  ws.reserve(panelStatsWorkspace + percentileWorkspace(400))
  var stats = newSeq[TraceStats](32)
  var s: PanelStats
  var p = 0.0
  let n = allocsIn:
    for _ in 0..<3:
      s = panelStats(g.data, meta, ws)
      traceStatsInto(g.data, meta, stats)
      p = percentileAbs(rowBlock(g.data).row(0), 99, ws)
  doAssert n == 0, $n & " allocations"
  doAssert s.rows == 32 and p > 0
