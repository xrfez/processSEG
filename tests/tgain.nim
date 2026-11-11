## Gain (gain/gain): fixed dB, normalization per trace and per panel, AGC
## against a brute-force reference (edges, 120 dB of dynamic range, both
## modes), decaying input, dead and zero traces, live lengths, gain removal,
## and no allocation.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

func near(a, b, rel: float64): bool =
  abs(a - b) <= rel * max(abs(a), abs(b))

proc gather(rows, cols: int): SynthGather[float64] =
  cmpGather[float64](0, newSeq[float64](rows), cols, dtMs = 2)

block fixed_gain_in_db:
  var x = @[1'f32, -2, 0.5]
  scaleDb(x, 20 * log10(2.0))
  doAssert x == @[2'f32, -4, 1]
  doAssert near(dbToFactor(-6), 0.501187233627, 1e-11)
  var g = gather(3, 4)
  for r in 0..<3:
    for j in 0..<4: g.data[r, j] = 1
  g.keys[1].status = tsDead
  g.setLive(2, 2)
  g.data[2, 2] = 0 # setLive zeroed it; keep a marker past the end
  let meta = initChunkMeta(g.rows, dtMs = 2)
  g.data.scaleDb(meta, 20)
  doAssert near(g.data[0, 3], 10, 1e-12) and g.data[1, 0] == 1
  doAssert near(g.data[2, 1], 10, 1e-12) and g.data[2, 2] == 0
  for bad in [NaN, Inf, -Inf]:
    doAssertRaises ValueError: g.data.scaleDb(meta, bad)

block normalize_per_trace:
  var ws = initWorkspace()
  for stat in NormStat:
    let plan = initNormalizePlan(stat, samples = 100, target = 2, percentile = 50)
    ws.reserve(plan.workspaceSize)
    var x = newSeq[float64](100)
    x.addNoise(1, seed = 3)
    x.addSpike(10, amp = 30)
    normalize(x, plan, ws)
    let s = traceStats(x)
    var w2 = initWorkspace()
    w2.reserve(percentileWorkspace(100))
    let got =
      case stat
      of normMax: s.maxAbs
      of normRms: s.rms
      of normPercentile: percentileAbs(x, 50, w2)
    doAssert near(got, 2, 1e-12), $stat & ": " & $got
  # Zero and NaN traces are left as they are.
  let plan = initNormalizePlan(normRms, samples = 10)
  var z = newSeq[float32](10)
  normalize(z, plan, ws)
  doAssert z == newSeq[float32](10)
  var n = @[1.0, NaN, 2]
  normalize(n, plan, ws)
  doAssert n[0] == 1 and n[2] == 2
  var tiny = @[1e-320, -1e-320] # target / level overflows: left alone
  normalize(tiny, plan, ws)
  doAssert tiny == @[1e-320, -1e-320]
  doAssertRaises ValueError: discard initNormalizePlan(normMax, samples = 0)
  doAssertRaises ValueError: discard initNormalizePlan(normMax, 10, target = 0)
  doAssertRaises ValueError: discard initNormalizePlan(normMax, 10, target = NaN)
  doAssertRaises ValueError: discard initNormalizePlan(normPercentile, 10, percentile = 120)

block normalize_block:
  var g = gather(5, 50)
  for r in 0..<5:
    for j in 0..<50: g.data[r, j] = float64(r + 1) * sin(float64(j) * 0.3)
  g.keys[1].status = tsDead
  g.setLive(3, 20)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var ws = initWorkspace()
  let trace = initNormalizePlan(normMax, samples = 50)
  var a = g.data.clone()
  a.normalize(meta, trace, ws)
  for r in [0, 2, 3]:
    var m = 0.0
    for j in 0..<50: m = max(m, abs(a[r, j]))
    doAssert near(m, 1, 1e-12), $r
  doAssert a[1, 5] == g.data[1, 5], "dead rows untouched"
  for j in 20..<50: doAssert a[3, j] == 0, "padding untouched"
  for stat in NormStat:
    let panel = initNormalizePlan(stat, samples = 50, scope = perPanel, target = 3)
    ws.reserve(max(panel.workspaceSize, panelStatsWorkspace))
    var b = g.data.clone()
    b.normalize(meta, panel, ws)
    let p = panelStats(b, meta, ws, clipPercentile = 99)
    let got =
      case stat
      of normMax: p.maxAbs
      of normRms: p.rms
      of normPercentile: p.clip
    doAssert near(got, 3, 1 / 128), $stat & ": " & $got
    # One factor: the row ratios are kept.
    doAssert near(b[2, 5] / b[0, 5], 3, 1e-12)
    doAssert b[1, 5] == g.data[1, 5]
  var wide = gather(5, 60)
  doAssertRaises ValueError:
    wide.data.normalize(initChunkMeta(wide.rows, dtMs = 2), trace, ws)

proc refAgc(x: seq[float64], h: int, mode: AgcMode, target, epsilon: float64): (
    seq[float64], seq[float64]) =
  ## Direct window sums in float64.
  let n = x.len
  var sq = 0.0
  for v in x: sq += v * v
  let eps = epsilon * sqrt(sq / float64(n))
  var y, g = newSeq[float64](n)
  for i in 0..<n:
    let lo = max(0, i - h)
    let e = min(n, i + h + 1)
    var s = 0.0
    for k in lo..<e: s += (if mode == agcRms: x[k] * x[k] else: abs(x[k]))
    let level = (if mode == agcRms: sqrt(s / float64(e - lo)) else: s / float64(e - lo))
    g[i] = target / (level + eps)
    y[i] = x[i] * g[i]
  (y, g)

block agc_matches_reference:
  var ws = initWorkspace()
  for mode in AgcMode:
    for (n, windowMs) in [(1, 6.0), (2, 6.0), (5, 6.0), (7, 6.0), (13, 6.0),
        (14, 6.0), (100, 6.0), (101, 50.0), (500, 104.0), (2001, 500.0), (30, 200.0)]:
      let plan = initAgcPlan(windowMs, dtMs = 2, samples = n, mode = mode, target = 2)
      ws.reserve(plan.workspaceSize)
      var x = newSeq[float64](n)
      x.addNoise(1, seed = n)
      x.addSpike(n div 3, amp = 20)
      let (want, wantG) = refAgc(x, plan.requiredMargin, mode, 2, 1e-9)
      var y = x
      var g = newSeq[float64](n)
      agc(y, plan, ws, g)
      var z = x
      agc(z, plan, ws)
      for i in 0..<n:
        doAssert near(y[i], want[i], 1e-12), $mode & " n " & $n & " at " & $i
        doAssert near(g[i], wantG[i], 1e-12)
        doAssert z[i] == y[i]
  # float32 runs its gain step in float32 (on means scaled to the trace
  # level): within a few float32 ulps of the float64 reference.
  for mode in AgcMode:
    for n in [3, 40, 777]:
      let plan = initAgcPlan(30, dtMs = 2, samples = n, mode = mode)
      ws.reserve(plan.workspaceSize)
      var x = newSeq[float64](n)
      x.addNoise(1e-20, seed = n) # far from 1: the scaling keeps float32 safe
      let (want, wantG) = refAgc(x, plan.requiredMargin, mode, 1, 1e-9)
      var f = newSeq[float32](n)
      for i in 0..<n: f[i] = float32(x[i])
      var g = newSeq[float32](n)
      agc(f, plan, ws, g)
      for i in 0..<n:
        doAssert near(float64(f[i]), want[i], 1e-5), $mode & " " & $n & " " & $i
        doAssert near(float64(g[i]), wantG[i], 1e-5)
  let p = initAgcPlan(500, dtMs = 2, samples = 2001)
  doAssert p.requiredMargin == 125 and p.windowSamples == 251 and p.windowMs == 502
  doAssert initAgcPlan(4, dtMs = 2, samples = 10).windowSamples == 3
  doAssertRaises ValueError: discard initAgcPlan(1.9, dtMs = 2, samples = 10)
  doAssertRaises ValueError: discard initAgcPlan(0, dtMs = 2, samples = 10)
  doAssertRaises ValueError: discard initAgcPlan(NaN, dtMs = 2, samples = 10)
  doAssertRaises ValueError: discard initAgcPlan(100, dtMs = 0, samples = 10)
  doAssertRaises ValueError: discard initAgcPlan(100, dtMs = 2, samples = 0)
  doAssertRaises ValueError: discard initAgcPlan(100, 2, 10, target = -1)
  doAssertRaises ValueError: discard initAgcPlan(100, 2, 10, epsilon = 0)

block agc_flattens_a_decay:
  # A 25 Hz sinusoid decaying 60 dB over 4 s comes out at constant amplitude.
  const n = 2001
  let plan = initAgcPlan(200, dtMs = 2, samples = n)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var x = newSeq[float32](n)
  x.addSinusoid(dtMs = 2, freqHz = 25)
  for i in 0..<n: x[i] *= float32(10.0 ^ (-3.0 * float64(i) / float64(n)))
  agc(x, plan, ws)
  # Sliding RMS over 200 ms windows: target 1 everywhere, within 2 %.
  for c in countup(50, n - 101, 50): # whole AGC windows (cut ones hold part of a period)
    var s = 0.0
    for k in c - 50 .. c + 50: s += float64(x[k]) ^ 2
    let r = sqrt(s / 101)
    doAssert abs(r - 1) < 0.02, $c & ": " & $r

block agc_precision_at_120_db:
  # A loud event, then a tail 120 dB down: the tail's gain matches the
  # direct sums to 1e-12, though the running window passed the loud part.
  const n = 3000
  let plan = initAgcPlan(100, dtMs = 2, samples = n, epsilon = 1e-12)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var x = newSeq[float64](n)
  x.addNoise(1, seed = 11)
  for i in 0..<n:
    x[i] *= (if i < 1000: 1.0 else: 1e-6)
  let (want, _) = refAgc(x, plan.requiredMargin, agcRms, 1, 1e-12)
  var y = x
  agc(y, plan, ws)
  for i in 0..<n:
    doAssert near(y[i], want[i], 1e-12), $i
  var tail = 0.0
  for i in 1200..<n: tail += y[i] * y[i]
  doAssert abs(sqrt(tail / float64(n - 1200)) - 1) < 0.05
  # float32 data: the float64 sums keep the tail at unit level too.
  var f = newSeq[float32](n)
  for i in 0..<n: f[i] = float32(x[i])
  agc(f, plan, ws)
  var ft = 0.0
  for i in 1200..<n: ft += float64(f[i]) ^ 2
  doAssert abs(sqrt(ft / float64(n - 1200)) - 1) < 0.05

block agc_block_rows:
  var g = gather(5, 300)
  for r in 0..<5:
    var x = newSeq[float64](300)
    x.addNoise(float64(r + 1), seed = r)
    for j in 0..<300: g.data[r, j] = x[j]
  g.keys[1].status = tsDead
  g.setLive(2, 120)
  for j in 0..<300: g.data[4, j] = 0 # an all-zero live trace
  g.data[3, 50] = NaN
  let failed = @[0]
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  let plan = initAgcPlan(40, dtMs = 2, samples = 300)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var a = g.data.clone()
  var gains = zeros[float64](5, 300)
  a.agc(meta, plan, ws, gains)
  doAssert a[0, 7] == g.data[0, 7] and a[1, 7] == g.data[1, 7], "failed, dead untouched"
  for j in 0..<300:
    doAssert gains[0, j] == 1 and gains[1, j] == 1 and gains[4, j] == 1
    doAssert a[4, j] == 0, "dead and zero traces stay zero"
  var live2 = newSeq[float64](120)
  for j in 0..<120: live2[j] = g.data[2, j]
  agc(live2, plan, ws)
  for j in 0..<120: doAssert a[2, j] == live2[j], "AGC stops at the live length"
  for j in 120..<300: doAssert a[2, j] == 0 and gains[2, j] == 1
  for j in 0..<300: # a NaN row is left as it is, like normalize does
    doAssert a[3, j] == g.data[3, j] or (j == 50 and a[3, j].isNaN)
    doAssert gains[3, j] == 1
  # The plain form gives the same samples; removing the gain restores them.
  var b = g.data.clone()
  b.agc(meta, plan, ws)
  for r in [0, 1, 2, 3, 4]:
    for j in 0..<300: doAssert b[r, j] == a[r, j] or (r == 3 and j == 50)
  a.removeGain(meta, gains)
  for r in [0, 1, 2, 4]:
    for j in 0..<300: doAssert near(a[r, j], g.data[r, j], 1e-12), $r & " " & $j
  var wrong = zeros[float64](5, 299)
  doAssertRaises ValueError: a.agc(meta, plan, ws, wrong)
  doAssertRaises ValueError: a.removeGain(meta, wrong)
  let short = initAgcPlan(40, dtMs = 2, samples = 299)
  doAssertRaises ValueError: a.agc(meta, short, ws)

block agc_zoom_margin:
  # A window decoded with requiredMargin each side matches the whole trace's
  # AGC (up to ε, which comes from the window's RMS: tiny here).
  const n = 1500
  var full = newSeq[float64](n)
  full.addNoise(1, seed = 2)
  full.addRicker(dtMs = 2, centreMs = 1500, freqHz = 20, amp = 10)
  let plan = initAgcPlan(300, dtMs = 2, samples = n)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var whole = full
  agc(whole, plan, ws)
  let m = plan.requiredMargin
  let (a, b) = (600, 900)
  var win = full[a - m ..< b + m]
  agc(win, plan, ws)
  for j in a..<b: doAssert near(win[j - a + m], whole[j], 1e-7), $j

block factors_that_do_not_fit_float32:
  # A factor finite in float64 but Inf in float32 would make zeros NaN: the
  # trace is left as it is (review 4.4 #2).
  var ws = initWorkspace()
  let big = initNormalizePlan(normMax, samples = 4, target = 1e20)
  var x = @[1e-20'f32, 0, 0, 0]
  normalize(x, big, ws)
  doAssert x == @[1e-20'f32, 0, 0, 0]
  for (target, level) in [(1e30, 1e-10), (1.0, 1e-39)]:
    let plan = initAgcPlan(6, dtMs = 2, samples = 8, target = target)
    ws.reserve(plan.workspaceSize)
    var y = @[float32(level), 0, 0, 0, float32(level), 0, 0, 0]
    let before = y
    var g = newSeq[float32](8)
    agc(y, plan, ws, g)
    doAssert y == before, $target
    for v in g: doAssert v == 1
  # float64 data has the range: the same AGC works there.
  let plan = initAgcPlan(6, dtMs = 2, samples = 8, target = 1e30)
  ws.reserve(plan.workspaceSize)
  var z = @[1e-10, 0, 0, 0, 1e-10, 0, 0, 0]
  agc(z, plan, ws)
  for v in z: doAssert v.classify in {fcNormal, fcZero}
  doAssert z[0] > 1e29

block plan_dt_must_match_the_rows:
  var g = gather(2, 50)
  let meta = initChunkMeta(g.rows, dtMs = 4)
  let plan = initAgcPlan(40, dtMs = 2, samples = 50)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  doAssertRaises ValueError: g.data.agc(meta, plan, ws)
  doAssert plan.target == 1 and plan.epsilon == 1e-9
  doAssert initNormalizePlan(normPercentile, 10, percentile = 95).percentile == 95

block no_allocation:
  var g = gather(16, 500)
  for r in 0..<16:
    for j in 0..<500: g.data[r, j] = sin(float64(j + r) * 0.1)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let plan = initAgcPlan(100, dtMs = 2, samples = 500)
  let np = initNormalizePlan(normPercentile, samples = 500)
  let pp = initNormalizePlan(normPercentile, samples = 500, scope = perPanel)
  var ws = initWorkspace()
  ws.reserve(max(plan.workspaceSize, max(np.workspaceSize, pp.workspaceSize)))
  var gains = zeros[float64](16, 500)
  let n = allocsIn:
    for _ in 0..<3:
      g.data.agc(meta, plan, ws)
      g.data.agc(meta, plan, ws, gains)
      g.data.removeGain(meta, gains)
      g.data.normalize(meta, np, ws)
      g.data.normalize(meta, pp, ws)
      g.data.scaleDb(meta, 3)
  doAssert n == 0, $n & " allocations"

block scale_db_overflow_is_refused:
  # 800 dB: a factor of 1e40, Inf in float32 (zeros would become NaN).
  var g = cmpGather[float32](0, [100.0, 200], 10, dtMs = 2)
  let meta = initChunkMeta(g.rows, 2)
  doAssertRaises ValueError: g.data.scaleDb(meta, 800)
  var h = cmpGather[float64](0, [100.0, 200], 10, dtMs = 2)
  let meta64 = initChunkMeta(h.rows, 2)
  h.data.scaleDb(meta64, 800)
  for v in h.data: doAssert v == 0
