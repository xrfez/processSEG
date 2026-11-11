## API review: a viewer loop written from the docs only (docs/integration.md,
## the module doc comments and parseSEG's prefetch docs).

import std/[assertions, math, os, strutils]
import parseSEG, processSEG
import utils/[demo, allocs]

type
  CacheEntry = object
    first, count: int
    raw: Tensor[float32]       # decoded once, kept for reprocessing
    failed: seq[int]           # the chunk's failedRows; the meta points into it
    processed: Tensor[float32] # allocated once per entry
    gen: uint64                # generation of the plan that filled `processed`
    meta: ChunkMeta

proc maxAbsDiff(a, b: Tensor[float32]): float64 =
  for r in 0 ..< a.shape[0]:
    for c in 0 ..< a.shape[1]:
      result = max(result, abs(a[r, c].float64 - b[r, c].float64))

# 1. open with keys, prefetch raw chunks (uniform) into a raw cache ----------
var f = openSeismic(demoDir() / "survey.sgy", keys = true)
doAssert f.len == 200
var cache: seq[CacheEntry]
block prefetch:
  var p = newPrefetcher[float32](f, 0 ..< f.len, chunkTraces = 64, uniform = true)
  var c: PrefetchChunk[float32]
  while p.next(c):
    # Check the adapter on the live chunk, while it is valid.
    let m = chunkMeta(c)
    doAssert m.len == c.count
    doAssert m.dtMs == 2.0
    doAssert m.firstSample == 0
    cache.add CacheEntry(first: c.first, count: c.count, raw: c.samples,
                         failed: c.failedRows)
  p.close()
doAssert cache.len == 4 # 64 + 64 + 64 + 8
# The chunk's view dies with the prefetcher, so the cache rebuilds its metas
# from the file, after `cache` has stopped growing (the meta points into
# `failed`).
for e in cache.mitems:
  e.meta = chunkMeta(f, e.first ..< e.first + e.count, e.failed)
  e.processed = zeros[float32](e.raw.shape[0], e.raw.shape[1])
  e.meta.checkShape(e.raw)
let ns = cache[0].raw.shape[1]
doAssert ns == 2001

# 2. a flow, compiled, applied with a pool, tagged with the generation -------
var flow = initPipeline([demeanOp(),
                         bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                         agcOp(windowMs = 500),
                         gainDbOp(6)])
var plan = flow.compile(float32, dtMs = cache[0].meta.dtMs, samples = ns)
var pool = newProcessPool(workers = 4)
let rawSnapshot = cache[0].raw.clone()
for e in cache.mitems:
  pool.apply(plan, e.raw, e.processed, e.meta)
  e.gen = plan.generation
block check_pool:
  doAssert maxAbsDiff(cache[0].raw, rawSnapshot) == 0, "raw cache untouched"
  doAssert maxAbsDiff(cache[0].raw, cache[0].processed) > 0
  # The pool gives the single-thread result.
  var ws = initWorkspace()
  ws.reserve(plan)
  var single = zeros[float32](cache[0].raw.shape[0], ns)
  plan.apply(cache[0].raw, single, cache[0].meta, ws)
  doAssert maxAbsDiff(single, cache[0].processed) == 0
  # AGC to 1, then +6 dB: live RMS near 2.
  var stats = newSeq[TraceStats](cache[0].meta.len)
  traceStatsInto(cache[0].processed, cache[0].meta, stats)
  for r, s in stats:
    if cache[0].meta.isLive(r):
      doAssert s.rms > 1.0 and s.rms < 3.0, $s

# 3. a slider moves: edit one op, recompile, reprocess only what is stale ----
block slider:
  flow[2].agc.windowMs = 300
  let old = plan.generation
  plan = flow.compile(float32, dtMs = cache[0].meta.dtMs, samples = ns)
  doAssert plan.generation > old
  # Only the visible chunks (0 and 1) are redrawn this frame.
  for i in 0 .. 1:
    if cache[i].gen != plan.generation:
      pool.apply(plan, cache[i].raw, cache[i].processed, cache[i].meta)
      cache[i].gen = plan.generation
  var stale = 0
  for e in cache:
    if e.gen != plan.generation: inc stale
  doAssert stale == 2
  for e in cache.mitems:
    if e.gen != plan.generation:
      pool.apply(plan, e.raw, e.processed, e.meta)
      e.gen = plan.generation
  # Steady state: a frame redone with the same plan allocates nothing.
  let e = addr cache[1]
  pool.apply(plan, e.raw, e.processed, e.meta) # warm (workers sized)
  let n = allocsIn:
    pool.apply(plan, e.raw, e.processed, e.meta)
  doAssert n == 0, "pool.apply allocated " & $n
  var ws = initWorkspace()
  ws.reserve(plan)
  let n1 = allocsIn:
    plan.apply(e.raw, e.processed, e.meta, ws)
  doAssert n1 == 0, "plan.apply allocated " & $n1
  # An invalid slider value fails at compile, naming the op.
  flow[2].agc.windowMs = -1
  try:
    discard flow.compile(float32, dtMs = 2, samples = ns)
    doAssert false
  except ValueError as err:
    doAssert "op 2" in err.msg, err.msg
  flow[2].agc.windowMs = 300

# 4. a zoom window decoded with the plan's margin ----------------------------
block zoom:
  # Linear ops only, so the zoom must match the whole trace within 1 % of
  # the peak (integration.md "Sample windows").
  let zflow = initPipeline([bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                            gainDbOp(6)])
  let zplan = zflow.compile(float32, dtMs = 2, samples = ns)
  let m = zplan.requiredMargin
  doAssert m > 0
  let a = 900
  let b = 1100
  let lo = max(0, a - m)
  let hi = min(ns - 1, b + m)
  let traces = 0 .. 63
  var failed: seq[int]
  var buf = zeros[float32](traces.len, hi - lo + 1)
  discard f.readTracesInto(traces, lo .. hi, buf, failed)
  let zmeta = chunkMeta(f, traces, lo .. hi, failed)
  doAssert zmeta.firstSample == lo
  doAssert abs(zmeta.sampleTimeMs(0, a - lo) - zmeta.t0Ms(0) - a.float64 * 2) < 1e-9
  var zout = zeros[float32](traces.len, hi - lo + 1)
  pool.apply(zplan, buf, zout, zmeta)
  # The whole traces through the same plan.
  var full = zeros[float32](traces.len, ns)
  pool.apply(zplan, cache[0].raw, full, cache[0].meta)
  var peak, err = 0.0
  for r in 0 ..< traces.len:
    for j in a .. b:
      peak = max(peak, abs(full[r, j].float64))
      err = max(err, abs(full[r, j].float64 - zout[r, j - lo].float64))
  var tracePeak = 0.0
  for r in 0 ..< traces.len:
    for j in 0 ..< ns: tracePeak = max(tracePeak, abs(full[r, j].float64))
  echo "zoom m=", m, " err/windowPeak=", err / peak, " err/tracePeak=", err / tracePeak
  doAssert peak > 0
  doAssert err <= 0.01 * tracePeak, "zoom error " & $(err / tracePeak)

# 5. moveout ticks over a chunk, tagged with the keys revision ---------------
block moveout:
  var g = openSeismic(demoDir() / "line.sg2", keys = true)
  let gm = chunkMeta(g, 0 ..< g.len)
  let first = linearMoveout(velocity = 1800, t0Ms = 0)
  var ticks = newSeq[float64](gm.len)
  var cols = newSeq[float64](gm.len)
  moveoutTimesInto(gm.rows, first, ticks)
  let tag = g.keysRevision
  let n = allocsIn:
    moveoutColumnsInto(gm, first, cols)
  doAssert n == 0
  for r in 0 ..< gm.len:
    let x = gm.keys(r).offset
    doAssert x.isNaN == ticks[r].isNaN
    if not x.isNaN:
      doAssert abs(ticks[r] - 1000 * abs(x) / 1800) < 1e-9
      doAssert abs(cols[r] - (ticks[r] - gm.t0Ms(r)) / gm.dtMs) < 1e-9
  doAssert not ticks[^1].isNaN, "line.sg2 should have offsets"
  doAssert tag == g.keysRevision # nothing changed the keys

# 6. display-gain stats and a spectrum of a selection ------------------------
block stats_spectrum:
  let e = addr cache[0]
  var stats = newSeq[TraceStats](e.meta.len)
  var ws = initWorkspace()
  ws.reserve(panelStatsWorkspace)
  let sp = initSpectrumPlan(float32, windowLen = 512, dtMs = e.meta.dtMs)
  ws.reserve(sp.workspaceSize)
  var amp, ampRaw, db = newSeq[float64](sp.bins)
  var ps: PanelStats
  var used, usedRaw: int
  # warm once, then a steady-state frame must not allocate
  traceStatsInto(e.processed, e.meta, stats)
  ps = panelStats(e.processed, e.meta, ws, clipPercentile = 99, rows = 10 .. 40)
  let n = allocsIn:
    traceStatsInto(e.processed, e.meta, stats)
    ps = panelStats(e.processed, e.meta, ws, clipPercentile = 99, rows = 10 .. 40)
    used = sp.averageAmplitudeInto(e.processed, e.meta, 500, amp, ws, rows = 10 .. 40)
    usedRaw = sp.averageAmplitudeInto(e.raw, e.meta, 500, ampRaw, ws, rows = 10 .. 40)
    toDbInto(amp, db)
  doAssert n == 0, "stats/spectrum allocated " & $n
  var live = 0
  for r in 10 .. 40:
    if e.meta.isLive(r): inc live
  doAssert ps.rows == live and used == live and usedRaw == live
  doAssert ps.clip > 0 and ps.clip <= ps.maxAbs and ps.rms > 0
  doAssert ps.samples == live * ns
  # 0 dB at the peak, and the band-pass removed what is below 4 Hz
  var peakDb = -Inf
  for v in db: peakDb = max(peakDb, v)
  doAssert peakDb == 0
  # The band-pass passes 8 .. 60 Hz: both spectra peak there (AGC adds some
  # low-frequency leakage, so no below-4 Hz comparison).
  var kp, kr = 0
  for k in 0 ..< sp.bins:
    if amp[k] > amp[kp]: kp = k
    if ampRaw[k] > ampRaw[kr]: kr = k
  echo "spectrum peak processed=", sp.binHz(kp), " Hz raw=", sp.binHz(kr), " Hz"
  doAssert sp.binHz(kp) in 4.0 .. 80.0 and sp.binHz(kr) in 4.0 .. 80.0

echo "tapireview: ok"
