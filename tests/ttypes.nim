## ChunkMeta (core/types): validation, liveness, live lengths, the time axis,
## NaN conventions and copying into a thread.

import std/[assertions, math, os, typedthreads]
import processSEG
import utils/[demo, synth]

proc liveGather(n, cols: int): SynthGather[float32] =
  var offsets = newSeq[float64](n)
  for i in 0..<n: offsets[i] = 100.0 * i.float64
  cmpGather[float32](0, offsets, cols, dtMs = 2)

block builds_from_synthetic_rows:
  let g = liveGather(4, 100)
  let meta = initChunkMeta(g.rows, dtMs = g.dtMs)
  doAssert meta.len == 4 and meta.dtMs == 2 and meta.firstSample == 0
  doAssert meta.failedCount == 0
  doAssert meta.keys(3).offset == 300, "keys are read from the view"
  meta.checkShape(g.data)

block keys_are_not_copied:
  var g = liveGather(2, 10)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  g.keys[1].offset = 999
  doAssert meta.keys(1).offset == 999, "the meta sees key edits in place"

block validation:
  let g = liveGather(3, 10)
  let rows = g.rows
  for bad in [0.0, -2.0, NaN, Inf, -Inf]:
    doAssertRaises ValueError:
      discard initChunkMeta(rows, dtMs = bad)
  doAssertRaises ValueError:
    discard initChunkMeta(rows, dtMs = 2, firstSample = -1)
  let outside = @[3]
  doAssertRaises ValueError:
    discard initChunkMeta(rows, dtMs = 2, failed = outside)
  let negative = @[-1]
  doAssertRaises ValueError:
    discard initChunkMeta(rows, dtMs = 2, failed = negative)
  let unsorted = @[2, 1]
  doAssertRaises ValueError:
    discard initChunkMeta(rows, dtMs = 2, failed = unsorted)
  let repeated = @[1, 1]
  doAssertRaises ValueError:
    discard initChunkMeta(rows, dtMs = 2, failed = repeated)

block validation_needs_keys:
  var f = openSeismic(demoDir() / "survey.sgy") # no stored keys
  defer: f.close()
  let rows = f.rowsOf(0 ..< 3)
  doAssert not rows.hasKeys
  doAssertRaises ValueError:
    discard initChunkMeta(rows, dtMs = 2)
  f.ensureKeys()
  doAssert initChunkMeta(f.rowsOf(0 ..< 3), dtMs = 2).len == 3,
    "a view made after ensureKeys has keys"

block empty_rows:
  let keys: seq[TraceKeys] = @[]
  let counts: seq[int] = @[]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 1)
  doAssert meta.len == 0 and not meta.isFailed(0)

block check_shape:
  let g = liveGather(3, 10)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  doAssertRaises ValueError:
    meta.checkShape(zeros[float32](2, 10))
  doAssertRaises ValueError:
    meta.checkShape(zeros[float32](30))
  let wide = zeros[float64](10, 3)
  doAssertRaises ValueError:
    meta.checkShape(wide.transpose) # a transposed view is not C-contiguous
  meta.checkShape(zeros[float64](3, 7)) # only the row count must agree

block liveness_rules:
  var g = liveGather(6, 10)
  g.keys[0].status = tsLive
  g.keys[1].status = tsUnknown
  g.keys[2].status = tsDead
  g.keys[3].status = tsAuxiliary
  g.keys[4].status = tsLive # failed below
  g.keys[5].status = tsUnknown # failed below
  let failed = @[4, 5]
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  doAssert meta.isLive(0), "tsLive"
  doAssert meta.isLive(1), "tsUnknown counts as live"
  doAssert not meta.isLive(2), "dead"
  doAssert not meta.isLive(3), "auxiliary"
  doAssert not meta.isLive(4) and not meta.isLive(5), "failed rows"
  doAssert meta.failedCount == 2 and meta.failedRow(0) == 4 and meta.failedRow(1) == 5
  for row in 0..<4: doAssert not meta.isFailed(row)

block failed_lookup:
  let g = liveGather(100, 4)
  var failed: seq[int]
  for r in countup(1, 99, 7): failed.add r
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  for r in 0..<100:
    doAssert meta.isFailed(r) == (r in failed), "row " & $r
  doAssert not meta.isFailed(-1) and not meta.isFailed(100)

block live_samples_without_window:
  var g = liveGather(3, 100)
  g.setLive(1, 60)
  g.setLive(2, 0)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  doAssert meta.liveSamples(0, 100) == 100
  doAssert meta.liveSamples(1, 100) == 60
  doAssert meta.liveSamples(2, 100) == 0
  doAssert meta.liveSamples(0, 80) == 80, "clamped to the block width"
  for r in 0..<3:
    doAssert meta.liveSamples(r, 100) == g.liveCols(r)

block live_samples_with_window:
  var full = liveGather(4, 1000)
  full.setLive(1, 500)
  full.setLive(2, 250)
  full.setLive(3, 0)
  let w = full.window(firstSample = 300, cols = 400)
  let meta = initChunkMeta(w.rows, dtMs = w.dtMs, firstSample = w.firstSample)
  doAssert w.rows.numSamples(0) == 1000, "the trace is longer than the block"
  doAssert meta.liveSamples(0, w.cols) == 400, "clamped to the block width"
  doAssert meta.liveSamples(1, w.cols) == 200, "500 − 300"
  doAssert meta.liveSamples(2, w.cols) == 0, "the trace ends before the window"
  doAssert meta.liveSamples(3, w.cols) == 0
  for r in 0..<4:
    doAssert meta.liveSamples(r, w.cols) == w.liveCols(r)

block time_axis:
  var full = liveGather(2, 1000)
  full.keys[1].t0Ms = 100
  full.addHyperbola(t0Ms = 900, velocity = 1e9, freqHz = 20)
  let w = full.window(firstSample = 300, cols = 200)
  let meta = initChunkMeta(w.rows, dtMs = w.dtMs, firstSample = w.firstSample)
  doAssert meta.t0Ms(0) == 0 and meta.t0Ms(1) == 100
  doAssert meta.sampleTimeMs(0, 0) == 600, "(300 + 0) · 2"
  doAssert meta.sampleTimeMs(1, 10) == 100 + 310 * 2
  doAssert w.data[0, 150] == 1 and meta.sampleTimeMs(0, 150) == 900,
    "the event sits where the time axis says"
  doAssert w.data[1, 100] == 1 and meta.sampleTimeMs(1, 100) == 900

block nan_conventions:
  var keys = @[initTraceKeys(), initTraceKeys()]
  keys[0].status = tsLive
  keys[0].t0Ms = 0
  keys[1].status = tsLive
  keys[1].t0Ms = NaN # an unknown SEG-D channel set
  let counts = @[10, 10]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
  doAssert meta.keys(0).offset.isNaN, "an offset without geometry stays NaN"
  doAssert meta.t0Ms(1).isNaN, "no default replaces an unknown t0"
  doAssert meta.sampleTimeMs(1, 3).isNaN, "derived times are NaN too"
  doAssert meta.sampleTimeMs(0, 3) == 6
  doAssert meta.isLive(1), "NaN keys do not change liveness"

block raises_none:
  # The accessors kernels use compile in a {.raises: [].} proc.
  proc liveTotal(m: ChunkMeta, cols: int): int {.raises: [].} =
    for row in 0..<m.len:
      if m.isLive(row): result += m.liveSamples(row, cols)
      discard m.keys(row).offset + m.sampleTimeMs(row, 0)
  var g = liveGather(3, 50)
  g.setLive(2, 20)
  let failed = @[0]
  doAssert liveTotal(initChunkMeta(g.rows, dtMs = 2, failed = failed), 50) == 70

# A meta is copied by value into a thread and read there (run under ASan by
# `nim asan`).
type ThreadArg = object
  meta: ChunkMeta
  cols: int
  live, samples: ptr int

proc worker(a: ThreadArg) {.thread.} =
  var live, samples = 0
  for row in 0..<a.meta.len:
    if a.meta.isLive(row):
      inc live
      samples += a.meta.liveSamples(row, a.cols)
      doAssert a.meta.keys(row).offset == 100.0 * row.float64
  a.live[] = live
  a.samples[] = samples

block meta_copied_into_a_thread:
  var g = liveGather(64, 200)
  for r in countup(0, 63, 4): g.setLive(r, 50)
  g.keys[9].status = tsDead
  let failed = @[3, 17]
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  var counts: array[4, int]
  var threads: array[2, Thread[ThreadArg]]
  for i in 0..1:
    createThread(threads[i], worker, ThreadArg(meta: meta, cols: g.cols,
      live: addr counts[2 * i], samples: addr counts[2 * i + 1]))
  joinThreads(threads)
  var wantLive, wantSamples = 0
  for row in 0..<meta.len:
    if meta.isLive(row):
      inc wantLive
      wantSamples += meta.liveSamples(row, g.cols)
  doAssert wantLive == 61
  for i in 0..1:
    doAssert counts[2 * i] == wantLive and counts[2 * i + 1] == wantSamples
