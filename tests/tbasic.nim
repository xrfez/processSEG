## Basic kernels (core/basic): exact values, live rows and live lengths, NaN
## containment, the pad-only taper, and no allocation.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

func near(a, b: float64, tol = 1e-9): bool = abs(a - b) <= tol

proc gather4(cols: int): SynthGather[float32] =
  cmpGather[float32](0, [0.0, 100, 200, 300], cols, dtMs = 2)

block demean_exact:
  var x = @[1.0, 2, 3, 4, 10]
  demean(x)
  doAssert x == @[-3.0, -2, -1, 0, 6]
  var f = @[5'f32, 5, 5]
  demean(f)
  doAssert f == @[0'f32, 0, 0]
  var empty: seq[float32]
  demean(empty)

block demean_large_dc_float32:
  # A sinusoid on a large DC, longer than the 8-lane blocks plus a tail.
  var x = zeroSignal[float32](1003)
  x.addSinusoid(dtMs = 2, freqHz = 10)
  var ref64 = newSeq[float64](x.len)
  for i, v in x: ref64[i] = float64(v)
  x.addDc(1e4)
  demean(x)
  var m = 0.0
  for v in ref64: m += v
  m /= float64(ref64.len)
  for i in 0..<x.len:
    doAssert near(float64(x[i]), ref64[i] - m, 2e-3), $i

block detrend_exact:
  var x = newSeq[float64](11)
  for i in 0..<x.len: x[i] = 3.0 + 0.5 * float64(i)
  detrend(x)
  for v in x: doAssert near(v, 0, 1e-12)
  # A line plus a symmetric bump: the line goes, the bump minus its mean stays.
  var y = @[0.0, 1, 2, 3, 4]
  var bump = @[0.0, 0, 5, 0, 0]
  for i in 0..<y.len: y[i] += bump[i]
  detrend(y)
  for i in 0..<y.len: doAssert near(y[i], bump[i] - 1.0, 1e-12), $i
  var one = @[7'f32]
  detrend(one)
  doAssert one == @[0'f32]

block detrend_removes_drift_float32:
  var x = zeroSignal[float32](2001)
  x.addDrift(dtMs = 2, slopePerS = 50)
  x.addDc(-300)
  detrend(x)
  for v in x: doAssert abs(v) < 1e-3

block scale_clip_polarity:
  var x = @[1'f32, -2, 3]
  scale(x, 0.5)
  doAssert x == @[0.5'f32, -1, 1.5]
  reversePolarity(x)
  doAssert x == @[-0.5'f32, 1, -1.5]
  var y = @[-5.0, -1, 0, 1, 5, NaN, Inf, -Inf]
  clip(y, -2, 3)
  doAssert y[0 .. 4] == @[-2.0, -1, 0, 1, 3]
  doAssert y[5].isNaN, "NaN stays NaN"
  doAssert y[6] == 3 and y[7] == -2
  var z = @[1.0]
  doAssertRaises ValueError: clip(z, 2, 1)
  doAssertRaises ValueError: clip(z, NaN, 1)
  clip(z, -Inf, Inf)
  doAssert z == @[1.0]

block cos_taper_and_pads:
  var ramp = newSeq[float64](3)
  cosTaperInto(ramp)
  for i in 0..<3:
    doAssert near(ramp[i], sin(PI / 2 * float64(i + 1) / 4) ^ 2)
  doAssert ramp[0] > 0 and ramp[2] < 1
  var x = @[1.0, 1, 1, 7, 8, 1, 1, 1]
  taperPads(x, ramp)
  doAssert x == @[ramp[0], ramp[1], ramp[2], 7, 8, ramp[2], ramp[1], ramp[0]],
    "the data in the middle is untouched"
  var short = @[1.0, 1, 1, 1, 1]
  doAssertRaises ValueError: taperPads(short, ramp)
  var exact = @[1.0, 1, 1, 1, 1, 1]
  taperPads(exact, ramp) # 2·3 == 6 fits
  var r32 = newSeq[float32](4)
  cosTaperInto(r32)
  doAssert near(float64(r32[3]), sin(PI / 2 * 4 / 5) ^ 2, 1e-7)

block block_forms_respect_liveness_and_live_length:
  var g = gather4(cols = 6)
  for r in 0..<4:
    for j in 0..<6: g.data[r, j] = float32(r + 1)
  g.setLive(0, 4) # zeroes columns 4, 5
  g.keys[1].status = tsDead
  g.keys[2].status = tsUnknown # unknown counts as live
  let failed = @[3]
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  g.data.scale(meta, 10)
  doAssert g.data[0, 0] == 10 and g.data[0, 3] == 10 and g.data[0, 4] == 0
  doAssert g.data[1, 0] == 2, "dead rows are untouched"
  doAssert g.data[2, 5] == 30
  doAssert g.data[3, 0] == 4, "failed rows are untouched"
  # Demean stops at the live length: the padding stays zero.
  g.data[0, 0] = 0
  g.data.demean(meta)
  doAssert near(float64(g.data[0, 0]), -7.5, 1e-6) and g.data[0, 4] == 0
  g.data.reversePolarity(meta)
  doAssert near(float64(g.data[0, 0]), 7.5, 1e-6) and g.data[1, 0] == 2
  g.data.clip(meta, -1, 1)
  doAssert g.data[0, 0] == 1 and g.data[1, 0] == 2
  doAssertRaises ValueError: g.data.clip(meta, 1, -1)
  for bad in [NaN, Inf, -Inf]:
    doAssertRaises ValueError: g.data.scale(meta, bad)

block detrend_block_on_a_window:
  # A trace of 20 samples decoded as columns 5 ..< 13; one row is cut short.
  var full = gather4(cols = 20)
  for r in 0..<4:
    for j in 0..<20: full.data[r, j] = float32(2 * j + 1)
  full.setLive(2, 9)
  var w = full.window(firstSample = 5, cols = 8)
  let meta = initChunkMeta(w.rows, dtMs = 2, firstSample = 5)
  w.data.detrend(meta)
  for j in 0..<8: doAssert abs(w.data[0, j]) < 1e-5
  doAssert meta.liveSamples(2, 8) == 4
  for j in 0..<4: doAssert abs(w.data[2, j]) < 1e-5
  for j in 4..<8: doAssert w.data[2, j] == 0, "past the trace's end"

block nan_stays_in_its_row:
  var g = gather4(cols = 8)
  for r in 0..<4:
    for j in 0..<8: g.data[r, j] = float32(j)
  g.data[1, 3] = NaN
  let meta = initChunkMeta(g.rows, dtMs = 2)
  g.data.detrend(meta)
  for j in 0..<8:
    doAssert g.data[1, j].isNaN
    doAssert abs(g.data[0, j]) < 1e-5 and abs(g.data[2, j]) < 1e-5

block float64_and_row_blocks:
  var buf = @[1.0, 2, 3, 10, 20, 30]
  let g = gather4(cols = 3)
  var keys = g.keys[0 .. 1]
  let counts = @[3, 3]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
  let b = rowBlock(buf, rows = 2, cols = 3)
  b.demean(meta)
  doAssert buf == @[-1.0, 0, 1, -10, 0, 10]
  let other = initChunkMeta(g.rows, dtMs = 2) # 4 rows
  doAssertRaises ValueError: b.demean(other)
  var t = zeros[float64](3, 3)
  doAssertRaises ValueError: t.demean(meta)

block no_allocation:
  var g = gather4(cols = 2001)
  for r in 0..<4: g.data[r, _] = g.data[r, _] +. 1'f32
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var ramp = newSeq[float32](100)
  cosTaperInto(ramp)
  let b = rowBlock(g.data)
  let n = allocsIn:
    for pass in 0..<10:
      g.data.demean(meta)
      g.data.detrend(meta)
      g.data.scale(meta, 1.5)
      g.data.clip(meta, -1, 1)
      g.data.reversePolarity(meta)
      taperPads(b.row(0), ramp)
  doAssert n == 0, "the kernels allocated " & $n & " times"
