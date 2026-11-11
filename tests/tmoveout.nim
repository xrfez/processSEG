## Moveout times (velocity/moveout): analytic values, the |x| convention on a
## split spread (signed and unsigned offsets), NaN offsets and t0, columns
## with a window and a t0, validation and no allocation.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

func near(a, b: float64, tol = 1e-12): bool = abs(a - b) <= tol * max(1, abs(b))

block analytic_values:
  doAssert lmoTime(1800, 1800, 0) == 1000
  doAssert lmoTime(-900, 1800, 20) == 520
  doAssert nmoTime(0, 2000, 800) == 800
  doAssert near(nmoTime(1500, 2000, 1000), 1250) # 3-4-5: sqrt(1000² + 750²)
  doAssert near(nmoTime(-1500, 2000, 1000), 1250)
  doAssert nmoTime(3000, 1500, 0) == 2000
  let l = linearMoveout(2500, t0Ms = 10)
  doAssert l.kind == mvLinear and l.velocity == 2500 and l.t0Ms == 10
  doAssert l.timeMs(500) == 210 and l.timeMs(-500) == 210
  let h = hyperbolicMoveout(2000, t0Ms = 1000)
  doAssert h.kind == mvHyperbolic and near(h.timeMs(1500), 1250)
  doAssert l.timeMs(NaN).isNaN and h.timeMs(NaN).isNaN

block split_spread_from_a_view:
  var g = shotGather[float32](1000, splitSpread(12, 25, 50), 100, dtMs = 2)
  var signed = newSeq[float64](g.len)
  let h = hyperbolicMoveout(1800, 400)
  let l = linearMoveout(1800, 0)
  moveoutTimesInto(g.rows, h, signed)
  for i in 0..<g.len:
    doAssert near(signed[i], hyperbolicMs(400, 1800, g.keys[i].offset))
  var lin = newSeq[float64](g.len)
  moveoutTimesInto(g.rows, l, lin)
  for i in 0..<g.len:
    doAssert near(lin[i], linearMs(0, 1800, g.keys[i].offset))
  doAssert g.keys[0].offset < 0 and g.keys[^1].offset > 0
  # Unsigned (derived) offsets give the same times.
  g.unsignOffsets
  var unsigned = newSeq[float64](g.len)
  moveoutTimesInto(g.rows, h, unsigned)
  doAssert unsigned == signed
  # The two sides of the spread are symmetric.
  for i in 0..<12: doAssert near(signed[i], signed[g.len - 1 - i])

block nan_offsets_and_columns:
  var g = cmpGather[float64](0, [0.0, 600, 1200, 1800], 500, dtMs = 4)
  g.keys[1].offset = NaN
  g.keys[2].t0Ms = -100
  g.keys[3].t0Ms = NaN
  let meta = initChunkMeta(g.rows, dtMs = 4, firstSample = 50)
  let l = linearMoveout(1200, 100)
  var t = newSeq[float64](4)
  moveoutTimesInto(meta.rows, l, t)
  doAssert t[0] == 100 and t[1].isNaN and t[2] == 1100 and t[3] == 1600
  var c = newSeq[float64](4)
  moveoutColumnsInto(meta, l, c)
  doAssert c[0] == 100 / 4 - 50 # t0Ms 0
  doAssert c[1].isNaN # NaN offset
  doAssert c[2] == (1100 + 100) / 4 - 50 # t0Ms −100
  doAssert c[3].isNaN # NaN t0Ms
  # A column maps back to the sample time.
  doAssert near(meta.sampleTimeMs(2, 0) + c[2] * 4, 1100)

block validation_and_allocation:
  for v in [0.0, -1, NaN, Inf]:
    doAssertRaises ValueError: discard linearMoveout(v)
    doAssertRaises ValueError: discard hyperbolicMoveout(v)
  doAssertRaises ValueError: discard linearMoveout(1000, t0Ms = NaN)
  doAssertRaises ValueError: discard hyperbolicMoveout(1000, t0Ms = -1)
  doAssert linearMoveout(1000, t0Ms = -50).timeMs(100) == 50 # a shift
  var g = cmpGather[float32](0, [100.0, 200], 10, dtMs = 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var short = newSeq[float64](1)
  doAssertRaises ValueError: moveoutTimesInto(g.rows, linearMoveout(1000), short)
  doAssertRaises ValueError: moveoutColumnsInto(meta, linearMoveout(1000), short)
  var t = newSeq[float64](2)
  let h = hyperbolicMoveout(2000)
  doAssert allocsIn(moveoutTimesInto(g.rows, h, t)) == 0
  var t2 = newSeq[float64](2)
  moveoutTimesInto(meta, h, t2)
  doAssert t2 == t
  doAssert allocsIn(moveoutColumnsInto(meta, h, t)) == 0

# ---------------------------------------------------- 6.2: velocity models --

func k(t, v: float64): VelocityKnot = VelocityKnot(tMs: t, vrms: v)

block hyperbola_from_a_function:
  let f = rmsVelocity([k(200, 1600), k(1000, 2000), k(2000, 2400)])
  let h = hyperbolicMoveout(f, t0Ms = 1000)
  doAssert h.velocity == f.vrmsAt(1000) and h.t0Ms == 1000
  doAssert near(h.timeMs(1500), sqrt(1000.0 ^ 2 + (1000 * 1500 / f.vrmsAt(
      1000)) ^ 2))
  let c = hyperbolicMoveout(constantVelocity(2000), t0Ms = 1000)
  doAssert near(c.timeMs(1500), 1250) # 3-4-5
  var d: VelocityFunction
  doAssertRaises ValueError: discard hyperbolicMoveout(d, 100)
  doAssertRaises ValueError: discard hyperbolicMoveout(f, -1)

block field_times_by_location:
  # Two CMPs, CDP 100 and 300, one 2D field between them.
  let a = constantVelocity(2000)
  let b = constantVelocity(3000)
  let field = initVelocityField([100.0, 300], [a, b], dtMs = 4, tMaxMs = 4000)
  var g = cmpGather[float32](0, [0.0, 1500, 1500, 1500, 1500], 100, dtMs = 4)
  for i, cdp in [100.0, 100, 300, 200, NaN]: g.keys[i].ensemble = cdp
  let meta = initChunkMeta(g.rows, dtMs = 4)
  var t = newSeq[float64](g.len)
  moveoutTimesInto(meta, field, t0Ms = 1000, t)
  doAssert t[0] == 1000
  doAssert near(t[1], 1250) # 1500 m at 2000 m/s
  doAssert near(t[2], sqrt(1000.0 ^ 2 + 500.0 ^ 2)) # at 3000 m/s
  let vMid = sqrt((2000.0 ^ 2 + 3000.0 ^ 2) / 2)
  doAssert near(t[3], nmoTime(1500, vMid, 1000))
  doAssert t[4].isNaN # no CDP
  # The rows form, and caller locations (a distance along the line).
  var t2 = newSeq[float64](g.len)
  moveoutTimesInto(g.rows, field, 1000, t2)
  doAssert t2[0 .. 3] == t[0 .. 3]
  moveoutTimesInto(g.rows, field, [300.0, 300, 100, 200, 300], 1000, t2)
  doAssert near(t2[1], sqrt(1000.0 ^ 2 + 500.0 ^ 2)) and near(t2[2], 1250) and
    not t2[4].isNaN
  # Columns: the t0 and window applied.
  var w = cmpGather[float32](0, [1500.0, 1500], 100, dtMs = 4)
  w.keys[0].ensemble = 100
  w.keys[1].ensemble = 300
  w.keys[1].t0Ms = -200
  let wm = initChunkMeta(w.rows, dtMs = 4, firstSample = 10)
  var c = newSeq[float64](2)
  moveoutColumnsInto(wm, field, 1000, c)
  doAssert near(c[0], 1250 / 4 - 10) and
    near(c[1], (sqrt(1000.0 ^ 2 + 500.0 ^ 2) + 200) / 4 - 10)
  doAssert allocsIn(moveoutTimesInto(meta, field, 1000, t)) == 0
  doAssert allocsIn(moveoutColumnsInto(wm, field, 1000, c)) == 0
  var short = newSeq[float64](2)
  doAssertRaises ValueError: moveoutTimesInto(meta, field, 1000, short)
  doAssertRaises ValueError: moveoutTimesInto(meta, field, -1, t)
  doAssertRaises ValueError: moveoutTimesInto(meta, field, NaN, t)
  doAssertRaises ValueError: moveoutTimesInto(g.rows, field, [1.0], 1000, t)

block field_times_3d:
  let fs = [constantVelocity(2000), constantVelocity(2000),
            constantVelocity(3000), constantVelocity(3000)]
  let field = initVelocityField([10.0, 20], [1.0, 2], fs, dtMs = 4, tMaxMs = 2000)
  var g = cmpGather[float64](0, [1500.0, 1500, 1500], 100, dtMs = 4)
  for i, (il, xl) in [(10.0, 1.0), (20.0, 2.0), (10.0, NaN)]:
    g.keys[i].inline = il
    g.keys[i].crossline = xl
  var t = newSeq[float64](3)
  moveoutTimesInto(g.rows, field, 1000, t)
  doAssert near(t[0], 1250) and near(t[1], sqrt(1000.0 ^ 2 + 500.0 ^ 2)) and t[2].isNaN
  doAssertRaises ValueError: moveoutTimesInto(g.rows, field, [1.0, 2, 3], 1000, t)

proc peakCol[T](g: SynthGather[T], r: int): int =
  var best = 0.0
  for j in 0..<g.cols:
    if abs(float64(g.data[r, j])) > best:
      best = abs(float64(g.data[r, j]))
      result = j

block lmo_flattens_and_returns:
  # 2000 m/s and offsets in steps of 20 m: shifts of 5 samples (2 ms) each.
  let offs = [0.0, 200, -400, 600, 800]
  for interp in [siLinear, siCubic]:
    var g = cmpGather[float64](0, offs, 400, dtMs = 2)
    g.addLinear(t0Ms = 100, velocity = 2000, freqHz = 25)
    let orig = g.data.clone
    let meta = initChunkMeta(g.rows, dtMs = 2)
    g.data.applyLmo(meta, velocity = 2000, biasMs = 20, interp = interp)
    for r in 0..<g.len: doAssert g.peakCol(r) == 60 # (100 + 20) / 2
    g.data.removeLmo(meta, velocity = 2000, biasMs = 20, interp = interp)
    # Integer shifts: the round trip is exact past what moved out at the top.
    for r in 0..<g.len:
      let lost = max(0, int(abs(offs[r]) / 2000 * 1000 / 2) - 10)
      for j in lost..<g.cols:
        doAssert abs(g.data[r, j] - orig[r, j]) <= 1e-12
  # A fractional shift: the peak still lands within one sample.
  var f = cmpGather[float32](0, [0.0, 333, 777], 400, dtMs = 2)
  f.addLinear(t0Ms = 100, velocity = 1900, freqHz = 20)
  let fm = initChunkMeta(f.rows, dtMs = 2)
  f.data.applyLmo(fm, velocity = 1900, interp = siCubic)
  for r in 0..<f.len: doAssert abs(f.peakCol(r) - 50) <= 1

block lmo_rows_left_alone:
  var g = cmpGather[float32](0, [400.0, 400, 400, 400, 400, 400], 50, dtMs = 2)
  for r in 0..<g.len:
    for j in 0..<g.cols: g.data[r, j] = float32(j + 1)
  g.keys[1].status = tsDead
  g.keys[2].offset = NaN
  g.keys[5].status = tsAuxiliary
  g.setLive(4, 30)
  for j in 30..<50: g.data[4, j] = 99 # a sentinel past the live end
  let orig = g.data.clone
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = [3])
  g.data.applyLmo(meta, velocity = 2000) # 200 ms: 100 samples, beyond the row
  for r in 0..<g.len:
    for j in 0..<g.cols:
      if r == 0 or (r == 4 and j < 30): doAssert g.data[r, j] == 0
      else: doAssert g.data[r, j] == orig[r, j]
  g.data.applyLmo(meta, velocity = 1e6) # tiny shift, row 0 of zeros stays 0
  doAssert allocsIn(g.data.applyLmo(meta, velocity = 4000)) == 0
  doAssert allocsIn(g.data.removeLmo(meta, velocity = 4000,
      interp = siCubic)) == 0
  for v in [0.0, -1, NaN]:
    doAssertRaises ValueError: g.data.applyLmo(meta, velocity = v)
  doAssertRaises ValueError: g.data.removeLmo(meta, velocity = 2000, biasMs = NaN)
  var wrong = zeros[float32](2, 50)
  doAssertRaises ValueError: wrong.applyLmo(meta, velocity = 2000)
