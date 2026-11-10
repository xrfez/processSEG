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
  doAssert c[0] == 100 / 4 - 50          # t0Ms 0
  doAssert c[1].isNaN                    # NaN offset
  doAssert c[2] == (1100 + 100) / 4 - 50 # t0Ms −100
  doAssert c[3].isNaN                    # NaN t0Ms
  # A column maps back to the sample time.
  doAssert near(meta.sampleTimeMs(2, 0) + c[2] * 4, 1100)

block validation_and_allocation:
  for v in [0.0, -1, NaN, Inf]:
    doAssertRaises ValueError: discard linearMoveout(v)
    doAssertRaises ValueError: discard hyperbolicMoveout(v)
  doAssertRaises ValueError: discard linearMoveout(1000, t0Ms = NaN)
  var g = cmpGather[float32](0, [100.0, 200], 10, dtMs = 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var short = newSeq[float64](1)
  doAssertRaises ValueError: moveoutTimesInto(g.rows, linearMoveout(1000), short)
  doAssertRaises ValueError: moveoutColumnsInto(meta, linearMoveout(1000), short)
  var t = newSeq[float64](2)
  let h = hyperbolicMoveout(2000)
  doAssert allocsIn(moveoutTimesInto(g.rows, h, t)) == 0
  doAssert allocsIn(moveoutColumnsInto(meta, h, t)) == 0
