## Geometry QC (qc/geometry): synthetic bad geometry is flagged, good
## geometry is not.

import std/[assertions, math]
import processSEG
import parseSEG

proc station(sx, sy, rx, ry: float64, offset = NaN): TraceKeys =
  result = initTraceKeys()
  result.srcX = sx
  result.srcY = sy
  result.rcvX = rx
  result.rcvY = ry
  result.offset = if offset.isNaN: hypot(rx - sx, ry - sy) else: offset
  result.cmpX = (sx + rx) / 2
  result.cmpY = (sy + ry) / 2
  result.status = tsLive

block flags_on_synthetic_geometry:
  var keys: seq[TraceKeys]
  # 0..9: a good split spread, signed offsets, rounded to whole metres.
  for i in 0 ..< 10:
    let rx = 1000.0 + 25.3 * float64(i - 5)
    keys.add station(1000, 500, rx, 500, offset = round(rx - 1000))
  keys[5].offset = 0 # 10: (index 5) source and receiver coincide: zero offset
  # 10: header offset 400, coordinates 250 apart.
  keys.add station(0, 0, 250, 0, offset = 400)
  # 11: no offset.
  keys.add station(0, 0, 250, 0)
  keys[^1].offset = NaN
  # 12: no receiver coordinates, header offset kept.
  keys.add station(0, 0, 250, 0)
  keys[^1].rcvX = NaN
  # 13: a midpoint 30 m off.
  keys.add station(0, 0, 300, 400)
  keys[^1].cmpX += 30
  # 14: a dead trace with bad geometry: not checked.
  keys.add station(0, 0, 250, 0, offset = 999)
  keys[^1].status = tsDead
  # 15: within the relative tolerance (1 % of 2000 m = 20 m).
  keys.add station(0, 0, 2000, 0, offset = 2015)
  var counts = newSeq[int](keys.len)
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
  var checks = newSeq[GeometryCheck](meta.len)
  checkGeometryInto(meta, checks)
  for i in 0 ..< 10:
    if i == 5:
      doAssert checks[i].flags == {gfZeroOffset}
    else:
      doAssert gfOffsetMismatch notin checks[i].flags and
        gfMidpointMismatch notin checks[i].flags
      doAssert (gfNegativeOffset in checks[i].flags) == (i < 5)
      doAssert abs(checks[i].offsetError) <= 0.5
  doAssert checks[10].flags == {gfOffsetMismatch} and checks[10].offsetError == 150
  doAssert checks[10].distance == 250
  doAssert checks[11].flags == {gfNoOffset} and checks[11].offsetError.isNaN
  doAssert checks[12].flags == {gfNoCoordinates} and checks[12].distance.isNaN
  doAssert checks[13].flags == {gfMidpointMismatch} and
    abs(checks[13].midpointError - 30) < 1e-9
  doAssert checks[14].flags == {} and checks[14].distance.isNaN
  doAssert checks[15].flags == {}
  doAssert flaggedRows(checks) == @[5, 10, 11]
  doAssert flaggedRows(checks, badGeometry + {gfMidpointMismatch}) == @[5, 10, 11, 13]
  doAssert flaggedRows(checks, {gfNegativeOffset}) == @[0, 1, 2, 3, 4]
  # The view form checks every row, dead ones too.
  var all = newSeq[GeometryCheck](meta.len)
  checkGeometryInto(meta.rows, all)
  doAssert all[14].flags == {gfOffsetMismatch}
  # Tighter and looser tolerances.
  checkGeometryInto(meta, checks, GeometryQc(tolerance: 0.1, relTolerance: 0))
  doAssert gfOffsetMismatch in checks[15].flags and gfOffsetMismatch in checks[0].flags
  checkGeometryInto(meta, checks, GeometryQc(tolerance: 200, relTolerance: 0))
  doAssert checks[10].flags == {} and gfZeroOffset in checks[0].flags

block single_trace_and_errors:
  let c = checkGeometry(station(0, 0, 3, 4, offset = -5))
  doAssert c.flags == {gfNegativeOffset} and c.distance == 5 and c.offsetError == 0
  var keys = @[station(0, 0, 3, 4)]
  var counts = @[0]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
  var short: seq[GeometryCheck]
  doAssertRaises(ValueError): checkGeometryInto(meta, short)
  var one = newSeq[GeometryCheck](1)
  doAssertRaises(ValueError): checkGeometryInto(meta, one, GeometryQc(tolerance: -1))
  doAssertRaises(ValueError): checkGeometryInto(meta, one, GeometryQc(relTolerance: NaN))

block infinite_coordinates:
  # An infinite coordinate is not an unknown one: no header offset matches
  # the distance, and the relative tolerance must not grow with it.
  for rx in [Inf, -Inf]:
    let c = checkGeometry(station(0, 0, rx, 0, offset = 100))
    doAssert gfOffsetMismatch in c.flags and gfNoCoordinates notin c.flags, $c.flags
    doAssert c.distance == Inf and c.offsetError == -Inf
  # Both ends at infinity: the distance is undefined, which is no match.
  let both = checkGeometry(station(Inf, 0, Inf, 0, offset = 100))
  doAssert gfOffsetMismatch in both.flags, $both.flags
  # A midpoint at infinity is not the midpoint of finite ends.
  var k = station(0, 0, 300, 400)
  k.cmpY = Inf
  doAssert checkGeometry(k).flags == {gfMidpointMismatch}
  # With no relative tolerance either (0 × Inf is NaN).
  let strict = checkGeometry(station(0, 0, Inf, 0, offset = 100), GeometryQc(relTolerance: 0))
  doAssert gfOffsetMismatch in strict.flags
