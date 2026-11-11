## CMP binning (stack/binning): header CDPs, a rotated 2D line, a rotated
## 3D grid (both handednesses) and header inline/crossline numbers binned
## from known synthetic geometry, bin locations round-tripping, edges, NaN,
## validation and no allocation.

import std/[assertions, math, random]
import processSEG
import utils/[allocs, synth]

func near(a, b: float64, tol = 1e-9): bool = abs(a - b) <= tol * max(1, abs(b))

proc keysAt(x, y: float64): TraceKeys =
  result = initTraceKeys()
  result.cmpX = x
  result.cmpY = y
  result.status = tsLive

block header_cdp:
  let b = cdpBinning(firstCdp = 101, count = 10, step = 2)
  doAssert b.kind == bnCdp and b.len == 10 and not b.is3D
  var k = initTraceKeys()
  for (cdp, want) in [(101.0, 0), (103, 1), (119, 9), (121, -1), (99, -1),
                      (102, 1), (101.9, 0), (NaN, -1)]:
    k.ensemble = cdp
    doAssert b.binOf(k) == want, $cdp
  doAssert b.location(3).cdp == 107 and b.location(3).x.isNaN

block split_spread_shots_on_a_line:
  # Shots every 50 m along x into a split spread at 25 m: midpoints every
  # 12.5 m, binned by projection onto the line and by header CDP.
  var keys: seq[TraceKeys]
  var numbers: seq[float64]
  for s in 0..<8:
    let g = shotGather[float32](1000 + 50 * float64(s), splitSpread(12, 25,
        12.5), 4, 2)
    for k in g.keys:
      var kk = k
      let cdp = (kk.cmpX - 706.25) / 12.5 + 1           # CDP 1 at x = 706.25
      doAssert cdp == round(cdp)
      kk.ensemble = cdp
      keys.add kk
      numbers.add cdp
  let counts = newSeq[int](keys.len)
  let rows = rowsOf(keys, counts)
  let line = lineBinning(706.25, 0, azimuthDeg = 90, binSize = 12.5, count = 200)
  let hdr = cdpBinning(1, 200)
  var a = newSeq[int](keys.len)
  var c = newSeq[int](keys.len)
  binsInto(rows, line, a)
  binsInto(rows, hdr, c)
  for i in 0..<keys.len:
    doAssert a[i] == int(numbers[i]) - 1 and c[i] == a[i]
    doAssert line.location(a[i]).cdp == numbers[i]
    doAssert near(line.location(a[i]).x, keys[i].cmpX)

block rotated_line:
  let az = 30.0
  let (ux, uy) = (sin(degToRad(az)), cos(degToRad(az)))
  let b = lineBinning(500, 800, az, binSize = 10, count = 50, firstCdp = 1001,
    maxOffLine = 15)
  var rng = initRand(3)
  for i in 0..<50:
    let along = float64(i) * 10 + rng.rand(-4.9 .. 4.9)
    let off = rng.rand(-14.9 .. 14.9)
    let k = keysAt(500 + along * ux + off * uy, 800 + along * uy - off * ux)
    doAssert b.binOf(k) == i, $i
    let loc = b.location(i)
    doAssert loc.cdp == 1001 + float64(i)
    doAssert near(loc.x, 500 + float64(i) * 10 * ux) and near(loc.y, 800 +
        float64(i) * 10 * uy)
  doAssert b.binOf(keysAt(500 + 20 * uy, 800 - 20 * ux)) == -1 # 20 m off the line
  doAssert b.binOf(keysAt(500 - 6 * ux, 800 - 6 * uy)) == -1 # before bin 0
  doAssert b.binOf(keysAt(500 + 5 * ux, 800 + 5 * uy)) == 1 # half-way: higher
  doAssert b.binOf(keysAt(NaN, 800)) == -1

block rotated_grid:
  for clockwise in [true, false]:
    let az = 120.0
    let b = gridBinning(10_000, 20_000, az, crosslineSpacing = 25,
      inlineSpacing = 12.5, inlines = 6, crosslines = 9, firstInline = 1001,
      firstCrossline = 2001, clockwise = clockwise)
    doAssert b.kind == bnGrid and b.len == 54 and b.inlines == 6 and
        b.crosslines == 9 and b.is3D
    let (ux, uy) = (sin(degToRad(az)), cos(degToRad(az)))
    let va = degToRad(az + (if clockwise: 90.0 else: -90.0))
    let (vx, vy) = (sin(va), cos(va))
    var rng = initRand(5)
    for i in 0..<6:
      for j in 0..<9:
        let a = float64(j) * 25 + rng.rand(-12.4 .. 12.4)
        let c = float64(i) * 12.5 + rng.rand(-6.2 .. 6.2)
        let k = keysAt(10_000 + a * ux + c * vx, 20_000 + a * uy + c * vy)
        let bin = b.binOf(k)
        doAssert bin == i * 9 + j, $i & " " & $j & " " & $bin
        let loc = b.location(bin)
        doAssert loc.inline == 1001 + float64(i) and loc.crossline == 2001 +
            float64(j) and loc.cdp.isNaN
        doAssert near(loc.x, 10_000 + float64(j) * 25 * ux + float64(i) * 12.5 * vx)
        doAssert near(loc.y, 20_000 + float64(j) * 25 * uy + float64(i) * 12.5 * vy)
    # One bin past the last inline, and before the first crossline.
    doAssert b.binOf(keysAt(10_000 + 6 * 12.5 * vx, 20_000 + 6 * 12.5 * vy)) == -1
    doAssert b.binOf(keysAt(10_000 - 13 * ux, 20_000 - 13 * uy)) == -1

block grid_handedness_pinned:
  # Azimuth 0: crosslines increase to the north (+y); clockwise inlines to
  # the east (+x), anticlockwise to the west.
  let cw = gridBinning(0, 0, 0, crosslineSpacing = 25, inlineSpacing = 12.5,
    inlines = 2, crosslines = 2)
  let ccw = gridBinning(100, 0, 0, 25, 12.5, 2, 2, clockwise = false)
  doAssert cw.binOf(keysAt(0, 25)) == 1 and cw.binOf(keysAt(12.5, 0)) == 2
  doAssert cw.binOf(keysAt(-12.5, 0)) == -1
  doAssert ccw.binOf(keysAt(87.5, 0)) == 2 and ccw.binOf(keysAt(112.5, 0)) == -1
  # Azimuth 90: crosslines to the east, clockwise inlines to the south.
  let east = gridBinning(0, 0, 90, 25, 12.5, 2, 2)
  doAssert east.binOf(keysAt(25, 0)) == 1 and east.binOf(keysAt(0, -12.5)) == 2
  doAssert abs(east.location(3).x - 25) < 1e-12 and abs(east.location(3).y +
      12.5) < 1e-12

block header_inline_crossline:
  let b = inlineCrosslineBinning(1001, 3, 10, 4, inlineStep = 2,
    crosslineStep = 5)
  doAssert b.kind == bnInlineCrossline and b.len == 12
  var k = initTraceKeys()
  for (il, xl, want) in [(1001.0, 10.0, 0), (1003, 15, 5), (1005, 25, 11),
                         (1007, 10, -1), (1001, 30, -1), (NaN, 10, -1)]:
    k.inline = il
    k.crossline = xl
    doAssert b.binOf(k) == want, $il & " " & $xl
  let loc = b.location(5)
  doAssert loc.inline == 1003 and loc.crossline == 15 and loc.x.isNaN

block validation_and_allocation:
  doAssertRaises ValueError: discard cdpBinning(NaN, 10)
  doAssertRaises ValueError: discard cdpBinning(1, 0)
  doAssertRaises ValueError: discard cdpBinning(1, 10, step = 0)
  doAssertRaises ValueError: discard lineBinning(0, 0, 0, 0, 10)
  doAssertRaises ValueError: discard lineBinning(0, 0, NaN, 10, 10)
  doAssertRaises ValueError: discard lineBinning(0, 0, 0, 10, 10,
      maxOffLine = 0)
  doAssertRaises ValueError: discard gridBinning(0, 0, 0, 25, -1, 2, 2)
  doAssertRaises ValueError: discard gridBinning(0, 0, 0, 25, 25, 0, 2)
  doAssertRaises ValueError: discard gridBinning(0, 0, 0, 25, 25, high(int), 2)
  doAssertRaises ValueError: discard inlineCrosslineBinning(1, 2, 1, 2,
      crosslineStep = Inf)
  let b = cdpBinning(1, 4)
  doAssertRaises ValueError: discard b.location(4)
  doAssertRaises ValueError: discard b.location(-1)
  var g = cmpGather[float32](0, [100.0, 200], 4, 2, cdp = 2)
  var bins = newSeq[int](2)
  doAssert allocsIn(binsInto(g.rows, b, bins)) == 0
  doAssert bins == @[1, 1]
  let meta = initChunkMeta(g.rows, dtMs = 2)
  bins = @[0, 0]
  binsInto(meta, b, bins)
  doAssert bins == @[1, 1]
  var short = newSeq[int](1)
  doAssertRaises ValueError: binsInto(g.rows, b, short)
