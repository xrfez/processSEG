## Survey fold maps (`qc/fold`): an orthogonal synthetic survey at several
## azimuths gives back its natural grid exactly, with every midpoint at a bin
## centre; overrides; errors. Pre-plot fold of an orthogonal swath with an X
## template: brute-force counts per bin (fold, offset classes, azimuth
## sectors, offset range), the nominal fold in the full-fold area, X and O
## alike. Morgantown (skipped when its files are absent).

import std/[assertions, math, os, strutils]
when defined(release) or defined(danger): import std/times
import parseSEG
import processSEG/stack/binning
import processSEG/qc/fold
import processSEG/core/types
import utils/allocs

proc close(a, b: float64, tol = 1e-6): bool = abs(a - b) <= tol

type Ortho = object
  azimuthDeg, x0, y0: float64
  receiverLines, stations: int
  interval, lineSpacing: float64             # receivers: along (a) and across (c) the azimuth
  sourceLines, sourcePoints: int
  sourceInterval, sourceLineSpacing: float64 # sources: across (c) and along (a)
  sourceA0, sourceC0: float64                # first source line (a), first source point (c)

func at(o: Ortho, a, c: float64): (float64, float64) =
  ## Frame position (a along the azimuth, c 90° clockwise of it) → (x, y).
  let t = degToRad(o.azimuthDeg)
  (o.x0 + a * sin(t) + c * cos(t), o.y0 + a * cos(t) - c * sin(t))

func receiverAt(o: Ortho, i, k: int): (float64, float64) =
  o.at(float64(k) * o.interval, float64(i) * o.lineSpacing)

func sourceAt(o: Ortho, m, q: int): (float64, float64) =
  o.at(o.sourceA0 + float64(m) * o.sourceLineSpacing, o.sourceC0 + float64(q) * o.sourceInterval)

proc geometry(o: Ortho): SurveyGeometry =
  result = initSurveyGeometry()
  for i in 0 ..< o.receiverLines:
    for k in 0 ..< o.stations:
      let (x, y) = o.receiverAt(i, k)
      result.addReceiver(float64(2001 + i), float64(101 + k), 1, x, y, 0)
  for m in 0 ..< o.sourceLines:
    for q in 0 ..< o.sourcePoints:
      let (x, y) = o.sourceAt(m, q)
      result.addSource(float64(5001 + m), float64(301 + q), 1, x, y, 0)

func initOrtho(az: float64): Ortho =
  # 6 receiver lines 300 apart, stations every 50 (21); 5 source lines 400
  # apart, between stations (a = 25 + 400 m), points every 60 (31) from c = −90.
  Ortho(azimuthDeg: az, x0: 612_345.0, y0: 4_401_234.0, receiverLines: 6, stations: 21,
    interval: 50, lineSpacing: 300, sourceLines: 5, sourcePoints: 31, sourceInterval: 60,
    sourceLineSpacing: 400, sourceA0: 25, sourceC0: -90)

proc midKeys(x, y: float64): TraceKeys =
  result = initTraceKeys()
  (result.cmpX, result.cmpY) = (x, y)

block natural_grid:
  for az in [90.0, 30.0, 200.0, 359.5]:
    let o = initOrtho(az)
    let b = surveyGrid(o.geometry)
    doAssert b.kind == bnGrid
    # midpoints: a in [(0 + 25)/2, (1000 + 1625)/2], c in [(0 − 90)/2, (1500 + 1710)/2]
    doAssert b.crosslines == int((1312.5 - 12.5) / 25) + 1, $b.crosslines # 53
    doAssert b.inlines == int((1605 - -45) / 30) + 1, $b.inlines # 56
    let first = b.location(0)
    let (x0, y0) = o.at(12.5, -45)
    doAssert first.inline == 1 and first.crossline == 1
    doAssert close(first.x, x0) and close(first.y, y0), $first
    # crossline numbers along the azimuth every 25, inline numbers clockwise every 30
    let (x1, y1) = o.at(12.5 + 25, -45)
    doAssert close(b.location(1).x, x1) and close(b.location(1).y, y1)
    let (x2, y2) = o.at(12.5, -45 + 30)
    let next = b.location(b.crosslines)
    doAssert next.inline == 2 and close(next.x, x2) and close(next.y, y2)
    # every source–receiver midpoint is in a bin, at its centre
    for i in 0 ..< o.receiverLines:
      for k in 0 ..< o.stations:
        let (rx, ry) = o.receiverAt(i, k)
        for m in 0 ..< o.sourceLines:
          for q in 0 ..< o.sourcePoints:
            let (sx, sy) = o.sourceAt(m, q)
            let (mx, my) = ((rx + sx) / 2, (ry + sy) / 2)
            let bin = b.binOf(midKeys(mx, my))
            doAssert bin >= 0
            let l = b.location(bin)
            doAssert close(l.x, mx, 1e-5) and close(l.y, my, 1e-5)
    # the corner bins hold the extreme midpoints: the grid is no larger than needed
    let (rx, ry) = o.receiverAt(o.receiverLines - 1, o.stations - 1)
    let (sx, sy) = o.sourceAt(o.sourceLines - 1, o.sourcePoints - 1)
    doAssert b.binOf(midKeys((rx + sx) / 2, (ry + sy) / 2)) == b.len - 1

block centring:
  # A stray receiver off the lattice moves the extent's corner by 7 along
  # and 11 across: the centred grid keeps every midpoint at a bin centre,
  # the corner grid does not; both cover every midpoint.
  for az in [90.0, 30.0]:
    let o = initOrtho(az)
    var g = o.geometry
    let (sx, sy) = o.at(-14, -22)
    g.addReceiver(1999, 1, 1, sx, sy, 0)
    let centred = surveyGrid(g)
    let corner = surveyGrid(g, centre = false)
    var offCentre = 0
    for i in 0 ..< o.receiverLines:
      for k in 0 ..< o.stations:
        let (rx, ry) = o.receiverAt(i, k)
        for m in 0 ..< o.sourceLines:
          for q in 0 ..< o.sourcePoints:
            let (qx, qy) = o.sourceAt(m, q)
            let (mx, my) = ((rx + qx) / 2, (ry + qy) / 2)
            let bin = centred.binOf(midKeys(mx, my))
            doAssert bin >= 0
            let l = centred.location(bin)
            doAssert close(l.x, mx, 1e-5) and close(l.y, my, 1e-5)
            let b2 = corner.binOf(midKeys(mx, my))
            doAssert b2 >= 0
            if not close(corner.location(b2).x, mx, 1e-5): inc offCentre
    doAssert offCentre > 0
    # the stray receiver's midpoints are covered too
    for m in [0, o.sourceLines - 1]:
      for q in [0, o.sourcePoints - 1]:
        let (qx, qy) = o.sourceAt(m, q)
        doAssert centred.binOf(midKeys((sx + qx) / 2, (sy + qy) / 2)) >= 0
  # A regular survey: centring changes nothing.
  let r = initOrtho(30).geometry
  let (c0, k0) = (surveyGrid(r), surveyGrid(r, centre = false))
  doAssert c0.len == k0.len
  doAssert close(c0.location(0).x, k0.location(0).x, 1e-6), $c0.location(0) & " " & $k0.location(0)
  doAssert close(c0.location(0).y, k0.location(0).y, 1e-6)

block strays:
  # A placeholder source at (0, 0) and a receiver 1e6 away: left out of the
  # extent by default (their pairs fall outside), covered when asked.
  let o = initOrtho(30)
  var g = o.geometry
  let plain = surveyGrid(g)
  g.addSource(9999, 1, 1, 0, 0, 0)
  let (fx, fy) = o.at(1e6, 0)
  g.addReceiver(9999, 1, 1, fx, fy, 0)
  let trimmed = surveyGrid(g)
  doAssert trimmed.len == plain.len, $trimmed.len & " " & $plain.len
  doAssert close(trimmed.location(0).x, plain.location(0).x) and
    close(trimmed.location(0).y, plain.location(0).y), $trimmed.location(0) & $plain.location(0)
  # Covering the strays takes 2.9e9 bins: more than the default budget.
  doAssertRaises(ValueError): discard surveyGrid(g, trimStrays = false)
  let full = surveyGrid(g, trimStrays = false, maxBins = high(int))
  doAssert full.len > 100 * plain.len
  # a compact survey loses no station: the extreme midpoints stay inside
  let (rx, ry) = o.receiverAt(o.receiverLines - 1, o.stations - 1)
  let (sx, sy) = o.sourceAt(o.sourceLines - 1, o.sourcePoints - 1)
  doAssert trimmed.binOf(midKeys((rx + sx) / 2, (ry + sy) / 2)) == trimmed.len - 1

block overrides:
  let o = initOrtho(30)
  let g = o.geometry
  let b = surveyGrid(g, binInline = 12.5, binCrossline = 15)
  doAssert b.crosslines == 105 and b.inlines == 111
  # a given azimuth: the extent is measured in its frame
  let r = surveyGrid(g, azimuthDeg = 120)
  let l = r.location(1)
  let l0 = r.location(0)
  doAssert close(arctan2(l.x - l0.x, l.y - l0.y).radToDeg.floorMod(360), 120)
  doAssert close(hypot(l.x - l0.x, l.y - l0.y), 25)
  # every midpoint is still covered
  for (i, k, m, q) in [(0, 0, 4, 30), (5, 20, 0, 0), (5, 0, 4, 0), (0, 20, 0, 30)]:
    let (rx, ry) = o.receiverAt(i, k)
    let (sx, sy) = o.sourceAt(m, q)
    doAssert r.binOf(midKeys((rx + sx) / 2, (ry + sy) / 2)) >= 0

block errors:
  let g = initOrtho(90).geometry
  for bad in [0.0, -1.0, Inf]:
    doAssertRaises(ValueError): discard surveyGrid(g, binInline = bad)
    doAssertRaises(ValueError): discard surveyGrid(g, binCrossline = bad)
  doAssertRaises(ValueError): discard surveyGrid(g, azimuthDeg = Inf)
  doAssertRaises(ValueError): discard surveyGrid(initSurveyGeometry())
  # one receiver and one source: nothing to measure unless given
  var one = initSurveyGeometry()
  one.addReceiver(1, 1, 1, 0, 0, 0)
  one.addSource(2, 1, 1, 100, 0, 0)
  doAssertRaises(ValueError): discard surveyGrid(one)
  doAssertRaises(ValueError): discard surveyGrid(one, azimuthDeg = 90, binInline = 10)
  let single = surveyGrid(one, binInline = 10, binCrossline = 10, azimuthDeg = 90)
  doAssert single.len == 1 and close(single.location(0).x, 50)

block bin_budget:
  # A bin size typed as 0.01 for 10: the midpoints span 1300 × 1650, which
  # would take 2.1e10 bins, 86 GB of fold counts. The error names both.
  let g = initOrtho(90).geometry
  try:
    discard surveyGrid(g, binInline = 0.01, binCrossline = 0.01)
    doAssert false, "a grid of 2.1e10 bins was made"
  except ValueError as e:
    doAssert "0.01 × 0.01" in e.msg and "1300 × 1650" in e.msg and "maxBins" in e.msg, e.msg
  # More bins than an int counts.
  doAssertRaises(ValueError): discard surveyGrid(g, binInline = 1e-12, binCrossline = 1e-12)
  doAssertRaises(ValueError): discard surveyGrid(g, binInline = 1e-300, binCrossline = 1e-300)
  # The budget is the caller's: the natural grid fits its own size exactly.
  let b = surveyGrid(g)
  doAssert surveyGrid(g, maxBins = b.len).len == b.len
  doAssertRaises(ValueError): discard surveyGrid(g, maxBins = b.len - 1)
  doAssertRaises(ValueError): discard surveyGrid(g, maxBins = 0)
  # A binning of the caller's own is checked where the counts are allocated.
  let big = gridBinning(0, 0, 90, 0.01, 0.02, 100_000, 100_000)
  try:
    discard initFoldMap(big)
    doAssert false, "a fold map of 1e10 bins was made"
  except ValueError as e:
    doAssert "0.01 × 0.02" in e.msg and "1000 × 2000" in e.msg and "maxBins" in e.msg, e.msg
  doAssertRaises(ValueError): discard g.preplotFold(big)
  doAssertRaises(ValueError): discard initFoldMap(cdpBinning(1, 1 shl 27))
  doAssert initFoldMap(b, maxBins = b.len).fold.size == b.len
  doAssert g.preplotFold(b, maxBins = b.len).fold.size == b.len
  doAssertRaises(ValueError): discard initFoldMap(b, maxBins = b.len - 1)
  doAssertRaises(ValueError): discard g.preplotFold(b, maxBins = b.len - 1)
  doAssertRaises(ValueError): discard initFoldMap(b, maxBins = -1)

# An orthogonal swath with X records: 10 receiver lines 200 apart (c = 200 i),
# 81 stations every 50 (a = 50 k); 20 source lines between stations
# (a = 25 + 200 m), 36 points every 50 (c = 25 + 50 q). Patch: the 4 receiver
# lines nearest the source (2 each side) × 32 stations (16 each side), clipped
# at the edges. Nominal fold: inline 32 × 50 / (2 × 200) = 4, crossline 4 / 2 = 2.
const
  swRl = 10
  swSt = 81
  swSl = 20
  swSp = 36

func swReceiver(i, k: int): (float64, float64) = (50.0 * float64(k), 200.0 * float64(i))
func swSource(m, q: int): (float64, float64) = (25 + 200.0 * float64(m), 25 + 50.0 * float64(q))

proc swathFile(o: Ortho): SpsFile =
  for i in 0 ..< swRl:
    for k in 0 ..< swSt:
      var p = initSpsPoint(pkReceiver)
      (p.line, p.point) = (float64(2001 + i), float64(101 + k))
      let (a, c) = swReceiver(i, k)
      (p.easting, p.northing) = o.at(a, c)
      p.elevation = 0
      result.points.add p
  var record = 1'i64
  for m in 0 ..< swSl:
    for q in 0 ..< swSp:
      var p = initSpsPoint(pkSource)
      (p.line, p.point) = (float64(5001 + m), float64(301 + q))
      let (a, c) = swSource(m, q)
      (p.easting, p.northing) = o.at(a, c)
      p.elevation = 0
      result.points.add p
      let i0 = int(floor(c / 200))
      var channel = 1'i64
      for i in max(0, i0 - 1) .. min(swRl - 1, i0 + 2):
        let (k0, k1) = (max(0, 4 * m - 15), min(swSt - 1, 4 * m + 16))
        var r = initSpsRelation()
        r.fieldRecord = record
        (r.srcLine, r.srcPoint) = (p.line, p.point)
        r.fromChannel = channel
        r.toChannel = channel + int64(k1 - k0)
        r.rcvLine = float64(2001 + i)
        (r.fromReceiver, r.toReceiver) = (float64(101 + k0), float64(101 + k1))
        result.relations.add r
        channel = r.toChannel + 1
      inc record

block preplot_swath:
  let bounds = @[0.0, 300, 800, 1500]
  let opts = FoldOptions(offsetBounds: bounds, azimuthSectors: 6, sectorStartDeg: 15,
    offsetRange: true)
  for az in [90.0, 30.0, 200.0]:
    let o = Ortho(azimuthDeg: az, x0: 512_000, y0: 4_300_000)
    var g = initSurveyGeometry()
    g.addSps(o.swathFile)
    let b = surveyGrid(g)
    doAssert close(b.location(0).x, o.at(12.5, 12.5)[0]) # bins 25 × 25 from (12.5, 12.5)
    let f = g.preplotFold(b, opts)
    doAssert f.fold.shape == [b.inlines, b.crosslines]
    doAssert f.offsetFold.shape == [3, b.inlines, b.crosslines]
    doAssert f.azimuthFold.shape == [6, b.inlines, b.crosslines]
    # brute force from the generator's own coordinates
    var fold = newSeq[int32](b.len)
    var byOffset = newSeq[int32](3 * b.len)
    var byAzimuth = newSeq[int32](6 * b.len)
    var lo = newSeq[float64](b.len)
    var hi = newSeq[float64](b.len)
    for x in lo.mitems: x = Inf
    for x in hi.mitems: x = -Inf
    var pairs = 0
    for r in o.swathFile.relations:
      let m = int(r.srcLine) - 5001
      let q = int(r.srcPoint) - 301
      let (sx, sy) = o.at(swSource(m, q)[0], swSource(m, q)[1])
      for k in int(r.fromReceiver) - 101 .. int(r.toReceiver) - 101:
        let i = int(r.rcvLine) - 2001
        let (a, c) = swReceiver(i, k)
        let (rx, ry) = o.at(a, c)
        let bin = b.binOf(midKeys((sx + rx) / 2, (sy + ry) / 2))
        doAssert bin >= 0
        inc pairs
        inc fold[bin]
        let off = hypot(rx - sx, ry - sy)
        for j in 0 ..< 3:
          if off >= bounds[j] and off < bounds[j + 1]: inc byOffset[j * b.len + bin]
        let bearing = floorMod(radToDeg(arctan2(rx - sx, ry - sy)) - 15, 360)
        inc byAzimuth[int(bearing / 60) * b.len + bin]
        lo[bin] = min(lo[bin], off)
        hi[bin] = max(hi[bin], off)
    doAssert f.binned == pairs and f.outside == 0 and f.noCoordinates == 0
    let fl = f.fold.toFlatSeq
    doAssert fl == fold
    doAssert f.offsetFold.toFlatSeq == byOffset
    doAssert f.azimuthFold.toFlatSeq == byAzimuth
    let (fmin, fmax) = (f.offsetMin.toFlatSeq, f.offsetMax.toFlatSeq)
    for i in 0 ..< b.len:
      if fold[i] == 0: doAssert fmin[i].isNaN and fmax[i].isNaN
      else:
        doAssert abs(fmin[i] - lo[i]) < 1e-3 and abs(fmax[i] - hi[i]) < 1e-3
    # nominal fold 4 × 2 = 8 in the full-fold area, never more
    doAssert max(fl) == 8
    var full = 0
    for i in 0 ..< b.inlines:
      for j in 0 ..< b.crosslines:
        let (a, c) = (12.5 + 25.0 * float64(j), 12.5 + 25.0 * float64(i))
        # far enough from the edges that no patch reaching it is clipped
        if a >= 800 + 400 and a <= 4000 - 800 - 400 and c >= 400 + 200 and c <= 1800 - 400 - 200:
          doAssert fl[i * b.crosslines + j] == 8, $(a, c, fl[i * b.crosslines + j])
          inc full
    doAssert full > 1000, $full
    # the same pairs from the receiver domain (O records)
    var go = initSurveyGeometry()
    var oFile = o.swathFile
    oFile.relations = g.receiverRelations
    go.addSps(oFile)
    let fo = go.preplotFold(b, opts, domain = rdReceiver)
    doAssert fo.fold.toFlatSeq == fold and fo.offsetFold.toFlatSeq == byOffset
    doAssert fo.azimuthFold.toFlatSeq == byAzimuth
    # a 2D line binning works too; a header binning is refused
    let line = lineBinning(o.at(0, 900)[0], o.at(0, 900)[1], az, 25, 200, maxOffLine = 100)
    let fl2 = g.preplotFold(line)
    doAssert fl2.fold.shape == [1, 200] and fl2.binned + fl2.outside == pairs
    doAssert fl2.binned > 0 and fl2.outside > 0
    doAssertRaises(ValueError): discard g.preplotFold(cdpBinning(1, 10))

block preplot_missing_coordinates:
  let o = Ortho(azimuthDeg: 90)
  var f = o.swathFile
  var dropped = 0
  for p in f.points.mitems:
    if p.kind == pkReceiver and p.line == 2005 and p.point == 140:
      (p.easting, p.northing) = (NaN, NaN)
  var g = initSurveyGeometry()
  g.addSps(f)
  for r in f.relations:
    if r.rcvLine == 2005 and r.fromReceiver <= 140 and r.toReceiver >= 140: inc dropped
  let m = g.preplotFold(surveyGrid(g))
  doAssert dropped > 0 and m.noCoordinates == dropped, $(m.noCoordinates, dropped)

block fold_options:
  let b = gridBinning(0, 0, 90, 10, 10, 3, 4)
  let m = initFoldMap(b)
  doAssert m.fold.shape == [3, 4] and m.offsetFold.size == 0 and m.offsetMin.size == 0
  for bad in [@[1.0], @[0.0, 0.0], @[2.0, 1.0], @[0.0, NaN], @[0.0, Inf]]:
    doAssertRaises(ValueError): discard initFoldMap(b, FoldOptions(offsetBounds: bad))
  doAssertRaises(ValueError): discard initFoldMap(b, FoldOptions(azimuthSectors: -1))
  doAssertRaises(ValueError): discard initFoldMap(b, FoldOptions(sectorStartDeg: NaN))

block clone_is_independent:
  # A map's counts are tensors, which share their storage when copied: a map
  # is moved or cloned, never copied (tests/reject/rfoldcopy.nim).
  let grid = gridBinning(0, 0, 0, 10, 10, 4, 4)
  var k = initTraceKeys()
  k.status = tsLive
  (k.srcX, k.srcY, k.rcvX, k.rcvY) = (0.0, 0.0, 20.0, 20.0)
  let keys = @[k]
  let counts = @[1]
  var a = initFoldMap(grid, FoldOptions(offsetBounds: @[0.0, 100], azimuthSectors: 2,
    offsetRange: true))
  a.addFold(rowsOf(keys, counts))
  var b = a.clone
  doAssert b.binned == 1 and b.fold.toFlatSeq == a.fold.toFlatSeq
  doAssert b.binning.len == a.binning.len and b.options.azimuthSectors == 2
  b.addFold(rowsOf(keys, counts))
  doAssert a.binned == 1 and b.binned == 2
  template total(t: untyped): int =
    var n = 0
    for v in t: n += int(v)
    n
  doAssert total(a.fold) == 1 and total(b.fold) == 2, "the clone has its own fold"
  doAssert total(a.offsetFold) == 1 and total(b.offsetFold) == 2
  doAssert total(a.azimuthFold) == 1 and total(b.azimuthFold) == 2
  b.clear
  doAssert total(a.fold) == 1 and total(b.fold) == 0
  var known = 0
  for v in a.offsetMin: known += int(not v.isNaN)
  doAssert known == 1, "clearing the clone leaves the original's offset range"
  # A map without the optional counts clones too; a returned map is a move.
  var plain = initFoldMap(grid).clone
  plain.addFold(rowsOf(keys, counts))
  doAssert plain.binned == 1 and plain.offsetFold.size == 0 and plain.offsetMin.size == 0

# Traces of the swath, one per pre-plot pair (keys as parseSEG's applySurvey
# leaves them); `drop` decides each pair's fate.
type Fate = enum fLive, fOmitted, fDead, fAux, fFailed

proc swathTraces(o: Ortho, fateOf: proc (n: int): Fate, keys: var seq[TraceKeys],
    failed: var seq[int], dropped: var seq[(float64, float64)]) =
  var n = 0
  for r in o.swathFile.relations:
    let m = int(r.srcLine) - 5001
    let q = int(r.srcPoint) - 301
    let (sx, sy) = o.at(swSource(m, q)[0], swSource(m, q)[1])
    for kk in int(r.fromReceiver) - 101 .. int(r.toReceiver) - 101:
      let i = int(r.rcvLine) - 2001
      let (rx, ry) = o.at(swReceiver(i, kk)[0], swReceiver(i, kk)[1])
      let fate = fateOf(n)
      inc n
      if fate != fLive: dropped.add ((sx + rx) / 2, (sy + ry) / 2)
      if fate == fOmitted: continue
      var k = initTraceKeys()
      (k.srcLine, k.srcPoint, k.srcX, k.srcY) = (r.srcLine, r.srcPoint, sx, sy)
      (k.rcvLine, k.rcvPoint, k.rcvX, k.rcvY) = (r.rcvLine, float64(101 + kk), rx, ry)
      k.offset = hypot(rx - sx, ry - sy)
      if n mod 2 == 0: (k.cmpX, k.cmpY) = ((sx + rx) / 2, (sy + ry) / 2) # else NaN: from coordinates
      k.status = case fate
        of fDead: tsDead
        of fAux: tsAuxiliary
        else: (if n mod 3 == 0: tsUnknown else: tsLive)
      if fate == fFailed: failed.add keys.len
      keys.add k

proc sameMaps(a, b: FoldMap): bool =
  a.fold.toFlatSeq == b.fold.toFlatSeq and a.offsetFold.toFlatSeq == b.offsetFold.toFlatSeq and
    a.azimuthFold.toFlatSeq == b.azimuthFold.toFlatSeq and
    $a.offsetMin.toFlatSeq == $b.offsetMin.toFlatSeq and
    $a.offsetMax.toFlatSeq == $b.offsetMax.toFlatSeq and a.binned == b.binned

block actual_fold:
  let opts = FoldOptions(offsetBounds: @[0.0, 300, 800, 1500], azimuthSectors: 6,
    offsetRange: true)
  let o = Ortho(azimuthDeg: 30, x0: 512_000, y0: 4_300_000)
  var g = initSurveyGeometry()
  g.addSps(o.swathFile)
  let b = surveyGrid(g)
  let pre = g.preplotFold(b, opts)
  # every pair live: actual equals pre-plot, counts and all
  block:
    var keys: seq[TraceKeys]
    var failed: seq[int]
    var dropped: seq[(float64, float64)]
    o.swathTraces(proc (n: int): Fate = fLive, keys, failed, dropped)
    var a = initFoldMap(b, opts)
    let counts = newSeq[int](keys.len)
    a.addFold(rowsOf(keys, counts))
    doAssert sameMaps(a, pre) and a.notLive == 0 and a.outside == 0
    doAssert max(foldDifference(pre, a).toFlatSeq) == 0 and min(foldDifference(pre, a).toFlatSeq) == 0
  # dropped channels: omitted, dead, auxiliary, failed rows
  let fate = proc (n: int): Fate =
    if n mod 37 == 5: fOmitted
    elif n mod 101 == 7: fDead
    elif n mod 211 == 3: fAux
    elif n mod 89 == 11: fFailed
    else: fLive
  var keys: seq[TraceKeys]
  var failed: seq[int]
  var dropped: seq[(float64, float64)]
  o.swathTraces(fate, keys, failed, dropped)
  var expected = newSeq[int32](b.len)
  for (x, y) in dropped: inc expected[b.binAt(x, y)]
  # one extra live trace whose midpoint is in the wrong place (a geometry
  # error), in a bin with no dropped trace
  let empty = expected.find(0'i32)
  doAssert empty >= 0
  var wrong = keys[100]
  (wrong.cmpX, wrong.cmpY) = (b.location(empty).x, b.location(empty).y)
  wrong.status = tsLive
  keys.add wrong
  dec expected[empty]
  let counts = newSeq[int](keys.len)
  let rows = rowsOf(keys, counts)
  # in order, in one block
  var whole = initFoldMap(b, opts)
  let meta = initChunkMeta(rows, dtMs = 2, failed = failed)
  doAssert allocsIn(whole.addFold(meta)) == 0
  var notLive = 0
  for k in keys:
    if k.status notin {tsLive, tsUnknown}: inc notLive
  doAssert whole.notLive == notLive + failed.len and whole.noCoordinates == 0
  # reversed, in uneven chunks (each with its own failed rows)
  var chunked = initFoldMap(b, opts)
  var hi = keys.len
  var size = 1
  while hi > 0:
    let lo = max(0, hi - size)
    var sub: seq[int]
    for r in failed:
      if r in lo ..< hi: sub.add r - lo
    chunked.addFold(initChunkMeta(rowsOf(keys.toOpenArray(lo, hi - 1),
      counts.toOpenArray(lo, hi - 1)), dtMs = 2, failed = sub))
    hi = lo
    size = size * 3 + 1
  doAssert sameMaps(whole, chunked) and whole.notLive == chunked.notLive
  let diff = foldDifference(pre, whole).toFlatSeq
  doAssert diff == expected
  doAssert diff[empty] == -1 # the misplaced trace
  # clear: back to empty, storage kept
  whole.clear
  doAssert max(whole.fold.toFlatSeq) == 0 and whole.binned == 0 and whole.notLive == 0
  doAssert whole.offsetMin.toFlatSeq[0].isNaN
  # no midpoint and no coordinates: counted, not binned
  var bare = initFoldMap(b)
  var lone = @[initTraceKeys()]
  bare.addFold(rowsOf(lone, @[0]))
  doAssert bare.noCoordinates == 1 and bare.binned == 0
  # different binnings are refused
  doAssertRaises(ValueError): discard foldDifference(pre, initFoldMap(surveyGrid(g,
      binInline = 10)))

block actual_header_binning:
  let b = inlineCrosslineBinning(101, 3, 1001, 4)
  var keys = newSeq[TraceKeys](5)
  for i, k in keys.mpairs:
    k = initTraceKeys()
    (k.inline, k.crossline, k.offset) = (101.0 + float64(i mod 3), 1001.0 + float64(i), -250.0)
  var m = initFoldMap(b, FoldOptions(offsetBounds: @[0.0, 200, 400], azimuthSectors: 4))
  m.addFold(rowsOf(keys, newSeq[int](5)))
  doAssert m.binned == 4 and m.outside == 1 # crossline 1005 is outside
  doAssert m.fold.toFlatSeq == @[1'i32, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
  # |offset| from the header; no coordinates: no bearing
  doAssert m.offsetFold.toFlatSeq[12 ..< 24] == m.fold.toFlatSeq
  doAssert max(m.azimuthFold.toFlatSeq) == 0

const
  dir = currentSourcePath().parentDir.parentDir / "exampleFiles" / "sps"
  base = dir / "PGI_NNE_West_Morgantown_3D_Post261019_NAD27_SPCS_WV_N_USFeet"

block morgantown:
  if not (fileExists(base & ".sps") and fileExists(base & ".rps")):
    echo "  skipped: Morgantown SPS files not present"
  else:
    var g = initSurveyGeometry()
    for ext in [".sps", ".rps"]: g.addSps(base & ext)
    let b = surveyGrid(g)
    let (l0, l1, l2) = (b.location(0), b.location(1), b.location(b.crosslines))
    let az = floorMod(radToDeg(arctan2(l1.x - l0.x, l1.y - l0.y)), 360.0)
    let binI = hypot(l1.x - l0.x, l1.y - l0.y)
    let binC = hypot(l2.x - l0.x, l2.y - l0.y)
    echo "  Morgantown grid: azimuth ", formatFloat(az, ffDecimal, 2), "°, bin ",
      formatFloat(binI, ffDecimal, 1), " × ", formatFloat(binC, ffDecimal, 1), " ft, ",
      b.inlines, " inlines × ", b.crosslines, " crosslines"
    doAssert abs(az - 91) < 1, $az
    doAssert abs(binI - 111.3) < 0.5, $binI
    # irregular sources: no significant phase, so centring keeps the corner
    let corner = surveyGrid(g, centre = false)
    doAssert corner.len == b.len and corner.location(0).x == b.location(0).x and
      corner.location(0).y == b.location(0).y
    if fileExists(base & ".xps"):
      g.addSps(base & ".xps")
      when defined(release) or defined(danger): (let t0 = cpuTime())
      let f = g.preplotFold(b, FoldOptions(offsetBounds: @[0.0, 5000, 10000, 20000, 40000],
        azimuthSectors: 8, offsetRange: true))
      when defined(release) or defined(danger):
        echo "  Morgantown preplotFold: ", formatFloat(cpuTime() - t0, ffDecimal, 2), " s"
      let fl = f.fold.toFlatSeq
      var live = 0
      for x in fl:
        if x > 0: inc live
      echo "  Morgantown pre-plot fold: ", f.binned, " pairs, max ", max(fl), ", ", live,
        " bins with fold; outside ", f.outside, ", no coordinates ", f.noCoordinates
      doAssert f.binned + f.noCoordinates == 11_180_874 and f.outside == 0
