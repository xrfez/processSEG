## Survey bin grid (`qc/fold`): an orthogonal synthetic survey at several
## azimuths gives back its natural grid exactly, with every midpoint at a bin
## centre; overrides; errors; Morgantown (skipped when its files are absent).

import std/[assertions, math, os, strutils]
import parseSEG
import processSEG/stack/binning
import processSEG/qc/fold

proc close(a, b: float64, tol = 1e-6): bool = abs(a - b) <= tol

type Ortho = object
  azimuthDeg, x0, y0: float64
  receiverLines, stations: int
  interval, lineSpacing: float64 # receivers: along (a) and across (c) the azimuth
  sourceLines, sourcePoints: int
  sourceInterval, sourceLineSpacing: float64 # sources: across (c) and along (a)
  sourceA0, sourceC0: float64 # first source line (a), first source point (c)

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
    doAssert b.inlines == int((1605 - -45) / 30) + 1, $b.inlines          # 56
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
