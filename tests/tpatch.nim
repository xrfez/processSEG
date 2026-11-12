## Patch description (`survey/patch`): a synthetic regular survey cut from a
## lines × channels template, straight and rotated, and the Morgantown 3D
## survey (skipped when its files are absent).

import std/[assertions, math, os, strutils]
when defined(release) or defined(danger): import std/times
import parseSEG
import processSEG/survey/patch
import utils/survey

proc close(a, b: float64, tol = 1e-6): bool = abs(a - b) <= tol

block layout:
  for az in [90.0, 30.0, 359.5, 200.0]:
    var s = initSyntheticSurvey()
    s.azimuthDeg = az
    (s.x0, s.y0) = (512_000.0, 4_300_000.0)
    let l = s.geometry.receiverLayout
    doAssert close(l.azimuthDeg, az, 1e-6), $l.azimuthDeg
    doAssert close(l.interval, 50) and close(l.lineSpacing, 200)
    doAssert l.lines.len == 8
    for line in l.lines:
      doAssert line.stations == 41 and close(line.inlineMax - line.inlineMin, 40 * 50)
    # crossline grows 90° clockwise of the azimuth: line numbers grow anticlockwise here
    doAssert l.lines[0].line == 1008 and l.lines[^1].line == 1001
    # a given azimuth is kept
    doAssert s.geometry.receiverLayout(azimuthDeg = 45).azimuthDeg == 45

block regular_template:
  for az in [90.0, 30.0, 200.0]:
    var s = initSyntheticSurvey()
    s.azimuthDeg = az
    (s.x0, s.y0) = (1000.0, -2000.0)
    let p = describePatches(s.geometry)
    let shots = s.shots
    doAssert p.records.len == shots.len and shots.len == 7 * 21
    doAssert p.regular and p.templateLines == 4 and p.templateChannels == 11, $p
    doAssert close(p.boxInline, 5 * 50) and close(p.boxCrossline, 1.5 * 200), $p
    doAssert p.noSource == 0
    var clipped = 0
    for i, r in p.records:
      let t = shots[i]
      doAssert r.record == t.record
      doAssert r.source == stationKey(float64(1001 + t.sourceLine), float64(100 + t.sourceStation), 1)
      let stations = t.lastStation - t.firstStation + 1
      doAssert r.channels == t.lines.len * stations and r.noCoordinates == 0
      doAssert r.lines.len == t.lines.len
      for j, l in r.lines:
        doAssert l.line == float64(1001 + t.lines[j]) and l.stations == stations
        doAssert l.firstPoint == float64(100 + t.firstStation)
        doAssert l.lastPoint == float64(100 + t.lastStation)
      # extent relative to the source
      doAssert close(r.inlineMin, float64(t.firstStation - t.sourceStation) * 50)
      doAssert close(r.inlineMax, float64(t.lastStation - t.sourceStation) * 50)
      let vLo = (float64(t.lines[0]) - (float64(t.sourceLine) + 0.5)) * 200
      let vHi = (float64(t.lines[^1]) - (float64(t.sourceLine) + 0.5)) * 200
      doAssert close(r.crosslineMin, -vHi) and close(r.crosslineMax, -vLo)
      let du = max(-r.inlineMin, r.inlineMax)
      let dv = max(-r.crosslineMin, r.crosslineMax)
      doAssert r.maxOffset <= hypot(du, dv) + 1e-6
      # clipped exactly when the template did not fit, on the sides it did not fit
      var expected: set[PatchSide]
      if t.sourceStation - 5 < 0: expected.incl psInlineBack
      if t.sourceStation + 5 > 40: expected.incl psInlineForward
      if t.sourceLine - 1 < 0: expected.incl psCrosslineRight  # lower lines: clockwise
      if t.sourceLine + 2 > 7: expected.incl psCrosslineLeft
      doAssert r.clipped == expected, $r.record & ": " & $r.clipped & " vs " & $expected
      if expected != {}: inc clipped
    doAssert p.clippedRecords == clipped and clipped == shots.len - 5 * 15
    doAssert ($p).startsWith("4 × 11 template, box ±250.0 inline × ±300.0 crossline"), $p

block irregular:
  # Drop one station from a full record's patch: the survey is no longer a template.
  var s = initSyntheticSurvey()
  var f = s.spsFile
  var g = initSurveyGeometry()
  var gap = 0 # first line of record 74: shot j = 3, k = 20, an unclipped one
  while f.relations[gap].fieldRecord != 74: inc gap
  let r = f.relations[gap]
  var a = r
  a.toReceiver = r.fromReceiver + 4
  a.toChannel = r.fromChannel + 4
  var b = r
  b.fromReceiver = r.fromReceiver + 6
  b.fromChannel = r.fromChannel + 5
  b.toChannel = r.toChannel - 1
  f.relations[gap] = a
  f.relations.insert(b, gap + 1)
  g.addSps(f)
  let p = describePatches(g)
  doAssert not p.regular and p.templateLines == 4 and p.templateChannels == 11, $p
  let rec = p.records[int(r.fieldRecord) - 1]
  doAssert rec.channels == 43 and rec.clipped == {}
  doAssert "nominal" in $p

block missing_coordinates:
  # Receivers listed in X but not in R are counted; a source without one has no extent.
  var s = initSyntheticSurvey()
  var f = s.spsFile
  var removed = 0
  for i in countdown(f.points.high, 0):
    let p = f.points[i]
    if p.kind == pkReceiver and p.line == 1004 and p.point == 120:
      f.points.delete i
      inc removed
    elif p.kind == pkSource and p.line == 1001 and p.point == 100:
      f.points.delete i
  doAssert removed == 1
  var g = initSurveyGeometry()
  g.addSps(f)
  let p = describePatches(g)
  doAssert p.noSource == 1 and p.records[0].sourceX.isNaN and p.records[0].maxOffset.isNaN
  var listed = 0
  for r in p.records: listed += r.noCoordinates
  doAssert listed > 0

block arguments:
  let g = initSyntheticSurvey().geometry
  doAssertRaises(ValueError): discard describePatches(g, clipTolerance = -1)
  doAssertRaises(ValueError): discard initSurveyGeometry().receiverLayout

block spread:
  let s = spreadOf([5.0, NaN, 1, 3, 2, 4])
  doAssert s.min == 1 and s.q1 == 2 and s.median == 3 and s.q3 == 4 and s.max == 5
  doAssert spreadOf(newSeq[float64]()).median.isNaN

const
  dir = currentSourcePath().parentDir.parentDir / "exampleFiles" / "sps"
  base = dir / "PGI_NNE_West_Morgantown_3D_Post261019_NAD27_SPCS_WV_N_USFeet"

block morgantown:
  if not (fileExists(base & ".sps") and fileExists(base & ".rps") and fileExists(base & ".xps")):
    echo "  skipped: Morgantown SPS files not present"
  else:
    var g = initSurveyGeometry()
    for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
    when defined(release) or defined(danger): (let t0 = cpuTime())
    let p = describePatches(g)
    when defined(release) or defined(danger):
      echo "  Morgantown describePatches: ", formatFloat(cpuTime() - t0, ffDecimal, 2), " s"
    echo "  Morgantown: ", p
    echo "    lines ", p.lines, "; channels ", p.channels
    echo "    inline half-width ", p.inlineHalfWidth, "; crossline ", p.crosslineHalfWidth
    echo "    max offset ", p.maxOffset, "; interval ", formatFloat(p.layout.interval, ffDecimal, 1),
      ", line spacing ", formatFloat(p.layout.lineSpacing, ffDecimal, 1)
    doAssert p.records.len == 5519 and p.noSource == 0
    doAssert abs(p.layout.azimuthDeg - 91) < 1, $p.layout.azimuthDeg
    doAssert abs(p.layout.interval - 222.6) < 1, $p.layout.interval
    doAssert not p.regular
    doAssert abs(p.boxInline - 19_340) < 200 and abs(p.boxCrossline - 10_060) < 200, $p
    doAssert p.lines.median in 22.0 .. 24.0 and p.channels.min == 648 and p.channels.max == 3154
    doAssert p.clippedRecords > 0 and p.clippedRecords < p.records.len
