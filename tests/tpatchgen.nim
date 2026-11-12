## X generation from a template (`survey/patch`, task 11.8): synthetic
## surveys whose X records are written by hand (`utils/survey`), in each
## edge mode, by coordinates and by numbers, rotated and irregularly
## numbered; R gaps and short lines; a write / read round trip; and the
## Morgantown survey against its own XPS (skipped when absent).

import std/[algorithm, assertions, math, os, sets, streams, strutils]
when defined(release) or defined(danger): import std/times
import parseSEG
import processSEG/survey/patch
import utils/survey

proc close(a, b: float64, tol = 1e-6): bool = abs(a - b) <= tol

type Run = tuple[record: int64, srcLine, srcPoint: float64, fromChannel, toChannel: int64,
  rcvLine, fromReceiver, toReceiver: float64]

func runs(rs: openArray[SpsRelation]): seq[Run] =
  for r in rs:
    result.add (r.fieldRecord, r.srcLine, r.srcPoint, r.fromChannel, r.toChannel, r.rcvLine,
      r.fromReceiver, r.toReceiver)

proc sameRuns(a, b: openArray[SpsRelation]): bool =
  let (x, y) = (runs(a), runs(b))
  if x.len != y.len:
    echo "  runs: ", x.len, " vs ", y.len
    return false
  for i in 0 ..< x.len:
    if x[i] != y[i]:
      echo "  first difference: ", x[i], " vs ", y[i]
      return false
  true

proc numbersTemplate(s: SyntheticSurvey, t: PatchTemplate): PatchTemplate =
  ## The same template by numbers: source line j sits half-way between
  ## receiver lines j and j + 1; source point k is receiver point k.
  result = t
  result.placement = ppNumbers
  result.lineMap = NumberMap(scale: 1, offset: 0.5 * float64(s.lineStep))

block hand_written:
  for edge in [peRoll, peShift, peStatic]:
    for az in [90.0, 30.0, 200.0]:
      var s = initSyntheticSurvey()
      (s.edge, s.azimuthDeg, s.x0, s.y0) = (edge, az, 7000.0, -3000.0)
      let f = s.spsFile
      var g = initSurveyGeometry()
      g.addSps(f)
      let shots = f.patchShots
      doAssert shots.len == 7 * 21 and shots[0].record == 1
      let t = initPatchTemplate(4, 11, edge)
      let byXy = g.generateRelations(shots, t)
      doAssert sameRuns(byXy, f.relations), $edge & " " & $az
      doAssert sameRuns(g.generateRelations(shots, s.numbersTemplate(t)), f.relations),
        "numbers: " & $edge & " " & $az

block irregular_numbering:
  # Rotated; lines numbered down in steps of 4, points down in steps of 2
  # (against the azimuth): the coordinates path still finds the patch, and
  # channels follow ascending line and point numbers.
  for edge in [peRoll, peShift]:
    var s = initSyntheticSurvey()
    (s.edge, s.azimuthDeg, s.lineStep, s.pointStep) = (edge, 123.0, -4, -2)
    (s.firstLine, s.firstPoint) = (2000, 900)
    let f = s.spsFile
    var g = initSurveyGeometry()
    g.addSps(f)
    let gen = g.generateRelations(f.patchShots, initPatchTemplate(4, 11, edge))
    doAssert sameRuns(gen, f.relations), $edge
    var tn = s.numbersTemplate(initPatchTemplate(4, 11, edge))
    tn.lineMap = NumberMap(scale: 1, offset: -2) # half of the −4 line step
    doAssert sameRuns(g.generateRelations(f.patchShots, tn), f.relations), "numbers " & $edge

block offsets:
  # An off-end spread: the centre 5 stations forward of the source.
  var s = initSyntheticSurvey()
  let f = s.spsFile
  var g = initSurveyGeometry()
  g.addSps(f)
  var t = initPatchTemplate(2, 11)
  t.stationOffset = 5
  t.lineOffset = 1 # the line pair one line higher in crossline
  let shot = PatchShot(record: 9, source: stationKey(1004, 120, 1))                   # j = 3, k = 20
  let gen = g.generateRelations([shot], t)
  # crossline grows towards lower line numbers here (lines grow anticlockwise)
  doAssert gen.len == 2 and gen[0].rcvLine == 1003 and gen[1].rcvLine == 1004, $runs(gen)
  doAssert gen[0].fromReceiver == 120 and gen[0].toReceiver == 130, $runs(gen)
  doAssert gen[1].fromChannel == 12 and gen[1].toChannel == 22

block r_gaps_and_short_lines:
  var s = initSyntheticSurvey()
  var f = s.spsFile
  # receiver 1004/120 missing from R: a hole, not a longer line
  for i in countdown(f.points.high, 0):
    if f.points[i].kind == pkReceiver and f.points[i].line == 1004 and f.points[i].point == 120:
      f.points.delete i
  # a short line of 3 stations 40 m off line 1005, around point 118-122
  for k in [18, 20, 22]:
    var p = initSpsPoint(pkReceiver)
    (p.line, p.point) = (5005.0, float64(100 + k))
    (p.easting, p.northing) = s.position(float64(k) * 50, 4 * 200 + 40)
    f.points.add p
  var g = initSurveyGeometry()
  g.addSps(f)
  let layout = g.receiverLayout
  doAssert layout.lines.len == 9 and layout.minLineStations == 21
  var short = 0
  for l in layout.lines:
    if l.short: inc short
  doAssert short == 1 and close(layout.lineSpacing, 200), $layout.lineSpacing
  let shot = PatchShot(record: 74, source: stationKey(1004, 120, 1)) # j = 3, k = 20
  let gen = g.generateRelations(layout, [shot], initPatchTemplate(4, 11))
  var total = 0'i64
  var lines: seq[float64]
  for r in gen:
    total += r.toChannel - r.fromChannel + 1
    lines.add r.rcvLine
  # 4 template lines × 11 − the hole + 3 short-line stations
  doAssert total == 44 - 1 + 3, $runs(gen)
  doAssert lines == @[1003.0, 1004, 1004, 1005, 1006, 5005], $lines
  doAssert gen[1].toReceiver == 119 and gen[2].fromReceiver == 121
  doAssert gen[2].fromChannel == gen[1].toChannel + 1, "no channel for the hole"
  doAssert gen[^1].fromReceiver == 118 and gen[^1].toReceiver == 122
  # the short-line run is 3 stations at step 2: one record
  doAssert gen[^1].toChannel - gen[^1].fromChannel == 2
  # line spacing as a parameter
  doAssert g.receiverLayout(lineSpacing = 250).lineSpacing == 250

block static_box:
  var s = initSyntheticSurvey()
  let f = s.spsFile
  var g = initSurveyGeometry()
  g.addSps(f)
  let layout = g.receiverLayout
  var t = initPatchTemplate(0, 0, peStatic)
  # stations 10..20 of lines 1002..1003 (crossline = −y at azimuth 90)
  t.staticBox = PatchBox(inlineMin: 499, inlineMax: 1001, crosslineMin: -401, crosslineMax: -199)
  let gen = g.generateRelations(layout, f.patchShots, t)
  doAssert gen.len == 2 * f.patchShots.len
  doAssert gen[0].rcvLine == 1002 and gen[0].fromReceiver == 110 and gen[0].toReceiver == 120
  doAssert gen[1].rcvLine == 1003 and gen[1].fromChannel == 12

block round_trip:
  var s = initSyntheticSurvey()
  s.azimuthDeg = 30
  var f = s.spsFile
  var g = initSurveyGeometry()
  g.addSps(f)
  f.relations = g.generateRelations(f.patchShots, initPatchTemplate(4, 11))
  f.revision = sps21
  var text = newStringStream()
  doAssert not f.writeSps(text).hasErrors
  let back = parseSps(text.data, "generated.sps")
  doAssert not back.log.hasErrors
  doAssert sameRuns(back.relations, f.relations)
  var g2 = initSurveyGeometry()
  g2.addSps(back)
  for r in g.records:
    var a, b: seq[StationRelation]
    for (_, rel) in g.relations(r): a.add rel
    for (_, rel) in g2.relations(r): b.add rel
    doAssert a == b

block arguments:
  let g = initSyntheticSurvey().geometry
  doAssertRaises(ValueError): discard g.generateRelations(g.patchShots, initPatchTemplate(4, 2_000_000))
  var bad = g.receiverLayout
  bad.interval = NaN
  doAssertRaises(ValueError): discard g.generateRelations(bad, g.patchShots, initPatchTemplate(4, 11))
  bad = g.receiverLayout
  bad.lineSpacing = 0
  doAssertRaises(ValueError): discard g.generateRelations(bad, g.patchShots, initPatchTemplate(4, 11))
  let shots = [PatchShot(record: 1, source: stationKey(1001, 100, 1))]
  doAssertRaises(ValueError): discard g.generateRelations(shots, initPatchTemplate(0, 11))
  doAssertRaises(ValueError): discard g.generateRelations(shots, initPatchTemplate(4, 0, peShift))
  var t = initPatchTemplate(4, 11)
  t.stationOffset = NaN
  doAssertRaises(ValueError): discard g.generateRelations(shots, t)
  doAssertRaises(ValueError): discard g.receiverLayout(lineSpacing = -1)
  # a source without coordinates gets no records by coordinates
  let none = [PatchShot(record: 1, source: stationKey(1, 1, 1))]
  doAssert g.generateRelations(none, initPatchTemplate(4, 11)).len == 0

block point_index_out_of_range:
  # An SPS record holds a one-digit point index, but a point built in code
  # may hold anything: the station key takes it clamped, as parseSEG's
  # survey join does, instead of a RangeDefect (or a wrapped index).
  var f: SpsFile
  for index in [100_000, -100_000, 3]:
    var p = initSpsPoint(pkSource)
    (p.line, p.point, p.index) = (1.0, 2.0, index)
    f.points.add p
  let shots = f.patchShots
  doAssert shots.len == 3 and shots[2].source.index == 3
  doAssert shots[0].source.index == high(int16) and shots[1].source.index == low(int16)

block shots_without_a_station_number:
  # A shot whose source line or point is unknown gets no records, like one
  # without coordinates, whatever the placement and the edge mode.
  let g = initSyntheticSurvey().geometry
  let good = PatchShot(record: 2, source: stationKey(1004, 120, 1))
  for bad in [StationKey(line: NaN, point: 120, index: 1),
      StationKey(line: 1004, point: NaN, index: 1), StationKey(line: Inf, point: 120, index: 1),
      StationKey(line: 1004, point: -Inf, index: 1)]:
    let shots = [PatchShot(record: 1, source: bad), good]
    for edge in [peRoll, peShift, peStatic]:
      for placement in [ppCoordinates, ppNumbers]:
        var t = initPatchTemplate(4, 11, edge)
        t.placement = placement
        let gen = g.generateRelations(shots, t)
        doAssert gen.len > 0, $edge & " " & $placement
        for r in gen: doAssert r.fieldRecord == 2, $edge & " " & $placement & " " & $bad
  # A finite number map that sends the source far outside the lines: no
  # receivers, and no conversion of 1e303 to an int.
  var t = initPatchTemplate(4, 11)
  t.placement = ppNumbers
  t.lineMap.scale = 1e300
  doAssert g.generateRelations([good], t).len == 0
  t.lineMap.scale = 1
  t.pointMap.scale = -1e300
  doAssert g.generateRelations([good], t).len == 0

const
  dir = currentSourcePath().parentDir.parentDir / "exampleFiles" / "sps"
  base = dir / "PGI_NNE_West_Morgantown_3D_Post261019_NAD27_SPCS_WV_N_USFeet"

block morgantown:
  if not (fileExists(base & ".sps") and fileExists(base & ".rps") and fileExists(base & ".xps")):
    echo "  skipped: Morgantown SPS files not present"
  else:
    var g = initSurveyGeometry()
    for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
    let p = describePatches(g)
    # the 11.7 box as a template: lines inside ±boxCrossline, slots inside ±boxInline
    let lines = 2 * int(round(p.boxCrossline / p.layout.lineSpacing))
    let channels = 2 * int(round(p.boxInline / p.layout.interval)) + 1
    var shots: seq[PatchShot]
    for r in p.records: shots.add PatchShot(record: r.record, source: r.source)
    when defined(release) or defined(danger): (let t0 = cpuTime())
    let gen = g.generateRelations(p.layout, shots, initPatchTemplate(lines, channels))
    when defined(release) or defined(danger):
      echo "  Morgantown generateRelations: ", formatFloat(cpuTime() - t0, ffDecimal, 2), " s"
    var f: SpsFile
    f.relations = gen
    var gg = initSurveyGeometry()
    gg.table = g.table
    gg.addSps(f)
    var ratios: seq[float64]
    for r in g.records:
      var a, b: HashSet[StationKey]
      for (_, rel) in g.relations(r): a.incl rel.receiver
      for (_, rel) in gg.relations(r): b.incl rel.receiver
      ratios.add float64(card(a * b)) / float64(max(1, card(a + b)))
    ratios.sort
    let q = spreadOf(ratios)
    var line = "  Morgantown " & $lines & " × " & $channels &
      " template vs XPS (listed ∩ generated / union, min/q1/median/q3/max):"
    for v in [q.min, q.q1, q.median, q.q3, q.max]: line.add " " & formatFloat(v, ffDecimal, 4)
    echo line
    doAssert ratios.len == 5519
