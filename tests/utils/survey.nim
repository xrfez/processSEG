## Synthetic SPS surveys for the survey tests: an orthogonal grid of receiver
## lines, source points between them, and X records cut from a lines ×
## channels template centred on each source and clipped at the survey edges
## (roll on / roll off).
##
## Layout before rotation: receiver line `i` (number `firstLine + i`) runs
## along +x at y = i·lineSpacing, station `k` (point `firstPoint + k`) at
## x = k·interval. Source line `j` lies half-way between receiver lines `j`
## and `j + 1`, its point `k` at x = k·interval (every `sourceEvery`
## stations). The whole survey is then rotated so the receiver lines run at
## `azimuthDeg` (clockwise from north) and moved to `(x0, y0)`.
##
## The template takes receiver lines `j − lines/2 + 1 .. j + lines/2`
## (`lines` even: as many on each side of the source) and stations
## `k − channels div 2 .. k + channels div 2` (`channels` odd), clipped at
## the edges (`edge = peRoll`), shifted inside (`peShift`), or every
## receiver (`peStatic`).
##
## Numbers: line `i` is `firstLine + i·lineStep`, station `k` is point
## `firstPoint + k·pointStep` (steps may be negative: numbered against the
## azimuth). X channels run 1..n in ascending line and point order.

import std/[algorithm, math]
import parseSEG
import processSEG/survey/patch

type
  SyntheticSurvey* = object
    receiverLines*, stations*: int
    interval*, lineSpacing*: float64
    sourceEvery*: int
    azimuthDeg*, x0*, y0*: float64
    lines*, channels*: int
    firstLine*, firstPoint*: int
    lineStep*, pointStep*: int
    edge*: PatchEdge

func initSyntheticSurvey*(): SyntheticSurvey =
  SyntheticSurvey(receiverLines: 8, stations: 41, interval: 50, lineSpacing: 200,
    sourceEvery: 2, azimuthDeg: 90, x0: 0, y0: 0, lines: 4, channels: 11,
    firstLine: 1001, firstPoint: 100, lineStep: 1, pointStep: 1, edge: peRoll)

func lineNumber*(s: SyntheticSurvey, i: int): float64 = float64(s.firstLine + i * s.lineStep)
func pointNumber*(s: SyntheticSurvey, k: int): float64 = float64(s.firstPoint + k * s.pointStep)

func position*(s: SyntheticSurvey, u, v: float64): (float64, float64) =
  ## Grid position (u along the lines, v across: line i at v = i·lineSpacing)
  ## → (x, y). v grows 90° anticlockwise of the azimuth.
  let a = degToRad(s.azimuthDeg)
  (s.x0 + u * sin(a) - v * cos(a), s.y0 + u * cos(a) + v * sin(a))

type ShotTemplate* = object
  record*: int64
  sourceLine*, sourceStation*: int ## j, k
  lines*: seq[int]                 ## receiver line indices in the patch
  firstStation*, lastStation*: int ## station range (clipped)

func shots*(s: SyntheticSurvey): seq[ShotTemplate] =
  ## Every shot with its (clipped) patch, records numbered from 1.
  var record = 1'i64
  for j in 0 .. s.receiverLines - 2:
    var k = 0
    while k < s.stations:
      var t = ShotTemplate(record: record, sourceLine: j, sourceStation: k)
      case s.edge
      of peRoll:
        t.firstStation = max(0, k - s.channels div 2)
        t.lastStation = min(s.stations - 1, k + s.channels div 2)
        for i in j - s.lines div 2 + 1 .. j + s.lines div 2:
          if i in 0 ..< s.receiverLines: t.lines.add i
      of peShift:
        t.firstStation = clamp(k - s.channels div 2, 0, s.stations - s.channels)
        t.lastStation = t.firstStation + s.channels - 1
        let first = clamp(j - s.lines div 2 + 1, 0, s.receiverLines - s.lines)
        for i in first ..< first + s.lines: t.lines.add i
      of peStatic:
        (t.firstStation, t.lastStation) = (0, s.stations - 1)
        for i in 0 ..< s.receiverLines: t.lines.add i
      result.add t
      inc record
      k += s.sourceEvery

proc spsFile*(s: SyntheticSurvey): SpsFile =
  ## R, S and X records of the survey.
  result.name = "synthetic"
  for i in 0 ..< s.receiverLines:
    for k in 0 ..< s.stations:
      var p = initSpsPoint(pkReceiver)
      p.line = s.lineNumber(i)
      p.point = s.pointNumber(k)
      (p.easting, p.northing) = s.position(float64(k) * s.interval, float64(i) * s.lineSpacing)
      p.elevation = 0
      result.points.add p
  for t in s.shots:
    var p = initSpsPoint(pkSource)
    p.line = s.lineNumber(t.sourceLine)
    p.point = s.pointNumber(t.sourceStation)
    (p.easting, p.northing) = s.position(float64(t.sourceStation) * s.interval,
      (float64(t.sourceLine) + 0.5) * s.lineSpacing)
    p.elevation = 0
    result.points.add p
    var lines = t.lines
    lines.sort(proc (a, b: int): int = cmp(s.lineNumber(a), s.lineNumber(b)))
    var channel = 1'i64
    for i in lines:
      var r = initSpsRelation()
      r.fieldRecord = t.record
      (r.srcLine, r.srcPoint) = (p.line, p.point)
      r.fromChannel = channel
      r.toChannel = channel + int64(t.lastStation - t.firstStation)
      r.rcvLine = s.lineNumber(i)
      let (a, b) = (s.pointNumber(t.firstStation), s.pointNumber(t.lastStation))
      (r.fromReceiver, r.toReceiver) = (min(a, b), max(a, b))
      result.relations.add r
      channel = r.toChannel + 1

proc geometry*(s: SyntheticSurvey): SurveyGeometry =
  result = initSurveyGeometry()
  result.addSps(s.spsFile)
