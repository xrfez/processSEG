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
## `k − channels div 2 .. k + channels div 2` (`channels` odd).

import std/math
import parseSEG

type
  SyntheticSurvey* = object
    receiverLines*, stations*: int
    interval*, lineSpacing*: float64
    sourceEvery*: int
    azimuthDeg*, x0*, y0*: float64
    lines*, channels*: int
    firstLine*, firstPoint*: int

func initSyntheticSurvey*(): SyntheticSurvey =
  SyntheticSurvey(receiverLines: 8, stations: 41, interval: 50, lineSpacing: 200,
    sourceEvery: 2, azimuthDeg: 90, x0: 0, y0: 0, lines: 4, channels: 11,
    firstLine: 1001, firstPoint: 100)

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
      var t = ShotTemplate(record: record, sourceLine: j, sourceStation: k,
        firstStation: max(0, k - s.channels div 2),
        lastStation: min(s.stations - 1, k + s.channels div 2))
      for i in j - s.lines div 2 + 1 .. j + s.lines div 2:
        if i in 0 ..< s.receiverLines: t.lines.add i
      result.add t
      inc record
      k += s.sourceEvery

proc spsFile*(s: SyntheticSurvey): SpsFile =
  ## R, S and X records of the survey.
  result.name = "synthetic"
  for i in 0 ..< s.receiverLines:
    for k in 0 ..< s.stations:
      var p = initSpsPoint(pkReceiver)
      p.line = float64(s.firstLine + i)
      p.lineName = $(s.firstLine + i)
      p.point = float64(s.firstPoint + k)
      p.pointName = $(s.firstPoint + k)
      (p.easting, p.northing) = s.position(float64(k) * s.interval, float64(i) * s.lineSpacing)
      p.elevation = 0
      result.points.add p
  for t in s.shots:
    var p = initSpsPoint(pkSource)
    p.line = float64(s.firstLine + t.sourceLine)
    p.lineName = $(s.firstLine + t.sourceLine)
    p.point = float64(s.firstPoint + t.sourceStation)
    p.pointName = $(s.firstPoint + t.sourceStation)
    (p.easting, p.northing) = s.position(float64(t.sourceStation) * s.interval,
      (float64(t.sourceLine) + 0.5) * s.lineSpacing)
    p.elevation = 0
    result.points.add p
    var channel = 1'i64
    for i in t.lines:
      var r = initSpsRelation()
      r.fieldRecord = t.record
      (r.srcLine, r.srcPoint) = (p.line, p.point)
      (r.srcLineName, r.srcPointName) = (p.lineName, p.pointName)
      r.fromChannel = channel
      r.toChannel = channel + int64(t.lastStation - t.firstStation)
      r.rcvLine = float64(s.firstLine + i)
      r.rcvLineName = $(s.firstLine + i)
      r.fromReceiver = float64(s.firstPoint + t.firstStation)
      r.toReceiver = float64(s.firstPoint + t.lastStation)
      result.relations.add r
      channel = r.toChannel + 1

proc geometry*(s: SyntheticSurvey): SurveyGeometry =
  result = initSurveyGeometry()
  result.addSps(s.spsFile)
