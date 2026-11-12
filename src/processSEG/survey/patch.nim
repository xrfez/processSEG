## Receiver patches of a survey: which receivers each shot was recorded on,
## described as a template (lines × channels) and as a distance box.
##
## ```nim
## var g = initSurveyGeometry()                   # parseSEG
## for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
## let p = describePatches(g)
## echo p                                         # "22 × 160 template, ..." or the box
## for r in p.records:
##   if r.clipped != {}: echo r.record, " clipped at ", r.clipped
## ```
##
## **Frame.** Distances are measured along the receiver lines (*inline*, the
## receiver-line azimuth, `ReceiverLayout.azimuthDeg`) and across them
## (*crossline*, 90° clockwise of it), relative to the record's source, in
## the survey's grid units. Azimuths are degrees clockwise from north (+y),
## as in `stack/binning`.
##
## **Layout.** `receiverLayout` measures the receiver lines once: the
## azimuth is the median bearing between consecutive stations of a line (in
## point order), the interval the median distance between them, the line
## spacing the median crossline distance between adjacent lines.
##
## **Records.** `describePatches` walks every X record (parseSEG
## `relations`): its receiver lines and the stations on each, its channels,
## the extent of its receivers (inline and crossline, each side), its
## maximum offset. A record's half-widths are its farther side in each
## direction, so a patch clipped on one side keeps its full width.
##
## **Survey.** The full patch is the median half-widths over the records.
## A record is *clipped* on a side when its receivers there stop short of
## the full patch by more than `clipTolerance` × the receiver interval
## (inline) or the line spacing (crossline): at the survey edges (roll on /
## roll off), or where receivers at the patch edge were dropped. (Whether
## the full patch would *fit* is not the test: short infill lines and
## ragged survey outlines would clip nearly every record of a real survey.)
## The template is taken from the unclipped records:
## *regular* when they all have the same number of lines and stations on
## every line (e.g. 22 × 160); otherwise the medians are nominal and the
## box describes the patch.

import std/[algorithm, math, sets, strutils, tables]
import parseSEG

type
  ReceiverLine* = object
    ## One receiver line in the layout frame.
    line*: float64                ## Line number.
    stations*: int                ## Receiver stations with coordinates.
    crossline*: float64           ## Mean crossline position of its stations.
    inlineMin*, inlineMax*: float64 ## Inline extent of its stations.

  ReceiverLayout* = object
    ## The receiver lines of a survey: direction, spacing and extent.
    azimuthDeg*: float64 ## Inline direction, degrees clockwise from north.
    interval*: float64   ## Median distance between consecutive stations of a line.
    lineSpacing*: float64 ## Median crossline distance between adjacent lines.
    lines*: seq[ReceiverLine] ## By ascending crossline position.

  PatchSide* = enum
    ## A side of a patch, relative to its source.
    psInlineBack = "inline back"       ## Towards decreasing inline position.
    psInlineForward = "inline forward" ## Along the azimuth.
    psCrosslineLeft = "crossline left" ## Anticlockwise of the azimuth.
    psCrosslineRight = "crossline right" ## Clockwise of the azimuth.

  PatchLine* = object
    ## The stations of one receiver line in a record.
    line*: float64
    stations*: int             ## Distinct stations (channels of one component).
    firstPoint*, lastPoint*: float64 ## Lowest and highest point number.

  RecordPatch* = object
    ## The receivers of one X record.
    record*: int64
    source*: StationKey
    sourceX*, sourceY*: float64 ## NaN when the source has no coordinates.
    channels*: int             ## Channels listed (all components).
    lines*: seq[PatchLine]     ## Ascending line number.
    noCoordinates*: int        ## Listed receiver stations without coordinates.
    inlineMin*, inlineMax*: float64 ## Receiver extent relative to the source (NaN: unknown).
    crosslineMin*, crosslineMax*: float64
    maxOffset*: float64
    clipped*: set[PatchSide]   ## Sides where the receivers stop short of the full patch.

  Spread* = object
    ## Five-number summary (nearest rank); NaN when empty.
    min*, q1*, median*, q3*, max*: float64

  PatchDescription* = object
    ## The patches of every X record of a survey, and their summary.
    layout*: ReceiverLayout
    records*: seq[RecordPatch] ## Ascending record number.
    lines*, channels*: Spread  ## Over all records.
    inlineHalfWidth*, crosslineHalfWidth*, maxOffset*: Spread ## Over records with coordinates.
    boxInline*, boxCrossline*: float64 ## The full patch: median half-widths.
    clippedRecords*: int
    noSource*: int             ## Records whose source has no coordinates.
    regular*: bool             ## Unclipped records share one lines × stations template.
    templateLines*, templateChannels*: int
      ## Lines × stations per line: exact when `regular`, else medians of the
      ## unclipped records (stations per line from channels / lines).

func inlineOf(azimuthDeg, x, y: float64): float64 {.inline.} =
  let a = degToRad(azimuthDeg)
  x * sin(a) + y * cos(a)

func crosslineOf(azimuthDeg, x, y: float64): float64 {.inline.} =
  let a = degToRad(azimuthDeg)
  x * cos(a) - y * sin(a)

func inlineOf*(layout: ReceiverLayout, x, y: float64): float64 {.inline.} =
  ## Position of (x, y) along the receiver lines.
  inlineOf(layout.azimuthDeg, x, y)

func crosslineOf*(layout: ReceiverLayout, x, y: float64): float64 {.inline.} =
  ## Position of (x, y) across the receiver lines (90° clockwise of the azimuth).
  crosslineOf(layout.azimuthDeg, x, y)

func median(v: var seq[float64]): float64 =
  if v.len == 0: return NaN
  v.sort
  v[(v.len - 1) div 2]

func spreadOf*(values: openArray[float64]): Spread =
  ## Min, quartiles (nearest rank) and max of the values that are not NaN.
  var v = newSeqOfCap[float64](values.len)
  for x in values:
    if not x.isNaN: v.add x
  if v.len == 0: return Spread(min: NaN, q1: NaN, median: NaN, q3: NaN, max: NaN)
  v.sort
  template at(q: float64): float64 = v[int(round(q * float64(v.high)))]
  Spread(min: v[0], q1: at(0.25), median: v[(v.len - 1) div 2], q3: at(0.75), max: v[^1])

proc lookup(t: Table[StationKey, Coordinates], s: StationKey, c: var Coordinates): bool =
  # parseSEG's rule: the exact station, then index 0 (any); a station with
  # no index (0) also matches index 1.
  let indices = [s.index, 0'i16, 1'i16]
  for n in 0 .. (if s.index == 0: 2 else: 1):
    let k = stationKey(s.line, s.point, indices[n])
    if k in t:
      c = t[k]
      return true

proc receiverLayout*(g: SurveyGeometry, azimuthDeg = NaN): ReceiverLayout =
  ## Measures the receiver lines of `g` (its R stations). The azimuth is
  ## measured unless given. Raises `ValueError` when no line has two
  ## stations with coordinates and no azimuth is given.
  var byLine: Table[float64, seq[(float64, Coordinates)]]
  for k, c in g.table.receivers:
    if not (c.x.isNaN or c.y.isNaN):
      byLine.mgetOrPut(k.line, @[]).add (k.point, c)
  var lineNumbers: seq[float64]
  for l in byLine.keys: lineNumbers.add l
  lineNumbers.sort
  var bearings, steps: seq[float64]
  for l in lineNumbers:
    var st = byLine[l]
    st.sort(proc (a, b: (float64, Coordinates)): int = cmp(a[0], b[0]))
    for i in 1 ..< st.len:
      let (dx, dy) = (st[i][1].x - st[i - 1][1].x, st[i][1].y - st[i - 1][1].y)
      let d = hypot(dx, dy)
      if d > 0:
        steps.add d
        bearings.add floorMod(radToDeg(arctan2(dx, dy)), 360.0)
    byLine[l] = st
  result.azimuthDeg = azimuthDeg
  if azimuthDeg.isNaN:
    if bearings.len == 0:
      raise newException(ValueError, "receiverLayout: no receiver line has two stations " &
        "with coordinates; give azimuthDeg")
    let reference = bearings[0] # unwrap around one bearing, so 359° and 1° agree
    for b in bearings.mitems: b = reference + floorMod(b - reference + 180.0, 360.0) - 180.0
    result.azimuthDeg = floorMod(median(bearings), 360.0)
  result.interval = median(steps)
  for l in lineNumbers:
    var line = ReceiverLine(line: l, stations: byLine[l].len, inlineMin: Inf, inlineMax: -Inf)
    var c = 0.0
    for (_, p) in byLine[l]:
      let a = inlineOf(result.azimuthDeg, p.x, p.y)
      line.inlineMin = min(line.inlineMin, a)
      line.inlineMax = max(line.inlineMax, a)
      c += crosslineOf(result.azimuthDeg, p.x, p.y)
    line.crossline = c / float64(line.stations)
    result.lines.add line
  result.lines.sort(proc (a, b: ReceiverLine): int = cmp(a.crossline, b.crossline))
  var gaps: seq[float64]
  for i in 1 ..< result.lines.len:
    gaps.add result.lines[i].crossline - result.lines[i - 1].crossline
  result.lineSpacing = median(gaps)

proc describeRecord(g: SurveyGeometry, layout: ReceiverLayout, record: int64): RecordPatch =
  result = RecordPatch(record: record, sourceX: NaN, sourceY: NaN, inlineMin: NaN,
    inlineMax: NaN, crosslineMin: NaN, crosslineMax: NaN, maxOffset: NaN)
  var first = true
  var sa, sc = NaN
  var seen: HashSet[StationKey] # distinct stations; multi-component channels share one
  for (_, rel) in g.relations(record):
    inc result.channels
    if first:
      first = false
      result.source = rel.source
      var c: Coordinates
      if g.table.sources.lookup(rel.source, c) and not (c.x.isNaN or c.y.isNaN):
        (result.sourceX, result.sourceY) = (c.x, c.y)
        sa = layout.inlineOf(c.x, c.y)
        sc = layout.crosslineOf(c.x, c.y)
        (result.inlineMin, result.inlineMax) = (Inf, -Inf)
        (result.crosslineMin, result.crosslineMax, result.maxOffset) = (Inf, -Inf, 0.0)
    let r = rel.receiver
    if seen.containsOrIncl(r): continue
    var j = result.lines.high
    while j >= 0 and result.lines[j].line != r.line: dec j
    if j < 0:
      result.lines.add PatchLine(line: r.line, firstPoint: r.point, lastPoint: r.point)
      j = result.lines.high
    inc result.lines[j].stations
    result.lines[j].firstPoint = min(result.lines[j].firstPoint, r.point)
    result.lines[j].lastPoint = max(result.lines[j].lastPoint, r.point)
    var c: Coordinates
    if not g.table.receivers.lookup(r, c) or c.x.isNaN or c.y.isNaN:
      inc result.noCoordinates
    elif not sa.isNaN:
      let a = layout.inlineOf(c.x, c.y) - sa
      let x = layout.crosslineOf(c.x, c.y) - sc
      result.inlineMin = min(result.inlineMin, a)
      result.inlineMax = max(result.inlineMax, a)
      result.crosslineMin = min(result.crosslineMin, x)
      result.crosslineMax = max(result.crosslineMax, x)
      result.maxOffset = max(result.maxOffset, hypot(c.x - result.sourceX, c.y - result.sourceY))
  if result.inlineMin == Inf: # a source with coordinates, but no receiver with them
    (result.inlineMin, result.inlineMax, result.crosslineMin, result.crosslineMax,
      result.maxOffset) = (NaN, NaN, NaN, NaN, NaN)
  result.lines.sort(proc (a, b: PatchLine): int = cmp(a.line, b.line))

func halfWidths(r: RecordPatch): (float64, float64) =
  (max(-r.inlineMin, r.inlineMax), max(-r.crosslineMin, r.crosslineMax))

func clippedSides(r: RecordPatch, layout: ReceiverLayout, boxInline, boxCrossline,
    tolerance: float64): set[PatchSide] =
  ## Sides where the record's receivers stop short of the full patch.
  if r.inlineMin.isNaN or boxInline.isNaN or boxCrossline.isNaN: return
  let tolI = tolerance * layout.interval
  let tolC = tolerance * layout.lineSpacing
  if -r.inlineMin < boxInline - tolI: result.incl psInlineBack
  if r.inlineMax < boxInline - tolI: result.incl psInlineForward
  if -r.crosslineMin < boxCrossline - tolC: result.incl psCrosslineLeft
  if r.crosslineMax < boxCrossline - tolC: result.incl psCrosslineRight

proc describePatches*(g: SurveyGeometry, layout: ReceiverLayout,
    clipTolerance = 0.75): PatchDescription =
  ## Describes the patch of every X record of `g` in `layout`'s frame (see
  ## the module documentation). `clipTolerance` is in receiver intervals
  ## (inline) and line spacings (crossline); below 1, so a patch one
  ## station short counts as clipped.
  if clipTolerance.isNaN or clipTolerance < 0:
    raise newException(ValueError, "describePatches: clipTolerance must be ≥ 0")
  result.layout = layout
  for r in g.records: result.records.add g.describeRecord(layout, r)
  var lines, channels, inl, xl, off: seq[float64]
  for r in result.records:
    lines.add float64(r.lines.len)
    channels.add float64(r.channels)
    if r.sourceX.isNaN: inc result.noSource
    let (a, c) = r.halfWidths
    inl.add a
    xl.add c
    off.add r.maxOffset
  result.lines = spreadOf(lines)
  result.channels = spreadOf(channels)
  result.inlineHalfWidth = spreadOf(inl)
  result.crosslineHalfWidth = spreadOf(xl)
  result.maxOffset = spreadOf(off)
  result.boxInline = result.inlineHalfWidth.median
  result.boxCrossline = result.crosslineHalfWidth.median
  var full: seq[int]
  for i, r in result.records.mpairs:
    r.clipped = r.clippedSides(layout, result.boxInline, result.boxCrossline, clipTolerance)
    if r.clipped != {}: inc result.clippedRecords
    elif not r.inlineMin.isNaN: full.add i
  if full.len == 0: # nothing measurable as unclipped: describe every record
    for i in 0 ..< result.records.len: full.add i
  if full.len > 0:
    let r0 = result.records[full[0]]
    let stations = if r0.lines.len > 0: r0.lines[0].stations else: 0
    result.regular = true
    var fl, fc: seq[float64]
    for i in full:
      let r = result.records[i]
      fl.add float64(r.lines.len)
      fc.add float64(r.channels)
      if r.lines.len != r0.lines.len: result.regular = false
      for l in r.lines:
        if l.stations != stations: result.regular = false
    if result.regular:
      (result.templateLines, result.templateChannels) = (r0.lines.len, stations)
    else:
      let ml = median(fl)
      result.templateLines = int(ml)
      result.templateChannels = if ml > 0: int(round(median(fc) / ml)) else: 0

proc describePatches*(g: SurveyGeometry, azimuthDeg = NaN,
    clipTolerance = 0.75): PatchDescription =
  ## `describePatches` in the measured receiver layout of `g` (the azimuth
  ## measured unless given).
  g.describePatches(g.receiverLayout(azimuthDeg), clipTolerance)

func `$`*(s: Spread): string =
  ## "min/q1/median/q3/max".
  template f(x: float64): string = formatFloat(x, ffDecimal, 1)
  f(s.min) & "/" & f(s.q1) & "/" & f(s.median) & "/" & f(s.q3) & "/" & f(s.max)

func `$`*(p: PatchDescription): string =
  ## One line: the template (or nominal size), the box, and the clipped records.
  let kind = if p.regular: "template" else: "nominal (irregular)"
  $p.templateLines & " × " & $p.templateChannels & " " & kind & ", box ±" &
    formatFloat(p.boxInline, ffDecimal, 1) & " inline × ±" &
    formatFloat(p.boxCrossline, ffDecimal, 1) & " crossline at " &
    formatFloat(p.layout.azimuthDeg, ffDecimal, 2) & "°, " & $p.records.len & " records, " &
    $p.clippedRecords & " clipped"
