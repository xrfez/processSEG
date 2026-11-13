## Receiver patches of a survey: which receivers each shot was recorded on,
## described as a template (lines × channels) and as a distance box, and X
## records generated from a template (`generateRelations`).
##
## ```nim
## var g = initSurveyGeometry()                   # parseSEG
## for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
## let p = describePatches(g)
## echo p                                         # "22 × 160 template, ..." or the box
## for r in p.records:
##   if r.clipped != {}: echo r.record, " clipped at ", r.clipped
## let xs = g.generateRelations(f.patchShots, initPatchTemplate(22, 160))
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
## `relationRuns`): its receiver lines and the stations on each, its channels,
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
##
## **Generation.** `generateRelations` lays a `PatchTemplate` over the R
## stations for each `PatchShot` (from an S file's order with `patchShots`,
## or the caller's record numbers): roll (clip), shift or static at the
## edges, the source placed by coordinates or by line and point numbers.
## The result is parseSEG `SpsRelation`s (D14: parseSEG writes them).

import std/[algorithm, math, sets, strutils, tables]
import parseSEG
import ./relwalk

func finite(x: float64): bool {.inline.} = x.classify notin {fcNan, fcInf, fcNegInf}

type
  LineStation* = object
    ## One receiver station of a line in the layout frame.
    point*: float64
    index*: int16                ## Point index, as the R record has it.
    inline*, crossline*: float64 ## Position along and across the receiver lines.

  ReceiverLine* = object
    ## One receiver line in the layout frame.
    line*: float64                  ## Line number.
    stations*: seq[LineStation]     ## Stations with coordinates, by point (then index).
    pointStep*: float64             ## Median point-number increment between its stations (NaN: one station).
    crossline*: float64             ## Mean crossline position of its stations.
    inlineMin*, inlineMax*: float64 ## Inline extent of its stations.
    short*: bool
      ## Fewer stations than `minLineStations`: an infill or relocated
      ## line, not a template line, and left out of the line spacing.

  ReceiverLayout* = object
    ## The receiver lines of a survey: direction, spacing and extent.
    azimuthDeg*: float64      ## Inline direction, degrees clockwise from north.
    interval*: float64        ## Median distance between consecutive stations of a line.
    lineSpacing*: float64     ## Median crossline distance between adjacent lines that are not short.
    minLineStations*: int     ## Stations a line needs not to be `short`.
    lines*: seq[ReceiverLine] ## By ascending crossline position.

  PatchSide* = enum
    ## A side of a patch, relative to its source.
    psInlineBack = "inline back"         ## Towards decreasing inline position.
    psInlineForward = "inline forward"   ## Along the azimuth.
    psCrosslineLeft = "crossline left"   ## Anticlockwise of the azimuth.
    psCrosslineRight = "crossline right" ## Clockwise of the azimuth.

  PatchLine* = object
    ## The stations of one receiver line in a record.
    line*: float64
    stations*: int                   ## Distinct stations (channels of one component).
    firstPoint*, lastPoint*: float64 ## Lowest and highest point number.

  RecordPatch* = object
    ## The receivers of one X record.
    record*: int64
    source*: StationKey
    sourceX*, sourceY*: float64     ## NaN when the source has no coordinates.
    channels*: int                  ## Channels listed (all components).
    lines*: seq[PatchLine]          ## Ascending line number.
    noCoordinates*: int             ## Listed receiver stations without coordinates.
    inlineMin*, inlineMax*: float64 ## Receiver extent relative to the source (NaN: unknown).
    crosslineMin*, crosslineMax*: float64
    maxOffset*: float64
    clipped*: set[PatchSide]        ## Sides where the receivers stop short of the full patch.

  Spread* = object
    ## Five-number summary (nearest rank); NaN when empty.
    min*, q1*, median*, q3*, max*: float64

  PatchDescription* = object
    ## The patches of every X record of a survey, and their summary.
    layout*: ReceiverLayout
    records*: seq[RecordPatch]                                ## Ascending record number.
    lines*, channels*: Spread                                 ## Over all records.
    inlineHalfWidth*, crosslineHalfWidth*, maxOffset*: Spread ## Over records with coordinates.
    boxInline*, boxCrossline*: float64                        ## The full patch: median half-widths.
    clippedRecords*: int
    noSource*: int                                            ## Records whose source has no coordinates.
    regular*: bool ## Unclipped records share one lines × stations template.
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

proc receiverLayout*(g: SurveyGeometry, azimuthDeg = NaN, interval = NaN, lineSpacing = NaN,
    minLineStations = 0): ReceiverLayout =
  ## Measures the receiver lines of `g` (its R stations). The azimuth, the
  ## station interval and the line spacing are measured unless given (> 0).
  ## A line with fewer than `minLineStations` stations is `short` (0: half
  ## the median line's count). Raises `ValueError` for a given value that is
  ## not positive and finite, or when a value cannot be measured (no line with
  ## two stations, or one line only) and is not given.
  for (v, name) in [(interval, "interval"), (lineSpacing, "lineSpacing")]:
    if not v.isNaN and not (v > 0 and v < Inf):
      raise newException(ValueError, "receiverLayout: " & name & " must be > 0 and finite")
  if not azimuthDeg.isNaN and abs(azimuthDeg) == Inf:
    raise newException(ValueError, "receiverLayout: azimuthDeg must be finite")
  if minLineStations < 0:
    raise newException(ValueError, "receiverLayout: minLineStations must be ≥ 0")
  var byLine: Table[float64, seq[(LineStation, Coordinates)]]
  for k, c in g.table.receivers:
    if finite(c.x) and finite(c.y):
      byLine.mgetOrPut(k.line, @[]).add (LineStation(point: k.point, index: k.index), c)
  var lineNumbers: seq[float64]
  for l in byLine.keys: lineNumbers.add l
  lineNumbers.sort
  var bearings, steps: seq[float64]
  for l in lineNumbers:
    var st = byLine[l]
    st.sort(proc (a, b: (LineStation, Coordinates)): int =
      result = cmp(a[0].point, b[0].point)
      if result == 0: result = cmp(a[0].index, b[0].index))
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
  result.interval = if interval.isNaN: median(steps) else: interval
  if result.interval.isNaN:
    raise newException(ValueError, "receiverLayout: no receiver line has two stations " &
      "with coordinates; give interval")
  var counts: seq[float64]
  for l in lineNumbers: counts.add float64(byLine[l].len)
  result.minLineStations = if minLineStations > 0: minLineStations
                           else: max(1, int(ceil(median(counts) / 2)))
  for l in lineNumbers:
    var line = ReceiverLine(line: l, inlineMin: Inf, inlineMax: -Inf, pointStep: NaN)
    var c = 0.0
    var pointSteps: seq[float64]
    for (st, p) in byLine[l]:
      var st = st
      st.inline = inlineOf(result.azimuthDeg, p.x, p.y)
      st.crossline = crosslineOf(result.azimuthDeg, p.x, p.y)
      line.inlineMin = min(line.inlineMin, st.inline)
      line.inlineMax = max(line.inlineMax, st.inline)
      c += st.crossline
      if line.stations.len > 0 and st.point > line.stations[^1].point:
        pointSteps.add st.point - line.stations[^1].point
      line.stations.add st
    line.crossline = c / float64(line.stations.len)
    line.pointStep = median(pointSteps)
    line.short = line.stations.len < result.minLineStations
    result.lines.add line
  result.lines.sort(proc (a, b: ReceiverLine): int = cmp(a.crossline, b.crossline))
  if not lineSpacing.isNaN:
    result.lineSpacing = lineSpacing
  else:
    var gaps: seq[float64]
    var last = NaN
    for line in result.lines:
      if not line.short:
        if not last.isNaN: gaps.add line.crossline - last
        last = line.crossline
    result.lineSpacing = median(gaps)
    if result.lineSpacing.isNaN:
      raise newException(ValueError, "receiverLayout: fewer than two receiver lines; " &
        "give lineSpacing")

proc describeRecord(g: SurveyGeometry, layout: ReceiverLayout, record: int64): RecordPatch =
  result = RecordPatch(record: record, sourceX: NaN, sourceY: NaN, inlineMin: NaN,
    inlineMax: NaN, crosslineMin: NaN, crosslineMax: NaN, maxOffset: NaN)
  var first = true
  var sa, sc = NaN
  var seen: HashSet[StationKey] # distinct stations; multi-component channels share one
  for st in g.stationChannels(record):
    let rel = st.relation
    result.channels += st.channels
    if first:
      first = false
      result.source = rel.source
      var c: Coordinates
      if g.table.sources.lookup(rel.source, c) and finite(c.x) and finite(c.y):
        (result.sourceX, result.sourceY) = (c.x, c.y)
        sa = layout.inlineOf(c.x, c.y)
        sc = layout.crosslineOf(c.x, c.y)
        (result.inlineMin, result.inlineMax) = (Inf, -Inf)
        (result.crosslineMin, result.crosslineMax, result.maxOffset) = (Inf, -Inf, 0.0)
    let r = rel.receiver
    if not seen.containsOrIncl(r):
      var j = result.lines.high
      while j >= 0 and result.lines[j].line != r.line: dec j
      if j < 0:
        result.lines.add PatchLine(line: r.line, firstPoint: r.point, lastPoint: r.point)
        j = result.lines.high
      inc result.lines[j].stations
      result.lines[j].firstPoint = min(result.lines[j].firstPoint, r.point)
      result.lines[j].lastPoint = max(result.lines[j].lastPoint, r.point)
      var c: Coordinates
      if not g.table.receivers.lookup(r, c) or not (finite(c.x) and finite(c.y)):
        inc result.noCoordinates
      elif not sa.isNaN:
        let a = layout.inlineOf(c.x, c.y) - sa
        let x = layout.crosslineOf(c.x, c.y) - sc
        result.inlineMin = min(result.inlineMin, a)
        result.inlineMax = max(result.inlineMax, a)
        result.crosslineMin = min(result.crosslineMin, x)
        result.crosslineMax = max(result.crosslineMax, x)
        result.maxOffset = max(result.maxOffset,
          hypot(c.x - result.sourceX, c.y - result.sourceY))
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

# ------------------------------------------------------------- generation --

type
  PatchEdge* = enum
    ## What a template patch does at the survey edges.
    peRoll = "roll"     ## Centred on the source, clipped at the edges (roll on / roll off).
    peShift = "shift"   ## Kept whole, shifted inside the survey: the source is off-centre.
    peStatic = "static" ## One fixed spread for every shot (`PatchTemplate.staticBox`).

  PatchPlacement* = enum
    ## How the source's place in the receiver grid is found.
    ppCoordinates = "coordinates" ## By position: lines by crossline, stations by inline.
    ppNumbers = "numbers"         ## By arithmetic on the line and point numbers (`NumberMap`).

  NumberMap* = object
    ## receiver value = `scale` × source value + `offset`.
    scale* = 1.0
    offset* = 0.0

  PatchBox* = object
    ## A box in the layout frame, absolute positions; everything by default.
    inlineMin* = -Inf
    inlineMax* = Inf
    crosslineMin* = -Inf
    crosslineMax* = Inf

  PatchTemplate* = object
    ## A lines × channels patch (see `generateRelations`).
    lines*: int               ## Template lines (`peRoll`, `peShift`).
    channels*: int            ## Station slots per line.
    lineOffset*: float64
      ## Patch centre relative to the source, in lines: towards higher
      ## crossline (`ppCoordinates`) or higher line numbers (`ppNumbers`).
    stationOffset*: float64
      ## Patch centre relative to the source, in stations: along the azimuth
      ## (`ppCoordinates`) or towards higher points (`ppNumbers`); e.g.
      ## `channels / 2` for an off-end spread.
    edge* = peRoll
    staticBox*: PatchBox ## `peStatic`: the receivers inside (default: all).
    placement* = ppCoordinates
    lineMap*, pointMap*: NumberMap
      ## `ppNumbers`: the (fractional) receiver line and point at the source,
      ## from its line and point numbers.
    # Distance field (11.9): limits around the source; Inf: none.
    inlineHalfWidth* = Inf ## Box half-width along `boxAzimuthDeg`.
    crosslineHalfWidth* = Inf ## Box half-width across it (90° clockwise).
    boxAzimuthDeg* = NaN ## The box's inline direction; NaN: the receiver-line azimuth.
    inlineDistance*, crosslineDistance*: float64
      ## Box (and radius) centre relative to the source, in distance along and
      ## across `boxAzimuthDeg`.
    radius* = Inf ## Distance from the box centre.
    maxOffset* = Inf ## Distance from the source.

  PatchShot* = object
    ## A field record to generate: its number and source station.
    record*: int64
    source*: StationKey

func initPatchTemplate*(lines, channels: int, edge = peRoll): PatchTemplate =
  ## A centred `lines × channels` template.
  PatchTemplate(lines: lines, channels: channels, edge: edge)

func initDistancePatch*(inlineHalfWidth, crosslineHalfWidth: float64, edge = peRoll,
    radius = Inf, maxOffset = Inf): PatchTemplate =
  ## A distance field alone: every receiver inside the box (and the radius
  ## and maximum offset), centred on the source.
  PatchTemplate(edge: edge, inlineHalfWidth: inlineHalfWidth,
    crosslineHalfWidth: crosslineHalfWidth, radius: radius, maxOffset: maxOffset)

func hasTemplate*(t: PatchTemplate): bool {.inline.} =
  ## The template part is set (lines or channels).
  t.lines != 0 or t.channels != 0

func hasDistance*(t: PatchTemplate): bool {.inline.} =
  ## A distance limit is set.
  t.inlineHalfWidth < Inf or t.crosslineHalfWidth < Inf or t.radius < Inf or t.maxOffset < Inf

func stationIndex(i: int): int16 {.inline.} =
  # A point index as the part of a station key. Survey records hold one
  # digit, but a point built in code may hold any int: out of range it
  # clamps, as parseSEG's survey join clamps it (its `idx`, which is not
  # exported), so both make the same key of the same point.
  int16(clamp(i, int(low(int16)), int(high(int16))))

proc patchShots*(f: SpsFile, firstRecord = 1'i64): seq[PatchShot] =
  ## One shot per S record with a numeric line and point, in file order,
  ## numbered from `firstRecord`. A point index beyond an `int16` is
  ## clamped, as parseSEG clamps it when it adds the file to a geometry.
  var record = firstRecord
  for p in f.points:
    if p.kind == pkSource and not (p.line.isNaN or p.point.isNaN):
      result.add PatchShot(record: record,
        source: stationKey(p.line, p.point, stationIndex(p.index)))
      inc record

proc patchShots*(g: SurveyGeometry, firstRecord = 1'i64): seq[PatchShot] =
  ## One shot per source station of `g`, in line, point, index order,
  ## numbered from `firstRecord`.
  var keys: seq[StationKey]
  for k in g.table.sources.keys: keys.add k
  keys.sort(proc (a, b: StationKey): int =
    result = cmp(a.line, b.line)
    if result == 0: result = cmp(a.point, b.point)
    if result == 0: result = cmp(a.index, b.index))
  for i, k in keys: result.add PatchShot(record: firstRecord + int64(i), source: k)

func roundUp(x: float64): int {.inline.} =
  ## Nearest integer, ties up (with slack for rounding error). A number map
  ## can send a source anywhere: beyond ±1e15 the result stays there, where
  ## the conversion to an int is still defined (NaN: the low end).
  const far = 1_000_000_000_000_000
  let y = floor(x + 0.5 + 1e-9)
  if y >= float64(far): far
  elif y > -float64(far): int(y)
  else: -far

func fractionalIndex(xs: openArray[float64], v, step: float64): float64 =
  ## Fractional position of `v` among ascending `xs`, extrapolated by `step`
  ## (NaN for a NaN `v`).
  if not (v > xs[0]): return (v - xs[0]) / step
  if v >= xs[^1]: return float64(xs.high) + (v - xs[^1]) / step
  let i = xs.upperBound(v) - 1
  float64(i) + (v - xs[i]) / (xs[i + 1] - xs[i])

func slotOf(line: ReceiverLine, i: int): int {.inline.} =
  if line.pointStep.isNaN: 0
  else: int(round((line.stations[i].point - line.stations[0].point) / line.pointStep))

func slotAt(line: ReceiverLine, inline, interval: float64): float64 =
  ## Fractional slot of an inline position, from the stations' positions
  ## (extrapolated by one slot per interval).
  let st = line.stations
  let d = if st[^1].inline >= st[0].inline: 1.0 else: -1.0
  let a = d * inline
  if st.len == 1 or a <= d * st[0].inline:
    return float64(line.slotOf(0)) + (a - d * st[0].inline) / interval
  if a >= d * st[^1].inline:
    return float64(line.slotOf(st.high)) + (a - d * st[^1].inline) / interval
  var lo = 0 # last station at or before a (positions assumed monotone)
  var hi = st.high
  while hi - lo > 1:
    let m = (lo + hi) div 2
    if d * st[m].inline <= a: lo = m else: hi = m
  let (a0, a1) = (d * st[lo].inline, d * st[hi].inline)
  let f = if a1 > a0: (a - a0) / (a1 - a0) else: 0.0
  float64(line.slotOf(lo)) + f * float64(line.slotOf(hi) - line.slotOf(lo))

type Picked = object
  line, step: float64
  st: LineStation

proc validate(t: PatchTemplate) =
  if t.edge != peStatic:
    if not (t.hasTemplate or t.hasDistance):
      raise newException(ValueError, "generateRelations: no template and no distance limit")
    if t.hasTemplate and (t.lines < 1 or t.channels < 1 or t.lines > 1_000_000 or
        t.channels > 1_000_000):
      raise newException(ValueError, "generateRelations: template lines and channels must be " &
        "in 1 .. 1 000 000")
    if t.edge == peShift and t.hasTemplate and t.hasDistance:
      raise newException(ValueError, "generateRelations: peShift takes a template or " &
        "distance limits, not both")
  for v in [t.lineOffset, t.stationOffset, t.lineMap.scale, t.lineMap.offset,
      t.pointMap.scale, t.pointMap.offset, t.inlineDistance, t.crosslineDistance]:
    if v.isNaN or abs(v) == Inf:
      raise newException(ValueError, "generateRelations: offsets and number maps must be finite")
  for v in [t.inlineHalfWidth, t.crosslineHalfWidth, t.radius, t.maxOffset]:
    if v.isNaN or v < 0:
      raise newException(ValueError, "generateRelations: distance limits must be ≥ 0 (Inf: none)")
  if not t.boxAzimuthDeg.isNaN and abs(t.boxAzimuthDeg) == Inf:
    raise newException(ValueError, "generateRelations: boxAzimuthDeg must be finite or NaN")

type
  Placed = object
    pick: Picked
    x, y: float64
    a, c: float64 # in the box frame

  LineExtent = object
    ## Where one layout line's stations lie in the layout frame.
    inlineMin, inlineMax, crosslineMin, crosslineMax: float64
    inlineDir: int ## 1: inline never decreases along the line; −1: never increases; 0: neither.

  DistanceField = object
    az: float64                     # box azimuth
    stations: seq[Placed]           # every layout station, line by line, with its grid position
    aMin, aMax, cMin, cMax: float64 # their extent in the box frame
                                    # For finding the stations near a box without visiting them all:
    lineStart: seq[int]             # per layout line, into `stations`; one more entry
    extents: seq[LineExtent]        # per layout line
    crossBefore: seq[float64]       # the highest crosslineMax of the lines up to each
    crossAfter: seq[float64]        # the lowest crosslineMin of the lines from each
    reachInline, reachCrossline: float64
      # The box's half extent along and across the receiver lines (Inf: none).

func xyOf(layout: ReceiverLayout, st: LineStation): (float64, float64) {.inline.} =
  let a = degToRad(layout.azimuthDeg)
  (st.inline * sin(a) + st.crossline * cos(a), st.inline * cos(a) - st.crossline * sin(a))

func extentOf(line: ReceiverLine): LineExtent =
  # A station without a position (NaN, in a layout built by hand) makes the
  # line unbounded and unordered: it is then searched whole.
  result = LineExtent(inlineMin: Inf, inlineMax: -Inf, crosslineMin: Inf, crosslineMax: -Inf)
  var (up, down, known) = (true, true, true)
  for i, st in line.stations:
    if st.inline.isNaN or st.crossline.isNaN: known = false
    result.inlineMin = min(result.inlineMin, st.inline)
    result.inlineMax = max(result.inlineMax, st.inline)
    result.crosslineMin = min(result.crosslineMin, st.crossline)
    result.crosslineMax = max(result.crosslineMax, st.crossline)
    if i > 0:
      if st.inline < line.stations[i - 1].inline: up = false
      if st.inline > line.stations[i - 1].inline: down = false
  if not known:
    result = LineExtent(inlineMin: -Inf, inlineMax: Inf, crosslineMin: -Inf, crosslineMax: Inf)
  else:
    result.inlineDir = if up: 1 elif down: -1 else: 0

func reach(alongHalf, acrossHalf, cosine, sine: float64): float64 =
  # Half the extent of a box of half-widths `alongHalf` × `acrossHalf` on an
  # axis at the angle of `cosine`, `sine` to its own (0 · Inf counts as 0: an
  # unlimited side does not reach along an axis it is square to).
  template part(half, factor: float64): float64 =
    (if factor == 0: 0.0 else: half * abs(factor))
  part(alongHalf, cosine) + part(acrossHalf, sine)

proc initDistanceField(layout: ReceiverLayout, t: PatchTemplate): DistanceField =
  result.az = if t.boxAzimuthDeg.isNaN: layout.azimuthDeg else: t.boxAzimuthDeg
  (result.aMin, result.aMax, result.cMin, result.cMax) = (Inf, -Inf, Inf, -Inf)
  for line in layout.lines:
    result.lineStart.add result.stations.len
    result.extents.add extentOf(line)
    for st in line.stations:
      let (x, y) = layout.xyOf(st)
      let (a, c) = (inlineOf(result.az, x, y), crosslineOf(result.az, x, y))
      result.stations.add Placed(pick: Picked(line: line.line, step: line.pointStep, st: st),
        x: x, y: y, a: a, c: c)
      result.aMin = min(result.aMin, a)
      result.aMax = max(result.aMax, a)
      result.cMin = min(result.cMin, c)
      result.cMax = max(result.cMax, c)
  result.lineStart.add result.stations.len
  # The layout's lines are in ascending mean crossline, which their stations
  # only follow roughly: the running extremes are what a search may rely on.
  let n = layout.lines.len
  result.crossBefore = newSeq[float64](n)
  result.crossAfter = newSeq[float64](n)
  for i in 0 ..< n:
    result.crossBefore[i] = max(result.extents[i].crosslineMax,
      (if i > 0: result.crossBefore[i - 1] else: -Inf))
  for i in countdown(n - 1, 0):
    result.crossAfter[i] = min(result.extents[i].crosslineMin,
      (if i < n - 1: result.crossAfter[i + 1] else: Inf))
  let turn = degToRad(result.az - layout.azimuthDeg)
  result.reachInline = reach(t.inlineHalfWidth, t.crosslineHalfWidth, cos(turn), sin(turn))
  result.reachCrossline = reach(t.crosslineHalfWidth, t.inlineHalfWidth, cos(turn), sin(turn))

func inside(t: PatchTemplate, x, y, a, c, sx, sy, ca, cc, cx, cy: float64): bool {.inline.} =
  ## (x, y) — (a, c) in the box frame — within the limits: box centred at
  ## (ca, cc) in the box frame, radius around (cx, cy), offset from the
  ## source (sx, sy).
  const eps = 1e-9
  # A limit of Inf holds for every position the box test lets through (a NaN
  # position fails that one), so its distance is not worked out.
  abs(a - ca) <= t.inlineHalfWidth * (1 + eps) + eps and
    abs(c - cc) <= t.crosslineHalfWidth * (1 + eps) + eps and
    (t.radius == Inf or hypot(x - cx, y - cy) <= t.radius * (1 + eps) + eps) and
    (t.maxOffset == Inf or hypot(x - sx, y - sy) <= t.maxOffset * (1 + eps) + eps)

func centreOf(f: DistanceField, t: PatchTemplate, sx, sy: float64): (float64, float64) =
  ## The box centre in the box frame: the source plus the distances, shifted
  ## inside the receivers' extent for `peShift` (centred on it when the box
  ## is wider).
  var ca = inlineOf(f.az, sx, sy) + t.inlineDistance
  var cc = crosslineOf(f.az, sx, sy) + t.crosslineDistance
  if t.edge == peShift:
    template fit(c: var float64, lo, hi, hw: float64) =
      if hw < Inf:
        c = if hi - lo >= 2 * hw: clamp(c, lo + hw, hi - hw) else: (lo + hi) / 2
    fit(ca, f.aMin, f.aMax, t.inlineHalfWidth)
    fit(cc, f.cMin, f.cMax, t.crosslineHalfWidth)
  (ca, cc)

func xyOfBox(az, a, c: float64): (float64, float64) {.inline.} =
  let r = degToRad(az)
  (a * sin(r) + c * cos(r), a * cos(r) - c * sin(r))

template firstIndex(count: int, i, test: untyped): int =
  ## The first `i` in `0 ..< count` for which `test` holds, given that it
  ## holds from some index on (`count` when it never does).
  block:
    var lo = 0
    var hi = count
    while lo < hi:
      let i = (lo + hi) shr 1
      if test: hi = i else: lo = i + 1
    lo

proc pickInside(f: DistanceField, layout: ReceiverLayout, t: PatchTemplate, sx, sy, ca, cc,
    cx, cy: float64, picked: var seq[Picked]) =
  ## Every layout station within the limits (`inside`), found from the box
  ## instead of by visiting them all. The limits lie inside a rectangle along
  ## the receiver lines; the lines that can reach it are a range (by the
  ## running extremes of their crosslines), and so are the stations of a
  ## line whose inline positions run one way. Every candidate still passes
  ## `inside`, so the result is that of testing every station (1e5
  ## receivers, a box of 10 lines × 100 stations: 113 µs a shot before, 15
  ## now).
  let (bi, bc) = (layout.inlineOf(cx, cy), layout.crosslineOf(cx, cy))
  var (iLo, iHi) = (bi - f.reachInline, bi + f.reachInline)
  var (cLo, cHi) = (bc - f.reachCrossline, bc + f.reachCrossline)
  template narrow(centreI, centreC, half: float64) =
    iLo = max(iLo, centreI - half)
    iHi = min(iHi, centreI + half)
    cLo = max(cLo, centreC - half)
    cHi = min(cHi, centreC + half)
  if t.radius < Inf: narrow(bi, bc, t.radius)
  if t.maxOffset < Inf: narrow(layout.inlineOf(sx, sy), layout.crosslineOf(sx, sy), t.maxOffset)
  # Slack for the rounding between the two frames and for `inside`'s own
  # tolerance: far more than either, and it only adds candidates.
  template slack(v: float64): float64 = 1e-6 * (1 + abs(v))
  (iLo, iHi, cLo, cHi) = (iLo - slack(iLo), iHi + slack(iHi), cLo - slack(cLo), cHi + slack(cHi))
  let lines = f.extents.len
  let firstLine = firstIndex(lines, l, f.crossBefore[l] >= cLo)
  let endLine = firstIndex(lines, l, f.crossAfter[l] > cHi)
  for l in firstLine ..< endLine:
    let e = f.extents[l]
    if e.crosslineMax >= cLo and e.crosslineMin <= cHi and e.inlineMax >= iLo and
        e.inlineMin <= iHi:
      let base = f.lineStart[l]
      let count = f.lineStart[l + 1] - base
      var (a, b) = (0, count)
      if e.inlineDir > 0:
        a = firstIndex(count, i, f.stations[base + i].pick.st.inline >= iLo)
        b = firstIndex(count, i, f.stations[base + i].pick.st.inline > iHi)
      elif e.inlineDir < 0:
        a = firstIndex(count, i, f.stations[base + i].pick.st.inline <= iHi)
        b = firstIndex(count, i, f.stations[base + i].pick.st.inline < iLo)
      for i in base + a ..< base + b:
        let p = f.stations[i] # a copy: reading it field by field measured 15 % slower
        if t.inside(p.x, p.y, p.a, p.c, sx, sy, ca, cc, cx, cy): picked.add p.pick

func slotsAscend(line: ReceiverLine): bool =
  ## The slots of the line's stations never decrease: its points ascend and
  ## its step is positive (as `receiverLayout` makes them; a layout built by
  ## hand may not).
  if not (line.pointStep.isNaN or (line.pointStep > 0 and line.pointStep < Inf)):
    return false
  for i in 1 ..< line.stations.len:
    if not (line.stations[i].point >= line.stations[i - 1].point): return false
  true

proc pickTemplate(layout: ReceiverLayout, main, shortLines: seq[int], t: PatchTemplate,
    cl, sa, srcPoint: float64, ascending: var seq[int8], picked: var seq[Picked]) =
  ## Roll or shift: the template lines around fractional line `cl` among the
  ## `main` lines, the station slots around the source on each, then the
  ## `shortLines` inside. `ascending[line]` remembers whether a layout
  ## line's slots ascend (1) or not (−1), found on its first use (0).
  var first = roundUp(cl - float64(t.lines - 1) / 2 + t.lineOffset)
  if t.edge == peShift:
    first = if main.len >= t.lines: clamp(first, 0, main.len - t.lines) else: 0
  var amin = Inf
  var amax = -Inf
  var xmin = Inf
  var xmax = -Inf
  for li in max(first, 0) ..< min(first + t.lines, main.len):
    template line: untyped = layout.lines[main[li]]
    let u = if t.placement == ppNumbers:
              (if line.pointStep.isNaN: 0.0
               else: (t.pointMap.scale * srcPoint + t.pointMap.offset - line.stations[0].point) /
                 line.pointStep) + t.stationOffset
            else:
              let d = if line.stations[^1].inline >= line.stations[0].inline: 1.0 else: -1.0
              line.slotAt(sa, layout.interval) + d * t.stationOffset
    var s0 = roundUp(u - float64(t.channels - 1) / 2)
    if t.edge == peShift:
      let maxSlot = line.slotOf(line.stations.high)
      s0 = if maxSlot + 1 >= t.channels: clamp(s0, 0, maxSlot - t.channels + 1) else: 0
    let s1 = s0 + t.channels - 1
    xmin = min(xmin, line.crossline)
    xmax = max(xmax, line.crossline)
    # The stations in slots s0 .. s1. Slots ascend along a line, so they are
    # a run found by bisection, not by visiting every station of the line
    # (10 lines of 1000 stations, 100 in the patch: 37 µs a shot before, 16
    # now, most of it the X records themselves).
    if ascending[main[li]] == 0: ascending[main[li]] = (if line.slotsAscend: 1 else: -1)
    var i = 0
    var last = line.stations.len
    if ascending[main[li]] > 0:
      i = firstIndex(last, k, line.slotOf(k) >= s0)
      last = firstIndex(last, k, line.slotOf(k) > s1)
    while i < last:
      let s = line.slotOf(i)
      if s >= s0 and s <= s1:
        picked.add Picked(line: line.line, step: line.pointStep, st: line.stations[i])
        amin = min(amin, line.stations[i].inline)
        amax = max(amax, line.stations[i].inline)
      inc i
  if amin <= amax:
    let half = layout.lineSpacing / 2
    for l in shortLines:
      template line: untyped = layout.lines[l]
      if line.crossline >= xmin - half and line.crossline <= xmax + half:
        for st in line.stations:
          if st.inline >= amin - 1e-9 and st.inline <= amax + 1e-9:
            picked.add Picked(line: line.line, step: line.pointStep, st: st)

proc emitRuns(picked: var seq[Picked], shot: PatchShot, relations: var seq[SpsRelation]) =
  ## X records: channels 1..n by ascending line and point, one record per
  ## run of consecutive stations (a numbering gap or an index change starts
  ## another).
  func byStation(a, b: Picked): int =
    result = cmp(a.line, b.line)
    if result == 0: result = cmp(a.st.point, b.st.point)
    if result == 0: result = cmp(a.st.index, b.st.index)
  # Picked line by line and point by point, they are often in order already:
  # seeing that costs a pass, sorting them twenty.
  if not picked.isSorted(byStation): picked.sort(byStation)
  var channel = 1'i64
  var i = 0
  while i < picked.len:
    var j = i
    while j + 1 < picked.len and picked[j + 1].line == picked[i].line and
        picked[j + 1].st.index == picked[i].st.index and not picked[i].step.isNaN and
        abs(picked[j + 1].st.point - picked[j].st.point - picked[i].step) <=
          1e-6 * max(1.0, abs(picked[i].step)):
      inc j
    var r = initSpsRelation()
    r.fieldRecord = shot.record
    (r.srcLine, r.srcPoint) = (shot.source.line, shot.source.point)
    r.srcIndex = max(1, int(shot.source.index))
    r.fromChannel = channel
    r.toChannel = channel + int64(j - i)
    r.rcvLine = picked[i].line
    (r.fromReceiver, r.toReceiver) = (picked[i].st.point, picked[j].st.point)
    r.rcvIndex = max(1, int(picked[i].st.index))
    relations.add r
    channel = r.toChannel + 1
    i = j + 1

proc generateRelations*(g: SurveyGeometry, layout: ReceiverLayout,
    shots: openArray[PatchShot], t: PatchTemplate): seq[SpsRelation] =
  ## X records for `shots` from template `t` over the receivers of
  ## `layout` (its stations: R stations with coordinates; stations missing
  ## from R are never listed).
  ##
  ## Template lines are the layout lines that are not `short`. The source's
  ## place: with `ppCoordinates` (the source needs coordinates in `g`, else
  ## the shot gets no records), its fractional line index among the template
  ## lines by crossline position, and on each line its fractional station
  ## slot by inline position; with `ppNumbers`, `lineMap` and `pointMap` of
  ## its numbers, among the template lines by line number. The patch is the
  ## `lines` template lines centred there (ties: the higher line) plus
  ## `lineOffset`, and on each the `channels` station slots centred on the
  ## source plus `stationOffset` (a slot is one `pointStep` of the line). A
  ## slot with no R station is a hole: the line gets fewer channels. Short
  ## lines join with their stations inside the patch: within half a line
  ## spacing of the chosen lines, and inside their inline extent. A shot
  ## whose source line or point is not a finite number gets no records, in
  ## any mode: an X record names its source.
  ## `peShift` keeps the line range and each line's slots inside the survey;
  ## `peStatic` lists, for every shot, the stations inside `staticBox`.
  ##
  ## **Distance field.** The distance limits (`inlineHalfWidth`,
  ## `crosslineHalfWidth` along `boxAzimuthDeg`, `radius` around the box
  ## centre, `maxOffset` from the source) keep the receivers inside them:
  ## every R station (short lines too) when there is no template
  ## (`lines = channels = 0`, see `initDistancePatch`), else the template's
  ## receivers (the intersection). They need the source's coordinates (else
  ## no records). The box is centred on the source plus `inlineDistance`,
  ## `crosslineDistance`; `peShift` moves it inside the receivers' extent in
  ## the box frame (an irregular outline counts by its bounding box), and
  ## takes either a template or distance limits, not both.
  ##
  ## Channels run 1..n in ascending line and point order; each run of
  ## consecutive stations (step `pointStep`, one point index) is one X
  ## record. Point index 0 is written as 1. Write the result with parseSEG
  ## (`SpsFile.relations`, `writeSps`); `receiverRelations` gives the O
  ## records. Raises `ValueError` for roll or shift with neither a template
  ## nor a distance limit, a template without lines or channels, shift with
  ## both, a non-finite offset or map, or a negative or NaN limit, and for
  ## a `layout` whose azimuth, interval or line spacing is not finite (or
  ## not positive).
  t.validate
  if not (abs(layout.azimuthDeg) < Inf and layout.interval > 0 and layout.interval < Inf and
      layout.lineSpacing > 0 and layout.lineSpacing < Inf):
    raise newException(ValueError, "generateRelations: the layout needs a finite azimuth " &
      "and a positive, finite interval and line spacing (build it with receiverLayout)")
  var main, shortLines: seq[int]
  for i, line in layout.lines:
    if line.short: shortLines.add i else: main.add i
  var ascending = newSeq[int8](layout.lines.len)
  if t.placement == ppNumbers:
    main.sort(proc (a, b: int): int = cmp(layout.lines[a].line, layout.lines[b].line))
  var mainPos: seq[float64]
  for i in main:
    mainPos.add(if t.placement == ppNumbers: layout.lines[i].line else: layout.lines[i].crossline)
  var posStep = layout.lineSpacing
  if t.placement == ppNumbers:
    var d: seq[float64]
    for i in 1 ..< mainPos.len: d.add mainPos[i] - mainPos[i - 1]
    posStep = if d.len > 0: median(d) else: 1.0
  var staticPicks: seq[Picked]
  if t.edge == peStatic:
    let b = t.staticBox
    for line in layout.lines:
      for st in line.stations:
        if st.inline >= b.inlineMin and st.inline <= b.inlineMax and
            st.crossline >= b.crosslineMin and st.crossline <= b.crosslineMax:
          staticPicks.add Picked(line: line.line, step: line.pointStep, st: st)
  let field = if t.edge != peStatic and t.hasDistance: initDistanceField(layout, t)
              else: DistanceField()
  var picked: seq[Picked]
  for shot in shots:
    picked.setLen 0
    var c: Coordinates
    # An X record names its source: a shot without a line or a point number
    # gets none, in any mode. One without coordinates gets none where its
    # position is needed.
    let numbered = finite(shot.source.line) and finite(shot.source.point)
    let located = numbered and t.edge != peStatic and
      g.table.sources.lookup(shot.source, c) and finite(c.x) and finite(c.y)
    if not numbered:
      discard
    elif t.edge == peStatic:
      picked.add staticPicks
    elif located or not (t.placement == ppCoordinates or t.hasDistance):
      if t.hasTemplate and main.len > 0:
        var cl, sa = NaN
        if t.placement == ppNumbers:
          cl = fractionalIndex(mainPos, t.lineMap.scale * shot.source.line + t.lineMap.offset,
            posStep)
        else:
          sa = layout.inlineOf(c.x, c.y)
          cl = fractionalIndex(mainPos, layout.crosslineOf(c.x, c.y), posStep)
        pickTemplate(layout, main, shortLines, t, cl, sa, shot.source.point, ascending,
          picked)
      if t.hasDistance:
        let (ca, cc) = field.centreOf(t, c.x, c.y)
        let (cx, cy) = xyOfBox(field.az, ca, cc)
        if t.hasTemplate:
          var kept = 0
          for i in 0 ..< picked.len:
            let (x, y) = layout.xyOf(picked[i].st)
            if t.inside(x, y, inlineOf(field.az, x, y), crosslineOf(field.az, x, y),
                c.x, c.y, ca, cc, cx, cy):
              picked[kept] = picked[i]
              inc kept
          picked.setLen kept
        else:
          field.pickInside(layout, t, c.x, c.y, ca, cc, cx, cy, picked)
    emitRuns(picked, shot, result)

proc generateRelations*(g: SurveyGeometry, shots: openArray[PatchShot],
    t: PatchTemplate): seq[SpsRelation] =
  ## `generateRelations` over the measured receiver layout of `g`.
  g.generateRelations(g.receiverLayout, shots, t)


# ------------------------------------------------------------- comparison --

type
  RecordAgreement* = object
    ## One field record in two relation sets (e.g. the supplied XPS and a
    ## generated one), by distinct receiver stations.
    record*: int64
    listed*, generated*, common*: int
    missing*: seq[StationKey] ## Listed, not generated (line, point order).
    extra*: seq[StationKey] ## Generated, not listed: e.g. receivers dropped in the field.

func jaccard*(a: RecordAgreement): float64 =
  ## common / (listed ∪ generated); 1 when both are empty.
  let u = a.listed + a.generated - a.common
  if u == 0: 1.0 else: float64(a.common) / float64(u)

func listedFraction*(a: RecordAgreement): float64 =
  ## The share of listed stations that were generated; 1 when none are listed.
  if a.listed == 0: 1.0 else: float64(a.common) / float64(a.listed)

func stationCmp(a, b: StationKey): int =
  result = cmp(a.line, b.line)
  if result == 0: result = cmp(a.point, b.point)
  if result == 0: result = cmp(a.index, b.index)

proc compareRelations*(listed, generated: SurveyGeometry): seq[RecordAgreement] =
  ## Per field record (X) of either geometry, ascending: its receiver
  ## stations in `listed` against those in `generated`. Index 0 and 1 count
  ## as the same station.
  var records: seq[int64]
  for r in listed.records: records.add r
  for r in generated.records: records.add r
  records.sort
  var last = 0'i64
  for i, r in records:
    if i > 0 and r == last: continue
    last = r
    var a, b: HashSet[StationKey]
    for st in listed.stationChannels(r):
      let s = st.relation.receiver
      a.incl stationKey(s.line, s.point, max(1'i16, s.index))
    for st in generated.stationChannels(r):
      let s = st.relation.receiver
      b.incl stationKey(s.line, s.point, max(1'i16, s.index))
    var e = RecordAgreement(record: r, listed: a.len, generated: b.len)
    for k in a:
      if k in b: inc e.common
      else: e.missing.add k
    for k in b:
      if k notin a: e.extra.add k
    e.missing.sort(stationCmp)
    e.extra.sort(stationCmp)
    result.add e


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
