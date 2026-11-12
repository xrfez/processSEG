## Fold maps of a survey: the bin grid of its pre-plot (SPS) geometry and
## the pre-plot fold of its X relations.
##
## ```nim
## var g = initSurveyGeometry()                   # parseSEG
## for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
## let grid = surveyGrid(g)                       # natural bins at the receiver azimuth
## let fine = surveyGrid(g, binInline = 25, binCrossline = 25)
## let pre = g.preplotFold(grid, FoldOptions(offsetBounds: @[0.0, 1000, 2000, 4000],
##                                           azimuthSectors: 8, offsetRange: true))
## echo pre.fold.shape                            # [inlines, crosslines]
## echo pre.offsetFold.shape                      # [3 classes, inlines, crosslines]
## ```
##
## **Grid.** `surveyGrid` builds a `gridBinning` (`stack/binning`) whose
## inlines run along the receiver lines: crossline numbers increase along
## the azimuth, inline numbers 90° clockwise of it, both from 1. Its bins
## cover the midpoint of every source–receiver pair of the geometry (so
## every pre-plot midpoint, whatever the X records say): along each grid
## axis a midpoint `(s + r) / 2` lies between half the sum of the source and
## receiver minima and half the sum of their maxima, which is the grid's
## extent. Only stations with coordinates count. The grid's origin is that
## minimum corner, a midpoint itself on a regular survey, so bins are centred
## on the midpoints there.
##
## **Defaults.** The azimuth is the receiver-line azimuth: the median
## bearing between consecutive stations of a receiver line, in point order
## (as `survey/patch.receiverLayout`). The *natural bin* is half the median
## receiver interval along the receiver lines (`binInline`) by half the
## median source interval across them (`binCrossline`), each interval the
## median distance between consecutive points of a line. That is the
## natural bin of an orthogonal survey (source lines across the receiver
## lines); for other layouts give `binCrossline`. Any argument given
## overrides its default, and a caller's own `Binning` works everywhere a
## `surveyGrid` does.
##
## **Fold.** A `FoldMap` counts source–receiver pairs per bin of its
## `Binning`: `fold` is `[inlines, crosslines]`, row-major like the bins (a
## 2D binning is `[1, count]`). On request (`FoldOptions`) it also counts
## them per **offset class** (`offsetFold`, `[classes, inlines,
## crosslines]`; class `k` is `offsetBounds[k] <= offset <
## offsetBounds[k + 1]`, offsets outside every class are in none) and per
## **azimuth sector** (`azimuthFold`, `[sectors, inlines, crosslines]`; the
## source → receiver bearing, sector `k` starting at `sectorStartDeg + k ×
## 360° / sectors`; a zero offset has bearing 0°), and keeps each bin's
## offset range (`offsetMin`, `offsetMax`, NaN for an empty bin). Offsets
## are source–receiver distances in grid units. A pair's bin is
## `Binning.binAt` of its midpoint, the maths `binOf` uses for a trace's
## midpoint, so a pre-plot map and one of traces align bin for bin.
##
## **Pre-plot.** `preplotFold` walks every relation of the geometry (X, or
## O with `domain = rdReceiver`) with parseSEG's `relations`. Station
## coordinates come from per-line arrays sorted by point, with a cursor that
## follows each run of consecutive stations, so a pair costs no hash
## lookup. Pairs whose stations have no coordinates are counted
## (`noCoordinates`), as are pairs outside the bins (`outside`).

import std/[algorithm, math, tables]
import parseSEG
import ../stack/binning

func finite(x: float64): bool {.inline.} = x.classify notin {fcNan, fcInf, fcNegInf}

func median(v: var seq[float64]): float64 =
  if v.len == 0: return NaN
  v.sort
  v[(v.len - 1) div 2]

proc lineSteps(t: Table[StationKey, Coordinates], bearings, steps: var seq[float64]) =
  ## Bearings and distances between consecutive stations of each line, in
  ## point (then index) order; stations without coordinates are skipped.
  var byLine: Table[float64, seq[(float64, int16, float64, float64)]]
  for k, c in t:
    if finite(c.x) and finite(c.y):
      byLine.mgetOrPut(k.line, @[]).add (k.point, k.index, c.x, c.y)
  for st in byLine.mvalues:
    st.sort(proc (a, b: (float64, int16, float64, float64)): int =
      result = cmp(a[0], b[0])
      if result == 0: result = cmp(a[1], b[1]))
    for i in 1 ..< st.len:
      let (dx, dy) = (st[i][2] - st[i - 1][2], st[i][3] - st[i - 1][3])
      let d = hypot(dx, dy)
      if d > 0:
        steps.add d
        bearings.add floorMod(radToDeg(arctan2(dx, dy)), 360.0)

func medianBearing(bearings: var seq[float64]): float64 =
  # Unwrapped around one bearing, so 359° and 1° agree.
  let reference = bearings[0]
  for b in bearings.mitems: b = reference + floorMod(b - reference + 180.0, 360.0) - 180.0
  floorMod(median(bearings), 360.0)

proc extent(t: Table[StationKey, Coordinates], azimuthDeg: float64,
    lo, hi: var array[2, float64]) =
  ## Min and max of the stations along (0) and across (1) the azimuth.
  let (s, c) = (sin(degToRad(azimuthDeg)), cos(degToRad(azimuthDeg)))
  lo = [Inf, Inf]
  hi = [-Inf, -Inf]
  for p in t.values:
    if finite(p.x) and finite(p.y):
      let a = p.x * s + p.y * c
      let x = p.x * c - p.y * s
      lo = [min(lo[0], a), min(lo[1], x)]
      hi = [max(hi[0], a), max(hi[1], x)]

proc badGrid(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, "surveyGrid: " & msg)

proc surveyGrid*(g: SurveyGeometry, binInline = NaN, binCrossline = NaN,
    azimuthDeg = NaN): Binning =
  ## The bin grid covering every source–receiver midpoint of `g` (see the
  ## module documentation): inlines along `azimuthDeg`, bins `binInline`
  ## along them (`gridBinning`'s `crosslineSpacing`) by `binCrossline`
  ## across (its `inlineSpacing`). NaN arguments take their defaults: the
  ## receiver-line azimuth, half the receiver interval, half the source
  ## interval. Raises `ValueError` for a given value that is not finite (or
  ## not positive, for the bins), when there are no receivers or no sources
  ## with coordinates, or when a default cannot be measured (no line with
  ## two stations at distinct positions).
  for (v, name) in [(binInline, "binInline"), (binCrossline, "binCrossline")]:
    if not v.isNaN and not (v > 0 and v < Inf):
      badGrid(name & " must be > 0 and finite, got " & $v)
  if not azimuthDeg.isNaN and not finite(azimuthDeg):
    badGrid("azimuthDeg must be finite, got " & $azimuthDeg)
  var azimuth = azimuthDeg
  var binI = binInline
  var binC = binCrossline
  if azimuth.isNaN or binI.isNaN:
    var bearings, steps: seq[float64]
    lineSteps(g.table.receivers, bearings, steps)
    if steps.len == 0:
      badGrid("no receiver line has two stations with coordinates; give " &
        (if azimuth.isNaN: "azimuthDeg" else: "binInline"))
    if azimuth.isNaN: azimuth = medianBearing(bearings)
    if binI.isNaN: binI = median(steps) / 2
  if binC.isNaN:
    var bearings, steps: seq[float64]
    lineSteps(g.table.sources, bearings, steps)
    if steps.len == 0:
      badGrid("no source line has two points with coordinates; give binCrossline")
    binC = median(steps) / 2
  var rLo, rHi, sLo, sHi: array[2, float64]
  extent(g.table.receivers, azimuth, rLo, rHi)
  extent(g.table.sources, azimuth, sLo, sHi)
  if rLo[0] == Inf: badGrid("no receiver has coordinates")
  if sLo[0] == Inf: badGrid("no source has coordinates")
  let lo = [(rLo[0] + sLo[0]) / 2, (rLo[1] + sLo[1]) / 2]
  let hi = [(rHi[0] + sHi[0]) / 2, (rHi[1] + sHi[1]) / 2]
  # Nearest-bin assignment: position u (in bins) needs u < n − 0.5.
  let crosslines = int(floor((hi[0] - lo[0]) / binI + 0.5)) + 1
  let inlines = int(floor((hi[1] - lo[1]) / binC + 0.5)) + 1
  let (s, c) = (sin(degToRad(azimuth)), cos(degToRad(azimuth)))
  let x0 = lo[0] * s + lo[1] * c
  let y0 = lo[0] * c - lo[1] * s
  gridBinning(x0, y0, azimuth, crosslineSpacing = binI, inlineSpacing = binC,
    inlines = inlines, crosslines = crosslines)

type
  FoldOptions* = object
    ## What a `FoldMap` counts besides the fold.
    offsetBounds*: seq[float64]
      ## Ascending offset class edges; n edges make n − 1 classes (none: no classes).
    azimuthSectors*: int    ## Sectors of the source → receiver bearing (0: none).
    sectorStartDeg*: float64 ## Start of sector 0, degrees clockwise from north.
    offsetRange*: bool      ## Keep each bin's minimum and maximum offset.

  FoldMap* = object
    ## Source–receiver pairs per bin; build it with `initFoldMap` or
    ## `preplotFold`.
    binning: Binning
    options: FoldOptions
    fold, offsetFold, azimuthFold: Tensor[int32]
    offsetMin, offsetMax: Tensor[float32]
    binned, outside, noCoordinates: int

proc initFoldMap*(binning: Binning, opts = FoldOptions()): FoldMap =
  ## An empty fold map over `binning`. Raises `ValueError` when the offset
  ## bounds are given but fewer than two, not finite or not strictly
  ## ascending, when `azimuthSectors` is negative, or `sectorStartDeg` not
  ## finite.
  let b = opts.offsetBounds
  if b.len == 1: raise newException(ValueError, "initFoldMap: offsetBounds needs " &
    "at least two edges")
  for i, e in b:
    if not finite(e) or (i > 0 and e <= b[i - 1]):
      raise newException(ValueError, "initFoldMap: offsetBounds must be finite and " &
        "strictly ascending, got " & $b)
  if opts.azimuthSectors < 0:
    raise newException(ValueError, "initFoldMap: azimuthSectors must be ≥ 0")
  if not finite(opts.sectorStartDeg):
    raise newException(ValueError, "initFoldMap: sectorStartDeg must be finite")
  let (n, m) = (binning.inlines, binning.crosslines)
  result = FoldMap(binning: binning, options: opts, fold: zeros[int32](n, m))
  if b.len > 1: result.offsetFold = zeros[int32](b.len - 1, n, m)
  if opts.azimuthSectors > 0: result.azimuthFold = zeros[int32](opts.azimuthSectors, n, m)
  if opts.offsetRange:
    result.offsetMin = newTensor[float32](n, m)
    result.offsetMin.apply_inline(NaN)
    result.offsetMax = result.offsetMin.clone

func binning*(m: FoldMap): lent Binning {.inline.} =
  ## The bins counted.
  m.binning

func options*(m: FoldMap): lent FoldOptions {.inline.} =
  ## What is counted besides the fold.
  m.options

func fold*(m: FoldMap): lent Tensor[int32] {.inline.} =
  ## Pairs per bin, `[inlines, crosslines]`.
  m.fold

func offsetFold*(m: FoldMap): lent Tensor[int32] {.inline.} =
  ## Pairs per offset class and bin, `[classes, inlines, crosslines]`;
  ## empty without `offsetBounds`.
  m.offsetFold

func azimuthFold*(m: FoldMap): lent Tensor[int32] {.inline.} =
  ## Pairs per azimuth sector and bin, `[sectors, inlines, crosslines]`;
  ## empty without `azimuthSectors`.
  m.azimuthFold

func offsetMin*(m: FoldMap): lent Tensor[float32] {.inline.} =
  ## Smallest offset per bin (NaN: empty), `[inlines, crosslines]`; empty
  ## without `offsetRange`.
  m.offsetMin

func offsetMax*(m: FoldMap): lent Tensor[float32] {.inline.} =
  ## Largest offset per bin, as `offsetMin`.
  m.offsetMax

func binned*(m: FoldMap): int {.inline.} =
  ## Pairs counted in a bin.
  m.binned

func outside*(m: FoldMap): int {.inline.} =
  ## Pairs whose midpoint is in no bin.
  m.outside

func noCoordinates*(m: FoldMap): int {.inline.} =
  ## Pairs left out because a station has no coordinates.
  m.noCoordinates

template raw[T](t: Tensor[T]): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](t.unsafe_raw_offset(aligned = false))

proc addPair(m: var FoldMap, sx, sy, rx, ry: float64) {.raises: [].} =
  ## Counts one pair with known coordinates.
  let bin = m.binning.binAt((sx + rx) * 0.5, (sy + ry) * 0.5)
  if bin < 0:
    inc m.outside
    return
  inc m.binned
  let size = m.binning.len
  {.push boundChecks: off, overflowChecks: off.}
  inc m.fold.raw[bin]
  template o: untyped = m.options
  if o.offsetBounds.len > 1 or o.offsetRange:
    let offset = hypot(rx - sx, ry - sy)
    template b: untyped = m.options.offsetBounds
    if b.len > 1 and offset >= b[0] and offset < b[^1]:
      let k = upperBound(b, offset) - 1 # b[k] <= offset < b[k + 1]
      inc m.offsetFold.raw[k * size + bin]
    if o.offsetRange:
      let f = float32(offset)
      let lo = m.offsetMin.raw
      let hi = m.offsetMax.raw
      if not (lo[bin] <= f): lo[bin] = f
      if not (hi[bin] >= f): hi[bin] = f
  if o.azimuthSectors > 0:
    let bearing = radToDeg(arctan2(rx - sx, ry - sy))
    let n = o.azimuthSectors
    let k = min(n - 1, int(floorMod(bearing - o.sectorStartDeg, 360.0) * float64(n) / 360.0))
    inc m.azimuthFold.raw[k * size + bin]
  {.pop.}

type
  StationAt = object
    point: float64
    index: int16
    x, y: float64

  StationIndex = object
    ## Stations with coordinates, by line, sorted by point then index.
    lineOf: Table[float64, int]
    lines: seq[seq[StationAt]]

  Cursor = object
    ## The last station found, so a run of consecutive stations resolves
    ## without a search.
    line: float64
    li, pos: int

proc initStationIndex(t: Table[StationKey, Coordinates]): StationIndex =
  for k, c in t:
    if finite(c.x) and finite(c.y):
      var li = result.lineOf.getOrDefault(k.line, -1)
      if li < 0:
        li = result.lines.len
        result.lineOf[k.line] = li
        result.lines.add @[]
      result.lines[li].add StationAt(point: k.point, index: k.index, x: c.x, y: c.y)
  for st in result.lines.mitems:
    st.sort(proc (a, b: StationAt): int =
      result = cmp(a.point, b.point)
      if result == 0: result = cmp(a.index, b.index))

func initCursor(): Cursor = Cursor(line: NaN, li: -1, pos: -1)

proc find(ix: StationIndex, c: var Cursor, s: StationKey): int {.raises: [].} =
  ## The position of `s` on its line (parseSEG's rule: the exact index,
  ## then index 0 (any); a station with no index (0) also matches index 1),
  ## or −1 (the cursor's line is then `c.li`).
  if s.line != c.line:
    c.line = s.line
    c.li = ix.lineOf.getOrDefault(s.line, -1)
    c.pos = -1
  if c.li < 0: return -1
  let st = ix.lines[c.li].addr
  template exact(p: int): bool =
    p >= 0 and p < st[].len and st[][p].point == s.point and st[][p].index == s.index
  if exact(c.pos): return c.pos
  if exact(c.pos + 1):
    inc c.pos
    return c.pos
  # binary search: the first station at the point, then the index rule
  var lo = 0
  var hi = st[].len
  while lo < hi:
    let mid = (lo + hi) div 2
    if st[][mid].point < s.point: lo = mid + 1 else: hi = mid
  var any, one = -1
  var p = lo
  while p < st[].len and st[][p].point == s.point:
    if st[][p].index == s.index:
      c.pos = p
      return p
    if st[][p].index == 0: any = p
    elif st[][p].index == 1: one = p
    inc p
  let found = if any >= 0: any elif s.index == 0: one else: -1
  if found >= 0: c.pos = found
  found

proc preplotFold*(g: SurveyGeometry, binning: Binning, opts = FoldOptions(),
    domain = rdSource): FoldMap =
  ## The pre-plot fold of `g`'s relations (X; O with `rdReceiver`) over
  ## `binning` (see the module documentation). Raises `ValueError` for bad
  ## options (`initFoldMap`) or a binning without positions (`bnCdp`,
  ## `bnInlineCrossline`: pairs have midpoints, not header numbers).
  if binning.kind notin {bnLine, bnGrid}:
    raise newException(ValueError, "preplotFold: the binning must be a line or grid " &
      "binning (midpoint positions), got " & $binning.kind)
  result = initFoldMap(binning, opts)
  let sources = initStationIndex(g.table.sources)
  let receivers = initStationIndex(g.table.receivers)
  var sc = initCursor()
  var rc = initCursor()
  for record in g.records(domain):
    for (_, rel) in g.relations(record, domain):
      let i = sources.find(sc, rel.source)
      let j = receivers.find(rc, rel.receiver)
      if i < 0 or j < 0:
        inc result.noCoordinates
      else:
        let s = sources.lines[sc.li][i]
        let r = receivers.lines[rc.li][j]
        result.addPair(s.x, s.y, r.x, r.y)
