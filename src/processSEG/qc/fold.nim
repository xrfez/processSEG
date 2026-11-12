## Fold maps of a survey: the bin grid of its pre-plot (SPS) geometry and
## the pre-plot fold of its X relations.
##
## ```nim
## var g = initSurveyGeometry()                   # parseSEG
## for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
## let grid = surveyGrid(g)                       # natural bins at the receiver azimuth
## let fine = surveyGrid(g, binInline = 25, binCrossline = 25)
## var actual = initFoldMap(grid)
## for chunk in chunks: actual.addFold(chunkMeta(chunk))   # any order
## let missing = foldDifference(pre, actual)       # pre-plot − actual
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
## every pre-plot midpoint, whatever the X records say, but for strays): along each grid
## axis a midpoint `(s + r) / 2` lies between half the sum of the source and
## receiver minima and half the sum of their maxima, which is the grid's
## extent. Only stations with coordinates count.
##
## **Strays.** A station far outside the survey (an SPS placeholder at
## (0, 0), a typo in a coordinate) would stretch the grid to millions of
## empty bins. With `trimStrays` (the default) the extent is taken over the
## stations of each set (sources, receivers) inside its fences on both
## axes: the quartiles `q1`, `q3` of the positions widened by `3·max(q3 −
## q1, 10 bins)` each way, which keeps any compact survey whole. The pairs
## of a station left out fall outside the grid and are counted as such
## (`FoldMap.outside`). `trimStrays = false` covers every station.
##
## **Centring.** The grid is one fixed lattice over the whole survey, and
## by default (`centre = true`) its bins are centred where the midpoints
## cluster. Along each axis, with `p` a position and `b` the bin, a
## midpoint `(s + r) / 2` has the bin phase `2π·p / b = π·s / b + π·r / b`:
## half a source's angle `π·s / b` plus half a receiver's. Each station
## set's dominant angle is its circular mean (the mean of the unit vectors
## `exp(iπ·p / b)`), taken again without the stations more than π/4 from
## it, so a few stray stations pull it little; the bin centres go at the
## sum of the two. That is O(stations), not O(pairs), and on a regular
## survey it puts every midpoint at a bin centre. A station set whose
## angles show no preference (the Rayleigh test: `n·R² ≤ ln 1000`, for `n`
## stations and a mean vector of length `R`; uniform angles pass one time in
## 1000) has none, and then the grid keeps the minimum corner as its first
## centre: midpoints of irregular source points scatter over the bins
## whatever their phase. Either way the first centre is within half a bin
## of the corner, and the grid covers every midpoint. `centre = false`
## always starts at the corner.
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
## (`noCoordinates`), as are pairs outside the bins (`outside`). Every
## channel is a pair: a multi-component survey counts each component (a 3C
## survey shows three times the conventional fold), as `addFold` counts
## each trace, so the two maps still align; `qc/record` and `survey/patch`
## count stations instead.
##
## **Actual.** `addFold` adds a block of traces (a `ChunkMeta` or a
## `TraceRows` view) to a map, chunk by chunk and in any order (counts
## only add up): every **live** trace (D12: status live or unknown, and not
## a failed row of the meta) is one pair. Its midpoint is the keys' `cmpX`,
## `cmpY`, or half-way between `srcX/srcY` and `rcvX/rcvY` when the headers
## have no midpoint (fill both with parseSEG's `applySurvey` first, from
## the survey). Its offset comes from the coordinates, else from
## `|offset|`; its bearing from the coordinates only (none: in no sector).
## Header binnings (`bnCdp`, `bnInlineCrossline`) bin by `binOf` (CDP,
## inline and crossline keys). `foldDifference(preplot, actual)` is
## pre-plot − actual per bin: positive where traces are missing, negative
## where traces fall where the pre-plot has fewer (geometry errors).

import std/[algorithm, math, tables]
import parseSEG
import ../core/types
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

proc stationFrame(t: Table[StationKey, Coordinates], azimuthDeg: float64,
    bins: array[2, float64], trim: bool): array[2, seq[float64]] =
  ## The positions of the stations along (0) and across (1) the azimuth;
  ## with `trim`, of those within the fences on both axes (module
  ## documentation).
  let (s, c) = (sin(degToRad(azimuthDeg)), cos(degToRad(azimuthDeg)))
  var pos: array[2, seq[float64]]
  for p in t.values:
    if finite(p.x) and finite(p.y):
      pos[0].add p.x * s + p.y * c
      pos[1].add p.x * c - p.y * s
  if not trim or pos[0].len == 0: return pos
  var fenceLo, fenceHi: array[2, float64]
  for a in 0 .. 1:
    var v = pos[a]
    v.sort
    let (q1, q3) = (v[(v.len - 1) div 4], v[(3 * (v.len - 1)) div 4])
    let w = 3 * max(q3 - q1, 10 * bins[a])
    (fenceLo[a], fenceHi[a]) = (q1 - w, q3 + w)
  for k in 0 ..< pos[0].len:
    let (a, x) = (pos[0][k], pos[1][k])
    if a >= fenceLo[0] and a <= fenceHi[0] and x >= fenceLo[1] and x <= fenceHi[1]:
      result[0].add a
      result[1].add x

func halfPhase(pos: seq[float64], bin: float64, angle: var float64): bool =
  ## The dominant `π·p / bin` of the station positions `pos` (module
  ## documentation): half a midpoint's bin phase. False when they show none.
  var (re, im) = (0.0, 0.0)
  for p in pos:
    re += cos(PI * p / bin)
    im += sin(PI * p / bin)
  # Rayleigh test: uniform phases give n·R² > ln 1000 one time in 1000.
  if pos.len == 0 or (re * re + im * im) / float64(pos.len) <= ln(1000.0): return false
  let mean = arctan2(im, re)
  (re, im) = (0.0, 0.0)
  for p in pos: # again, without the stations more than π/4 from the mean
    let a = PI * p / bin
    if abs(floorMod(a - mean + PI, 2 * PI) - PI) <= PI / 4:
      re += cos(a)
      im += sin(a)
  angle = arctan2(im, re)
  true

func centred(rcv, src: seq[float64], bin, lo: float64): float64 =
  ## The bin centre within half a bin of `lo` at the midpoints' dominant
  ## phase (module documentation); `lo` itself when there is none.
  var ar, br: float64
  if not halfPhase(rcv, bin, ar) or not halfPhase(src, bin, br): return lo
  let phase = (ar + br) / (2 * PI) * bin # a centre, modulo bin
  result = phase + bin * round((lo - phase) / bin)
  if result - lo > 0.5 * bin * (1 - 1e-9): result -= bin # lo must round into bin 0

proc badGrid(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, "surveyGrid: " & msg)

proc surveyGrid*(g: SurveyGeometry, binInline = NaN, binCrossline = NaN,
    azimuthDeg = NaN, centre = true, trimStrays = true): Binning =
  ## The bin grid covering every source–receiver midpoint of `g` (see the
  ## module documentation): inlines along `azimuthDeg`, bins `binInline`
  ## along them (`gridBinning`'s `crosslineSpacing`) by `binCrossline`
  ## across (its `inlineSpacing`). NaN arguments take their defaults: the
  ## receiver-line azimuth, half the receiver interval, half the source
  ## interval. `centre` places the bin centres on the midpoints' dominant
  ## positions (module documentation); `false` starts the grid at the
  ## minimum corner. `trimStrays` leaves stations far outside the survey out
  ## of the extent (module documentation). Raises `ValueError` for a given
  ## value that is not finite (or
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
  let rcv = stationFrame(g.table.receivers, azimuth, [binI, binC], trimStrays)
  let src = stationFrame(g.table.sources, azimuth, [binI, binC], trimStrays)
  if rcv[0].len == 0: badGrid("no receiver has coordinates")
  if src[0].len == 0: badGrid("no source has coordinates")
  var lo, hi: array[2, float64]
  for a in 0 .. 1:
    lo[a] = (min(rcv[a]) + min(src[a])) / 2
    hi[a] = (max(rcv[a]) + max(src[a])) / 2
  if centre:
    lo = [centred(rcv[0], src[0], binI, lo[0]), centred(rcv[1], src[1], binC, lo[1])]
  # Nearest-bin assignment: position u (in bins) needs u < n − 0.5 (with
  # the low side's slack, so rounding in binAt cannot drop the last one).
  let crosslines = int(floor((hi[0] - lo[0]) / binI + 0.5 + 1e-9)) + 1
  let inlines = int(floor((hi[1] - lo[1]) / binC + 0.5 + 1e-9)) + 1
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
    azimuthSectors*: int     ## Sectors of the source → receiver bearing (0: none).
    sectorStartDeg*: float64 ## Start of sector 0, degrees clockwise from north.
    offsetRange*: bool       ## Keep each bin's minimum and maximum offset.

  FoldMap* = object
    ## Source–receiver pairs per bin; build it with `initFoldMap` or
    ## `preplotFold`.
    binning: Binning
    options: FoldOptions
    fold, offsetFold, azimuthFold: Tensor[int32]
    offsetMin, offsetMax: Tensor[float32]
    binned, outside, noCoordinates, notLive: int

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
  ## Pairs left out because a station has no coordinates (traces: no
  ## midpoint, and no source and receiver coordinates to make one).
  m.noCoordinates

func notLive*(m: FoldMap): int {.inline.} =
  ## Traces left out by `addFold` as not live (dead, auxiliary, failed).
  m.notLive

template raw[T](t: Tensor[T]): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](t.unsafe_raw_offset(aligned = false))

proc clear*(m: var FoldMap) {.raises: [].} =
  ## Zeroes every count (offset ranges back to NaN); the storage is kept.
  template zero(t: Tensor[int32]) =
    if t.size > 0: zeroMem(t.raw, t.size * sizeof(int32))
  template unset(t: Tensor[float32]) =
    for i in 0 ..< t.size: t.raw[i] = NaN
  zero(m.fold)
  zero(m.offsetFold)
  zero(m.azimuthFold)
  unset(m.offsetMin)
  unset(m.offsetMax)
  (m.binned, m.outside, m.noCoordinates, m.notLive) = (0, 0, 0, 0)

proc addToBin(m: var FoldMap, bin: int, offset, bearing: float64) {.raises: [].} =
  ## Counts one pair in bin `bin` (≥ 0); a NaN offset or bearing is in no
  ## class, range or sector.
  inc m.binned
  let size = m.binning.len
  {.push boundChecks: off, overflowChecks: off.}
  inc m.fold.raw[bin]
  template b: untyped = m.options.offsetBounds
  if b.len > 1 and offset >= b[0] and offset < b[^1]:
    let k = upperBound(b, offset) - 1 # b[k] <= offset < b[k + 1]
    inc m.offsetFold.raw[k * size + bin]
  if m.options.offsetRange and not offset.isNaN:
    let f = float32(offset)
    let lo = m.offsetMin.raw
    let hi = m.offsetMax.raw
    if not (lo[bin] <= f): lo[bin] = f
    if not (hi[bin] >= f): hi[bin] = f
  let n = m.options.azimuthSectors
  if n > 0 and not bearing.isNaN:
    let k = min(n - 1, int(floorMod(bearing - m.options.sectorStartDeg, 360.0) *
      float64(n) / 360.0))
    inc m.azimuthFold.raw[k * size + bin]
  {.pop.}

func bearingOf(sx, sy, rx, ry: float64): float64 {.inline.} =
  ## Source → receiver bearing in degrees (NaN when a coordinate is).
  radToDeg(arctan2(rx - sx, ry - sy))

proc addPair(m: var FoldMap, sx, sy, rx, ry: float64) {.raises: [].} =
  ## Counts one pair with known coordinates.
  let bin = m.binning.binAt((sx + rx) * 0.5, (sy + ry) * 0.5)
  if bin < 0:
    inc m.outside
    return
  let bearing = if m.options.azimuthSectors > 0: bearingOf(sx, sy, rx, ry) else: NaN
  m.addToBin(bin, hypot(rx - sx, ry - sy), bearing)

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

proc addTrace(m: var FoldMap, k: TraceKeys) {.raises: [].} =
  let coords = finite(k.srcX) and finite(k.srcY) and finite(k.rcvX) and finite(k.rcvY)
  var bin: int
  if m.binning.kind in {bnLine, bnGrid}:
    var (x, y) = (k.cmpX, k.cmpY)
    if not (finite(x) and finite(y)):
      if not coords:
        inc m.noCoordinates
        return
      (x, y) = ((k.srcX + k.rcvX) * 0.5, (k.srcY + k.rcvY) * 0.5)
    bin = m.binning.binAt(x, y)
  else:
    bin = m.binning.binOf(k)
  if bin < 0:
    inc m.outside
    return
  let offset = if coords: hypot(k.rcvX - k.srcX, k.rcvY - k.srcY) else: abs(k.offset)
  let bearing = if coords and m.options.azimuthSectors > 0:
                  bearingOf(k.srcX, k.srcY, k.rcvX, k.rcvY)
                else: NaN
  m.addToBin(bin, offset, bearing)

proc addFold*(m: var FoldMap, meta: ChunkMeta) {.raises: [].} =
  ## Adds the live rows of `meta` (see the module documentation). Allocates
  ## nothing.
  for row in 0 ..< meta.len:
    if meta.isLive(row): m.addTrace(meta.keys(row))
    else: inc m.notLive

proc addFold*(m: var FoldMap, rows: TraceRows) =
  ## Adds the live rows of `rows` (status live or unknown). Raises
  ## `ValueError` when the view has no keys. Allocates nothing.
  if not rows.hasKeys:
    raise newException(ValueError, "the rows have no keys: open the file with " &
      "keys = true (or call ensureKeys) before making the view")
  for row in 0 ..< rows.len:
    let k = rows.keysAt(row)
    if k.status in {tsLive, tsUnknown}: m.addTrace(k)
    else: inc m.notLive

func sameValue(a, b: float64): bool {.inline.} = a == b or (a.isNaN and b.isNaN)

func sameBins(a, b: Binning): bool =
  if a.kind != b.kind or a.inlines != b.inlines or a.crosslines != b.crosslines: return false
  for bin in [0, a.len - 1]:
    let (p, q) = (a.location(bin), b.location(bin))
    if not (sameValue(p.cdp, q.cdp) and sameValue(p.inline, q.inline) and
        sameValue(p.crossline, q.crossline) and sameValue(p.x, q.x) and
        sameValue(p.y, q.y)): return false
  true

proc foldDifference*(preplot, actual: FoldMap): Tensor[int32] =
  ## Pre-plot − actual fold per bin, `[inlines, crosslines]`. Raises
  ## `ValueError` when the two maps are not over the same bins (kind,
  ## counts, first and last bin).
  if not sameBins(preplot.binning, actual.binning):
    raise newException(ValueError, "foldDifference: the maps have different binnings")
  result = newTensor[int32](preplot.fold.shape[0], preplot.fold.shape[1])
  let (p, a, d) = (preplot.fold.raw, actual.fold.raw, result.raw)
  {.push overflowChecks: off.}
  for i in 0 ..< result.size: d[i] = p[i] - a[i]
  {.pop.}
