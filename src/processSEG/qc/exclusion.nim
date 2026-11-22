## Noisy and weak traces of one gather, judged outside an exclusion area:
## the traces near the common point (the source of a shot gather, the
## receiver of a receiver gather) are loud for physical reasons and are not
## called noisy. `qc/amplitude.recordAmplitude` judges every trace of a
## record against one median; this module takes the near-source zone out of
## that judgement, compares like with like (the hydrophone apart, nodes as
## wholes, components after rotation) and adds a second finding, a trace
## louder than its neighbours.
##
## ```nim
## # amps from traceAmplitudeInto; picks from qc/firstbreak (or none)
## var roles = newSeq[TraceRole](meta.len)
## meta.traceRolesInto(gdCommonSource, [], roles)  # rotated: one bool per row, or empty
## var res: NoiseResult
## var ws = initWorkspace()
## ws.reserve(noiseWorkspace(meta.len))
## noiseQc(meta, amps, picks, roles, AmplitudeParams(), ExclusionParams(),
##   gdCommonSource, res, ws)
## echo res.rule, " ", res.excluded, " traces not judged"
## for r, a in amps:
##   if afNoisy in a.flags: echo r, " noisy, z = ", a.robustZ
## ```
##
## **Terms.** A *gather* holds the traces of one record that share a common
## point; the *far end* of a trace is its other station (the receiver in a
## shot gather, the source in a receiver gather). A *unit* is what the
## statistics compare: a trace, or for unrotated multi-component geophones a
## *node* (the motion traces of one far-end station). A *group* is the units
## compared with each other: single-component data is one group, the
## hydrophone one group, unrotated geophone nodes one group, rotated
## geophones one group per component. A *line* is the far end's line and the
## *station* its position along it.
##
## **Roles** (`traceRolesInto`, or fill `TraceRole` yourself). The sensor
## class comes from SEG-D `sensorType` (1 pressure, 2-9 motion) or the
## SEG-Y trace code (11 pressure, 12-14 motion, 15-17 motion rotated to
## V/T/R); a record with at most one distinct non-zero component is
## single-component. A row is rotated when the caller says so (`rotated`,
## filled from the file's orientation header) or its SEG-Y code is 15-17;
## SEG-D without that header counts as unrotated. Lines come from the far
## end's header line, else from its coordinates (a new line where the next
## station is over 3 median spacings away and over 2 spacings off the local
## axis of the last 5, or far back along it: extra splits are harmless),
## else from its station numbers (a number going backwards or jumping 10
## typical steps), else the record is one line.
##
## **The area.** The first rule that applies sets it, and `NoiseResult.rule`
## and `reasons` say which and why the earlier ones did not apply:
##
## 1. `arModel`: the refraction model (`qc/shotfit`), when the picks and
##    offsets allow it: its expected arrival time at each trace's offset.
## 2. `arTrend`: the trend of the first-break picks along each line: a
##    running median over the ±`trendTraces` neighbours of the confident
##    picks (widened up to 4 times while fewer than 3 are found), then the
##    same median over the picks that lie within `clamp(5 % of the trend,
##    20, 60)` ms of it. The trace's own pick is not used: a noisy trace
##    would place itself outside, and a burst before the break would give
##    it an early pick and put it inside.
## 3. `arRadius`: the offsets alone, from a survey or from header
##    coordinates (`offsetOrigin` records which).
## 4. `arSteps`: `steps` stations either side of the station nearest the
##    source, counted on each line separately, on the lines the shot reaches
##    (`stepsOnAllLines`: every line).
## 5. `arNone`: every trace is judged, as `recordAmplitude` does.
##
## Rules 1 to 3 place the *edge* where the level stops depending on arrival:
## the traces of a group are sorted by arrival; `L` is the median level of
## the later half (and of the traces without an arrival); a running median
## of the level over ±max(5, n/20) traces is followed to its maximum and the
## edge is the arrival of the first position after it that is at most `L +
## levelDb` (6 dB). Units arriving before the edge are *inside*: not judged
## noisy (those arriving at the edge too: a trend has runs of equal values).
## Fewer than 20 units with an arrival is too few for the rule. The user may
## instead give a fixed time (`acFixedTime`, on the model or trend arrival),
## a fixed radius (`acFixedRadius`), a step count (`acSteps`), or turn the
## area off (`acOff`).
##
## **Judging.** Over the usable units outside the area the median `M` and
## `S = 1.4826·MAD` of the level (`rmsDb`) are taken; each usable unit gets
## `robustZ = (rmsDb − M) / S`. *Noisy* is outside and above `outlierZ`;
## *weak* is below `−outlierZ` and is judged everywhere, inside the area
## too. A node's level is the dB of the mean energy of its usable motion
## traces; its flags go to all of its usable traces. With 2 of 3 usable
## traces the node is judged and marked `partial`; with fewer it is
## `nsIncompleteNode`. *Louder than its neighbours* compares a unit outside
## the area with the median of up to `neighbours` usable units outside the
## area on each side on its own line (never across a line change); `neighbourZ`
## of the residuals' own robust z-score calls it louder. `groups[].medianDb`
## is each group's level, for comparing the gathers of a folder.
##
## **Limits, measured on field and synthetic records.**
## - A run of 3 or more adjacent traces with a burst before the break and
##   beyond the visible breaks makes coherent early picks: with the trend
##   they fall inside the area (the model, which a few dozen false picks do
##   not move, judges them).
## - An uncorrelated vibroseis record has no first breaks: pass no picks (or
##   those of its correlated version); offsets or station numbers may still
##   give a radius or steps.
## - Unrotated geophone components are judged per node, on their summed
##   energy: one noisy component can hide behind two quiet ones.
## - A step count of 3 or 4 removes only about half of the near-source
##   flags on real patches (8, the default, about all); the automatic edge
##   needs no choice.
## - The area is computed per gather from the arrivals of all its traces;
##   only the edge differs per group.
##
## **Allocation.** `NoiseResult` keeps its scratch: after the first gather of
## a size, `noiseQc` with a `ShotFit` passed in, or without picks, allocates
## nothing. The form that fits the shot itself allocates in `fitShot`.

import std/[math, tables]
import parseSEG
import ../core/[types, workspace]
import ./[amplitude, firstbreak, shotfit]

type
  GatherDomain* = enum
    gdCommonSource = "common source"     ## A shot gather: the far end is the receiver.
    gdCommonReceiver = "common receiver" ## A receiver gather: the far end is the source.

  SensorClass* = enum
    scSingle = "single"
    scPressure = "pressure"
    scMotion = "motion"

  AreaChoice* = enum
    acAuto = "automatic"
    acFixedTime = "fixed time"
    acFixedRadius = "fixed radius"
    acSteps = "station steps"
    acOff = "off"

  AreaRule* = enum
    arModel = "refraction model"
    arTrend = "first-break trend"
    arRadius = "radius"
    arSteps = "station steps"
    arFixedTime = "fixed time"
    arFixedRadius = "fixed radius"
    arNone = "none"

  AreaReason* = enum ## Why a rule did not apply (a set in `NoiseResult.reasons`).
    rnNoPicks = "no first breaks"
    rnFewPicks = "too few coherent first breaks"
    rnNoOffsets = "no offsets"
    rnNoPosition = "no source position along any line"
    rnFewTraces = "too few usable traces"
    rnNoLoudness = "no near-source loudness"
    rnOverrideNeedsPicks = "a fixed time needs first breaks"
    rnOverrideNeedsOffsets = "a fixed radius needs offsets"

  OffsetOrigin* = enum
    ooUnknown = ""
    ooHeaders = "trace headers"
    ooSurvey = "survey"

  LineSource* = enum ## How the lines of a gather were found.
    lsHeaders = "header line numbers"
    lsCoordinates = "coordinates"
    lsStations = "station numbers"
    lsUnknown = "unknown (the record taken as one line)"

  NoiseState* = enum
    nsJudged = "judged"
    nsInArea = "not judged: inside the exclusion area"
    nsUnusable = "not judged: no usable samples"
    nsIncompleteNode = "not judged: node incomplete"

  NeighbourState* = enum
    nbJudged = "judged"
    nbInArea = "not judged: inside the exclusion area"
    nbUnusable = "not judged: no usable samples"
    nbFewNeighbours = "not judged: too few neighbours on its line"

  TraceRole* = object
    ## Where a trace stands in the statistics (module documentation).
    sensor*: SensorClass
    rotated*: bool
    group*: int32     ## Statistics group (`traceRolesInto` numbers them from 0); -1: none.
    node*: int32      ## Motion rows of one node share it; -1 for none.
    line*: int32      ## The far end's line, record-local; -1 unknown.
    station*: float64 ## The far end's position along its line.

  ExclusionParams* = object
    ## What `noiseQc` does; times in ms, distances in offset units.
    choice*: AreaChoice = acAuto
    minConfidence*: float64 = 0.5           ## Picks below it are not used.
    trendTraces*: int = 7                   ## The trend's running-median half window, in rows.
    levelDb*: float64 = 6                   ## The level drop that places the edge.
    edgeMs*: float64 = NaN                  ## `acFixedTime`.
    radius*: float64 = NaN                  ## `acFixedRadius`, offset units.
    steps*: int = 8                         ## `acSteps`, and the automatic fallback.
    stepsOnAllLines*: bool = false          ## Steps on every line, not only those reached.
    minArrivals*: int = 8                   ## Coherent picks needed (and `minFraction` of rows).
    minFraction*: float64 = 0.05
    neighbours*: int = 7                    ## Units each side for the neighbour median.
    minNeighbours*: int = 4
    neighbourZ*: float64 = 3.5              ## Inf: no neighbour finding.
    offsetOrigin*: OffsetOrigin = ooUnknown ## Echoed into the result's wording only.

  GroupNoise* = object
    ## One group's counts, in units (a node is one unit).
    sensor*: SensorClass
    component*: uint8          ## Rotated motion: the component; else 0.
    units*, judged*, inArea*, noisy*, weak*, louder*: int
    medianDb*, madDb*: float64 ## Of the judged units: the gather's level.
    edge*: float64             ## This group's edge (time or distance); NaN: none.

  SortKey = object
    a: int32
    b: float64
    row: int32

  NoiseScratch = object
    fit: ShotFit
    live, usableRow, inside2: seq[bool]
    ordered, sorted: seq[SortKey]
    segStart: seq[int32]
    segOf, rep, nextMember, nodeFirst, nodeLast: seq[int32]
    uval, uarr, uz, absOff, trendT1, pickMs, coherentMs, trendArr, nbRes: seq[float64]
    tmp, tmp2, vals, level: seq[float64]
    srcStation, stepSize: seq[float64]
    edges, edges2: seq[float64]
    uUsable, uNode, seenGroup: seq[bool]
    elig: seq[int32]

  NoiseResult* = object
    ## What `noiseQc` found; per-row seqs are indexed by row.
    domain*: GatherDomain
    rule*: AreaRule
    reasons*: set[AreaReason]
    offsetOrigin*: OffsetOrigin
    edgeMs*: float64                    ## Time rules: the first group's edge; NaN else.
    radius*: float64                    ## Radius rules; NaN else.
    steps*: int                         ## `arSteps`; 0 else.
    lines*, linesReached*: int          ## Lines found; lines with a unit inside the area.
    linesFromHeaders*: bool             ## False: from coordinates, station numbers or one line.
    lineSource*: LineSource
    agreement*: float64                 ## Model and trend both applied: share agreeing.
    excluded*, judged*, notJudged*: int ## Traces (rows).
    arrival*: seq[float64]              ## Per row: the rule's arrival; NaN: none.
    inside*: seq[bool]
    state*: seq[NoiseState]
    partial*: seq[bool]                 ## Its node was judged on 2 of 3 components.
    neighbourDb*: seq[float64]          ## The residual against its line neighbours; NaN none.
    neighbourZ*: seq[float64]
    louder*: seq[bool]                  ## "Louder than its neighbours".
    neighbourState*: seq[NeighbourState]
    earlyBreak*: seq[bool]              ## Own pick earlier than the edge, outside the area.
    groups*: seq[GroupNoise]
    sc: NoiseScratch

proc badNoise(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func finite(x: float64): bool {.inline.} = x - x == 0

proc validate*(p: ExclusionParams) =
  ## Raises `ValueError` when `p` is not valid: `levelDb`, `minConfidence`
  ## (0 .. 1) or `minFraction` (0 .. 1) not finite or out of range;
  ## `trendTraces`, `minArrivals`, `neighbours` or `minNeighbours` below 1;
  ## `steps` below 0; `neighbourZ` NaN or not above 0; a NaN `edgeMs` with
  ## `acFixedTime`, or a `radius` that is not positive with `acFixedRadius`.
  if not (finite(p.levelDb) and p.levelDb >= 0):
    badNoise("levelDb must be finite and ≥ 0, got " & $p.levelDb)
  if not (p.minConfidence >= 0 and p.minConfidence <= 1):
    badNoise("minConfidence must be in 0 .. 1, got " & $p.minConfidence)
  if not (p.minFraction >= 0 and p.minFraction <= 1):
    badNoise("minFraction must be in 0 .. 1, got " & $p.minFraction)
  if p.trendTraces < 1: badNoise("trendTraces must be ≥ 1, got " & $p.trendTraces)
  if p.minArrivals < 1: badNoise("minArrivals must be ≥ 1, got " & $p.minArrivals)
  if p.neighbours < 1: badNoise("neighbours must be ≥ 1, got " & $p.neighbours)
  if p.minNeighbours < 1: badNoise("minNeighbours must be ≥ 1, got " & $p.minNeighbours)
  if p.steps < 0: badNoise("steps must be ≥ 0, got " & $p.steps)
  if not (p.neighbourZ > 0): badNoise("neighbourZ must be > 0, got " & $p.neighbourZ)
  if p.choice == acFixedTime and not finite(p.edgeMs):
    badNoise("a fixed time needs a finite edgeMs, got " & $p.edgeMs)
  if p.choice == acFixedRadius and not (p.radius > 0 and p.radius < Inf):
    badNoise("a fixed radius needs a radius > 0, got " & $p.radius)

# ----------------------------------------------------------------- helpers --

func `<`(x, y: SortKey): bool =
  if x.a != y.a: x.a < y.a
  elif x.b != y.b: x.b < y.b
  else: x.row < y.row

proc siftDown[T](a: var openArray[T], root, e: int) =
  var r = root
  while 2 * r + 1 < e:
    var c = 2 * r + 1
    if c + 1 < e and a[c] < a[c + 1]: inc c
    if a[r] < a[c]:
      swap(a[r], a[c])
      r = c
    else:
      return

proc heapSort[T](a: var openArray[T]) =
  ## An in-place sort that allocates nothing (the stdlib's merge sort does).
  let n = a.len
  for i in countdown(n div 2 - 1, 0): siftDown(a, i, n)
  for e in countdown(n - 1, 1):
    swap(a[0], a[e])
    siftDown(a, 0, e)

func select(x: var openArray[float64], k: int): float64 =
  # The k-th smallest of x (from 0), reordering x: a quickselect.
  var lo = 0
  var hi = x.high
  while lo < hi:
    let mid = (lo + hi) shr 1
    if x[mid] < x[lo]: swap(x[mid], x[lo])
    if x[hi] < x[lo]: swap(x[hi], x[lo])
    if x[hi] < x[mid]: swap(x[hi], x[mid])
    let pivot = x[mid]
    var i = lo
    var j = hi
    while i <= j:
      while x[i] < pivot: inc i
      while pivot < x[j]: dec j
      if i <= j:
        swap(x[i], x[j])
        inc i
        dec j
    if k <= j: hi = j
    elif k >= i: lo = i
    else: return x[k]
  x[k]

func medianOf(x: var openArray[float64]): float64 =
  ## The median of `x` (reordered): the middle one, or the mean of the two
  ## middle ones; NaN when empty.
  if x.len == 0: return NaN
  let m = x.len div 2
  let upper = select(x, m)
  if x.len mod 2 == 1: upper
  else:
    var lower = x[0]
    for i in 1 ..< m: lower = max(lower, x[i])
    (lower + upper) / 2

func lowerBound(win: ptr UncheckedArray[float64], n: int, v: float64): int =
  var lo = 0
  var hi = n
  while lo < hi:
    let mid = (lo + hi) shr 1
    if win[mid] < v: lo = mid + 1 else: hi = mid
  lo

proc slidingMedianInto(vals: openArray[float64], w: int, win: ptr UncheckedArray[float64],
    dst: var openArray[float64]) =
  ## `dst[j]` = the median of `vals[j − w .. j + w]` (cut at the ends), by a
  ## sorted window in `win` (`min(vals.len, 2·w + 2)` elements): O(n · w)
  ## moves, which is a memmove each.
  let m = vals.len
  var size = 0
  var lo = 0
  var hi = -1
  for j in 0 ..< m:
    let newHi = min(m - 1, j + w)
    while hi < newHi:
      inc hi
      let p = lowerBound(win, size, vals[hi])
      if p < size: moveMem(addr win[p + 1], addr win[p], (size - p) * sizeof(float64))
      win[p] = vals[hi]
      inc size
    let newLo = max(0, j - w)
    while lo < newLo:
      let p = lowerBound(win, size, vals[lo])
      if p + 1 < size: moveMem(addr win[p], addr win[p + 1], (size - p - 1) * sizeof(float64))
      dec size
      inc lo
    dst[j] = (if size mod 2 == 1: win[size div 2] else: (win[size div 2 - 1] + win[size div 2]) / 2)

proc fill[T](s: var seq[T], n: int, v: T) =
  s.setLen n
  for i in 0 ..< n: s[i] = v

func farLine(k: TraceKeys, d: GatherDomain): float64 {.inline.} =
  if d == gdCommonSource: k.rcvLine else: k.srcLine

func farPoint(k: TraceKeys, d: GatherDomain): float64 {.inline.} =
  if d == gdCommonSource: k.rcvPoint else: k.srcPoint

func farIndex(k: TraceKeys, d: GatherDomain): int16 {.inline.} =
  if d == gdCommonSource: k.rcvIndex else: k.srcIndex

func farX(k: TraceKeys, d: GatherDomain): float64 {.inline.} =
  if d == gdCommonSource: k.rcvX else: k.srcX

func farY(k: TraceKeys, d: GatherDomain): float64 {.inline.} =
  if d == gdCommonSource: k.rcvY else: k.srcY

func nearLine(k: TraceKeys, d: GatherDomain): float64 {.inline.} =
  if d == gdCommonSource: k.srcLine else: k.rcvLine

func nearPoint(k: TraceKeys, d: GatherDomain): float64 {.inline.} =
  if d == gdCommonSource: k.srcPoint else: k.rcvPoint

func offsetOf(k: TraceKeys): float64 =
  ## |offset|: the header's, else from both stations' coordinates; NaN when unknown.
  if finite(k.offset): abs(k.offset)
  elif finite(k.srcX) and finite(k.srcY) and finite(k.rcvX) and finite(k.rcvY):
    hypot(k.rcvX - k.srcX, k.rcvY - k.srcY)
  else: NaN

func sentinel(x: float64): float64 {.inline.} = (if finite(x): x else: -1.2345e300)

type StationKey = (float64, float64, int16, float64, float64, uint8, uint8)

func stationKey(k: TraceKeys, d: GatherDomain): StationKey =
  (sentinel(farLine(k, d)), sentinel(farPoint(k, d)), farIndex(k, d), sentinel(farX(k, d)),
    sentinel(farY(k, d)), k.depthIndex, k.reshootIndex)

# ------------------------------------------------------------------- roles --

func sensorOf(k: TraceKeys): (SensorClass, bool) =
  ## The row's sensor class and whether its code says rotated.
  case k.traceCode
  of 11: (scPressure, false)
  of 12 .. 14: (scMotion, false)
  of 15 .. 17: (scMotion, true)
  else:
    case k.sensorType
    of 1: (scPressure, false)
    of 2 .. 9: (scMotion, false)
    else: (scSingle, false)

func gatherDomainOf*(meta: ChunkMeta): GatherDomain =
  ## `gdCommonReceiver` when every seismic row has the same receiver station
  ## and more than one source station; else `gdCommonSource`.
  var first = -1
  var sameReceiver = true
  var otherSource = false
  for r in 0 ..< meta.len:
    if meta.keys(r).isSeismic:
      if first < 0: first = r
      else:
        let a = meta.keys(first)
        let b = meta.keys(r)
        if stationKey(a, gdCommonSource) != stationKey(b, gdCommonSource): sameReceiver = false
        if stationKey(a, gdCommonReceiver) != stationKey(b, gdCommonReceiver):
          otherSource = true
  if first >= 0 and sameReceiver and otherSource: gdCommonReceiver else: gdCommonSource

proc linesFromCoordinates(meta: ChunkMeta, domain: GatherDomain, dst: var openArray[TraceRole]) =
  ## Record-order line ids from the far end's coordinates (module documentation).
  var spacings: seq[float64]
  var prevX, prevY = NaN
  for r in 0 ..< meta.len:
    if dst[r].group >= 0:
      let x = farX(meta.keys(r), domain)
      let y = farY(meta.keys(r), domain)
      let d = hypot(x - prevX, y - prevY)
      if d > 0: spacings.add d
      prevX = x
      prevY = y
  let sp = medianOf(spacings)
  var line = 0'i32
  var recent: seq[(float64, float64)]
  for r in 0 ..< meta.len:
    if dst[r].group >= 0:
      let p = (farX(meta.keys(r), domain), farY(meta.keys(r), domain))
      if recent.len > 0 and (p[0] != recent[^1][0] or p[1] != recent[^1][1]):
        let last = recent[^1]
        var split = false
        if sp > 0:
          let vx = p[0] - last[0]
          let vy = p[1] - last[1]
          let dist = hypot(vx, vy)
          var along, perp = 0.0
          if recent.len >= 2:
            let ax = last[0] - recent[0][0]
            let ay = last[1] - recent[0][1]
            let al = hypot(ax, ay)
            if al > 0:
              along = (vx * ax + vy * ay) / al
              perp = abs(vx * ay - vy * ax) / al
            else:
              perp = dist
          else:
            perp = dist
          split = (dist > 3 * sp and perp > 2 * sp) or along < -3 * sp
        if split:
          inc line
          recent.setLen 0
        recent.add p
        if recent.len > 5: recent.delete 0
      elif recent.len == 0:
        recent.add p
      dst[r].line = line

proc linesFromStations(meta: ChunkMeta, domain: GatherDomain, dst: var openArray[TraceRole]) =
  var steps: seq[float64]
  var prev = NaN
  for r in 0 ..< meta.len:
    if dst[r].group >= 0:
      let p = farPoint(meta.keys(r), domain)
      if finite(prev) and p > prev: steps.add p - prev
      prev = p
  let typical = medianOf(steps)
  var line = 0'i32
  prev = NaN
  for r in 0 ..< meta.len:
    if dst[r].group >= 0:
      let p = farPoint(meta.keys(r), domain)
      if finite(prev) and p != prev and (p < prev or (typical > 0 and p - prev > 10 * typical)):
        inc line
      prev = p
      dst[r].line = line

proc traceRolesInto*(meta: ChunkMeta, domain: GatherDomain, rotated: openArray[bool],
    dst: var openArray[TraceRole]) =
  ## Each row's sensor class, group, node, line and station from the keys
  ## (module documentation). `rotated` has one flag per row, or is empty
  ## ("none rotated"; SEG-Y codes 15-17 are rotated always). A row that is
  ## not live (auxiliary, dead by header, failed) gets group -1. Raises
  ## `ValueError` when `dst.len != meta.len` or `rotated` has another length.
  let n = meta.len
  if dst.len != n: badNoise("dst holds " & $dst.len & " roles for " & $n & " rows")
  if rotated.len != 0 and rotated.len != n:
    badNoise("rotated holds " & $rotated.len & " flags for " & $n & " rows")
  var comps: set[uint8]
  for r in 0 ..< n:
    if meta.isLive(r) and meta.keys(r).component != 0: comps.incl meta.keys(r).component
  let multi = comps.card > 1
  var groupIds = initTable[int, int32]()
  var nodeIds = initTable[StationKey, int32]()
  var linePoints = true
  var lineHeaders = true
  var coords = true
  var live = 0
  for r in 0 ..< n:
    dst[r] = TraceRole(sensor: scSingle, group: -1, node: -1, line: -1, station: NaN)
    if meta.isLive(r):
      inc live
      let k = meta.keys(r)
      if not finite(farLine(k, domain)): lineHeaders = false
      if not finite(farPoint(k, domain)): linePoints = false
      if not (finite(farX(k, domain)) and finite(farY(k, domain))): coords = false
      let (sensor, codeRotated) = (if multi: sensorOf(k) else: (scSingle, false))
      let isRot = codeRotated or (rotated.len > 0 and rotated[r])
      var key = 0
      case sensor
      of scSingle: key = 0
      of scPressure: key = 1
      of scMotion: key = (if isRot: 3 + int(k.component) else: 2)
      var g = groupIds.getOrDefault(key, -1)
      if g < 0:
        g = int32(groupIds.len)
        groupIds[key] = g
      dst[r].sensor = sensor
      dst[r].rotated = isRot and sensor == scMotion
      dst[r].group = g
      if sensor == scMotion:
        let sk = stationKey(k, domain)
        var nd = nodeIds.getOrDefault(sk, -1)
        if nd < 0:
          nd = int32(nodeIds.len)
          nodeIds[sk] = nd
        dst[r].node = nd
  # Lines: header numbers, else coordinates, else station numbers, else one.
  if live > 0:
    if lineHeaders:
      var ids = initTable[float64, int32]()
      for r in 0 ..< n:
        if dst[r].group >= 0:
          let v = farLine(meta.keys(r), domain)
          var id = ids.getOrDefault(v, -1)
          if id < 0:
            id = int32(ids.len)
            ids[v] = id
          dst[r].line = id
    elif coords:
      linesFromCoordinates(meta, domain, dst)
    elif linePoints:
      linesFromStations(meta, domain, dst)
    else:
      for r in 0 ..< n:
        if dst[r].group >= 0: dst[r].line = 0
  # Stations: the point number when known, else the rank along the line.
  var ranks = initTable[(int32, StationKey), float64]()
  var counts = initTable[int32, int]()
  for r in 0 ..< n:
    if dst[r].group >= 0:
      let k = meta.keys(r)
      if linePoints:
        dst[r].station = farPoint(k, domain)
      else:
        let key = (dst[r].line, stationKey(k, domain))
        var rank = ranks.getOrDefault(key, -1.0)
        if rank < 0:
          rank = float64(counts.getOrDefault(dst[r].line, 0))
          counts[dst[r].line] = int(rank) + 1
          ranks[key] = rank
        dst[r].station = rank

func noiseWorkspace*(rows: int): WorkspaceSize {.raises: [].} =
  ## The scratch `noiseQc` needs for a gather of up to `rows` traces.
  WorkspaceSize(f64: 2 * max(rows, 0) + 4)

# ------------------------------------------------------------------ stages --

func traceUsable(a: TraceAmplitude): bool {.inline.} =
  a.samples > 0 and finite(a.rmsDb) and a.flags * {afDead, afConstant} == {}

func isNodeUnit(r: TraceRole): bool {.inline.} =
  r.sensor == scMotion and not r.rotated and r.node >= 0

proc prepareUnits(meta: ChunkMeta, amps: var openArray[TraceAmplitude],
    roles: openArray[TraceRole], res: var NoiseResult) =
  ## Usable rows, the units (nodes group their motion rows), their levels
  ## and the groups.
  template s: untyped = res.sc
  let n = meta.len
  s.live.fill(n, false)
  s.usableRow.fill(n, false)
  s.rep.fill(n, 0'i32)
  s.nextMember.fill(n, -1'i32)
  s.nodeFirst.fill(n, -1'i32)
  s.nodeLast.fill(n, -1'i32)
  s.uval.fill(n, NaN)
  s.uUsable.fill(n, false)
  s.uNode.fill(n, false)
  var gMax = -1
  for r in 0 ..< n:
    if roles[r].group >= n or roles[r].node >= n or roles[r].group < -1 or roles[r].node < -1:
      badNoise("row " & $r & ": group and node ids must be in -1 ..< " & $n)
    amps[r].flags.excl {afWeak, afNoisy}
    amps[r].robustZ = NaN
    s.live[r] = meta.isLive(r)
    s.usableRow[r] = s.live[r] and roles[r].group >= 0 and traceUsable(amps[r])
    gMax = max(gMax, int(roles[r].group))
    s.rep[r] = int32(r)
    if isNodeUnit(roles[r]) and roles[r].group >= 0:
      let nd = roles[r].node
      let f = s.nodeFirst[nd]
      if f < 0:
        s.nodeFirst[nd] = int32(r)
        s.nodeLast[nd] = int32(r)
      elif roles[f].group == roles[r].group:
        s.rep[r] = f
        s.nextMember[s.nodeLast[nd]] = int32(r)
        s.nodeLast[nd] = int32(r)
  res.groups.setLen(gMax + 1)
  for g in 0 ..< res.groups.len:
    res.groups[g] = GroupNoise(medianDb: NaN, madDb: NaN, edge: NaN)
  res.partial.fill(n, false)
  s.seenGroup.fill(res.groups.len, false)
  for r in 0 ..< n:
    let g = roles[r].group
    if g >= 0 and not s.seenGroup[g]:
      s.seenGroup[g] = true
      res.groups[g].sensor = roles[r].sensor
      if roles[r].sensor == scMotion and roles[r].rotated:
        res.groups[g].component = meta.keys(r).component
  for r in 0 ..< n:
    if s.rep[r] == int32(r) and roles[r].group >= 0:
      if isNodeUnit(roles[r]):
        var sum = 0.0
        var k = 0
        var m = int32(r)
        while m >= 0:
          if s.usableRow[m]:
            sum += pow(10.0, amps[m].rmsDb / 10)
            inc k
          m = s.nextMember[m]
        s.uNode[r] = true
        s.uUsable[r] = k >= 2
        if k >= 1: s.uval[r] = 10 * log10(sum / float64(k))
        if k == 2:
          m = int32(r)
          while m >= 0:
            res.partial[m] = true
            m = s.nextMember[m]
      else:
        s.uUsable[r] = s.usableRow[r]
        s.uval[r] = amps[r].rmsDb

proc orderLines(meta: ChunkMeta, roles: openArray[TraceRole], res: var NoiseResult) =
  ## The live rows by (line, station), cut into line segments.
  template s: untyped = res.sc
  let n = meta.len
  s.ordered.setLen 0
  s.segOf.fill(n, -1'i32)
  for r in 0 ..< n:
    if s.live[r]:
      let st = roles[r].station
      s.ordered.add SortKey(a: roles[r].line, b: (if finite(st): st else: float64(r)),
        row: int32(r))
  heapSort(s.ordered)
  s.segStart.setLen 0
  var prev = low(int32)
  for p in 0 ..< s.ordered.len:
    if p == 0 or s.ordered[p].a != prev:
      s.segStart.add int32(p)
      prev = s.ordered[p].a
    s.segOf[s.ordered[p].row] = int32(s.segStart.len - 1)
  s.segStart.add int32(s.ordered.len)

func pickOk(q: FirstBreakPick, minConf: float64): bool {.inline.} =
  finite(q.timeMs) and q.confidence >= minConf

proc windowMedian(res: var NoiseResult, seg, p, k0: int, vals: openArray[float64]): float64 =
  ## The median of the finite `vals` (by row) over the positions `p ± k` of
  ## the line segment `seg`, k = `k0`, doubled up to 4 × `k0` until 3 are found;
  ## NaN with fewer than 3.
  template s: untyped = res.sc
  let a = int(s.segStart[seg])
  let b = int(s.segStart[seg + 1])
  var k = k0
  while true:
    s.tmp.setLen 0
    for q in max(a, p - k) ..< min(b, p + k + 1):
      let v = vals[s.ordered[q].row]
      if finite(v): s.tmp.add v
    if s.tmp.len >= 3 or k >= 4 * k0: break
    k *= 2
  if s.tmp.len >= 3: medianOf(s.tmp) else: NaN

proc trendInto(meta: ChunkMeta, picks: openArray[FirstBreakPick], params: ExclusionParams,
    res: var NoiseResult): int =
  ## The pick trend along each line into `trendArr`; returns the number of coherent picks.
  template s: untyped = res.sc
  let n = meta.len
  s.pickMs.fill(n, NaN)
  s.coherentMs.fill(n, NaN)
  s.trendT1.fill(n, NaN)
  s.trendArr.fill(n, NaN)
  for r in 0 ..< n:
    if s.live[r] and pickOk(picks[r], params.minConfidence): s.pickMs[r] = picks[r].timeMs
  for seg in 0 ..< s.segStart.len - 1:
    for p in int(s.segStart[seg]) ..< int(s.segStart[seg + 1]):
      s.trendT1[s.ordered[p].row] = windowMedian(res, seg, p, params.trendTraces, s.pickMs)
  for r in 0 ..< n:
    let t1 = s.trendT1[r]
    if finite(t1) and finite(s.pickMs[r]) and
        abs(s.pickMs[r] - t1) <= clamp(0.05 * abs(t1), 20.0, 60.0):
      s.coherentMs[r] = s.pickMs[r]
      inc result
  for seg in 0 ..< s.segStart.len - 1:
    for p in int(s.segStart[seg]) ..< int(s.segStart[seg + 1]):
      s.trendArr[s.ordered[p].row] = windowMedian(res, seg, p, params.trendTraces, s.coherentMs)

proc unitArrivals(arr: openArray[float64], res: var NoiseResult) =
  ## `uarr[rep]`: a trace's arrival, a node's median over its traces.
  template s: untyped = res.sc
  let n = arr.len
  s.uarr.fill(n, NaN)
  for r in 0 ..< n:
    if s.rep[r] == int32(r):
      if s.uNode[r]:
        s.tmp.setLen 0
        var m = int32(r)
        while m >= 0:
          if finite(arr[m]): s.tmp.add arr[m]
          m = s.nextMember[m]
        s.uarr[r] = medianOf(s.tmp)
      else:
        s.uarr[r] = arr[r]

proc edgeFor(arr: openArray[float64], roles: openArray[TraceRole], params: ExclusionParams,
    fixed: float64, mask: var seq[bool], edges: var seq[float64], res: var NoiseResult,
    ws: var Workspace, noLoud: var bool): int =
  ## Marks `mask[rep]` for the units arriving before the edge: the level
  ## rule per group (`fixed` NaN), or the edge `fixed`. Returns the groups the
  ## rule applied to (20 or more units with an arrival; every group for a
  ## fixed edge); `edges[g]` is the edge, -Inf for "nothing".
  template s: untyped = res.sc
  let n = arr.len
  mask.fill(n, false)
  edges.fill(res.groups.len, NaN)
  unitArrivals(arr, res)
  let win = ws.scratch(float64, n + 2)
  for g in 0 ..< res.groups.len:
    s.sorted.setLen 0
    s.tmp2.setLen 0
    for r in 0 ..< n:
      if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r]:
        if finite(s.uarr[r]): s.sorted.add SortKey(b: s.uarr[r], row: int32(r))
        else: s.tmp2.add s.uval[r]
    let m = s.sorted.len
    var edge = NaN
    if finite(fixed):
      edge = fixed
      inc result
    elif m >= 20:
      heapSort(s.sorted)
      s.vals.setLen m
      for j in 0 ..< m: s.vals[j] = s.uval[s.sorted[j].row]
      s.tmp.setLen 0
      for j in m div 2 ..< m: s.tmp.add s.vals[j]
      for v in s.tmp2: s.tmp.add v
      let base = medianOf(s.tmp)
      s.level.setLen m
      slidingMedianInto(s.vals, max(5, m div 20), win, s.level)
      var top = 0
      for j in 1 ..< m:
        if s.level[j] > s.level[top]: top = j
      if s.level[top] <= base + params.levelDb:
        edge = -Inf
        noLoud = true
      else:
        var j = top
        while j < m and s.level[j] > base + params.levelDb: inc j
        edge = s.sorted[min(j, m - 1)].b
      inc result
    edges[g] = edge
    if not edge.isNaN:
      for r in 0 ..< n:
        if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r] and
            finite(s.uarr[r]) and (s.uarr[r] < edge or (not finite(fixed) and s.uarr[r] == edge)):
          mask[r] = true
  for r in 0 ..< n:
    mask[r] = mask[s.rep[r]]

proc typicalSpacing(meta: ChunkMeta, domain: GatherDomain, res: var NoiseResult): float64 =
  ## The median distance between consecutive distinct far-end stations along the lines.
  template s: untyped = res.sc
  s.tmp.setLen 0
  for seg in 0 ..< s.segStart.len - 1:
    for p in int(s.segStart[seg]) + 1 ..< int(s.segStart[seg + 1]):
      let a = meta.keys(s.ordered[p - 1].row)
      let b = meta.keys(s.ordered[p].row)
      let d = hypot(farX(b, domain) - farX(a, domain), farY(b, domain) - farY(a, domain))
      if finite(d) and d > 0: s.tmp.add d
  medianOf(s.tmp)

proc sourceFitScore(px, py, pt: openArray[float64], cx, cy: float64,
    buf: var seq[float64]): float64 =
  ## The median absolute residual of a line `t = a + b·d` fitted to the
  ## picks' times against their distances `d` from (`cx`, `cy`); Inf when
  ## the slope is not positive.
  let m = px.len
  var sd, st, sdd, sdt = 0.0
  buf.setLen m
  for i in 0 ..< m:
    let d = hypot(px[i] - cx, py[i] - cy)
    buf[i] = d
    sd += d
    st += pt[i]
    sdd += d * d
    sdt += d * pt[i]
  let fm = float64(m)
  let den = fm * sdd - sd * sd
  if den <= 0: return Inf
  let b = (fm * sdt - sd * st) / den
  if b <= 0: return Inf
  let a = (st - b * sd) / fm
  for i in 0 ..< m: buf[i] = abs(pt[i] - (a + b * buf[i]))
  medianOf(buf)

proc fitSourceXY(meta: ChunkMeta, domain: GatherDomain, params: ExclusionParams,
    res: var NoiseResult, sx, sy: var float64): bool =
  ## The common point's XY that best explains the coherent picks as a line in
  ## the distance: a grid search over at most 400 far-end positions, refined
  ## to half a spacing. False without 5 coherent picks with coordinates.
  template s: untyped = res.sc
  var px, py, pt: seq[float64]
  var all = 0
  for r in 0 ..< meta.len:
    if finite(s.coherentMs[r]) and finite(farX(meta.keys(r), domain)) and
        finite(farY(meta.keys(r), domain)): inc all
  if all < 5: return false
  let stride = max(1, all div 500)
  var seen = 0
  for r in 0 ..< meta.len:
    if finite(s.coherentMs[r]) and finite(farX(meta.keys(r), domain)) and
        finite(farY(meta.keys(r), domain)):
      if seen mod stride == 0:
        px.add farX(meta.keys(r), domain)
        py.add farY(meta.keys(r), domain)
        pt.add s.coherentMs[r]
      inc seen
  var cands = 0
  for r in 0 ..< meta.len:
    if s.live[r] and finite(farX(meta.keys(r), domain)) and finite(farY(meta.keys(r), domain)):
      inc cands
  let cstride = max(1, cands div 400)
  var best = Inf
  var seenC = 0
  var buf: seq[float64]
  for r in 0 ..< meta.len:
    if s.live[r] and finite(farX(meta.keys(r), domain)) and finite(farY(meta.keys(r), domain)):
      if seenC mod cstride == 0:
        let cx = farX(meta.keys(r), domain)
        let cy = farY(meta.keys(r), domain)
        let sc = sourceFitScore(px, py, pt, cx, cy, buf)
        if sc < best:
          best = sc
          sx = cx
          sy = cy
      inc seenC
  if best == Inf: return false
  let h = typicalSpacing(meta, domain, res) / 2
  if finite(h) and h > 0:
    var improved = true
    var rounds = 0
    while improved and rounds < 40:
      improved = false
      inc rounds
      for dx in [-h, 0.0, h]:
        for dy in [-h, 0.0, h]:
          if dx != 0 or dy != 0:
            let sc = sourceFitScore(px, py, pt, sx + dx, sy + dy, buf)
            if sc < best:
              best = sc
              sx += dx
              sy += dy
              improved = true
  true

proc stepsInside(meta: ChunkMeta, roles: openArray[TraceRole], domain: GatherDomain,
    params: ExclusionParams, coherentOk: bool, res: var NoiseResult,
    mask: var seq[bool]): bool =
  ## The station-step area: false when no line has a source position.
  template s: untyped = res.sc
  let n = meta.len
  let segs = s.segStart.len - 1
  s.srcStation.fill(segs, NaN)
  s.stepSize.fill(segs, 1.0)
  mask.fill(n, false)
  # The common point's position along each line.
  var sx, sy = NaN
  var fitted = false
  var fitTried = false
  var any = false
  for seg in 0 ..< segs:
    let a = int(s.segStart[seg])
    let b = int(s.segStart[seg + 1])
    var bestOff = Inf
    var pos = NaN
    for p in a ..< b:
      let o = s.absOff[s.ordered[p].row]
      if finite(o) and o < bestOff:
        bestOff = o
        pos = s.ordered[p].b
    if not finite(pos) and coherentOk:
      if not fitTried:
        fitTried = true
        fitted = fitSourceXY(meta, domain, params, res, sx, sy)
      if fitted:
        var bd = Inf
        for p in a ..< b:
          let k = meta.keys(s.ordered[p].row)
          let d = hypot(farX(k, domain) - sx, farY(k, domain) - sy)
          if finite(d) and d < bd:
            bd = d
            pos = s.ordered[p].b
    if not finite(pos):
      # The common point's own line and point, when it is this line (2-D).
      let k0 = meta.keys(s.ordered[a].row)
      let cl = nearLine(k0, domain)
      let cp = nearPoint(k0, domain)
      var pointsKnown = true
      for p in a ..< b:
        if not finite(farPoint(meta.keys(s.ordered[p].row), domain)): pointsKnown = false
      if finite(cl) and finite(cp) and pointsKnown and cl == farLine(k0, domain):
        var bd = Inf
        for p in a ..< b:
          let d = abs(s.ordered[p].b - cp)
          if d < bd:
            bd = d
            pos = s.ordered[p].b
    s.srcStation[seg] = pos
    any = any or finite(pos)
    # The most common station increment.
    s.tmp.setLen 0
    for p in a + 1 ..< b:
      let d = s.ordered[p].b - s.ordered[p - 1].b
      if d > 0: s.tmp.add d
    if s.tmp.len > 0:
      heapSort(s.tmp)
      var bestRun = 0
      var run = 0
      for i in 0 ..< s.tmp.len:
        run = (if i > 0 and s.tmp[i] == s.tmp[i - 1]: run + 1 else: 1)
        if run > bestRun:
          bestRun = run
          s.stepSize[seg] = s.tmp[i]
  if not any: return false
  # Inside: within the steps of the source position, on lines the shot reaches.
  for g in 0 ..< res.groups.len:
    s.tmp.setLen 0
    for r in 0 ..< n:
      if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r]:
        s.tmp.add s.uval[r]
    let gm = medianOf(s.tmp)
    for seg in 0 ..< segs:
      if finite(s.srcStation[seg]):
        let reach = float64(params.steps) * s.stepSize[seg] + 1e-9
        s.vals.setLen 0
        for p in int(s.segStart[seg]) ..< int(s.segStart[seg + 1]):
          let r = int(s.ordered[p].row)
          if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r] and
              abs(s.ordered[p].b - s.srcStation[seg]) <= reach:
            s.vals.add s.uval[r]
        if s.vals.len > 0 and
            (params.stepsOnAllLines or medianOf(s.vals) > gm + params.levelDb):
          for p in int(s.segStart[seg]) ..< int(s.segStart[seg + 1]):
            let r = int(s.ordered[p].row)
            if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r] and
                abs(s.ordered[p].b - s.srcStation[seg]) <= reach:
              mask[r] = true
  for r in 0 ..< n:
    mask[r] = mask[s.rep[r]]
  true

proc judge(roles: openArray[TraceRole], amp: AmplitudeParams, res: var NoiseResult,
    amps: var openArray[TraceAmplitude]) =
  ## The statistics outside the area, the flags, `robustZ`.
  template s: untyped = res.sc
  let n = amps.len
  s.uz.fill(n, NaN)
  for g in 0 ..< res.groups.len:
    var lowest = Inf
    var m = 0
    for r in 0 ..< n:
      if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r] and
          not res.inside[r]:
        lowest = min(lowest, s.uval[r])
        inc m
    if m > 0:
      s.tmp.setLen 0
      for r in 0 ..< n:
        if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r] and
            not res.inside[r]:
          s.tmp.add s.uval[r] - lowest
      let median = medianOf(s.tmp) + lowest
      var k = 0
      for r in 0 ..< n:
        if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r] and
            not res.inside[r]:
          s.tmp[k] = abs(s.uval[r] - median)
          inc k
      let mad = medianOf(s.tmp)
      res.groups[g].medianDb = median
      res.groups[g].madDb = mad
      let scale = 1.4826 * mad
      for r in 0 ..< n:
        if s.rep[r] == int32(r) and roles[r].group == int32(g) and s.uUsable[r]:
          let d = s.uval[r] - median
          s.uz[r] = (if scale > 0: d / scale elif d == 0: 0.0 elif d > 0: Inf else: -Inf)
  for r in 0 ..< n:
    let u = s.rep[r]
    if s.usableRow[r] and s.uUsable[u] and not s.uz[u].isNaN:
      amps[r].robustZ = s.uz[u]
      if s.uz[u] > amp.outlierZ and not res.inside[r]: amps[r].flags.incl afNoisy
      elif s.uz[u] < -amp.outlierZ: amps[r].flags.incl afWeak

proc neighbourFinding(roles: openArray[TraceRole], params: ExclusionParams, res: var NoiseResult) =
  ## "Louder than its neighbours", per group, per line.
  template s: untyped = res.sc
  let n = res.inside.len
  s.nbRes.fill(n, NaN)
  for g in 0 ..< res.groups.len:
    # The eligible units in (line, station) order, with their line segment.
    s.elig.setLen 0
    for p in 0 ..< s.ordered.len:
      let r = s.ordered[p].row
      if s.rep[r] == r and roles[r].group == int32(g) and s.uUsable[r] and not res.inside[r]:
        s.elig.add int32(p)
    for e in 0 ..< s.elig.len:
      let r = s.ordered[s.elig[e]].row
      s.tmp.setLen 0
      var lo = e - 1
      while lo >= 0 and e - lo <= params.neighbours and
          s.segOf[s.ordered[s.elig[lo]].row] == s.segOf[r]:
        s.tmp.add s.uval[s.ordered[s.elig[lo]].row]
        dec lo
      var hi = e + 1
      while hi < s.elig.len and hi - e <= params.neighbours and
          s.segOf[s.ordered[s.elig[hi]].row] == s.segOf[r]:
        s.tmp.add s.uval[s.ordered[s.elig[hi]].row]
        inc hi
      if s.tmp.len >= params.minNeighbours:
        s.nbRes[r] = s.uval[r] - medianOf(s.tmp)
    # Their robust z-scores.
    s.tmp.setLen 0
    for e in 0 ..< s.elig.len:
      let v = s.nbRes[s.ordered[s.elig[e]].row]
      if finite(v): s.tmp.add v
    if s.tmp.len > 0:
      let med = medianOf(s.tmp)
      for v in s.tmp.mitems: v = abs(v - med)
      let scale = 1.4826 * medianOf(s.tmp)
      for e in 0 ..< s.elig.len:
        let r = s.ordered[s.elig[e]].row
        let v = s.nbRes[r]
        if finite(v):
          let d = v - med
          let z = (if scale > 0: d / scale elif d == 0: 0.0 elif d > 0: Inf else: -Inf)
          res.neighbourZ[r] = z

proc finish(meta: ChunkMeta, picks: openArray[FirstBreakPick], roles: openArray[TraceRole],
    params: ExclusionParams, timeEdge: bool, res: var NoiseResult,
    amps: openArray[TraceAmplitude]) =
  ## States, per-row copies of the unit findings, the early-break note and the counts.
  template s: untyped = res.sc
  let n = meta.len
  for r in 0 ..< n:
    let u = s.rep[r]
    if not s.usableRow[r]:
      res.state[r] = nsUnusable
      res.neighbourState[r] = nbUnusable
    elif not s.uUsable[u]:
      res.state[r] = nsIncompleteNode
      res.neighbourState[r] = nbUnusable
    elif res.inside[r]:
      res.state[r] = nsInArea
      res.neighbourState[r] = nbInArea
    else:
      res.state[r] = nsJudged
      res.neighbourState[r] = (if finite(s.nbRes[u]): nbJudged else: nbFewNeighbours)
    if res.neighbourState[r] == nbJudged:
      res.neighbourDb[r] = s.nbRes[u]
      res.neighbourZ[r] = res.neighbourZ[u]
      res.louder[r] = res.neighbourZ[u] > params.neighbourZ
    if timeEdge and picks.len == n and s.usableRow[r] and not res.inside[r] and
        pickOk(picks[r], params.minConfidence):
      let g = roles[r].group
      if g >= 0 and finite(res.groups[g].edge) and picks[r].timeMs < res.groups[g].edge:
        res.earlyBreak[r] = true
  res.excluded = 0
  res.judged = 0
  res.notJudged = 0
  for r in 0 ..< n:
    case res.state[r]
    of nsJudged: inc res.judged
    of nsInArea:
      inc res.excluded
      inc res.notJudged
    else: inc res.notJudged
  res.linesReached = 0
  for seg in 0 ..< s.segStart.len - 1:
    var reached = false
    for p in int(s.segStart[seg]) ..< int(s.segStart[seg + 1]):
      if res.inside[s.ordered[p].row]: reached = true
    if reached: inc res.linesReached
  for r in 0 ..< n:
    if s.rep[r] == int32(r) and roles[r].group >= 0 and s.uUsable[r]:
      let g = int(roles[r].group)
      inc res.groups[g].units
      if res.inside[r]: inc res.groups[g].inArea else: inc res.groups[g].judged
      if afNoisy in amps[r].flags: inc res.groups[g].noisy
      if afWeak in amps[r].flags: inc res.groups[g].weak
      if res.louder[r]: inc res.groups[g].louder

proc noiseImpl(meta: ChunkMeta, amps: var openArray[TraceAmplitude],
    picks: openArray[FirstBreakPick], hasFit: bool, fitUsed: int, modelMs: openArray[float64],
    roles: openArray[TraceRole], amp: AmplitudeParams, params: ExclusionParams,
    domain: GatherDomain, res: var NoiseResult, ws: var Workspace) =
  template s: untyped = res.sc
  let n = meta.len
  res.domain = domain
  res.rule = arNone
  res.reasons = {}
  res.offsetOrigin = params.offsetOrigin
  res.edgeMs = NaN
  res.radius = NaN
  res.steps = 0
  res.lines = 0
  res.linesReached = 0
  res.agreement = NaN
  res.arrival.fill(n, NaN)
  res.inside.fill(n, false)
  res.state.fill(n, nsUnusable)
  res.neighbourDb.fill(n, NaN)
  res.neighbourZ.fill(n, NaN)
  res.louder.fill(n, false)
  res.neighbourState.fill(n, nbUnusable)
  res.earlyBreak.fill(n, false)
  prepareUnits(meta, amps, roles, res)
  orderLines(meta, roles, res)
  res.lines = s.segStart.len - 1
  # How the lines were found.
  var headers = n > 0
  var live = 0
  var withOffsets = 0
  var withModel = 0
  s.absOff.fill(n, NaN)
  for r in 0 ..< n:
    if s.live[r]:
      inc live
      let k = meta.keys(r)
      if not finite(farLine(k, domain)): headers = false
      s.absOff[r] = offsetOf(k)
      if finite(s.absOff[r]): inc withOffsets
      if hasFit and finite(modelMs[r]): inc withModel
  res.linesFromHeaders = headers and live > 0
  res.lineSource = (if res.linesFromHeaders: lsHeaders
    elif res.lines <= 1: lsUnknown
    else:
      var all = true
      for r in 0 ..< n:
        if s.live[r] and not (finite(farX(meta.keys(r), domain)) and
            finite(farY(meta.keys(r), domain))): all = false
      if all: lsCoordinates else: lsStations)
  let minArr = max(params.minArrivals, int(ceil(params.minFraction * float64(live))))
  let havePicks = picks.len == n and n > 0
  let offsetsOk = withOffsets * 2 > live
  let modelOk = hasFit and withModel * 2 > live and fitUsed >= minArr
  var coherent = 0
  if havePicks: coherent = trendInto(meta, picks, params, res)
  let trendOk = havePicks and coherent >= minArr
  var noLoud = false
  var applied = 0
  var timeEdge = false
  template adopt(arr: openArray[float64], ruleName: AreaRule, fixedEdge: float64) =
    applied = edgeFor(arr, roles, params, fixedEdge, res.inside, s.edges, res, ws, noLoud)
    if applied > 0:
      res.rule = ruleName
      for r in 0 ..< n: res.arrival[r] = arr[r]
  case params.choice
  of acOff: discard
  of acSteps:
    if stepsInside(meta, roles, domain, params, trendOk or coherent >= 5, res, res.inside):
      res.rule = arSteps
      res.steps = params.steps
    else:
      res.reasons.incl rnNoPosition
  of acFixedTime:
    if modelOk: adopt(modelMs, arFixedTime, params.edgeMs)
    elif trendOk: adopt(s.trendArr, arFixedTime, params.edgeMs)
    else: res.reasons.incl rnOverrideNeedsPicks
    timeEdge = res.rule == arFixedTime
  of acFixedRadius:
    if offsetsOk: adopt(s.absOff, arFixedRadius, params.radius)
    else: res.reasons.incl rnOverrideNeedsOffsets
  of acAuto:
    if modelOk:
      adopt(modelMs, arModel, NaN)
      if res.rule == arModel: timeEdge = true
      else: res.reasons.incl rnFewTraces
    else:
      if not hasFit: res.reasons.incl rnNoPicks
      elif not offsetsOk: res.reasons.incl rnNoOffsets
      else: res.reasons.incl rnFewPicks
    if res.rule == arNone and trendOk:
      adopt(s.trendArr, arTrend, NaN)
      if res.rule == arTrend: timeEdge = true
      else: res.reasons.incl rnFewTraces
    elif res.rule == arNone:
      res.reasons.incl (if havePicks: rnFewPicks else: rnNoPicks)
    if res.rule == arNone and offsetsOk:
      adopt(s.absOff, arRadius, NaN)
      if res.rule != arRadius: res.reasons.incl rnFewTraces
    elif res.rule == arNone:
      res.reasons.incl rnNoOffsets
    if res.rule == arNone:
      if stepsInside(meta, roles, domain, params, trendOk or coherent >= 5, res, res.inside):
        res.rule = arSteps
        res.steps = params.steps
      else:
        res.reasons.incl rnNoPosition
  if res.rule == arModel or res.rule == arTrend or res.rule == arFixedTime or
      res.rule == arRadius or res.rule == arFixedRadius:
    for g in 0 ..< res.groups.len: res.groups[g].edge = s.edges[g]
    for g in 0 ..< res.groups.len:
      if finite(s.edges[g]):
        if res.rule == arRadius or res.rule == arFixedRadius:
          if not finite(res.radius): res.radius = s.edges[g]
        elif not finite(res.edgeMs): res.edgeMs = s.edges[g]
    if noLoud: res.reasons.incl rnNoLoudness
  # Model and trend both there: how much their areas agree.
  if params.choice == acAuto and res.rule == arModel and trendOk:
    var dummy = false
    s.inside2.setLen n
    discard edgeFor(s.trendArr, roles, params, NaN, s.inside2, s.edges2, res, ws, dummy)
    var agree = 0
    var total = 0
    for r in 0 ..< n:
      if s.usableRow[r]:
        inc total
        if s.inside2[r] == res.inside[r]: inc agree
    if total > 0: res.agreement = float64(agree) / float64(total)
  judge(roles, amp, res, amps)
  neighbourFinding(roles, params, res)
  finish(meta, picks, roles, params, timeEdge, res, amps)

proc checkInputs(meta: ChunkMeta, amps: openArray[TraceAmplitude],
    picks: openArray[FirstBreakPick], roles: openArray[TraceRole], amp: AmplitudeParams,
    params: ExclusionParams, ws: var Workspace) =
  validate(amp)
  validate(params)
  let n = meta.len
  if amps.len != n: badNoise("amps holds " & $amps.len & " values for " & $n & " rows")
  if roles.len != n: badNoise("roles holds " & $roles.len & " values for " & $n & " rows")
  if picks.len != 0 and picks.len != n:
    badNoise("picks holds " & $picks.len & " values for " & $n & " rows")
  if not ws.fits(noiseWorkspace(n)):
    badNoise("the workspace holds " & $ws.capacity & "; reserve noiseWorkspace(" & $n & ")")

proc noiseQc*(meta: ChunkMeta, amps: var openArray[TraceAmplitude],
    picks: openArray[FirstBreakPick], fit: ShotFit, roles: openArray[TraceRole],
    amp: AmplitudeParams, params: ExclusionParams, domain: GatherDomain, res: var NoiseResult,
    ws: var Workspace) =
  ## The noisy and weak traces of one gather outside its exclusion area, and
  ## the units louder than their neighbours (module documentation). `amps`
  ## come from `traceAmplitudeInto`; this sets `robustZ`, `afNoisy` and
  ## `afWeak` (replacing what `recordAmplitude` does for the record) and
  ## fills `res`. `fit` is the shot's `fitShot` result (a viewer's
  ## first-break overlay computes one: pass it, so a record is neither
  ## picked nor fitted twice); `picks` may be empty (the rules that need
  ## them are skipped). `ws` must hold `noiseWorkspace(meta.len)`. Raises
  ## `ValueError` for mismatched lengths (`fit` too), invalid parameters or a
  ## workspace too small.
  checkInputs(meta, amps, picks, roles, amp, params, ws)
  if fit.modelMs.len != meta.len:
    badNoise("fit holds " & $fit.modelMs.len & " model times for " & $meta.len & " rows")
  noiseImpl(meta, amps, picks, true, fit.used, fit.modelMs, roles, amp, params, domain, res, ws)

proc noiseQc*(meta: ChunkMeta, amps: var openArray[TraceAmplitude],
    picks: openArray[FirstBreakPick], roles: openArray[TraceRole], amp: AmplitudeParams,
    params: ExclusionParams, domain: GatherDomain, res: var NoiseResult, ws: var Workspace) =
  ## `noiseQc` that fits the shot itself from `picks` (`fitShot` without
  ## correlation; it allocates a few times). Without picks there is no fit.
  checkInputs(meta, amps, picks, roles, amp, params, ws)
  if picks.len == meta.len and picks.len > 0:
    fitShot(meta, picks, ShotFitParams(minConfidence: params.minConfidence), res.sc.fit)
    noiseImpl(meta, amps, picks, true, res.sc.fit.used, res.sc.fit.modelMs, roles, amp, params,
      domain, res, ws)
  else:
    noiseImpl(meta, amps, picks, false, 0, [], roles, amp, params, domain, res, ws)
