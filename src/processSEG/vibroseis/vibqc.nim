## Vibrator QC of a record (`analyzeVibrators`): for each vibrator unit, its
## reference (the pilot), ground force (GF), reaction-mass (M) and baseplate
## (B) accelerations, as the caller assigns them to rows; the ground force
## generated from M and B; every measure of `sweeptrack` against the unit's
## reference; and, where the caller sets limits, the points beyond them and a
## summary per signal. The viewer draws; this module fills its buffers.
##
## ```nim
## let tp = initSweepTrackPlan(float32, TrackParams(), dtMs = 1, samples = 8001)
## var ws = initWorkspace()
## ws.reserve(vibratorWorkspace(tp))
## var units = vibUnitsInOrder(count = 2,                  # Ref, GF, M, B per unit
##   reactionMass = mass(2450, muKilogram), baseplateMass = mass(1180, muKilogram))
## var results: seq[UnitSignature[float32]]
## tp.analyzeVibrators(samples, meta, units, VibLimits(phaseDeg: 5), results, ws)
## # results[0].tracks[vsGroundForce].phaseDeg, .amplitude[0] (fundamental
## # force), .thdPct, .peak; results[0].weightedSum: GF from M and B;
## # results[0].summaries[vsGroundForce].failed, .flags[vsGroundForce]
## ```
##
## **Roles.** `VibUnit.rows` gives each role's row in the block (−1: the unit
## has none); the reference is required. The library never takes a role from
## a row's position: `vibUnitsInOrder` builds the usual layout (each unit's
## channels in the order Ref, GF, M, B, or an order given) for the caller to
## pass, and any other assignment is a `VibUnit` of the caller's.
##
## **Ground force from M and B.** `GF = Mr·a_M + Mb·a_B` (the weighted sum, as
## vibrator controllers form it), with the reaction mass `Mr` and the
## baseplate mass `Mb` entered with their unit (`Mass`): accelerations in
## m/s² give newtons. The sign is the accelerometers': if the generated force
## comes out at 180° to a recorded one, they are wired the other way:
## reverse both rows (`reversePolarity` on the block's M and B rows) before
## the call, since reversing `weightedSum` after it leaves its track as it
## was.
##
## **Units.** The rows are taken in the unit the caller says they are in
## (`VibUnit.units`, `auRecorded` by default; for calibrated rows the unit
## from `resolveInto`). A role's `Sensitivity`, when set, takes a row in volts
## on to its quantity: M and B in V/g or V/(m/s²) to m/s², a ground-force
## channel in V/N, V/kN or V/lbf to N (a calibration gives auxiliary rows
## volts only, so wireline channels get their physical units here). The
## measures are in the resulting unit (`UnitSignature.units`); the generated
## ground force is in N when both accelerations are in m/s², and
## `auRecorded` otherwise.
##
## **Limits** (`VibLimits`, every field NaN, i.e. off, by default). At each
## point between `fromMs` and `toMs` (NaN: no bound; set them to leave out
## the sweep's tapers): `|phase| > phaseDeg`, `THD > thdPct`, any harmonic's
## `100·A_k/A_1 > harmonicPct`, the fundamental below `forceLow` or above
## `forceHigh`, the peak above `peakForce`; over those points, the mean
## `|phase|` above `meanPhaseDeg`, the mean THD above `meanThdPct`, the mean
## fundamental below `meanForceLow`. `limitFlagsInto` gives each point the
## set of limits it exceeds, `summarize` the statistics and the limits
## failed. In `analyzeVibrators` the limits apply to the two ground forces,
## recorded and generated; the other signals are summarized without them.
##
## **Data.** A failed row's measures are NaN (and so is the generated force
## when M or B failed); a dead row is analysed as it is. Bad arguments raise
## `ValueError` before any result changes.

import std/math
import parseSEG
import ../core/[types, rows, units, workspace]
import sweeptrack

type
  VibSignal* = enum
    ## A vibrator unit's signals: the four roles a row can have, and the
    ## ground force generated from M and B.
    vsReference = "Ref" ## The pilot (as recorded, or a generated sweep).
    vsGroundForce = "GF" ## The ground force (recorded).
    vsMass = "M" ## The reaction-mass acceleration.
    vsBaseplate = "B" ## The baseplate acceleration.
    vsWeightedSum = "GF (M, B)" ## `Mr·a_M + Mb·a_B`, generated.

  VibRole* = range[vsReference .. vsBaseplate]
    ## The roles a row can have.

  VibUnit* = object
    ## One vibrator unit's rows and constants; see the module documentation.
    rows*: array[VibRole, int] = [vsReference: -1, vsGroundForce: -1, vsMass: -1,
      vsBaseplate: -1]   ## Each role's row in the block; −1: none.
    units*: array[VibRole, AmplitudeUnit]
      ## The unit each row is in as passed (`auRecorded` by default).
    sensitivity*: array[VibRole, Sensitivity]
      ## Optional: takes a row in volts on to its quantity (M, B: acceleration;
      ## GF: force); not set (`Sensitivity()`) by default.
    reactionMass*: Mass ## `Mr`, for the generated ground force.
    baseplateMass*: Mass ## `Mb`, for the generated ground force.

  LimitKind* = enum
    ## A limit a signal exceeds.
    lkPhase = "phase"          ## `|phase| > phaseDeg` at a point.
    lkThd = "THD"              ## `THD > thdPct` at a point.
    lkHarmonic = "harmonic"    ## A harmonic's `100·A_k/A_1 > harmonicPct` at a point.
    lkForceLow = "force low"   ## The fundamental below `forceLow` at a point.
    lkForceHigh = "force high" ## The fundamental above `forceHigh` at a point.
    lkPeakForce = "peak force" ## The peak above `peakForce` at a point.
    lkMeanPhase = "mean phase" ## The mean `|phase|` above `meanPhaseDeg`.
    lkMeanThd = "mean THD"     ## The mean THD above `meanThdPct`.
    lkMeanForce = "mean force" ## The mean fundamental below `meanForceLow`.

  VibLimits* = object
    ## Optional limits (module documentation); NaN: none.
    phaseDeg*: float64 = NaN     ## Largest `|phase|` at a point, degrees.
    thdPct*: float64 = NaN       ## Largest THD at a point, %.
    harmonicPct*: float64 = NaN  ## Largest level of any one harmonic at a point, % of the fundamental.
    forceLow*: float64 = NaN     ## Smallest fundamental at a point, in the signal's unit.
    forceHigh*: float64 = NaN    ## Largest fundamental at a point.
    peakForce*: float64 = NaN    ## Largest peak at a point (the hold-down weight).
    meanPhaseDeg*: float64 = NaN ## Largest mean `|phase|`.
    meanThdPct*: float64 = NaN   ## Largest mean THD.
    meanForceLow*: float64 = NaN ## Smallest mean fundamental.
    fromMs*: float64 = NaN       ## Points before this are not checked (NaN: from the start).
    toMs*: float64 = NaN         ## Points after this are not checked (NaN: to the end).

  TrackSummary* = object
    ## One signal over its measured points between `fromMs` and `toMs`.
    points*: int                               ## Points with a fundamental.
    meanAbsPhaseDeg*, maxAbsPhaseDeg*: float64 ## Of `|phase|`; NaN without points.
    meanThdPct*, maxThdPct*: float64           ## Of THD where measured.
    meanForce*, minForce*, maxForce*: float64  ## Of the fundamental's amplitude.
    maxPeak*: float64                          ## The largest peak.
    exceeded*: array[LimitKind, int]           ## Points beyond each per-point limit (0 for the mean kinds).
    failed*: set[LimitKind]                    ## The limits exceeded at a point or by a mean.

  UnitSignature*[T: SampleType] = object
    ## The measures of one unit; see the module documentation. Filled by
    ## `analyzeVibrators`, which sizes it on its first call and refills it
    ## after.
    signals*: set[VibSignal] ## The signals the unit has (its roles with a row, and the generated force).
    reference*: SweepReference ## The unit's prepared reference.
    tracks*: array[VibSignal, SweepTrack] ## Each signal's measures (all NaN for one it has not).
    units*: array[VibSignal, AmplitudeUnit] ## Each signal's unit.
    summaries*: array[VibSignal, TrackSummary] ## Each signal's summary.
    flags*: array[VibSignal, seq[set[LimitKind]]] ## Per point, the limits exceeded (force signals).
    weightedSum*: seq[T] ## The generated ground force's samples (empty without one).
    planSamples: int # the samples of the plan `reference` and `tracks` are sized for

const
  defaultOrder* = [vsReference, vsGroundForce, vsMass, vsBaseplate]
    ## The usual channel order of a unit on a wireline record.
  pointLimits = {lkPhase, lkThd, lkHarmonic, lkForceLow, lkForceHigh, lkPeakForce}

proc badVib(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc badUnit(u: int, msg: string) {.noinline, noreturn.} =
  badVib("unit " & $u & ": " & msg)

proc initVibUnit*(reference: int, groundForce = -1, mass = -1, baseplate = -1,
    reactionMass = Mass(), baseplateMass = Mass()): VibUnit =
  ## A unit with these rows (−1: none) and masses; units and sensitivities
  ## left as their defaults (rows as recorded, no sensitivity).
  result = VibUnit(reactionMass: reactionMass, baseplateMass: baseplateMass)
  result.rows = [vsReference: reference, vsGroundForce: groundForce, vsMass: mass,
    vsBaseplate: baseplate]

proc vibUnitsInOrder*(count: int, firstRow = 0, order: openArray[VibSignal] = defaultOrder,
    reactionMass = Mass(), baseplateMass = Mass()): seq[VibUnit] =
  ## `count` units whose channels follow each other in `order` from
  ## `firstRow` (by default Ref, GF, M, B: unit `u`'s reference is row
  ## `firstRow + 4u`). Roles not in `order` have no row. Raises `ValueError`
  ## for a negative `count` or `firstRow`, an `order` without the reference,
  ## with a role twice or with `vsWeightedSum` (which is no row).
  if count < 0 or firstRow < 0:
    badVib("count and firstRow must be ≥ 0, got " & $count & " and " & $firstRow)
  var seen: set[VibSignal]
  for r in order:
    if r == vsWeightedSum: badVib("the generated ground force has no row of its own")
    if r in seen: badVib("the role " & $r & " appears twice in the order")
    seen.incl r
  if vsReference notin seen: badVib("the order must hold the reference")
  for u in 0 ..< count:
    var unit = VibUnit(reactionMass: reactionMass, baseplateMass: baseplateMass)
    for i, r in order: unit.rows[r] = firstRow + u * order.len + i
    result.add unit

# ------------------------------------------------------------ ground force --

proc weightedSumKernel[T](m, b: openArray[T], km, kb: float64, dst: var openArray[T]) {.
    raises: [].} =
  ## `dst[j] = km·m[j] + kb·b[j]` (lengths checked by the caller).
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0 ..< dst.len: dst[j] = T(km * float64(m[j]) + kb * float64(b[j]))
  {.pop.}

proc groundForceInto*[T: SampleType](mass, baseplate: openArray[T], reactionMass,
    baseplateMass: Mass, dst: var openArray[T]) {.raises: [].} =
  ## The ground force `Mr·a_M + Mb·a_B` from the reaction-mass and baseplate
  ## accelerations into `dst` (all three of one length: an `AssertionDefect`
  ## otherwise), the masses in kilograms: accelerations in m/s² give N. A
  ## mass that is not set makes it NaN.
  doAssert mass.len == dst.len and baseplate.len == dst.len,
    "mass, baseplate and dst need the same length"
  weightedSumKernel(mass, baseplate, reactionMass.kilograms, baseplateMass.kilograms, dst)

# ------------------------------------------------------------------ limits --

func inRange(l: VibLimits, tMs: float64): bool {.inline.} =
  not (tMs < l.fromMs) and not (tMs > l.toMs) # a NaN bound is no bound

func exceeds(t: SweepTrack, l: VibLimits, m: int): set[LimitKind] {.raises: [].} =
  ## The per-point limits point `m` exceeds (none where not measured).
  let a1 = t.amplitude[0][m]
  if a1.isNaN or not l.inRange(t.timeMs[m]): return
  if abs(t.phaseDeg[m]) > l.phaseDeg: result.incl lkPhase
  if t.thdPct[m] > l.thdPct: result.incl lkThd
  if l.harmonicPct == l.harmonicPct and a1 > 0:
    for k in 1 ..< t.harmonics:
      if 100 * t.amplitude[k][m] / a1 > l.harmonicPct: result.incl lkHarmonic
  if a1 < l.forceLow: result.incl lkForceLow
  if a1 > l.forceHigh: result.incl lkForceHigh
  if t.peak[m] > l.peakForce: result.incl lkPeakForce

proc limitFlagsInto*(track: SweepTrack, limits: VibLimits,
    flags: var openArray[set[LimitKind]]) {.raises: [].} =
  ## Each point's per-point limits exceeded (module documentation): empty
  ## where the point is not measured, outside `fromMs .. toMs`, or within
  ## every limit. `flags` has one entry per point (an `AssertionDefect`
  ## otherwise).
  doAssert flags.len == track.len, "flags needs one entry per point"
  for m in 0 ..< track.len: flags[m] = track.exceeds(limits, m)

func summarize*(track: SweepTrack, limits = VibLimits()): TrackSummary {.raises: [].} =
  ## The statistics of `track` over its measured points between
  ## `limits.fromMs` and `limits.toMs`, the points beyond each per-point
  ## limit and the limits failed (module documentation).
  var sumPhase, sumThd, sumForce = 0.0
  var nPhase, nThd = 0
  result.maxAbsPhaseDeg = NaN
  result.maxThdPct = NaN
  result.minForce = NaN
  result.maxForce = NaN
  result.maxPeak = NaN
  for m in 0 ..< track.len:
    let a1 = track.amplitude[0][m]
    if a1.isNaN or not limits.inRange(track.timeMs[m]): continue
    inc result.points
    sumForce += a1
    if not (a1 >= result.minForce): result.minForce = a1
    if not (a1 <= result.maxForce): result.maxForce = a1
    let p = abs(track.phaseDeg[m])
    if not p.isNaN:
      sumPhase += p
      inc nPhase
      if not (p <= result.maxAbsPhaseDeg): result.maxAbsPhaseDeg = p
    let d = track.thdPct[m]
    if not d.isNaN:
      sumThd += d
      inc nThd
      if not (d <= result.maxThdPct): result.maxThdPct = d
    let pk = track.peak[m]
    if not pk.isNaN and not (pk <= result.maxPeak): result.maxPeak = pk
    for k in track.exceeds(limits, m): inc result.exceeded[k]
  result.meanForce = (if result.points > 0: sumForce / float64(result.points) else: NaN)
  result.meanAbsPhaseDeg = (if nPhase > 0: sumPhase / float64(nPhase) else: NaN)
  result.meanThdPct = (if nThd > 0: sumThd / float64(nThd) else: NaN)
  for k in pointLimits:
    if result.exceeded[k] > 0: result.failed.incl k
  if result.meanAbsPhaseDeg > limits.meanPhaseDeg: result.failed.incl lkMeanPhase
  if result.meanThdPct > limits.meanThdPct: result.failed.incl lkMeanThd
  if result.meanForce < limits.meanForceLow: result.failed.incl lkMeanForce

# --------------------------------------------------------------- analysis --

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func vibratorWorkspace*[T](tp: SweepTrackPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `analyzeVibrators` needs: the plan's and one row.
  result = tp.workspaceSize
  when T is float32: result.f32 += tp.samples
  else: result.f64 += tp.samples

proc fillNaN(t: var SweepTrack) {.raises: [].} =
  for s in [addr t.referenceHz, addr t.frequencyHz, addr t.phaseDeg, addr t.thdPct, addr t.peak]:
    for v in s[].mitems: v = NaN
  for a in t.amplitude.mitems:
    for v in a.mitems: v = NaN

proc checkUnits[T](tp: SweepTrackPlan[T], b: RowBlock[T], meta: ChunkMeta,
    units: openArray[VibUnit]) =
  meta.checkShape(b)
  if not sameInterval(meta.dtMs, tp.dtMs):
    badVib("the rows are at " & $meta.dtMs & " ms, the plan at " & $tp.dtMs & " ms")
  if meta.firstSample != 0:
    badVib("vibrator QC needs whole records from sample 0, got firstSample " &
      $meta.firstSample)
  if b.cols > tp.samples:
    badVib("the rows are " & $b.cols & " samples, the plan takes " & $tp.samples)
  for u, unit in units:
    if unit.rows[vsReference] < 0: badUnit(u, "a reference row is required")
    for role in VibRole:
      let r = unit.rows[role]
      if r < -1 or r >= b.len:
        badUnit(u, "the " & $role & " row " & $r & " is outside 0 ..< " & $b.len)
      let s = unit.sensitivity[role]
      if s.isSet:
        if role == vsReference: badUnit(u, "the reference takes no sensitivity")
        if unit.units[role] != auVolt:
          badUnit(u, "a sensitivity takes a row in volts on to its quantity: the " &
            $role & " row is in " & $unit.units[role] & " (set units[" & $role & "] = auVolt)")
        let want = (if role == vsGroundForce: auForce else: auAcceleration)
        if s.unit != want:
          badUnit(u, "the " & $role & " sensitivity is per " & $s.unit & ", not " & $want)
    if unit.reactionMass.isSet != unit.baseplateMass.isSet:
      badUnit(u, "the generated ground force needs both masses (or neither)")

proc analyzeVibrators*[T: SampleType](tp: SweepTrackPlan[T], b: RowBlock[T],
    meta: ChunkMeta, units: openArray[VibUnit], limits: VibLimits,
    results: var seq[UnitSignature[T]], ws: var Workspace) =
  ## Every measure of every unit of the record `b` (module documentation)
  ## into `results`, one per unit (resized; sized for the plan on the first
  ## call, refilled without allocation after). `ws` holds
  ## `vibratorWorkspace(tp)`. Raises `ValueError`, before any result
  ## changes, for a block that does not match the meta, another interval, a
  ## window (`firstSample > 0`), rows wider than the plan, a unit without a
  ## reference row, a row outside the block, a sensitivity on the reference,
  ## on a row not in volts or of another quantity, or one mass without the
  ## other.
  tp.checkUnits(b, meta, units)
  doAssert ws.fits(vibratorWorkspace(tp)), "reserve vibratorWorkspace(tp) first"
  if results.len != units.len: results.setLen(units.len)
  let cols = b.cols
  let own = tp.workspaceSize.elems(T)
  for u, unit in units:
    var res = addr results[u]
    # Sized for another plan (a record of another length can have as many
    # points): sized again.
    if res.planSamples != tp.samples or res.tracks[vsReference].len != tp.points or
        res.tracks[vsReference].harmonics != tp.params.harmonics:
      res.reference = initSweepReference(tp)
      for s in VibSignal:
        res.tracks[s] = initSweepTrack(tp)
        res.flags[s] = newSeq[set[LimitKind]](tp.points)
      res.planSamples = tp.samples
    res.signals = {}
    for s in VibSignal:
      res.units[s] = auRecorded
      res.summaries[s] = TrackSummary()
      for f in res.flags[s].mitems: f = {}
      res.tracks[s].fillNaN
    template live(r: int): int = meta.liveSamples(r, cols)
    let refRow = unit.rows[vsReference]
    if meta.isFailed(refRow):
      tp.referenceInto(b.row(refRow, 0), res.reference, ws) # no sweep: every measure NaN
    else:
      tp.referenceInto(b.row(refRow, live(refRow)), res.reference, ws)
    let scaled = ws.scratch(T, cols, own)
    # The roles' rows, each in its unit (a sensitivity: volts to the quantity).
    var factor: array[VibRole, float64]
    for role in VibRole:
      let r = unit.rows[role]
      if r < 0: continue
      res.signals.incl role
      let s = unit.sensitivity[role]
      factor[role] = (if s.isSet: 1 / s.voltsPerUnit else: 1.0)
      res.units[role] = (if s.isSet: s.unit else: unit.units[role])
      if meta.isFailed(r): continue # NaN
      let n = live(r)
      if s.isSet:
        let x = b.rowPtr(r)
        for j in 0 ..< n: scaled[j] = T(float64(x[j]) * factor[role])
        tp.trackInto(scaled.toOpenArray(0, n - 1), res.reference, res.tracks[role], ws)
      else:
        tp.trackInto(b.row(r, n), res.reference, res.tracks[role], ws)
    # The ground force from M and B.
    let rm = unit.rows[vsMass]
    let rb = unit.rows[vsBaseplate]
    if rm >= 0 and rb >= 0 and unit.reactionMass.isSet:
      res.signals.incl vsWeightedSum
      res.units[vsWeightedSum] =
        (if res.units[vsMass] == auAcceleration and res.units[vsBaseplate] == auAcceleration:
          auForce else: auRecorded)
      if res.weightedSum.len != cols: res.weightedSum.setLen(cols)
      if meta.isFailed(rm) or meta.isFailed(rb):
        for v in res.weightedSum.mitems: v = T(NaN)
      else:
        weightedSumKernel(b.row(rm), b.row(rb), unit.reactionMass.kilograms * factor[vsMass],
          unit.baseplateMass.kilograms * factor[vsBaseplate], res.weightedSum)
        tp.trackInto(res.weightedSum.toOpenArray(0, max(live(rm), live(rb)) - 1),
          res.reference, res.tracks[vsWeightedSum], ws)
    else:
      res.weightedSum.setLen(0)
    for s in res.signals:
      let l = (if s in {vsGroundForce, vsWeightedSum}: limits else: VibLimits())
      res.summaries[s] = summarize(res.tracks[s], l)
      if s in {vsGroundForce, vsWeightedSum}:
        limitFlagsInto(res.tracks[s], l, res.flags[s])

proc analyzeVibrators*[T: SampleType](tp: SweepTrackPlan[T], samples: Tensor[T],
    meta: ChunkMeta, units: openArray[VibUnit], limits: VibLimits,
    results: var seq[UnitSignature[T]], ws: var Workspace) =
  ## `analyzeVibrators` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  tp.analyzeVibrators(rowBlock(samples), meta, units, limits, results, ws)
