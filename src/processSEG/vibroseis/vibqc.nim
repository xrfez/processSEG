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
## for u in units.mitems:                                  # phase near 0 on any system
##   u.polarity = [poAsRecorded, poAutomatic, poAutomatic, poAutomatic]
## var results: seq[UnitSignature[float32]]
## tp.analyzeVibrators(samples, meta, units, VibLimits(phaseDeg: 5), results, ws)
## # results[0].tracks[vsGroundForce].phaseDeg, .amplitude[0] (fundamental
## # force), .thdPct, .peak; results[0].weightedSum: GF from M and B;
## # results[0].summaries[vsGroundForce].failed, .flags[vsGroundForce];
## # results[0].judged (per point), .reversed (the polarity used)
## ```
##
## **Roles.** `VibUnit.rows` gives each role's row in the block (−1: the unit
## has none); the reference is required. The library never takes a role from
## a row's position, nor from its trace status (live, unknown, dead and
## auxiliary rows are analysed alike): `vibUnitsInOrder` builds the usual
## layout (each unit's channels in the order Ref, GF, M, B, or an order
## given) for the caller to pass, and any other assignment is a `VibUnit` of
## the caller's.
##
## **Ground force from M and B.** `GF = Mr·a_M + Mb·a_B` (the weighted sum, as
## vibrator controllers form it), with the reaction mass `Mr` and the
## baseplate mass `Mb` entered with their unit (`Mass`): accelerations in
## m/s² give newtons, from M and B each in its polarity (below). M and B must
## be in one unit (after their sensitivities): one in m/s² and the other as
## recorded is refused, not summed.
##
## **Polarity** (`VibUnit.polarity`, per role, applied before every measure).
## `poAsRecorded` (the default) or `poReversed` (the row times −1) for any
## role; `poAutomatic` reverses a signal whose mean phase to the reference
## (circular, weighted by the fundamental) over the judged part of the sweep
## is beyond ±90°, so that phase reads near 0 for a healthy unit on a system
## whose ground force is at 180° to its pilot. The ground force is decided on
## its own. **M and B are never decided apart**: the baseplate moves against
## the mass, near 180° from it by physics, and flipping one of them turns the
## weighted sum into a difference; both `poAutomatic` (and both rows and
## masses: the generated force decides) reverse together when the ground
## force generated from them is beyond ±90°. The reference is the phase
## origin: as recorded or reversed, never automatic. An automatic reversal is
## the track's phase turned by 180° (`reversePolarity`; nothing else of a
## signal times −1 changes) and the generated samples negated, with no
## second pass. The polarity used is reported: `UnitSignature.reversed`,
## `TrackSummary.reversed`. Automatic corrects wiring; a ground force at
## +100° by a fault is reversed to −80° too, and still fails its phase limit.
## The decision is doubtful where the mean phase is near ±90° (after it,
## `TrackSummary.meanPhaseDeg` near ±90°): it can go either way from one
## sweep to the next there. Once a system's polarity is known, give it.
##
## **Units.** The rows are taken in the unit the caller says they are in
## (`VibUnit.units`, `auRecorded` by default; for calibrated rows the unit
## from `resolveInto`). A role's `Sensitivity`, when set, takes a row in volts
## on to its quantity: M and B in V/g or V/(m/s²) to m/s², a ground-force
## channel in V/N, V/kN or V/lbf to N. A calibration (`core/calibrate`) treats rows by
## their status: auxiliary rows to volts only, the others by the seismic
## rules (a catch-all geophone rule would make a ground-force channel m/s,
## and its sensitivity here is then refused). Wireline channels may be
## marked either way, so calibrate a vibrator record with the same rules and
## `target = ctVolts`: every row ends in volts whatever its status, and the
## roles' sensitivities take it on. The measures are in the resulting unit
## (`UnitSignature.units`); the generated ground force is in N when both
## accelerations are in m/s², and `auRecorded` otherwise.
##
## **The judged part.** Limits and summaries judge the sweep without a part
## at its start and at its end, where the force ramps and the measures run
## to extremes: by default its tapers (`skipStartMs`, `skipEndMs` NaN), from
## the reference's envelope (`SweepExtent.flatStartMs .. flatEndMs`), or
## lengths in ms from the sweep's first and last sample (0: none), within
## `fromMs .. toMs` (record time, NaN: no bound). The points outside are
## still measured and returned; `judgedInto` (`UnitSignature.judged`) flags
## each point in or out, for a viewer to leave out or dim.
##
## **A trace shorter than its reference.** The reference's sweep sets the
## judged part, and a signal row that ends earlier (a record cut short, or a
## stored reference longer than the traces) is zeros after its last sample:
## those points are measured as a force of 0 and a phase of noise, and fail
## every limit. `toMs` is the way out, and the caller's: pass the trace's own
## end (the last sample's time, less half a window: the window at a point
## reaches half of its length past it) in `VibLimits.toMs` to `analyzeVibrators`,
## `summarize`, `limitFlagsInto`, `judgedInto`, `automaticPolarity` and
## `windowStats`; every one of them judges `judgedPart` and so honours it.
## The library does not guess the trace's end from its samples.
##
## **Windows of the sweep** (`windowStats`): the statistics per 500 ms window
## aligned to the sweep's start, and their average and peak over the sweep, as
## a vibrator controller's QC log gives them (a mean per window, then the mean
## and the worst of the windows, rather than the worst point).
##
## **Limits** (`VibLimits`, every limit NaN, i.e. off, by default). At each
## judged point: `|phase| > phaseDeg`, `THD > thdPct`, any harmonic's
## `100·A_k/A_1 > harmonicPct`, the fundamental below `forceLow` or above
## `forceHigh`, the peak above `peakForce`; over those points, the mean
## `|phase|` above `meanPhaseDeg`, the mean THD above `meanThdPct`, the mean
## fundamental below `meanForceLow`. `limitFlagsInto` gives each point the
## set of limits it exceeds, `summarize` the statistics and the limits
## failed. In `analyzeVibrators` the limits apply to the two ground forces,
## recorded and generated; the other signals are summarized over the same
## judged part without them.
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

  Polarity* = enum
    ## How a role's row is taken (module documentation, "Polarity").
    poAsRecorded = "as recorded" ## As it is.
    poReversed = "reversed"      ## Times −1.
    poAutomatic = "automatic"
      ## Reversed when its mean phase to the reference, over the judged part of
      ## the sweep, is beyond ±90° (GF; M and B as a pair, from the ground
      ## force generated from them; not the reference).

  VibUnit* = object
    ## One vibrator unit's rows and constants; see the module documentation.
    rows*: array[VibRole, int] = [vsReference: -1, vsGroundForce: -1, vsMass: -1,
      vsBaseplate: -1]   ## Each role's row in the block; −1: none.
    units*: array[VibRole, AmplitudeUnit]
      ## The unit each row is in as passed (`auRecorded` by default).
    sensitivity*: array[VibRole, Sensitivity]
      ## Optional: takes a row in volts on to its quantity (M, B: acceleration;
      ## GF: force); not set (`Sensitivity()`) by default.
    polarity*: array[VibRole, Polarity]
      ## How each row is taken (`poAsRecorded` by default), before every measure.
    reactionMass*: Mass ## `Mr`, for the generated ground force.
    baseplateMass*: Mass ## `Mb`, for the generated ground force.

  LimitKind* = enum
    ## A limit a signal exceeds.
    lkPhase = "phase"          ## `|phase| > phaseDeg` at a point.
    lkThd = "THD"              ## `THD > thdPct` at a point.
    lkHarmonic = "harmonic"
      ## A harmonic's `100·A_k/A_1 > harmonicPct` at a point (with a THD band,
      ## only the harmonics inside it there).
    lkForceLow = "force low" ## The fundamental below `forceLow` at a point.
    lkForceHigh = "force high" ## The fundamental above `forceHigh` at a point.
    lkPeakForce = "peak force" ## The peak above `peakForce` at a point.
    lkMeanPhase = "mean phase" ## The mean `|phase|` above `meanPhaseDeg`.
    lkMeanThd = "mean THD" ## The mean THD above `meanThdPct`.
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
    skipStartMs*: float64 = NaN
      ## The part of the sweep's start left out of the judgement, ms from its
      ## first sample (NaN: its start taper, `SweepExtent.flatStartMs`; 0:
      ## none).
    skipEndMs*: float64 = NaN
      ## The part of its end left out, ms back from its last sample (NaN: its
      ## end taper).
    fromMs*: float64 = NaN ## Points before this are not judged either (NaN: no bound).
    toMs*: float64 = NaN ## Points after this are not judged either (NaN: no bound).

  TrackSummary* = object
    ## One signal over its measured points in the judged part of the sweep.
    points*: int                               ## Points with a fundamental.
    phasePoints*, thdPoints*: int
      ## Of those, the points with a phase (a fundamental above 0) and with
      ## a THD (every harmonic it sums measured, by default: not where one
      ## is past the harmonic limit). A limit is judged over these alone: one
      ## whose count is 0 was not judged, which `failed` cannot tell from a
      ## pass.
    meanPhaseDeg*: float64
      ## The phase's mean, circular and weighted by the fundamental (what
      ## `poAutomatic` decides from); NaN without points.
    meanAbsPhaseDeg*, maxAbsPhaseDeg*: float64 ## Of `|phase|`; NaN without points.
    meanThdPct*, maxThdPct*: float64 ## Of THD where measured.
    meanForce*, minForce*, maxForce*: float64 ## Of the fundamental's amplitude.
    maxPeak*: float64 ## The largest peak.
    exceeded*: array[LimitKind, int] ## Points beyond each per-point limit (0 for the mean kinds).
    failed*: set[LimitKind] ## The limits exceeded at a point or by a mean.
    reversed*: bool
      ## The signal was reversed before it was measured (its polarity, given
      ## or automatic: `analyzeVibrators` sets it; `summarize` leaves it false).

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
    judged*: seq[bool] ## Per point: inside the judged part of the sweep (`judgedInto`).
    reversed*: set[VibSignal]
      ## The signals reversed before they were measured (given or automatic;
      ## `vsWeightedSum` when both M and B were).
    weightedSum*: seq[T]
      ## The generated ground force's samples (empty without one), from M and
      ## B in their polarity, negated when they were reversed automatically:
      ## in its polarity already.
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

proc validate*(l: VibLimits) =
  ## Checks the judged part's lengths: `skipStartMs` and `skipEndMs` NaN, or
  ## finite and ≥ 0. Raises `ValueError` (`analyzeVibrators` calls it; the
  ## per-track calls take a negative length as 0). The limit values are the
  ## caller's: any value, NaN for none.
  for (name, v) in [("skipStartMs", l.skipStartMs), ("skipEndMs", l.skipEndMs)]:
    if not (v.isNaN or (v >= 0 and v < Inf)):
      badVib(name & " must be NaN (the taper) or a finite length ≥ 0, got " & $v)

func judgedPart*(e: SweepExtent, limits = VibLimits()): Slice[float64] {.raises: [].} =
  ## The part of the sweep `e` that limits and summaries judge, as the
  ## first and last point time in ms: the sweep without `skipStartMs` at its
  ## start and `skipEndMs` at its end (NaN: its tapers, `e.flatStartMs ..
  ## e.flatEndMs`; a negative length counts as 0), within `fromMs .. toMs`.
  ## `NaN .. NaN` (no point) without a sweep.
  if e.startMs.isNaN: return NaN .. NaN
  let skipA = max(0.0, limits.skipStartMs) # (NaN: the taper)
  let skipB = max(0.0, limits.skipEndMs)
  var a = (if limits.skipStartMs.isNaN: e.flatStartMs else: e.startMs + skipA)
  var b = (if limits.skipEndMs.isNaN: e.flatEndMs else: e.endMs - skipB)
  if limits.fromMs > a: a = limits.fromMs # a NaN bound is no bound
  if limits.toMs < b: b = limits.toMs
  a .. b

proc judgedInto*(track: SweepTrack, limits: VibLimits, dst: var openArray[bool]) {.
    raises: [].} =
  ## Per point, whether its time lies in the judged part of the track's
  ## sweep (`judgedPart(track.extent, limits)`), measured or not: a viewer
  ## leaves the other points out of a plot or draws them dimmed. `dst` has
  ## one entry per point (an `AssertionDefect` otherwise).
  doAssert dst.len == track.len, "dst needs one entry per point"
  let span = track.extent.judgedPart(limits)
  for m in 0 ..< track.len: dst[m] = track.timeMs[m] in span

func exceeds(t: SweepTrack, l: VibLimits, span: Slice[float64], m: int): set[LimitKind] {.
    raises: [].} =
  ## The per-point limits point `m` exceeds (none where not measured or
  ## not judged).
  let a1 = t.amplitude[0][m]
  if a1.isNaN or t.timeMs[m] notin span: return
  if abs(t.phaseDeg[m]) > l.phaseDeg: result.incl lkPhase
  if t.thdPct[m] > l.thdPct: result.incl lkThd
  if l.harmonicPct == l.harmonicPct and a1 > 0:
    for k in 1 ..< t.harmonics:
      # (with a band: the harmonics inside it at this point only, as THD)
      if not t.bandHz.isNaN and float64(k + 1) * t.referenceHz[m] > t.bandHz: break
      if 100 * t.amplitude[k][m] / a1 > l.harmonicPct: result.incl lkHarmonic
  if a1 < l.forceLow: result.incl lkForceLow
  if a1 > l.forceHigh: result.incl lkForceHigh
  if t.peak[m] > l.peakForce: result.incl lkPeakForce

proc limitFlagsInto*(track: SweepTrack, limits: VibLimits,
    flags: var openArray[set[LimitKind]]) {.raises: [].} =
  ## Each point's per-point limits exceeded (module documentation): empty
  ## where the point is not measured, outside the judged part, or within
  ## every limit. `flags` has one entry per point (an `AssertionDefect`
  ## otherwise).
  doAssert flags.len == track.len, "flags needs one entry per point"
  let span = track.extent.judgedPart(limits)
  for m in 0 ..< track.len: flags[m] = track.exceeds(limits, span, m)

func meanPhase(track: SweepTrack, span: Slice[float64]): float64 {.raises: [].} =
  ## The circular mean of the phase over the judged points, weighted by the
  ## fundamental, degrees; NaN without such points.
  var re, im = 0.0
  for m in 0 ..< track.len:
    let a1 = track.amplitude[0][m]
    let p = track.phaseDeg[m]
    if a1.isNaN or p.isNaN or track.timeMs[m] notin span: continue
    re += a1 * cos(degToRad(p))
    im += a1 * sin(degToRad(p))
  if re == 0 and im == 0: NaN else: radToDeg(arctan2(im, re))

func summarize*(track: SweepTrack, limits = VibLimits()): TrackSummary {.raises: [].} =
  ## The statistics of `track` over its measured points in the judged part
  ## of the sweep (`judgedPart`), the points beyond each per-point limit and
  ## the limits failed (module documentation).
  let span = track.extent.judgedPart(limits)
  var sumPhase, sumThd, sumForce = 0.0
  var nPhase, nThd = 0
  result.maxAbsPhaseDeg = NaN
  result.maxThdPct = NaN
  result.minForce = NaN
  result.maxForce = NaN
  result.maxPeak = NaN
  for m in 0 ..< track.len:
    let a1 = track.amplitude[0][m]
    if a1.isNaN or track.timeMs[m] notin span: continue
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
    for k in track.exceeds(limits, span, m): inc result.exceeded[k]
  result.meanPhaseDeg = track.meanPhase(span)
  result.meanForce = (if result.points > 0: sumForce / float64(result.points) else: NaN)
  result.meanAbsPhaseDeg = (if nPhase > 0: sumPhase / float64(nPhase) else: NaN)
  result.meanThdPct = (if nThd > 0: sumThd / float64(nThd) else: NaN)
  result.phasePoints = nPhase
  result.thdPoints = nThd
  for k in pointLimits:
    if result.exceeded[k] > 0: result.failed.incl k
  if result.meanAbsPhaseDeg > limits.meanPhaseDeg: result.failed.incl lkMeanPhase
  if result.meanThdPct > limits.meanThdPct: result.failed.incl lkMeanThd
  if result.meanForce < limits.meanForceLow: result.failed.incl lkMeanForce

# ----------------------------------------------------------- windows --

type
  WindowSummary* = object
    ## A sweep's window statistics (`WindowStats`): the mean of the windows'
    ## means ("average") and the largest window ("peak"), as a vibrator
    ## controller's log gives them, each with the start of the window it is
    ## from. NaN where no window has the measure.
    windows*: int ## The windows kept.
    meanAbsPhaseDeg*: float64 ## Mean of the windows' mean `|phase|`.
    maxAbsPhaseDeg*, maxAbsPhaseAtMs*: float64 ## The largest window mean `|phase|`, and its window's start.
    meanForce*: float64 ## Mean of the windows' mean fundamental.
    minForce*, minForceAtMs*: float64 ## The smallest window mean fundamental, and its window's start.
    maxForce*, maxForceAtMs*: float64 ## The largest window mean fundamental, and its window's start.
    meanThdPct*: float64 ## Mean of the windows' mean THD.
    maxThdPct*, maxThdAtMs*: float64 ## The largest window mean THD, and its window's start.
    maxPeak*, maxPeakAtMs*: float64 ## The largest window peak, and its window's start.

  WindowStats* = object
    ## A track's statistics per window of the sweep (`windowStats`): the
    ## windows kept, one value each in the sequences, and the sweep's
    ## summary of them.
    windowMs*: float64 ## The window's length, ms.
    startMs*: seq[float64] ## Each kept window's start, ms from the rows' first sample.
    points*: seq[int] ## Each window's measured, judged points (those with a fundamental).
    meanAbsPhaseDeg*: seq[float64] ## Mean `|phase|` of the window's points with a phase (NaN: none).
    meanForce*: seq[float64] ## Mean fundamental.
    peak*: seq[float64] ## The largest `SweepTrack.peak` of the window: its largest `|signal|`.
    meanThdPct*: seq[float64] ## Mean THD of the window's points with a THD (NaN: none).
    summary*: WindowSummary ## The sweep's average and peak of the windows.

proc windowStats*(track: SweepTrack, limits = VibLimits(), windowMs = 500.0,
    minFill = 0.5, originMs = NaN): WindowStats =
  ## The statistics of `track` per window of `windowMs` (a vibrator
  ## controller's 500 ms), and the sweep's "average" and "peak" of them
  ## (`WindowSummary`). The windows are aligned to the sweep's start
  ## (`track.extent.startMs`, or `originMs` when given: the true sweep start
  ## or time break, ms from the rows' first sample): window `i` is `start + i·windowMs ..< start +
  ## (i + 1)·windowMs`. Only the measured points inside the judged part
  ## (`judgedPart(track.extent, limits)`: the tapers or `skipStartMs` and
  ## `skipEndMs`, within `fromMs .. toMs`) count, so a range leaves out the
  ## tapers or a zero-padded tail of a trace shorter than its reference.
  ##
  ## **Partial windows.** A window the judged part or the measured points cut
  ## short is averaged over the points it has, and kept when those points
  ## span at least `minFill` of `windowMs` (their count times the points'
  ## interval); a thinner one is left out and counted nowhere. `minFill = 0`
  ## keeps every window with a point, `1` only the complete ones. The average
  ## of the windows is of the kept windows, each as one.
  ##
  ## Empty (no window) for a track without a sweep or without points in the
  ## judged part. Raises `ValueError` for a `windowMs` that is not finite and
  ## above 0, or a `minFill` outside 0 .. 1.
  if not (windowMs > 0 and windowMs < Inf):
    badVib("windowMs must be finite and positive, got " & $windowMs)
  if not (minFill >= 0 and minFill <= 1):
    badVib("minFill must be in 0 .. 1, got " & $minFill)
  result.windowMs = windowMs
  var sum = WindowSummary(meanAbsPhaseDeg: NaN, maxAbsPhaseDeg: NaN, maxAbsPhaseAtMs: NaN,
    meanForce: NaN, minForce: NaN, minForceAtMs: NaN, maxForce: NaN, maxForceAtMs: NaN,
    meanThdPct: NaN, maxThdPct: NaN, maxThdAtMs: NaN, maxPeak: NaN, maxPeakAtMs: NaN)
  result.summary = sum
  let origin = (if originMs.isNaN: track.extent.startMs else: originMs)
  let span = track.extent.judgedPart(limits)
  if origin.isNaN or span.a.isNaN or span.b.isNaN or track.len < 2: return
  let step = track.timeMs[1] - track.timeMs[0]
  if not (step > 0): return
  type Acc = object
    n, nPhase, nThd: int
    phase, force, thd, peak: float64
  let count = int(floor((span.b - origin) / windowMs)) + 1
  if count <= 0: return
  var accs = newSeq[Acc](count)
  for m in 0 ..< track.len:
    let t = track.timeMs[m]
    let a1 = track.amplitude[0][m]
    if a1.isNaN or t notin span or t < origin: continue
    let i = int(floor((t - origin) / windowMs))
    if i >= count: continue
    template acc: untyped = accs[i]
    inc acc.n
    acc.force += a1
    let p = abs(track.phaseDeg[m])
    if not p.isNaN:
      acc.phase += p
      inc acc.nPhase
    let d = track.thdPct[m]
    if not d.isNaN:
      acc.thd += d
      inc acc.nThd
    let pk = track.peak[m]
    if not pk.isNaN and not (pk <= acc.peak): acc.peak = pk
  var nPhase, nThd, nPeak = 0
  var sumPhase, sumForce, sumThd = 0.0
  for i, acc in accs:
    if acc.n == 0 or float64(acc.n) * step < minFill * windowMs * (1 - 1e-9): continue
    let at = origin + float64(i) * windowMs
    let phase = (if acc.nPhase > 0: acc.phase / float64(acc.nPhase) else: NaN)
    let force = acc.force / float64(acc.n)
    let thd = (if acc.nThd > 0: acc.thd / float64(acc.nThd) else: NaN)
    let peak = (if acc.peak > 0: acc.peak else: NaN)
    result.startMs.add at
    result.points.add acc.n
    result.meanAbsPhaseDeg.add phase
    result.meanForce.add force
    result.peak.add peak
    result.meanThdPct.add thd
    inc sum.windows
    sumForce += force
    if not (force >= sum.minForce): (sum.minForce, sum.minForceAtMs) = (force, at)
    if not (force <= sum.maxForce): (sum.maxForce, sum.maxForceAtMs) = (force, at)
    if not phase.isNaN:
      sumPhase += phase
      inc nPhase
      if not (phase <= sum.maxAbsPhaseDeg): (sum.maxAbsPhaseDeg, sum.maxAbsPhaseAtMs) = (phase, at)
    if not thd.isNaN:
      sumThd += thd
      inc nThd
      if not (thd <= sum.maxThdPct): (sum.maxThdPct, sum.maxThdAtMs) = (thd, at)
    if not peak.isNaN:
      inc nPeak
      if not (peak <= sum.maxPeak): (sum.maxPeak, sum.maxPeakAtMs) = (peak, at)
  if sum.windows > 0: sum.meanForce = sumForce / float64(sum.windows)
  if nPhase > 0: sum.meanAbsPhaseDeg = sumPhase / float64(nPhase)
  if nThd > 0: sum.meanThdPct = sumThd / float64(nThd)
  result.summary = sum

proc reversePolarity*(track: var SweepTrack) {.raises: [].} =
  ## The measures of the signal reversed: every phase turned by 180°
  ## (wrapped to (−180, 180]); the amplitudes, THD, peak and frequencies of
  ## a signal times −1 are its own. No second pass over the samples.
  for v in track.phaseDeg.mitems:
    if not v.isNaN: v = wrapDeg(v + 180)

proc automaticPolarity*(track: var SweepTrack, limits = VibLimits()): bool {.raises: [].} =
  ## `poAutomatic` for one track: reverses it (`reversePolarity`) when its
  ## mean phase over the judged part of the sweep (`judgedPart(track.extent,
  ## limits)`; `TrackSummary.meanPhaseDeg`) is beyond ±90°, and says whether
  ## it did; a track without judged points is left as it is.
  result = abs(track.meanPhase(track.extent.judgedPart(limits))) > 90
  if result: track.reversePolarity

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
  for v in t.thdCount.mitems: v = 0
  t.lastFundamentalMs = NaN
  t.bandHz = NaN
  t.fundamentalCut = false
  for a in t.amplitude.mitems:
    for v in a.mitems: v = NaN
  t.extent = SweepExtent()

proc copyTimes[T](b: RowBlock[T], r, n: int, k: float64, dst: ptr UncheckedArray[T]) {.
    raises: [].} =
  ## The first `n` samples of row `r` times `k`, into `dst`.
  let x = b.rowPtr(r)
  for j in 0 ..< n: dst[j] = T(float64(x[j]) * k)

func unitAfter(unit: VibUnit, role: VibRole): AmplitudeUnit {.inline.} =
  ## A role's unit once its sensitivity (if any) is applied.
  if unit.sensitivity[role].isSet: unit.sensitivity[role].unit else: unit.units[role]

proc checkUnits[T](tp: SweepTrackPlan[T], b: RowBlock[T], meta: ChunkMeta,
    units: openArray[VibUnit], limits: VibLimits) =
  meta.checkShape(b)
  if not sameInterval(meta.dtMs, tp.dtMs):
    badVib("the rows are at " & $meta.dtMs & " ms, the plan at " & $tp.dtMs & " ms")
  if meta.firstSample != 0:
    badVib("vibrator QC needs whole records from sample 0, got firstSample " &
      $meta.firstSample)
  if b.cols > tp.samples:
    badVib("the rows are " & $b.cols & " samples, the plan takes " & $tp.samples)
  limits.validate
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
            $role & " row is in " & $unit.units[role] & " (set units[" & $role &
            "] = auVolt; a channel calibrated by seismic rules gets their factors " &
            "whatever its role: calibrate the record with target ctVolts)")
        let want = (if role == vsGroundForce: auForce else: auAcceleration)
        if s.unit != want:
          badUnit(u, "the " & $role & " sensitivity is per " & $s.unit & ", not " & $want)
    if unit.reactionMass.isSet != unit.baseplateMass.isSet:
      badUnit(u, "the generated ground force needs both masses (or neither)")
    if unit.polarity[vsReference] == poAutomatic:
      badUnit(u, "the reference is the phase origin: its polarity is as recorded or " &
        "reversed, not automatic")
    let pair = unit.rows[vsMass] >= 0 and unit.rows[vsBaseplate] >= 0 and
      unit.reactionMass.isSet
    let autoM = unit.rows[vsMass] >= 0 and unit.polarity[vsMass] == poAutomatic
    let autoB = unit.rows[vsBaseplate] >= 0 and unit.polarity[vsBaseplate] == poAutomatic
    if (autoM or autoB) and not pair:
      badUnit(u, "an automatic polarity of M and B is decided from the ground force " &
        "generated from them: it needs both rows and both masses")
    if autoM != autoB:
      badUnit(u, "M and B take one polarity, decided together: both poAutomatic, or " &
        "each poAsRecorded or poReversed")
    if pair and unit.unitAfter(vsMass) != unit.unitAfter(vsBaseplate):
      badUnit(u, "the ground force generated from M and B needs both in one unit: M is " &
        "in " & $unit.unitAfter(vsMass) & ", B in " & $unit.unitAfter(vsBaseplate))

proc analyzeVibrators*[T: SampleType](tp: SweepTrackPlan[T], b: RowBlock[T],
    meta: ChunkMeta, units: openArray[VibUnit], limits: VibLimits,
    results: var seq[UnitSignature[T]], ws: var Workspace) =
  ## Every measure of every unit of the record `b` (module documentation)
  ## into `results`, one per unit (resized; sized for the plan on the first
  ## call, refilled without allocation after). `ws` holds
  ## `vibratorWorkspace(tp)`. Raises `ValueError`, before any result
  ## changes, for a block that does not match the meta, another interval, a
  ## window (`firstSample > 0`), rows wider than the plan, limits `validate`
  ## refuses, a unit without a reference row, a row outside the block, a
  ## sensitivity on the reference, on a row not in volts or of another
  ## quantity, one mass without the other, an automatic polarity on the
  ## reference, on M or B without the generated ground force or on one of
  ## them only, or M and B in different units with both masses.
  tp.checkUnits(b, meta, units, limits)
  doAssert ws.fits(vibratorWorkspace(tp)), "reserve vibratorWorkspace(tp) first"
  if results.len != units.len: results.setLen(units.len)
  let cols = b.cols
  let own = tp.workspaceSize.elems(T)
  # The judged part without limit values, for M, B and the reference.
  let judging = VibLimits(skipStartMs: limits.skipStartMs, skipEndMs: limits.skipEndMs,
    fromMs: limits.fromMs, toMs: limits.toMs)
  for u, unit in units:
    var res = addr results[u]
    # Sized for another plan (a record of another length can have as many
    # points): sized again.
    if res.planSamples != tp.samples or res.tracks[vsReference].len != tp.points or
        res.tracks[vsReference].harmonics != tp.params.trackedHarmonics:
      res.reference = initSweepReference(tp)
      for s in VibSignal:
        res.tracks[s] = initSweepTrack(tp)
        res.flags[s] = newSeq[set[LimitKind]](tp.points)
      res.judged = newSeq[bool](tp.points)
      res.planSamples = tp.samples
    res.signals = {}
    res.reversed = {}
    for s in VibSignal:
      res.units[s] = auRecorded
      res.summaries[s] = TrackSummary()
      for f in res.flags[s].mitems: f = {}
      res.tracks[s].fillNaN
    template live(r: int): int = meta.liveSamples(r, cols)
    let scaled = ws.scratch(T, cols, own)
    # Each role's row times its factor: the sensitivity (volts to the
    # quantity) and the polarity, before every measure.
    var factor: array[VibRole, float64]
    for role in VibRole:
      let s = unit.sensitivity[role]
      factor[role] = (if unit.polarity[role] == poReversed: -1.0 else: 1.0) *
        (if s.isSet: 1 / s.voltsPerUnit else: 1.0)
    let refRow = unit.rows[vsReference]
    if meta.isFailed(refRow):
      tp.referenceInto(b.row(refRow, 0), res.reference, ws) # no sweep: every measure NaN
    elif factor[vsReference] == 1:
      tp.referenceInto(b.row(refRow, live(refRow)), res.reference, ws)
    else:
      let n = live(refRow)
      b.copyTimes(refRow, n, factor[vsReference], scaled)
      tp.referenceInto(scaled.toOpenArray(0, n - 1), res.reference, ws)
    for role in VibRole:
      let r = unit.rows[role]
      if r < 0: continue
      res.signals.incl role
      if unit.polarity[role] == poReversed: res.reversed.incl role
      res.units[role] = unit.unitAfter(role)
      if meta.isFailed(r): continue # NaN
      let n = live(r)
      if factor[role] == 1:
        tp.trackInto(b.row(r, n), res.reference, res.tracks[role], ws)
      else:
        b.copyTimes(r, n, factor[role], scaled)
        tp.trackInto(scaled.toOpenArray(0, n - 1), res.reference, res.tracks[role], ws)
    # The ground force from M and B, each with its factor.
    let rm = unit.rows[vsMass]
    let rb = unit.rows[vsBaseplate]
    if rm >= 0 and rb >= 0 and unit.reactionMass.isSet:
      res.signals.incl vsWeightedSum
      if {vsMass, vsBaseplate} <= res.reversed: res.reversed.incl vsWeightedSum
      res.units[vsWeightedSum] = (if res.units[vsMass] == auAcceleration: auForce else: auRecorded)
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
    # Automatic polarity, from the mean phase over the judged part: the
    # ground force on its own, M and B together from the force they make.
    if unit.rows[vsGroundForce] >= 0 and unit.polarity[vsGroundForce] == poAutomatic:
      if res.tracks[vsGroundForce].automaticPolarity(judging):
        res.reversed.incl vsGroundForce
    if vsWeightedSum in res.signals and unit.polarity[vsMass] == poAutomatic:
      if res.tracks[vsWeightedSum].automaticPolarity(judging):
        res.tracks[vsMass].reversePolarity
        res.tracks[vsBaseplate].reversePolarity
        for v in res.weightedSum.mitems: v = -v
        res.reversed.incl {vsMass, vsBaseplate, vsWeightedSum}
    let span = res.reference.extent.judgedPart(limits)
    for m in 0 ..< tp.points: res.judged[m] = tp.pointMs(m) in span
    for s in res.signals:
      let l = (if s in {vsGroundForce, vsWeightedSum}: limits else: judging)
      res.summaries[s] = summarize(res.tracks[s], l)
      res.summaries[s].reversed = s in res.reversed
      if s in {vsGroundForce, vsWeightedSum}:
        limitFlagsInto(res.tracks[s], l, res.flags[s])

proc analyzeVibrators*[T: SampleType](tp: SweepTrackPlan[T], samples: Tensor[T],
    meta: ChunkMeta, units: openArray[VibUnit], limits: VibLimits,
    results: var seq[UnitSignature[T]], ws: var Workspace) =
  ## `analyzeVibrators` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  tp.analyzeVibrators(rowBlock(samples), meta, units, limits, results, ws)
