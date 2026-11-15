## Vibrator QC of a record (task 15.6): the ground force from mass and
## baseplate in newtons (masses in kg and lb); a synthetic wireline record
## of three units, two in the default order and one assigned out of order,
## with its channels in volts and the roles' sensitivities; the measures per
## role, their units, the generated force against the recorded one; limits
## flagging exactly the points a direct check finds; failed rows; refusals;
## no allocation on a second call. Task 15.9, on records generated apart
## from the library (`utils/chirps`): polarity per role (given and
## automatic, M and B as a pair), the judged part of the sweep, rows of any
## status calibrated to volts, M and B in different units.

import std/[assertions, math, strutils]
import parseSEG, processSEG
import utils/[allocs, chirps]

const
  dt = 1.0
  g = 9.80665

let spec = linearSweep(8, 80, 6000, dt, initSweepTaper(300, 300, tpCos2))
let n = spec.sweepLen + 1000
let mr = mass(2450, muKilogram)
let mb = mass(2600, muPound) # 1179.3 kg

proc taperAt(tMs: float64): float64 =
  if tMs < 0 or tMs > 6000: return 0
  result = 1
  if tMs < 300: result = taperWeight(tpCos2, tMs / 300)
  if 6000 - tMs < 300: result = taperWeight(tpCos2, (6000 - tMs) / 300)

proc wave(amp, phaseDeg: float64, delayMs = 0.0, h2 = 0.0): seq[float64] =
  ## `amp·w·sin(φ(t − τ) + θ)` plus a 2nd harmonic of `h2·amp`.
  result = newSeq[float64](n)
  for j in 0 ..< n:
    let t = float64(j) * dt - delayMs
    let w = taperAt(t)
    if w > 0:
      let p = spec.phaseAt(t)
      result[j] = amp * w * (sin(p + degToRad(phaseDeg)) + h2 * sin(2 * p))

block ground_force_from_mass_and_baseplate:
  var a, b = newSeq[float64](100)
  for j in 0 ..< 100:
    a[j] = sin(float64(j) * 0.1) * 30 # m/s²
    b[j] = cos(float64(j) * 0.1) * 12
  var f = newSeq[float64](100)
  groundForceInto(a, b, mr, mb, f)
  for j in 0 ..< 100:
    doAssert abs(f[j] - (2450 * a[j] + 2600 * 0.45359237 * b[j])) < 1e-9 * 1e5
  var f32 = newSeq[float32](100)
  var a32, b32 = newSeq[float32](100)
  for j in 0 ..< 100: (a32[j], b32[j]) = (float32(a[j]), float32(b[j]))
  groundForceInto(a32, b32, mr, mb, f32)
  doAssert abs(float64(f32[7]) - f[7]) < 1e-6 * abs(f[7]) + 1e-2
  groundForceInto(a, b, Mass(), mb, f)
  doAssert f[3].isNaN

block units_in_order:
  let u = vibUnitsInOrder(2)
  doAssert u.len == 2 and u[1].rows == [vsReference: 4, vsGroundForce: 5, vsMass: 6,
    vsBaseplate: 7]
  let v = vibUnitsInOrder(3, firstRow = 1, order = [vsGroundForce, vsReference])
  doAssert v[2].rows[vsReference] == 6 and v[2].rows[vsGroundForce] == 5 and
    v[2].rows[vsMass] == -1
  doAssertRaises(ValueError): discard vibUnitsInOrder(1, order = [vsGroundForce])
  doAssertRaises(ValueError): discard vibUnitsInOrder(1, order = [vsReference, vsMass, vsMass])
  doAssertRaises(ValueError): discard vibUnitsInOrder(-1)
  let w = initVibUnit(3, groundForce = 9, reactionMass = mr, baseplateMass = mb)
  doAssert w.rows[vsReference] == 3 and w.rows[vsBaseplate] == -1 and w.reactionMass.isSet

# A record of three units. The accelerations: M leads the pilot by 10°, B
# lags it by 170° (the plate moves against the mass), at 30 and 25 m/s²
# with a 3 % 2nd harmonic on B; the ground force is their weighted sum. The
# channels are volts: GF at 0.25 V/kN, M at 0.1 V/g, B at 0.05 V/g. Unit 1
# is unit 0 delayed by 2 ms; unit 2 is unit 0 again, its rows assigned in
# another order (B, Ref, M, GF).
let sGf = sensitivity(0.25, suVPerKN)
let sM = sensitivity(0.1, suVPerG)
let sB = sensitivity(0.05, suVPerG)
let aM = wave(30, 10)
let aB = wave(25, -170, h2 = 0.03)
var force = newSeq[float64](n)
groundForceInto(aM, aB, mr, mb, force)
let pilot = wave(1, 0)

proc volts(x: seq[float64], s: Sensitivity): seq[float64] =
  result = newSeq[float64](x.len)
  for j, v in x: result[j] = v * s.voltsPerUnit

var keys = newSeq[TraceKeys](12)
var counts = newSeq[int](12)
for r in 0 ..< 12:
  keys[r] = initTraceKeys()
  keys[r].status = tsAuxiliary
  keys[r].t0Ms = 0
  counts[r] = n
var data = zeros[float64](12, n)
proc put(row: int, x: seq[float64]) =
  for j in 0 ..< n: data[row, j] = x[j]
put(0, pilot)
put(1, volts(force, sGf))
put(2, volts(aM, sM))
put(3, volts(aB, sB))
block:
  # Unit 1: every signal 2 ms late (built from the delayed accelerations).
  let dM = wave(30, 10, 2)
  let dB = wave(25, -170, 2, h2 = 0.03)
  var dF = newSeq[float64](n)
  groundForceInto(dM, dB, mr, mb, dF)
  put(4, wave(1, 0, 2))
  put(5, volts(dF, sGf))
  put(6, volts(dM, sM))
  put(7, volts(dB, sB))
put(8, volts(aB, sB))
put(9, pilot)
put(10, volts(aM, sM))
put(11, volts(force, sGf))

proc wireline(u: VibUnit): VibUnit =
  result = u
  for role in [vsGroundForce, vsMass, vsBaseplate]: result.units[role] = auVolt
  result.sensitivity[vsGroundForce] = sGf
  result.sensitivity[vsMass] = sM
  result.sensitivity[vsBaseplate] = sB

var units = vibUnitsInOrder(2, reactionMass = mr, baseplateMass = mb)
units.add initVibUnit(9, groundForce = 11, mass = 10, baseplate = 8, reactionMass = mr,
  baseplateMass = mb)
for u in units.mitems: u = wireline(u)
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = dt)
let tp = initSweepTrackPlan(float64, TrackParams(), dt, n)
var ws = initWorkspace()
ws.reserve(vibratorWorkspace(tp))

proc flat(t: float64): bool = t > 900 and t < 5100

# The fundamental of the weighted sum: Mr·30∠10° + Mb·25∠−170° (44.0 kN).
let kgB = 2600 * 0.45359237
let fRe = 2450 * 30 * cos(degToRad(10.0)) + kgB * 25 * cos(degToRad(-170.0))
let fIm = 2450 * 30 * sin(degToRad(10.0)) + kgB * 25 * sin(degToRad(-170.0))
let wantF = hypot(fRe, fIm)
let wantPhase = radToDeg(arctan2(fIm, fRe))

block measures_per_role_and_units:
  var results: seq[UnitSignature[float64]]
  tp.analyzeVibrators(data, meta, units, VibLimits(), results, ws)
  doAssert results.len == 3
  let r0 = results[0]
  doAssert r0.signals == {vsReference, vsGroundForce, vsMass, vsBaseplate, vsWeightedSum}
  doAssert r0.units[vsGroundForce] == auForce and r0.units[vsMass] == auAcceleration and
    r0.units[vsBaseplate] == auAcceleration and r0.units[vsWeightedSum] == auForce and
    r0.units[vsReference] == auRecorded
  var seen = 0
  for m in 0 ..< tp.points:
    let t = tp.pointMs(m)
    if not flat(t): continue
    let gf = r0.tracks[vsGroundForce]
    doAssert abs(gf.amplitude[0][m] - wantF) < 1e-4 * wantF, $gf.amplitude[0][m] & " vs " & $wantF
    doAssert abs(gf.phaseDeg[m] - wantPhase) < 0.05
    doAssert abs(r0.tracks[vsMass].amplitude[0][m] - 30) < 3e-3 and
      abs(r0.tracks[vsMass].phaseDeg[m] - 10) < 0.05
    doAssert abs(r0.tracks[vsBaseplate].amplitude[0][m] - 25) < 3e-3 and
      abs(wrapDeg(r0.tracks[vsBaseplate].phaseDeg[m] + 170)) < 0.05 and
      abs(r0.tracks[vsBaseplate].thdPct[m] - 3) < 0.01
    # The generated force is the recorded one (both are the weighted sum).
    let ws0 = r0.tracks[vsWeightedSum]
    doAssert abs(ws0.amplitude[0][m] - gf.amplitude[0][m]) < 1e-9 * wantF and
      abs(ws0.phaseDeg[m] - gf.phaseDeg[m]) < 1e-9
    doAssert abs(r0.tracks[vsReference].phaseDeg[m]) < 1e-3
    # Unit 2 is unit 0 with its rows in another order: the same measures.
    for s in VibSignal:
      doAssert r0.tracks[s].amplitude[0][m] == results[2].tracks[s].amplitude[0][m]
      doAssert r0.tracks[s].phaseDeg[m] == results[2].tracks[s].phaseDeg[m]
    # Unit 1 is unit 0 two ms late, against its own (late) reference: the same.
    doAssert abs(results[1].tracks[vsGroundForce].phaseDeg[m - 0] - wantPhase) < 0.1
    inc seen
  doAssert seen > 300
  for k in 0 ..< n:
    doAssert abs(r0.weightedSum[k] - force[k]) < 1e-9 * 1e5
  doAssert r0.summaries[vsGroundForce].points > 500 and
    abs(r0.summaries[vsGroundForce].maxForce - wantF) < 1e-4 * wantF
  doAssert r0.summaries[vsGroundForce].failed == {}

block limits_flag_exactly:
  # Limits on the ground force: a phase limit unit 1's delay reaches partway
  # along the sweep against unit 0's reference, a THD limit, a force range
  # and a peak, between fromMs and toMs; the flags are exactly what a direct
  # check of each point finds, and the summary counts them.
  var late = units[0]
  late.rows[vsGroundForce] = 5 # unit 1's (late) ground force against unit 0's pilot
  late.rows[vsMass] = 6
  late.rows[vsBaseplate] = 7
  # B's 3 % 2nd harmonic makes the ground force's 2.0 %.
  let limits = VibLimits(phaseDeg: 2, thdPct: 2.5, harmonicPct: 1.9,
    forceLow: 0.95 * wantF, forceHigh: 1.05 * wantF, peakForce: 1.0 * wantF,
    meanPhaseDeg: 5, meanThdPct: 2.5, meanForceLow: 2 * wantF, fromMs: 500, toMs: 5500)
  var results: seq[UnitSignature[float64]]
  tp.analyzeVibrators(data, meta, [late], limits, results, ws)
  let t = results[0].tracks[vsGroundForce]
  let flags = results[0].flags[vsGroundForce]
  var count: array[LimitKind, int]
  var points = 0
  for m in 0 ..< t.len:
    var want: set[LimitKind]
    let a1 = t.amplitude[0][m]
    let time = t.timeMs[m]
    if not a1.isNaN and time >= 500 and time <= 5500:
      inc points
      if abs(t.phaseDeg[m]) > 2: want.incl lkPhase
      if t.thdPct[m] > 2.5: want.incl lkThd
      for k in 2 .. t.harmonics:
        if 100 * t.amplitude[k - 1][m] / a1 > 1.9: want.incl lkHarmonic
      if a1 < 0.95 * wantF: want.incl lkForceLow
      if a1 > 1.05 * wantF: want.incl lkForceHigh
      if t.peak[m] > 1.0 * wantF: want.incl lkPeakForce
    doAssert flags[m] == want, $time & ": " & $flags[m] & " vs " & $want
    for k in want: inc count[k]
  let s = results[0].summaries[vsGroundForce]
  doAssert s.points == points and s.exceeded == count
  doAssert count[lkPhase] > 50 and count[lkPhase] < points # the delay crosses the limit
  doAssert count[lkForceLow] == 0 and count[lkThd] == 0 # the tapers are left out
  # The waveform's peak is 1.0076 × its fundamental (the 2nd harmonic adds to
  # one lobe: the other is 0.994 ×), its 2nd harmonic 2.0 %: every point
  # exceeds those limits.
  doAssert count[lkHarmonic] == points and count[lkPeakForce] == points, $count
  doAssert lkMeanForce in s.failed and lkMeanThd notin s.failed
  doAssert (lkPhase in s.failed) == (count[lkPhase] > 0)
  # The mass and baseplate get summaries without limits.
  doAssert results[0].summaries[vsMass].failed == {} and results[0].flags[vsMass][300] == {}
  # The whole sweep (no part of its start or end left out): the tapers'
  # low force is flagged.
  var open = limits
  open.fromMs = NaN
  open.toMs = NaN
  open.skipStartMs = 0
  open.skipEndMs = 0
  open.peakForce = 0.5 * wantF
  tp.analyzeVibrators(data, meta, [late], open, results, ws)
  let o = results[0].summaries[vsGroundForce]
  doAssert o.exceeded[lkForceLow] > 0 and o.points > s.points
  doAssert o.exceeded[lkPeakForce] > 0 and o.exceeded[lkPeakForce] < o.points

block failed_rows:
  let failed = @[5]
  let m2 = initChunkMeta(rowsOf(keys, counts), dtMs = dt, failed = failed)
  var results: seq[UnitSignature[float64]]
  tp.analyzeVibrators(data, m2, units, VibLimits(), results, ws)
  doAssert results[1].tracks[vsGroundForce].amplitude[0][300].isNaN
  doAssert results[1].summaries[vsGroundForce].points == 0
  doAssert not results[1].tracks[vsMass].amplitude[0][300].isNaN
  let failRef = @[0]
  let m3 = initChunkMeta(rowsOf(keys, counts), dtMs = dt, failed = failRef)
  tp.analyzeVibrators(data, m3, units, VibLimits(), results, ws)
  doAssert results[0].reference.span.len == 0 and
    results[0].tracks[vsGroundForce].amplitude[0][300].isNaN
  let failM = @[2]
  let m4 = initChunkMeta(rowsOf(keys, counts), dtMs = dt, failed = failM)
  tp.analyzeVibrators(data, m4, units, VibLimits(), results, ws)
  doAssert results[0].weightedSum[100].isNaN and vsWeightedSum in results[0].signals

block refusals:
  var results: seq[UnitSignature[float64]]
  template refused(us: openArray[VibUnit]) =
    doAssertRaises(ValueError): tp.analyzeVibrators(data, meta, us, VibLimits(), results, ws)
  var u = units[0]
  u.rows[vsReference] = -1
  refused([u])
  u = units[0]
  u.rows[vsMass] = 12
  refused([u])
  u = units[0]
  u.units[vsMass] = auRecorded # a sensitivity on a row not in volts
  refused([u])
  u = units[0]
  u.sensitivity[vsMass] = sGf # a force sensitivity on an acceleration
  refused([u])
  u = units[0]
  u.sensitivity[vsReference] = sM
  u.units[vsReference] = auVolt
  refused([u])
  u = units[0]
  u.baseplateMass = Mass()
  refused([u])
  let wide = initSweepTrackPlan(float64, TrackParams(), dt, n - 1)
  var ws2 = initWorkspace()
  ws2.reserve(vibratorWorkspace(wide))
  doAssertRaises(ValueError): wide.analyzeVibrators(data, meta, units, VibLimits(), results, ws2)
  let other = initSweepTrackPlan(float64, TrackParams(), 2, n)
  doAssertRaises(ValueError): other.analyzeVibrators(data, meta, units, VibLimits(), results, ws)
  doAssert results.len == 0 # nothing changed

block results_across_record_lengths:
  # One `results` kept across records of other lengths whose plans have as
  # many points (7001 and 7005 samples at a 10-sample step): resized, not an
  # AssertionDefect (the reference was sized for the first plan).
  var results: seq[UnitSignature[float64]]
  tp.analyzeVibrators(data, meta, units, VibLimits(), results, ws)
  let longer = n + 4
  var keys2 = keys
  var counts2 = newSeq[int](12)
  for c in counts2.mitems: c = longer
  var data2 = zeros[float64](12, longer)
  for r in 0 ..< 12:
    for j in 0 ..< n: data2[r, j] = data[r, j]
  let meta2 = initChunkMeta(rowsOf(keys2, counts2), dtMs = dt)
  let tp2 = initSweepTrackPlan(float64, TrackParams(), dt, longer)
  doAssert tp2.points == tp.points
  var ws2 = initWorkspace()
  ws2.reserve(vibratorWorkspace(tp2))
  tp2.analyzeVibrators(data2, meta2, units, VibLimits(), results, ws2)
  doAssert results[0].reference.span.len > 0
  doAssert abs(results[0].tracks[vsGroundForce].phaseDeg[300] - wantPhase) < 0.05
  tp.analyzeVibrators(data, meta, units, VibLimits(), results, ws) # and back
  doAssert abs(results[2].tracks[vsGroundForce].amplitude[0][300] - wantF) < 1e-4 * wantF

block no_allocation_on_a_second_call:
  var results: seq[UnitSignature[float64]]
  tp.analyzeVibrators(data, meta, units, VibLimits(phaseDeg: 3), results, ws)
  let count = allocsIn:
    tp.analyzeVibrators(data, meta, units, VibLimits(phaseDeg: 3), results, ws)
  doAssert count == 0, $count & " allocations"

# --------------------------------------------------------------- 15.9 --

# A record generated apart from the library: the pilot cos φ (8-80 Hz, 6 s,
# cos² tapers of 300 ms, 1 ms, a 1 s listen time), and per unit a ground
# force and the accelerations of its mass and baseplate, in N and m/s².
let chirp15 = Chirp(law: clLinear, f0: 8, f1: 80, lengthMs: 6000, startMs: 300, endMs: 300)
let n15 = 7001

proc record(rows: openArray[seq[float64]], status = tsAuxiliary): (Tensor[float64], ChunkMeta) =
  var keys = newSeq[TraceKeys](rows.len)
  var counts = newSeq[int](rows.len)
  var data = zeros[float64](rows.len, n15)
  for r in 0 ..< rows.len:
    keys[r] = initTraceKeys()
    keys[r].status = status
    keys[r].t0Ms = 0
    counts[r] = n15
    for j in 0 ..< n15: data[r, j] = rows[r][j]
  (data, initChunkMeta(rowsOf(keys, counts), dtMs = 1))

proc tone(amp, phaseDeg: float64, h2 = 0.0): seq[float64] =
  chirp(chirp15, 1, n15, [Tone(amp: amp, phaseDeg: phaseDeg),
    Tone(order: 2, amp: h2 * amp)])

proc negated(x: seq[float64]): seq[float64] =
  result = x
  for v in result.mitems: v = -v

let tp15 = initSweepTrackPlan(float64, TrackParams(), 1, n15)
var ws15 = initWorkspace()
ws15.reserve(vibratorWorkspace(tp15))

proc inFlat(t: float64): bool = t > 900 and t < 5100

proc physical(u: VibUnit): VibUnit =
  ## The unit's rows in N and m/s² as passed.
  result = u
  result.units[vsGroundForce] = auForce
  result.units[vsMass] = auAcceleration
  result.units[vsBaseplate] = auAcceleration

block polarity_given_and_automatic:
  # The ground force at 180° + 3° to the pilot (the system's convention),
  # the accelerometers both wired the other way (M at +10°, B at −170°
  # reversed), and a second ground force at +100° (a fault, not wiring).
  let aM = tone(30, 10)
  let aB = tone(25, -170, h2 = 0.03)
  var gf = newSeq[float64](n15)
  groundForceInto(aM, aB, mr, mb, gf)
  let (data, meta) = record([tone(1, 0), tone(44e3, 183), aM.negated, aB.negated,
    tone(44e3, 100), tone(1, 0).negated, tone(44e3, 3)])
  var auto = initVibUnit(0, groundForce = 1, mass = 2, baseplate = 3, reactionMass = mr,
    baseplateMass = mb).physical
  auto.polarity = [vsReference: poAsRecorded, vsGroundForce: poAutomatic,
    vsMass: poAutomatic, vsBaseplate: poAutomatic]
  var given = auto
  given.polarity = [vsReference: poAsRecorded, vsGroundForce: poReversed,
    vsMass: poReversed, vsBaseplate: poReversed]
  var recorded = auto
  recorded.polarity = [poAsRecorded, poAsRecorded, poAsRecorded, poAsRecorded]
  var fault = initVibUnit(0, groundForce = 4).physical
  fault.polarity[vsGroundForce] = poAutomatic
  var refReversed = initVibUnit(5, groundForce = 6).physical # the pilot wired the other way
  refReversed.polarity[vsReference] = poReversed
  let limits = VibLimits(phaseDeg: 5, meanPhaseDeg: 5)
  var results: seq[UnitSignature[float64]]
  tp15.analyzeVibrators(data, meta, [auto, given, recorded, fault, refReversed], limits,
    results, ws15)
  # The weighted sum of the unreversed pair: Mr·30∠10° + Mb·25∠−170°.
  let want = radToDeg(arctan2(2450 * 30 * sin(degToRad(10.0)) + kgB * 25 * sin(degToRad(-170.0)),
    2450 * 30 * cos(degToRad(10.0)) + kgB * 25 * cos(degToRad(-170.0))))
  for u in 0 .. 1:
    let r = results[u]
    doAssert r.reversed == {vsGroundForce, vsMass, vsBaseplate, vsWeightedSum}, $r.reversed
    for s in [vsGroundForce, vsMass, vsBaseplate, vsWeightedSum]:
      doAssert r.summaries[s].reversed
    doAssert not r.summaries[vsReference].reversed
    # Reversed M and B give the weighted sum of the unreversed pair, exactly.
    for j in 0 ..< n15: doAssert r.weightedSum[j] == gf[j], $u & " at " & $j
    var seen = 0
    for m in 0 ..< tp15.points:
      if not inFlat(tp15.pointMs(m)): continue
      let p = r.tracks[vsGroundForce].phaseDeg[m]
      doAssert abs(p - 3) < 0.05, $p
      doAssert abs(r.tracks[vsMass].phaseDeg[m] - 10) < 0.05 and
        abs(wrapDeg(r.tracks[vsBaseplate].phaseDeg[m] + 170)) < 0.05 # B not turned to M
      doAssert abs(r.tracks[vsWeightedSum].phaseDeg[m] - want) < 0.05
      inc seen
    doAssert seen > 300
    # Limits and summaries see the corrected phase.
    doAssert r.summaries[vsGroundForce].failed == {} and
      abs(r.summaries[vsGroundForce].meanPhaseDeg - 3) < 0.05 and
      abs(r.summaries[vsGroundForce].meanAbsPhaseDeg - 3) < 0.05
  # The given and the automatic polarities measure alike.
  for m in 0 ..< tp15.points:
    for s in VibSignal:
      let a = results[0].tracks[s].phaseDeg[m]
      let b = results[1].tracks[s].phaseDeg[m]
      doAssert (a.isNaN and b.isNaN) or abs(wrapDeg(a - b)) < 1e-9
      doAssert results[0].tracks[s].amplitude[0][m] == results[1].tracks[s].amplitude[0][m] or
        results[0].tracks[s].amplitude[0][m].isNaN
  # As recorded: the ground force reads −177° and fails the limits.
  let r2 = results[2]
  doAssert r2.reversed == {} and lkPhase in r2.summaries[vsGroundForce].failed and
    abs(r2.summaries[vsGroundForce].meanPhaseDeg + 177) < 0.05
  for j in 0 ..< n15: doAssert r2.weightedSum[j] == -gf[j]
  # +100° is beyond ±90°: reversed to −80°, and the summary says so.
  let r3 = results[3]
  doAssert r3.reversed == {vsGroundForce} and r3.summaries[vsGroundForce].reversed and
    abs(r3.summaries[vsGroundForce].meanPhaseDeg + 80) < 0.05 and
    {lkPhase, lkMeanPhase} <= r3.summaries[vsGroundForce].failed
  # A reversed pilot, given: its ground force reads +3°, not −177°.
  let r4 = results[4]
  doAssert r4.reversed == {vsReference} and
    abs(r4.summaries[vsGroundForce].meanPhaseDeg - 3) < 0.05 and
    abs(r4.summaries[vsReference].meanPhaseDeg) < 1e-3
  # A track alone: `reversePolarity` is the track of the reversed signal.
  var t1 = initSweepTrack(tp15)
  var t2 = initSweepTrack(tp15)
  var reference = initSweepReference(tp15)
  tp15.referenceInto(tone(1, 0), reference, ws15)
  let x = tone(2, 183, h2 = 0.05)
  tp15.trackInto(x, reference, t1, ws15)
  tp15.trackInto(x.negated, reference, t2, ws15)
  t1.reversePolarity
  for m in 0 ..< t1.len:
    doAssert (t1.phaseDeg[m].isNaN and t2.phaseDeg[m].isNaN) or
      abs(wrapDeg(t1.phaseDeg[m] - t2.phaseDeg[m])) < 1e-9
    for k in 0 ..< t1.harmonics:
      doAssert t1.amplitude[k][m] == t2.amplitude[k][m] or t1.amplitude[k][m].isNaN
    doAssert t1.peak[m] == t2.peak[m] or t1.peak[m].isNaN
    doAssert t1.thdPct[m] == t2.thdPct[m] or t1.thdPct[m].isNaN
  doAssert not t1.automaticPolarity # now at +3°
  t1.reversePolarity
  doAssert t1.automaticPolarity and abs(summarize(t1).meanPhaseDeg - 3) < 0.05
  # No allocation on a second call, automatic polarities included.
  let count = allocsIn:
    tp15.analyzeVibrators(data, meta, [auto, given, recorded, fault, refReversed], limits,
      results, ws15)
  doAssert count == 0, $count & " allocations"

block polarity_refusals:
  let (data, meta) = record([tone(1, 0), tone(44e3, 3), tone(30, 10), tone(25, -170)])
  var results: seq[UnitSignature[float64]]
  template refused(u: VibUnit) =
    doAssertRaises(ValueError): tp15.analyzeVibrators(data, meta, [u], VibLimits(), results, ws15)
  let base = initVibUnit(0, groundForce = 1, mass = 2, baseplate = 3, reactionMass = mr,
    baseplateMass = mb).physical
  var u = base
  u.polarity[vsReference] = poAutomatic # the reference is the phase origin
  refused(u)
  u = base
  u.polarity[vsMass] = poAutomatic # M decided without B
  refused(u)
  u = base
  u.polarity[vsBaseplate] = poAutomatic
  u.polarity[vsMass] = poReversed
  refused(u)
  u = initVibUnit(0, groundForce = 1, mass = 2, baseplate = 3).physical # no masses
  u.polarity[vsMass] = poAutomatic
  u.polarity[vsBaseplate] = poAutomatic
  refused(u)
  u = initVibUnit(0, mass = 2, reactionMass = mr, baseplateMass = mb).physical # no B
  u.polarity[vsMass] = poAutomatic
  refused(u)
  u = base
  u.units[vsBaseplate] = auRecorded # M in m/s², B as recorded: no sum of the two
  refused(u)
  u = base
  u.units[vsMass] = auVolt
  u.sensitivity[vsMass] = sensitivity(0.1, suVPerG) # m/s² by its sensitivity: one unit
  tp15.analyzeVibrators(data, meta, [u], VibLimits(), results, ws15)
  doAssert results[0].units[vsWeightedSum] == auForce
  for v in [-1.0, -Inf, Inf]:
    doAssertRaises(ValueError):
      tp15.analyzeVibrators(data, meta, [base], VibLimits(skipStartMs: v), results, ws15)
    doAssertRaises(ValueError):
      tp15.analyzeVibrators(data, meta, [base], VibLimits(skipEndMs: v), results, ws15)
  # Automatic polarity on the ground force needs nothing else, and a role
  # without a row takes any polarity.
  u = initVibUnit(0, groundForce = 1).physical
  u.polarity = [poAsRecorded, poAutomatic, poAutomatic, poAutomatic]
  tp15.analyzeVibrators(data, meta, [u], VibLimits(), results, ws15)
  doAssert results[0].reversed == {}

block judged_part:
  # The tapers (300 ms, cos²) are left out by default, from the pilot's
  # envelope: the judged part starts within a period of 300 ms and ends
  # within one of 5700 ms; the measures there are returned and flagged; the
  # summaries are over the judged points alone. Lengths in ms instead, from
  # the sweep's ends, and fromMs/toMs within them. An untapered sweep is
  # judged whole.
  let (data, meta) = record([tone(1, 0), tone(44e3, 2, h2 = 0.02)])
  let unit = initVibUnit(0, groundForce = 1).physical
  var results: seq[UnitSignature[float64]]
  tp15.analyzeVibrators(data, meta, [unit], VibLimits(), results, ws15)
  let r = results[0]
  let t = r.tracks[vsGroundForce]
  var first, last = -1
  var outside = 0
  for m in 0 ..< tp15.points:
    if r.judged[m]:
      if first < 0: first = m
      last = m
    elif not t.amplitude[0][m].isNaN:
      doAssert not t.phaseDeg[m].isNaN and not t.thdPct[m].isNaN and not t.peak[m].isNaN
      inc outside # measured, finite, flagged outside
  doAssert abs(tp15.pointMs(first) - 300) <= 1000 / chirp15.freqAt(300) and
    abs(tp15.pointMs(last) - 5700) <= 1000 / chirp15.freqAt(5700),
    $tp15.pointMs(first) & " .. " & $tp15.pointMs(last)
  doAssert outside > 20
  var judged = newSeq[bool](t.len)
  t.judgedInto(VibLimits(), judged)
  doAssert judged == r.judged
  # The summary against a direct one over the judged points.
  var pts, nThd = 0
  var sumF, sumP, sumT, maxP, maxT, maxPk, re, im = 0.0
  var minF = Inf
  var maxF = 0.0
  for m in 0 ..< t.len:
    let a1 = t.amplitude[0][m]
    if not judged[m] or a1.isNaN: continue
    inc pts
    sumF += a1
    minF = min(minF, a1)
    maxF = max(maxF, a1)
    sumP += abs(t.phaseDeg[m])
    maxP = max(maxP, abs(t.phaseDeg[m]))
    re += a1 * cos(degToRad(t.phaseDeg[m]))
    im += a1 * sin(degToRad(t.phaseDeg[m]))
    if not t.thdPct[m].isNaN:
      inc nThd
      sumT += t.thdPct[m]
      maxT = max(maxT, t.thdPct[m])
    maxPk = max(maxPk, t.peak[m])
  let s = r.summaries[vsGroundForce]
  doAssert s.points == pts and abs(s.meanForce - sumF / float64(pts)) < 1e-9 * maxF and
    s.minForce == minF and s.maxForce == maxF and
    abs(s.meanAbsPhaseDeg - sumP / float64(pts)) < 1e-12 and s.maxAbsPhaseDeg == maxP and
    abs(s.meanThdPct - sumT / float64(nThd)) < 1e-12 and s.maxThdPct == maxT and
    s.maxPeak == maxPk and abs(s.meanPhaseDeg - radToDeg(arctan2(im, re))) < 1e-9
  doAssert abs(s.meanPhaseDeg - 2) < 0.05 and abs(s.meanThdPct - 2) < 0.01
  # Lengths in ms from the sweep's ends, and fromMs/toMs within them.
  let e = t.extent
  for (l, a, b) in [(VibLimits(skipStartMs: 500, skipEndMs: 1000), e.startMs + 500, e.endMs - 1000),
      (VibLimits(skipStartMs: 0, skipEndMs: 0), e.startMs, e.endMs),
      (VibLimits(fromMs: 2000, toMs: 2500), 2000.0, 2500.0),
      (VibLimits(skipStartMs: 500, fromMs: 100, toMs: 3000), e.startMs + 500, 3000.0)]:
    let span = e.judgedPart(l)
    doAssert span.a == a and span.b == b, $span
    tp15.analyzeVibrators(data, meta, [unit], l, results, ws15)
    for m in 0 ..< tp15.points:
      doAssert results[0].judged[m] == (tp15.pointMs(m) >= a and tp15.pointMs(m) <= b)
  doAssert e.judgedPart(VibLimits(skipStartMs: -5)).a == e.startMs # a negative length is 0
  doAssert SweepExtent().judgedPart.a.isNaN
  # An untapered sweep: every measured point is judged.
  let bare = Chirp(law: clLinear, f0: 8, f1: 80, lengthMs: 6000, ramp: rmNone)
  let (d2, m2) = record([chirp(bare, 1, n15), chirp(bare, 1, n15, [Tone(amp: 44e3, phaseDeg: 2)])])
  tp15.analyzeVibrators(d2, m2, [unit], VibLimits(), results, ws15)
  var measured = 0
  for m in 0 ..< tp15.points:
    if not results[0].tracks[vsGroundForce].amplitude[0][m].isNaN:
      doAssert results[0].judged[m], $tp15.pointMs(m)
      inc measured
  doAssert measured > 500

block roles_whatever_the_status:
  # Wireline channels "may not be labeled as aux": a record in counts whose
  # channels are marked auxiliary, live, unknown or dead, calibrated with a
  # seismic catch-all rule (a descale and a geophone's sensitivity). The
  # documented route: the vibrator record to volts (target ctVolts), every
  # row whatever its status, then the roles' sensitivities on to m/s² and
  # N: the same measures for every marking. To physical units, the rows not
  # marked auxiliary get the geophone's m/s, and the roles' sensitivities
  # are refused, the message naming the route.
  let lsb = descale(2, duMicroVPerCount)
  let sGf15 = sensitivity(0.25, suVPerKN)
  let sM15 = sensitivity(0.1, suVPerG)
  let sB15 = sensitivity(0.05, suVPerG)
  let aM = tone(30, 10)
  let aB = tone(25, -170, h2 = 0.03)
  var gf = newSeq[float64](n15)
  groundForceInto(aM, aB, mr, mb, gf)
  proc counts(x: seq[float64], voltsPerUnit: float64): seq[float64] =
    result = newSeq[float64](x.len)
    for j, v in x: result[j] = v * voltsPerUnit / lsb.voltsPerCount
  let rows = [counts(tone(1, 0), 1), counts(gf, sGf15.voltsPerUnit),
    counts(aM, sM15.voltsPerUnit), counts(aB, sB15.voltsPerUnit)]
  let seismic = initCalibration([CalibrationRule(descale: lsb,
    sensitivity: sensitivity(28.8, suVPerMPerS))])
  var toVolts = seismic
  toVolts.target = ctVolts
  var unit = vibUnitsInOrder(1, reactionMass = mr, baseplateMass = mb)[0]
  unit.sensitivity = [vsReference: Sensitivity(), vsGroundForce: sGf15, vsMass: sM15,
    vsBaseplate: sB15]
  var first: seq[UnitSignature[float64]]
  for status in [tsAuxiliary, tsLive, tsUnknown, tsDead]:
    var (data, meta) = record(rows, status)
    let plan = initCalibrationPlan(toVolts)
    var cal = newSeq[RowCalibration](4)
    discard plan.resolveInto(meta, cal)
    plan.calibrate(data, meta)
    var u = unit
    for role in VibRole:
      doAssert cal[u.rows[role]].unit == auVolt, $status
      u.units[role] = cal[u.rows[role]].unit
    var results: seq[UnitSignature[float64]]
    tp15.analyzeVibrators(data, meta, [u], VibLimits(), results, ws15)
    let r = results[0]
    doAssert r.units[vsGroundForce] == auForce and r.units[vsMass] == auAcceleration and
      r.units[vsWeightedSum] == auForce
    if first.len == 0:
      first = results
      for m in 0 ..< tp15.points:
        if inFlat(tp15.pointMs(m)):
          doAssert abs(r.tracks[vsMass].amplitude[0][m] - 30) < 3e-3 and
            abs(r.tracks[vsWeightedSum].amplitude[0][m] - r.tracks[vsGroundForce].amplitude[0][m]) <
            1e-6 * r.tracks[vsGroundForce].amplitude[0][m]
    else:
      for s in VibSignal:
        for m in 0 ..< tp15.points:
          let a = r.tracks[s].amplitude[0][m]
          let b = first[0].tracks[s].amplitude[0][m]
          doAssert (a.isNaN and b.isNaN) or a == b, $status & " " & $s
          doAssert r.tracks[s].phaseDeg[m] == first[0].tracks[s].phaseDeg[m] or
            r.tracks[s].phaseDeg[m].isNaN
        doAssert r.summaries[s].points == first[0].summaries[s].points
    if status != tsAuxiliary:
      # To physical units: m/s for these rows, and the roles' sensitivities refused.
      var (d2, m2) = record(rows, status)
      let phys = initCalibrationPlan(seismic)
      var c2 = newSeq[RowCalibration](4)
      discard phys.resolveInto(m2, c2)
      doAssert c2[1].unit == auVelocity
      var v = unit
      for role in VibRole: v.units[role] = c2[v.rows[role]].unit
      try:
        tp15.analyzeVibrators(d2, m2, [v], VibLimits(), results, ws15)
        doAssert false, "a sensitivity on a row in m/s"
      except ValueError as err:
        doAssert "ctVolts" in err.msg, err.msg
