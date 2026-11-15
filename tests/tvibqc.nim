## Vibrator QC of a record (task 15.6): the ground force from mass and
## baseplate in newtons (masses in kg and lb); a synthetic wireline record
## of three units, two in the default order and one assigned out of order,
## with its channels in volts and the roles' sensitivities; the measures per
## role, their units, the generated force against the recorded one; limits
## flagging exactly the points a direct check finds; failed rows; refusals;
## no allocation on a second call.

import std/[assertions, math]
import parseSEG, processSEG
import utils/allocs

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
  # The whole sweep: the tapers' low force is flagged.
  var open = limits
  open.fromMs = NaN
  open.toMs = NaN
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
