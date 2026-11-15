## Tracking against sweep time (task 15.3): on synthetic sweeps of several
## laws, with tapers, at 1 and 2 ms, each measure recovers what was put in:
## a phase shift, a delay, harmonics at known levels, the fundamental's
## amplitude (with a DC offset and with noise), the frequency law, the peak
## per period; NaN outside the sweep and above the harmonic limit; float32;
## comparisons; refusals; no allocation. On the example files (skipped when
## absent): a recorded pilot against itself and against its law.

import std/[assertions, math, os, random]
import parseSEG, processSEG
import utils/allocs

type Harmonic = tuple[order: int, amp, phaseDeg: float64]

proc taperAt(spec: SweepSpec, tMs: float64): float64 =
  ## The sweep's taper weight at `tMs` (0 outside the sweep).
  let T = spec.lengthMs
  if tMs < 0 or tMs > T: return 0
  let s = spec.taper.startMs
  let e = spec.taper.endMs
  result = 1
  if s > 0 and tMs < s: result = taperWeight(spec.taper.shape, tMs / s, spec.taper.param)
  if e > 0 and T - tMs < e:
    result = min(result, taperWeight(spec.taper.shape, (T - tMs) / e, spec.taper.param))

proc phaseTable(spec: SweepSpec, dtMs: float64, n: int): seq[float64] =
  ## 2π∫₀ᵗ f at each sample, by Simpson's rule on 64 steps per sample (for
  ## the laws without a closed form).
  result = newSeq[float64](n)
  var acc = 0.0
  const sub = 64
  let h = dtMs / sub
  for j in 1 ..< n:
    let a = float64(j - 1) * dtMs
    var s = spec.frequencyAt(a) + spec.frequencyAt(a + dtMs)
    for i in 1 ..< sub:
      s += (if i mod 2 == 1: 4.0 else: 2.0) * spec.frequencyAt(a + float64(i) * h)
    acc += s * h / 3 / 1000
    result[j] = 2 * PI * acc

proc phiAt(spec: SweepSpec, table: seq[float64], tMs, dtMs: float64): float64 =
  ## The sweep's phase (without its start phase) at `tMs`.
  if spec.law in {slLinear, slExponential, slTable}:
    spec.phaseAt(tMs) - degToRad(spec.startPhaseDeg)
  else:
    let j = int(round(tMs / dtMs))
    doAssert abs(float64(j) * dtMs - tMs) < 1e-9, "on the grid only"
    table[clamp(j, 0, table.len - 1)]

proc signal(spec: SweepSpec, n: int, harmonics: openArray[Harmonic], delayMs = 0.0,
    dc = 0.0): seq[float64] =
  ## `dc + Σ amp·w(t − τ)·sin(k·φ(t − τ) + phase)`, the sweep's taper `w`.
  let table = (if spec.law in {slLinear, slExponential, slTable}: @[] else:
    phaseTable(spec, spec.dtMs, n))
  result = newSeq[float64](n)
  for j in 0 ..< n:
    let t = float64(j) * spec.dtMs - delayMs
    let w = taperAt(spec, t)
    var v = dc
    if w > 0:
      let p = phiAt(spec, table, t, spec.dtMs)
      for h in harmonics: v += h.amp * w * sin(float64(h.order) * p + degToRad(h.phaseDeg))
    result[j] = v

proc flat(spec: SweepSpec, tMs, marginMs: float64): bool =
  ## `tMs` is inside the sweep's flat part, `marginMs` away from the tapers.
  let s = spec.taper.startMs
  let e = spec.taper.endMs
  tMs > s + marginMs and tMs < spec.lengthMs - e - marginMs

proc setup[T](spec: SweepSpec, n: int, params = TrackParams()): (SweepTrackPlan[T],
    SweepReference, SweepTrack) =
  let tp = initSweepTrackPlan(T, params, spec.dtMs, n)
  (tp, initSweepReference(tp), initSweepTrack(tp))

proc toT[T](x: seq[float64]): seq[T] =
  result = newSeq[T](x.len)
  for j, v in x: result[j] = T(v)

let linear1 = linearSweep(8, 80, 6000, 1, initSweepTaper(300, 300, tpCos2))
let linear2 = linearSweep(8, 80, 6000, 2, initSweepTaper(300, 300, tpCos2))
let expo = exponentialSweep(6, 96, 8000, 1, initSweepTaper(500, 250, tpBlackman))
let octave = dbPerOctaveSweep(6, 90, 8000, 2, 3, initSweepTaper(400, 400, tpCos2))
let octave1 = dbPerOctaveSweep(6, 90, 8000, 1, 3, initSweepTaper(400, 400, tpCos2))
let table = tableSweep([SweepPoint(tMs: 0, fHz: 10), SweepPoint(tMs: 2000, fHz: 30),
  SweepPoint(tMs: 4000, fHz: 35), SweepPoint(tMs: 6000, fHz: 70)], 1,
  initSweepTaper(250, 250, tpLinear))

block phase_shifts_and_amplitude:
  # The pilot (as generated) against a copy shifted by θ and scaled: phase θ
  # and amplitude A in the flat part, for every law, at 1 and 2 ms.
  for spec in [linear1, linear2, expo, octave, table]:
    let n = spec.sweepLen + 1000 # a listen time after the sweep
    var pilot = spec.sweepSamples(float64)
    pilot.setLen(n)
    var (tp, reference, track) = setup[float64](spec, n)
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    tp.referenceInto(pilot, reference, ws)
    for (theta, amp) in [(0.0, 1.0), (30.0, 2.5), (-75.0, 0.4), (179.0, 1.0)]:
      let s = signal(spec, n, [(1, amp, theta)])
      tp.trackInto(s, reference, track, ws)
      var seen = 0
      for m in 0 ..< track.len:
        if not flat(spec, track.timeMs[m], 600): continue
        doAssert abs(wrapDeg(track.phaseDeg[m] - theta)) < 0.05,
          $spec.law & " θ " & $theta & " at " & $track.timeMs[m] & ": " & $track.phaseDeg[m]
        doAssert abs(track.amplitude[0][m] - amp) < 1e-4 * amp,
          $spec.law & " at " & $track.timeMs[m] & ": " & $track.amplitude[0][m]
        doAssert track.thdPct[m] < 0.01 and track.amplitude[1][m] < 1e-4 * amp,
          $spec.law & " dt " & $spec.dtMs & " at " & $track.timeMs[m] & ": H2 " &
          $track.amplitude[1][m] & ", THD " & $track.thdPct[m]
        inc seen
      doAssert seen > 300, $seen

block harmonics_at_known_levels:
  # The 2nd at 10 %, the 3rd at 3 %, the 5th at 0.5 % (the 4th absent), with
  # phases of their own and a DC offset: each recovered, and THD both ways.
  # Every harmonic stays below Nyquist (recorded data are anti-alias
  # filtered: a 5th harmonic past Nyquist would fold onto the 3rd).
  let want = [0.0, 0.10, 0.03, 0.0, 0.005]
  let thdF = 100 * sqrt(0.1 ^ 2 + 0.03 ^ 2 + 0.005 ^ 2)
  let thdT = 100 * sqrt((0.1 ^ 2 + 0.03 ^ 2 + 0.005 ^ 2) / (1 + 0.1 ^ 2 + 0.03 ^ 2 + 0.005 ^ 2))
  for spec in [linear1, expo, octave1]:
    let n = spec.sweepLen
    let pilot = spec.sweepSamples(float64)
    let s = signal(spec, n, [(1, 1.0, 12.0), (2, 0.1, -40.0), (3, 0.03, 100.0),
      (5, 0.005, 0.0)], dc = 0.3)
    for thd in ThdReference:
      var (tp, reference, track) = setup[float64](spec, n, TrackParams(thd: thd))
      var ws = initWorkspace()
      ws.reserve(tp.workspaceSize)
      tp.referenceInto(pilot, reference, ws)
      tp.trackInto(s, reference, track, ws)
      var hd2 = newSeq[float64](track.len)
      track.distortionInto(2, hd2)
      let nyq = 500 / spec.dtMs
      var seen = 0
      for m in 0 ..< track.len:
        if not flat(spec, track.timeMs[m], 700): continue
        let f = spec.frequencyAt(track.timeMs[m])
        doAssert abs(track.phaseDeg[m] - 12) < 0.05, $track.phaseDeg[m]
        for k in 1 .. 5:
          let a = track.amplitude[k - 1][m]
          if float64(k) * f * 1.1 > 0.8 * nyq: continue # near or above the limit
          let w = (if k == 1: 1.0 else: want[k - 1])
          doAssert abs(a - w) < 2e-4, $spec.law & " H" & $k & " at " & $track.timeMs[m] &
            ": " & $a
        if 5 * f * 1.1 < 0.8 * nyq:
          let expect = (if thd == thdFundamental: thdF else: thdT)
          doAssert abs(track.thdPct[m] - expect) < 0.02, $track.thdPct[m] & " vs " & $expect
          doAssert abs(hd2[m] - 10) < 0.02
          inc seen
      doAssert seen > 100, $spec.law & " " & $seen

block harmonics_above_the_limit_are_nan:
  # At 2 ms (Nyquist 250 Hz, limit 200 Hz) the 3rd harmonic of an 8-80 Hz
  # sweep is measured up to 66.7 Hz only; THD then uses the 2nd alone.
  let spec = linear2
  let n = spec.sweepLen
  var (tp, reference, track) = setup[float64](spec, n, TrackParams(harmonics: 3))
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  tp.referenceInto(spec.sweepSamples(float64), reference, ws)
  tp.trackInto(signal(spec, n, [(1, 1.0, 0.0), (2, 0.1, 0.0), (3, 0.03, 0.0)]), reference,
    track, ws)
  var above, below = 0
  for m in 0 ..< track.len:
    if not flat(spec, track.timeMs[m], 300): continue
    let f = spec.frequencyAt(track.timeMs[m])
    if 3 * f > 205:
      doAssert track.amplitude[2][m].isNaN and abs(track.thdPct[m] - 10) < 0.02
      inc above
    elif 3 * f < 195:
      doAssert abs(track.amplitude[2][m] - 0.03) < 2e-4
      inc below
  doAssert above > 20 and below > 100

block delay_and_frequency:
  # The pilot delayed by τ = 3.5 ms: phase −360·∫f over the delay, the
  # signal's fundamental at f(t − τ), the reference's frequency the law.
  for spec in [linear1, expo]:
    let n = spec.sweepLen + 500
    var pilot = spec.sweepSamples(float64)
    pilot.setLen(n)
    let tau = 3.5
    let s = signal(spec, n, [(1, 1.0, 0.0)], delayMs = tau)
    var (tp, reference, track) = setup[float64](spec, n)
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    tp.referenceInto(pilot, reference, ws)
    tp.trackInto(s, reference, track, ws)
    var seen = 0
    for m in 0 ..< track.len:
      let t = track.timeMs[m]
      if not flat(spec, t, 600): continue
      let want = wrapDeg(radToDeg(spec.phaseAt(t - tau) - spec.phaseAt(t)))
      doAssert abs(wrapDeg(track.phaseDeg[m] - want)) < 0.1,
        $spec.law & " at " & $t & ": " & $track.phaseDeg[m] & " vs " & $want
      doAssert abs(track.referenceHz[m] - spec.frequencyAt(t)) < 0.05, $track.referenceHz[m]
      doAssert abs(track.frequencyHz[m] - spec.frequencyAt(t - tau)) < 0.05,
        $track.frequencyHz[m] & " vs " & $spec.frequencyAt(t - tau)
      inc seen
    doAssert seen > 300

block peak_per_period:
  # A fundamental of 2 and a 2nd harmonic of 0.5: the peak of each period is
  # the waveform's maximum, beyond the fundamental: max |2·sin p + 0.5·cos 2p|
  # = 2.5 (at p = −π/2). At 1 ms, from 9 Hz (110 samples a period) to 80 Hz
  # (12.5), and at 2 ms to 6.25 samples a period (the 2nd harmonic to 3.1):
  # the largest sample alone was 0.25 % (4.5 %) short even with a parabola
  # through it; the band-limited value is within 2e-4.
  for spec in [linear1, linear2]:
    let n = spec.sweepLen
    let s = signal(spec, n, [(1, 2.0, 0.0), (2, 0.5, 90.0)])
    var (tp, reference, track) = setup[float64](spec, n)
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    tp.referenceInto(spec.sweepSamples(float64), reference, ws)
    tp.trackInto(s, reference, track, ws)
    var worst = 0.0
    for m in 0 ..< track.len:
      let t = track.timeMs[m]
      if not flat(spec, t, 600): continue
      worst = max(worst, abs(track.peak[m] - 2.5) / 2.5)
      doAssert abs(track.amplitude[0][m] - 2) < 2e-4
    doAssert worst < 2e-4, $spec.dtMs & ": " & $worst # measured 3.6e-5 and 4.3e-5

block noise:
  # White noise 60 dB below the signal: amplitude and phase stay close.
  let spec = linear1
  let n = spec.sweepLen
  var s = signal(spec, n, [(1, 1.0, 20.0)])
  var rng = initRand(5)
  for v in s.mitems: v += rng.gauss(0, 1e-3)
  var (tp, reference, track) = setup[float64](spec, n)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  tp.referenceInto(spec.sweepSamples(float64), reference, ws)
  tp.trackInto(s, reference, track, ws)
  for m in 0 ..< track.len:
    if not flat(spec, track.timeMs[m], 600): continue
    doAssert abs(track.amplitude[0][m] - 1) < 2e-3 and abs(track.phaseDeg[m] - 20) < 0.15

block reference_offset_and_drift:
  # A recorded reference with a 10 % DC offset and a 2 % drift at 0.3 Hz:
  # both are removed before its phase is taken. Left in, they would put a
  # ripple once per period into its phase, read as a false 2nd harmonic of
  # half their size in every signal tracked against it.
  let spec = linear1
  let n = spec.sweepLen
  var pilot = spec.sweepSamples(float64)
  for j in 0 ..< n: pilot[j] += 0.1 + 0.02 * sin(2 * PI * 0.3 * float64(j) / 1000)
  var (tp, reference, track) = setup[float64](spec, n)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  tp.referenceInto(pilot, reference, ws)
  tp.trackInto(signal(spec, n, [(1, 1.0, 30.0)]), reference, track, ws)
  var seen = 0
  for m in 0 ..< track.len:
    if not flat(spec, track.timeMs[m], 600): continue
    doAssert abs(track.phaseDeg[m] - 30) < 0.05 and track.amplitude[1][m] < 1e-4,
      $track.timeMs[m] & ": " & $track.phaseDeg[m] & ", H2 " & $track.amplitude[1][m]
    inc seen
  doAssert seen > 300

block outside_the_sweep:
  # Before the sweep reaches −40 dB, in the listen time after it, and within
  # half a window of its ends: NaN. A reference of zeros or with a NaN: no
  # sweep at all.
  let spec = linear1
  let n = spec.sweepLen + 2000
  var pilot = spec.sweepSamples(float64)
  pilot.setLen(n)
  var (tp, reference, track) = setup[float64](spec, n)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  tp.referenceInto(pilot, reference, ws)
  let sp = reference.span
  # −40 dB of the cos² taper: sin²(π·t/600) = 0.01 at t = 19.1 ms. At the
  # 8 Hz start the analytic envelope is a little above the taper (12 ms).
  let edge = 600 / PI * arcsin(0.1)
  doAssert sp.a in 8 .. 22 and abs(float64(sp.b) - (6000 - edge)) <= 2, $sp
  tp.trackInto(pilot, reference, track, ws)
  doAssert track.phaseDeg[0].isNaN and track.amplitude[0][0].isNaN and track.peak[0].isNaN
  for m in 0 ..< track.len:
    let t = track.timeMs[m]
    if t > 6000 or t < float64(sp.a):
      doAssert track.amplitude[0][m].isNaN and not reference.isMeasured(m)
    # Half a window (two periods, by the law's phase) inside the span:
    # measured; reaching out of it: not.
    let pc = spec.phaseAt(t)
    if pc - 4 * PI > spec.phaseAt(float64(sp.a) + 2) and
        pc + 4 * PI < spec.phaseAt(float64(sp.b) - 2):
      doAssert reference.isMeasured(m) and abs(track.phaseDeg[m]) < 0.05, $t
    if pc - 4 * PI < spec.phaseAt(float64(sp.a) - 2) or
        pc + 4 * PI > spec.phaseAt(float64(sp.b) + 2):
      doAssert not reference.isMeasured(m) and track.phaseDeg[m].isNaN, $t
  var zeros = newSeq[float64](n)
  tp.referenceInto(zeros, reference, ws)
  doAssert reference.span.len == 0
  tp.trackInto(pilot, reference, track, ws)
  for m in 0 ..< track.len: doAssert track.amplitude[0][m].isNaN
  pilot[100] = NaN
  tp.referenceInto(pilot, reference, ws)
  doAssert reference.span.len == 0

block tapers_followed:
  # In the start taper the fundamental follows the ramp: it rises, and stays
  # within the taper's range over the window.
  let spec = linearSweep(10, 80, 6000, 1, initSweepTaper(1500, 300, tpLinear))
  let n = spec.sweepLen
  var (tp, reference, track) = setup[float64](spec, n)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  let pilot = spec.sweepSamples(float64)
  tp.referenceInto(pilot, reference, ws)
  tp.trackInto(pilot, reference, track, ws)
  var last = 0.0
  var seen = 0
  for m in 0 ..< track.len:
    let t = track.timeMs[m]
    let a = track.amplitude[0][m]
    if a.isNaN or t > 1300: continue
    let win = 2000 / spec.frequencyAt(t) # half a window, ms
    doAssert a >= taperAt(spec, t - win) - 1e-3 and a <= taperAt(spec, t + win) + 1e-3, $t
    doAssert a >= last - 1e-9
    # The amplitude changes within the window, so the conjugate term no longer
    # cancels exactly: at the ramp's start (the window from near zero to a
    # quarter) the phase is off by 0.1°.
    doAssert abs(track.phaseDeg[m]) < 0.25, $t & ": " & $track.phaseDeg[m] & " a " & $a
    last = a
    inc seen
  doAssert seen > 50

block float32_plan:
  let spec = linear2
  let n = spec.sweepLen
  let pilot = toT[float32](spec.sweepSamples(float64))
  let s = toT[float32](signal(spec, n, [(1, 3.0, -30.0), (2, 0.3, 0.0)]))
  var (tp, reference, track) = setup[float32](spec, n)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  tp.referenceInto(pilot, reference, ws)
  tp.trackInto(s, reference, track, ws)
  var seen = 0
  for m in 0 ..< track.len:
    if not flat(spec, track.timeMs[m], 600): continue
    doAssert abs(track.phaseDeg[m] + 30) < 0.05 and abs(track.amplitude[0][m] - 3) < 1e-3
    doAssert abs(track.thdPct[m] - 10) < 0.05
    inc seen
  doAssert seen > 200

block comparisons:
  let spec = linear1
  let n = spec.sweepLen
  var (tp, reference, a) = setup[float64](spec, n)
  var b = initSweepTrack(tp)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  tp.referenceInto(spec.sweepSamples(float64), reference, ws)
  tp.trackInto(signal(spec, n, [(1, 2.0, 30.0)]), reference, a, ws)
  tp.trackInto(signal(spec, n, [(1, 0.5, -75.0)]), reference, b, ws)
  var phase, ratio = newSeq[float64](a.len)
  compareTracksInto(a, b, phase, ratio)
  for m in 0 ..< a.len:
    if not flat(spec, a.timeMs[m], 600): continue
    doAssert abs(phase[m] - 105) < 0.05 and abs(ratio[m] - 20 * log10(4.0)) < 1e-3
  doAssert phase[0].isNaN and ratio[0].isNaN

block refusals_and_shapes:
  doAssertRaises(ValueError): discard initSweepTrackPlan(float64, TrackParams(harmonics: 0), 1, 100)
  doAssertRaises(ValueError): discard initSweepTrackPlan(float64, TrackParams(harmonics: 17), 1, 100)
  doAssertRaises(ValueError): discard initSweepTrackPlan(float64, TrackParams(cycles: 2), 1, 100)
  doAssertRaises(ValueError): discard initSweepTrackPlan(float64, TrackParams(stepMs: 0), 1, 100)
  doAssertRaises(ValueError): discard initSweepTrackPlan(float64, TrackParams(minReferenceDb: 0), 1, 100)
  doAssertRaises(ValueError):
    discard initSweepTrackPlan(float64, TrackParams(maxHarmonicFraction: 1.5), 1, 100)
  doAssertRaises(ValueError): discard initSweepTrackPlan(float64, TrackParams(), NaN, 100)
  doAssertRaises(ValueError): discard initSweepTrackPlan(float64, TrackParams(), 1, 1)
  let tp = initSweepTrackPlan(float64, TrackParams(stepMs: 4), 2, 1001)
  doAssert tp.stepSamples == 2 and tp.points == 501 and tp.pointMs(10) == 40
  let other = initSweepTrackPlan(float64, TrackParams(), 2, 1001)
  var reference = initSweepReference(tp)
  var track = initSweepTrack(other)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  var x = newSeq[float64](1001)
  doAssertRaises(AssertionDefect): tp.trackInto(x, reference, track, ws) # another plan's track
  var t2 = initSweepTrack(tp)
  doAssertRaises(ValueError): t2.distortionInto(1, x)
  doAssertRaises(ValueError): t2.distortionInto(6, x)
  var long = newSeq[float64](1002)
  doAssertRaises(AssertionDefect): tp.referenceInto(long, reference, ws)
  # A track whose fields were resized is not sized for the plan either.
  tp.referenceInto(x, reference, ws)
  t2.peak.setLen(3)
  doAssertRaises(AssertionDefect): tp.trackInto(x, reference, t2, ws)
  t2 = initSweepTrack(tp)
  t2.amplitude[2].setLen(0)
  doAssertRaises(AssertionDefect): tp.trackInto(x, reference, t2, ws)
  # A step past the trace (in samples beyond an int): one point, not a step of 1.
  for stepMs in [5000.0, 1e19, 1e300]:
    let one = initSweepTrackPlan(float64, TrackParams(stepMs: stepMs), 2, 1001)
    doAssert one.points == 1 and one.stepSamples >= 1001, $stepMs & ": " & $one.points

block nan_stays_in_its_windows:
  # A NaN or an infinite sample makes NaN the points whose window holds it,
  # and only those: the running sums skip it (before, every later point was
  # NaN, 300 of them here).
  let spec = linear1
  let n = spec.sweepLen
  var (tp, reference, clean) = setup[float64](spec, n)
  var track = initSweepTrack(tp)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  tp.referenceInto(spec.sweepSamples(float64), reference, ws)
  let s = signal(spec, n, [(1, 1.0, 20.0), (2, 0.1, 0.0)])
  tp.trackInto(s, reference, clean, ws)
  for bad in [NaN, Inf, -Inf]:
    var x = s
    x[2000] = bad
    tp.trackInto(x, reference, track, ws)
    var blank, kept = 0
    for m in 0 ..< track.len:
      if clean.amplitude[0][m].isNaN:
        doAssert track.amplitude[0][m].isNaN
        continue
      let t = track.timeMs[m]
      let half = 2000 / spec.frequencyAt(t) # half a window (two periods), ms
      if abs(t - 2000) < half - 2:
        doAssert track.amplitude[0][m].isNaN and track.phaseDeg[m].isNaN and
          track.thdPct[m].isNaN and track.peak[m].isNaN, $bad & " at " & $t
        inc blank
      elif abs(t - 2000) > half + 2:
        for k in 0 ..< 5:
          let a = track.amplitude[k][m]
          let c = clean.amplitude[k][m]
          doAssert (a.isNaN and c.isNaN) or abs(a - c) < 1e-9, $bad & " at " & $t & ": " & $a
        doAssert abs(track.phaseDeg[m] - clean.phaseDeg[m]) < 1e-7 and
          abs(track.peak[m] - clean.peak[m]) < 1e-9 and
          (track.frequencyHz[m].isNaN or abs(track.frequencyHz[m] - clean.frequencyHz[m]) < 1e-6),
          $bad & " at " & $t
        inc kept
    doAssert blank >= 10 and kept > 500, $blank & " blank, " & $kept & " kept" # 12 and 560

block window_selectivity:
  # What the window lets through besides its harmonic (module documentation):
  # a subharmonic at f/2 reads 1/6 of its level in the fundamental over 4
  # periods (the weight's cos(2x/C) term is at f/2), nothing over 6; and a
  # reference with a 2nd harmonic of its own puts about half of it into
  # every signal's 2nd harmonic.
  let spec = linear1
  let n = spec.sweepLen
  let pilot = spec.sweepSamples(float64)
  for (cycles, leak) in [(4, 1 / 6), (6, 0.0)]:
    var (tp, reference, track) = setup[float64](spec, n, TrackParams(cycles: cycles))
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    tp.referenceInto(pilot, reference, ws)
    tp.trackInto(signal(spec, n, [(1, 1.0, 0.0)]), reference, track, ws)
    let a1 = track.amplitude[0]
    var sub = signal(spec, n, [(1, 1.0, 0.0)])
    let table = phaseTable(spec, spec.dtMs, n)
    for j in 0 ..< n:
      let t = float64(j) * spec.dtMs
      sub[j] += 0.05 * taperAt(spec, t) * cos(0.5 * phiAt(spec, table, t, spec.dtMs))
    tp.trackInto(sub, reference, track, ws)
    var worst = 0.0
    for m in 0 ..< track.len:
      if flat(spec, track.timeMs[m], 800):
        worst = max(worst, abs(track.amplitude[0][m] - a1[m]))
    doAssert abs(worst - 0.05 * leak) < 1e-3, $cycles & ": " & $worst
  var (tp, reference, track) = setup[float64](spec, n)
  var ws = initWorkspace()
  ws.reserve(tp.workspaceSize)
  tp.referenceInto(signal(spec, n, [(1, 1.0, 0.0), (2, 0.05, 0.0)]), reference, ws)
  tp.trackInto(pilot, reference, track, ws)
  for m in 0 ..< track.len:
    if flat(spec, track.timeMs[m], 600):
      doAssert track.amplitude[1][m] > 0.02 and track.amplitude[1][m] < 0.035,
        $track.amplitude[1][m]

block no_allocation:
  let spec = linear2
  let n = spec.sweepLen
  let pilot = toT[float32](spec.sweepSamples(float64))
  let s = toT[float32](signal(spec, n, [(1, 1.0, 10.0), (3, 0.1, 0.0)]))
  for harmonics in [5, 12]: # the 8- and the 16-lane scans
    var (tp, reference, track) = setup[float32](spec, n, TrackParams(harmonics: harmonics))
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    tp.referenceInto(pilot, reference, ws)
    tp.trackInto(s, reference, track, ws)
    let count = allocsIn:
      tp.referenceInto(pilot, reference, ws)
      tp.trackInto(s, reference, track, ws)
    doAssert count == 0, $count & " allocations"
    var seen = 0
    for m in 0 ..< track.len:
      if flat(spec, track.timeMs[m], 600):
        doAssert abs(track.amplitude[2][m] - 0.1) < 2e-4 or track.amplitude[2][m].isNaN
        inc seen
    doAssert seen > 200

block example_pilot:
  # exampleFiles/Sweep0.sgy: one recorded pilot, a linear 8-110 Hz sweep of
  # 12 s at 1 ms (its headers do not describe it: `pilotFromTrace` gives no
  # sweep). Against itself: phase 0, no distortion, its own law as the
  # reference's frequency. A generated 8-110 Hz sweep against it: the same
  # frequency, a phase that does not move along the sweep.
  let path = currentSourcePath().parentDir / ".." / "exampleFiles" / "Sweep0.sgy"
  if not fileExists(path):
    echo "  skipped (no exampleFiles/Sweep0.sgy)"
  else:
    var f = openSeismic(path, keys = true)
    let p = pilotFromTrace(f, 0, float64)
    f.close()
    doAssert p.dtMs == 1 and p.len == 12001
    let law = linearSweep(8, 110, 12000, 1)
    let tp = initSweepTrackPlan(float64, TrackParams(), 1, p.len)
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    var reference = initSweepReference(tp)
    var self, gen = initSweepTrack(tp)
    tp.referenceInto(p.samples, reference, ws)
    tp.trackInto(p.samples, reference, self, ws)
    tp.trackInto(law.sweepSamples(float64), reference, gen, ws)
    var lo, hi = gen.phaseDeg[100]
    var seen = 0
    for m in 0 ..< self.len:
      let t = self.timeMs[m]
      if t < 1000 or t > 11000: continue
      doAssert abs(self.phaseDeg[m]) < 1e-3 and self.thdPct[m] < 0.01, $t
      doAssert abs(self.referenceHz[m] - law.frequencyAt(t)) < 0.05,
        $t & ": " & $self.referenceHz[m] & " vs " & $law.frequencyAt(t)
      doAssert abs(gen.frequencyHz[m] - self.referenceHz[m]) < 0.05
      lo = min(lo, gen.phaseDeg[m])
      hi = max(hi, gen.phaseDeg[m])
      inc seen
    doAssert seen == 1001 and hi - lo < 1, $lo & " .. " & $hi
