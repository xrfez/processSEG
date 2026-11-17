## Tracking against sweep time: on synthetic sweeps of several
## laws, with tapers, at 1 and 2 ms, each measure recovers what was put in:
## a phase shift, a delay, harmonics at known levels, the fundamental's
## amplitude (with a DC offset and with noise), the frequency law, the peak
## per period; NaN outside the sweep and above the harmonic limit; float32;
## comparisons; refusals; no allocation. On signals generated apart from
## the library (`utils/chirps`): the window's minimum time, the flat part
## of a reference, THD's harmonics and their count, a reference whose phase
## steps back. On the example
## files (skipped when absent): a recorded pilot against itself and against
## its law.

import std/[assertions, math, os, random]
import parseSEG, processSEG
import utils/[allocs, chirps]

proc main*() =
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

  func expectedCycles(f: float64, cycles = 4, minWindowMs = 100.0): int =
    ## The window's periods at `f` (module documentation): `cycles`, or the
    ## smallest even count reaching `minWindowMs`.
    let need = minWindowMs * f / 1000
    if need <= float64(cycles): cycles else: 2 * int(ceil(need / 2))

  func nearEdge(f: float64, minWindowMs = 100.0): bool =
    ## `f` within 0.2 % of a frequency where the window's periods change.
    let need = minWindowMs * f / 1000
    abs(need / 2 - round(need / 2)) < 2e-3 * need

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
          # THD over harmonics 2-5, each of them (thFixed): NaN where the 5th
          # is past the limit (above 40 Hz at 2 ms).
          let all = not track.amplitude[4][m].isNaN
          doAssert track.amplitude[1][m] < 1e-4 * amp and (if all: track.thdPct[m] < 0.01 and
            track.thdCount[m] == 4 else: track.thdPct[m].isNaN and track.thdCount[m] == 0),
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
    # sweep is measured up to 66.7 Hz only; THD then uses the 2nd alone when
    # it counts all that fit, and is NaN when it counts 2-3 (fixed).
    let spec = linear2
    let n = spec.sweepLen
    for mode in ThdHarmonics:
      var (tp, reference, track) = setup[float64](spec, n,
        TrackParams(harmonics: 3, thdHarmonics: mode))
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
          doAssert track.amplitude[2][m].isNaN
          if mode == thAllThatFit:
            doAssert abs(track.thdPct[m] - 10) < 0.02 and track.thdCount[m] == 1
          else:
            doAssert track.thdPct[m].isNaN and track.thdCount[m] == 0
          inc above
        elif 3 * f < 195:
          doAssert abs(track.amplitude[2][m] - 0.03) < 2e-4 and track.thdCount[m] == 2 and
            abs(track.thdPct[m] - 100 * sqrt(0.1 ^ 2 + 0.03 ^ 2)) < 0.02, $mode & " " & $f
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
      if t > 6000 or nearEdge(spec.frequencyAt(t)): continue
      # Half a window (C/2 periods, by the law's phase; C = 4 up to 40 Hz,
      # then 6 and 8 by the 100 ms minimum) inside the span: measured;
      # reaching out of it: not.
      let half = PI * float64(expectedCycles(spec.frequencyAt(t)))
      let pc = spec.phaseAt(t)
      if pc - half > spec.phaseAt(float64(sp.a) + 2) and
          pc + half < spec.phaseAt(float64(sp.b) - 2):
        doAssert reference.isMeasured(m) and abs(track.phaseDeg[m]) < 0.05, $t
        doAssert reference.cyclesAt(m) == expectedCycles(spec.frequencyAt(t)), $t
      if pc - half < spec.phaseAt(float64(sp.a) - 2) or
          pc + half > spec.phaseAt(float64(sp.b) + 2):
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
      if track.amplitude[4][m].isNaN: doAssert track.thdPct[m].isNaN # above 40 Hz at 2 ms
      else: doAssert abs(track.thdPct[m] - 10) < 0.05
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
    doAssertRaises(ValueError): discard initSweepTrackPlan(float64, TrackParams(minReferenceDb: 0),
        1, 100)
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
        let half = 500 * float64(reference.cyclesAt(m)) / spec.frequencyAt(t) # half a window, ms
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
      var (tp, reference, track) = setup[float64](spec, n,
        TrackParams(cycles: cycles, minWindowMs: 0))
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
        doAssert abs(self.phaseDeg[m]) < 1e-3, $t
        # THD over harmonics 2-5: NaN from 80 Hz, where the 5th passes the limit.
        doAssert self.thdPct[m] < 0.01 or (self.thdPct[m].isNaN and self.amplitude[4][m].isNaN), $t
        doAssert abs(self.referenceHz[m] - law.frequencyAt(t)) < 0.05,
          $t & ": " & $self.referenceHz[m] & " vs " & $law.frequencyAt(t)
        doAssert abs(gen.frequencyHz[m] - self.referenceHz[m]) < 0.05
        lo = min(lo, gen.phaseDeg[m])
        hi = max(hi, gen.phaseDeg[m])
        inc seen
      doAssert seen == 1001 and hi - lo < 1, $lo & " .. " & $hi

  # ------------------------------------- polarity, window, judged part, THD --

  proc tracked(c: Chirp, dtMs: float64, n: int, x: seq[float64],
      params = TrackParams()): (SweepTrack, SweepReference) =
    ## `x` tracked against the clean chirp `c` (both from `utils/chirps`).
    let tp = initSweepTrackPlan(float64, params, dtMs, n)
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    var r = initSweepReference(tp)
    var t = initSweepTrack(tp)
    tp.referenceInto(chirp(c, dtMs, n), r, ws)
    tp.trackInto(x, r, t, ws)
    (t, r)

  let up = Chirp(law: clLinear, f0: 8, f1: 80, lengthMs: 6000, startMs: 300, endMs: 300)

  proc clear(c: Chirp, r: SweepReference, t: SweepTrack, m: int): bool =
    ## Point `m` is measured and its window is clear of the tapers.
    if not r.isMeasured(m): return false
    let tMs = t.timeMs[m]
    let half = 500 * float64(r.cyclesAt(m)) / c.freqAt(tMs)
    tMs - half > c.startMs and tMs + half < c.lengthMs - c.endMs

  block window_minimum_time:
    # 4 periods up to 40 Hz, then the smallest even count reaching 100 ms: 6
    # to 60 Hz, 8 to 80 Hz. A clean fundamental is measured as well over any
    # whole number of periods, so nothing jumps where the count changes; a
    # subharmonic at f/2 leaks 1/6 of its level over 4 periods and nothing
    # over the even counts; on white noise the phase's rms falls by √(C/4).
    let n = 6001
    let (t, r) = tracked(up, 1, n, chirp(up, 1, n, [Tone(amp: 2, phaseDeg: -75)]))
    var seen: array[10, int]
    for m in 0 ..< t.len:
      if not up.clear(r, t, m): continue
      let f = up.freqAt(t.timeMs[m])
      if nearEdge(f): continue
      let C = r.cyclesAt(m)
      doAssert C == expectedCycles(f), $t.timeMs[m] & ": " & $C
      # (Just past the start taper, over 4 periods, a false H2 of 1.3e-4 of A1,
      # as over 4 periods everywhere; over 6 and 8 periods at most 5e-7.)
      doAssert abs(t.phaseDeg[m] + 75) < 0.002 and abs(t.amplitude[0][m] - 2) < 2e-4 and
        t.amplitude[1][m] < 2 * 2e-4, $t.timeMs[m]
      if C >= 6: doAssert t.amplitude[1][m] < 2 * 1e-5 and abs(t.phaseDeg[m] + 75) < 1e-5
      if m > 0 and r.cyclesAt(m - 1) != C and up.clear(r, t, m - 1): # no jump
        doAssert abs(t.phaseDeg[m] - t.phaseDeg[m - 1]) < 2e-3 and
          abs(t.amplitude[0][m] - t.amplitude[0][m - 1]) < 1e-5
      inc seen[C]
    doAssert seen[4] > 200 and seen[6] > 100 and seen[8] > 100, $seen
    # An odd `cycles` stays where it is long enough; raised, it is even.
    let (t5, r5) = tracked(up, 1, n, chirp(up, 1, n), TrackParams(cycles: 5))
    for m in 0 ..< t5.len:
      if r5.isMeasured(m):
        let f = up.freqAt(t5.timeMs[m])
        if not nearEdge(f): doAssert r5.cyclesAt(m) == expectedCycles(f, cycles = 5)
    # The subharmonic.
    let sub = chirp(up, 1, n, [Tone(), Tone(order: 0.5, amp: 0.05)])
    let (ts, _) = tracked(up, 1, n, sub)
    let (tc, _) = tracked(up, 1, n, chirp(up, 1, n))
    var leak: array[10, float64]
    for m in 0 ..< ts.len:
      if up.clear(r, ts, m): leak[r.cyclesAt(m)] = max(leak[r.cyclesAt(m)],
        abs(ts.amplitude[0][m] - tc.amplitude[0][m]))
    doAssert abs(leak[4] - 0.05 / 6) < 3e-4 and leak[6] < 1e-5 and leak[8] < 1e-5, $leak
    # White noise of 1 % over 12 realizations: the phase's rms per band.
    var sq: array[2, array[10, float64]]
    var count: array[10, int]
    for seed in 1 .. 12:
      let x = chirp(up, 1, n, [Tone(phaseDeg: 20)], noise = 0.01, seed = seed)
      let (t4, _) = tracked(up, 1, n, x, TrackParams(minWindowMs: 0))
      let (tm, rm) = tracked(up, 1, n, x)
      for m in 0 ..< tm.len:
        let f = up.freqAt(tm.timeMs[m])
        if f < 20 or f > 76 or nearEdge(f) or tm.phaseDeg[m].isNaN or t4.phaseDeg[m].isNaN:
          continue
        let C = rm.cyclesAt(m)
        sq[0][C] += wrapDeg(t4.phaseDeg[m] - 20) ^ 2
        sq[1][C] += wrapDeg(tm.phaseDeg[m] - 20) ^ 2
        inc count[C]
    for C in [4, 6, 8]:
      let ratio = sqrt(sq[0][C] / sq[1][C])
      doAssert count[C] > 1000 and abs(ratio / sqrt(float64(C) / 4) - 1) < 0.1,
        $C & ": " & $ratio # measured 1.000, 1.240, 1.417

  block flat_part:
    # The flat part between the tapers, from the reference's envelope: within
    # a period (of the reference there) of each taper's end, for cos², linear
    # and Blackman tapers of 0.3 and 1 s, at 1 and 2 ms; an untapered sweep is
    # flat from within a period of its start (before its first measured
    # point) to within a period of its end.
    for dtMs in [1.0, 2.0]:
      let n = int(7000 / dtMs) + 1
      for ramp in [rmCos2, rmLinear, rmBlackman, rmNone]:
        for taperMs in [300.0, 1000.0]:
          if ramp == rmNone and taperMs > 300: continue
          let c = Chirp(law: clLinear, f0: 8, f1: 80, lengthMs: 6000, ramp: ramp,
            startMs: (if ramp == rmNone: 0.0 else: taperMs),
            endMs: (if ramp == rmNone: 0.0 else: taperMs))
          let tp = initSweepTrackPlan(float64, TrackParams(), dtMs, n)
          var ws = initWorkspace()
          ws.reserve(tp.workspaceSize)
          var r = initSweepReference(tp)
          tp.referenceInto(chirp(c, dtMs, n), r, ws)
          let e = r.extent
          doAssert e.startMs == float64(r.span.a) * dtMs and e.endMs == float64(r.span.b) * dtMs
          doAssert e.flatStartMs == float64(r.flatSpan.a) * dtMs
          let a = c.startMs
          let b = c.lengthMs - c.endMs
          doAssert abs(e.flatStartMs - a) <= 1000 / c.freqAt(a) and
            abs(e.flatEndMs - b) <= 1000 / c.freqAt(b),
            $ramp & " " & $taperMs & " at " & $dtMs & " ms: " & $e
          if ramp == rmNone:
            # The first measured point is half a window (2 periods) in.
            var first = 0
            while not r.isMeasured(first): inc first
            doAssert e.flatStartMs <= float64(first) * 10, $e
    # A sloped plateau is still flat: the amplitude rising or falling by 30 %
    # of its mean over the sweep, or rising by 60 %, between tapers of 0.3, 1
    # and 2 s.
    # Where the slope passes 2 % of the tapers' fastest rise (a 1 s taper and
    # 30 %), the tapers' ends are found with looser criteria, not left in
    # the judged part (a falling plateau's 1 s end taper, judged, failed a
    # healthy signal's phase and THD limits with 1 % noise).
    for (taperMs, tilt) in [(300.0, 0.3), (1000.0, 0.3), (1000.0, -0.3), (300.0, 0.6),
        (1000.0, 0.6), (2000.0, 0.3)]:
      let c = Chirp(law: clLinear, f0: 8, f1: 80, lengthMs: 6000, startMs: taperMs,
        endMs: taperMs, tilt: tilt)
      let n = 7001
      let tp = initSweepTrackPlan(float64, TrackParams(), 1, n)
      var ws = initWorkspace()
      ws.reserve(tp.workspaceSize)
      var r = initSweepReference(tp)
      tp.referenceInto(chirp(c, 1, n), r, ws)
      let e = r.extent
      let a = c.startMs
      let b = c.lengthMs - c.endMs
      # (Within 3 periods or 4 % of the taper: the slope moves the tapers'
      # ends; measured 59 ms, 3 % past the start of a 2 s end taper.)
      doAssert abs(e.flatStartMs - a) <= max(3 * 1000 / c.freqAt(a), 0.04 * taperMs) and
        abs(e.flatEndMs - b) <= max(3 * 1000 / c.freqAt(b), 0.04 * taperMs),
        $taperMs & " " & $tilt & ": " & $e
    # An amplitude that rises all through the sweep's first half never
    # settles there: the judged part runs from the sweep's start.
    block:
      let c = Chirp(law: clLinear, f0: 8, f1: 80, lengthMs: 6000, ramp: rmLinear,
        startMs: 3000, endMs: 300)
      let n = 7001
      let tp = initSweepTrackPlan(float64, TrackParams(), 1, n)
      var ws = initWorkspace()
      ws.reserve(tp.workspaceSize)
      var r = initSweepReference(tp)
      tp.referenceInto(chirp(c, 1, n), r, ws)
      let e = r.extent
      doAssert e.flatStartMs == e.startMs or abs(e.flatStartMs - 3000) <= 1000 / c.freqAt(3000), $e
    # No sweep, no extent.
    let tp = initSweepTrackPlan(float64, TrackParams(), 1, 1000)
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    var r = initSweepReference(tp)
    tp.referenceInto(newSeq[float64](1000), r, ws)
    doAssert r.extent.startMs.isNaN and r.extent.flatEndMs.isNaN and r.flatSpan.len == 0
    var t = initSweepTrack(tp)
    tp.trackInto(newSeq[float64](1000), r, t, ws)
    doAssert t.extent.flatStartMs.isNaN

  block thd_harmonic_count:
    # 16 harmonics at known levels (A_k = 0.1/k, phases of their own), each
    # faded out before Nyquist as an anti-alias filter would, at 0.5 ms on an
    # 8-80 Hz sweep: every harmonic measured within 2e-4 of A1, THD the direct
    # sum over the harmonics it counts, and the count per point the number of
    # harmonics below the limit (0.8 of Nyquist all through the window):
    # all 15 up to 50 Hz, falling to 9 at 80 Hz. Counted all that fit, THD is
    # that sum; counted fixed (2-16), it is NaN once one is past the limit.
    let n = 12001
    var tones: seq[Tone]
    for k in 1 .. 16:
      tones.add Tone(order: float64(k), amp: (if k == 1: 1.0 else: 0.1 / float64(k)),
        phaseDeg: 17 * float64(k))
    let x = chirp(up, 0.5, n, tones, nyquistFade = true)
    for mode in ThdHarmonics:
      let (t, r) = tracked(up, 0.5, n, x, TrackParams(harmonics: 16, thdHarmonics: mode))
      var counts: array[17, int]
      for m in 0 ..< t.len:
        if not up.clear(r, t, m): continue
        let tMs = t.timeMs[m]
        let fTop = up.freqAt(tMs + 500 * float64(r.cyclesAt(m)) / up.freqAt(tMs))
        var sum, own = 0.0 # the levels put in, and as measured
        var measured = 0
        for k in 1 .. 16:
          let a = t.amplitude[k - 1][m]
          let below = float64(k) * fTop < 800
          if abs(float64(k) * fTop - 800) > 2: doAssert below == not a.isNaN, $k & " at " & $tMs
          if a.isNaN: continue
          doAssert abs(a - tones[k - 1].amp) < 2e-4, $k & " at " & $tMs & ": " & $a
          if k > 1:
            sum += tones[k - 1].amp ^ 2
            own += a ^ 2
            inc measured
        if mode == thAllThatFit or measured == 15:
          doAssert t.thdCount[m] == measured and
            abs(t.thdPct[m] - 100 * sqrt(own) / t.amplitude[0][m]) < 1e-9 and
            abs(t.thdPct[m] - 100 * sqrt(sum)) < 5e-3, # 1e-3 from 700 ms
            $mode & " at " & $tMs & ": " & $t.thdPct[m] & " vs " & $(100 * sqrt(sum))
        else:
          doAssert t.thdPct[m].isNaN and t.thdCount[m] == 0
        inc counts[t.thdCount[m]]
      if mode == thAllThatFit:
        for c in 9 .. 15: doAssert counts[c] > 0, $counts
      else:
        doAssert counts[0] > 100 and counts[15] > 200, $counts
    # THD's noise floor grows as the square root of the harmonics it counts:
    # 1 % white noise on a clean fundamental reads 0.29 % over 2-5 and 0.58 %
    # over 2-16 (0.5 ms, 8-40 Hz: all 16 measured).
    let low = Chirp(law: clLinear, f0: 8, f1: 40, lengthMs: 6000, startMs: 300, endMs: 300)
    var floorPct: array[2, float64]
    for i, h in [5, 16]:
      var cnt = 0
      for seed in 1 .. 3:
        let (t, _) = tracked(low, 0.5, n, chirp(low, 0.5, n, noise = 0.01, seed = seed),
          TrackParams(harmonics: h))
        for m in 0 ..< t.len:
          if t.timeMs[m] > 1000 and t.timeMs[m] < 5000:
            floorPct[i] += t.thdPct[m]
            inc cnt
      floorPct[i] /= float64(cnt)
    doAssert abs(floorPct[0] - 0.29) < 0.03 and
      abs(floorPct[1] / floorPct[0] / sqrt(15 / 4) - 1) < 0.1, $floorPct

  block phase_steps_back:
    # A reference whose phase steps back (a 300 ms dropout to noise at -50 dB,
    # two sweeps beating, 30 % noise): the windows found from ψ do not follow
    # the points in order there, and are made to (a scan cannot go back; it
    # ran past the end of the rows without that). The track ends, and away
    # from the dropout it is the clean one's.
    let n = 7001
    var rng = initRand(3)
    let clean = chirp(up, 1, n)
    var dropout = clean
    for j in 3000 ..< 3300: dropout[j] = 3e-3 * rng.gauss()
    var beating = clean
    let slow = Chirp(law: clLinear, f0: 4.8, f1: 48, lengthMs: 6000, startMs: 300, endMs: 300)
    let other = chirp(slow, 1, n)
    for j in 3000 ..< 4000:
      beating[j] += 1.5 * sin(PI * float64(j - 3000) / 1000) ^ 2 * other[j]
    let noisy = chirp(up, 1, n, noise = 0.3, seed = 5)
    for minW in [0.0, 100.0]:
      for (name, refx, lo, hi) in [("dropout", dropout, 2800.0, 3500.0),
          ("beating", beating, 2800.0, 4200.0), ("noise", noisy, 0.0, 0.0)]:
        var x = refx
        for v in x.mitems: v *= 4e4 # the signal is the reference: phase 0
        let tp = initSweepTrackPlan(float64, TrackParams(minWindowMs: minW), 1, n)
        var ws = initWorkspace()
        ws.reserve(tp.workspaceSize)
        var r = initSweepReference(tp)
        var own = initSweepTrack(tp)
        tp.referenceInto(refx, r, ws)
        tp.trackInto(x, r, own, ws)
        var seen = 0
        for m in 0 ..< own.len:
          let tm = own.timeMs[m]
          doAssert own.phaseDeg[m].isNaN or abs(own.phaseDeg[m]) <= 180
          if name == "noise" or own.phaseDeg[m].isNaN or tm < 700 or tm > 5300 or
              (tm > lo and tm < hi): continue
          doAssert abs(own.phaseDeg[m]) < 1e-2 and abs(own.amplitude[0][m] / 4e4 /
            up.weightAt(tm) - 1) < 1e-3, name & " " & $tm & ": " & $own.phaseDeg[m]
          inc seen
        doAssert name == "noise" or seen > 300, name & ": " & $seen

  block new_parameters_refused:
    for w in [-1.0, NaN, Inf, 2e6]:
      doAssertRaises(ValueError):
        discard initSweepTrackPlan(float64, TrackParams(minWindowMs: w), 1, 100)
    doAssert initSweepTrackPlan(float64, TrackParams(minWindowMs: 0), 1, 100).params.minWindowMs == 0

when isMainModule: main()
