## Distortion within the sweep's band, the 500 ms window statistics and the
## measured limits of the sweep tracker, on synthetic sweeps of known content
## (`utils/chirps`; linear up and down, 0.5, 1 and 2 ms): THD over the
## harmonics up to a band edge (equal to the planted in-band harmonics, 0 once
## the 2nd leaves the band); the reference's top frequency; where the
## fundamental stops being measured and what a higher limit gives; the peak of
## a pure fundamental from 0.05 to 0.8 of Nyquist; window statistics against
## hand-computed values; the judged range of a trace shorter than its
## reference.
##
## Each test was seen to fail with its change undone: THD summing 2 .. 5
## (`harmonicBandHz` ignored); the fundamental's limit as the harmonics' (the
## cut and last-measured checks); the 8-tap interpolator (`peakHalf = 8`,
## `peakBeta = 8`: the peak scan reads up to 4 % low above 0.7 of Nyquist);
## partial windows kept whole (`minFill` ignored).

import std/[assertions, math]
import parseSEG, processSEG
import utils/chirps

proc main*() =
  const edge = 220.0

  proc sweepOf(dtMs, f0, f1: float64, lengthMs = 12000.0): Chirp =
    Chirp(law: clLinear, f0: f0, f1: f1, lengthMs: lengthMs, ramp: rmCos2,
      startMs: 500, endMs: 500)

  proc track(tp: SweepTrackPlan[float64], ref0, x: seq[float64], ws: var Workspace): (
      SweepReference, SweepTrack) =
    var r = initSweepReference(tp)
    var t = initSweepTrack(tp)
    tp.referenceInto(ref0, r, ws)
    tp.trackInto(x, r, t, ws)
    (r, t)

  block in_band_thd:
    # H2 10 % and H3 5 % planted, 6-220 Hz up and 220-6 Hz down. With the band
    # at 220 Hz THD is √(0.1² + 0.05²) = 11.18 % where both are in band
    # (k·f ≤ 220: up to 73 Hz), 10 % up to 110 Hz, and 0 above 110 Hz, where
    # no harmonic is: the same at 0.5, 1 and 2 ms (at 2 ms an in-band
    # harmonic past 0.8 of Nyquist is not measured: THD is NaN there).
    doAssert TrackParams().trackedHarmonics == 5
    doAssert TrackParams(harmonicBandHz: edge).trackedHarmonics == 16
    for dtMs in [0.5, 1.0, 2.0]:
      for (f0, f1) in [(6.0, 220.0), (220.0, 6.0)]:
        let ch = sweepOf(dtMs, f0, f1)
        let n = int(12000 / dtMs) + 1
        let ref0 = chirp(ch, dtMs, n, [Tone()])
        let x = chirp(ch, dtMs, n, [Tone(amp: 1000, phaseDeg: -30),
          Tone(order: 2, amp: 100, phaseDeg: 40), Tone(order: 3, amp: 50, phaseDeg: -60)],
          nyquistFade = true)
        let tp = initSweepTrackPlan(float64, TrackParams(harmonicBandHz: edge), dtMs, n)
        var ws = initWorkspace()
        ws.reserve(tp.workspaceSize)
        let (r, t) = tp.track(ref0, x, ws)
        doAssert t.harmonics == 16
        var seen = [0, 0, 0] # points checked with both, the 2nd only, none in band
        for m in 0 ..< t.len:
          let f = t.referenceHz[m]
          let tm = t.timeMs[m]
          if f.isNaN or tm < 1000 or tm > 11000 or t.amplitude[0][m].isNaN: continue
          let nyq = 500 / dtMs
          let kTop = min(16, int(floor(edge / f))) # the highest harmonic in band
          if abs(2 * f - edge) < 10 or abs(3 * f - edge) < 10: continue # the window straddles it
          if 3 * f > 0.8 * nyq and 3 * f < 0.95 * nyq: continue # faded
          if float64(kTop) * (f - 4) > 0.8 * nyq: # one in band is past the limit
            doAssert t.thdPct[m].isNaN, $dtMs & " at " & $f
            continue
          if kTop >= 2 and float64(kTop) * (f + 4) > 0.8 * nyq: continue # at it: the window's spread
          let want = (if 3 * f < edge: 11.1803 elif 2 * f < edge: 10.0 else: 0.0)
          doAssert abs(t.thdPct[m] - want) < 0.05, $dtMs & " " & $f & " Hz: " & $t.thdPct[m]
          doAssert t.thdCount[m] == max(0, kTop - 1), $kTop & " " & $t.thdCount[m]
          inc seen[(if 3 * f < edge: 0 elif 2 * f < edge: 1 else: 2)]
        doAssert seen[0] > 30 and seen[1] > 30 and seen[2] > 100, $seen
        # The reference's top, to default the band to.
        doAssert abs(r.topFrequencyHz - 220) < 2.2,
          $r.topFrequencyHz
        # The other THD of the default parameters: 2 .. 5 whatever the band.
        let tp0 = initSweepTrackPlan(float64, TrackParams(), dtMs, n)
        ws.reserve(tp0.workspaceSize)
        let (_, t0) = tp0.track(ref0, x, ws)
        doAssert t0.harmonics == 5
        for m in 0 ..< t0.len:
          if abs(t0.referenceHz[m] - 60) < 1 and t0.timeMs[m] > 1000 and
              not t0.thdPct[m].isNaN:
            doAssert t0.thdCount[m] == 4 and abs(t0.thdPct[m] - 11.18) < 0.05

  block top_frequency:
    # The top of the reference's band, from a clean one to 5 % rms noise,
    # untapered, down and exponential: within 1 % (the largest instantaneous
    # frequency read 925 Hz of a 220 Hz sweep under 1 % noise).
    let up = sweepOf(1, 6, 220)
    let down = sweepOf(1, 220, 6)
    let bare = Chirp(law: clLinear, f0: 8, f1: 80, lengthMs: 8000, ramp: rmNone)
    let expo = Chirp(law: clExponential, f0: 6, f1: 96, lengthMs: 8000, ramp: rmBlackman,
      startMs: 250, endMs: 500)
    var seed = 1
    for (ch, top, noise) in [(up, 220.0, 0.0), (up, 220.0, 0.001), (up, 220.0, 0.01),
        (up, 220.0, 0.05), (down, 220.0, 0.01), (down, 220.0, 0.05), (bare, 80.0, 0.0),
        (bare, 80.0, 0.01), (expo, 96.0, 0.0), (expo, 96.0, 0.01), (expo, 96.0, 0.05)]:
      inc seed
      let n = int(ch.lengthMs) + 1
      let ref0 = chirp(ch, 1, n, [Tone()], noise = noise, seed = seed)
      let tp = initSweepTrackPlan(float64, TrackParams(), 1, n)
      var ws = initWorkspace()
      ws.reserve(tp.workspaceSize)
      var r = initSweepReference(tp)
      doAssert initSweepReference(tp).topFrequencyHz.isNaN # no sweep yet
      tp.referenceInto(ref0, r, ws)
      doAssert abs(r.topFrequencyHz / top - 1) < 0.01,
        $ch.law & " " & $top & " noise " & $noise & ": " & $r.topFrequencyHz
      var zeros = newSeq[float64](n)
      tp.referenceInto(zeros, r, ws)
      doAssert r.topFrequencyHz.isNaN

  block auto_band:
    # `harmonicBandAuto` is the band at the reference's top, in one pass: the
    # same THD as the typed 220 Hz (within the edge's 1 %), the edge reported.
    let ch = sweepOf(1, 6, 220)
    let n = 12001
    let ref0 = chirp(ch, 1, n, [Tone()], noise = 0.01)
    let x = chirp(ch, 1, n, [Tone(amp: 1000), Tone(order: 2, amp: 100), Tone(order: 3, amp: 50)])
    var ws = initWorkspace()
    let tpa = initSweepTrackPlan(float64, TrackParams(harmonicBandAuto: true), 1, n)
    let tpt = initSweepTrackPlan(float64, TrackParams(harmonicBandHz: edge), 1, n)
    ws.reserve(tpa.workspaceSize)
    let (_, ta) = tpa.track(ref0, x, ws)
    let (_, tt) = tpt.track(ref0, x, ws)
    doAssert ta.harmonics == 16 and abs(ta.bandHz / 220 - 1) < 0.01 and tt.bandHz == edge
    var seen = 0
    for m in 0 ..< ta.len:
      let f = ta.referenceHz[m]
      if f.isNaN or ta.timeMs[m] < 1000 or ta.timeMs[m] > 11000: continue
      if abs(2 * f - 220) < 12 or abs(3 * f - 220) < 12: continue
      doAssert abs(ta.thdPct[m] - tt.thdPct[m]) < 0.3, $f
      inc seen
    doAssert seen > 800
    doAssert initSweepTrack(tpa).bandHz.isNaN
    let tp0 = initSweepTrackPlan(float64, TrackParams(), 1, n)
    ws.reserve(tp0.workspaceSize)
    doAssert tp0.track(ref0, x, ws)[1].bandHz.isNaN

  block band_stepping_down:
    # At 2 ms the 2nd harmonic of 101-109 Hz is in band and not measured, and
    # nothing else is: THD is NaN, not 0 % (0 only where none is in band); at
    # 52 Hz the 4th (208 Hz) is the one not measured: NaN (fixed), or the
    # 2nd and 3rd's 10 % (all that fit).
    let ch = sweepOf(2, 6, 220)
    let n = 6001
    let ref0 = chirp(ch, 2, n, [Tone()])
    let x = chirp(ch, 2, n, [Tone(amp: 1), Tone(order: 2, amp: 0.1)])
    var ws = initWorkspace()
    for mode in [thFixed, thAllThatFit]:
      let tp = initSweepTrackPlan(float64, TrackParams(harmonicBandHz: edge,
        thdHarmonics: mode), 2, n)
      ws.reserve(tp.workspaceSize)
      let (_, t) = tp.track(ref0, x, ws)
      var seen = [0, 0]
      for m in 0 ..< t.len:
        if t.referenceHz[m].isNaN: continue
        if abs(t.referenceHz[m] - 105) < 2:
          doAssert t.thdPct[m].isNaN, $mode & ": " & $t.thdPct[m]
          inc seen[0]
        if abs(t.referenceHz[m] - 52) < 0.5 and t.timeMs[m] > 1000:
          doAssert (mode == thFixed) == t.thdPct[m].isNaN, $mode & ": " & $t.thdPct[m]
          if mode == thAllThatFit: doAssert abs(t.thdPct[m] - 10) < 0.1
          inc seen[1]
      doAssert seen[0] > 3 and seen[1] > 0, $seen

  block fundamental_limit:
    # 6-220 Hz at 2 ms (Nyquist 250 Hz): the fundamental stops at 0.8 of
    # Nyquist, 200 Hz: the track says so; to 0.95 it is measured to the end of
    # the sweep, within 0.5 % and 0.3° (clean tone).
    let ch = sweepOf(2, 6, 220)
    let n = 6001
    let ref0 = chirp(ch, 2, n, [Tone()])
    let x = chirp(ch, 2, n, [Tone(amp: 1000, phaseDeg: 30)])
    var ws = initWorkspace()
    let tp = initSweepTrackPlan(float64, TrackParams(), 2, n)
    ws.reserve(tp.workspaceSize)
    let (_, cut) = tp.track(ref0, x, ws)
    doAssert cut.fundamentalCut
    doAssert cut.lastFundamentalMs > 10000 and cut.lastFundamentalMs < 11000,
      $cut.lastFundamentalMs
    for m in 0 ..< cut.len:
      if cut.timeMs[m] > cut.lastFundamentalMs:
        doAssert cut.amplitude[0][m].isNaN and cut.phaseDeg[m].isNaN
    let tp2 = initSweepTrackPlan(float64, TrackParams(maxFundamentalFraction: 0.95), 2, n)
    ws.reserve(tp2.workspaceSize)
    let (_, wide) = tp2.track(ref0, x, ws)
    doAssert not wide.fundamentalCut
    doAssert wide.lastFundamentalMs > 11600, $wide.lastFundamentalMs
    var checked = 0
    for m in 0 ..< wide.len:
      if wide.amplitude[0][m].isNaN or wide.timeMs[m] < 1000 or wide.timeMs[m] > 11400: continue
      doAssert abs(wide.amplitude[0][m] / 1000 - 1) < 0.005 and
        abs(wide.phaseDeg[m] - 30) < 0.3, $wide.timeMs[m] & ": " & $wide.phaseDeg[m]
      inc checked
    doAssert checked > 1000
    # At 1 ms the same sweep is far from the limit.
    let ch1 = sweepOf(1, 6, 220)
    let tp1 = initSweepTrackPlan(float64, TrackParams(), 1, 12001)
    ws.reserve(tp1.workspaceSize)
    let (_, ok) = tp1.track(chirp(ch1, 1, 12001, [Tone()]), chirp(ch1, 1, 12001, [Tone()]), ws)
    doAssert not ok.fundamentalCut and ok.lastFundamentalMs > 11400
    doAssertRaises(ValueError):
      discard initSweepTrackPlan(float64, TrackParams(maxFundamentalFraction: 1.5), 1, 100)
    doAssertRaises(ValueError):
      discard initSweepTrackPlan(float64, TrackParams(harmonicBandHz: -3), 1, 100)
    doAssertRaises(ValueError):
      discard initSweepTrackPlan(float64, TrackParams(harmonicBandHz: Inf), 1, 100)

  block peak_of_a_pure_fundamental:
    # A fundamental of 1000 from 0.05 to 0.8 of Nyquist: every point's peak is
    # 1000 within 3e-4 (8 taps each side with a Kaiser window of 8 read up to
    # 1.8 % low from 0.7 of Nyquist; measured now 1.3e-4).
    for dtMs in [0.5, 1.0, 2.0, 4.0]:
      let n = int(12000 / dtMs) + 1
      let nyq = 500 / dtMs
      let ch = sweepOf(dtMs, 0.05 * nyq, 0.84 * nyq)
      let x = chirp(ch, dtMs, n, [Tone(amp: 1000)])
      let tp = initSweepTrackPlan(float64, TrackParams(), dtMs, n)
      var ws = initWorkspace()
      ws.reserve(tp.workspaceSize)
      let (_, t) = tp.track(x, x, ws)
      var worst, top = 0.0
      var seen = 0
      for m in 0 ..< t.len:
        if t.peak[m].isNaN or t.timeMs[m] < 700 or t.timeMs[m] > 11300: continue
        worst = max(worst, abs(t.peak[m] / 1000 - 1))
        top = max(top, t.referenceHz[m] / nyq)
        inc seen
      doAssert seen > 800 and top > 0.77, $seen & " " & $top
      doAssert worst < 3e-4, $dtMs & ": " & $worst
    # A tone at 0.3 of Nyquist with a 3rd harmonic at 0.9: the waveform's
    # maximum 600·cos p + 400·cos 3p is 1000 (p = 0): no period reads above it,
    # and the largest period's peak reads within 0.2 %.
    block:
      let n = 12001
      let ch = Chirp(law: clLinear, f0: 150, f1: 151, lengthMs: 12000, ramp: rmNone)
      let x = chirp(ch, 1, n, [Tone(amp: 600), Tone(order: 3, amp: 400)])
      let tp = initSweepTrackPlan(float64, TrackParams(), 1, n)
      var ws = initWorkspace()
      ws.reserve(tp.workspaceSize)
      let (_, t) = tp.track(x, x, ws)
      var best = 0.0
      for m in 0 ..< t.len:
        if not t.peak[m].isNaN: best = max(best, t.peak[m])
      doAssert abs(best / 1000 - 1) < 0.002, $best

  block window_statistics:
    # A track filled by hand: 201 points every 10 ms from 0 to 2000 ms, the
    # sweep from 0 and judged throughout. Window w (500 ms: w = t div 500)
    # has |phase| 2(w + 1)° (alternating in sign), force 1000 − 100w, THD
    # 5(w + 1) %, and peak 1500 + 10w, plus 7 at the window's last point.
    let tp = initSweepTrackPlan(float64, TrackParams(), 1, 2001)
    var t = initSweepTrack(tp)
    t.extent = SweepExtent(startMs: 0, endMs: 2000, flatStartMs: 0, flatEndMs: 2000)
    for m in 0 ..< t.len:
      let w = m div 50
      t.phaseDeg[m] = (if m mod 2 == 0: 1.0 else: -1.0) * 2.0 * float64(w + 1)
      t.amplitude[0][m] = 1000.0 - 100.0 * float64(w)
      t.thdPct[m] = 5.0 * float64(w + 1)
      t.peak[m] = 1500.0 + 10.0 * float64(w) + (if m mod 50 == 49: 7.0 else: 0.0)
    block all_judged:
      let s = t.windowStats()
      doAssert s.startMs == @[0.0, 500, 1000, 1500] # the 5th, one point, is thinner than half
      doAssert s.points == @[50, 50, 50, 50]
      doAssert s.meanAbsPhaseDeg == @[2.0, 4, 6, 8]
      doAssert s.meanForce == @[1000.0, 900, 800, 700]
      doAssert s.meanThdPct == @[5.0, 10, 15, 20]
      doAssert s.peak == @[1507.0, 1517, 1527, 1537]
      let u = s.summary
      doAssert u.windows == 4
      doAssert u.meanAbsPhaseDeg == 5 and u.maxAbsPhaseDeg == 8 and u.maxAbsPhaseAtMs == 1500
      doAssert u.meanForce == 850 and u.minForce == 700 and u.minForceAtMs == 1500
      doAssert u.maxForce == 1000 and u.maxForceAtMs == 0
      doAssert u.meanThdPct == 12.5 and u.maxThdPct == 20 and u.maxThdAtMs == 1500
      doAssert u.maxPeak == 1537 and u.maxPeakAtMs == 1500
    block partial_windows:
      let s = t.windowStats(minFill = 0)
      doAssert s.startMs.len == 5 and s.points[4] == 1 and s.meanForce[4] == 600
      doAssert t.windowStats(minFill = 1).startMs == @[0.0, 500, 1000, 1500]
      # Judged from 250 to 1250 ms: windows 0 and 2 hold half, 1 all of it,
      # 3 none; a longer window and an unaligned start.
      let r = t.windowStats(VibLimits(skipStartMs: 0, fromMs: 250, toMs: 1250))
      doAssert r.startMs == @[0.0, 500, 1000] and r.points == @[25, 50, 26]
      doAssert r.meanForce == @[1000.0, 900, 800]
      doAssert abs(r.summary.meanForce - 900) < 1e-12 and r.summary.maxForceAtMs == 0
      doAssert t.windowStats(VibLimits(skipStartMs: 0, fromMs: 250, toMs: 1250),
        minFill = 0.6).startMs == @[500.0]
      let one = t.windowStats(windowMs = 1000)
      doAssert one.startMs == @[0.0, 1000] and one.meanForce == @[950.0, 750]
      doAssert one.meanAbsPhaseDeg == @[3.0, 7]
    block origin:
      # Windows from 250 ms: the first holds 25 points of window 0 and 25 of 1.
      let s = t.windowStats(originMs = 250)
      doAssert s.startMs[0] == 250 and s.startMs[1] == 750 and s.meanForce[0] == 950 and
        s.meanForce[1] == 850
    block band_harmonic_limit:
      # H2 10 % and H3 50 % at 100 Hz: with the band at 220 Hz only the 2nd is in
      # it, and a 20 % limit on any one harmonic is not exceeded; without a band
      # it is (by the 3rd).
      var u = t
      for m in 0 ..< u.len:
        u.referenceHz[m] = 100
        u.amplitude[1][m] = 100
        u.amplitude[2][m] = 500
      let lim = VibLimits(harmonicPct: 20, skipStartMs: 0)
      var flags = newSeq[set[LimitKind]](u.len)
      u.limitFlagsInto(lim, flags)
      doAssert lkHarmonic in flags[10]
      u.bandHz = 220
      u.limitFlagsInto(lim, flags)
      doAssert lkHarmonic notin flags[10]
      u.bandHz = 320
      u.limitFlagsInto(lim, flags)
      doAssert lkHarmonic in flags[10]
    block missing_values:
      var u = t
      for m in 0 ..< 25: u.thdPct[m] = NaN # half of window 0 has no THD
      for m in 50 ..< 100: u.amplitude[0][m] = NaN # window 1 is not measured
      let s = u.windowStats()
      doAssert s.startMs == @[0.0, 1000, 1500] and s.meanThdPct[0] == 5
      u.peak[10] = NaN
      doAssert u.windowStats().peak[0] == 1507
    block no_sweep_and_refusals:
      let e = initSweepTrack(tp).windowStats()
      doAssert e.startMs.len == 0 and e.summary.windows == 0 and e.summary.meanForce.isNaN
      for w in [0.0, -5.0, NaN, Inf]:
        doAssertRaises(ValueError):
          discard t.windowStats(windowMs = w)
      for f in [-0.1, 1.1, NaN]:
        doAssertRaises(ValueError):
          discard t.windowStats(minFill = f)

  block window_statistics_of_a_sweep:
    # A 12 s sweep, 20° phase, force 1000, H2 10 % at 1 ms: every 500 ms window
    # gives 20°, 1000 and the peak 1000..1100 (the fundamental plus the 2nd
    # at its largest: between 1000 and 1.1 of it).
    let dtMs = 1.0
    let n = 13001
    let ch = sweepOf(dtMs, 6, 220)
    let ref0 = chirp(ch, dtMs, n, [Tone()])
    let x = chirp(ch, dtMs, n, [Tone(amp: 1000, phaseDeg: 20), Tone(order: 2, amp: 100)])
    let tp = initSweepTrackPlan(float64, TrackParams(harmonicBandHz: edge), dtMs, n)
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    let (r, t) = tp.track(ref0, x, ws)
    let s = t.windowStats()
    doAssert s.summary.windows >= 22 and s.summary.windows <= 24, $s.summary.windows
    doAssert floorMod(s.startMs[0] - r.extent.startMs, 500.0) == 0
    doAssert s.startMs[1] - s.startMs[0] == 500
    for w in 0 ..< s.startMs.len:
      doAssert abs(s.meanAbsPhaseDeg[w] - 20) < 0.1 and abs(s.meanForce[w] - 1000) < 1,
        $s.startMs[w]
      doAssert s.peak[w] > 995 and s.peak[w] < 1101
      # H2 in band up to 110 Hz: the mean THD is 10 % there and 0 above.
      let f = t.referenceHz[int(s.startMs[w] / 10) + 25]
      if abs(2 * f - edge) > 20: doAssert abs(s.meanThdPct[w] - (if 2 * f < edge: 10.0 else: 0.0)) < 0.1
    doAssert abs(s.summary.meanAbsPhaseDeg - 20) < 0.1 and abs(s.summary.maxAbsPhaseDeg - 20) < 0.1
    doAssert s.summary.maxThdPct > 9.5 and s.summary.maxThdPct < 10.1 and
      s.summary.meanThdPct < s.summary.maxThdPct

  block trace_shorter_than_its_reference:
    # The trace ends at 8 s of a 12 s reference, zeros after: the statistics
    # over the whole judged part include the empty end (a force of 0), over
    # `toMs` set to the trace's end they are the sweep's.
    let dtMs = 1.0
    let n = 12001
    let ch = sweepOf(dtMs, 6, 220)
    let ref0 = chirp(ch, dtMs, n, [Tone()])
    var x = chirp(ch, dtMs, n, [Tone(amp: 1000)])
    for j in 8000 ..< n: x[j] = 0
    let tp = initSweepTrackPlan(float64, TrackParams(), dtMs, n)
    var ws = initWorkspace()
    ws.reserve(tp.workspaceSize)
    let (_, t) = tp.track(ref0, x, ws)
    doAssert t.summarize().minForce < 1
    let endMs = 8000.0 - 100 # half a window before the data ends
    doAssert t.summarize(VibLimits(toMs: endMs)).minForce > 990
    doAssert t.windowStats().summary.minForce < 1
    doAssert t.windowStats(VibLimits(toMs: endMs)).summary.minForce > 990

when isMainModule: main()
