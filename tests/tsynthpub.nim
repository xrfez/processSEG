## Public synthetic signals: wavelet peak and spectrum, gather arrivals,
## seeded noise, and a distorted sweep read back by the vibrator QC.
##
## Each block was seen to fail with the implementation broken, then restored:
## - wavelet_peak: `addRicker` centred one sample late (the peak left centreMs);
## - wavelet_spectrum: `ricker` used `2·freqHz` (the spectral peak moved);
## - gather_arrivals: `hyperbolicMs` dropped the offset term (arrivals went flat);
## - noise_seed: `addNoise` ignored the seed (same seed no longer matched);
## - harmonics_qc: `sweepHarmonicsInto` dropped each term's phase (the QC
##   read 0° instead of the 4° that was put in);
## - harmonics_fundamental_nyquist: `sweepHarmonics` cut an order-1 term at
##   Nyquist (the last sample went to 0);
## - live_columns_checked_first: a bad `liveCols` entry raised after the rows
##   before it were written.

import std/[assertions, math]
import parseSEG, processSEG

func near(a, b, tol: float64): bool = abs(a - b) <= tol

proc peakAt(x: openArray[float64]): int =
  for j in 1 ..< x.len:
    if abs(x[j]) > abs(x[result]): result = j

block wavelet_peak:
  # Peak time: the centre sample is the unique maximum, and it is 1.
  doAssert ricker(0, 25) == 1
  doAssert ricker(10, 25) == ricker(-10, 25)
  var x = newSeq[float64](501)
  x.addRicker(dtMs = 2, centreMs = 400, freqHz = 25, amp = 3)
  doAssert peakAt(x) == 200, "400 ms at 2 ms is sample 200, got " & $peakAt(x)
  doAssert x[200] == 3
  doAssert x[0] == 0

block wavelet_spectrum:
  # The Ricker amplitude spectrum peaks at freqHz. A 30 Hz wavelet in a
  # 1024-sample window (1 ms, no taper, no demean) must peak within one bin.
  const freq = 30.0
  let n = 1024
  var x = newSeq[float64](n)
  x.addRicker(dtMs = 1, centreMs = 512, freqHz = freq)
  let sp = initSpectrumPlan(float64, dtMs = 1.0, windowLen = n, taperFraction = 0,
    demean = false)
  var ws = initWorkspace()
  ws.reserve(sp.workspaceSize)
  var amp = newSeq[float64](sp.bins)
  var phase = newSeq[float64](sp.bins)
  sp.amplitudeInto(x, amp, phase, ws)
  var peak = 1
  for k in 2 ..< sp.bins:
    if amp[k] > amp[peak]: peak = k
  doAssert abs(sp.binHz(peak) - freq) <= sp.binHz(1),
    "spectrum peak at " & $sp.binHz(peak) & " Hz, want " & $freq

block gather_arrivals:
  # Arrival time of each row is the travel-time formula, to the sample.
  let offsets = [-600.0, -200.0, 0.0, 200.0, 600.0]
  let dtMs = 2.0
  var shot = zeros[float64](offsets.len, 1001)
  shot.addHyperbola(offsets, dtMs, t0Ms = 400, velocity = 2000, freqHz = 30)
  let rows = rowBlock(shot)
  for r, off in offsets:
    let want = hyperbolicMs(400, 2000, off)
    doAssert near(want, sqrt(400.0 ^ 2 + (abs(off) / 2000 * 1000) ^ 2), 1e-9)
    let col = peakAt(rows.row(r))
    doAssert near(col.float64 * dtMs, want, dtMs / 2),
      "hyperbola row " & $r & " at " & $(col.float64 * dtMs) & " want " & $want
  shot.addLinear(offsets, dtMs, t0Ms = 20, velocity = 1500, freqHz = 40, amp = 2)
  # The direct wave is earlier than the reflection on the far offsets, and
  # the stronger of the two at the near offsets where they do not overlap.
  for r, off in [0.0, 600.0]:
    let row = if off == 0: 2 else: 4
    let want = linearMs(20, 1500, off)
    doAssert near(want, 20 + abs(off) / 1500 * 1000, 1e-9)
    doAssert near(peakAt(rows.row(row)).float64 * dtMs, want, dtMs / 2),
      "linear row " & $row

block noise_seed:
  var a = newSeq[float64](20000)
  var b = newSeq[float64](20000)
  var c = newSeq[float64](20000)
  a.addNoise(sigma = 2, seed = 7)
  b.addNoise(sigma = 2, seed = 7)
  c.addNoise(sigma = 2, seed = 8)
  doAssert a == b, "the same seed gives the same samples"
  doAssert a != c, "another seed gives other samples"
  var s, s2 = 0.0
  for v in a:
    s += v
    s2 += v * v
  let mean = s / a.len.float64
  let sd = sqrt(s2 / a.len.float64 - mean * mean)
  doAssert abs(mean) < 0.05, "mean " & $mean
  doAssert abs(sd - 2) < 0.05, "sd " & $sd
  var again = newSeq[float32](64)
  var once = newSeq[float32](64)
  again.addNoise(1.5, 99)
  once.addNoise(1.5, 99)
  doAssert again == once

block harmonics_qc:
  # Tolerances are the guide's: phase within 0.05°, a harmonic within 0.02
  # percentage points of the level put in, away from the tapers.
  let spec = exponentialSweep(8, 48, 3000, 1, initSweepTaper(250, 250))
  let n = spec.sweepLen
  let pilot = spec.sweepSamples(float64)
  let gf = spec.sweepHarmonics(float64, [
    SweepHarmonic(phaseDeg: 4),
    SweepHarmonic(order: 2, amp: 0.08),
    SweepHarmonic(order: 3, amp: 0.03)])
  doAssert pilot == spec.sweepHarmonics(float64), "no harmonics is the sweep"
  var keys = newSeq[TraceKeys](2)
  var counts = newSeq[int](2)
  var data = zeros[float64](2, n)
  for r in 0 ..< 2:
    keys[r] = initTraceKeys()
    keys[r].t0Ms = 0
    counts[r] = n
  for j in 0 ..< n:
    data[0, j] = pilot[j]
    data[1, j] = gf[j]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 1)
  var unit = initVibUnit(0, groundForce = 1)
  let tp = initSweepTrackPlan(float64, TrackParams(harmonics: 3), dtMs = 1, samples = n)
  var ws = initWorkspace()
  ws.reserve(vibratorWorkspace(tp))
  var results: seq[UnitSignature[float64]] = @[]
  tp.analyzeVibrators(data, meta, [unit], VibLimits(), results, ws)
  let track = results[0].tracks[vsGroundForce]
  let m = 150 # 1500 ms, inside the flat part
  doAssert abs(track.phaseDeg[m] - 4) < 0.05, "phase " & $track.phaseDeg[m]
  var hd2 = newSeq[float64](track.len)
  var hd3 = newSeq[float64](track.len)
  track.distortionInto(2, hd2)
  track.distortionInto(3, hd3)
  doAssert abs(hd2[m] - 8) < 0.02, "2nd " & $hd2[m]
  doAssert abs(hd3[m] - 3) < 0.02, "3rd " & $hd3[m]
  # A dB/octave law has no closed-form phaseAt; the fundamental still matches.
  let dbo = dbPerOctaveSweep(10, 40, 2000, 2, 6, initSweepTaper(100, 100))
  doAssert dbo.sweepHarmonics(float64) == dbo.sweepSamples(float64)
  # Cut, not fold: a 5th harmonic of 10-80 Hz at 2 ms crosses Nyquist (250 Hz)
  # at 50 Hz and is zero from there on.
  let up = linearSweep(10, 80, 2000, 2, noTaper(), maxNyquistFraction = 1)
  let cut = up.sweepHarmonics(float64, [SweepHarmonic(order: 5, amp: 1)])
  var heard = 0
  for j in 0 ..< cut.len:
    let f = up.frequencyAt(float64(j) * 2)
    if 5 * f >= 250:
      doAssert cut[j] == 0
    elif abs(cut[j]) > 1e-6:
      inc heard
  doAssert heard > 100
  doAssertRaises(ValueError):
    discard up.sweepHarmonics(float64, [SweepHarmonic(order: 0)])

block harmonics_fundamental_nyquist:
  # An explicit order-1 term is the fundamental, which a valid spec keeps at or
  # below Nyquist, so it is not cut there (only orders above 1 are). Up to
  # Nyquist exactly, it equals the empty list sample for sample.
  let spec = linearSweep(10, 250, 1002, 2, noTaper(), maxNyquistFraction = 1)
  let want = spec.sweepSamples(float64)
  let got = spec.sweepHarmonics(float64, [SweepHarmonic()])
  doAssert got == want, "order 1 equals the sweep"
  doAssert got[^1] == want[^1], "last sample " & $got[^1]

block live_columns_checked_first:
  # A live count outside 0 .. cols is refused before any row is written: row 0
  # (valid) must keep its prefilled samples when row 1 is refused.
  var g = zeros[float64](3, 80)
  for j in 0 ..< 3 * 80: g[j div 80, j mod 80] = 7
  doAssertRaises(ValueError):
    g.addLinear([0.0, 100.0, 200.0], 2, 0, 1500, 30, liveCols = [80, 81, 80])
  for r in 0 ..< 3:
    for j in 0 ..< 80:
      doAssert g[r, j] == 7, "row " & $r & " col " & $j & " was written"
