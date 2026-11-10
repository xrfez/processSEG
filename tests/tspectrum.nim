## Spectra (fft/spectrum): peak frequency of a sinusoid within one bin, the
## amplitude and dB scales, the power average over live rows, and no
## allocation in a steady state.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

func near(a, b, tol: float64): bool = abs(a - b) <= tol

proc peakBin(a: openArray[float64]): int =
  for k in 1..<a.len:
    if a[k] > a[result]: result = k

proc gather(rows, cols: int, dtMs = 2.0): SynthGather[float32] =
  cmpGather[float32](cmpX = 0, offsets = newSeq[float64](rows), cols, dtMs)

block plan_validation:
  let sp = initSpectrumPlan(float32, windowLen = 501, dtMs = 2)
  doAssert sp.windowLen == 501 and sp.nfft == 512 and sp.bins == 257
  doAssert near(sp.binHz(256), 250, 1e-9), "Nyquist at 2 ms"
  doAssert near(sp.binHz(1), 1000 / (512 * 2), 1e-12)
  doAssert initSpectrumPlan(float64, 100, 4, nfft = 1000).nfft == 1000
  for bad in [0, -3]:
    doAssertRaises ValueError: discard initSpectrumPlan(float32, bad, 2)
  for bad in [0.0, -1, NaN, Inf]:
    doAssertRaises ValueError: discard initSpectrumPlan(float32, 10, bad)
  for bad in [-0.1, 0.6, NaN]:
    doAssertRaises ValueError:
      discard initSpectrumPlan(float32, 10, 2, taperFraction = bad)
  doAssertRaises ValueError: discard initSpectrumPlan(float32, 100, 2, nfft = 64)
  doAssertRaises ValueError: discard initSpectrumPlan(float32, 100, 2, nfft = 101)

block peak_within_one_bin:
  # Frequencies on and between bins, several tapers and paddings.
  var ws = initWorkspace()
  for (win, nfft, taper) in [(501, 0, 0.1), (400, 1024, 0.5), (1000, 0, 0.0),
      (257, 0, 0.25)]:
    let sp = initSpectrumPlan(float64, win, dtMs = 2, taperFraction = taper,
      nfft = nfft)
    ws.reserve(sp.workspaceSize)
    for f in [7.3, 25.0, 61.7, 180.2]:
      var x = newSeq[float64](win)
      x.addSinusoid(2, f, amp = 3)
      var amp, phase = newSeq[float64](sp.bins)
      sp.amplitudeInto(x, amp, phase, ws)
      let k = peakBin(amp)
      let df = sp.binHz(1)
      doAssert abs(sp.binHz(k) - f) <= df, "f = " & $f & " peak at " &
        $sp.binHz(k) & " Hz, bin width " & $df

block amplitude_scale_on_a_bin:
  # A sinusoid of amplitude A on bin 40 reads A, whatever the taper and padding.
  var ws = initWorkspace()
  for (taper, nfft) in [(0.0, 0), (0.1, 0), (0.5, 0), (0.5, 4096)]:
    let sp = initSpectrumPlan(float64, 1000, dtMs = 1, taperFraction = taper,
      nfft = nfft)
    ws.reserve(sp.workspaceSize)
    let f = sp.binHz(if nfft == 0: 40 else: 160) # same frequency, 40 Hz
    var x = newSeq[float64](1000)
    x.addSinusoid(1, f, amp = 2.5, phaseRad = 0.3)
    var amp, phase = newSeq[float64](sp.bins)
    sp.amplitudeInto(x, amp, phase, ws)
    let k = peakBin(amp)
    doAssert near(amp[k], 2.5, 0.01), "taper " & $taper & ": " & $amp[k]
    # sin(ωt + 0.3) = cos(ωt + 0.3 − π/2): phase −π/2 + 0.3 at t = 0 (boxcar).
    if taper == 0: doAssert near(phase[k], 0.3 - PI / 2, 1e-6), $phase[k]

block dc_and_demean:
  var ws = initWorkspace()
  var x = newSeq[float64](200)
  x.addDc(4)
  for demean in [true, false]:
    let sp = initSpectrumPlan(float64, 200, 2, taperFraction = 0,
        demean = demean)
    ws.reserve(sp.workspaceSize)
    var amp, phase = newSeq[float64](sp.bins)
    sp.amplitudeInto(x, amp, phase, ws)
    doAssert near(amp[0], (if demean: 0.0 else: 4.0), 1e-9), "DC is not doubled"

block db_scales:
  let amp = @[0.0, 0.1, 1.0, 10.0]
  var db = newSeq[float64](4)
  toDbInto(amp, db)
  doAssert db[3] == 0 and near(db[2], -20, 1e-12) and near(db[1], -40, 1e-12)
  doAssert db[0] == -120, "zero is clamped to the floor"
  toDbInto(amp, db, dbAbsolute, floorDb = -30)
  doAssert near(db[3], 20, 1e-12) and db[2] == 0 and db[1] == -20 and db[0] == -30
  toDbInto([0.0, 0.0], db.toOpenArray(0, 1))
  doAssert db[0] == -120 and db[1] == -120

block amplitude_phase_kernel:
  var amp, phase = newSeq[float32](2)
  amplitudePhaseInto([3'f32, 0], [4'f32, -2], amp, phase)
  doAssert amp == @[5'f32, 2] and near(phase[1], -PI / 2, 1e-6)

block average_over_live_rows:
  # Rows 0..5 carry 20 Hz at amplitudes 1..6 (row r: r + 1); row 2 is dead,
  # row 4 failed, row 5 is live for only 100 samples, which is before the
  # window. The power average of the rest (amplitudes 1, 2, 4) is sqrt(21/3).
  var g = gather(rows = 6, cols = 600)
  for r in 0..<6:
    var row = newSeq[float32](600)
    row.addSinusoid(2, 25.0, amp = float64(r + 1))
    for j in 0..<600: g.data[r, j] = row[j]
  g.keys[2].status = tsDead
  g.setLive(5, 100)
  let failed = @[4]
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  let sp = initSpectrumPlan(float32, 400, dtMs = 2, taperFraction = 0.5)
  var ws = initWorkspace()
  ws.reserve(sp.workspaceSize)
  var amp = newSeq[float64](sp.bins)
  let before = g.data.clone()
  let used = sp.averageAmplitudeInto(g.data, meta, firstCol = 150, amp, ws)
  doAssert used == 3, $used
  doAssert g.data == before, "the samples are not changed"
  let k = peakBin(amp)
  doAssert abs(sp.binHz(k) - 25) <= sp.binHz(1)
  # 25 Hz is bin 20 of 400 samples at 2 ms (nfft = 400), so it is on a bin.
  doAssert sp.nfft == 400 and k == 20
  doAssert near(amp[k], sqrt(21.0 / 3), 1e-3), $amp[k]
  # A subset of rows, and the single-trace form agrees with a one-row average.
  var one = newSeq[float64](sp.bins)
  doAssert sp.averageAmplitudeInto(g.data, meta, 150, one, ws, rows = 3..3) == 1
  doAssert near(one[k], 4, 1e-3)
  var a32, p32 = newSeq[float32](sp.bins)
  var row3 = newSeq[float32](400)
  for j in 0..<400: row3[j] = g.data[3, 150 + j]
  sp.amplitudeInto(row3, a32, p32, ws)
  for i in 0..<sp.bins: doAssert near(float64(a32[i]), one[i], 1e-4)
  # Nothing live: zeros.
  doAssert sp.averageAmplitudeInto(g.data, meta, 150, one, ws, rows = 2..2) == 0
  doAssert one[k] == 0
  doAssertRaises ValueError:
    discard sp.averageAmplitudeInto(g.data, meta, -1, one, ws)
  doAssertRaises AssertionDefect:
    discard sp.averageAmplitudeInto(g.data, meta, 0, one.toOpenArray(0, 3), ws)

block many_rows_and_partial_windows:
  # 19 rows (two full batches and a partial one); the window runs past the
  # live end of some rows, which are zero-padded and still count.
  var g = gather(rows = 19, cols = 300, dtMs = 4)
  for r in 0..<19:
    var row = newSeq[float32](300)
    row.addSinusoid(4, 50.0, amp = 1)
    for j in 0..<300: g.data[r, j] = row[j]
    if r mod 3 == 0: g.setLive(r, 250)
  let meta = initChunkMeta(g.rows, dtMs = 4)
  let sp = initSpectrumPlan(float32, 128, dtMs = 4)
  var ws = initWorkspace()
  ws.reserve(sp.workspaceSize)
  var amp = newSeq[float64](sp.bins)
  doAssert sp.averageAmplitudeInto(g.data, meta, 160, amp, ws) == 19
  doAssert abs(sp.binHz(peakBin(amp)) - 50) <= sp.binHz(1)
  let n = allocsIn:
    for _ in 0..<3:
      discard sp.averageAmplitudeInto(g.data, meta, 160, amp, ws)
  doAssert n == 0, $n & " allocations"

block nan_rows_are_left_out:
  # A NaN or Inf in one row's window must not blank the average.
  var g = gather(rows = 4, cols = 400)
  for r in 0..<4:
    var row = newSeq[float32](400)
    row.addSinusoid(2, 25.0, amp = 2)
    for j in 0..<400: g.data[r, j] = row[j]
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let sp = initSpectrumPlan(float32, 200, dtMs = 2)
  var ws = initWorkspace()
  ws.reserve(sp.workspaceSize)
  var amp, db = newSeq[float64](sp.bins)
  for bad in [NaN, Inf, -Inf]:
    g.data[2, 10] = bad
    doAssert sp.averageAmplitudeInto(g.data, meta, 0, amp, ws) == 3
    for a in amp: doAssert a == a and a < Inf
    doAssert near(amp[peakBin(amp)], 2, 0.02) # 25 Hz is bin 10 of 200 at 2 ms
    doAssert sp.averageAmplitudeInto(g.data, meta, 200, amp, ws) == 4,
      "outside the window, the bad sample does not matter"
  # One trace with a NaN: NaN in that trace only, and dB at the floor.
  var x = newSeq[float32](200)
  x[3] = NaN
  var a32, p32 = newSeq[float32](sp.bins)
  sp.amplitudeInto(x, a32, p32, ws)
  doAssert a32[5].isNaN
  toDbInto([NaN, 1.0, Inf, 0.1], db.toOpenArray(0, 3))
  doAssert db[0] == -120 and db[1] == 0 and db[2] == Inf and near(db[3], -20, 1e-12),
    "NaN is the floor; Inf is not the peak"

block short_windows_scale_like_full_ones:
  # Rows that end inside the window get their own taper end and Σw, so a
  # sinusoid on a bin still reads its amplitude.
  let sp = initSpectrumPlan(float64, 400, dtMs = 1, taperFraction = 0.2)
  var ws = initWorkspace()
  ws.reserve(sp.workspaceSize)
  var amp, phase = newSeq[float64](sp.bins)
  # 300 samples of a 400-sample window: 50 Hz is bin 20 of nfft 400 at 1 ms
  # and 15 whole cycles of the 300 samples.
  var x = newSeq[float64](300)
  x.addSinusoid(1, 50, amp = 3)
  sp.amplitudeInto(x, amp, phase, ws)
  doAssert peakBin(amp) == 20 and near(amp[20], 3, 0.03), $amp[20]
  # The same through the average, float64 plan (its accumulator follows the
  # window buffers in the float64 scratch), rows cut at different lengths.
  var g = cmpGather[float64](cmpX = 0, offsets = newSeq[float64](11), 500, 1)
  for r in 0..<11:
    var row = newSeq[float64](500)
    row.addSinusoid(1, 50, amp = 3)
    for j in 0..<500: g.data[r, j] = row[j]
    if r mod 2 == 1: g.setLive(r, 300 + 100 * (r mod 3)) # windows of 200 or 300
  let meta = initChunkMeta(g.rows, dtMs = 1)
  var avg = newSeq[float64](sp.bins)
  doAssert sp.averageAmplitudeInto(g.data, meta, 100, avg, ws) == 11
  doAssert peakBin(avg) == 20 and near(avg[20], 3, 0.05), $avg[20]
