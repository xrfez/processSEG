## Vibroseis correlation (tasks 10.3b and 10.4): FFT against direct
## correlation, reflectivity recovered, normalization, deconvolution, the
## pilot's sub-sample start time, the instrument filter's phase, rows,
## allocation and the pool.

import std/[assertions, math, complex]
import parseSEG, processSEG
import utils/[synth, allocs]

proc direct(x: seq[float64], p: seq[float64], m: int): float64 =
  ## Σ x[j + m]·p[j] over the overlap.
  for j in 0 ..< p.len:
    let i = j + m
    if i >= 0 and i < x.len: result += x[i] * p[j]

proc convolve(refl: seq[float64], p: seq[float64], n: int): seq[float64] =
  ## The record of reflectivity `refl` (spikes) with the pilot, n samples.
  result = newSeq[float64](n)
  for i, r in refl:
    if r != 0:
      for j in 0 ..< p.len:
        if i + j < n: result[i + j] += r * p[j]

proc metaFor(rows, n: int, dtMs: float64, keys: var seq[TraceKeys],
    counts: var seq[int]): ChunkMeta =
  keys = newSeq[TraceKeys](rows)
  counts = newSeq[int](rows)
  for r in 0 ..< rows:
    keys[r] = initTraceKeys()
    keys[r].status = tsLive
    keys[r].t0Ms = 0
    counts[r] = n
  initChunkMeta(rowsOf(keys, counts), dtMs)

block fft_equals_direct:
  let spec = linearSweep(8, 60, 1000, 2, initSweepTaper(100, 100))
  let pilot = generatePilot(spec)
  let n = 1201
  var x = newSeq[float64](n)
  x.addNoise(1.0, 7)
  let plan = initCorrelationPlan(float64, pilot, n, 2, normalize = false)
  doAssert plan.outSamples == n - pilot.len + 1
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var y = newSeq[float64](plan.outSamples)
  plan.correlateInto(x, y, ws)
  var worst, peak = 0.0
  for m in 0 ..< y.len:
    let d = direct(x, pilot.samples, m)
    worst = max(worst, abs(y[m] - d))
    peak = max(peak, abs(d))
  doAssert worst < 1e-10 * peak, "FFT vs direct: " & $worst
  # Lags into the record's end correlate with zeros, not its start.
  let long = initCorrelationPlan(float64, pilot, n, 2, outSamples = n, normalize = false)
  var y2 = newSeq[float64](n)
  ws.reserve(long.workspaceSize)
  long.correlateInto(x, y2, ws)
  for m in [n - pilot.len, n - 100, n - 1]:
    doAssert abs(y2[m] - direct(x, pilot.samples, m)) < 1e-10 * peak

block whole_sample_pilot_start_times:
  # Pilots starting before or after the time break by whole samples: the
  # output at column m is the direct correlation at lag m + t0/dt.
  let base = generatePilot(linearSweep(8, 60, 1000, 2, initSweepTaper(100, 100)))
  let n = 1801
  var x = newSeq[float64](n)
  x.addNoise(1.0, 11)
  for t0 in [-20.0, 20.0, -900.0, 3000.0]:
    var p = base
    p.t0Ms = t0
    let plan = initCorrelationPlan(float64, p, n, 2, outSamples = n, normalize = false)
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var y = newSeq[float64](n)
    plan.correlateInto(x, y, ws)
    let s = int(t0 / 2)
    var worst, peak = 0.0
    for m in 0 ..< n:
      let lag = m + s
      let d = if lag <= -p.len or lag >= n: 0.0 else: direct(x, p.samples, lag)
      worst = max(worst, abs(y[m] - d))
      peak = max(peak, abs(d))
    doAssert worst < 1e-10 * peak, "t0 " & $t0 & ": " & $worst
  # A start time no lag can reach is refused (and never sizes a huge FFT).
  var far = base
  far.t0Ms = 1e30
  doAssertRaises(ValueError): discard initCorrelationPlan(float64, far, n, 2)
  far.t0Ms = 2e7
  doAssertRaises(ValueError): discard initCorrelationPlan(float64, far, n, 2)

block reflectivity_to_klauder:
  # Spikes convolved with the sweep correlate back to spikes ⊛ Klauder at
  # their times, with the normalized Klauder peak = the spike's amplitude.
  let pilot = generatePilot(linearSweep(10, 80, 4000, 2, initSweepTaper(200, 200)))
  let n = 3501
  var refl = newSeq[float64](n)
  refl[100] = 1.0
  refl[400] = -0.5
  refl[900] = 0.25
  let x = convolve(refl, pilot.samples, n)
  let plan = initCorrelationPlan(float64, pilot, n, 2)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var y = newSeq[float64](plan.outSamples)
  plan.correlateInto(x, y, ws)
  doAssert abs(y[100] - 1.0) < 1e-3 and abs(y[400] + 0.5) < 1e-3 and abs(y[900] - 0.25) < 1e-3
  # The peaks are where the spikes are.
  for (at, amp) in [(100, 1.0), (400, -0.5), (900, 0.25)]:
    for d in [-3, -2, -1, 1, 2, 3]:
      doAssert abs(y[at + d]) < abs(amp)

block normalization_is_independent_of_the_pilot:
  # The same reflector through two pilots of different amplitude and length.
  let n = 6001
  var refl = newSeq[float64](n)
  refl[300] = 0.7
  for (amp, lengthMs) in [(1.0, 4000.0), (3.5, 8000.0)]:
    let pilot = generatePilot(linearSweep(10, 80, lengthMs, 2, initSweepTaper(200, 200),
      amplitude = amp))
    let x = convolve(refl, pilot.samples, n)
    let plan = initCorrelationPlan(float64, pilot, n, 2)
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var y = newSeq[float64](plan.outSamples)
    plan.correlateInto(x, y, ws)
    doAssert abs(y[300] - 0.7) < 1e-6 * 0.7, $y[300]
    var energy = 0.0
    for v in pilot.samples: energy += v * v
    doAssert abs(plan.scale * energy - 1) < 1e-9

block deconvolution_flattens_the_spectrum:
  let pilot = generatePilot(dbPerOctaveSweep(8, 90, 6000, 2, 4, initSweepTaper(200, 200)))
  let n = 4001
  var refl = newSeq[float64](n)
  refl[1000] = 1.0
  let x = convolve(refl, pilot.samples, n)
  var spectra: array[CorrelationMode, seq[float64]]
  for mode in CorrelationMode:
    let plan = initCorrelationPlan(float64, pilot, n, 2, outSamples = 2000, mode = mode,
      epsPercent = 0.1)
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var y = newSeq[float64](plan.outSamples)
    plan.correlateInto(x, y, ws)
    doAssert abs(y[1000] - 1) < 1e-6, $mode & ": peak " & $y[1000]
    # The wavelet's amplitude spectrum (512 samples around the spike).
    let fp = initFftPlan(float64, 512)
    var w = initWorkspace()
    w.reserve(fp.workspaceSize(batched = false))
    var re, im = newSeq[float64](fp.bins)
    fp.forward(y.toOpenArray(744, 1255), re, im, w)
    spectra[mode] = newSeq[float64](fp.bins)
    for k in 0 ..< fp.bins: spectra[mode][k] = 20 * log10(hypot(re[k], im[k]))
  # In the band, away from the tapers (bins of 1000/(512·2) Hz: 15–80 Hz).
  proc spread(s: seq[float64]): float64 =
    var lo = Inf
    var hi = -Inf
    for k in 16 .. 82:
      lo = min(lo, s[k])
      hi = max(hi, s[k])
    hi - lo
  doAssert spread(spectra[cmDeconvolve]) < 1.0, "decon: " & $spread(spectra[cmDeconvolve])
  # Correlation keeps |P|²: +8 dB/octave over 15–80 Hz, about 19 dB.
  doAssert spread(spectra[cmCorrelate]) > 15
  # A large ε tends to correlation's shape.
  let big = initCorrelationPlan(float64, pilot, n, 2, outSamples = 2000,
    mode = cmDeconvolve, epsPercent = 1e6)
  let corr = initCorrelationPlan(float64, pilot, n, 2, outSamples = 2000)
  var ws = initWorkspace()
  ws.reserve(big.workspaceSize)
  var a, b = newSeq[float64](2000)
  big.correlateInto(x, a, ws)
  corr.correlateInto(x, b, ws)
  for m in 0 ..< 2000: doAssert abs(a[m] - b[m]) < 1e-4

block sub_sample_pilot_start:
  # A recorded pilot whose first sample is 0.6 ms (0.3 samples) after the
  # time break: correlation removes that start time, so an arrival of the
  # sweep at 400 ms reads at sample 200, not 200.3.
  let dt = 2.0
  let spec = linearSweep(10, 70, 3000, dt, initSweepTaper(200, 200))
  proc sweepAt(t: float64): float64 =
    if t < 0 or t > 3000: return 0
    sin(spec.phaseAt(t)) * taperWeight(tpCos2, min(t / 200, (3000 - t) / 200))
  let n = 2501
  var x = newSeq[float64](n)
  for i in 0 ..< n: x[i] = sweepAt(float64(i) * dt - 400)
  var late = newSeq[float64](spec.sweepLen)
  for j in 0 ..< late.len: late[j] = sweepAt(0.6 + float64(j) * dt)
  proc arrival(pilot: Pilot[float64]): float64 =
    let plan = initCorrelationPlan(float64, pilot, n, dt)
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var y = newSeq[float64](plan.outSamples)
    plan.correlateInto(x, y, ws)
    var best = 1
    for m in 1 ..< y.len - 1:
      if y[m] > y[best]: best = m
    float64(best) + 0.5 * (y[best - 1] - y[best + 1]) /
      (y[best - 1] - 2 * y[best] + y[best + 1])
  let at = arrival(initPilot(late, dt, t0Ms = 0.6))
  let naive = arrival(initPilot(late, dt))
  doAssert abs(at - 200) < 0.02, "with the start time: " & $at
  doAssert abs(naive - 200.3) < 0.02, "without it: " & $naive

block instrument_filter_phase:
  # Data recorded through a minimum-phase anti-alias filter: correlating with
  # the filtered synthetic pilot gives a zero-phase wavelet; with the raw
  # pilot, the filter's phase remains.
  let dt = 2.0
  let spec = linearSweep(8, 80, 4000, dt, initSweepTaper(200, 200))
  let raw = generatePilot(spec)
  let inst = InstrumentFilter(highCutHz: 100, highSlopeDbPerOctave: 36,
    lowCutHz: 3, lowSlopeDbPerOctave: 36, phase: ipMinimumPhase)
  let filtered = raw.applyInstrumentFilter(inst)
  let n = 4001
  var refl = newSeq[float64](n)
  refl[1000] = 1
  # The record through the instrument, simulated independently of
  # applyInstrumentFilter: the reflected sweep run through the minimum-phase
  # IIR cascades directly, from rest (zeros ahead of the record).
  var rec = initPilot(convolve(refl, raw.samples, n), dt)
  block:
    const lead = 2000
    var buf = newSeq[float64](lead + n)
    for j in 0 ..< n: buf[lead + j] = rec.samples[j]
    var ws = initWorkspace()
    for spec in [butterLowCut(3, 36), butterHighCut(100, 36)]:
      let plan = initIirPlan(float64, spec, dt, buf.len, ipMinimum, trNone)
      ws.reserve(plan.workspaceSize)
      buf.applyFilter(plan, ws)
    for j in 0 ..< n: rec.samples[j] = buf[lead + j]
  proc phaseErrorDeg(pilot: Pilot[float64]): float64 =
    let plan = initCorrelationPlan(float64, pilot, n, dt, outSamples = 2001)
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var y = newSeq[float64](2001)
    plan.correlateInto(rec.samples, y, ws)
    # The phase of the wavelet centred on sample 1000, 15–70 Hz.
    let fp = initFftPlan(float64, 1000)
    var w = initWorkspace()
    w.reserve(fp.workspaceSize(batched = false))
    # buf[0] is sample 1000 (lag 0), negative lags wrap to the end.
    var buf = newSeq[float64](1000)
    for i in 0 ..< 1000:
      let m = 1000 + (if i < 500: i else: i - 1000)
      buf[i] = y[m]
    var re, im = newSeq[float64](fp.bins)
    fp.forward(buf, re, im, w)
    for k in 0 ..< fp.bins:
      let f = float64(k) * 1000 / (1000 * dt)
      if f >= 15 and f <= 70:
        result = max(result, abs(radToDeg(arctan2(im[k], re[k]))))
  let good = phaseErrorDeg(filtered)
  let bad = phaseErrorDeg(raw)
  doAssert good < 1.0, "with the instrument filter: " & $good & "°"
  doAssert bad > 20, "without it the minimum-phase filter remains: " & $bad & "°"

block rows_meta_float32_and_allocation:
  let pilot = generatePilot(linearSweep(10, 70, 2000, 2, initSweepTaper(100, 100)), float32)
  let n = 1601
  let rows = 11
  var keys: seq[TraceKeys]
  var counts: seq[int]
  let meta = metaFor(rows, n, 2, keys, counts)
  var refl = newSeq[float64](n)
  refl[50] = 1
  let rec = convolve(refl, generatePilot(linearSweep(10, 70, 2000, 2,
    initSweepTaper(100, 100))).samples, n)
  var src = zeros[float32](rows, n)
  for r in 0 ..< rows:
    for j in 0 ..< n: src[r, j] = float32(rec[j] * float64(r + 1))
  keys[3].status = tsDead
  keys[4].status = tsAuxiliary
  counts[5] = 1300 # a shorter record: what is past its live length is ignored
  for j in 1300 ..< n: src[5, j] = 1e6
  let meta2 = initChunkMeta(rowsOf(keys, counts), 2)
  let plan = initCorrelationPlan(float32, pilot, n, 2)
  var dst = zeros[float32](rows, plan.outSamples)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  plan.correlateInto(src, dst, meta2, ws)
  for r in 0 ..< rows:
    if r == 3:
      for m in 0 ..< plan.outSamples: doAssert dst[r, m] == 0
    else:
      doAssert abs(float64(dst[r, 50]) - float64(r + 1)) < 2e-3 * float64(r + 1), $r
  # No allocation per call once reserved.
  let allocs = allocsIn:
    for i in 0 ..< 3: plan.correlateInto(src, dst, meta2, ws)
  doAssert allocs == 0, $allocs & " allocations"
  # The pool gives the same result.
  var pool = newProcessPool(3)
  var pd = zeros[float32](rows, plan.outSamples)
  pool.correlateInto(plan, src, pd, meta2)
  doAssert pd == dst
  # The allocating form.
  doAssert plan.correlate(src, meta2) == dst
  discard meta

block validation:
  let pilot = generatePilot(linearSweep(10, 70, 2000, 2))
  doAssertRaises(ValueError): discard initCorrelationPlan(float32, pilot, 1601, 1)
  doAssertRaises(ValueError): discard initCorrelationPlan(float32, pilot, 500, 2)
  discard initCorrelationPlan(float32, pilot, 500, 2, outSamples = 500)
  doAssertRaises(ValueError): discard initCorrelationPlan(float32, pilot, 500, 2, outSamples = 501)
  doAssertRaises(ValueError):
    discard initCorrelationPlan(float32, pilot, 1601, 2, mode = cmDeconvolve, epsPercent = 0)
  let plan = initCorrelationPlan(float32, pilot, 1601, 2)
  var keys: seq[TraceKeys]
  var counts: seq[int]
  let meta = metaFor(2, 1601, 2, keys, counts)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var src = zeros[float32](2, 1601)
  var dst = zeros[float32](2, 600)
  doAssertRaises(ValueError): plan.correlateInto(src, dst, meta, ws)
  var wide = zeros[float32](2, 1700)
  var dst2 = zeros[float32](2, plan.outSamples)
  doAssertRaises(ValueError): plan.correlateInto(wide, dst2, meta, ws)
  let m4 = initChunkMeta(rowsOf(keys, counts), 4)
  doAssertRaises(ValueError): plan.correlateInto(src, dst2, m4, ws)
