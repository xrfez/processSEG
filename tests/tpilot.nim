## Pilots (tasks 10.3 and 10.3b): generated and recorded pilots, the SEG-Y
## sweep fields round trip through parseSEG, resampling and instrument
## filters.

import std/[assertions, math, os]
import parseSEG, processSEG
import processSEG/vibroseis/kaiser

block generated:
  let spec = linearSweep(8, 80, 8000, 2, initSweepTaper(250, 300, tpLinear))
  let p = generatePilot(spec, float32)
  doAssert p.len == 4001 and p.dtMs == 2 and p.t0Ms == 0 and p.lengthMs == 8000
  doAssert p.sweep.isSome and p.sweep.get.f1Hz == 8
  doAssert p.samples == spec.sweepSamples(float32)

block validation:
  doAssertRaises(ValueError): discard initPilot(@[1.0], 2)
  doAssertRaises(ValueError): discard initPilot(@[0.0, 0.0, 0.0], 2)
  doAssertRaises(ValueError): discard initPilot(@[1.0, NaN], 2)
  doAssertRaises(ValueError): discard initPilot(@[1.0, 1.0], 0)
  doAssertRaises(ValueError): discard initPilot(@[1.0, 1.0], 2, t0Ms = NaN)
  let s = linearSweep(8, 80, 100, 2)
  doAssertRaises(ValueError): discard initPilot(@[1.0, 1.0], 1, sweep = some(s))
  doAssert initPilot(@[1.0, -1.0], 2, 0.6).t0Ms == 0.6

block sweep_overrides:
  let spec = linearSweep(8, 80, 12000, 2, initSweepTaper(250, 500, tpCos2))
  let o = spec.sweepOverrides
  doAssert o.len == 7 and o[0].name == "trace.sweepFreqStart"
  doAssert o[0].value.toFloat == 8 and o[1].value.toFloat == 80
  doAssert o[2].value.toFloat == 12000 and o[3].value.toFloat == 1
  doAssert o[4].value.toFloat == 250 and o[5].value.toFloat == 500 and o[6].value.toFloat == 2
  doAssert spec.sweepOverrides(binary = true)[3].name == "binary.sweepType"
  # A sweep over 32.767 s does not fit the int16 length field.
  doAssertRaises(ValueError): discard linearSweep(8, 80, 40000, 2).sweepOverrides
  # The fields hold whole Hz and ms. A 7.5 Hz start written as 8 would be
  # read back as the pilot's sweep, a wrong one: refused, not rounded.
  doAssertRaises(ValueError): discard linearSweep(7.5, 80, 12000, 2).sweepOverrides
  doAssertRaises(ValueError): discard linearSweep(8, 79.6, 12000, 2).sweepOverrides
  doAssertRaises(ValueError):
    discard linearSweep(8, 80, 12000.5, 0.5, maxNyquistFraction = 0.1).sweepOverrides
  doAssertRaises(ValueError):
    discard linearSweep(8, 80, 12000, 2, initSweepTaper(250.5, 0)).sweepOverrides

block round_trip_through_segy:
  # Written with parseSEG from pilotKeys and sweepOverrides, read back as a pilot.
  let spec = linearSweep(6, 90, 6000, 2, initSweepTaper(200, 400, tpLinear))
  let p = generatePilot(spec, float32)
  let path = getTempDir() / "processseg_tpilot.sgy"
  block:
    var w = openSeismicWriter(path, SegyOutputOptions())
    w.writeTrace(p.pilotKeys, p.samples, overrides = spec.sweepOverrides)
    doAssert not w.close().isFatal
  var f = openSeismic(path, keys = true)
  doAssert f.traceKeys(0).status == tsAuxiliary and f.traceKeys(0).traceCode == 6
  let back = pilotFromTrace(f, 0, float32)
  doAssert back.samples == p.samples and back.dtMs == 2 and back.t0Ms == 0
  doAssert back.sweep.isSome
  let s = back.sweep.get
  doAssert s.law == slLinear and s.f1Hz == 6 and s.f2Hz == 90 and s.lengthMs == 6000
  doAssert s.taper.startMs == 200 and s.taper.endMs == 400 and s.taper.shape == tpLinear
  # The same through a meta, as a viewer holds the trace (no sweep fields).
  let data = f.readTraces(0 .. 0, float32)
  let meta = chunkMeta(f, 0 .. 0)
  let r = pilotFromRow(meta, data, 0)
  doAssert r.samples == p.samples and r.dtMs == 2 and r.t0Ms == 0 and r.sweep.isNone
  doAssertRaises(ValueError): discard pilotFromRow(meta, data, 1)
  doAssertRaises(ValueError): discard pilotFromTrace(f, 1)
  f.close()
  # Other sweep types give no sweep from the headers.
  doAssert sweepFromHeaders(8, 80, 8000, 4, 0, 0, 1, 2).isNone
  doAssert sweepFromHeaders(8, 80, 8000, 3, 0, 0, 3, 2).get.law == slExponential
  removeFile(path)

const sweepFile = currentSourcePath().parentDir.parentDir / "exampleFiles" / "Sweep0.sgy"

block headers_that_do_not_fit_the_trace:
  # Fields no trace at that interval can hold are not its description.
  doAssert sweepFromHeaders(80, 1100, 12000, 1, 500, 300, 5, 1).isNone # above Nyquist
  doAssert sweepFromHeaders(8, 110, 12000, 1, 500, 300, 5, 1).isSome
  doAssert sweepFromHeaders(8, 110, 12000, 1, 500, 300, 5, 1, samples = 12001).isSome
  doAssert sweepFromHeaders(8, 110, 12000, 1, 500, 300, 5, 1, samples = 16001).isSome
  doAssert sweepFromHeaders(8, 110, 12000, 1, 500, 300, 5, 1, samples = 6001).isNone
  doAssert sweepFromHeaders(8, 110, 12000, 1, 9000, 9000, 1, 1).isNone # tapers overlap
  doAssert sweepFromHeaders(8, 110, 12000.5, 1, 0, 0, 1, 1).isNone # not whole samples
  doAssert sweepFromHeaders(8, 110, 12000, 1, 0, 0, 1, NaN).isNone
  # Frequencies ten times the sweep's (a controller that writes 0.1 Hz)
  # pass every header check at 0.25 ms; the samples say otherwise.
  let spec = linearSweep(8, 80, 4000, 0.25, initSweepTaper(200, 200),
    maxNyquistFraction = 0.1)
  let p = generatePilot(spec, float32)
  doAssert abs(p.energyAbove(44) - 0.5) < 0.01 and p.energyAbove(400) < 1e-6
  doAssert p.energyAbove(-1) == 1 and p.energyAbove(2000) == 0
  let path = getTempDir() / "processseg_tpilot_tenths.sgy"
  for (scale, described) in [(1.0, true), (10.0, false)]:
    var o = spec.sweepOverrides
    o[0] = headerOverride("trace.sweepFreqStart", int16(8 * scale))
    o[1] = headerOverride("trace.sweepFreqEnd", int16(80 * scale))
    block:
      var w = openSeismicWriter(path, SegyOutputOptions())
      w.writeTrace(p.pilotKeys, p.samples, overrides = o)
      doAssert not w.close().isFatal
    var f = openSeismic(path, keys = true)
    let back = pilotFromTrace(f, 0, float32)
    doAssert back.samples == p.samples and back.sweep.isSome == described, $scale
    f.close()
  removeFile(path)

block the_example_pilot:
  # A controller's pilot: 8–110 Hz over 12 s at 1 ms, with 80 and 1100 in
  # its header's frequency fields.
  if not fileExists(sweepFile):
    echo "  skipped: Sweep0.sgy is not present"
  else:
    var f = openSeismic(sweepFile, keys = true)
    let p = pilotFromTrace(f, 0)
    f.close()
    doAssert p.len == 12001 and p.dtMs == 1 and p.sweep.isNone
    doAssert p.energyAbove(8) > 0.99 and p.energyAbove(110) < 0.01
    doAssert p.resamplePilot(2).len == 6001

block from_row_window_and_t0:
  var keys = @[initTraceKeys()]
  keys[0].t0Ms = -1.5
  keys[0].intervalUs = 2000
  let counts = @[6]
  let failed: seq[int] = @[]
  var data = zeros[float64](1, 8)
  for j in 0 ..< 6: data[0, j] = float64(j + 1)
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2, firstSample = 1, failed = failed)
  let p = pilotFromRow(meta, data, 0)
  # Column 0 is trace sample 1: t0 = −1.5 + 2; the live length is 6 − 1.
  doAssert p.t0Ms == 0.5 and p.len == 5
  doAssert pilotFromRow(meta, data, 0, t0Ms = 3).t0Ms == 3

proc maxDiff(a, b: seq[float64], skip: int): float64 =
  for j in skip ..< min(a.len, b.len) - skip: result = max(result, abs(a[j] - b[j]))

block resample_down_and_up:
  # 0.5 ms → 2 ms, against the sweep generated at 2 ms; the sweep stays
  # below 0.8 · 250 Hz, the cutoff is 0.9 · 250 Hz.
  let taper = initSweepTaper(200, 200)
  let fine = generatePilot(linearSweep(8, 160, 4000, 0.5, taper, maxNyquistFraction = 0.2))
  let coarse = fine.resamplePilot(2)
  let want = generatePilot(linearSweep(8, 160, 4000, 2, taper))
  doAssert coarse.len == want.len and coarse.dtMs == 2 and coarse.sweep.get.dtMs == 2
  let down = maxDiff(coarse.samples, want.samples, 0)
  doAssert down < 2e-3, "downsampled pilot differs by " & $down
  # 2 ms → 0.5 ms.
  let up = want.resamplePilot(0.5)
  let wantFine = generatePilot(linearSweep(8, 160, 4000, 0.5, taper, maxNyquistFraction = 0.2))
  doAssert up.len == wantFine.len
  let upErr = maxDiff(up.samples, wantFine.samples, 0)
  doAssert upErr < 2e-3, "upsampled pilot differs by " & $upErr
  # A pilot with energy above the new limit is refused, whatever it says of
  # its sweep (160 Hz > 0.9 · 125), unless cutting it is what is asked for.
  doAssertRaises(ValueError): discard fine.resamplePilot(4)
  var anon = fine
  anon.sweep = none(SweepSpec)
  doAssertRaises(ValueError): discard anon.resamplePilot(4)
  let cut = fine.resamplePilot(4, bandLimit = true)
  doAssert cut.len == 1001 and cut.sweep.isNone # no longer the sweep it was
  # A description the samples contradict does not stop the resampling.
  var wrong = fine
  wrong.sweep = some(linearSweep(80, 800, 4000, 0.5, taper, maxNyquistFraction = 1))
  let ok = wrong.resamplePilot(2)
  doAssert ok.samples == coarse.samples and ok.sweep.isNone
  # A length that is no whole number of the new samples: resampled, undescribed.
  let odd = fine.resamplePilot(3, bandLimit = true)
  doAssert odd.len == 1334 and odd.sweep.isNone
  doAssertRaises(ValueError): discard fine.resamplePilot(0)
  doAssertRaises(ValueError): discard anon.resamplePilot(1e-9) # over 2³¹ samples
  doAssertRaises(ValueError): discard fine.resamplePilot(2, cutoffFraction = 0)

proc bandAmplitude(x: openArray[float64], fHz, dtMs: float64): float64 =
  ## The RMS spectral amplitude of `x` over `fHz ± 1` Hz (a direct DFT).
  var e = 0.0
  for k in -10 .. 10:
    var re, im = 0.0
    let w = 2 * PI * (fHz + 0.1 * float64(k)) * dtMs / 1000
    for j, v in x:
      re += v * cos(w * float64(j))
      im += v * sin(w * float64(j))
    e += re * re + im * im
  sqrt(e / 21)

block resampling_keeps_its_passband_flat:
  # The kernel's passband ends a transition width below the cutoff: with the
  # defaults at 0.9 − 2.736 / 32 = 0.8145 of the lower Nyquist, 203.6 Hz at
  # 2 ms. A sweep is taken up to there and comes out unattenuated; one that
  # runs into the transition band (it used to lose 0.45 dB at 210 Hz and
  # 1.3 dB at 215) is refused.
  let taper = initSweepTaper(300, 300)
  let fine = generatePilot(linearSweep(10, 200, 8000, 0.5, taper, maxNyquistFraction = 0.25))
  let res = fine.resamplePilot(2)
  let want = generatePilot(linearSweep(10, 200, 8000, 2, taper))
  doAssert res.len == want.len
  doAssert abs(res.sweep.get.maxNyquistFraction - (0.9 - sqrt(1 + (8 / PI) ^ 2) / 32)) < 1e-12
  for f in [50.0, 150.0, 190.0, 195.0, 199.0]:
    let gain = bandAmplitude(res.samples, f, 2) / bandAmplitude(want.samples, f, 2)
    doAssert abs(20 * log10(gain)) < 0.01, $f & " Hz: " & $(20 * log10(gain)) & " dB"
  doAssert maxDiff(res.samples, want.samples, 0) < 2e-3
  for top in [210.0, 220.0]:
    let over = generatePilot(linearSweep(10, top, 8000, 0.5, taper, maxNyquistFraction = 0.25))
    doAssertRaises(ValueError): discard over.resamplePilot(2)
  # A sweep generated with the default limit (0.8 of Nyquist) at the new
  # interval fits the default passband, down and up.
  discard generatePilot(linearSweep(10, 200, 8000, 1, taper, maxNyquistFraction = 0.5))
    .resamplePilot(2)
  discard generatePilot(linearSweep(10, 100, 8000, 4, taper)).resamplePilot(1)
  # A window too short to have a passband.
  doAssertRaises(ValueError): discard fine.resamplePilot(2, halfWidth = 3)
  doAssertRaises(ValueError): discard fine.resamplePilot(2, halfWidth = 0)
  # At the pilot's ends part of the kernel has no samples under it: the
  # rest is rescaled to the whole (an untapered cosine start read 0.61).
  let step = generatePilot(linearSweep(10, 100, 2000, 0.5, maxNyquistFraction = 0.25,
    startPhaseDeg = 90))
  let ends = step.resamplePilot(2)
  doAssert abs(ends.samples[0] - 1) < 0.02, $ends.samples[0]
  doAssert abs(ends.samples[1] - step.samples[4]) < 0.02, $ends.samples[1]

proc interpolated(p: Pilot[float64], dtMs: float64, halfWidth = 32): seq[float64] =
  ## `resamplePilot`'s definition, tap by tap: the Kaiser-windowed sinc at
  ## 0.9 of the lower Nyquist, each output divided by the sum of its taps.
  let coarse = max(dtMs, p.dtMs)
  let fc = 0.9 * 500 / coarse
  let w = float64(halfWidth) * coarse
  for j in 0 .. int(floor(p.lengthMs / dtMs * (1 + 1e-9))):
    let t = float64(j) * dtMs
    var acc, sum = 0.0
    for i, v in p.samples:
      let u = (float64(i) * p.dtMs - t) / w
      if abs(u) < 1:
        let x = 2 * fc / 1000 * (float64(i) * p.dtMs - t)
        let h = (if x == 0: 1.0 else: sin(PI * x) / (PI * x)) *
          besselI0(8 * sqrt(1 - u * u)) / besselI0(8.0)
        acc += v * h
        sum += h
    result.add acc / sum

block resampling_by_tabulated_kernels:
  # Intervals in a ratio of small whole numbers share their kernels (one
  # per phase); any other ratio computes one per output. Both are the
  # definition, to rounding, ends included.
  let p = generatePilot(linearSweep(5, 60, 1000, 2, startPhaseDeg = 40))
  for dtMs in [8.0, 4.0, 3.0, 2.5, 1.0, 0.5, 0.75, 2.0 * sqrt(2.0), 0.3 * PI, 2.0]:
    let got = p.resamplePilot(dtMs, bandLimit = true)
    let want = interpolated(p, dtMs)
    doAssert got.len == want.len, $dtMs
    doAssert maxDiff(got.samples, want, 0) < 1e-12, $dtMs & ": " &
      $maxDiff(got.samples, want, 0)
  # float32 samples, a window longer than the pilot.
  let short = generatePilot(linearSweep(5, 60, 100, 2, startPhaseDeg = 40), float32)
  var wide = newSeq[float64](short.len)
  for j, v in short.samples: wide[j] = float64(v)
  let got = short.resamplePilot(4, bandLimit = true)
  let want = interpolated(initPilot(wide, 2), 4)
  for j in 0 ..< want.len: doAssert abs(float64(got.samples[j]) - want[j]) < 1e-6

block instrument_filter_causality:
  var s = newSeq[float64](400)
  s[200] = 1
  let spike = initPilot(s, 2)
  let mp = spike.applyInstrumentFilter(InstrumentFilter(highCutHz: 100,
    highSlopeDbPerOctave: 36, phase: ipMinimumPhase))
  for j in 0 ..< 200: doAssert mp.samples[j] == 0, "a minimum-phase filter is causal"
  doAssert mp.samples[200] != 0
  let zp = spike.applyInstrumentFilter(InstrumentFilter(highCutHz: 100,
    highSlopeDbPerOctave: 36, phase: ipZeroPhase))
  for k in 1 .. 30: doAssert abs(zp.samples[200 - k] - zp.samples[200 + k]) < 1e-9
  doAssert zp.len == 400 and zp.t0Ms == 0
  # A low-cut alone, and a band-pass with equal slopes.
  discard spike.applyInstrumentFilter(InstrumentFilter(lowCutHz: 3,
    lowSlopeDbPerOctave: 12, phase: ipMinimumPhase))
  discard spike.applyInstrumentFilter(InstrumentFilter(lowCutHz: 3, lowSlopeDbPerOctave: 12,
    highCutHz: 100, highSlopeDbPerOctave: 12, phase: ipZeroPhase))
  doAssertRaises(ValueError): discard spike.applyInstrumentFilter(InstrumentFilter())
  # Each cut has its own slope, as the SEG-Y header gives a recorder's: a
  # 3 Hz low-cut at 6 dB/octave with a 200 Hz alias filter at 72. The
  # filtered spike is the response: −3 dB at both corners (minimum phase),
  # flat between them.
  var long = newSeq[float64](8000)
  long[100] = 1
  let recorder = initPilot(long, 2).applyInstrumentFilter(InstrumentFilter(lowCutHz: 3,
    lowSlopeDbPerOctave: 6, highCutHz: 200, highSlopeDbPerOctave: 72, phase: ipMinimumPhase))
  proc gainDb(x: seq[float64], fHz: float64): float64 =
    var re, im = 0.0
    for j, v in x:
      re += v * cos(2 * PI * fHz * float64(j) * 0.002)
      im += v * sin(2 * PI * fHz * float64(j) * 0.002)
    10 * log10(re * re + im * im)
  doAssert abs(gainDb(recorder.samples, 3) + 3.01) < 0.05, $gainDb(recorder.samples, 3)
  doAssert abs(gainDb(recorder.samples, 200) + 3.01) < 0.05, $gainDb(recorder.samples, 200)
  doAssert abs(gainDb(recorder.samples, 50)) < 0.05, $gainDb(recorder.samples, 50)
  doAssert gainDb(recorder.samples, 0.75) < -11 and gainDb(recorder.samples, 0.75) > -13
  doAssert gainDb(recorder.samples, 240) < -18 # 72 dB/octave: well below the 6 dB side
  discard spike.applyInstrumentFilter(InstrumentFilter(lowCutHz: 3, lowSlopeDbPerOctave: 12,
    highCutHz: 100, highSlopeDbPerOctave: 24, phase: ipZeroPhase))
  doAssertRaises(ValueError): discard spike.applyInstrumentFilter(InstrumentFilter(
    highCutHz: 100, highSlopeDbPerOctave: 7, phase: ipMinimumPhase))
