## Recording filter: a reference through a typed filter with its bulk delay
## backed out. Numbers are from an independent scipy check (prewarped
## bilinear Butterworth, least-squares band delay, band phase residual).
## Seen to fail: the shift index in `applyRecordingFilter` changed by one
## sample (the exact-shift and the zero-residual blocks fail), the Chebyshev
## even-order scale dropped (the ripple block fails).

import std/[assertions, complex, math]
import parseSEG, processSEG

proc bw(fc: float64, order: int, shift = 0): RecordingFilter =
  RecordingFilter(kind: rfkMinimumPhase, family: rffButterworth, highCutHz: fc,
    highOrder: order, shiftSamples: shift)

proc phaseDeg(f: RecordingFilter, dtMs, hz: float64): float64 =
  let h = f.responseAt(dtMs, hz)
  arctan2(h.im, h.re) * 180 / PI

proc main*() =
  let spec = linearSweep(6, 220, 12000, 1, initSweepTaper(250, 250))
  let sweep = generatePilot(spec, float64)

  block none_is_bit_identical:
    let q = sweep.applyRecordingFilter(RecordingFilter())
    doAssert q.samples == sweep.samples and q.dtMs == sweep.dtMs and q.t0Ms == sweep.t0Ms
    doAssert RecordingFilter().kind == rfkNone
    doAssert RecordingFilter().shiftSamples == 0

  block length_is_exact:
    doAssert sweep.len == 12000
    for s in [0, 1, 3, -3, 105]:
      let q = sweep.applyRecordingFilter(bw(400, 6, s))
      doAssert q.len == 12000 and q.t0Ms == 0 and q.dtMs == 1, $s
    var lin = RecordingFilter(kind: rfkLinearPhase, taps: 211, highCutHz: 438,
      highSlopeDbPerOctave: 580, shiftSamples: 105)
    doAssert sweep.applyRecordingFilter(lin).len == 12000
    doAssertRaises(ValueError): discard sweep.applyRecordingFilter(bw(400, 6, 12000))
    doAssertRaises(ValueError): discard sweep.applyRecordingFilter(bw(400, 6, -12000))

  block exact_shift:
    # The shifted output is the unshifted one moved, the ring-out included.
    var s = newSeq[float64](100)
    s[97] = 1 # 2 samples before the end of a 100-sample pilot
    let p = initPilot(s, 2)
    let base = p.applyRecordingFilter(bw(100, 6))
    # The same filter on a pilot padded with zeros.
    var long = s
    long.setLen 130
    let padded = initPilot(long, 2).applyRecordingFilter(bw(100, 6))
    let shifted = p.applyRecordingFilter(bw(100, 6, 5))
    for j in 0 ..< 100: doAssert abs(shifted.samples[j] - padded.samples[j + 5]) < 1e-12
    doAssert abs(shifted.samples[95]) > 1e-3, "the ring-out is inside"
    doAssert base.samples[97] != 0
    let later = p.applyRecordingFilter(bw(100, 6, -3))
    for j in 0 ..< 3: doAssert later.samples[j] == 0
    for j in 3 ..< 100: doAssert abs(later.samples[j] - padded.samples[j - 3]) < 1e-12
    # A typed spike response comes out as the coefficients themselves.
    var sp = newSeq[float64](50)
    sp[10] = 1
    let h = @[0.5, 0.25, -0.125]
    let q = initPilot(sp, 1).applyRecordingFilter(
      RecordingFilter(kind: rfkMinimumPhase, coefficients: h, shiftSamples: 1))
    doAssert q.samples[9] == 0.5 and q.samples[10] == 0.25 and q.samples[11] == -0.125
    doAssert q.samples[8] == 0 and q.samples[12] == 0

  block butterworth_matches_instrument_filter:
    let a = sweep.applyRecordingFilter(bw(100, 6))
    let b = sweep.applyInstrumentFilter(InstrumentFilter(highCutHz: 100,
      highSlopeDbPerOctave: 36, phase: ipMinimum))
    for j in 0 ..< a.len: doAssert abs(a.samples[j] - b.samples[j]) < 1e-9
    let c = sweep.applyRecordingFilter(RecordingFilter(kind: rfkMinimumPhase,
      family: rffButterworth, lowCutHz: 3, lowOrder: 2, highCutHz: 100, highOrder: 3))
    let d = sweep.applyInstrumentFilter(InstrumentFilter(lowCutHz: 3, lowSlopeDbPerOctave: 12,
      highCutHz: 100, highSlopeDbPerOctave: 18, phase: ipMinimum))
    for j in 0 ..< a.len: doAssert abs(c.samples[j] - d.samples[j]) < 1e-9

  block bulk_delay:
    proc d(f: RecordingFilter, dt = 1.0, hi = 220.0): float64 =
      f.bulkDelaySamples(dt, 6, hi)
    doAssert abs(d(bw(400, 6)) - 0.70) < 0.02
    doAssert abs(d(bw(400, 12)) - 1.39) < 0.02
    doAssert abs(d(bw(400, 24)) - 2.77) < 0.02
    doAssert abs(d(bw(438, 12)) - 0.84) < 0.02
    doAssert abs(d(bw(207, 6), 2, 200) - 0.90) < 0.03
    doAssert suggestedShift(bw(400, 6), 1, 6, 220) == 1
    doAssert suggestedShift(bw(400, 12), 1, 6, 220) == 1
    doAssert suggestedShift(bw(400, 24), 1, 6, 220) == 3
    doAssert suggestedShift(bw(438, 12), 1, 6, 220) == 1
    doAssert suggestedShift(bw(207, 6), 2, 6, 200) == 1
    # A low-cut alone has no delay to remove; none = 0; linear is exact.
    doAssert abs(d(RecordingFilter(kind: rfkMinimumPhase, lowCutHz: 3, lowOrder: 2))) < 0.05
    doAssert d(RecordingFilter()) == 0
    doAssert d(RecordingFilter(kind: rfkLinearPhase, taps: 211, highCutHz: 438,
      highSlopeDbPerOctave: 580)) == 105
    doAssertRaises(ValueError): discard bw(400, 6).bulkDelaySamples(1, 6, 500)
    doAssertRaises(ValueError): discard bw(400, 6).bulkDelaySamples(1, 0, 100)

  block shift_residual:
    var r = bw(400, 6).shiftResidualDeg(1, 6, 220, 0)
    doAssert abs(r.meanDeg - 28.0) < 0.1 and abs(r.peakDeg - 60.1) < 0.1, $r
    r = bw(400, 6).shiftResidualDeg(1, 6, 220, 1)
    doAssert abs(r.meanDeg - 12.7) < 0.1 and abs(r.peakDeg - 19.4) < 0.1, $r
    r = bw(438, 12).shiftResidualDeg(1, 6, 220, 1)
    doAssert abs(r.meanDeg - 7.1) < 0.1 and abs(r.peakDeg - 9.9) < 0.1, $r

  block low_cut_phase:
    let l3 = RecordingFilter(kind: rfkMinimumPhase, lowCutHz: 3, lowOrder: 2)
    doAssert abs(phaseDeg(l3, 1, 6) - 43.3) < 0.1
    let l1 = RecordingFilter(kind: rfkMinimumPhase, lowCutHz: 1, lowOrder: 1)
    doAssert abs(phaseDeg(l1, 1, 6) - 9.5) < 0.1

  block linear_phase_residual:
    # Odd length: the delay (L - 1)/2 is whole, and one shift leaves nothing.
    let lin = RecordingFilter(kind: rfkLinearPhase, taps: 211, highCutHz: 438,
      highSlopeDbPerOctave: 580)
    doAssert linearDelaySamples(211) == 105
    var r = lin.shiftResidualDeg(1, 6, 220, 105)
    doAssert r.meanDeg < 0.005 and r.peakDeg < 0.005, $r
    # Even length: half a sample is left (0.5 sample = 39.6 deg at 220 Hz).
    var ev = lin
    ev.taps = 64
    r = ev.shiftResidualDeg(1, 6, 220, 31)
    doAssert abs(r.meanDeg - 20.3) < 0.5 and abs(r.peakDeg - 39.6) < 0.5, $r
    # In the time domain: filtered and shifted equals the raw sweep in the
    # band (the filter does nothing there), away from the tapers.
    var shifted = lin
    shifted.shiftSamples = 105
    let q = sweep.applyRecordingFilter(shifted)
    var err = 0.0
    for j in 400 ..< 11600: err = max(err, abs(q.samples[j] - sweep.samples[j]))
    doAssert err < 0.01, $err
    # The design is symmetric.
    let h = linearPhaseTaps(lin, 1)
    doAssert h.len == 211
    for k in 0 ..< 105: doAssert abs(h[k] - h[210 - k]) < 1e-12

  block linear_phase_roll_off:
    # A 100 Hz high-cut: -3 dB at the corner, flat below, falling above.
    let f = RecordingFilter(kind: rfkLinearPhase, taps: 401, highCutHz: 100,
      highSlopeDbPerOctave: 48)
    proc db(hz: float64): float64 = 20 * log10(abs(f.responseAt(2, hz)))
    doAssert abs(db(100) + 3.0) < 0.3, $db(100)
    doAssert abs(db(20)) < 0.1
    doAssert db(200) < -30 and db(200) > -60, $db(200)

  block chebyshev:
    let f = RecordingFilter(kind: rfkMinimumPhase, family: rffChebyshev,
      highCutHz: 100, highOrder: 4, rippleDb: 1)
    proc db(hz: float64): float64 = 20 * log10(abs(f.responseAt(2, hz)))
    # Even order: the passband sits within [-1, 0] dB, the edge at -1 dB.
    doAssert abs(db(0.001) + 1.0) < 0.01, $db(0.001)
    doAssert abs(db(100) + 1.0) < 0.02, $db(100)
    var lo = 0.0
    for i in 1 .. 99: lo = min(lo, db(float64(i)))
    doAssert lo > -1.02 and lo <= -0.99
    # Odd order: 0 dB at 0 Hz.
    var odd = f
    odd.highOrder = 5
    doAssert abs(20 * log10(abs(odd.responseAt(2, 0.001)))) < 0.01
    doAssert abs(20 * log10(abs(odd.responseAt(2, 100))) + 1.0) < 0.02
    # A Chebyshev's filtered sweep is causal and has the pilot's length.
    doAssert sweep.applyRecordingFilter(f).len == 12000

  block chebyshev_low_cut:
    # scipy cheby1 highpass, prewarped bilinear (checks3.py): 20 Hz corner, 1 dB, 2 ms.
    for (order, want) in [(4, @[(20.0, -1.0), (40.0, -0.2394), (10.0, -34.03), (5.0, -59.98),
        (100.0, -0.6134), (249.9, -1.0)]), (5, @[(20.0, -1.0), (40.0, -0.3153),
        (10.0, -45.50), (5.0, -77.95), (100.0, -0.6154), (249.9, 0.0)])]:
      let f = RecordingFilter(kind: rfkMinimumPhase, family: rffChebyshev, lowCutHz: 20,
        lowOrder: order, rippleDb: 1)
      for (hz, db) in want:
        let got = 20 * log10(abs(f.responseAt(2, hz)))
        doAssert abs(got - db) < 0.02, $order & " " & $hz & ": " & $got

  block linear_phase_gain_at_twice_the_cutoff:
    # frequency-sampled Hamming design, checked in numpy: 211 taps, 100 Hz, 48 dB/oct, 2 ms.
    let f = RecordingFilter(kind: rfkLinearPhase, taps: 211, highCutHz: 100,
      highSlopeDbPerOctave: 48)
    proc db(hz: float64): float64 = 20 * log10(abs(f.responseAt(2, hz)))
    doAssert abs(db(20)) < 1e-6 and abs(db(100) + 3.0238) < 0.01, $db(100)
    doAssert abs(db(200) + 48.144) < 0.05, $db(200)
    doAssert abs(db(240) + 60.82) < 0.5, $db(240)

  block even_taps_and_typed_linear_coefficients:
    doAssert linearShiftSamples(64) == 31 and linearShiftSamples(211) == 105
    doAssert linearShiftSamples(1) == 0 and linearShiftSamples(low(int)) == 0
    var ev = RecordingFilter(kind: rfkLinearPhase, taps: 64, highCutHz: 100,
      highSlopeDbPerOctave: 24)
    doAssert suggestedShift(ev, 2, 6, 200) == 31
    doAssert coefficientsSymmetric(@[1.0, 2.0, 3.0, 2.0, 1.0])
    doAssert not coefficientsSymmetric(@[1.0, 2.0, 3.0, 2.5, 1.0])
    # typed taps under linear phase replace the design
    let typed = RecordingFilter(kind: rfkLinearPhase, coefficients: @[0.25, 0.5, 0.25],
      shiftSamples: 1)
    var sp = newSeq[float64](20)
    sp[5] = 1
    let q = initPilot(sp, 1).applyRecordingFilter(typed)
    doAssert q.samples[4] == 0.25 and q.samples[5] == 0.5 and q.samples[6] == 0.25
    doAssert suggestedShift(typed, 1, 6, 200) == 1
    doAssert bulkDelaySamples(typed, 1, 6, 200) == 1

  block extreme_numbers_and_non_finite_output:
    var big = bw(100, 3)
    for v in [low(int), high(int)]:
      big.shiftSamples = v
      doAssertRaises(ValueError): big.checkFilter(1, 100)
      doAssertRaises(ValueError): discard sweep.applyRecordingFilter(big)
    doAssertRaises(ValueError): bw(100, low(int)).checkFilter(1)
    doAssertRaises(ValueError): bw(100, high(int)).checkFilter(1)
    for t in [low(int), high(int), 1]:
      doAssertRaises(ValueError): RecordingFilter(kind: rfkLinearPhase, taps: t,
        highCutHz: 100, highSlopeDbPerOctave: 24).checkFilter(1)
    doAssert linearDelaySamples(low(int)) < 0 # no overflow
    # a ripple that overflows the poles, a coefficient that overflows the sum
    doAssertRaises(ValueError): RecordingFilter(kind: rfkMinimumPhase, family: rffChebyshev,
      highCutHz: 100, highOrder: 4, rippleDb: 1e-300).checkFilter(1)
    doAssertRaises(ValueError): discard initPilot(@[1.0, 1e300, 3.0], 1).applyRecordingFilter(
      RecordingFilter(kind: rfkMinimumPhase, coefficients: @[1e300, 1e300]))
    doAssertRaises(ValueError): discard initPilot(@[1.0'f32, 2, 3], 1).applyRecordingFilter(
      RecordingFilter(kind: rfkMinimumPhase, coefficients: @[1e300]))

  block refusals:
    proc bad(f: RecordingFilter, samples = 100) =
      doAssertRaises(ValueError): f.checkFilter(1, samples)
    bad(bw(500, 3)) # Nyquist at 1 ms
    bad(bw(100, 0))
    bad(bw(100, 33))
    bad(RecordingFilter(kind: rfkMinimumPhase, highOrder: 2)) # no cut
    bad(RecordingFilter(kind: rfkMinimumPhase, lowCutHz: 50, lowOrder: 2, highCutHz: 40,
      highOrder: 2)) # low above high
    bad(RecordingFilter(kind: rfkMinimumPhase, family: rffChebyshev, highCutHz: 100,
      highOrder: 3)) # no ripple
    bad(RecordingFilter(kind: rfkMinimumPhase, coefficients: @[0.0, 0.0]))
    bad(RecordingFilter(kind: rfkMinimumPhase, coefficients: @[1.0, NaN]))
    bad(RecordingFilter(kind: rfkLinearPhase, taps: 2, highCutHz: 100,
      highSlopeDbPerOctave: 24))
    bad(RecordingFilter(kind: rfkLinearPhase, taps: 31, highCutHz: 100)) # no roll-off
    bad(bw(100, 3, 100), 100)
    doAssertRaises(ValueError): discard linearPhaseTaps(RecordingFilter(kind: rfkLinearPhase,
      taps: 5000, highCutHz: 100, highSlopeDbPerOctave: 24), 1)
    RecordingFilter().checkFilter(1)

when isMainModule: main()
