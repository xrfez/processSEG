# Changelog

## Unreleased

- `vibroseis/recfilter`: a reference through a recorder's filter with its bulk delay backed out. `RecordingFilter` (`rfkNone`; `rfkLinearPhase` with `taps`, cutoffs and roll-offs; `rfkMinimumPhase` with typed `coefficients` or a Butterworth/Chebyshev `family`, orders, cutoffs and `rippleDb`; `shiftSamples`, whole samples), `applyRecordingFilter` (`rfkNone` returns the pilot bit for bit; the length and `t0Ms` are kept), `checkFilter`, `linearPhaseTaps`, `linearDelaySamples`, `responseAt`, `bulkDelaySamples`, `suggestedShift`, `shiftResidualDeg`. Nothing is read from headers: every parameter is the caller's. Typed `coefficients` replace the design under either kind; `linearShiftSamples`, `coefficientsSymmetric`, `minRippleDb`. A non-finite output and a ripple below 0.001 dB are refused.

## 0.9.0

- Behaviour change, one rule for lengths: a length in time is N × dt (12000 ms at 1 ms is 12000 samples, the first at 0 ms, the last at 11999 ms). `samplesInLength(lengthMs, dtMs)` and `lengthOfSamples(samples, dtMs)` are the shared helpers (a length that is not a whole multiple of dt is cut down to whole samples).
- `sweepLen` is `lengthMs / dtMs` (was `+ 1`): a sweep of 8000 ms at 2 ms has 4000 samples, not 4001; the end taper's ramp ends at `lengthMs`, one interval after the last sample. `Pilot.lengthMs` is `len · dtMs` (was `(len − 1) · dtMs`); `resamplePilot` gives `lengthMs / dtMs` samples. `applyTaper` takes the sweep to last `x.len · dtMs`.
- `initCorrelationPlan`: the default `outSamples` is the listen time, `recordSamples − pilot.len` (was `+ 1`).


## 0.8.0

- `TrackParams.harmonicBandHz`: THD over the harmonics within a band (`k·f ≤` the edge), as a vibrator controller's distortion; `trackedHarmonics`, `topFrequencyHz` (an estimate of the reference's top frequency).
- `TrackParams.maxFundamentalFraction` (default 0.8, as before); `SweepTrack.fundamentalCut` and `lastFundamentalMs` say where the fundamental stops being measured.
- `TrackParams.harmonicBandAuto` sets the band at the reference's top frequency inside `trackInto` (one pass); `SweepTrack.bandHz` is the edge used. In-band THD is NaN where harmonics are in band but none was measured (0 only where none is in band), and `lkHarmonic` judges the in-band harmonics only. `topFrequencyHz` is an estimate (a chord fit: about 0.4 % on a clean reference with tapers to 0.5 s; wrong with noise after the sweep, tapers over 1 s, short steep exponential sweeps or a cut reference): give `harmonicBandHz` from the sweep's design for QC. `largestChordHz` is a lower bound of the top.
- Behaviour change: `maxHarmonicFraction` no longer limits the fundamental (it follows `maxFundamentalFraction`, 0.8 by default): a caller that set a non-default `maxHarmonicFraction` gets the fundamental measured to 0.8 of Nyquist regardless.
- `windowStats` takes `originMs` (default the extent's start).
- `windowStats`, `WindowStats`, `WindowSummary`: per-window (500 ms) mean phase, force, peak and THD and their average and worst window over the sweep.
- `sweeptrack`: the peak interpolator has 16 taps each side (was 8): a pure fundamental reads within 3e-4 up to 0.8 of Nyquist (it read up to 4 % low from 0.7).

## 0.7.1

- `analyticInto`: the instantaneous frequency was wrong above a quarter of the sampling rate (it read the true value minus fs/2); the phase step is now taken per sample (circular mean of the two steps around the sample), so it is right up to Nyquist for a clean tone.
- `circularMeanDeg`: the circular mean of angles in degrees and the resultant length.

## 0.7.0

Exports added since 0.6.0:

- `timeShiftInto`, `timeShift`: a trace shifted by a whole number of samples, length kept. `wholeSamples` snaps milliseconds, round to nearest, ties away from zero.
- `FkAxes`, `FkPlan`, `initFkPlan`, `axes`, `nFreq`, `nK`, `dfHz`, `dkPerTrace`, `fkRows`, `fkAmplitudeInto`: f-k amplitude of a gather.
- `analyticInto`: envelope, phase and instantaneous Hz from one Hilbert.
- `averageWorkspace`, `averageSpectrogramInto`: a spectrogram averaged over traces.
- `unwrapPhaseInto`, `removeDelayInto`: phase unwrap and linear-delay removal. `SpectrumAccumulator`, `initSpectrumAccumulator`, `used`, `clear`, `amplitudesInto`: a power sum of spectra.
- `heldGain`: a gain curve held flat after a time.
- `appliedResponseInto`: the response of a flow as applied.
- `traces`: the trace count of an f-k plan. `samples`, `dtMs`, `workspaceSize`: overloads for `FkPlan`, and `dtMs` and `workspaceSize` also for `SpectrumAccumulator`.
- `bins`, `windowLen`, `nfft`, `binHz`: overloads for `SpectrumAccumulator`, as for `SpectrumPlan`.
- `add`: overloads for `SpectrumAccumulator`, adding a row block or a tensor.
- `toDbInto` gained a defaulted `refAmplitude = 1.0` parameter: with `dbAbsolute`, 0 dB is that amplitude. Calls that do not pass it compile unchanged.
