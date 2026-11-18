# Changelog

## Unreleased

- `TrackParams.harmonicBandHz`: THD over the harmonics within a band (`k·f ≤` the edge), as a vibrator controller's distortion; `trackedHarmonics`, `topFrequencyHz` (the reference's top frequency).
- `TrackParams.maxFundamentalFraction` (default 0.8, as before); `SweepTrack.fundamentalCut` and `lastFundamentalMs` say where the fundamental stops being measured.
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
