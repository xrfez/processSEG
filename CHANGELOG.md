# Changelog

## 0.7.0 (unreleased)

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
