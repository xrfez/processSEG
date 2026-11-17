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
