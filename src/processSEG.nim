## processSEG: seismic processing on top of parseSEG.
##
## This umbrella module re-exports the whole public API of processSEG. It does
## not re-export parseSEG: callers import it themselves for files, tensors and
## views (`import parseSEG, processSEG`), so the two packages stay separate:
## - `processSEG/core/types`: `ChunkMeta`, the metadata of a block of rows;
## - `processSEG/core/rows`: `RowBlock`, zero-copy rows of a tensor or buffer;
## - `processSEG/core/workspace`: `Workspace`, grow-only per-thread scratch;
## - `processSEG/core/basic`: demean, detrend, scale, clip, polarity, pad taper;
## - `processSEG/core/mute`: top and bottom mutes (`MuteCurve`);
## - `processSEG/core/units`: amplitude units; sensitivities and descales entered with their
##   units (`Sensitivity`, `Descale`, `FileScale`);
## - `processSEG/core/calibrate`: recorded values to volts or physical units per row
##   (`Calibration`, `CalibrationPlan`, `calibrate`, `resolveInto`);
## - `processSEG/fft/plan`: `FftPlan`, real mixed-radix FFTs, batched across traces;
## - `processSEG/fft/spectrum`: `SpectrumPlan`, amplitude spectra and dB for display;
## - `processSEG/fft/hilbert`: `HilbertPlan`, the Hilbert transform and the envelope of a trace;
## - `processSEG/fft/spectrogram`: `SpectrogramPlan`, short-time Fourier amplitudes (time-frequency);
## - `processSEG/fft/fk`: `FkPlan`, the f-k amplitude of a gather;
## - `processSEG/filter/design`: `FilterSpec`, cos² trapezoid, Butterworth and notch responses;
## - `processSEG/filter/apply`: `FilterPlan`, zero-phase FFT filtering with edge handling;
## - `processSEG/filter/iir`: `IirPlan`, Butterworth biquads, zero and minimum phase;
## - `processSEG/gain/gain`: fixed dB gain, normalization (`NormalizePlan`), AGC (`AgcPlan`);
## - `processSEG/gain/curves`: time-variant gain (`GainCurve`, `GainCurvePlan`);
## - `processSEG/gain/stats`: per-trace and panel RMS, max and percentile clip;
## - `processSEG/core/shift`: fractional time shifts of a row (`shiftSamples`);
## - `processSEG/velocity/model`: velocity functions, Dix, fields (`VelocityField`);
## - `processSEG/velocity/formats`: VELF, HANDVEL and CSV velocity picks;
## - `processSEG/velocity/moveout`: LMO and NMO times for overlays (`Moveout`), LMO on samples;
## - `processSEG/statics/elevation`: datum statics from elevations (`DatumStatics`);
## - `processSEG/stack/nmo`: NMO correction with a stretch mute (`applyNmo`, `nmoInto`);
## - `processSEG/stack/binning`: CMP bins, 2D and 3D (`Binning`, `binsInto`);
## - `processSEG/stack/stack`: `StackAccumulator`, any order, per-sample fold;
## - `processSEG/batch`: `bruteStack` of a file (prefetch transform, per-worker stacks), `writeStack`;
## - `processSEG/qc/geometry`: geometry QC flags per trace (`checkGeometryInto`);
## - `processSEG/qc/firstbreak`: first-break pickers with a confidence (`FirstBreakPlan`, `pickInto`);
## - `processSEG/qc/snr`: windowed SNR per trace, coherence SNR per gather versus time (`windowSnrInto`, `coherenceSnrInto`);
## - `processSEG/qc/shotfit`: robust per-shot refraction fit, swapped-geometry flags (`fitShot`);
## - `processSEG/pipeline`: `Op`, `Pipeline`, `compile` → `PipelinePlan`, `apply`;
## - `processSEG/core/pool`: `ProcessPool`, persistent workers that apply plans;
## - `processSEG/adapters`: `chunkMeta` from a prefetch chunk or a trace read, `pilotFromTrace`;
## - `processSEG/filescales`: what a file's headers say about its scale (`fileScalesInto`,
##   `scaleReport`, `segyUnitOverrides`);
## - `processSEG/vibroseis/taper`: sweep-end tapers (`SweepTaper`);
## - `processSEG/vibroseis/sweep`: sweep generation (`SweepSpec`, `sweepSamples`,
##   `sweepHarmonics`);
## - `processSEG/vibroseis/pilot`: generated and recorded pilots (`Pilot`), resampling,
##   instrument filters, SEG-Y sweep headers;
## - `processSEG/vibroseis/correlate`: correlation and sweep deconvolution (`CorrelationPlan`);
## - `processSEG/vibroseis/vstack`: `RecordStack`, vertical and diversity stacks of repeated records;
## - `processSEG/vibroseis/sweeptrack`: vibrator signals against sweep time: phase, fundamental,
##   harmonics, THD, peak and frequency (`SweepTrackPlan`, `trackInto`);
## - `processSEG/vibroseis/crosscorrelate`: a signal against its reference: correlation over lags,
##   envelope, the lag window's spectrum and phase, comparisons (`CrossCorrelationPlan`);
## - `processSEG/vibroseis/vibqc`: vibrator QC of a record: roles, ground force from M and B, limits,
##   summaries (`VibUnit`, `analyzeVibrators`);
## - `processSEG/continuous/slice`: continuous recordings cut into records at shot times
##   (`planSlices`, `extractInto`);
## - `processSEG/survey/patch`: receiver patches of an X survey, template and distance box
##   (`describePatches`, `receiverLayout`);
## - `processSEG/qc/fold`: fold maps of a survey: its bin grid (`surveyGrid`), pre-plot and actual fold (`preplotFold`, `addFold`, `foldDifference`);
## - `processSEG/qc/record`: field records against the XPS (`recordCompleteness`, `summarize`);
## - `processSEG/qc/amplitude`: amplitude QC per trace and per record (`traceAmplitudeInto`, `recordAmplitude`);
## - `processSEG/synth`: synthetic traces and gathers (`ricker`, `addSinusoid`, `addRicker`,
##   `addNoise`, `linearMs`, `hyperbolicMs`, `addLinear`, `addHyperbola`).

import processSEG/core/[types, rows, workspace, basic, mute, shift, units, calibrate]
import processSEG/fft/[plan, spectrum, hilbert, spectrogram, fk]
import processSEG/filter/[design, apply, iir]
import processSEG/gain/[gain, curves, stats]
import processSEG/[pipeline, adapters, batch, filescales]
import processSEG/core/pool
import processSEG/velocity/[model, formats, moveout]
import processSEG/statics/elevation
import processSEG/stack/[nmo, binning, stack]
import processSEG/qc/[geometry, firstbreak, snr, shotfit, fold, record, amplitude]
import processSEG/vibroseis/[taper, sweep, pilot, correlate, vstack, sweeptrack]
import processSEG/vibroseis/[crosscorrelate, vibqc]
import processSEG/continuous/slice
import processSEG/survey/patch
import processSEG/synth
export types, rows, workspace, basic, mute, plan, spectrum, hilbert, spectrogram, fk, design,
    apply, iir, gain, curves, stats, pipeline, adapters, pool, shift, model,
    formats, moveout, elevation, nmo, binning, stack, batch, geometry, firstbreak, snr,
    shotfit, filescales
export taper, sweep, pilot, correlate, vstack, slice, patch, fold, record, amplitude, units,
    calibrate, sweeptrack, crosscorrelate, vibqc, synth
