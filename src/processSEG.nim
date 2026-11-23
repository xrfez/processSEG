## processSEG: seismic processing on top of parseSEG.
##
## This umbrella module re-exports the whole public API of processSEG. It does
## not re-export parseSEG: callers import it themselves for files, tensors and
## views (`import parseSEG, processSEG`), so the two packages stay separate:
## - `processSEG/core/types`: `ChunkMeta`, the metadata of a block of rows;
##   `StopCheck`, a caller's abandon check for a long kernel, and `Stopped`;
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
##   a stored trace placed later (`placedWhole`, `placedSinc`);
## - `processSEG/core/timeshift`: a whole-sample shift of a trace (`timeShift`,
##   `wholeSamples`);
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
## - `processSEG/qc/breakfit`: robust start and trimmed model of first-break picks (`trimmedModel`);
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
## - `processSEG/vibroseis/sweepedge`: the top frequency of a reference sweep (`readTop`,
##   `TopReading`, `TopReason`);
## - `processSEG/vibroseis/recfilter`: a reference through a recorder's filter, its bulk delay backed out
##   (`RecordingFilter`, `applyRecordingFilter`, `suggestedShift`);
## - `processSEG/vibroseis/crosscorrelate`: a signal against its reference: correlation over lags,
##   envelope, the lag window's spectrum and phase, comparisons (`CrossCorrelationPlan`);
## - `processSEG/vibroseis/matchpeak`: the correlation envelope's peak, its time shift and the
##   correlation's phase there (`matchOf`, `peakAt`, `rotationOf`);
## - `processSEG/vibroseis/vibqc`: vibrator QC of a record: roles, ground force from M and B, limits,
##   summaries (`VibUnit`, `analyzeVibrators`);
## - `processSEG/continuous/slice`: continuous recordings cut into records at shot times
##   (`planSlices`, `extractInto`);
## - `processSEG/survey/patch`: receiver patches of an X survey, template and distance box
##   (`describePatches`, `receiverLayout`);
## - `processSEG/qc/fold`: fold maps of a survey: its bin grid (`surveyGrid`), pre-plot and actual fold (`preplotFold`, `addFold`, `foldDifference`);
## - `processSEG/qc/record`: field records against the XPS (`recordCompleteness`, `summarize`);
## - `processSEG/qc/amplitude`: amplitude QC per trace and per record (`traceAmplitudeInto`, `recordAmplitude`);
## - `processSEG/qc/exclusion`: noisy and weak traces outside the exclusion area near the source, nodes and components, louder than neighbours (`noiseQc`, `traceRolesInto`);
## - `processSEG/qc/signalend`: where a trace's signal ends (`blockLevels`,
##   `lastLoudMs`, `signalEndMs`);
## - `processSEG/synth`: synthetic traces and gathers (`ricker`, `addSinusoid`, `addRicker`,
##   `addNoise`, `linearMs`, `hyperbolicMs`, `addLinear`, `addHyperbola`).

import processSEG/core/[types, rows, workspace, basic, mute, shift, timeshift, units, calibrate]
import processSEG/fft/[plan, spectrum, hilbert, spectrogram, fk]
import processSEG/filter/[design, apply, iir]
import processSEG/gain/[gain, curves, stats]
import processSEG/[pipeline, adapters, batch, filescales]
import processSEG/core/pool
import processSEG/velocity/[model, formats, moveout]
import processSEG/statics/elevation
import processSEG/stack/[nmo, binning, stack]
import processSEG/qc/[geometry, firstbreak, snr, shotfit, breakfit, fold, record]
import processSEG/qc/[amplitude, exclusion, signalend]
import processSEG/vibroseis/[taper, sweep, pilot, correlate, vstack, sweeptrack, sweepedge]
import processSEG/vibroseis/[crosscorrelate, matchpeak, vibqc, recfilter]
import processSEG/continuous/slice
import processSEG/survey/patch
import processSEG/synth
export spectrum except addAmplitudeRows # a pool batch helper: scaled power, not an amplitude
export types, rows, workspace, basic, mute, plan, hilbert, spectrogram, fk,
    timeshift, design,
    apply, iir, gain, curves, stats, pipeline, adapters, pool, shift, model,
    formats, moveout, elevation, nmo, binning, stack, batch, geometry, firstbreak, snr,
    shotfit, breakfit, filescales
export taper, sweep, pilot, correlate, vstack, slice, patch, fold, record, amplitude, units,
    calibrate, sweeptrack, sweepedge, crosscorrelate, matchpeak, vibqc, recfilter, synth,
    exclusion, signalend
