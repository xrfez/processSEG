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
## - `processSEG/fft/plan`: `FftPlan`, real mixed-radix FFTs, batched across traces;
## - `processSEG/fft/spectrum`: `SpectrumPlan`, amplitude spectra and dB for display;
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
## - `processSEG/qc/shotfit`: robust per-shot refraction fit, swapped-geometry flags (`fitShot`);
## - `processSEG/pipeline`: `Op`, `Pipeline`, `compile` → `PipelinePlan`, `apply`;
## - `processSEG/core/pool`: `ProcessPool`, persistent workers that apply plans;
## - `processSEG/adapters`: `chunkMeta` from a prefetch chunk or a trace read;
## - `processSEG/vibroseis/taper`: sweep-end tapers (`SweepTaper`).

import processSEG/core/[types, rows, workspace, basic, mute, shift]
import processSEG/fft/[plan, spectrum]
import processSEG/filter/[design, apply, iir]
import processSEG/gain/[gain, curves, stats]
import processSEG/[pipeline, adapters, batch]
import processSEG/core/pool
import processSEG/velocity/[model, formats, moveout]
import processSEG/statics/elevation
import processSEG/stack/[nmo, binning, stack]
import processSEG/qc/[geometry, firstbreak, shotfit]
import processSEG/vibroseis/[taper]
export types, rows, workspace, basic, mute, plan, spectrum, design,
    apply, iir, gain, curves, stats, pipeline, adapters, pool, shift, model,
    formats, moveout, elevation, nmo, binning, stack, batch, geometry, firstbreak, shotfit
export taper
