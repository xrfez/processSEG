## processSEG: seismic processing on top of parseSEG.
##
## This umbrella module re-exports the whole public API, and parseSEG too
## (`Tensor`, `SeismicFile`, the prefetcher, `TraceRows`, ...), so a viewer
## needs only `import processSEG`:
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
## - `processSEG/velocity/moveout`: LMO and NMO times for overlays (`Moveout`);
## - `processSEG/pipeline`: `Op`, `Pipeline`, `compile` → `PipelinePlan`, `apply`;
## - `processSEG/core/pool`: `ProcessPool`, persistent workers that apply plans;
## - `processSEG/adapters`: `chunkMeta` from a prefetch chunk or a trace read.

import parseSEG
import processSEG/core/[types, rows, workspace, basic, mute]
import processSEG/fft/[plan, spectrum]
import processSEG/filter/[design, apply, iir]
import processSEG/gain/[gain, curves, stats]
import processSEG/[pipeline, adapters]
import processSEG/core/pool
import processSEG/velocity/moveout
export parseSEG, types, rows, workspace, basic, mute, plan, spectrum, design,
    apply, iir, gain, curves, stats, pipeline, adapters, pool, moveout
