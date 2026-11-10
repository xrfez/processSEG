## processSEG: seismic processing on top of parseSEG.
##
## This umbrella module re-exports the whole public API:
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
## - `processSEG/adapters`: `chunkMeta` from a prefetch chunk or a trace read.

import processSEG/core/[types, rows, workspace, basic, mute]
import processSEG/fft/[plan, spectrum]
import processSEG/filter/[design, apply, iir]
import processSEG/adapters
export types, rows, workspace, basic, mute, plan, spectrum, design, apply, iir, adapters
