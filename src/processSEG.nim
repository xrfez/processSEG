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
## - `processSEG/adapters`: `chunkMeta` from a prefetch chunk or a trace read.

import processSEG/core/[types, rows, workspace, basic, mute]
import processSEG/fft/[plan, spectrum]
import processSEG/filter/design
import processSEG/adapters
export types, rows, workspace, basic, mute, plan, spectrum, design, adapters
