## processSEG: seismic processing on top of parseSEG.
##
## This umbrella module re-exports the whole public API:
## - `processSEG/core/types`: `ChunkMeta`, the metadata of a block of rows;
## - `processSEG/core/rows`: `RowBlock`, zero-copy rows of a tensor or buffer;
## - `processSEG/core/workspace`: `Workspace`, grow-only per-thread scratch.

import processSEG/core/[types, rows, workspace]
export types, rows, workspace
