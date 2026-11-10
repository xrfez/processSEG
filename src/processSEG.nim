## processSEG: seismic processing on top of parseSEG.
##
## This umbrella module re-exports the whole public API:
## - `processSEG/core/types`: `ChunkMeta`, the metadata of a block of rows.

import processSEG/core/types
export types
