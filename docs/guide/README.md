# processSEG usage guides

Task-oriented guides to the public API, for Nim developers who know seismic processing but
not this library. Two users are in mind: the developer of a seismic **viewer** (pipelines,
the process pool, zoom windows, display gain, overlays) and someone running **batch jobs**
(brute stacks, first breaks, geometry QC, vibroseis correlation).

[`../api.md`](../api.md) is the reference (every exported name, one line each, and the
conventions every call follows); [`../integration.md`](../integration.md) is the contract
between parseSEG, processSEG and a viewer. These pages show how the pieces fit together.
Every `nim` code block here is a complete program that `tests/tdocs.nim` compiles and runs
(blocks marked `nim skip` are not run). The examples build their own data, synthetic keys
or a small SEG-Y written with parseSEG, so they need no files.

## Which page to read

| You want to | Read |
|---|---|
| Understand the model: `ChunkMeta`, `RowBlock`, `Workspace`, live rows, the time axis; a first kernel call; metas from a file | [getting-started.md](getting-started.md) (start here) |
| Build, edit and compile a processing flow; apply it in place or `src → dst` | [pipelines.md](pipelines.md#build-compile-apply) |
| Process a zoom window with enough margin (`requiredMargin`, `decodeWindow`) | [pipelines.md](pipelines.md#zoom-windows) |
| Spread the work over threads (`ProcessPool`, or your own threads with plan parts) | [pipelines.md](pipelines.md#the-process-pool) |
| Apply fixed, normalized, AGC or time-variant gain | [gain.md](gain.md) |
| Compute statistics for a GPU display gain (`traceStatsInto`, `panelStats`) | [gain.md](gain.md#statistics-for-a-shader-uniform) |
| Show an amplitude spectrum of a selection | [gain.md](gain.md#amplitude-spectra) |
| Build velocity functions and fields; read VELF, HANDVEL or CSV picks | [velocity-and-stack.md](velocity-and-stack.md) |
| Draw moveout ticks over a gather; flatten first breaks (LMO) | [velocity-and-stack.md](velocity-and-stack.md#moveout-overlays) |
| Apply datum statics, NMO, bin and stack | [velocity-and-stack.md](velocity-and-stack.md#nmo-and-stacking-by-hand) |
| Brute-stack a whole file and write the stack as SEG-Y | [velocity-and-stack.md](velocity-and-stack.md#a-brute-stack-of-a-file) |
| Check geometry from the keys, pick first breaks, find swapped receivers | [qc.md](qc.md) |
| Measure signal-to-noise per trace or per gather | [qc.md](qc.md#signal-to-noise-ratios) |
| Generate a sweep or read a recorded pilot; write it as SEG-Y with its sweep fields | [vibroseis.md](vibroseis.md#sweeps-and-pilots) |
| Resample a pilot, or pass it through the recorder's filter | [vibroseis.md](vibroseis.md#matching-the-pilot-to-the-data) |
| Correlate vibroseis records, or deconvolve the sweep | [vibroseis.md](vibroseis.md#correlation-and-sweep-deconvolution) |
| Stack repeated sweeps (vertical, diversity) | [vibroseis.md](vibroseis.md#stacking-repeated-sweeps) |
| Cut continuous node recordings into records at shot times | [vibroseis.md](vibroseis.md#continuous-recordings) |

## Related documents

- `nim docs`: the generated HTML from the doc comments (formulas, limits, edge cases).
- `bench/README.md`: measured costs of the kernels, the pool and the batch jobs.
- parseSEG's own guides (`deps/parseSEG/docs/guide/`): opening, prefetching, reading
  sample windows, editing headers, writing files.
