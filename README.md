# processSEG

Seismic processing in Nim on top of [parseSEG](https://github.com/xrfez/parseSEG), built for
an interactive viewer: kernels work in place on parseSEG's sample tensors, chunk by chunk,
with no hidden allocation and no hidden threads.

- **Filters:** zero-phase FFT band-pass (cos² trapezoid corners, Butterworth magnitude,
  notches) with edge handling to each trace's live length; Butterworth IIR, zero or minimum
  phase.
- **Gain:** fixed dB, per-trace or per-panel normalization (max, RMS, percentile), AGC,
  time-variant curves (knots, tᵖ, exponential); statistics for GPU uniforms; spectra.
- **Pipelines:** an editable list of ops compiled once to a plan, applied in place or
  src → dst, with the sample margin a zoom window needs; a persistent `ProcessPool`
  runs a plan across cores.
- **Pre-stack:** mutes, datum statics, LMO and NMO (stretch mute), velocity functions and
  fields (VELF, HANDVEL, CSV), CMP binning (2D, 3D), stacking in any order with per-sample
  fold, a brute stack of a whole file, written as SEG-Y (or described by `stackKeys` for
  parseSEG's writer).
- **QC:** geometry checks (header offset against coordinates), first-break pickers
  (STA/LTA, modified energy ratio, modified Coppens) with a confidence per pick, and a
  robust per-shot refraction fit that flags swapped receivers; windowed and coherence SNR.
- **Vibroseis:** sweep generation (linear, dB/octave, dB/Hz, exponential, table; nine taper
  shapes), generated or recorded pilots (resampling, instrument filters, SEG-Y sweep
  headers), correlation and sweep deconvolution, vertical and diversity stacking of
  repeated sweeps, and continuous node recordings cut into records at shot times.
- **Survey QC:** receiver patches of an X survey described as a template or a distance box,
  X records generated from either (roll, shift, static edges), automatic bin grids,
  pre-plot and actual fold maps (offset classes, azimuth sectors) and their difference,
  field records checked against the XPS by station (missing, unlisted, mismatched records),
  and amplitude QC per trace and record (dead, clipped, spiky, DC, weak, noisy traces).

float32 and float64 throughout; units are ms, Hz, dB and distance units per second.

## Quick start

```nim
import parseSEG, processSEG

# A file opened with keys gives the rows' geometry; here, 48 synthetic traces.
var keys = newSeq[TraceKeys](48)
var counts = newSeq[int](48)
for i in 0 ..< 48:
  keys[i] = initTraceKeys()
  keys[i].offset = 25.0 * float64(i)
  keys[i].t0Ms = 0
  counts[i] = 2001
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
var raw = zeros[float32](48, 2001)          # decoded samples, [traces, samples]

let flow = initPipeline([demeanOp(),
                         bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                         agcOp(windowMs = 500)])
let plan = flow.compile(float32, dtMs = 2, samples = 2001)

# One thread: a Workspace per thread, reserved once.
var ws = initWorkspace()
ws.reserve(plan)
var processed = zeros[float32](48, 2001)
plan.apply(raw, processed, meta, ws)        # src → dst, no allocation

# Or every core: a persistent pool, created once.
let pool = newProcessPool(workers = 4)
pool.apply(plan, raw, processed, meta)

# A zoom on samples 500 .. 899 decodes a margin on each side.
let window = decodeWindow(500 .. 899, plan.requiredMargin, traceLen = 2001)
echo "decode ", window, ", tag the result with generation ", plan.generation
```

With a real file, the meta comes from parseSEG's prefetcher (`chunkMeta(chunk)`) or a
trace read (`chunkMeta(f, traces, failed)`); see the guide.

## Documentation

- [docs/guide/](docs/guide/README.md): task-oriented guides with tested examples.
- [docs/api.md](docs/api.md): the API reference.
- [docs/integration.md](docs/integration.md): the contract between parseSEG, processSEG and
  a viewer (who reads the file, buffers and plans, zoom margins, display gain, overlays).
- [bench/README.md](bench/README.md): measured throughput (a 2000-trace panel: a band-pass
  in about 6 ms on 4 workers).
- `nim docs`: the generated HTML reference, in `htmldocs/`.

## Setup

Requires Nim 2.4.0. Dependencies are managed by [Atlas](https://github.com/nim-lang/atlas)
and pinned in `atlas.lock`; they are checked out into `deps/`, and `nim.cfg` holds the search
paths Atlas generates for them.

```sh
atlas rep atlas.lock    # fresh clone: check out the pinned dependencies
```

To add or change a dependency, edit the `requires` lines in `processSEG.nimble` (or run
`atlas use <url>`), run `atlas install`, then `atlas pin` to update `atlas.lock`. Do not
edit the Atlas section of `nim.cfg` by hand: Atlas rewrites it.

## Directory Layout

- `src/processSEG.nim`: the umbrella module; `src/processSEG/<area>/<module>.nim`: the
  library (core, fft, filter, gain, velocity, statics, stack, qc, pipeline, batch).
- `tests/`: test suites (`tester.nim` runs every `t*.nim`; helpers go in `tests/utils/`;
  `tests/reject/` holds programs that must not compile; `tests/fuzz/` the fuzz targets).
- `bench/`: benchmarks and their recorded results.
- `docs/`: guides, API reference and the integration contract.
- `tools/`: build helper wrappers (`tools/zigcc` for Zig CC invocations).
- `deps/`: Atlas dependency checkouts (only `deps/atlas.config` is committed).
- `bin/`: compiled binaries.
- `.github/workflows/`: GitHub Actions CI pipeline.

## Build Tasks

Build modes are defined in `config.nims` and output to `./bin/`:

| Task | Toolchain | Profile Description | Key Flags |
|---|---|---|---|
| `nim nimRelease` | GCC/Default | Native Host Optimized | `-d:release -d:danger --opt:speed -march=native` |
| `nim nimPortable` | GCC/Default | Generic Portable Size | `-d:release --opt:size -march=x86-64 -mtune=generic -s` |
| `nim nimDebug` | GCC/Default | Full Debug & Diagnostics | `--debuginfo:on --assertions:on --stackTrace:on` |
| `nim zigRelease` | Zig CC | Native Host Optimized | `-d:release -d:danger --opt:speed -O3 -march=native` |
| `nim zigPortable` | Zig CC | Generic Portable Size | `-d:release --opt:size -Oz -mcpu=baseline -s` |
| `nim zigDebug` | Zig CC | Full Debug & Diagnostics | `--debuginfo:on` |

**CPU target.** parseSEG is compiled into this program, so the build flags decide its decode
speed. On x86-64, every build that does not choose its own CPU target (`nim test`, the debug
builds) targets x86-64-v3 (AVX2: Intel 2013+ / AMD Zen 2017+), which runs parseSEG's sample
decoders up to 2x faster than baseline x86-64. A v3 binary crashes on CPUs without AVX2: use
the portable builds there (they pass `-d:baselineCpu`). Other CPUs (arm64) keep the compiler's
default.

## Development Commands

- Run test suite: `nim test`
- Run test suite in release and danger builds: `nim testRelease`
- Run test suite under AddressSanitizer: `nim asan`
- Run the fuzz targets (libFuzzer, ASan, UBSan; clang with compiler-rt):
  `nim fuzz` (`FUZZ_SECONDS=N` per target, default 60)
- Run the benchmarks: `nim bench` (see `bench/README.md`)
- Format codebase: `nim fmt`
- Generate documentation: `nim docs`

CI (`.github/workflows/ci.yml`) replays `atlas.lock`, checks formatting, and runs the tests in
debug, release and danger builds and the docs build on Linux, and the tests on Windows and
macOS.

## Acknowledgements & Attributions

- **[parseSEG](https://github.com/xrfez/parseSEG)**: SEG-Y / SEG-D reading, writing and indexing.
- **[nimlangserver](https://github.com/nim-lang/langserver)**: Official Nim Language Server with MCP support.
- **[planetis-m/skills](https://github.com/planetis-m/skills)**: Curated agent skill repository for Nim development by [planetis-m](https://github.com/planetis-m).
- **[setup-nim-action](https://github.com/jiro4989/setup-nim-action)**: Nim GitHub Actions runner by [jiro4989](https://github.com/jiro4989).
