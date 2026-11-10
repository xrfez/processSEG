# processSEG

Seismic SEG-Y / SEG-D processing in Nim, built on [parseSEG](https://github.com/xrfez/parseSEG).

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

- `src/`: Source code modules.
- `tests/`: Test suites (`tester.nim` runs every `t*.nim`; helpers go in `tests/utils/`).
- `tools/`: Build helper wrappers (`tools/zigcc` for Zig CC invocations).
- `deps/`: Atlas dependency checkouts (only `deps/atlas.config` is committed).
- `bin/`: Compiled binary target directory.
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
