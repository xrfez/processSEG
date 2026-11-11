# Getting started

This page covers the three types every processSEG call takes (`ChunkMeta`, `RowBlock`,
`Workspace`) and a first kernel call, on synthetic data and on a file.

## Build and import

processSEG needs Nim 2.4 and parseSEG, whose checkouts are pinned by Atlas in `deps/`
(`nim.cfg` holds the paths). Add `src/` to your search path, and import both packages:
processSEG does not re-export parseSEG, which owns files, `TraceKeys`, the `TraceRows`
view and the Arraymancer tensor API.

```nim skip
import parseSEG, processSEG
```

```sh
nim test          # the test suite, these examples included (tests/tdocs.nim)
nim docs          # the doc comments as HTML, into htmldocs/
```

parseSEG is compiled into your program, so your flags set its decode speed and
processSEG's: build with `-d:release` (or `-d:danger`) and, on x86-64, `-march=x86-64-v3`
or `-march=native` (the project's `config.nims` does this for its own builds).

## The model

processSEG never opens a file. It processes a block of samples that you (or parseSEG's
prefetcher) decoded, described by a `ChunkMeta`:

- **samples**: a `Tensor[float32]` (or `float64`) shaped `[rows, cols]`, row-major and
  C-contiguous, as `readTraces`, `readTracesInto` and the prefetcher deliver it. Rows past
  a shorter trace's end are zero;
- **`ChunkMeta`**: parseSEG's `TraceRows` view of the same rows (it must have keys), the
  sample interval `dtMs`, the window origin `firstSample` and the rows that failed to
  decode. It copies no header value: `meta.keys(row)` reads the view;
- **`RowBlock[T]`**: a raw view of the tensor's rows; `b.row(r)` is a row as an
  `openArray`, in place. Every kernel exists for one row (`var openArray[T]`) and for a
  whole block (a tensor or a `RowBlock`, with the meta);
- **`Workspace`**: grow-only scratch for kernels that need temporary buffers. Reserve it
  once per plan; applying then allocates nothing.

The rules that every kernel follows are on one list in
[api.md, Conventions](../api.md#conventions): live rows, the time axis, NaN for unknown
keys, units, errors.

## A first kernel call

Synthetic data needs only keys: `initTraceKeys()` gives a trace with every key unknown
(NaN), and you fill in what the processing reads. `rowsOf(keys, numSamples)` is the view.

```nim
import parseSEG, processSEG

const dtMs = 2.0
const ns = 501
var keys = newSeq[TraceKeys](4)
var numSamples = newSeq[int](4)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].offset = 100.0 * float64(i + 1)
  keys[i].t0Ms = 0
  keys[i].intervalUs = dtMs * 1000
  keys[i].status = tsLive
  numSamples[i] = ns
keys[2].status = tsDead                 # dead rows are passed through
numSamples[3] = 300                     # a short trace: zero past sample 300

let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs) # validates once
var samples = zeros[float32](keys.len, ns)
for r in 0 ..< keys.len:
  for j in 0 ..< numSamples[r]:
    samples[r, j] = 5 + float32(j mod 7) # a DC level plus a ripple

samples.demean(meta)                    # in place: live rows, live samples
var mean = 0.0
for j in 0 ..< ns: mean += float64(samples[0, j]) / ns
doAssert abs(mean) < 1e-5               # the mean is gone
doAssert samples[2, 0] == 5             # the dead row is untouched
doAssert samples[3, 400] == 0           # padding stays zero
doAssert meta.liveSamples(3, ns) == 300
doAssert meta.sampleTimeMs(1, 10) == 20 # t0Ms + (firstSample + col)·dtMs
```

The block form on a `RowBlock` is the same call; a row kernel takes the live part of one
row (`b.row(r, n)`), so you decide which rows and samples it sees:

```nim
import parseSEG, processSEG

var keys = @[initTraceKeys(), initTraceKeys()]
for k in keys.mitems:
  k.status = tsLive
  k.intervalUs = 4000
let numSamples = @[100, 100]
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 4)
var samples = zeros[float64](2, 100)
for j in 0 ..< 100:
  samples[0, j] = float64(j)            # a linear trend
  samples[1, j] = -float64(j)

let b = rowBlock(samples)               # checks rank 2 and C-contiguity once
b.detrend(meta)                         # block form
doAssert abs(samples[0, 50]) < 1e-9

for r in 0 ..< b.len:                   # row kernels on the live part
  if meta.isLive(r):
    reversePolarity(b.row(r, meta.liveSamples(r, b.cols)))
b.sub(1 .. 1).scale(meta.sub(1 .. 1), 2.0) # one row of each, as a thread would
```

## Plans and workspaces

Anything with precomputed tables (a filter's response, an AGC window, an FFT) is a plan,
built once per sample interval and trace length. It is immutable, validated when built
(bad parameters raise `ValueError` there, never later), and says how much scratch it
needs:

```nim
import std/math
import parseSEG, processSEG

const dtMs = 2.0
const ns = 1001
var keys = newSeq[TraceKeys](8)
var numSamples = newSeq[int](8)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].status = tsLive
  keys[i].t0Ms = 0
  keys[i].intervalUs = dtMs * 1000
  numSamples[i] = ns
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs)
var samples = zeros[float32](keys.len, ns)
for r in 0 ..< keys.len:
  for j in 0 ..< ns:
    let t = float64(j) * dtMs / 1000
    samples[r, j] = float32(sin(2 * PI * 3 * t) + sin(2 * PI * 30 * t))

let plan = initFilterPlan(float32, bandPass(Corners(f1: 10, f2: 15, f3: 60, f4: 80)),
  dtMs, samples = ns)                   # ValueError for a bad spec, here
var ws = initWorkspace()
ws.reserve(plan.workspaceSize)          # grows once; a no-op when it fits
samples.applyFilter(meta, plan, ws)     # allocates nothing
let stats = block:
  var s = newSeq[TraceStats](meta.len)
  traceStatsInto(samples, meta, s)
  s
doAssert abs(stats[0].rms - sqrt(0.5)) < 0.02 # the 30 Hz sine passes, 3 Hz is gone
doAssert plan.requiredMargin > 0        # samples of context each side, for zooms

doAssertRaises ValueError:
  discard initFilterPlan(float32, lowCut(0, 300), dtMs, ns) # above Nyquist
```

One `Workspace` per thread: it cannot be copied, and several threads may share one plan.
A `PipelinePlan` chains many ops in one apply, and a `ProcessPool` spreads it over
threads with their own workspaces: see [pipelines.md](pipelines.md).

## From a file

With a real file, parseSEG decodes and the adapters build the meta: `chunkMeta(c)` for a
prefetch chunk, `chunkMeta(f, traces[, samples], failed)` for a `readTracesInto`. Open the
file with `keys = true` (or call `ensureKeys` before making any view). This example
writes a small SEG-Y first, with parseSEG's writer of traces from keys:

```nim
import std/math
import parseSEG, processSEG

block write:
  var w = openSeismicWriter("first.sgy", SegyOutputOptions())
  var x = newSeq[float32](500)
  for i in 0 ..< 12:
    var k = initTraceKeys()
    k.offset = 50.0 * float64(i + 1)
    k.t0Ms = 0
    k.intervalUs = 2000
    k.status = tsLive
    for j in 0 ..< x.len: x[j] = float32(100 + 10 * sin(float64(j) / 5))
    w.writeTrace(k, x)
  discard w.close()

var f = openSeismic("first.sgy", keys = true)
var samples = newTraceTensor[float32](f.len, f.maxSamples)
var failed: seq[int]                    # reused: cleared, capacity kept
discard f.readTracesInto(0 ..< f.len, samples, failed)
let meta = chunkMeta(f, 0 ..< f.len, failed) # dtMs from the keys' intervalUs
doAssert meta.dtMs == 2 and meta.failedCount == 0
samples.demean(meta)
doAssert abs(samples[0, 0]) < 1
f.close()
```

`failed` (and a trace list passed to `chunkMeta`) is held by pointer: it must outlive the
meta. A meta is cheap to build, so build it per apply rather than storing it. The rules a
viewer must follow around these views (keys revisions, `applySurvey`, raw caches) are in
[integration.md](../integration.md#data-contract).
