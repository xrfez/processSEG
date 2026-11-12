# Pipelines, zoom windows and the process pool

A viewer applies a user-edited processing flow to every chunk it draws. processSEG keeps
the flow as plain data (`Pipeline`), compiles it once per sample interval and width
(`PipelinePlan`), and applies the plan with no allocation, on one thread or across a
`ProcessPool`. [integration.md](../integration.md#plans-and-buffers-the-interactive-loop)
is the viewer contract (stale buffers, raw caches, what allocates); this page shows the
calls.

## Build, compile, apply

```nim
import std/[math, strutils]
import parseSEG, processSEG

const dtMs = 2.0
const ns = 1001
var keys = newSeq[TraceKeys](48)
var numSamples = newSeq[int](48)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].offset = 25.0 * float64(i + 1)
  keys[i].t0Ms = 0
  keys[i].intervalUs = dtMs * 1000
  keys[i].status = tsLive
  numSamples[i] = ns
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs)
var raw = zeros[float32](keys.len, ns)  # the viewer's raw cache entry
for r in 0 ..< keys.len:
  for j in 0 ..< ns:
    let t = float64(j) * dtMs / 1000
    raw[r, j] = float32(50 + sin(2 * PI * 25 * t) * exp(-t))

var flow = initPipeline([demeanOp(),
                         bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                         muteTopOp(lmoMute(velocity = 2000, padMs = -20), taperMs = 20),
                         agcOp(windowMs = 500)])
let plan = flow.compile(float32, dtMs, samples = ns) # ValueError names a bad op
var ws = initWorkspace()
ws.reserve(plan)                        # once per plan; a no-op when it fits

var processed = zeros[float32](keys.len, ns) # allocated once per cached chunk
plan.apply(raw, processed, meta, ws)    # src → dst: raw kept, the copy fused
doAssert raw[0, 100] > 40 and abs(processed[0, 100]) < 3
var tag = plan.generation               # tag the buffer; 0 means "never processed"

flow[3].agc.windowMs = 300              # a slider moved: edit the op, recompile
let plan2 = flow.compile(float32, dtMs, samples = ns)
doAssert plan2.generation > tag
ws.reserve(plan2)
plan2.apply(raw, processed, meta, ws)
tag = plan2.generation

flow.insert(scaleOp(NaN), 0)            # nothing is checked until compile
try:
  discard flow.compile(float32, dtMs, ns)
  doAssert false
except ValueError as e:
  doAssert e.msg.startsWith("op 0 (opScale)")
flow.delete(0)
```

- Ops are a variant `Op` with a parameter group per kind (`op.agc`, `op.filter.spec`,
  `op.mute.taperMs`, ...), read and edited through accessors that raise `ValueError` for
  an op of another kind; a `Pipeline` supports `add`, `insert`, `delete`, `[]`, `[]=`,
  `len`, `items` and `pairs`.
- Compiling is cheap (microseconds to a few ms), so recompile on every slider event. Keep
  one plan per sample interval: a uniform prefetch of a mixed SEG-D file gives chunks at
  several.
- A buffer is stale when the plan's `generation` **or** the file's `keysRevision`
  changed: mutes read offsets, gain curves `t0Ms`, every op the trace status.
- `plan.apply(samples, meta, ws)` runs in place, for batch jobs without a raw cache. `ws`
  must hold `plan.workspaceSize`: `apply` never grows it.

## Zoom windows

Filters and AGC need context beyond the samples on screen. `requiredMargin` is the number
of samples to decode on each side (the sum of the filters' impulse lengths and the half
AGC windows), and `decodeWindow` widens the shown range by it, clamped to the trace. The
`Pipeline` gives the margin without compiling, so you can size the decode before building
the plan for the window's width; a plan also takes any block narrower than its
`samples`, so one compiled for whole traces works on a window too.

```nim
import std/math
import parseSEG, processSEG

const ns = 2001
block write:
  var w = openSeismicWriter("zoom.sgy", SegyOutputOptions())
  var x = newSeq[float32](ns)
  for i in 0 ..< 8:
    var k = initTraceKeys()
    k.offset = 100.0 * float64(i + 1)
    k.t0Ms = 0
    k.intervalUs = 2000
    k.status = tsLive
    for j in 0 ..< ns:
      let t = float64(j) * 0.002
      x[j] = float32(sin(2 * PI * 20 * t + float64(i)) + 0.2 * sin(2 * PI * 2 * t))
    w.writeTrace(k, x)
  discard w.close()

var f = openSeismic("zoom.sgy", keys = true)
let flow = initPipeline([bandPassOp(Corners(f1: 5, f2: 10, f3: 40, f4: 60))])
let shown = 1200 .. 1400                # the samples on screen
let win = decodeWindow(shown, flow.requiredMargin(dtMs = 2), ns)

var failed: seq[int]
var zoom = newTraceTensor[float32](f.len, win.len)
discard f.readTracesInto(0 ..< f.len, win, zoom, failed) # only the window is decoded
let meta = chunkMeta(f, 0 ..< f.len, win, failed)        # firstSample = win.a
doAssert meta.firstSample == win.a
# The window's width, and the record's length for tables on the time axis.
let plan = flow.compile(float32, meta.dtMs, samples = win.len, recordSamples = ns)
var ws = initWorkspace()
ws.reserve(plan)
plan.apply(zoom, meta, ws)
# Draw columns shown.a - win.a .. shown.b - win.a: they match the whole trace
# filtered, here to within 2.5 % of the traces' peak (1.2) next to the margin
# and far closer inside. (Steady sinusoids are the worst case; broadband data
# is within about 0.5 %.)
var whole = f.readTraces(0 ..< f.len)
let all = chunkMeta(f, 0 ..< f.len)
let full = flow.compile(float32, all.dtMs, samples = ns)
ws.reserve(full)
full.apply(whole, all, ws)
for r in 0 ..< f.len:
  for j in shown:
    doAssert abs(zoom[r, j - win.a] - whole[r, j]) < 0.025 * 1.2
f.close()
```

What the margin promises, and where it does not hold (an AGC or per-trace normalization
after a filter next to much stronger events, whole-trace statistics such as `demean`), is
in [integration.md, "Sample windows and edge effects"](../integration.md#sample-windows-and-edge-effects).
With parseSEG's prefetcher, decode the window through its `firstSample` and `cols`, and
`chunkMeta(c)` picks up `c.firstSample`.

## The process pool

`newProcessPool(workers)` starts persistent threads once (at viewer start-up); a job then
costs a wake-up, not a thread start. `pool.apply` takes the same arguments as
`plan.apply` without the workspace (each worker owns one), splits the rows across the
workers, and blocks until they are done. The result is identical to the single-thread
apply.

```nim
import parseSEG, processSEG

var keys = newSeq[TraceKeys](200)
var numSamples = newSeq[int](200)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].offset = 12.5 * float64(i)
  keys[i].t0Ms = 0
  keys[i].intervalUs = 2000
  keys[i].status = tsLive
  numSamples[i] = 1001
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 2)
var raw = zeros[float32](200, 1001)
for r in 0 ..< 200:
  for j in 0 ..< 1001: raw[r, j] = float32((r * 7 + j * 13) mod 17) - 8

let plan = initPipeline([detrendOp(), normalizeOp(normRms, perPanel),
                         agcOp(windowMs = 250)]).compile(float32, 2, 1001)
var pool = newProcessPool(workers = 4)  # once; cannot be copied, joined on destroy
var processed = zeros[float32](200, 1001)
pool.apply(plan, raw, processed, meta)  # per frame or chunk

var ws = initWorkspace()
ws.reserve(plan)
var want = zeros[float32](200, 1001)
plan.apply(raw, want, meta, ws)
for r in 0 ..< 200:
  for j in 0 ..< 1001: doAssert processed[r, j] == want[r, j]
```

One job runs at a time per pool; a second caller waits. Measured on a Ryzen 9 3950X
([bench/README.md](../../bench/README.md)): 6–12 µs per job, and a demean + band-pass + AGC
flow over a 2000 × 2001 float32 panel takes 32 ms on one thread, 8.6 ms on 4 workers.

## Your own threads

A viewer with its own job system calls `plan.apply` from each of its threads, with one
`Workspace` per thread and disjoint rows (`RowBlock.sub` with `ChunkMeta.sub`). A
per-panel normalization needs the whole panel, so such a plan is split into parts, and
each part that starts with one needs its factor computed over every row first. This is
what the pool does; here the two "threads" run one after the other:

```nim
import parseSEG, processSEG

var keys = newSeq[TraceKeys](64)
var numSamples = newSeq[int](64)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].intervalUs = 4000
  keys[i].status = tsLive
  numSamples[i] = 500
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 4)
var raw = zeros[float32](64, 500)
for r in 0 ..< 64:
  for j in 0 ..< 500: raw[r, j] = float32(r + 1) * float32(j mod 9)

let plan = initPipeline([demeanOp(), normalizeOp(normMax, perPanel),
                         gainDbOp(6)]).compile(float32, 4, 500)
var processed = zeros[float32](64, 500)
let src = rowBlock(raw)
let dst = rowBlock(processed)
plan.checkApply(src, dst, meta)         # once, on the caller's thread: may raise
var perThread = [initWorkspace(), initWorkspace()] # one per thread
for ws in perThread.mitems: ws.reserve(plan)
let halves = [0 ..< 32, 32 ..< 64]
for part in 0 ..< plan.partCount:       # 2: the normalization starts part 1
  var factor = NaN
  if plan.partNeedsPanel(part):         # over every row, on one thread
    factor = plan.panelFactorOf(part, (if part == 0: src else: dst), meta,
      perThread[0])
  for h, rows in halves:                # each half on its own thread
    let s = if part == 0: src.sub(rows) else: RowBlock[float32]() # copy in part 0 only
    plan.runPart(part, s, dst.sub(rows), meta.sub(rows), factor, perThread[h])

var ws = initWorkspace()
ws.reserve(plan)
var want = zeros[float32](64, 500)
plan.apply(raw, want, meta, ws)
for r in 0 ..< 64:
  for j in 0 ..< 500: doAssert processed[r, j] == want[r, j]
```

A plan holds seqs: share it by pointer (or keep it where every thread can reach it),
never copy it into a thread, since a copy allocates there. Metas and blocks can be copied
freely.
