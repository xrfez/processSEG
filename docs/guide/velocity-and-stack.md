# Velocities, moveout, statics and stacking

From velocity picks to a stacked section: velocity functions and fields, the formats they
are read from, moveout overlays, datum statics, NMO, CMP binning and stacking, by hand
with a `StackAccumulator` or for a whole file with `bruteStack`. Velocities are RMS
velocities against two-way time, in distance units per second (the file's coordinate
units); times are absolute ms on each trace's own axis (`t0Ms`).

## Velocity functions and fields

A `VelocityFunction` is a constant, a linear interval-velocity gradient, RMS knots, or
layers. Knots are interpolated linearly in V²·t (so the interval velocity is constant
between knots), and a `VelocityField` holds functions at nodes resampled onto one time
grid, 2D (CDP numbers or distances) or 3D (an inline × crossline grid).

```nim
import parseSEG, processSEG

let c = constantVelocity(2000)
let g = gradientVelocity(v0 = 1800, k = 600)   # Vint = 1800 + 600·t(s)
let k = rmsVelocity([VelocityKnot(tMs: 0, vrms: 1500),
                     VelocityKnot(tMs: 1000, vrms: 2000),
                     VelocityKnot(tMs: 2000, vrms: 2500)])
let layers = intervalVelocity(tMs = [1000.0, 2000], vint = [1500.0, 2500])
doAssert c.vrmsAt(1234) == 2000 and g.vintAt(1000) == 2400
doAssert abs(layers.vintAt(1500) - 2500) < 1e-6
doAssert abs(k.vrmsAt(1000) - 2000) < 1e-9

var vint = newSeq[float64](3)               # Dix, and back
dixIntervalInto([500.0, 1000, 2000], [1500.0, 2000, 2500], vint)
var vrms = newSeq[float64](3)
rmsFromIntervalInto([500.0, 1000, 2000], vint, vrms)
doAssert abs(vrms[2] - 2500) < 1e-6

# A 2D field: functions at CDPs 100 and 200, on a 4 ms grid to 3 s.
let field = initVelocityField(locations = [100.0, 200], functions = [c, k],
  dtMs = 4, tMaxMs = 3000)
doAssert field.kind == fk2D and field.len == 2
doAssert abs(field.vrmsAt(location = 150, tMs = 1000) - 2000) < 1e-6
var column = newSeq[float64](751)
field.vrmsInto(location = 120, startMs = 0, dtMs = 4, vrms = column)
```

Velocity files load into `VelocityPick`s: `readVelocities(path)` (or `parseVelocities`
on text) detects HANDVEL blocks, VELF cards or CSV; `initVelocityField(picks, dtMs,
tMaxMs)` builds a 2D field by CDP, or a 3D one when every pick has an inline and
crossline (they must cover a full grid).

```nim
import parseSEG, processSEG

let handvel = """
HANDVEL     1001
        0     1500      500     1800
     1000     2100
HANDVEL     1101
        0     1550     1000     2200
"""
let picks = parseVelocities(handvel)
doAssert picks.len == 2 and picks[0].cdp == 1001 and picks[0].inline.isNaN
let field = initVelocityField(picks, dtMs = 4, tMaxMs = 4000)
doAssert abs(field.vrmsAt(1001, 500) - 1800) < 1e-6

let csv = "inline,crossline,tMs,vrms\n10,1,0,1500\n10,1,2000,2500\n" &
  "10,2,0,1500\n10,2,2000,2600\n"
let field3 = initVelocityField(parseVelocityCsv(csv), dtMs = 4, tMaxMs = 2000)
doAssert field3.kind == fk3D and field3.vrmsAt(10, 2, 2000) == 2600
doAssertRaises ValueError:              # errors name the line
  discard parseVelocities("HANDVEL 1\n0 1500 500\n")
```

## Moveout overlays

The viewer draws ticks over a gather to check geometry and velocities. A `Moveout` is a
linear (`t0 + |x|/v`) or hyperbolic (`sqrt(t0² + x²/v²)`) curve; `moveoutTimesInto`
fills one time per row from its offset, and `moveoutColumnsInto` the same as fractional
columns of the block (each row's `t0Ms` and the window's `firstSample` applied), ready to
draw. A NaN offset gives a NaN tick. Tag the ticks with the file's `keysRevision`.

```nim
import parseSEG, processSEG

var keys = newSeq[TraceKeys](24)
var numSamples = newSeq[int](24)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].offset = 50.0 * float64(i + 1)
  keys[i].t0Ms = -40                    # recording started 40 ms before time zero
  keys[i].intervalUs = 2000
  keys[i].status = tsLive
  numSamples[i] = 1501
keys[3].offset = NaN                    # no geometry yet
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 2, firstSample = 100)

var ticks = newSeq[float64](meta.len)   # once per chunk, reused
moveoutTimesInto(meta, linearMoveout(velocity = 2500, t0Ms = 10), ticks)
doAssert ticks[0] == 10 + 1000 * 50 / 2500 and ticks[3].isNaN
moveoutColumnsInto(meta, hyperbolicMoveout(2200, t0Ms = 800), ticks)
doAssert abs(meta.sampleTimeMs(0, int(ticks[0])) - nmoTime(50, 2200, 800)) < 2

let field = initVelocityField([1.0, 2.0], [constantVelocity(2000),
  constantVelocity(3000)], dtMs = 4, tMaxMs = 3000)
for k in keys.mitems: k.ensemble = 2    # the field is read at each row's CDP
moveoutTimesInto(meta, field, t0Ms = 1000, ticks)
doAssert abs(ticks[0] - nmoTime(50, 3000, 1000)) < 1e-9
```

To flatten first breaks on the display, `applyLmo(samples, meta, velocity, biasMs)`
shifts every live row up by `1000·|x|/v − biasMs` ms in place, and `removeLmo` shifts it
back (apart from what moved out of the row). It is a display transform, not a pipeline
op.

## Datum statics

A static is the time **added** to every event of a trace: the sample at `t` moves to
`t + static`. A station above the datum delays the events, so its datum static is
negative. `staticsInto` computes one per row from the keys' `srcZ` and `rcvZ` (NaN when
either is unknown, and a NaN static leaves its row alone); parseSEG's `traceStatics(f,
traces)` reads the header datums, depths and recorded statics for the other forms.

```nim
import parseSEG, processSEG

var keys = newSeq[TraceKeys](3)
var numSamples = @[500, 500, 500]
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].t0Ms = 0
  keys[i].intervalUs = 2000
  keys[i].status = tsLive
  keys[i].srcZ = 120
  keys[i].rcvZ = 110 + 10 * float64(i)
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 2)
let d = datumStatics(datumZ = 100, vReplacement = 2000)
var st = newSeq[float64](meta.len)
staticsInto(meta, d, st)
doAssert st[0] == -(20 + 10) * 1000 / 2000 # -15 ms

var samples = zeros[float32](3, 500)
for r in 0 ..< 3: samples[r, 200] = 1   # an event at 400 ms
samples.applyStatics(meta, st, siCubic)
doAssert samples[0, 200 - 7] > 0.4      # moved up by 15 ms (7.5 samples)
samples.applyStatics(meta, d)           # the datum form: no statics buffer
```

## NMO and stacking by hand

`applyNmo` corrects each live row in place from its own `t0Ms`, with a `VelocityFunction`
or a `VelocityField` read at each row's location (the keys' `ensemble` in 2D, `inline`
and `crossline` in 3D), and mutes where the stretch exceeds `maxStretch`. (The kernel
under it is `nmoInto`, one row with a velocity per output sample; to correct many rows
with one velocity column yourself, convert the column once with `vrmsToSloth` and call
`nmoSlothInto`, which gives the same samples without a division per sample.) A
`StackAccumulator` then sums traces into bins in any order with a **per-sample** fold:
`addLive` counts only the span between a trace's first and last non-zero samples, so
muted zeros do not count.

```nim
import std/math
import parseSEG, processSEG

const dtMs = 4.0
const ns = 751
proc ricker(tauMs, freqHz: float64): float64 =
  let a = (PI * freqHz * tauMs / 1000) ^ 2
  (1 - 2 * a) * exp(-a)

# Three CMP gathers (CDPs 10, 11, 12) of 8 offsets, one reflection at 1200 ms.
var keys: seq[TraceKeys]
var numSamples: seq[int]
for cdp in 10 .. 12:
  for i in 0 ..< 8:
    var k = initTraceKeys()
    k.ensemble = float64(cdp)
    k.offset = 100 + 200 * float64(i)
    k.t0Ms = 0
    k.intervalUs = dtMs * 1000
    k.status = tsLive
    keys.add k
    numSamples.add ns
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs)
let v = constantVelocity(2200)
var samples = zeros[float32](keys.len, ns)
for r in 0 ..< keys.len:
  let t = nmoTime(keys[r].offset, 2200, 1200)
  for j in 0 ..< ns: samples[r, j] = float32(ricker(float64(j) * dtMs - t, 25))

var ws = initWorkspace()
ws.reserve(nmoWorkspace(float32, ns))
samples.applyNmo(meta, v, initNmoParams(siCubic, maxStretch = 0.3, taperMs = 20), ws)
for r in 0 ..< keys.len:                # flat at 1200 ms on every offset
  doAssert abs(samples[r, 300] - 1) < 0.02

let binning = cdpBinning(firstCdp = 10, count = 3)
var bins = newSeq[int](meta.len)
binsInto(meta, binning, bins)           # -1 for a trace in no bin
var acc = initStackAccumulator[float32](binning.len, ns, dtMs)
let b = rowBlock(samples)
for r in countdown(meta.len - 1, 0):    # any order
  if meta.isLive(r):
    discard acc.addLive(bins[r], b.row(r))
var stack = zeros[float32](acc.bins, acc.samples)
var fold = zeros[int32](acc.bins, acc.samples)
acc.stackInto(stack, fnFold)            # mean per sample
acc.foldInto(fold)
doAssert abs(stack[1, 300] - 1) < 0.02 and fold[1, 300] == 8
doAssert fold[1, 50] < 8                # the stretch mute removed far offsets early
doAssert binning.location(2).cdp == 12
```

`stackInto` and `foldInto` copy the stack into your tensors and leave the accumulator
open for more traces. For a large stack, `acc.finishInto(data, fold, norm)` normalizes in
place and hands the accumulator's storage over instead, so the stack is held once.

With threads, either correct the traces on the workers and add them on one thread in a
fixed order (what `bruteStack` does: the sums then do not depend on the threads), or give
each thread its own accumulator and `merge` them at the end (the last bits of the sums
then depend on which thread got which trace).

## A brute stack of a file

`bruteStack` does all of this over a file: it reads through parseSEG's prefetcher
(`uniform = true`) and, on each decode worker while the chunk is in cache, applies
optional datum statics, the pre-stack flow (`opts.flow`, a `Pipeline`) and NMO onto the
stack's time axis; the calling thread takes the corrected chunks in file order and stacks
each trace into its bin. Velocities come from a function, or from a field read at each
bin's location. `writeStack` saves the result as SEG-Y, one
trace per bin, whatever the input standard. To write it your own way (other options,
extra header overrides, some bins only), give parseSEG's `writeTrace(w, keys, samples,
overrides)` each bin's `stackKeys(res, bin)` and `stackOverrides(res, bin)`: processSEG
describes its results and parseSEG writes them.

```nim
import std/math
import parseSEG, processSEG

proc ricker(tauMs, freqHz: float64): float64 =
  let a = (PI * freqHz * tauMs / 1000) ^ 2
  (1 - 2 * a) * exp(-a)

const ns = 1001
block write:                            # 20 CMPs of 12 offsets, events at 600 and 1400 ms
  var w = openSeismicWriter("cmps.sgy", SegyOutputOptions())
  var x = newSeq[float32](ns)
  for cdp in 1 .. 20:
    for i in 0 ..< 12:
      var k = initTraceKeys()
      k.ensemble = float64(cdp)
      k.offset = 50 + 150 * float64(i)
      k.t0Ms = 0
      k.intervalUs = 2000
      k.status = tsLive
      for j in 0 ..< ns:
        x[j] = float32(ricker(float64(j) * 2 - nmoTime(k.offset, 2000, 600), 20) +
          ricker(float64(j) * 2 - nmoTime(k.offset, 2400, 1400), 20))
      w.writeTrace(k, x)
  discard w.close()

var f = openSeismic("cmps.sgy", keys = true)
let velocity = rmsVelocity([VelocityKnot(tMs: 600, vrms: 2000),
                            VelocityKnot(tMs: 1400, vrms: 2400)])
var opts = initBruteStackOptions(
  flow = initPipeline([bandPassOp(Corners(f1: 4, f2: 8, f3: 50, f4: 70))]),
  nmo = initNmoParams(siCubic, maxStretch = 0.5, taperMs = 20))
opts.workers = 2
var calls = 0
let res = bruteStack(f, float32, cdpBinning(firstCdp = 1, count = 20), velocity, opts,
  progress = proc (done, total: int64): bool =
    inc calls                           # on the calling thread, after each chunk
    true)                               # false cancels: SegCancelledError
doAssert res.stacked == 240 and res.skipped == 0 and res.failed == 0
doAssert res.data.shape == [20, ns] and res.fold[5, 700] == 12
doAssert res.binTraces[5] == 12         # traces per bin: a fold map
doAssert res.data[5, 300] > 0.8 and res.data[5, 700] > 0.8 and calls > 0

var near: seq[int]                      # a list of traces: the near offsets only
for t in 0 ..< f.len:
  if f.traceKeys(t).offset < 500: near.add t
let nearStack = bruteStack(f, float32, near, cdpBinning(firstCdp = 1, count = 20),
  velocity, opts)
doAssert nearStack.stacked == near.len and nearStack.fold[5, 700] == 3

let log = writeStack("stack.sgy", res)  # SEG-Y Rev 2.1, trace sorting 4
doAssert not log.isFatal
var s = openSeismic("stack.sgy", keys = true)
doAssert s.len == 20 and s.traceKeys(4).ensemble == 5
s.close()
f.close()
```

- Run `bruteStack` on a background thread of your own; it blocks until done, and calls
  `progress` on that thread. The job only reads the file. While it runs, other threads
  may read the same `SeismicFile` (`readTracesInto`, `traceKeys`, views, the prefetchers
  they have) and, with parseSEG 0.10.0 or later, create and close prefetchers on it
  (before 0.10.0 the map's pin count is not thread-safe: create them before the job
  starts and close them after it has returned). No thread may `applySurvey`, edit
  headers or `close` the file until `bruteStack` has returned or raised: cancel through
  `progress` and wait for it first.
- The traces are summed in file order whatever `opts.workers` is, so the stack is the
  same bit for bit for any number of workers (only a `perPanel` normalization in the
  flow depends on `opts.chunkTraces`).
- `res.skipped` counts live traces outside every bin or with a NaN offset, `t0Ms` or
  static, no interval, non-finite samples, or fully stretch-muted.
- `res.binTraces[bin]` is the number of traces stacked into each bin (a fold map), and
  `res.cmpX[bin]`, `res.cmpY[bin]` their mean midpoint: with `cdpBinning` or
  `inlineCrosslineBinning` that is where the bin lies on a map, and what `stackKeys` and
  `writeStack` give as its CDP X/Y (line and grid bins keep their centre).
- Memory: one stack of bins × samples × (sizeof(T) + 4) bytes, whatever the number of
  workers (the sums are normalized in place and become the result), plus the prefetch
  ring and each worker's scratch and velocity columns (`opts.velocityCacheBytes`, 8 MiB:
  a field is read once per bin while a shot's bins fit it; raise it for 3D shots spread
  over thousands of bins); `bruteStackMemory` estimates it. A float32 3D stack of
  100 000 bins × 2001 samples is 1.6 GB.
- Part of a file: pass a range, `bruteStack(f, float32, 0 ..< 5000, binning, ...)`, or a
  list of trace indices (a map selection, one channel set of a mixed SEG-D record),
  `bruteStack(f, float32, picked, binning, ...)`; a list is stacked in its own order.
- 3D: `gridBinning` (a rotated grid of midpoints) or `inlineCrosslineBinning` (header
  numbers), and a 3D `VelocityField`.
