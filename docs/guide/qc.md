# Geometry QC and first breaks

Three levels of geometry checking, from the cheapest: flags from the keys alone, first
breaks picked on the samples, and a robust refraction fit per shot that flags traces
whose header position does not match their arrival times. All three work on a
`ChunkMeta` (and samples), so they run on a viewer's chunks or in a batch loop over a
file's shots; the QC modules are experimental for land data. Offsets and distances are in
the file's coordinate units, velocities in those units per second.

## Flags from the keys

`checkGeometryInto` compares each live row's header offset with the source–receiver
distance from the coordinates, and the header midpoint with the halfway point, within the
`GeometryQc` tolerance (by default the larger of 1 unit and 1 % of the distance).
`flaggedRows` lists the rows with any of the `badGeometry` flags: no offset, a zero
offset, an offset that disagrees with the coordinates.

```nim
import parseSEG, processSEG

var keys = newSeq[TraceKeys](12)
var numSamples = newSeq[int](12)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].srcX = 1000
  keys[i].srcY = 0
  keys[i].rcvX = 1000 + 50 * float64(i - 6) # a split spread
  keys[i].rcvY = 0
  keys[i].offset = keys[i].rcvX - keys[i].srcX # signed, as SEG-Y may hold it
  keys[i].cmpX = (keys[i].srcX + keys[i].rcvX) / 2
  keys[i].cmpY = 0
  keys[i].intervalUs = 2000
  keys[i].status = tsLive
  numSamples[i] = 1000
keys[6].rcvX = 1025                     # back the zero offset off the source
keys[6].offset = 25
keys[6].cmpX = 1012.5
keys[2].offset = -150                   # the header disagrees with the coordinates
keys[9].offset = NaN                    # no offset at all
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 2)

var checks = newSeq[GeometryCheck](meta.len)
checkGeometryInto(meta, checks)
doAssert gfOffsetMismatch in checks[2].flags and checks[2].offsetError == -50
doAssert gfNegativeOffset in checks[0].flags # normal on one side of a split spread
doAssert flaggedRows(checks) == @[2, 9]
doAssert flaggedRows(checks, {gfNegativeOffset}) == @[0, 1, 2, 3, 4, 5]
```

Add `gfMidpointMismatch` to the flags when the file holds true midpoints (processed and
3D SEG-Y often hold the bin centre as the CDP X/Y).

## First breaks

A `FirstBreakPlan` picks every live row: an optional minimum-phase band-pass
(`lowCutHz`, `highCutHz`), one of three pickers on the trace energy (`pmStaLta`,
`pmEnergyRatio`, `pmCoppens`), in a fixed search window (`searchStartMs`, `searchEndMs`)
or one guided by a linear moveout (`guideVelocity`, `guideT0Ms`, `beforeMs`,
`afterMs`). Each pick has a time (on the meta's time axis), a fractional column and a
confidence from 0 to 1, mapped from the SNR around the pick (`lowDb` .. `highDb`). The
input samples are not changed: the workspace holds the filtered copy.

```nim
import std/[math, random]
import parseSEG, processSEG

const dtMs = 2.0
const ns = 750
var keys = newSeq[TraceKeys](48)
var numSamples = newSeq[int](48)
var truth = newSeq[float64](48)
var samples = zeros[float32](48, ns)
var rng = initRand(7)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].offset = 25 * float64(i + 1)
  keys[i].t0Ms = 0
  keys[i].intervalUs = dtMs * 1000
  keys[i].status = tsLive
  numSamples[i] = ns
  truth[i] = min(1000 * keys[i].offset / 1000, 1000 * keys[i].offset / 3000 + 80)
  for j in 0 ..< ns:                    # a causal 30 Hz onset in weak noise
    let tau = (float64(j) * dtMs - truth[i]) / 1000
    let onset = if tau >= 0: sin(2 * PI * 30 * tau) * exp(-tau * 1000 / 30) else: 0.0
    samples[i, j] = float32(onset + rng.gauss(0, 0.02))
keys[20].status = tsDead
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs)

let params = FirstBreakParams(pickMethod: pmCoppens,
  guideVelocity: 3000, guideT0Ms: 80, beforeMs: 80, afterMs: 80) # the refractor
let plan = initFirstBreakPlan(float32, params, dtMs, ns)
var ws = initWorkspace()
ws.reserve(plan.workspaceSize)
var picks = newSeq[FirstBreakPick](meta.len) # once per chunk, reused
plan.pickInto(samples, meta, ws, picks)
for r in 0 ..< meta.len:
  if r == 20:
    doAssert picks[r].timeMs.isNaN and picks[r].confidence == 0 # not live: noPick()
  else:
    doAssert abs(picks[r].timeMs - truth[r]) < 3 and picks[r].confidence > 0.9
```

The overlay is `picks[row].timeMs` (or `.col` over the block). For one trace,
`plan.pick(x, ws, offset, t0Ms)` gives the same pick; the plan is shareable across
threads, one `Workspace` each. The module documentation compares the pickers and their
biases (energy arrival vs onset).

## A refraction fit per shot

`fitShot` fits the confident picks of one shot (a `meta.sub` of its rows) against the
header `|offset|`: a two-segment L1 fit with a crossover (`rmTwoSegment`) or a median per
offset bin (`rmOffsetBinned`). Residuals are then taken against the neighbours along the
spread, which removes a refractor that varies smoothly; a trace whose header puts it at
another station jumps, and is `suspect`. With samples, each trace is also
cross-correlated with its neighbours, which catches traces without a confident pick.
`asymmetryMs` compares the two sides of a split spread (a misplaced source).

```nim
import std/[math, random]
import parseSEG, processSEG

const dtMs = 2.0
const ns = 750
proc breakMs(offset: float64): float64 =
  let d = abs(offset)
  min(1000 * d / 1000, 1000 * d / 3000 + 80) # direct wave, then a refractor

var keys = newSeq[TraceKeys](96)
var numSamples = newSeq[int](96)
var samples = zeros[float32](96, ns)
var rng = initRand(3)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].srcX = 5000
  keys[i].srcY = 0
  keys[i].rcvX = 5000 + 25 * float64(i - 48) + (if i >= 48: 25 else: 0)
  keys[i].rcvY = 0
  keys[i].offset = keys[i].rcvX - keys[i].srcX
  keys[i].channel = float64(i + 1)
  keys[i].t0Ms = 0
  keys[i].intervalUs = dtMs * 1000
  keys[i].status = tsLive
  numSamples[i] = ns
  for j in 0 ..< ns:
    let tau = (float64(j) * dtMs - breakMs(keys[i].offset)) / 1000
    let onset = if tau >= 0: sin(2 * PI * 30 * tau) * exp(-tau * 1000 / 30) else: 0.0
    samples[i, j] = float32(onset + rng.gauss(0, 0.02))
# Two receivers' headers trade places, five stations apart: the recorded
# traces stay where they were.
swap(keys[70].rcvX, keys[75].rcvX)
swap(keys[70].offset, keys[75].offset)
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs)

let plan = initFirstBreakPlan(float32, FirstBreakParams(pickMethod: pmCoppens), dtMs, ns)
var ws = initWorkspace()
ws.reserve(plan.workspaceSize)
var picks = newSeq[FirstBreakPick](meta.len)
plan.pickInto(samples, meta, ws, picks)

var fit: ShotFit                        # its seqs are resized and reused per shot
fitShot(samples, meta, picks, ShotFitParams(), fit)
var suspect: seq[int]
for row, bad in fit.suspect:
  if bad: suspect.add row
doAssert suspect == @[70, 75]
doAssert abs(fit.v1 - 1000) < 50 and abs(fit.v2 - 3000) < 150
doAssert abs(fit.asymmetryMs) < 2
```

What a swap must exceed to be found: about `Δx / v` against the model (`residualMs`,
12 ms by default) or against the neighbours' lags (`lagToleranceMs`, 8 ms). An
adjacent-channel swap on a fast refractor is below both; lower the thresholds for clean
data and short groups. Pass each receiver line of a 3D shot separately. `fitShot`
allocates a little scratch per call: it is QC, not a per-frame kernel.

## Signal-to-noise ratios

`qc/snr` measures the SNR two ways, in dB. Per trace, `windowSnrInto` compares the mean
power in a signal window with that in a noise window, at fixed times or relative to a
pick or an LMO guide: `seRatio` is the classic energy ratio (noise alone gives 0 dB, the
first-break confidence uses it), `seExcess` subtracts the noise power and estimates the
true signal over noise. Per gather, `coherenceSnrInto` estimates the SNR versus time from
how alike the flattened (NMO- or LMO-corrected) traces are: the mean trace is the signal,
the residuals the noise. An estimate of no signal returns the floor (`floorDb`, −40 dB by
default); NaN means it could not be measured (a dead row, an empty window).

```nim
import std/[math, random]
import parseSEG, processSEG

const dtMs = 2.0
const ns = 500
proc breakMs(offset: float64): float64 = 40 + 1000 * abs(offset) / 2500

var keys = newSeq[TraceKeys](60)
var numSamples = newSeq[int](60)
var samples = zeros[float32](60, ns)
var rng = initRand(11)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].offset = 20 * float64(i + 1)
  keys[i].t0Ms = 0
  keys[i].intervalUs = dtMs * 1000
  keys[i].status = tsLive
  numSamples[i] = ns
  for j in 0 ..< ns:                    # a 25 Hz Ricker on the LMO line, in noise
    let tau = (float64(j) * dtMs - breakMs(keys[i].offset)) / 1000
    let a = (PI * 25 * tau) ^ 2
    samples[i, j] = float32((1 - 2 * a) * exp(-a) + rng.gauss(0, 0.1))
keys[7].status = tsDead
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs)

# Per trace: ±20 ms around the guide over 100 .. 60 ms before it.
let guide = linearMoveout(velocity = 2500, t0Ms = 40)
let p = WindowSnrParams(signalStartMs: -20, signalEndMs: 20,
  noiseStartMs: -100, noiseEndMs: -60, estimator: seExcess)
var perTrace = newSeq[float64](meta.len)
windowSnrInto(samples, meta, p, guide, perTrace)
doAssert perTrace[7].isNaN              # not live
doAssert perTrace[30] > 5

# Per gather: flatten the event at 40 + 60 ms, then the SNR versus time.
var flat = samples.clone
flat.applyLmo(meta, velocity = 2500, biasMs = 60)
var ws = initWorkspace()
ws.reserve(coherenceSnrWorkspace(ns))
var perSample = newSeq[float64](ns)
coherenceSnrInto(flat, meta, windowMs = 20, ws, perSample)
doAssert perSample[50] > 5              # 100 ms: the flattened event
doAssert perSample[300] <= -10          # 600 ms: noise only
```

The windowed estimate needs long windows at low SNR (about 10⁵ samples per window for
±1 dB at −10 dB, 10³ at 0 dB); the coherence estimate pools traces, so a window of a few
hundred samples on 100 traces does the same. Coherence counts each row over its
`liveSpan`, so mutes and padding are not taken for signal or noise; residual moveout and
amplitude changes along the gather count as noise.
