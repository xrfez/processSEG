# Vibroseis: sweeps, pilots, correlation and record stacking

A vibroseis record holds the earth's response to a long sweep. Before it can be processed
it is compressed with the **pilot**, the reference sweep: by correlation (each reflection
becomes the Klauder wavelet) or by sweep deconvolution. This page goes through the steps
in their usual order:

1. make a sweep, or take a recorded pilot;
2. match the pilot to the data: the same sample interval, and the same instrument filter;
3. stack repeated sweeps at one source point (vertical or diversity), before or after
   correlation;
4. correlate;
5. for nodes that record continuously, cut the recording into records at the shot
   times first.

processSEG does the signal processing and returns values plus their header description;
parseSEG reads and writes the files (decision D14). [qc.md](qc.md#signal-to-noise-ratios)
covers the SNR measures, which help judge a stack or a correlation.

## Sweeps and pilots

A `SweepSpec` describes the sweep: the frequency law, start and end frequencies, the
length, the sample interval, the start and end tapers. The constructors validate it:
frequencies must stay below `maxNyquistFraction` of Nyquist (0.8 by default, under the
data's anti-alias filter), and the tapers may meet but not overlap.

| Law | Constructor | Spectrum |
|---|---|---|
| linear | `linearSweep(f1Hz, f2Hz, lengthMs, dtMs)` | flat |
| dB per octave | `dbPerOctaveSweep(..., dtMs, boostDb)` | rises by `boostDb` per octave |
| dB per Hz | `dbPerHzSweep(..., dtMs, boostDb)` | rises by `boostDb` per Hz |
| exponential | `exponentialSweep(f1Hz, f2Hz, lengthMs, dtMs)` | −3 dB per octave |
| table | `tableSweep(points, dtMs)` | your (time, frequency) points, linear between them |

`generatePilot` turns a spec into a `Pilot`, the type every later step takes. A pilot
recorded in the data (an auxiliary sweep or ground-force trace) becomes the same type
through `pilotFromRow` (a row of a block you read) or `pilotFromTrace` (a trace of a
file). `pilotKeys` and `sweepOverrides` give what parseSEG's writer needs to save a pilot
as SEG-Y with its sweep fields.

```nim
import std/os
import parseSEG, processSEG

let spec = dbPerOctaveSweep(f1Hz = 6, f2Hz = 96, lengthMs = 8000, dtMs = 2, boostDb = 3,
  taper = initSweepTaper(startMs = 300, endMs = 300, shape = tpBlackman))
doAssert spec.sweepLen == 4001                       # both ends included
doAssert abs(spec.frequencyAt(0) - 6) < 1e-9 and abs(spec.frequencyAt(8000) - 96) < 1e-9

let pilot = generatePilot(spec, float32)
doAssert pilot.len == 4001 and pilot.t0Ms == 0 and pilot.sweep.isSome

# Save it as a one-trace SEG-Y (trace code 6, sweep fields in bytes 127-140) ...
let path = getTempDir() / "guide_pilot.sgy"
var w = openSeismicWriter(path, SegyOutputOptions())
w.writeTrace(pilot.pilotKeys, pilot.samples, overrides = spec.sweepOverrides)
doAssert not w.close().isFatal

# ... and read it back as a recorded pilot. SEG-Y names only linear and exponential
# sweeps, so the dB/octave law comes back as samples without a sweep description.
var f = openSeismic(path, keys = true)
let back = pilotFromTrace(f, 0, float32)
doAssert back.samples == pilot.samples and back.dtMs == 2 and back.sweep.isNone
f.close()
removeFile(path)
```

The SEG-Y binary header has the same sweep fields (3233-3248). After writing,
`sweepOverrides(spec, binary = true)` gives them as `binary.` names for parseSEG's
`HeaderEditor.setFileField`.

## Matching the pilot to the data

**Sample interval.** Correlation refuses a pilot at another interval than the records.
Vibrator controllers often log the pilot at 0.5 or 1 ms while the data are at 2 ms:
`resamplePilot` converts it with a band-limited (Kaiser-windowed sinc) interpolation, and
refuses when the sweep reaches above the new cutoff. A pilot you generate yourself is best
made at the data's interval directly.

**Instrument filter.** The data reached the recorder through its anti-alias filter (and
perhaps a low-cut), often minimum phase. A recorded pilot went through the same filter; a
synthetic one did not, which leaves the filter's phase in the correlated data.
`applyInstrumentFilter` passes the pilot through an `InstrumentFilter` described as
Butterworth responses (the SEG-Y trace bytes 141-152 give the corners and slopes).

**Start time.** A recorded pilot can start a fraction of a sample after the time break
(`Pilot.t0Ms`). Correlation removes that start time exactly, so the output's time axis is
the record's.

```nim
import processSEG

# A pilot logged at 0.5 ms, for data at 2 ms.
let fine = generatePilot(linearSweep(8, 120, 6000, 0.5, initSweepTaper(250, 250),
  maxNyquistFraction = 0.25))
let pilot = fine.resamplePilot(dtMs = 2)
doAssert pilot.dtMs == 2 and pilot.len == 3001

# The recorder's filters, as the decimated 2 ms data went through them: a minimum-phase
# 3 Hz low-cut and 200 Hz high-cut, 18 dB/octave each.
let inst = InstrumentFilter(lowCutHz: 3, lowSlopeDbPerOctave: 18,
  highCutHz: 200, highSlopeDbPerOctave: 18, phase: ipMinimumPhase)
let matched = pilot.applyInstrumentFilter(inst)
doAssert matched.len == pilot.len and matched.dtMs == 2
```

## Correlation and sweep deconvolution

A `CorrelationPlan` holds the pilot's spectrum, computed once. The output is a new block
(the record length changes): by default the listen time, `recordSamples − pilot.len + 1`
columns. Two modes:

- `cmCorrelate` (the default): `X · conj(P)`, the matched filter. A reflection becomes
  the Klauder wavelet, zero phase, with the sweep's spectrum squared.
- `cmDeconvolve`: `X · conj(P) / (|P|² + ε)`, sweep deconvolution: a flat spectrum over
  the sweep's band, so a sharper wavelet, at the cost of the noise outside it; best with a
  ground-force pilot. `epsPercent` sets ε as a percentage of the peak `|P|²`.

With `normalize` (the default) the output is divided by its wavelet's peak, so a
pilot-shaped arrival of amplitude `a` correlates to `a` whatever the pilot's level or
length, and records correlated with different pilots compare.

```nim
import parseSEG, processSEG

const dt = 2.0
let pilot = generatePilot(linearSweep(10, 80, 4000, dt, initSweepTaper(200, 200)), float32)

# Three uncorrelated records of 6 s: reflectors at 300 and 900 ms, as the sweep arrives.
const n = 3001
var keys = newSeq[TraceKeys](3)
var counts = newSeq[int](3)
var raw = zeros[float32](3, n)
for r in 0 ..< 3:
  keys[r] = initTraceKeys()
  keys[r].status = tsLive
  keys[r].t0Ms = 0
  counts[r] = n
  for (atMs, amp) in [(300.0, 1.0), (900.0, -0.5)]:
    let first = int(atMs / dt)
    for j in 0 ..< pilot.len:
      raw[r, first + j] = raw[r, first + j] + float32(amp) * pilot.samples[j]
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = dt)

let plan = initCorrelationPlan(float32, pilot, recordSamples = n, dtMs = dt)
doAssert plan.outSamples == n - pilot.len + 1             # 1001: a 2 s listen time
var ws = initWorkspace()
ws.reserve(plan.workspaceSize)
var correlated = zeros[float32](3, plan.outSamples)
plan.correlateInto(raw, correlated, meta, ws)
doAssert abs(correlated[0, 150] - 1.0) < 1e-3              # 300 ms, amplitude 1
doAssert abs(correlated[2, 450] + 0.5) < 1e-3              # 900 ms, amplitude −0.5

# The same with sweep deconvolution, on every core.
let decon = initCorrelationPlan(float32, pilot, n, dt, mode = cmDeconvolve, epsPercent = 0.5)
var pool = newProcessPool(workers = 2)
var sharp = zeros[float32](3, decon.outSamples)
pool.correlateInto(decon, raw, sharp, meta)
doAssert abs(sharp[1, 150] - 1.0) < 1e-2
```

Dead and failed rows give zeros; live, unknown and auxiliary rows are correlated (a
recorded pilot correlates to the Klauder wavelet, a quick QC). The correlated block feeds
the usual pipeline: filters, gain, mutes.

## Stacking repeated sweeps

Several sweeps at one source point are stacked channel by channel, before or after
correlation. A `RecordStack` accumulates traces in any order; you say which output row
each record row goes to (`rowMap`, or one trace at a time with `add`).

- `rsVertical`: the mean per sample, counting only live traces.
- `rsDiversity`: each trace weighted per sample by the inverse of its power in a sliding
  window (`windowMs`), so a noise burst (traffic, wind) in one sweep is suppressed.
  `powerFloor` stops quiet windows from dominating; `clipLevel` clips spikes.

```nim
import std/math
import parseSEG, processSEG

const n = 1000
const sweeps = 4
var ws = initWorkspace()
var acc = initRecordStack[float64](rows = 1, samples = n, dtMs = 2,
  params = RecordStackParams(mode: rsDiversity, windowMs: 100))
ws.reserve(acc.workspaceSize)
for s in 0 ..< sweeps:
  var x = newSeq[float64](n)
  for j in 0 ..< n: x[j] = sin(2 * PI * 25 * float64(j) * 0.002)    # the coherent signal
  if s == 2:
    for j in 400 ..< 600: x[j] += 50 * sin(float64(j) * 1.7)       # a burst in one sweep
  doAssert acc.add(0, x, ws)
var stacked = zeros[float64](1, n)
acc.stackInto(stacked)
# Outside the burst the stack is the signal; inside it, the burst is largely gone.
doAssert abs(stacked[0, 100] - sin(2 * PI * 25 * 100 * 0.002)) < 1e-9
var leak = 0.0
for j in 400 ..< 600: leak = max(leak, abs(stacked[0, j] - sin(2 * PI * 25 * float64(j) * 0.002)))
doAssert leak < 1.0                                              # vertical: about 12
```

Record blocks go in with `addRecord(samples, meta, ws, rowMap)`, which adds only the live
rows. For threads, give each its own `RecordStack` and `merge` them at the end.

## Continuous recordings

Nodes record continuously, as consecutive segments per receiver. `planSlices` lays out
one trace per (receiver, shot) from the segments' start times (`TraceKeys.time`, or your
own array) and a window around each shot time. A receiver is its station (line and point
to 1/100 with the point index, the rule of [survey-qc.md](survey-qc.md)), and at one
station each channel and component is a stream of its own; your own array of receiver
ids overrides that. Sorting the segments into streams is the costly part (a quarter of a
second for a million rows) and does not depend on the shots: `initSliceStreams(meta)`
does it once, and `planSlices(streams, shots, window)` then costs only the traces of
those shots (`planSlices(meta, shots, window)` does both in one call). `extractInto`
copies and stitches the segments into a new block. The
plan lists `tracesNeeded`, so only those traces are read (`rowsNeeded` are the same rows
counted in the planning view, which are trace numbers only when the view starts at trace
0); gaps are zero-filled and reported; overlapping segments are trimmed (the later one
wins) and checked. Each output trace's keys (`sliceKeys`) carry the shot time as `time`
and the window start as `t0Ms`, so its time axis is relative to the source, as a shot
record's is. For uncorrelated data, make the window the listen time plus the sweep
length, then correlate.

```nim
import parseSEG, processSEG

# One receiver: three 2 s segments at 2 ms, the first starting at t = 1000 s.
const seg = 1000
var keys = newSeq[TraceKeys](3)
var counts = newSeq[int](3)
var data = zeros[float32](3, seg)
for r in 0 ..< 3:
  keys[r] = initTraceKeys()
  keys[r].status = tsLive
  keys[r].t0Ms = 0
  keys[r].intervalUs = 2000
  keys[r].time = 1000.0 + 2.0 * float64(r)
  keys[r].rcvLine = 1
  keys[r].rcvPoint = 7
  counts[r] = seg
  for j in 0 ..< seg: data[r, j] = float32(r * seg + j)       # the stream's sample index
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)

# A shot at t = 1001.5 s and a 1 s window from it: it straddles segments 0 and 1.
let streams = initSliceStreams(meta)                 # once per dataset: keep it
let plan = planSlices(streams, [initSliceShot(1001.5, id = 1)],
  SliceWindow(startMs: 0, lengthMs: 1000))
doAssert plan.len == 1 and plan.samples == 500 and plan.rowsNeeded == @[0, 1]
var report = initSliceReport(plan)
var ws = initWorkspace()
ws.reserve(sliceWorkspace(float32, plan))
var rec = zeros[float32](plan.len, plan.samples)
plan.extractInto(data, meta, rec, report, ws)
doAssert rec[0, 0] == 750 and rec[0, 499] == 1249 and report.gapSamples(0) == 0
let k = plan.sliceKeys(0)
doAssert k.time == 1001.5 and k.t0Ms == 0 and k.record == 1
```

Clock drift is corrected before planning, on the times: `driftCorrectedTimes(times, t0,
offset0, t1, offset1)` for a linear drift between two clock checks of one receiver. The
default is no correction: land nodes are normally GPS-disciplined and corrected when
harvested.
