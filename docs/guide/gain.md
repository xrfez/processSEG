# Display gain, statistics and spectra

Viewers offer fixed, normalized and AGC gain. Fixed and normalized gain are one factor
per trace or per panel, so a GPU viewer can apply them as a shader uniform from
statistics processSEG computes; AGC and time-variant curves change samples with time and
run on the CPU, on their own or as pipeline ops (`gainDbOp`, `normalizeOp`, `agcOp`,
`gainCurveOp`). See [integration.md, "Display gain"](../integration.md#display-gain-cpu-or-gpu)
for the viewer's side.

## Fixed gain, normalization and AGC

```nim
import std/math
import parseSEG, processSEG

const dtMs = 2.0
const ns = 1501
var keys = newSeq[TraceKeys](10)
var numSamples = newSeq[int](10)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].t0Ms = 0
  keys[i].intervalUs = dtMs * 1000
  keys[i].status = tsLive
  numSamples[i] = ns
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs)
var samples = zeros[float32](keys.len, ns)
for r in 0 ..< keys.len:                # a 30 Hz sine decaying with time
  for j in 0 ..< ns:
    let t = float64(j) * dtMs / 1000
    samples[r, j] = float32(float64(r + 1) * sin(2 * PI * 30 * t) * exp(-2 * t))

samples.scaleDb(meta, 6.0206)           # fixed gain: ×2
doAssert abs(dbToFactor(-6.0206) - 0.5) < 1e-4

var ws = initWorkspace()
let norm = initNormalizePlan(normMax, ns, perTrace, target = 1)
ws.reserve(norm.workspaceSize)          # none unless normPercentile
samples.normalize(meta, norm, ws)       # each trace's max |x| becomes 1
var peak = 0.0
for j in 0 ..< ns: peak = max(peak, abs(float64(samples[4, j])))
doAssert abs(peak - 1) < 1e-6

let agcPlan = initAgcPlan(windowMs = 400, dtMs, ns) # agcRms by default
ws.reserve(agcPlan.workspaceSize)
var gains = zeros[float32](keys.len, ns)
var before = samples.clone()
samples.agc(meta, agcPlan, ws, gains)   # keep each sample's gain
var late = 0.0                          # the decay is gone: RMS ~1 late and early
for j in 1200 ..< 1300: late += float64(samples[0, j])^2 / 100
doAssert abs(sqrt(late) - 1) < 0.1
samples.removeGain(meta, gains)         # and back
doAssert abs(samples[0, 700] - before[0, 700]) < 1e-4
doAssert agcPlan.requiredMargin == agcPlan.windowSamples div 2 # half the window
```

`agcMeanAbs` divides by the window's mean |x| instead of its RMS. AGC costs 4.5–7 ms
per 2000 × 2001 float32 panel on one thread, for windows of 100 ms to 2 s
([bench/README.md](../../bench/README.md)).

## Time-variant gain curves

A curve (`knotGain`, `powerGain`, `exponentialGain`) is a gain against absolute time. A
`GainCurvePlan` tabulates it once per sample interval and record length; applying it
multiplies each live row by the table at the row's own times (its `t0Ms`, the meta's
`firstSample`).

```nim
import std/math
import parseSEG, processSEG

var keys = @[initTraceKeys(), initTraceKeys()]
for k in keys.mitems:
  k.intervalUs = 4000
  k.status = tsLive
keys[0].t0Ms = 0
keys[1].t0Ms = 1000                     # this trace starts 1 s later
let numSamples = @[1001, 1001]
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 4)
var samples = ones[float32](2, 1001)

let curve = knotGain([GainKnot(tMs: 0, dB: 0), GainKnot(tMs: 4000, dB: 24)])
doAssert abs(curve.gainDbAt(2000) - 12) < 1e-9
# The second trace runs to 5000 ms: widen the table to cover it.
let plan = initGainCurvePlan(float32, curve, dtMs = 4, samples = 1001, tMaxMs = 5000)
samples.applyGainCurve(meta, plan)
doAssert abs(samples[0, 500] - dbToFactor(12)) < 1e-4  # 2000 ms
doAssert abs(samples[1, 0] - dbToFactor(6)) < 1e-4     # 1000 ms

let spherical = powerGain(2)            # (t / 1 s)², unity at 1000 ms
doAssert abs(spherical.factorAt(2000) - 4) < 1e-9
```

## Statistics for a shader uniform

`traceStatsInto` gives RMS and max |x| per row (NaN for dead, auxiliary and failed rows);
`panelStats` gives them over the live rows of a panel or a selection of rows, with a
percentile clip value from a histogram (0.4 % relative resolution). Recompute them when
the data or the selection changes, not per frame: about 0.9 ms and 4 ms for a
2000 × 2001 panel.

```nim
import std/math
import parseSEG, processSEG

var keys = newSeq[TraceKeys](6)
var numSamples = newSeq[int](6)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].intervalUs = 2000
  keys[i].status = tsLive
  numSamples[i] = 1000
keys[5].status = tsAuxiliary
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 2)
var samples = zeros[float32](6, 1000)
for r in 0 ..< 6:
  for j in 0 ..< 1000: samples[r, j] = float32(r + 1) * float32(j mod 10 - 5)

var stats = newSeq[TraceStats](meta.len)
traceStatsInto(samples, meta, stats)
doAssert stats[0].maxAbs == 5 and stats[5].rms.isNaN # auxiliary: no statistics

var ws = initWorkspace()
ws.reserve(panelStatsWorkspace)
let p = panelStats(samples, meta, ws, clipPercentile = 99)
doAssert p.rows == 5 and p.maxAbs == 25 # the auxiliary row is left out
let gainUniform = 1 / p.clip            # display: clip at p.clip, scale by 1/p.clip
let sel = panelStats(samples, meta, ws, clipPercentile = NaN, rows = 0 .. 1)
doAssert sel.rows == 2 and sel.clip.isNaN # a selection; no histogram pass

# The factor a per-panel normalization would apply, as a uniform:
let norm = initNormalizePlan(normRms, samples = 1000, scope = perPanel)
let factor = panelFactor(rowBlock(samples), meta, norm, ws)
doAssert abs(factor - 1 / p.rms) < 1e-6 * factor
doAssert gainUniform > 0
```

## Amplitude spectra

A `SpectrumPlan` is made once per window length and sample interval; it demeans,
Tukey-tapers and zero-pads each window (to the next power of two by default), and
averages power over the live rows of a selection. Amplitudes are scaled so that a
sinusoid of amplitude A reads A. It does not change the samples, so it runs on the raw or
the processed buffer.

```nim
import std/math
import parseSEG, processSEG

const dtMs = 2.0
var keys = newSeq[TraceKeys](4)
var numSamples = newSeq[int](4)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].t0Ms = 0
  keys[i].intervalUs = dtMs * 1000
  keys[i].status = tsLive
  numSamples[i] = 1000
let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs)
var samples = zeros[float32](4, 1000)
for r in 0 ..< 4:
  for j in 0 ..< 1000:
    samples[r, j] = float32(3 * sin(2 * PI * 31.25 * float64(j) * dtMs / 1000))

let sp = initSpectrumPlan(float32, windowLen = 512, dtMs)
var ws = initWorkspace()
ws.reserve(sp.workspaceSize)
var amp = newSeq[float64](sp.bins)
var db = newSeq[float64](sp.bins)
let firstCol = 200                      # the user's time window, in block columns
let used = sp.averageAmplitudeInto(samples, meta, firstCol, amp, ws, rows = 0 .. 3)
toDbInto(amp, db)                       # 0 dB at the peak; or dbAbsolute
var k = 0
for i in 0 ..< sp.bins:
  if amp[i] > amp[k]: k = i
doAssert used == 4 and sp.binHz(k) == 31.25 and abs(amp[k] - 3) < 0.05
doAssert db[k] == 0
```

`averageAmplitudeInto` returns the rows it used: rows with a NaN or Inf in the window,
and dead, auxiliary and failed rows, are left out. Pass `nfft` for another
2·3·5-smooth length, `demean = false` to keep the DC bin.
