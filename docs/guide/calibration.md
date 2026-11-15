# Recorded values to physical units

Many recorders, nodes above all, write samples as quantization values: counts of the
analogue-to-digital converter. Two factors turn a count into ground motion or pressure:

- the **descale**, volts at the recorder's input per count (a SEG-D channel set's MP factor
  or descale multiplier, a SEG-2 DESCALING_FACTOR, or the recorder's documentation);
- the **sensitivity** of the sensor, volts out per unit of what it measures: V/(m/s) for a
  geophone, V/(m/s²) for an accelerometer, V/Pa for a hydrophone.

A row's factor is `descale / (10^(gain/20) · sensitivity)`, where `gain` is a recorder gain
in dB that the descale does not already hold (usually none). The result is in SI units: V,
then m/s, m/s² or Pa (`AmplitudeUnit`). Different sensors on one unit and different units
get their own factors, so the scale is per row. Amplitudes in physical units compare across
equipment; amplitudes in counts do not.

The case this was built for is a **marine node** whose recorder writes counts: each node a
hydrophone and three geophones, so the factor differs by component within a unit, and one of
them is a pressure sensor ([the worked example](#rules-which-rows-get-which-factor)). Most
files need none of this.

**Nothing changes for code that does not calibrate.** Every new argument is optional
(`scales`), and a pipeline without `calibrateOp` compiles to the same steps and gives the
same samples, bit for bit, whether or not its metas carry a file-scale column.

Calibrate before anything that changes amplitudes: gain, AGC, normalization, statistics and
QC then see physical values ([pipelines](#in-a-pipeline) below).

## Entering a sensitivity

A sensitivity is never a bare number: it is entered with its unit, and held in SI.

- **As wired** (the usual case): `sensitivity(value, unit)`, the figure that holds with the
  sensor connected to the recorder. For a geophone this is the **damped** sensitivity of its
  spec sheet: the coil works into a shunt (damping) resistor and the recorder's input, which
  divide its output. For a string of sensors, the string's figure as wired.
- **Open circuit**: `openCircuitSensitivity(value, unit, coilOhms, shuntOhms, inputOhms =
  Inf, inSeries = 1, inParallel = 1)` computes the as-wired figure from the open-circuit
  (intrinsic) one: `S = n·S₀ · R_L / (n·R_c/m + R_L)`, with `S₀` one sensor's open-circuit
  sensitivity, `R_c` its coil resistance, `n` sensors in series per string, `m` strings in
  parallel, and `R_L` the shunt in parallel with the recorder's input impedance (leave
  `inputOhms` out when it is large against the shunt).

Units: `suVPerMPerS` V/(m/s), `suVPerInPerS` V/(in/s), `suVPerMPerS2` V/(m/s²), `suVPerG` V/g,
`suVPerBar` V/bar, `suVPerPa` V/Pa, `suMicroVPerPa` µV/Pa, `suDbReVPerMicroPa` dB re 1 V/µPa.

```nim
import std/math
import parseSEG, processSEG

let damped = sensitivity(22.5, suVPerMPerS)          # a spec sheet's damped figure
doAssert damped.unit == auVelocity and damped.voltsPerUnit == 22.5

let inches = sensitivity(0.7, suVPerInPerS)          # the same kind of figure in V/(in/s)
doAssert abs(inches.voltsPerUnit - 0.7 / 0.0254) < 1e-12

# Open circuit 28.8 V/(m/s), a 375 Ω coil, a 1339 Ω shunt: 28.8 · 1339 / 1714.
let fromSheet = openCircuitSensitivity(28.8, suVPerMPerS, coilOhms = 375, shuntOhms = 1339)
doAssert abs(fromSheet.voltsPerUnit - 28.8 * 1339 / 1714) < 1e-12
echo fromSheet   # 22.4989 V/(m/s) (open circuit 28.8 V/(m/s), coil 375 Ω, shunt 1339 Ω)

let hydrophone = sensitivity(20, suVPerBar)          # 2e-4 V/Pa
doAssert abs(hydrophone.valueIn(suMicroVPerPa) - 200) < 1e-9
doAssert abs(hydrophone.valueIn(suDbReVPerMicroPa) - 20 * log10(2e-10)) < 1e-9 # −194 dB
```

**The flat band only.** One sensitivity holds above the sensor's natural frequency. Near and
below it a geophone's response falls off (12 dB per octave below it), and how it falls
depends on the damping: amplitudes there come out low after calibration. processSEG does
not correct the response.

**Gain.** SEG-D defines its MP factor and descale multiplier to include the recorder's gain,
so normally `gainDb` stays unset. If a file's descale stays the same between records taken
at two gain settings, it does not include the gain: put the gain in the rule's `gainDb`.

## Rules: which rows get which factor

A `Calibration` is plain data: a list of `CalibrationRule`s. A rule matches rows by any of
SEG-D sensor type, component, channel set, channel and receiver station (one recording unit),
and gives any of a descale, a sensitivity and a gain. The three factors resolve separately,
each from the **most specific** rule that gives it:

station > channel > channel set > component > sensor type > a rule without criteria (the
catch-all default).

A rule with more criteria of the same rank wins (a channel set and a component beat a
channel set alone); two rules with the same criteria and values are refused. `resolveInto`
says, per row, the factor, the unit, where each factor came from and whether the row is
converted, unresolved or kept, without touching a sample; `calibrate` applies it.

```nim
import std/[math, options]
import parseSEG, processSEG

# A record of three marine nodes whose recorder writes counts: each node a
# hydrophone (component 1) and three geophones (2 vertical, 3 inline, 4 crossline).
var keys = newSeq[TraceKeys](12)
for i in 0 ..< keys.len:
  keys[i] = initTraceKeys()
  keys[i].status = tsLive
  keys[i].component = uint8(1 + i mod 4)
  keys[i].sensorType = uint8(1 + i mod 4)  # SEG-D: 1 hydrophone, 2-4 geophones
  (keys[i].rcvLine, keys[i].rcvPoint) = (10.0, float64(101 + i div 4))
var counts = newSeq[int](keys.len)
for c in counts.mitems: c = 50
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)

# The table, by component, from the sensors' spec sheets (the figures as wired).
let cal = initCalibration([
  CalibrationRule(match: RuleMatch(component: 1),
                  sensitivity: sensitivity(-192, suDbReVPerMicroPa)),
  CalibrationRule(match: RuleMatch(component: 2), sensitivity: sensitivity(28.8, suVPerMPerS)),
  CalibrationRule(match: RuleMatch(component: 3), sensitivity: sensitivity(28.5, suVPerMPerS)),
  CalibrationRule(match: RuleMatch(component: 4), sensitivity: sensitivity(28.6, suVPerMPerS)),
  # Node 10/102's vertical geophone was calibrated on its own:
  CalibrationRule(match: RuleMatch(station: some stationKey(10, 102, 0), component: 2),
                  sensitivity: sensitivity(27.9, suVPerMPerS)),
  # Every row (a file's own descale, when it has one, comes through `scales`):
  CalibrationRule(descale: descale(0.298, duMicroVPerCount))])
echo cal
let plan = initCalibrationPlan(cal)

var rows = newSeq[RowCalibration](meta.len)
let summary = plan.resolveInto(meta, rows)
echo summary                     # 12 converted (9 m/s, 3 Pa), 0 unresolved, 0 kept
doAssert rows[0].unit == auPressure and rows[1].unit == auVelocity
doAssert rows[5].sensitivityRule == 4 and rows[1].sensitivityRule == 1

var samples = newTensor[float32](meta.len, 50)
for r in 0 ..< meta.len:
  for j in 0 ..< 50: samples[r, j] = 1000          # counts
plan.calibrate(samples, meta)
let lsb = 0.298e-6                                 # V per count
doAssert abs(samples[1, 0] / (1000 * lsb / 28.8) - 1) < 1e-6           # m/s
let hydrophone = pow(10.0, -192 / 20) * 1e6                           # V/Pa
doAssert abs(samples[0, 0] / (1000 * lsb / hydrophone) - 1) < 1e-6     # Pa
```

Rows without the factors their target needs are never given a wrong scale:

- a **live** row is filled with NaN and counted as `unresolved` (`noDescale`,
  `noSensitivity`). A row of counts beside rows in m/s would be millions of times too large
  for every step after it; NaN is what panel statistics, spectra and stacks already leave
  out, and it stays in its row. The raw data is kept by the `src → dst` forms. A factor
  that, or whose inverse, does not fit a float32 (about 3e-39 to 3e38; real factors lie
  near 1e-12 to 1e3) comes from a broken header or extreme entries and is not applied
  either;
- a **dead** row is converted when it can be, else kept;
- an **auxiliary** row (pilot, time break) is converted to volts when its descale is known
  (`scaleAuxiliary`, on by default), and never given a sensitivity;
- a **failed** row is never touched.

`target = ctVolts` stops at volts and needs no sensitivity.

## The file's own factors

A file may say something about its scale itself. `scaleReport` shows what its headers hold,
grouped, so you can check them against the recorder's documentation and the sensor's spec
sheet before applying anything; `fileScalesInto` turns them into a per-row `FileScale`
column that a meta points at.

```nim
import std/math
import parseSEG, processSEG

# A small SEG-Y whose trace headers give a descale (N = 10: 2⁻¹⁰ V per count)
# and a transduction constant (50 m/s per V: 0.02 V/(m/s)).
var w = openSeismicWriter("calib_demo.sgy", SegyOutputOptions())
var k = initTraceKeys()
k.intervalUs = 2000
k.t0Ms = 0
k.status = tsLive
for i in 0 ..< 3:
  w.writeTrace(k, newSeq[float32](100), [headerOverride("trace.traceWeightingFactor", 10),
    headerOverride("trace.traceValueUnit", 2), headerOverride("trace.transductionMantissa", 5),
    headerOverride("trace.transductionExponent", 1), headerOverride("trace.transductionUnits", 6)])
doAssert w.close().len == 0

var f = openSeismic("calib_demo.sgy", keys = true)
for g in f.scaleReport(): echo g   # 3 traces from 0: trace.traceWeightingFactor = 10, ...

var scales = newSeq[FileScale](f.len)              # once per chunk, reused
f.fileScalesInto(0 ..< f.len, scales)
doAssert abs(scales[0].descale - pow(2.0, -10)) < 1e-15 and scales[0].unit == auVelocity
let meta = chunkMeta(f, 0 ..< f.len, scales = scales)  # the meta points at the column

let plan = initCalibrationPlan(initCalibration())  # no rules: the file's factors alone
var rows = newSeq[RowCalibration](meta.len)
discard plan.resolveInto(meta, rows)
doAssert rows[0].descaleFrom == fsFile and abs(rows[0].factor - pow(2.0, -10) / 0.02) < 1e-12
f.close()
```

`fileFactors` decides between the file and the rules:

- `ffFallback` (default): the specific rules, then the file, then the catch-all rule. What you
  entered for a row wins over the file, and the file over a blanket default;
- `ffFirst`: the file, then the rules;
- `ffIgnore`: the rules only.

Neither overrides the other silently: a row where the file and a rule both give a factor
more than 0.1 % apart is flagged (`RowCalibration.conflict`, `CalibrationSummary.conflicts`),
whichever won. A sensitivity whose quantity does not fit the SEG-D sensor type (a V/(m/s)
figure on a hydrophone) is flagged as a `mismatch`.

## Comparing two candidate tables

When it is not clear which figure of a spec sheet applies (the damped one or the open-circuit
one), build both and compare: `ratioDbInto(a, b, meta, dst)` gives, per row, how much larger
data comes out under `a` than under `b`, in dB, and the range over the rows.

```nim
import std/math
import parseSEG, processSEG

var keys = @[initTraceKeys()]
keys[0].status = tsLive
keys[0].sensorType = 2
let counts = @[50]
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
let lsb = descale(0.5, duMicroVPerCount)
let asWired = initCalibrationPlan(initCalibration([
  CalibrationRule(descale: lsb, sensitivity: sensitivity(28.8, suVPerMPerS))]))
let shunted = initCalibrationPlan(initCalibration([
  CalibrationRule(descale: lsb, sensitivity: openCircuitSensitivity(28.8, suVPerMPerS,
    coilOhms = 375, shuntOhms = 1339))]))
var db = newSeq[float64](meta.len)
let spread = ratioDbInto(shunted, asWired, meta, db)
doAssert abs(db[0] - 20 * log10(1714 / 1339)) < 1e-9   # 2.1 dB louder if 28.8 was open circuit
doAssert spread.rows == 1
```

## In a pipeline

`calibrateOp(cal)` converts in a flow; `compile` refuses it after any op that changes
amplitudes (scale, gain, clip, AGC, normalization, gain curve), so everything after it sees
physical values. `plan.amplitudeUnits` tells a viewer how to label the output: recorded,
calibrated (the per-row units of `resolveInto`) or relative (a gain op follows).

```nim
import parseSEG, processSEG

var keys = newSeq[TraceKeys](3)
for i in 0 ..< 3:
  keys[i] = initTraceKeys()
  keys[i].status = tsLive
  keys[i].sensorType = 2
let counts = @[200, 200, 200]
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2)
let cal = initCalibration([CalibrationRule(descale: descale(0.5, duMicroVPerCount),
  sensitivity: sensitivity(22.5, suVPerMPerS))])

var flow = initPipeline([calibrateOp(cal), demeanOp()])
let plan = flow.compile(float32, dtMs = 2, samples = 200)
doAssert plan.amplitudeUnits == paCalibrated
var raw = newTensor[float32](3, 200)
for r in 0 ..< 3:
  for j in 0 ..< 200: raw[r, j] = float32(1000 * (j mod 7))
var processed = newTensor[float32](3, 200)
var ws = initWorkspace()
ws.reserve(plan)
plan.apply(raw, processed, meta, ws)        # m/s, the raw counts kept

var stats = newSeq[TraceStats](3)
traceStatsInto(processed, meta, stats)      # RMS in m/s
doAssert stats[0].rms > 0 and stats[0].rms < 1e-3

flow.add agcOp(windowMs = 100)              # after the op: allowed, but relative
doAssert flow.compile(float32, 2, 200).amplitudeUnits == paRelative
flow.insert(gainDbOp(6), 0)                 # before it: refused
doAssertRaises(ValueError): discard flow.compile(float32, 2, 200)
```

## Writing calibrated data

`uncalibrate` (and `uncalibrateOp`) is the inverse: rows back to counts with the same
factors, to write an integer format with the input's descale. In float32 the round trip
holds to a few parts in 10⁷, so 24-bit counts come back within one count; process in
float64 when the counts must come back exactly. Physical samples written to
SEG-Y carry their unit with `segyUnitOverrides(unit)` (trace bytes 203-204 set, N and the
transduction constant zero), and `fileScalesInto` reads them back as physical, so a
calibration does not scale them twice.

```nim
import parseSEG, processSEG

var w = openSeismicWriter("calib_out.sgy", SegyOutputOptions())
var k = initTraceKeys()
k.intervalUs = 2000
k.t0Ms = 0
k.status = tsLive
w.writeTrace(k, @[1e-6'f32, -2e-6, 3e-6], segyUnitOverrides(auVelocity))   # m/s
doAssert w.close().len == 0

var f = openSeismic("calib_out.sgy", keys = true)
var scales = newSeq[FileScale](1)
f.fileScalesInto(0 ..< 1, scales)
doAssert scales[0].physical and scales[0].unit == auVelocity and scales[0].descale == 1
f.close()
```
