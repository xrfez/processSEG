# Vibrator QC: the signatures of a wireline test

At start-up and during a survey, each vibrator's signals are recorded on a wireline record:
per unit a **reference** (the pilot it is driven with), its **ground force** (GF), and the
accelerations of its **reaction mass** (M) and **baseplate** (B). The plots of a vibrator
QC compare them with the reference: their spectra, their correlation with it, their phase
and force against sweep time, their harmonic distortion, one unit against another, and,
as an option, limits. processSEG computes what those plots show, values and their axes,
into the caller's buffers; drawing is the viewer's.

| Plot | Call | Axis |
|---|---|---|
| Signal | the rows; `UnitSignature.weightedSum` (GF from M and B) | `sampleTimeMs` |
| Spectrum: amplitude, phase | `SpectrumPlan.amplitudeInto`; against the reference: the correlation's `lagSpectrumInto` | `binHz` |
| Correlation: raw, envelope | `cp.crossCorrelateInto(x, reference, lags, envelope, ws)` | `cp.lagMs(i)` |
| Correlation: spectrum, phase | `cp.lagSpectrumInto(lags, amp, phaseDeg, ws)` | `cp.binHz(k)` |
| QC: phase, fundamental force, peak force, frequency | `track.phaseDeg`, `track.amplitude[0]`, `track.peak`, `track.frequencyHz` and `track.referenceHz` | `track.timeMs` |
| Distortion: each harmonic, THD | `track.distortionInto(k, dst)`, `track.thdPct` | `track.timeMs` |
| Distortion: time-variant spectrum | `sg.spectrogramInto(x, amp, ws)` (then `toDbInto`) | `sg.frameCol(i)`, `sg.binHz(k)` |
| Comparison against time | `compareTracksInto(a, b, phaseDeg, ratioDb)` | `track.timeMs` |
| Comparison against frequency | `cp.compareInto(a, b, ratioDb, phaseDeg, ws)` | `cp.binHz(k)` |
| Limits (optional) | `limitFlagsInto`, `summarize`; `UnitSignature.flags`, `.summaries` | `track.timeMs` |
| Everything of a record | `tp.analyzeVibrators(samples, meta, units, limits, results, ws)` | |

## Roles and units

The caller says which row is which: a `VibUnit` holds each role's row (−1: none), and the
library never takes a role from a row's position. `vibUnitsInOrder(count)` builds the usual
layout, each unit's channels in the order Ref, GF, M, B (another `order` can be given), for
the caller to pass or edit.

Rows are taken in the unit the caller says they are in (`VibUnit.units`, `auRecorded` by
default; a calibrated row's from `resolveInto`, [calibration](calibration.md)). Wireline
channels are often auxiliary traces, which a calibration takes to volts only; a role's
`Sensitivity` takes such a row on to its quantity: M and B in V/g or V/(m/s²) to m/s², the
ground-force channel in V/N, V/kN or V/lbf to N. The ground force generated from M and B
is the weighted sum `Mr·a_M + Mb·a_B`, with the reaction mass and the baseplate mass entered
with their unit (`mass(2450, muKilogram)`, `mass(5400, muPound)`): in newtons when both
accelerations are in m/s². Its sign is the accelerometers': if it comes out at 180° to a
recorded ground force, they are wired the other way. Reverse both rows (`reversePolarity` on
the M and B rows) before `analyzeVibrators`: reversing `weightedSum` after it leaves the
generated force's track as it was.

## Everything at once

`analyzeVibrators` takes a record's rows and the units, prepares each unit's reference,
tracks each of its signals and the generated force against it, and summarizes them, into
one `UnitSignature` per unit. The results are sized on the first call and refilled after,
without allocating.

```nim
import std/[complex, math]
import parseSEG, processSEG

# One unit, its channels in the usual order: Ref, GF, M, B. A 2 s sweep at 1 ms.
let spec = linearSweep(10, 60, 2000, 1, initSweepTaper(200, 200))
let n = spec.sweepLen + 500
proc sweep(amp, phaseDeg: float64): seq[float64] =
  result = linearSweep(10, 60, 2000, 1, initSweepTaper(200, 200), startPhaseDeg = phaseDeg,
    amplitude = amp).sweepSamples(float64)
  result.setLen(n)
let mr = mass(2450, muKilogram)
let mb = mass(1180, muKilogram)
let aM = sweep(20, 5)       # m/s²: the mass leads the pilot by 5°
let aB = sweep(15, -175)    # the baseplate moves against it
var force = newSeq[float64](n)
groundForceInto(aM, aB, mr, mb, force)

# The channels as recorded: volts, with the sensors' and the controller's scales.
let (sGf, sM, sB) = (sensitivity(0.25, suVPerKN), sensitivity(0.1, suVPerG),
  sensitivity(0.1, suVPerG))
var keys = newSeq[TraceKeys](4)
var counts = newSeq[int](4)
var data = zeros[float64](4, n)
for r in 0 ..< 4:
  keys[r] = initTraceKeys()
  keys[r].status = tsAuxiliary
  keys[r].t0Ms = 0
  counts[r] = n
let pilot = sweep(1, 0)
for j in 0 ..< n:
  data[0, j] = pilot[j]
  data[1, j] = force[j] * sGf.voltsPerUnit
  data[2, j] = aM[j] * sM.voltsPerUnit
  data[3, j] = aB[j] * sB.voltsPerUnit
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 1)

var units = vibUnitsInOrder(1, reactionMass = mr, baseplateMass = mb)
for (role, s) in [(vsGroundForce, sGf), (vsMass, sM), (vsBaseplate, sB)]:
  units[0].units[role] = auVolt
  units[0].sensitivity[role] = s

let tp = initSweepTrackPlan(float64, TrackParams(), dtMs = 1, samples = n)
var ws = initWorkspace()
ws.reserve(vibratorWorkspace(tp))
var results: seq[UnitSignature[float64]]
let limits = VibLimits(phaseDeg: 3, fromMs: 300, toMs: 1700) # leave out the tapers
tp.analyzeVibrators(data, meta, units, limits, results, ws)

# The ground force's fundamental: 2450 kg × 20 m/s² ∠5° + 1180 kg × 15 m/s² ∠−175°.
let want = 2450.0 * 20 * exp(complex(0.0, degToRad(5.0))) +
  1180.0 * 15 * exp(complex(0.0, degToRad(-175.0)))
let u = results[0]
let gf = u.tracks[vsGroundForce]
let m = 100                                   # the point at 1000 ms
doAssert gf.timeMs[m] == 1000 and u.units[vsGroundForce] == auForce
doAssert abs(gf.amplitude[0][m] - abs(want)) < 1e-3 * abs(want)   # N
doAssert abs(gf.phaseDeg[m] - radToDeg(phase(want))) < 0.05      # 5.0°
doAssert abs(gf.referenceHz[m] - spec.frequencyAt(1000)) < 0.05
# The force from M and B is the recorded one, and over the 3° limit.
doAssert abs(u.tracks[vsWeightedSum].amplitude[0][m] - gf.amplitude[0][m]) < 1e-6 * abs(want)
doAssert lkPhase in u.summaries[vsGroundForce].failed and lkPhase in u.flags[vsGroundForce][m]
doAssert u.summaries[vsMass].failed == {}     # M and B have no limits
```

## Against sweep time

A `SweepTrackPlan` measures one signal against one reference at points every `stepMs`
(10 ms) along the sweep. The reference's phase ψ comes from its own analytic signal, so any
sweep law (linear, dB/octave, exponential, a table, a recorded pilot) is handled the same
way. At each point, harmonic `k`'s amplitude and phase are a Fourier coefficient over
`cycles` (4) periods of the reference with a cos⁴ weight in phase: over whole periods it is
orthogonal to every other harmonic and to DC, whatever the law, which is what a vibrator
controller's tracking filter does. The measures, one value per point, NaN where not
measured (outside the sweep, within half a window of its ends, a harmonic above
`maxHarmonicFraction` of Nyquist):

- `phaseDeg`: the fundamental's phase minus the reference's, degrees in (−180, 180],
  positive when the signal leads (the reference delayed by τ reads −360·f·τ);
- `amplitude[k − 1]`: harmonic k's peak amplitude; `amplitude[0]` of a ground force is the
  **fundamental force**;
- `thdPct`: total harmonic distortion, `100·√(Σ_{k≥2} A_k²)/A_1` (`thdFundamental`) or over
  `√(Σ_{k≥1} A_k²)` (`thdTotal`); `distortionInto(k, dst)`: one harmonic's `100·A_k/A_1`;
- `peak`: the largest |s| within one period, harmonics and noise included, of the
  band-limited signal (the **peak force**, against the hold-down weight);
- `referenceHz` and `frequencyHz`: the reference's instantaneous frequency (the sweep law
  as it was driven) and the signal fundamental's.

```nim
import std/math
import processSEG

# A ground force 3° ahead of its pilot, with a 2nd harmonic of 8 % and a 3rd of 3 %;
# a second unit 4 ms late. An exponential sweep at 1 ms.
let spec = exponentialSweep(8, 64, 3000, 1, initSweepTaper(250, 250))
let n = spec.sweepLen
let pilot = spec.sweepSamples(float64)
proc signal(delayMs: float64): seq[float64] =
  result = newSeq[float64](n)
  for j in 0 ..< n:
    let t = float64(j) - delayMs
    if t < 0: continue
    let w = (if t < 250: taperWeight(tpCos2, t / 250) elif t > 2750:
      taperWeight(tpCos2, (3000 - t) / 250) else: 1.0)
    let p = spec.phaseAt(t)
    result[j] = w * (sin(p + degToRad(3.0)) + 0.08 * sin(2 * p) + 0.03 * sin(3 * p))

let tp = initSweepTrackPlan(float64, TrackParams(harmonics: 3), dtMs = 1, samples = n)
var ws = initWorkspace()
ws.reserve(tp.workspaceSize)
var reference = initSweepReference(tp)
var a = initSweepTrack(tp)
var b = initSweepTrack(tp)
tp.referenceInto(pilot, reference, ws)    # once per reference
tp.trackInto(signal(0), reference, a, ws)
tp.trackInto(signal(4), reference, b, ws)

let m = 150                               # 1500 ms
let f = spec.frequencyAt(1500)
doAssert abs(a.phaseDeg[m] - 3) < 0.05 and abs(a.amplitude[0][m] - 1) < 1e-4
var hd2 = newSeq[float64](a.len)
a.distortionInto(2, hd2)
doAssert abs(hd2[m] - 8) < 0.02 and abs(a.thdPct[m] - 100 * sqrt(0.08^2 + 0.03^2)) < 0.02
doAssert abs(a.peak[m] - 0.99695) < 1e-3   # the waveform's largest |s| (the harmonics' phases)
# The second unit against the first: 360·f·4 ms behind.
var dPhase, dDb = newSeq[float64](a.len)
compareTracksInto(b, a, dPhase, dDb)
doAssert abs(wrapDeg(dPhase[m] + 360 * f * 0.004)) < 0.2 and abs(dDb[m]) < 1e-3
```

## Correlation with the reference

A `CrossCorrelationPlan` correlates a signal with its reference over the lags within
`maxLagMs` of zero (`c(m) = Σ s[j + m]·r[j]`: a delay τ peaks at +τ), as a correlation
coefficient by default (`cnCoefficient`: 1 for the reference's shape, whatever the units),
by the reference's energy (`cnReference`) or raw (`cnNone`). Its envelope peaks at the delay
whatever the phase. `lagSpectrumInto` gives the spectrum of the tapered lag window with the
phase measured from lag zero: the signal's phase relative to the reference against
frequency (a delay τ: −360·f·τ). `compareInto` compares two signals against frequency: the
ratio of their amplitude spectra in dB and the phase of one minus the other.

```nim
import std/math
import processSEG

let spec = linearSweep(10, 80, 3000, 1, initSweepTaper(200, 200))
let n = spec.sweepLen + 500
var pilot = spec.sweepSamples(float64)
pilot.setLen(n)
var late = newSeq[float64](n)    # the pilot 6 ms late, at 0.5 of its level
for j in 6 ..< n: late[j] = 0.5 * pilot[j - 6]

let cp = initCrossCorrelationPlan(float64, maxLagMs = 250, dtMs = 1, samples = n)
var ws = initWorkspace()
ws.reserve(cp.workspaceSize)
var lags, envelope = newSeq[float64](cp.lags)
cp.crossCorrelateInto(late, pilot, lags, envelope, ws)
var best = 0
for i in 0 ..< cp.lags:
  if lags[i] > lags[best]: best = i
doAssert cp.lagMs(best) == 6 and abs(lags[best] - 1) < 1e-6   # the shape is the pilot's

var amp, phaseDeg = newSeq[float64](cp.bins)
cp.lagSpectrumInto(lags, amp, phaseDeg, ws)
var ratioDb, dPhase = newSeq[float64](cp.bins)
cp.compareInto(late, pilot, ratioDb, dPhase, ws)
for k in 0 ..< cp.bins:
  let f = cp.binHz(k)
  if f > 20 and f < 70:
    doAssert abs(wrapDeg(phaseDeg[k] + 360 * f * 0.006)) < 0.5
    doAssert abs(ratioDb[k] - 20 * log10(0.5)) < 0.1
```

## Time-variant spectrum

`SpectrogramPlan` is a short-time Fourier transform: Hann frames of `windowLen` samples
centred every `hop` samples, scaled so that a sinusoid of amplitude A on a bin reads A (also
in the frames the trace's ends cut). A frame of W ms resolves about 2/W kHz: 256 ms at 1 ms
separates lines 7.8 Hz apart. A sweep shows as a line along its law, its harmonics as the
lines at twice, three times that frequency: overlay `k·track.referenceHz`.

```nim
import std/math
import processSEG

let spec = linearSweep(10, 90, 4000, 1, initSweepTaper(200, 200))
let x = spec.sweepSamples(float64)
let sg = initSpectrogramPlan(float64, windowLen = 255, hop = 20, dtMs = 1, samples = x.len)
var ws = initWorkspace()
ws.reserve(sg.workspaceSize)
var amp = newSeq[float64](sg.frames * sg.bins)
sg.spectrogramInto(x, amp, ws)              # frame i, bin k: amp[i·bins + k]
let i = 100                                 # the frame at column 2000
var best = 0
for k in 0 ..< sg.bins:
  if amp[i * sg.bins + k] > amp[i * sg.bins + best]: best = k
doAssert sg.frameCol(i) == 2000
doAssert abs(sg.binHz(best) - spec.frequencyAt(2000)) <= sg.binHz(1)
var db = newSeq[float64](sg.bins)
toDbInto(amp.toOpenArray(i * sg.bins, (i + 1) * sg.bins - 1), db)   # 0 dB at the peak
doAssert db[best] == 0
```

## Limits

Limits are optional, all off (NaN) by default: per point, `phaseDeg` (|phase|), `thdPct`,
`harmonicPct` (any one harmonic), `forceLow` and `forceHigh` (the fundamental), `peakForce`;
over the sweep, `meanPhaseDeg`, `meanThdPct` and `meanForceLow`; all checked between
`fromMs` and `toMs` (set them to leave out the tapers, where the force ramps). For any
track, `limitFlagsInto(track, limits, flags)` gives each point the set of limits it exceeds
(`set[LimitKind]`), for the viewer to mark, and `summarize(track, limits)` the statistics
(points measured; mean and maximum |phase| and THD; the fundamental's mean, minimum and
maximum; the largest peak), the points beyond each limit and the limits failed, for a pass
or fail table per unit. `analyzeVibrators` applies them to the two ground forces, recorded
and generated.

## Conventions, and what is still to confirm

Vibrator QC systems differ in some definitions; processSEG takes these, and states them:

- **Phase:** signal minus reference, positive when the signal leads.
- **Distortion:** THD over harmonics 2 to 5 (`harmonics`) relative to the fundamental;
  `thdTotal` relative to the fundamental and harmonics together. Noise and subharmonics are
  not counted as such, but the window lets through what lies within its main lobe of a
  harmonic: noise in the harmonics' bands, and a subharmonic at f/2 or 3f/2 at 1/6 of its
  level into the fundamental and the 2nd harmonic over 4 periods (0.04 over 5, nothing over
  6).
- **Reference:** a clean sweep (the pilot, as recorded or generated). Harmonics of its own
  bend its phase: a 5 % 2nd harmonic in the reference reads as a false 2nd harmonic of about
  3 % in every signal tracked against it. Compare two signals with `compareTracksInto`, not
  by taking one as the other's reference.
- **Peak force:** the largest |GF| per period of the fundamental, harmonics included; the
  summary also has the largest fundamental force.
- **Window:** 4 periods of the fundamental (`cycles`), points every 10 ms (`stepMs`).
- **Ground force from M and B:** `Mr·a_M + Mb·a_B`, the accelerometers' polarity.
- **Accuracy** (the test suite, on synthetic sweeps): phase within 0.05°, harmonic levels
  within 0.02 % of the fundamental (0.04 % at the slow start of a strongly nonlinear law),
  the fundamental within 1e-4, the peak within 2e-4, the
  frequency within 0.05 Hz, away from the sweep's ends; in a taper's first periods, where the
  amplitude changes within the window, the phase is off by up to 0.1°. Near the harmonic
  limit (2.5 samples a period) the fundamental's phase is within 0.4° and its amplitude
  0.5 %. With noise the ends of a sweep are the least steady part, the end most (its windows
  are short and its amplitude small): leave the tapers out of limits with `fromMs`, `toMs`.
- **NaN and infinite samples** blank the points whose window holds them, and only those.
