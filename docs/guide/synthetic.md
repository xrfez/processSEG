# Synthetic records

A viewer that needs a small, correct sample set builds it here, in memory: a shot
gather with a direct wave, a refraction, a reflection and ground roll, or a vibrator
record whose ground force has a known phase error and known harmonic distortion.
Nothing in this module opens a file. The formulas and the noise contract are in
[`../api.md`](../api.md) (`synth`, `vibroseis/sweep`).

## A shot gather

Offsets are distances, signed the way a split spread is (`rcvX − srcX`). A linear
event is a direct wave, a refraction or ground roll, depending on the velocity and
the frequency; a reflection is a hyperbola. `addNoise` uses `std/random`: the same
seed gives the same samples on one machine and one Nim version, and is not a
promise across Nim versions or platforms. A left-out seed is `DefaultNoiseSeed`, so
it repeats too. Two traces with the same seed, or none, get the same noise: give
each trace its own seed. Seed 0 draws the same noise as 2147483647, so avoid 0.

```nim
import parseSEG, processSEG

let dtMs = 2.0
let offsets = [-100.0, -50.0, 0.0, 50.0, 100.0]
var shot = zeros[float64](offsets.len, 501)                 # 1 s at 2 ms
shot.addLinear(offsets, dtMs, t0Ms = 8, velocity = 1800, freqHz = 45)       # direct
shot.addLinear(offsets, dtMs, t0Ms = 40, velocity = 3200, freqHz = 25, amp = 0.6)
shot.addHyperbola(offsets, dtMs, t0Ms = 280, velocity = 2000, freqHz = 30)  # reflection
shot.addLinear(offsets, dtMs, t0Ms = 0, velocity = 400, freqHz = 12, amp = 2) # ground roll
# The zero-offset reflection is alone at 280 ms, and the wavelet peaks at 1.
let col = int(hyperbolicMs(280, 2000, 0) / dtMs)
doAssert col == 140 and abs(shot[2, col] - 1.0) < 1e-9
shot.addNoise(0.02, seed = 7)
doAssert abs(shot[2, col] - 1.0) < 0.15                     # the same seed, next run
```

## A distorted ground force

`sweepHarmonics` is the pilot's sweep plus harmonics. Harmonic k is
`amp·sin(k·φ + phase)`, with `φ` the fundamental's phase, so a phase error against
the pilot is the fundamental's own phase and does not multiply into the harmonics.
A term whose instantaneous frequency would pass Nyquist is cut at that sample, not
folded back. The vibrator QC reads the phase and the distortion back; the
tolerances are the ones in [vibrator QC](vibrator-qc.md#conventions-and-what-is-still-to-confirm).

```nim
import parseSEG, processSEG

let spec = linearSweep(10, 60, 2000, 1, initSweepTaper(200, 200))
let pilot = spec.sweepSamples(float64)
var force = spec
force.amplitude = 4e4                                       # newtons
let gf = force.sweepHarmonics(float64, [
  SweepHarmonic(phaseDeg: 4),                               # 4° ahead of the pilot
  SweepHarmonic(order: 2, amp: 0.08)])                      # 8% second harmonic
let n = spec.sweepLen
var keys = newSeq[TraceKeys](2)
var counts = newSeq[int](2)
var data = zeros[float64](2, n)
for r in 0 ..< 2:
  keys[r] = initTraceKeys()
  keys[r].t0Ms = 0
  counts[r] = n
for j in 0 ..< n:
  data[0, j] = pilot[j]
  data[1, j] = gf[j]
let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 1)
var unit = initVibUnit(0, groundForce = 1)
unit.units[vsGroundForce] = auForce
let tp = initSweepTrackPlan(float64, TrackParams(harmonics: 2), dtMs = 1, samples = n)
var ws = initWorkspace()
ws.reserve(vibratorWorkspace(tp))
var results: seq[UnitSignature[float64]] = @[]
tp.analyzeVibrators(data, meta, [unit], VibLimits(phaseDeg: 5), results, ws)
let track = results[0].tracks[vsGroundForce]
let m = 100                                                 # 1000 ms, past the taper
doAssert abs(track.phaseDeg[m] - 4) < 0.05
var hd2 = newSeq[float64](track.len)
track.distortionInto(2, hd2)
doAssert abs(hd2[m] - 8) < 0.02
doAssert abs(track.amplitude[0][m] - 4e4) < 1e-3 * 4e4
doAssert results[0].summaries[vsGroundForce].failed == {}  # inside the 5° limit
```
