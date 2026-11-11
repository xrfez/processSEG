## Self-tests of the synthetic generators in tests/utils/synth.nim.

import std/[assertions, math]
import utils/synth

func near(a, b: float64, tol = 1e-9): bool = abs(a - b) <= tol

proc argMaxAbs[T](g: SynthGather[T], row: int): int =
  result = 0
  for j in 1..<g.liveCols(row):
    if abs(g.data[row, j]) > abs(g.data[row, result]):
      result = j

block sinusoid_values:
  var x = zeroSignal[float64](100)
  x.addSinusoid(dtMs = 1, freqHz = 10, amp = 2)
  doAssert x[0] == 0
  doAssert near(x[25], 2), "a quarter period of 10 Hz at 1 ms is sample 25"
  doAssert near(x[50], 0)
  doAssert near(x[75], -2)

block sinusoid_float32:
  var x = zeroSignal[float32](40)
  x.addSinusoid(dtMs = 2, freqHz = 12.5, phaseRad = PI / 2)
  doAssert near(x[0].float64, 1, 1e-6), "a cosine starts at its peak"
  doAssert near(x[20].float64, -1, 1e-5), "half a period of 12.5 Hz is 40 ms = 20 samples"

block spike_and_dc:
  var x = zeroSignal[float64](10)
  x.addDc(3)
  x.addSpike(4, amp = -5)
  doAssert x[0] == 3 and x[9] == 3
  doAssert x[4] == -2

block drift_is_linear:
  var x = zeroSignal[float64](1001)
  x.addDrift(dtMs = 2, slopePerS = 30, start = 1)
  doAssert x[0] == 1
  doAssert near(x[500], 1 + 30 * 1.0), "500 samples of 2 ms is 1 s"
  doAssert near(x[1000] - x[999], 30 * 0.002)

block ricker_shape:
  let f = 25.0
  doAssert ricker(0, f) == 1
  doAssert near(ricker(10, f), ricker(-10, f)), "symmetric"
  let zeroMs = 1000 / (PI * f * sqrt(2.0))
  doAssert near(ricker(zeroMs, f), 0, 1e-12), "zero crossing at 1/(π f √2)"
  doAssert ricker(1.5 * zeroMs, f) < 0, "negative side lobes"
  var x = zeroSignal[float64](501)
  x.addRicker(dtMs = 2, centreMs = 400, freqHz = f, amp = 3)
  doAssert x[200] == 3
  doAssert x[0] == 0, "far from the centre stays zero"

block ricker_with_t0:
  var x = zeroSignal[float64](101)
  x.addRicker(dtMs = 4, centreMs = 300, freqHz = 20, t0Ms = 100)
  doAssert x[50] == 1, "sample 50 is at 100 + 50·4 = 300 ms"

block noise_seeded:
  var a = zeroSignal[float64](20000)
  var b = zeroSignal[float64](20000)
  var c = zeroSignal[float64](20000)
  a.addNoise(sigma = 2, seed = 7)
  b.addNoise(sigma = 2, seed = 7)
  c.addNoise(sigma = 2, seed = 8)
  doAssert a == b, "the same seed gives the same noise"
  doAssert a != c
  var s, s2 = 0.0
  for v in a:
    s += v
    s2 += v * v
  let mean = s / a.len.float64
  let sd = sqrt(s2 / a.len.float64 - mean * mean)
  doAssert abs(mean) < 0.05, $mean
  doAssert abs(sd - 2) < 0.05, $sd

block spreads:
  let s = splitSpread(perSide = 3, groupInterval = 25, nearOffset = 50)
  doAssert s == @[-100.0, -75, -50, 50, 75, 100]
  doAssert endOn(3, 10, 5) == @[5.0, 15, 25]

block shot_gather_geometry:
  var g = shotGather[float32](srcX = 1000, rcvRel = splitSpread(4, 25, 25),
      cols = 500,
    dtMs = 2, srcZ = 10, rcvZ = [1.0, 2, 3, 4, 5, 6, 7, 8], record = 12)
  doAssert g.len == 8 and g.cols == 500
  doAssert g.data.shape[0] == 8
  let k = g.keys[0]
  doAssert k.srcX == 1000 and k.rcvX == 900
  doAssert k.offset == -100, "offset is signed: rcvX − srcX"
  doAssert g.keys[7].offset == 100
  doAssert k.cmpX == 950
  doAssert k.srcZ == 10 and k.rcvZ == 1
  doAssert k.record == 12 and k.channel == 1 and g.keys[7].channel == 8
  doAssert k.intervalUs == 2000 and k.t0Ms == 0
  doAssert k.status == tsLive and k.isSeismic
  doAssert k.ensemble.isNaN

block cmp_gather_geometry:
  let g = cmpGather[float64](cmpX = 500, offsets = [100.0, 200, 300], cols = 10,
      dtMs = 4,
    cdp = 42)
  for i, off in [100.0, 200, 300]:
    doAssert g.keys[i].offset == off
    doAssert g.keys[i].cmpX == 500
    doAssert g.keys[i].srcX == 500 - off / 2 and g.keys[i].rcvX == 500 + off / 2
    doAssert g.keys[i].ensemble == 42

block rows_view:
  var g = cmpGather[float32](0, [0.0, 100, 200], cols = 50, dtMs = 2)
  g.setLive(1, 30)
  let r = g.rows
  doAssert r.len == 3 and r.hasKeys
  doAssert r.traceOf(2) == 2
  doAssert r.numSamples(0) == 50 and r.numSamples(1) == 30
  doAssert r.traceKeys(2).offset == 200

block hyperbola_peaks:
  let offsets = [0.0, 500, 1000, 1500, 2000]
  var g = cmpGather[float64](0, offsets, cols = 1001, dtMs = 2)
  g.addHyperbola(t0Ms = 800, velocity = 2000, freqHz = 30)
  for r, off in offsets:
    let want = sqrt(800.0 ^ 2 + (off / 2000 * 1000) ^ 2)
    doAssert near(hyperbolicMs(800, 2000, off), want)
    doAssert abs(g.argMaxAbs(r).float64 * 2 - want) <= 1, "peak within half a sample"
  doAssert g.argMaxAbs(0) == 400 and g.data[0, 400] == 1

block linear_event_split_spread:
  var g = shotGather[float64](0, splitSpread(5, 50, 50), cols = 1001, dtMs = 1)
  g.addLinear(t0Ms = 20, velocity = 1500, freqHz = 40)
  for r in 0..<g.len:
    let want = 20 + abs(g.keys[r].offset) / 1500 * 1000
    doAssert abs(g.argMaxAbs(r).float64 - want) <= 0.5
  doAssert g.argMaxAbs(0) == g.argMaxAbs(g.len - 1), "symmetric about the source"

block elevation_shifts:
  var g = shotGather[float64](0, endOn(3, 100, 100), cols = 1001, dtMs = 1,
      srcZ = 120,
    rcvZ = [100.0, 150, 200])
  let shifts = g.elevationShiftsMs(datumZ = 100, vReplacement = 2000)
  doAssert near(shifts[0], 20 / 2000 * 1000)
  doAssert near(shifts[2], 120 / 2000 * 1000)
  g.addHyperbola(t0Ms = 500, velocity = 2500, freqHz = 30, shiftsMs = shifts)
  for r in 0..<g.len:
    let want = hyperbolicMs(500, 2500, g.keys[r].offset) + shifts[r]
    doAssert abs(g.argMaxAbs(r).float64 - want) <= 0.5

block varying_live_length:
  var g = cmpGather[float32](0, [0.0, 0, 0], cols = 200, dtMs = 2)
  g.setLive(1, 120)
  g.setLive(2, 0)
  g.addHyperbola(t0Ms = 300, velocity = 2000, freqHz = 10)
  doAssert g.data[0, 150] == 1
  for j in 120..<200:
    doAssert g.data[1, j] == 0, "zero past the live length"
  doAssert g.data[1, 119] != 0, "the wavelet tail inside the live part stays"
  for j in 0..<200:
    doAssert g.data[2, j] == 0
  doAssert g.rows.numSamples(2) == 0

block t0_shifts_events:
  var g = cmpGather[float64](0, [0.0], cols = 300, dtMs = 2)
  g.keys[0].t0Ms = 100
  g.addHyperbola(t0Ms = 400, velocity = 2000, freqHz = 20)
  doAssert g.argMaxAbs(0) == 150, "400 ms is sample (400 − 100)/2"

block unsigned_offsets:
  var g = shotGather[float64](0, splitSpread(2, 50, 50), cols = 10, dtMs = 1)
  g.unsignOffsets()
  doAssert g.keys[0].offset == 100 and g.keys[3].offset == 100
  doAssert g.keys[0].rcvX - g.keys[0].srcX < 0, "the side stays in the coordinates"

block window_of_a_gather:
  var full = cmpGather[float64](0, [0.0, 0], cols = 500, dtMs = 2)
  full.setLive(1, 300)
  full.addHyperbola(t0Ms = 600, velocity = 2000, freqHz = 20)
  let w = full.window(firstSample = 200, cols = 150)
  doAssert w.firstSample == 200 and w.cols == 150
  doAssert w.numSamples == @[500, 300], "trace lengths are kept, longer than the window"
  doAssert w.liveCols(0) == 150 and w.liveCols(1) == 100
  doAssert w.data[0, 100] == 1, "600 ms is trace sample 300, column 100"
  for j in 100..<150:
    doAssert w.data[1, j] == 0, "zero past the trace's end"
  doAssert w.rows.numSamples(0) == 500

block window_events_use_first_sample:
  var w = cmpGather[float64](0, [0.0], cols = 1000, dtMs = 2).window(400, 100)
  w.addHyperbola(t0Ms = 900, velocity = 2000, freqHz = 20)
  doAssert w.argMaxAbs(0) == 50, "900 ms is trace sample 450, column 50"
  w.setLive(0, 420)
  doAssert w.liveCols(0) == 20 and w.data[0, 20] == 0
