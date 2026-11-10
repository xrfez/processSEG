## Time-variant gain (gain/curves): exact values at the knots, interpolation
## in dB, flat extrapolation, the power and exponential curves, each row's
## t0 (on and off the grid, before and beyond the table), NaN t0, liveness,
## and no allocation.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

func near(a, b, rel: float64): bool =
  abs(a - b) <= rel * max(abs(a), abs(b))

let knots = [GainKnot(tMs: 100, dB: 0), GainKnot(tMs: 1100, dB: 20),
  GainKnot(tMs: 2100, dB: 14)]

block curve_values:
  let c = knotGain(knots)
  doAssert c.gainDbAt(100) == 0 and c.gainDbAt(1100) == 20 and c.gainDbAt(2100) == 14
  doAssert c.gainDbAt(600) == 10, "linear in dB"
  doAssert c.gainDbAt(1600) == 17
  doAssert c.gainDbAt(-50) == 0 and c.gainDbAt(5000) == 14, "flat outside"
  doAssert near(c.factorAt(1100), 10, 1e-15)
  doAssert near(c.factorAt(600), sqrt(10.0), 1e-15)
  doAssert c.gainDbAt(NaN).isNaN and c.factorAt(NaN).isNaN
  let one = knotGain([GainKnot(tMs: 0, dB: 6)])
  doAssert one.gainDbAt(-1) == 6 and one.gainDbAt(1e6) == 6
  let p = powerGain(2)
  doAssert p.factorAt(1000) == 1 and near(p.factorAt(2000), 4, 1e-15)
  doAssert near(p.factorAt(500), 0.25, 1e-15)
  doAssert p.factorAt(0) == p.factorAt(1), "t below minMs is taken as minMs"
  doAssert near(p.gainDbAt(2000), 20 * log10(4.0), 1e-12)
  let e = exponentialGain(6)
  doAssert e.gainDbAt(0) == 0 and e.gainDbAt(2000) == 12 and e.gainDbAt(-1000) == -6
  doAssert near(e.factorAt(2000), dbToFactor(12), 1e-15)
  doAssertRaises ValueError: discard knotGain([])
  doAssertRaises ValueError: discard knotGain([GainKnot(tMs: 1, dB: 0), GainKnot(tMs: 1, dB: 3)])
  doAssertRaises ValueError: discard knotGain([GainKnot(tMs: 2, dB: 0), GainKnot(tMs: 1, dB: 3)])
  doAssertRaises ValueError: discard knotGain([GainKnot(tMs: NaN, dB: 0)])
  doAssertRaises ValueError: discard knotGain([GainKnot(tMs: 0, dB: Inf)])
  doAssertRaises ValueError: discard powerGain(NaN)
  doAssertRaises ValueError: discard powerGain(2, refMs = 0)
  doAssertRaises ValueError: discard powerGain(2, minMs = -1)
  doAssertRaises ValueError: discard exponentialGain(Inf)

block plan_table:
  let plan = initGainCurvePlan(float64, knotGain(knots), dtMs = 2, samples = 1001)
  doAssert plan.tableLen == 1001 and plan.tableMs == 0.0 .. 2000.0
  let wide = initGainCurvePlan(float32, powerGain(2), 2, 1001, tMinMs = -100, tMaxMs = 3000)
  doAssert wide.tableLen == 1551 and wide.tableMs == -100.0 .. 3000.0
  doAssertRaises ValueError: discard initGainCurvePlan(float32, powerGain(2), 0, 10)
  doAssertRaises ValueError: discard initGainCurvePlan(float32, powerGain(2), 2, 0)
  doAssertRaises ValueError: discard initGainCurvePlan(float32, powerGain(2), 2, 10,
    tMinMs = 10, tMaxMs = 5)
  doAssertRaises ValueError: discard initGainCurvePlan(float32, powerGain(2), 2, 10,
    tMinMs = Inf)

proc ones(n: int): seq[float64] =
  result = newSeq[float64](n)
  for v in result.mitems: v = 1

block knots_on_the_grid:
  # t0 = 0: the samples at the knot times carry exactly the knot gains.
  let plan = initGainCurvePlan(float64, knotGain(knots), dtMs = 2, samples = 1501)
  var x = ones(1501)
  x.applyGainCurve(plan, startMs = 0)
  doAssert x[50] == dbToFactor(0) and x[550] == dbToFactor(20) and x[1050] == dbToFactor(14)
  doAssert near(x[300], dbToFactor(10), 1e-15), "half way: 10 dB"
  doAssert x[0] == 1 and x[10] == 1, "flat before the first knot"
  doAssert x[1500] == dbToFactor(14), "flat after the last"

block t0_shifts:
  let c = knotGain(knots)
  let plan = initGainCurvePlan(float64, c, dtMs = 2, samples = 1001)
  for startMs in [100.0, 2.0, 1001.0, 0.7, -20.0, -3.3, 1500.0, 1999.0, 5000.0, -1e7]:
    var x = ones(1001)
    x.applyGainCurve(plan, startMs)
    for j in 0..<1001:
      let want = c.factorAt(startMs + float64(j) * 2)
      # Off the grid: (ΔdB·ln10/20)²/8 with 0.04 dB per sample: 2.7e-6.
      doAssert near(x[j], want, 3e-6), $startMs & " at " & $j & ": " & $x[j] & " vs " & $want
    if startMs == 100:
      doAssert x[0] == 1 and x[500] == 10, "on the grid: exact table values"
  # A power curve off the grid: each factor within the documented bound,
  # (ΔdB·ln10/20)²/8, ΔdB the gain change over one sample (largest early on).
  let p = powerGain(2)
  let pp = initGainCurvePlan(float64, p, dtMs = 2, samples = 2001)
  var x = ones(2001)
  x.applyGainCurve(pp, startMs = 51.3)
  for j in 0..<2001:
    let t = 51.3 + float64(j) * 2
    let dDb = p.gainDbAt(ceil(t / 2) * 2) - p.gainDbAt(floor(t / 2) * 2)
    let bound = (dDb * ln(10.0) / 20) ^ 2 / 8
    doAssert abs(x[j] / p.factorAt(t) - 1) <= 1.01 * bound + 1e-12, $j
  # A negative power: g''/g = p(p − 1)/t², larger than (g'/g)².
  let inv = powerGain(-1, refMs = 100)
  let ip = initGainCurvePlan(float64, inv, dtMs = 2, samples = 1001)
  var y = ones(1001)
  y.applyGainCurve(ip, startMs = 21)
  for j in 0..<1001:
    let t0 = floor((21 + float64(j) * 2) / 2) * 2
    let bound = 2.0 * (2 / t0) ^ 2 / 8 # |p(p − 1)|·(dt/t)²/8 at the interval start
    doAssert abs(y[j] / inv.factorAt(21 + float64(j) * 2) - 1) <= 1.01 * bound, $j
  # A knot between grid points (1101 ms): first order there, within
  # |Δs|·dt/4 (s = ln10/20 × dB per ms); second order elsewhere.
  let k = knotGain([GainKnot(tMs: 0, dB: 0), GainKnot(tMs: 1101, dB: 20),
    GainKnot(tMs: 2000, dB: 0)])
  let kp = initGainCurvePlan(float64, k, dtMs = 2, samples = 1001)
  var z = ones(1001)
  z.applyGainCurve(kp, startMs = 1)
  let s1 = ln(10.0) / 20 * 20 / 1101
  let s2 = ln(10.0) / 20 * -20 / 899
  var atKink = 0.0
  for j in 0..<1001:
    let t = 1 + float64(j) * 2
    let err = abs(z[j] / k.factorAt(t) - 1)
    if abs(t - 1101) < 2: atKink = max(atKink, err)
    else: doAssert err <= (max(abs(s1), abs(s2)) * 2) ^ 2 / 8 * 1.01, $j
  doAssert atKink > 1e-4 and atKink <= abs(s1 - s2) * 2 / 4 * 1.01, $atKink
  var nanT = ones(10)
  nanT.applyGainCurve(pp, startMs = NaN)
  doAssert nanT == ones(10), "a NaN start time leaves the trace alone"

block block_form_follows_each_rows_time:
  var g = cmpGather[float32](0, [0.0, 0, 0, 0, 0, 0], 400, dtMs = 2)
  for r in 0..<6:
    for j in 0..<400: g.data[r, j] = 1
  g.keys[0].t0Ms = 0
  g.keys[1].t0Ms = 100
  g.keys[2].t0Ms = 51 # off the grid
  g.keys[3].t0Ms = NaN
  g.keys[4].t0Ms = 0
  g.keys[4].status = tsDead
  g.keys[5].t0Ms = 0
  g.setLive(5, 150)
  var w = g.window(firstSample = 50, cols = 300)
  let meta = initChunkMeta(w.rows, dtMs = 2, firstSample = 50)
  let c = exponentialGain(10)
  let plan = initGainCurvePlan(float32, c, dtMs = 2, samples = 300)
  w.data.applyGainCurve(meta, plan)
  for r in [0, 1, 2]:
    for j in 0..<300:
      let want = c.factorAt(meta.sampleTimeMs(r, j))
      doAssert near(float64(w.data[r, j]), want, 2e-6), $r & " " & $j
  for j in 0..<300:
    doAssert w.data[3, j] == 1, "NaN t0: untouched"
    doAssert w.data[4, j] == 1, "dead: untouched"
  for j in 0..<100: doAssert near(float64(w.data[5, j]), c.factorAt(100 + 2 * float64(j)), 2e-6)
  for j in 100..<300: doAssert w.data[5, j] == 0, "padding untouched"
  var wide = zeros[float32](6, 301)
  doAssertRaises ValueError: wide.applyGainCurve(meta, plan)

block unbuilt_curves_and_dt:
  var c: GainCurve # default: no knots
  doAssert c.gainDbAt(10).isNaN and c.factorAt(10).isNaN
  doAssertRaises ValueError: discard initGainCurvePlan(float32, c, 2, 10)
  var g = cmpGather[float32](0, [0.0, 0], 50, dtMs = 4)
  let meta = initChunkMeta(g.rows, dtMs = 4)
  let plan = initGainCurvePlan(float32, powerGain(2), dtMs = 2, samples = 50)
  doAssertRaises ValueError: g.data.applyGainCurve(meta, plan)

block no_allocation:
  var g = cmpGather[float32](0, [0.0, 0, 0, 0], 500, dtMs = 2)
  g.keys[1].t0Ms = 1.5
  g.keys[2].t0Ms = -40
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let plan = initGainCurvePlan(float32, knotGain(knots), dtMs = 2, samples = 500)
  let n = allocsIn:
    for _ in 0..<3: g.data.applyGainCurve(meta, plan)
  doAssert n == 0, $n & " allocations"
