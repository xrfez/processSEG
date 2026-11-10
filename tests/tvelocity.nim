## Velocity models (velocity/model, velocity/formats): constant and gradient
## functions, knots interpolated in V²t at and between the nodes, the Dix
## round trip and its NaN, non-monotonic input reported, 2D and 3D fields,
## and VELF, HANDVEL and CSV picks (tests/data/sample.velf).

import std/[assertions, math, os, strutils]
import processSEG
import utils/allocs

func near(a, b: float64, tol = 1e-12): bool = abs(a - b) <= tol * max(1, abs(b))

template raisesWith(want: string, body: untyped) =
  try:
    body
    doAssert false, "expected a ValueError with " & want
  except ValueError as e:
    doAssert want in e.msg, "'" & e.msg & "' lacks '" & want & "'"

func same(a, b: float64): bool = a == b or a.isNaN and b.isNaN

func `~=`(a, b: seq[VelocityPick]): bool =
  ## Equal picks (NaN locations compare equal).
  if a.len != b.len: return false
  for i in 0..<a.len:
    if not (same(a[i].cdp, b[i].cdp) and same(a[i].inline, b[i].inline) and
        same(a[i].crossline, b[i].crossline) and a[i].velocity == b[i].velocity):
      return false
  true

func k(t, v: float64): VelocityKnot = VelocityKnot(tMs: t, vrms: v)

block constant_and_gradient:
  let c = constantVelocity(2000)
  doAssert c.kind == vkConstant
  for t in [0.0, 500, 1e6]:
    doAssert c.vrmsAt(t) == 2000 and c.vintAt(t) == 2000
  doAssert c.vrmsAt(NaN).isNaN
  let g = gradientVelocity(1500, 500) # Vint = 1500 + 500·t[s]
  doAssert g.vintAt(0) == 1500 and g.vintAt(2000) == 2500
  # Vrms² = (1/s) ∫ (v0 + kτ)² dτ, against a midpoint sum.
  for tMs in [1.0, 1000, 3000]:
    let s = tMs / 1000
    var sum = 0.0
    const n = 100_000
    for i in 0..<n:
      let tau = (float64(i) + 0.5) * s / n
      sum += (1500 + 500 * tau) ^ 2
    doAssert near(g.vrmsAt(tMs), sqrt(sum / n), 1e-9)
  doAssert g.vrmsAt(0) == 1500 and g.vrmsAt(-5) == 1500
  let neg = gradientVelocity(1500, -1000)
  doAssert neg.vintAt(1000) == 500 and neg.vintAt(1500).isNaN
  doAssert neg.vrmsAt(1500) > 0
  for v in [0.0, -1, NaN, Inf]:
    doAssertRaises ValueError: discard constantVelocity(v)
    doAssertRaises ValueError: discard gradientVelocity(v, 0)
  doAssertRaises ValueError: discard gradientVelocity(1500, NaN)
  # A default function gives NaN, never a crash.
  var d: VelocityFunction
  doAssert d.vrmsAt(100).isNaN and d.vintAt(100).isNaN

block knots_at_and_between_nodes:
  let ks = [k(200, 1600), k(1000, 2000), k(2000, 2200)]
  let f = rmsVelocity(ks)
  doAssert f.kind == vkKnots and f.knots == @ks
  for kn in ks: doAssert near(f.vrmsAt(kn.tMs), kn.vrms)
  # Before the first knot: V₁ (one layer from 0); after the last: Vₙ.
  doAssert f.vrmsAt(0) == 1600 and f.vrmsAt(100) == 1600
  doAssert f.vrmsAt(5000) == 2200
  # Between knots, W = V²·t is linear.
  for (a, b) in [(0, 1), (1, 2)]:
    let (t1, t2) = (ks[a].tMs, ks[b].tMs)
    let (w1, w2) = (ks[a].vrms ^ 2 * t1, ks[b].vrms ^ 2 * t2)
    for frac in [0.1, 0.5, 0.9]:
      let t = t1 + frac * (t2 - t1)
      doAssert near(f.vrmsAt(t) ^ 2 * t, w1 + frac * (w2 - w1))
  # The interval velocity is the Dix velocity of the layer, constant inside.
  let v12 = sqrt((2000.0 ^ 2 * 1000 - 1600.0 ^ 2 * 200) / 800)
  doAssert near(f.vintAt(500), v12) and near(f.vintAt(999), v12)
  doAssert f.vintAt(100) == 1600 and f.vintAt(2000) == 2200
  # A knot at t = 0 sets only the value at 0 (module documentation).
  let z = rmsVelocity([k(0, 1500), k(1000, 2000)])
  doAssert z.vrmsAt(0) == 1500 and near(z.vrmsAt(500), 2000)
  # One knot: constant.
  let one = rmsVelocity([k(800, 1700)])
  doAssert one.vrmsAt(0) == 1700 and one.vrmsAt(5000) == 1700
  var out1 = newSeq[float64](5)
  f.vrmsInto(startMs = 0, dtMs = 500, out1)
  for j in 0..<5: doAssert out1[j] == f.vrmsAt(float64(j) * 500)
  doAssert allocsIn(f.vrmsInto(0, 500, out1)) == 0
  doAssertRaises ValueError: f.vrmsInto(0, 0, out1)

block non_monotonic_input_reported:
  raisesWith("knot 2"): discard rmsVelocity([k(0, 1500), k(1000, 1800), k(800, 1900)])
  raisesWith("knot 1"): discard rmsVelocity([k(500, 1500), k(500, 1800)])
  raisesWith("knot 0"): discard rmsVelocity([k(-1, 1500)])
  raisesWith("knot 1: vrms"): discard rmsVelocity([k(0, 1500), k(100, 0)])
  raisesWith("knot 0"): discard rmsVelocity([k(NaN, 1500)])
  raisesWith("at least one knot"): discard rmsVelocity([])
  var vint = newSeq[float64](3)
  raisesWith("time 2"): dixIntervalInto([100.0, 200, 150], [1500.0, 1600, 1700], vint)
  raisesWith("same length"): dixIntervalInto([100.0, 200], [1500.0, 1600, 1700], vint)

block dix_round_trip:
  let t = [120.0, 400, 850, 1300, 2100, 3600]
  let layers = [1480.0, 1750, 2100, 1900, 2600, 3200] # with an inversion
  var vrms = newSeq[float64](t.len)
  rmsFromIntervalInto(t, layers, vrms)
  doAssert vrms[0] == layers[0]
  var back = newSeq[float64](t.len)
  dixIntervalInto(t, vrms, back)
  for i in 0..<t.len: doAssert near(back[i], layers[i], 1e-12)
  # And rms → interval → rms.
  var again = newSeq[float64](t.len)
  rmsFromIntervalInto(t, back, again)
  for i in 0..<t.len: doAssert near(again[i], vrms[i], 1e-12)
  # The knots of a function interpolated in V²t give the same layers.
  var knots: seq[VelocityKnot]
  for i in 0..<t.len: knots.add k(t[i], vrms[i])
  let f = rmsVelocity(knots)
  for i in 1..<t.len:
    doAssert near(f.vintAt((t[i - 1] + t[i]) / 2), layers[i], 1e-12)
  # intervalVelocity: the layers come back, and the last one continues.
  let fi = intervalVelocity(t, layers)
  for i in 0..<t.len: doAssert near(fi.vrmsAt(t[i]), vrms[i], 1e-12)
  for i in 1..<t.len:
    doAssert near(fi.vintAt((t[i - 1] + t[i]) / 2), layers[i], 1e-12)
  doAssert near(fi.vintAt(5000), layers[^1], 1e-12)
  doAssert near(fi.vrmsAt(3000), f.vrmsAt(3000), 1e-12)
  # A first time of 0: a layer of no thickness, vint[0] = vrms[0].
  var v0 = newSeq[float64](2)
  dixIntervalInto([0.0, 1000], [1500.0, 2000], v0)
  doAssert v0 == @[1500.0, 2000]
  rmsFromIntervalInto([0.0, 1000], [1500.0, 2000], v0)
  doAssert v0 == @[1500.0, 2000]

block dix_negative_is_nan:
  # W = V²t drops from 2000²·1000 to 1500²·1200: Vint² < 0.
  var vint = newSeq[float64](3)
  dixIntervalInto([1000.0, 1200, 2000], [2000.0, 1500, 2500], vint)
  doAssert vint[0] == 2000 and vint[1].isNaN and not vint[2].isNaN
  let f = rmsVelocity([k(1000, 2000), k(1200, 1500), k(2000, 2500)])
  doAssert f.vintAt(1100).isNaN
  doAssert f.vrmsAt(1100) > 0 # still a positive RMS velocity
  doAssertRaises ValueError:
    rmsFromIntervalInto([100.0, 200], [1500.0, -1], vint.toOpenArray(0, 1))

block field_2d:
  let a = rmsVelocity([k(0, 1500), k(1000, 2000), k(2000, 2400)])
  let b = rmsVelocity([k(0, 1600), k(800, 2200), k(2400, 2800)])
  # One node, knots on the 4 ms grid: the field is the function, at and
  # between grid times.
  let one = initVelocityField([500.0], [a], dtMs = 4, tMaxMs = 3000)
  doAssert one.kind == fk2D and one.len == 1 and one.samples == 751
  for t in [0.0, 1, 4, 6, 999, 1000, 1002, 2001, 2999.5, 3000, 4000]:
    doAssert near(one.vrmsAt(500, t), a.vrmsAt(t), 1e-12), $t
    doAssert near(one.vrmsAt(-1e9, t), a.vrmsAt(t), 1e-12)
  # Two nodes: V² interpolated linearly in space; clamped outside.
  let f = initVelocityField([100.0, 300], [a, b], dtMs = 4, tMaxMs = 3000)
  doAssert f.locations == @[100.0, 300]
  for t in [0.0, 400, 1000, 1700, 2500]:
    doAssert near(f.vrmsAt(100, t), a.vrmsAt(t), 1e-12)
    doAssert near(f.vrmsAt(300, t), b.vrmsAt(t), 1e-12)
    doAssert near(f.vrmsAt(50, t), a.vrmsAt(t), 1e-12)
    doAssert near(f.vrmsAt(900, t), b.vrmsAt(t), 1e-12)
    doAssert near(f.vrmsAt(150, t) ^ 2, 0.75 * a.vrmsAt(t) ^ 2 + 0.25 *
        b.vrmsAt(t) ^ 2,
      1e-12)
  # Off-grid knots: the grid resampling is close, not exact (worst at a
  # knot, where W has a kink between grid times: 3.3e-4 here).
  let c = rmsVelocity([k(0, 1500), k(1001, 2000), k(2003, 2400)])
  let fc = initVelocityField([0.0], [c], dtMs = 4, tMaxMs = 3000)
  for t in countup(0, 3000, 7):
    doAssert near(fc.vrmsAt(0, float64(t)), c.vrmsAt(float64(t)), 5e-4)
  doAssert f.vrmsAt(NaN, 100).isNaN and f.vrmsAt(200, NaN).isNaN
  doAssert f.vrmsAt(200, 1, 100).isNaN # a 3D query on a 2D field
  var v = newSeq[float64](1001)
  f.vrmsInto(location = 220, startMs = 1, dtMs = 2, v)
  for j in 0..<v.len: doAssert v[j] == f.vrmsAt(220, 1 + 2 * float64(j))
  doAssert allocsIn(f.vrmsInto(220, 1, 2, v)) == 0
  f.vrmsInto(NaN, 0, 2, v)
  doAssert v[0].isNaN and v[^1].isNaN
  doAssertRaises ValueError: f.vrmsInto(1, 2, 0, 2, v) # 3D form
  doAssertRaises ValueError: f.vrmsInto(1, 0, -2, v)
  raisesWith("location 1"):
    discard initVelocityField([100.0, 100], [a, b], 4, 3000)
  raisesWith("functions 1"): discard initVelocityField([100.0, 200], [a], 4, 3000)
  raisesWith("dtMs"): discard initVelocityField([100.0], [a], 0, 3000)
  raisesWith("tMaxMs"): discard initVelocityField([100.0], [a], 4, NaN)
  var d: VelocityFunction
  raisesWith("not initialized"): discard initVelocityField([100.0], [d], 4, 100)
  var empty: VelocityField
  doAssert empty.vrmsAt(0, 0).isNaN and empty.len == 0

block field_3d:
  let fs = [constantVelocity(1500), constantVelocity(1700),
            constantVelocity(1900), constantVelocity(2100),
            constantVelocity(2300), constantVelocity(2500)]
  # inlines [10, 20], crosslines [1, 2, 3]: fs[i·3 + j].
  let f = initVelocityField([10.0, 20], [1.0, 2, 3], fs, dtMs = 8, tMaxMs = 1000)
  doAssert f.kind == fk3D and f.len == 6 and f.crosslines == @[1.0, 2, 3]
  for i, il in [10.0, 20]:
    for j, xl in [1.0, 2, 3]:
      doAssert near(f.vrmsAt(il, xl, 500), fs[i * 3 + j].vrmsAt(0))
  # Bilinear in V².
  let mid = f.vrmsAt(15, 1.5, 300) ^ 2
  doAssert near(mid, (1500.0 ^ 2 + 1700.0 ^ 2 + 2100.0 ^ 2 + 2300.0 ^ 2) / 4, 1e-12)
  let edge = f.vrmsAt(12.5, 3, 300) ^ 2 # on the last crossline
  doAssert near(edge, 0.75 * 1900.0 ^ 2 + 0.25 * 2500.0 ^ 2, 1e-12)
  doAssert near(f.vrmsAt(0, 0, 0), 1500) and near(f.vrmsAt(99, 99, 0), 2500)
  doAssert f.vrmsAt(15, NaN, 0).isNaN and f.vrmsAt(15, 0).isNaN
  var v = newSeq[float64](10)
  f.vrmsInto(15, 1.5, 0, 4, v)
  doAssert near(v[3] ^ 2, mid, 1e-12)
  doAssert allocsIn(f.vrmsInto(15, 1.5, 0, 4, v)) == 0
  doAssertRaises ValueError: f.vrmsInto(15, 0, 4, v) # 2D form
  raisesWith("needs 6 functions"):
    discard initVelocityField([10.0, 20], [1.0, 2, 3], fs[0..4], 8, 1000)
  raisesWith("crossline 2"):
    discard initVelocityField([10.0, 20], [1.0, 3, 2], fs, 8, 1000)

block field_from_picks:
  let a = constantVelocity(1500)
  let b = constantVelocity(2500)
  let p2 = [VelocityPick(cdp: 300, inline: NaN, crossline: NaN, velocity: b),
            VelocityPick(cdp: 100, inline: NaN, crossline: NaN, velocity: a)]
  let f2 = initVelocityField(p2, dtMs = 4, tMaxMs = 100)
  doAssert f2.kind == fk2D and f2.locations == @[100.0, 300]
  doAssert near(f2.vrmsAt(100, 50), 1500) and near(f2.vrmsAt(300, 50), 2500)
  var p3: seq[VelocityPick]
  for il in [20.0, 10]:
    for xl in [5.0, 1]:
      p3.add VelocityPick(cdp: NaN, inline: il, crossline: xl,
        velocity: constantVelocity(1000 + il * 10 + xl))
  let f3 = initVelocityField(p3, dtMs = 4, tMaxMs = 100)
  doAssert f3.kind == fk3D and f3.locations == @[10.0, 20] and
    f3.crosslines == @[1.0, 5]
  doAssert near(f3.vrmsAt(20, 1, 0), 1201) and near(f3.vrmsAt(10, 5, 0), 1105)
  raisesWith("is missing"): discard initVelocityField(p3[0..2], 4, 100)
  raisesWith("repeats"): discard initVelocityField(p3 & p3[0], 4, 100)
  raisesWith("is 3D"): discard initVelocityField(@p2 & p3[0], 4, 100)
  raisesWith("location 1"): discard initVelocityField(@p2 & p2[0], 4, 100)
  raisesWith("at least one pick"): discard initVelocityField(newSeq[
      VelocityPick](), 4, 100)

block sample_velf_parses:
  let picks = readVelocities(currentSourcePath.parentDir / "data" / "sample.velf")
  doAssert picks.len == 3
  doAssert picks[0].cdp == 101 and picks[1].cdp == 121 and picks[2].cdp == 141
  doAssert picks[0].inline.isNaN
  doAssert picks[0].velocity.knots == @[k(0, 1500), k(400, 1650), k(900, 1900),
    k(1500, 2150), k(2200, 2450), k(3000, 2700)] # continuation card joined
  doAssert picks[1].velocity.knots.len == 4
  doAssert picks[2].velocity.knots[^1] == k(3500, 2900)
  let field = initVelocityField(picks, dtMs = 4, tMaxMs = 4000)
  doAssert near(field.vrmsAt(121, 1000), 1950, 1e-12)
  doAssert parseVelf(readFile(currentSourcePath.parentDir / "data" /
    "sample.velf")) ~= picks
  raisesWith("line 2: cdp 7: time 500 after 1000"):
    discard parseVelf("VELF 7 0 1500 1000 1800\nVELF 500 1600 1400 1700\n")
  raisesWith("line 1: a VELF continuation"): discard parseVelf("VELF 0 1500\n")
  raisesWith("line 1: 'x15' is not a number"): discard parseVelf("VELF 7 0 x15\n")

block handvel:
  let text = """
* a comment
HANDVEL     1001
        0     1500      500     1800
     1000     2100

HANDVEL     1021
      100     1550     1200     2200
END
"""
  let picks = parseHandvel(text)
  doAssert picks.len == 2 and picks[0].cdp == 1001 and picks[1].cdp == 1021
  doAssert picks[0].velocity.knots == @[k(0, 1500), k(500, 1800), k(1000, 2100)]
  doAssert parseVelocities(text) ~= picks
  let p3 = parseHandvel("HANDVEL 100 200\n0 1500 1000 2000\nHANDVEL 100 210\n0 1500\n")
  doAssert p3.len == 2 and p3[0].inline == 100 and p3[0].crossline == 200 and
    p3[0].cdp.isNaN
  raisesWith("line 3: cdp 5: time 0 after 0"):
    discard parseHandvel("HANDVEL 5\n0 1500\n0 1600\n")
  raisesWith("line 2: an odd count"): discard parseHandvel("HANDVEL 5\n0 1500 100\n")
  raisesWith("line 1: numbers outside"): discard parseHandvel("0 1500\n")
  raisesWith("line 1: cdp 5 has no time-velocity pairs"):
    discard parseHandvel("HANDVEL 5\nHANDVEL 6\n0 1500\n")
  raisesWith("line 1: a location"): discard parseHandvel("HANDVEL 1 2 3\n")

block csv:
  let text = "cdp,tMs,vrms\n100,0,1500\n200,0,1600\n100,1000,2000\n200, 900 ,2100\n"
  let picks = parseVelocityCsv(text)
  doAssert picks.len == 2 and picks[0].cdp == 100 and picks[1].cdp == 200
  doAssert picks[0].velocity.knots == @[k(0, 1500), k(1000, 2000)]
  doAssert picks[1].velocity.knots == @[k(0, 1600), k(900, 2100)]
  doAssert parseVelocities(text) ~= picks
  let p3 = parseVelocities("10,1,0,1500\n10,2,0,1600\n10,1,500,1700\n")
  doAssert p3.len == 2 and p3[0].inline == 10 and p3[0].crossline == 1 and
    p3[0].velocity.knots.len == 2
  raisesWith("line 4: cdp 100: time 400 after 500"):
    discard parseVelocityCsv("cdp,tMs,vrms\n100,500,1500\n200,0,1600\n100,400,2000\n")
  raisesWith("line 2: 4 columns"): discard parseVelocityCsv("1,0,1500\n1,0,1500,3\n")
  raisesWith("line 1: a velocity CSV"): discard parseVelocityCsv("1,2\n")
  raisesWith("line 2: 'abc'"): discard parseVelocityCsv("1,0,1500\nabc,1,2\n")
  doAssert parseVelocities("").len == 0
