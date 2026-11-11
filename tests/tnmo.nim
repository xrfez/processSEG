## NMO (stack/nmo): hyperbolas flattened to within one sample (constant
## and time-variant velocities, a 2D field by CDP, linear and cubic), zero
## offset is the identity, the stretch mute against the analytic stretch
## and its taper, each row's own t0Ms, resampling onto another time axis,
## NaN and dead rows, validation and no allocation.

import std/[assertions, math, random]
import processSEG
import utils/[allocs, synth]

const dt = 2.0

func peakCol(x: openArray[float64], near: int, halfWidth: int): float64 =
  ## The fractional column of the largest sample within `halfWidth` of
  ## `near` (parabolic interpolation of the peak).
  var best = max(0, near - halfWidth)
  for j in max(0, near - halfWidth) .. min(x.high, near + halfWidth):
    if x[j] > x[best]: best = j
  if best == 0 or best == x.high: return float64(best)
  let (a, b, c) = (x[best - 1], x[best], x[best + 1])
  float64(best) + 0.5 * (a - c) / (a - 2 * b + c)

proc rowOf(g: SynthGather[float64], r: int): seq[float64] =
  result = newSeq[float64](g.cols)
  for j in 0..<g.cols: result[j] = g.data[r, j]

proc checkFlat(g: SynthGather[float64], t0s: openArray[float64],
    first = 0) =
  for r in first..<g.len:
    let x = g.rowOf(r)
    for t0 in t0s:
      let c = peakCol(x, int(t0 / dt), 8)
      doAssert abs(c * dt - t0) <= dt, "row " & $r & " t0 " & $t0 & ": " & $(c * dt)

block constant_velocity_flattened:
  let offsets = [0.0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
  for interp in [siLinear, siCubic]:
    var g = cmpGather[float64](0, offsets, 1000, dt)
    for t0 in [600.0, 1000, 1600]: g.addHyperbola(t0, 2200, freqHz = 25)
    let meta = initChunkMeta(g.rows, dt)
    var ws = initWorkspace()
    ws.reserve(nmoWorkspace(float64, g.cols))
    applyNmo(g.data, meta, constantVelocity(2200), initNmoParams(interp,
        maxStretch = Inf), ws)
    checkFlat(g, [600.0, 1000, 1600])

block time_variant_velocity_flattened:
  # Reflections at the RMS velocity of a gradient function at their t0.
  let f = gradientVelocity(1600, 900)
  let offsets = [0.0, 400, 800, 1200, 1600, 2000, 2400]
  var g = cmpGather[float64](0, offsets, 1500, dt, cdp = 20)
  let t0s = [500.0, 1200, 2400]
  for t0 in t0s: g.addHyperbola(t0, f.vrmsAt(t0), freqHz = 25)
  var h = g.clone
  let meta = initChunkMeta(g.rows, dt)
  var ws = initWorkspace()
  ws.reserve(nmoWorkspace(float64, g.cols))
  let p = initNmoParams(siCubic, maxStretch = Inf)
  applyNmo(g.data, meta, f, p, ws)
  checkFlat(g, t0s)
  # The same through a 2D field by CDP (ensemble 20 lies between the nodes),
  # within the field's off-grid accuracy (≲ 1e-3 in velocity), below the
  # early, heavily stretched times.
  let field = initVelocityField([10.0, 30], [f, f], dtMs = 4, tMaxMs = 3000)
  applyNmo(h.data, meta, field, p, ws)
  var worst = 0.0
  for r in 0..<g.len:
    for j in 100..<g.cols:
      worst = max(worst, abs(g.data[r, j] - h.data[r, j]))
  doAssert worst <= 2e-3

block zero_offset_is_identity:
  var g = cmpGather[float32](0, [0.0, 0.0], 300, dt)
  g.data[0, 10] = 1
  g.data[1, 200] = -2
  let before = g.data.clone
  let meta = initChunkMeta(g.rows, dt)
  var ws = initWorkspace()
  ws.reserve(nmoWorkspace(float32, g.cols))
  for interp in [siLinear, siCubic]:
    applyNmo(g.data, meta, constantVelocity(2000), initNmoParams(interp), ws)
    doAssert g.data == before

block stretch_mute:
  let v = 2000.0
  let x = 1500.0
  let maxS = 0.3
  # The analytic mute time: t/t0 − 1 = maxS, t0 = (x/v)/sqrt((1+s)² − 1).
  let muteMs = 1000 * x / v / sqrt((1 + maxS) ^ 2 - 1)
  doAssert abs(nmoStretch(x, v, muteMs) - maxS) <= 1e-12
  doAssert nmoStretch(x, v, 0) == Inf and nmoStretch(0, v, 0) == 0
  var src = newSeq[float64](2000) # longer than dst: t > t0
  for j in 0..<src.len: src[j] = 1
  var vel = newSeq[float64](1000)
  for v0 in vel.mitems: v0 = v
  var dst = newSeq[float64](1000)
  let first = nmoInto(src, 0, dst, 0, dt, x, vel, initNmoParams(
      maxStretch = maxS))
  # The discrete stretch is within a sample of the analytic one.
  doAssert abs(float64(first) * dt - muteMs) <= dt, $first & " " & $muteMs
  for j in 0..<first: doAssert dst[j] == 0
  for j in first..<1000: doAssert dst[j] == 1
  # A taper ramps in over taperMs before the mute, on the muted side.
  let tapered = nmoInto(src, 0, dst, 0, dt, x, vel, initNmoParams(
      maxStretch = maxS, taperMs = 40))
  doAssert tapered == first - 19 # the ramp starts at first − 20, weight 0
  for j in 0..<tapered: doAssert dst[j] == 0
  for j in tapered..<first:
    doAssert dst[j] > 0 and dst[j] < 1 and (j == tapered or dst[j] > dst[j - 1])
  for j in first..<1000: doAssert dst[j] == 1
  # A smaller limit mutes deeper; Inf mutes nothing below t0 = 0.
  doAssert nmoInto(src, 0, dst, 0, dt, x, vel, initNmoParams(
      maxStretch = 0.1)) > first
  doAssert nmoInto(src, 0, dst, 0, dt, x, vel, initNmoParams(
      maxStretch = Inf)) == 0
  # A velocity that folds the mapping back is muted (Δt ≤ 0).
  for j in 0..<500: vel[j] = 400 + 0.5 * float64(j)
  doAssert nmoInto(src, 0, dst, 0, dt, x, vel, initNmoParams(
      maxStretch = 1e9)) > 0

block own_t0_and_resampling:
  # A row starting at −100 ms: the event at absolute t is found at t.
  var g = cmpGather[float64](0, [1200.0], 800, dt)
  g.keys[0].t0Ms = -100
  g.addHyperbola(700, 2400, freqHz = 25)
  let meta = initChunkMeta(g.rows, dt)
  var ws = initWorkspace()
  ws.reserve(nmoWorkspace(float64, g.cols))
  applyNmo(g.data, meta, constantVelocity(2400), initNmoParams(
      maxStretch = Inf), ws)
  doAssert abs(peakCol(g.rowOf(0), (700 + 100) div 2, 8) * dt - 100 - 700) <= dt
  # The kernel onto an axis starting at 200 ms (a stack's).
  var h = cmpGather[float64](0, [1200.0], 800, dt)
  h.addHyperbola(700, 2400, freqHz = 25)
  var dst = newSeq[float64](400)
  var vel = newSeq[float64](400)
  for v in vel.mitems: v = 2400
  discard nmoInto(h.rowOf(0), 0, dst, 200, dt, 1200, vel, initNmoParams(
      siCubic, maxStretch = Inf))
  doAssert abs(peakCol(dst, 250, 8) * dt + 200 - 700) <= dt

block nan_and_dead_rows:
  var g = cmpGather[float32](0, [500.0, 600, 700, 800, 900], 100, dt, cdp = 5)
  for r in 0..<g.len:
    for j in 0..<g.cols: g.data[r, j] = float32(r + 1)
  g.keys[0].offset = NaN
  g.keys[1].t0Ms = NaN
  g.keys[2].status = tsDead
  g.keys[3].ensemble = NaN # no location for a field
  let before = g.data.clone
  let meta = initChunkMeta(g.rows, dt)
  var ws = initWorkspace()
  ws.reserve(nmoWorkspace(float32, g.cols))
  let field = initVelocityField([5.0], [constantVelocity(2000)], dtMs = 4,
    tMaxMs = 400)
  applyNmo(g.data, meta, field, initNmoParams(), ws)
  for r in 0..3:
    for j in 0..<g.cols: doAssert g.data[r, j] == before[r, j]
  doAssert g.data[4, 0] == 0 # corrected: the top is stretch-muted
  # The kernel zeroes its output for a NaN offset.
  var dst = [1.0, 2, 3]
  doAssert nmoInto([1.0, 2, 3], 0, dst, 0, dt, NaN, [2000.0, 2000, 2000],
    initNmoParams()) == 3
  doAssert dst == [0.0, 0, 0]

block validation_and_allocation:
  for s in [0.0, -0.1, NaN]:
    doAssertRaises ValueError: discard initNmoParams(maxStretch = s)
  for t in [-1.0, NaN, Inf]:
    doAssertRaises ValueError: discard initNmoParams(taperMs = t)
  let p = initNmoParams(siCubic, 0.5, 10)
  doAssert p.interp == siCubic and p.maxStretch == 0.5 and p.taperMs == 10
  var g = cmpGather[float32](0, [500.0, 1000], 500, dt)
  g.addHyperbola(400, 2000, freqHz = 30)
  let meta = initChunkMeta(g.rows, dt)
  var ws = initWorkspace()
  doAssertRaises AssertionDefect:
    applyNmo(g.data, meta, constantVelocity(2000), p, ws)
  ws.reserve(nmoWorkspace(float32, g.cols))
  doAssertRaises ValueError:
    applyNmo(g.data, meta, constantVelocity(2000), NmoParams(), ws)
  let f = constantVelocity(2000)
  var b = rowBlock(g.data)
  doAssert allocsIn(applyNmo(b, meta, f, p, ws)) == 0
  let field = initVelocityField([1.0], [f], dtMs = 4, tMaxMs = 1000)
  doAssert allocsIn(applyNmo(b, meta, field, p, ws)) == 0

block nmo_matches_a_reference:
  # nmoInto against a direct evaluation of the NMO mapping with linear
  # interpolation (rows longer than the 256-sample column blocks, random
  # start times, offsets and velocities): exact without a stretch mute, and
  # from the first unmuted column with one.
  proc refNmo(src: seq[float64], srcStart: float64, m: int,
      dstStart, dt, off: float64, vel: seq[float64]): seq[float64] =
    result = newSeq[float64](m)
    let n = src.len
    proc at(i: int): float64 = (if i >= 0 and i < n: src[i] else: 0.0)
    for j in 0..<m:
      let t0 = dstStart + float64(j) * dt
      if t0 < 0: continue
      let tx = 1000 * abs(off) / vel[j]
      let c = (sqrt(t0 * t0 + tx * tx) - srcStart) / dt
      if c.isNaN or c <= -2 or c >= float64(n + 1): continue
      let k = int(floor(c))
      let f = c - float64(k)
      result[j] = (1 - f) * at(k) + f * at(k + 1)
  var r = initRand(3)
  for it in 0..<200:
    let n = r.rand(1..1500)
    let m = r.rand(1..1500)
    var src = newSeq[float64](n)
    for v in src.mitems: v = r.gauss()
    var vel = newSeq[float64](m)
    for v in vel.mitems: v = r.rand(1500.0 .. 3000.0)
    let dt = [1.0, 2.0, 4.0][r.rand(2)]
    let ss = r.rand(-100.0 .. 100.0)
    let ds = r.rand(-100.0 .. 100.0)
    let off = r.rand(-3000.0 .. 3000.0)
    let want = refNmo(src, ss, m, ds, dt, off, vel)
    var dst = newSeq[float64](m)
    discard src.nmoInto(ss, dst, ds, dt, off, vel, initNmoParams(maxStretch = Inf))
    for j in 0..<m: doAssert abs(dst[j] - want[j]) <= 1e-12, $it & " col " & $j
    var muted = newSeq[float64](m)
    let first = src.nmoInto(ss, muted, ds, dt, off, vel, initNmoParams(maxStretch = 0.3))
    for j in 0..<first: doAssert muted[j] == 0
    for j in first..<m: doAssert abs(muted[j] - want[j]) <= 1e-12
