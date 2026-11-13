## Property tests (task 9.1): random parameters and data, fixed seeds.
##
## - random pipelines on random gathers (sizes, dt, live lengths, dead,
##   auxiliary and failed rows, amplitudes from 1e-6 to 1e6, all-zero rows):
##   `compile` either succeeds or raises `ValueError`; `apply` gives finite
##   samples, leaves non-live rows and samples past each live length alone,
##   and in place equals src → dst exactly;
## - filter linearity: FFT and IIR filters, f(a·x + b·y) = a·f(x) + b·f(y);
## - scale invariance of AGC and normalization;
## - shifts and NMO with random (also NaN, huge) shifts, offsets and
##   velocities: finite output, a mute column in range;
## - first-break picks on random traces: a pick in the trace or none, a
##   confidence in [0, 1];
## - velocity parsers on random and mutated text: picks or `ValueError`.
##
## Out-of-bounds access in the `boundChecks: off` kernels shows up under the
## ASan run (`nim asan`), which runs this file too.

import std/[algorithm, assertions, math, random, strutils]
import processSEG
import utils/synth

const propIterations {.intdefine.} = 400
const propSeed {.intdefine.} = 0
const iterations = propIterations
const seedBase = propSeed

proc finiteAll[T](x: Tensor[T]): bool =
  for v in x:
    if classify(float64(v)) in {fcNan, fcInf, fcNegInf}: return false
  true

proc uniform(r: var Rand, lo, hi: float64): float64 = lo + r.rand(1.0) * (hi - lo)

proc logUniform(r: var Rand, lo, hi: float64): float64 =
  exp(r.uniform(ln(lo), ln(hi)))

# -------------------------------------------------------------- random ops --

proc randomCorners(r: var Rand, nyq: float64): Corners =
  ## Corners below Nyquist, sorted, sometimes very narrow ramps.
  var f = [r.uniform(0.5, nyq), r.uniform(0.5, nyq), r.uniform(0.5, nyq),
           r.uniform(0.5, nyq)]
  f.sort
  Corners(f1: f[0], f2: f[1], f3: f[2], f4: f[3])

proc randomSpec(r: var Rand, nyq: float64, butter: bool): FilterSpec =
  if butter:
    let slope = 6.0 * float64(r.rand(1 .. 8))
    case r.rand(2)
    of 0: butterLowCut(r.uniform(0.5, nyq * 0.9), slope)
    of 1: butterHighCut(r.uniform(1, nyq * 0.95), slope)
    else:
      let lo = r.uniform(0.5, nyq * 0.5)
      butterBandPass(lo, r.uniform(lo * 1.1, nyq * 0.95), slope)
  else:
    var s = case r.rand(2)
      of 0: bandPass(r.randomCorners(nyq))
      of 1:
        let a = r.uniform(0.5, nyq * 0.8)
        lowCut(a, r.uniform(a, nyq * 0.9))
      else:
        let a = r.uniform(1, nyq * 0.8)
        highCut(a, r.uniform(a, nyq))
    if r.rand(3) == 0: s = s.withNotch(r.uniform(1, nyq * 0.9), r.uniform(0.5, 5))
    if r.rand(3) == 0: s = s.withZeroDc
    s

proc randomOp(r: var Rand, dtMs: float64): Op =
  let nyq = nyquistHz(dtMs)
  case r.rand(14)
  of 0: demeanOp()
  of 1: detrendOp()
  of 2: polarityOp()
  of 3: scaleOp(r.logUniform(1e-3, 1e3) * (if r.rand(1) == 0: -1.0 else: 1.0))
  of 4: gainDbOp(r.uniform(-60, 60))
  of 5:
    let a = r.uniform(-5, 5)
    clipOp(a, a + r.uniform(0, 10))
  of 6:
    let curve = case r.rand(2)
      of 0: timeMute(r.uniform(-100, 2000))
      of 1: lmoMute(r.uniform(300, 6000), r.uniform(-50, 200), r.uniform(-50, 50))
      else: tableMute([0.0, 500, 2000], [0.0, r.uniform(0, 500), r.uniform(0, 1500)])
    if r.rand(1) == 0: muteTopOp(curve, r.uniform(0, 100))
    else: muteBottomOp(curve, r.uniform(0, 100))
  of 7: filterOp(r.randomSpec(nyq, butter = r.rand(1) == 0),
      TrendRemoval(r.rand(2)))
  of 8: iirOp(r.randomSpec(nyq, butter = true), IirPhase(r.rand(1)),
      TrendRemoval(r.rand(2)))
  of 9: agcOp(r.uniform(dtMs, 1000), AgcMode(r.rand(1)), r.logUniform(0.1, 10))
  of 10: normalizeOp(NormStat(r.rand(2)), NormScope(r.rand(1)),
      r.logUniform(0.1, 10), r.uniform(50, 100))
  of 11:
    let knots = [GainKnot(tMs: 0, dB: r.uniform(-20, 20)),
                 GainKnot(tMs: r.uniform(10, 3000), dB: r.uniform(-20, 40))]
    gainCurveOp(knotGain(knots))
  of 12: gainCurveOp(powerGain(r.uniform(0, 2.5), r.uniform(100, 2000)))
  of 13: gainCurveOp(exponentialGain(r.uniform(-10, 10)))
  else:
    # Out-of-range values: compile must raise ValueError, never a defect.
    case r.rand(4)
    of 0: agcOp(r.uniform(-10, 0))
    of 1: filterOp(highCut(nyq * 1.5, nyq * 2))
    of 2: clipOp(1, 0)
    of 3: scaleOp(NaN)
    else: iirOp(bandPass(r.randomCorners(nyq)))

# ----------------------------------------------------------- random gathers --

proc randomGather[T](r: var Rand, rows, cols: int, dtMs: float64,
    failed: var seq[int]): SynthGather[T] =
  var offs = newSeq[float64](rows)
  for i in 0..<rows: offs[i] = r.uniform(-3000, 3000)
  result = cmpGather[T](0, offs, cols, dtMs)
  failed.setLen 0
  for i in 0..<rows:
    case r.rand(9)
    of 0: result.keys[i].status = tsDead
    of 1: result.keys[i].status = tsAuxiliary
    of 2: failed.add i
    else: discard
    let live = case r.rand(5)
      of 0: 0
      of 1: r.rand(1 .. min(cols, 4))
      of 2: r.rand(0 .. cols)
      else: cols
    result.setLive(i, live)
    let amp = case r.rand(5)
      of 0: 0.0
      of 1: 1e-6
      of 2: 1e6
      else: r.logUniform(1e-2, 1e2)
    var x = newSeq[T](live)
    x.addNoise(amp, seed = r.rand(high(int32)).int64)
    if live > 0 and r.rand(3) == 0: x.addDc(amp * 50)
    if r.rand(4) == 0 and live > 0: x[r.rand(live - 1)] += T(amp * 1000)
    for j in 0..<live: result.data[i, j] = x[j]

proc checkPipelines[T](seed: int64) =
  var r = initRand(seed + seedBase * 1000)
  var compiled, rejected = 0
  var failed: seq[int]
  var ws = initWorkspace()
  for it in 0..<iterations:
    let dtMs = [0.25, 0.5, 1.0, 2.0, 4.0][r.rand(4)]
    let rows = r.rand(1 .. 24)
    let cols = r.rand(1 .. 600)
    var g = randomGather[T](r, rows, cols, dtMs, failed)
    var flow = initPipeline()
    for k in 0 ..< r.rand(1 .. 5): flow.add r.randomOp(dtMs)
    var plan: PipelinePlan[T]
    try:
      plan = flow.compile(T, dtMs, cols)
      inc compiled
    except ValueError:
      inc rejected
      continue
    let meta = initChunkMeta(g.rows, dtMs, failed = failed)
    ws.reserve(plan)
    var inPlace = g.data.clone
    plan.apply(inPlace, meta, ws)
    var dst = zeros[T](rows, cols)
    plan.apply(g.data, dst, meta, ws)
    let ctx = "seed " & $seed & " it " & $it & " " & $T & " ops " & $flow.len
    doAssert inPlace.finiteAll, "non-finite output: " & ctx
    for i in 0..<rows:
      let live = meta.isLive(i)
      for j in 0..<cols:
        doAssert inPlace[i, j] == dst[i, j], "in place != src → dst: " & ctx
        if not live or j >= g.liveCols(i):
          doAssert inPlace[i, j] == g.data[i, j],
            "row " & $i & " col " & $j & " changed (not live): " & ctx
  doAssert compiled > iterations div 2, "too few plans compiled: " & $compiled
  doAssert rejected > 0, "no invalid op was generated"

block random_pipelines:
  checkPipelines[float32](1)
  checkPipelines[float64](2)

proc checkNanContained[T](seed: int64) =
  ## A NaN or Inf sample in one row reaches no other row: the other rows come
  ## out as when that row is a failed one (left out of panel statistics too).
  var r = initRand(seed + seedBase * 1000)
  var failed, failed2: seq[int]
  var ws = initWorkspace()
  var checked = 0
  for it in 0..<iterations:
    let dtMs = [0.5, 1.0, 2.0, 4.0][r.rand(3)]
    let rows = r.rand(2 .. 20)
    let cols = r.rand(4 .. 500)
    var g = randomGather[T](r, rows, cols, dtMs, failed)
    var flow = initPipeline()
    for k in 0 ..< r.rand(1 .. 5):
      # A clip or a mute makes a non-finite sample finite (clamped, or
      # zeroed), and the row then rightly joins panel statistics: left out.
      var op = r.randomOp(dtMs)
      while op.kind in {opClip, opMuteTop, opMuteBottom}: op = r.randomOp(dtMs)
      flow.add op
    let plan = try: flow.compile(T, dtMs, cols) except ValueError: continue
    var bad = -1
    for i in 0..<rows:
      if i notin failed and g.keys[i].status notin {tsDead, tsAuxiliary} and
          g.liveCols(i) > 0:
        bad = i
        break
    if bad < 0: continue
    g.data[bad, r.rand(g.liveCols(bad) - 1)] = T([NaN, Inf, -Inf][r.rand(2)])
    failed2 = failed
    failed2.add bad
    failed2.sort
    let meta = initChunkMeta(g.rows, dtMs, failed = failed)
    let meta2 = initChunkMeta(g.rows, dtMs, failed = failed2)
    ws.reserve(plan)
    var a = zeros[T](rows, cols)
    var b = zeros[T](rows, cols)
    plan.apply(g.data, a, meta, ws)
    plan.apply(g.data, b, meta2, ws)
    var kinds: seq[string]
    for op in flow: kinds.add $op
    let ctx = "seed " & $seed & " it " & $it & " " & $T & " " & kinds.join(" | ") & ", row " & $bad
    for i in 0..<rows:
      if i == bad: continue
      # Rows run in SIMD lane groups: moving a row to another lane (the bad
      # row live or failed) may change the last bits of a float64 IIR, so
      # the tolerance is relative to the row's scale, not to each sample (a
      # sample near zero carries the rounding of the row's larger values).
      var scale = 0.0
      for j in 0..<cols: scale = max(scale, abs(float64(b[i, j])))
      for j in 0..<cols:
        let x = float64(a[i, j])
        let y = float64(b[i, j])
        doAssert x.classify notin {fcNan, fcInf, fcNegInf} and
          abs(x - y) <= 1e-12 * scale, "a non-finite sample leaked: " & ctx &
          " into row " & $i & " col " & $j & ": " & $a[i, j] & " vs " & $b[i, j]
    inc checked
  doAssert checked > iterations div 3

block nan_contained:
  checkNanContained[float32](10)
  checkNanContained[float64](11)

# ---------------------------------------------------------------- linearity --

proc linearityError[T](r: var Rand, n: int, apply: proc (x: var seq[T])): float64 =
  var x = newSeq[T](n)
  var y = newSeq[T](n)
  x.addNoise(1, seed = r.rand(high(int32)).int64)
  y.addNoise(1, seed = r.rand(high(int32)).int64)
  x.addDrift(2, r.uniform(-50, 50))
  let a = r.uniform(-3, 3)
  let b = r.uniform(-3, 3)
  var z = newSeq[T](n)
  var scale = 0.0 # relative to the input: a filter may remove nearly all of it
  for j in 0..<n:
    z[j] = T(a) * x[j] + T(b) * y[j]
    scale = max(scale, abs(float64(z[j])))
  apply(x)
  apply(y)
  apply(z)
  for j in 0..<n:
    result = max(result, abs(float64(z[j]) - (a * float64(x[j]) + b * float64(y[j]))))
  result /= max(scale, 1e-30)

block filter_linearity:
  var r = initRand(3 + seedBase * 1000)
  var ws = initWorkspace()
  var iirChecked = 0
  for it in 0..<60:
    let dtMs = [0.5, 1.0, 2.0, 4.0][r.rand(3)]
    let n = r.rand(8 .. 1500)
    let nyq = nyquistHz(dtMs)
    let trend = TrendRemoval(r.rand(2))
    let fspec = r.randomSpec(nyq, butter = r.rand(1) == 0)
    try:
      let fp = initFilterPlan(float64, fspec, dtMs, n, trend)
      ws.reserve(fp.workspaceSize)
      let e = linearityError[float64](r, r.rand(1 .. n),
        proc (x: var seq[float64]) = fp.applyFilter(x, ws))
      doAssert e < 1e-9, "FFT filter not linear: " & $e & " " & $fspec
    except ValueError: discard # a ramp too narrow for n: rejected
    let ispec = r.randomSpec(nyq, butter = true)
    let phase = IirPhase(r.rand(1))
    try:
      let ip = initIirPlan(float64, ispec, dtMs, n, phase, trend)
      ws.reserve(ip.workspaceSize)
      let e = linearityError[float64](r, r.rand(1 .. n),
        proc (x: var seq[float64]) = ip.applyFilter(x, ws))
      doAssert e < 1e-9, "IIR filter not linear: " & $e & " " & $ispec &
        " dt " & $dtMs & " n " & $n & " " & $trend & " " & $phase
      # float32: the same within single precision.
      let ip32 = initIirPlan(float32, ispec, dtMs, n, phase, trend)
      ws.reserve(ip32.workspaceSize)
      let e32 = linearityError[float32](r, r.rand(1 .. n),
        proc (x: var seq[float32]) = ip32.applyFilter(x, ws))
      doAssert e32 < 1e-3, "float32 IIR filter not linear: " & $e32
      inc iirChecked
    except ValueError: discard # an odd slope for a zero-phase cascade
  doAssert iirChecked > 20

block gain_scale_invariance:
  var r = initRand(4 + seedBase * 1000)
  var ws = initWorkspace()
  for it in 0..<60:
    let n = r.rand(1 .. 1200)
    var x = newSeq[float64](n)
    x.addNoise(r.logUniform(1e-3, 1e3), seed = it)
    let c = r.logUniform(1e-3, 1e3)
    var y = x
    for v in y.mitems: v *= c
    # ε is relative to the trace's RMS, so the AGC is scale invariant.
    let agc = initAgcPlan(r.uniform(6, 800), 2, n, AgcMode(r.rand(1)))
    ws.reserve(agc.workspaceSize)
    var ax = x
    var ay = y
    agc.agc(ax, ws)
    agc.agc(ay, ws)
    for j in 0..<n:
      doAssert ax[j].classify == fcNormal or ax[j] == 0
      doAssert abs(ax[j] - ay[j]) <= 1e-9 * max(1.0, abs(ax[j])), "AGC scale"
    let norm = initNormalizePlan(NormStat(r.rand(2)), n, perTrace,
      percentile = r.uniform(50, 100))
    ws.reserve(norm.workspaceSize)
    var nx = x
    var ny = y
    norm.normalize(nx, ws)
    norm.normalize(ny, ws)
    for j in 0..<n:
      # The percentile histogram bins |x| relative to the maximum, so it is
      # scale invariant up to rounding as well.
      doAssert abs(nx[j] - ny[j]) <= 1e-9 * max(1.0, abs(nx[j])), "normalize scale"

block zero_traces_stay_zero:
  # All-zero input: every op yields zeros (no 0/0 from AGC or normalize).
  var r = initRand(5 + seedBase * 1000)
  var ws = initWorkspace()
  for it in 0..<100:
    let n = r.rand(1 .. 500)
    var flow = initPipeline()
    for k in 0 ..< r.rand(1 .. 4):
      var op = r.randomOp(2)
      while op.kind in {opClip}: op = r.randomOp(2) # clip may move 0 into [lo, hi]
      flow.add op
    let plan = try: flow.compile(float32, 2, n) except ValueError: continue
    ws.reserve(plan)
    var g = cmpGather[float32](0, [100.0, 200, 300], n, dtMs = 2)
    let meta = initChunkMeta(g.rows, 2)
    plan.apply(g.data, meta, ws)
    for v in g.data: doAssert v == 0, "zero in, " & $v & " out"

# ----------------------------------------------------------- shifts and NMO --

block random_shifts:
  var r = initRand(6 + seedBase * 1000)
  for it in 0..<400:
    let n = r.rand(0 .. 300)
    var x = newSeq[float32](n)
    x.addNoise(1, seed = it)
    let shift = case r.rand(5)
      of 0: NaN
      of 1: Inf
      of 2: r.uniform(-1e9, 1e9)
      of 3: float64(r.rand(-n - 3 .. n + 3))
      else: r.uniform(-float64(n) - 3, float64(n) + 3)
    let orig = x
    x.shiftSamples(shift, ShiftInterp(r.rand(1)))
    for j in 0..<n:
      doAssert x[j].classify in {fcNormal, fcSubnormal, fcZero, fcNegZero}
      if shift.isNaN or shift == Inf: doAssert x[j] == orig[j]
    if shift == 0:
      doAssert x == orig

block random_nmo:
  var r = initRand(7 + seedBase * 1000)
  for it in 0..<300:
    let ns = r.rand(1 .. 400)
    let nd = r.rand(1 .. 400)
    var src = newSeq[float32](ns)
    src.addNoise(1, seed = it)
    var dst = newSeq[float32](nd)
    var vel = newSeq[float64](nd)
    let v = case r.rand(3)
      of 0: 1e-3
      of 1: 1e9
      else: r.uniform(300, 8000)
    for j in 0..<nd: vel[j] = v * r.uniform(0.9, 1.1)
    let offset = case r.rand(4)
      of 0: NaN
      of 1: 0.0
      of 2: 1e7
      else: r.uniform(-5000, 5000)
    let p = initNmoParams(ShiftInterp(r.rand(1)), r.uniform(0, 3), r.uniform(0, 50))
    let dt = [0.5, 1.0, 2.0, 4.0][r.rand(3)]
    let first = src.nmoInto(r.uniform(-200, 200), dst, r.uniform(-200, 200),
      dt, offset, vel, p)
    doAssert first in 0 .. nd
    for j in 0..<nd:
      doAssert dst[j].classify in {fcNormal, fcSubnormal, fcZero, fcNegZero},
        "NMO output " & $dst[j]
      if j < first: doAssert dst[j] == 0

# ------------------------------------------------------------- first breaks --

block random_first_breaks:
  var r = initRand(8 + seedBase * 1000)
  var ws = initWorkspace()
  for it in 0..<120:
    let dtMs = [0.5, 1.0, 2.0][r.rand(2)]
    let n = r.rand(1 .. 1500)
    var params = FirstBreakParams(pickMethod: PickMethod(r.rand(2)),
      windowMs: r.uniform(dtMs, 60), ltaMs: r.uniform(dtMs, 300),
      snrWindowMs: r.uniform(dtMs, 80))
    if r.rand(1) == 0:
      params.lowCutHz = r.uniform(1, 10)
      params.highCutHz = r.uniform(20, nyquistHz(dtMs) * 0.9)
    if r.rand(1) == 0:
      params.guideVelocity = r.uniform(500, 6000)
      params.beforeMs = r.uniform(0, 100)
      params.afterMs = r.uniform(0, 200)
    if r.rand(2) == 0:
      params.searchStartMs = r.uniform(-100, float64(n) * dtMs)
      params.searchEndMs = params.searchStartMs + r.uniform(-50, 500)
    let plan = try: initFirstBreakPlan(float32, params, dtMs, n)
               except ValueError: continue
    ws.reserve(plan.workspaceSize)
    let len = r.rand(0 .. n)
    var x = newSeq[float32](len)
    case r.rand(2)
    of 0: discard # zeros
    of 1: x.addNoise(r.logUniform(1e-6, 1e6), seed = it)
    else:
      x.addNoise(0.1, seed = it)
      if len > 0: x.addRicker(dtMs, r.uniform(0, float64(len) * dtMs), 30)
    let t0 = r.uniform(-50, 50)
    let p = plan.pick(x, ws, offset = r.uniform(-3000, 3000), t0Ms = t0)
    doAssert p.confidence >= 0 and p.confidence <= 1, $p
    if p.col < 0:
      doAssert p.timeMs.isNaN and p.confidence == 0
    else:
      doAssert p.col < len, $p & " len " & $len
      doAssert abs(p.timeMs - (t0 + float64(p.col) * dtMs)) < 1e-9
      doAssert p.snrDb.isNaN or p.snrDb.classify != fcInf

# -------------------------------------------------------- velocity parsers --

const velfSeed = """
VELF       101                 1001  2001
VELF       101                  0  1500  1000  2000  2000  2800
VELF       201                  0  1500  1200  2100
"""
const handvelSeed = """
HANDVEL     101
    0  1500  1000  2000  2000  2800
HANDVEL     201
    0  1500  1200  2100
"""
const csvSeed = "cdp,time,velocity\n101,0,1500\n101,1000,2000\n201,0,1500\n"

const nasty = ["-", "1e400", "nan", "inf", ",", "\n", " ", "-0", "99999999999999999999"]

proc mutate(r: var Rand, s: string): string =
  result = s
  for k in 0 .. r.rand(6):
    if result.len == 0: result.add 'x'
    let at = r.rand(result.len - 1)
    case r.rand(5)
    of 0: result[at] = char(r.rand(255))
    of 1: result.delete(at .. at)
    of 2: result.insert($char(r.rand(32 .. 126)), at)
    of 3: result.insert(nasty[r.rand(nasty.high)], at)
    of 4: result.setLen(at)
    else: result = result & result[0 .. at]

block velocity_parsers:
  var r = initRand(9 + seedBase * 1000)
  var ok, bad = 0
  for it in 0..<3000:
    let text = r.mutate([velfSeed, handvelSeed, csvSeed][r.rand(2)])
    for parse in [parseVelf, parseHandvel, parseVelocityCsv, parseVelocities]:
      try:
        let picks = parse(text)
        inc ok
        for p in picks:
          for k in p.velocity.knots:
            doAssert k.tMs.classify != fcNan and k.vrms > 0, $k
          doAssert p.velocity.vrmsAt(500).classify notin {fcNan, fcInf}
        try: discard initVelocityField(picks, 2, 3000) except ValueError: discard
      except ValueError:
        inc bad
  doAssert ok > 0 and bad > 0

# ------------------------------------------------- fuzzer findings (9.1) --

block extreme_parameters:
  # libFuzzer + UBSan found float → int conversions of values no int holds
  # (undefined in C; one gave a garbage FFT size and an 18 s run).
  # A ramp of 1e-30 Hz: refused at once, not sized.
  doAssertRaises ValueError:
    discard initFilterPlan(float32, bandPass(Corners(f1: 10, f2: 10 + 1e-30,
      f3: 60, f4: 80)), 2, 1000)
  # An NMO taper of 1e300 ms: the ramp reaches column 0 (the mute ends
  # inside the trace).
  var src = newSeq[float32](2000)
  src.addNoise(1, seed = 1)
  var dst = newSeq[float32](2000)
  var vel = newSeq[float64](2000)
  for v in vel.mitems: v = 2000
  let first = src.nmoInto(0, dst, 0, 2, 1500, vel,
    initNmoParams(maxStretch = 0.3, taperMs = 1e300))
  doAssert first == 0
  for v in dst: doAssert v.classify in {fcZero, fcNegZero, fcNormal, fcSubnormal}
  # First-break and correlation windows of 1e300 ms: clamped to the trace.
  let plan = initFirstBreakPlan(float32, FirstBreakParams(windowMs: 1e300,
    ltaMs: 1e300, snrWindowMs: 1e300), 2, 300)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var x = newSeq[float32](300)
  x.addNoise(1, seed = 2)
  discard plan.pick(x, ws)
