## Fuzz target: a pipeline built from the input (ops with in-range, extreme
## or raw float parameters), compiled and applied to samples from the input
## (raw float32 bit patterns: NaN, Inf, subnormals) on rows with random
## status, live lengths and failed rows. `compile` may raise `ValueError`;
## `apply` must not fail, must leave rows that are not live alone, and must
## give the same samples in place as src → dst. Also shifts and NMO on the
## same rows.

import std/math
import parseSEG, processSEG
import fuzzutil

proc corners(r: var ByteReader, nyq: float64): Corners =
  Corners(f1: r.range(0, nyq), f2: r.range(0, nyq), f3: r.range(0, nyq),
    f4: r.range(0, nyq * 1.2))

proc spec(r: var ByteReader, nyq: float64): FilterSpec =
  result = case r.pick(6)
    of 0: bandPass(r.corners(nyq))
    of 1: lowCut(r.range(0, nyq), r.range(0, nyq))
    of 2: highCut(r.range(0, nyq), r.range(0, nyq))
    of 3: butterLowCut(r.range(0, nyq), 6.0 * float64(r.pick(10)))
    of 4: butterHighCut(r.range(0, nyq), 6.0 * float64(r.pick(10)))
    else: butterBandPass(r.range(0, nyq), r.range(0, nyq), 6.0 * float64(r.pick(10)))
  if r.pick(4) == 0: result = result.withNotch(r.range(0, nyq), r.range(0, 10))
  if r.pick(4) == 0: result = result.withZeroDc

proc op(r: var ByteReader, dtMs: float64): Op =
  let nyq = nyquistHz(dtMs)
  case r.pick(16)
  of 0: demeanOp()
  of 1: detrendOp()
  of 2: polarityOp()
  of 3: scaleOp(r.range(-1e3, 1e3))
  of 4: gainDbOp(r.range(-100, 100))
  of 5: clipOp(r.range(-10, 10), r.range(-10, 10))
  of 6, 7:
    let curve = case r.pick(3)
      of 0: timeMute(r.range(-100, 3000))
      of 1: lmoMute(r.range(-100, 8000), r.range(-100, 500), r.range(-100, 100))
      else: tableMute([0.0, r.range(0, 1000), r.range(0, 4000)],
                      [r.range(0, 500), r.range(0, 1000), r.range(0, 2000)])
    if r.pick(2) == 0: muteTopOp(curve, r.range(-1, 200))
    else: muteBottomOp(curve, r.range(-1, 200))
  of 8: filterOp(r.spec(nyq), TrendRemoval(r.pick(3)))
  of 9: iirOp(r.spec(nyq), IirPhase(r.pick(2)), TrendRemoval(r.pick(3)))
  of 10: agcOp(r.range(-10, 2000), AgcMode(r.pick(2)), r.range(0, 10),
      r.range(0, 1e-3))
  of 11: normalizeOp(NormStat(r.pick(3)), NormScope(r.pick(2)), r.range(0, 10),
      r.range(0, 100))
  of 12: gainCurveOp(knotGain([GainKnot(tMs: r.range(0, 1000), dB: r.range(-60, 60)),
      GainKnot(tMs: r.range(0, 4000), dB: r.range(-60, 60))]), r.range(0, 1000),
      if r.pick(2) == 0: NaN else: r.range(0, 4000))
  of 13: gainCurveOp(powerGain(r.range(-1, 4), r.range(0, 3000), r.range(0, 100)))
  of 14: gainCurveOp(exponentialGain(r.range(-60, 60)))
  else: gainCurveOp(GainCurve())

when defined(fuzzStats):
  var applied, ops: int

fuzzTarget:
  var r = ByteReader(data: data, len: len)
  let dtMs = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0][r.pick(6)]
  let rows = 1 + r.pick(12)
  let cols = 1 + r.u16 mod 700
  let first = r.pick(4) * r.u8
  var keys = newSeq[TraceKeys](rows)
  var live = newSeq[int](rows)
  var failed: seq[int]
  for i in 0..<rows:
    keys[i] = TraceKeys(offset: r.range(-4000, 4000), t0Ms: r.range(-100, 100),
      status: TraceStatus(r.pick(4)), ensemble: 1, intervalUs: dtMs * 1000)
    live[i] = first + r.u16 mod (cols + 3)
    if r.pick(8) == 0: failed.add i
  let view = try: rowsOf(keys, live) except ValueError: return
  var flow = initPipeline()
  for k in 0 ..< r.pick(6):
    try: flow.add r.op(dtMs)
    except ValueError: discard # a builder refused its arguments
  var src = zeros[float32](rows, cols)
  for i in 0..<rows:
    for j in 0 ..< max(0, min(cols, live[i] - first)):
      src[i, j] = if r.left > 0: r.rawF32 else: float32(j mod 7)
  let meta = try: initChunkMeta(view, dtMs, firstSample = first, failed = failed)
             except ValueError: return
  let plan = try: flow.compile(float32, dtMs, cols)
             except ValueError: return
  when defined(fuzzStats): inc applied
  var ws = initWorkspace()
  var a = src.clone
  var b = zeros[float32](rows, cols)
  try:
    ws.reserve(plan)
    plan.apply(a, meta, ws)
    plan.apply(src, b, meta, ws)
  except ValueError:
    doAssert false, "reserve or apply raised on a matching block"
  for i in 0..<rows:
    for j in 0..<cols:
      doAssert sameSample(a[i, j], b[i, j]), "in place != src → dst"
      if not meta.isLive(i) or j >= meta.liveSamples(i, cols):
        doAssert sameSample(a[i, j], src[i, j]), "a sample that is not live changed"
  # Shifts and NMO on the processed rows.
  let interp = ShiftInterp(r.pick(2))
  var row = newSeq[float32](cols)
  for j in 0..<cols: row[j] = a[0, j]
  row.shiftSamples(r.range(-1000, 1000), interp)
  var vel = newSeq[float64](cols)
  for j in 0..<cols: vel[j] = r.range(-100, 9000)
  var dst = newSeq[float32](cols)
  let p = try: initNmoParams(interp, r.range(-1, 5), r.range(-1, 100))
          except ValueError: NmoParams()
  let m = row.nmoInto(r.range(-500, 500), dst, r.range(-500, 500), dtMs,
    r.range(-5000, 5000), vel, p)
  doAssert m in 0 .. cols
  when defined(fuzzStats): ops += flow.len

when defined(fuzzStats):
  echo "applied ", applied, ", ops ", ops
