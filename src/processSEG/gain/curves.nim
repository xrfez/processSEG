## Time-variant gain: a gain curve in dB against time (`GainCurve`), applied
## from a per-sample table precomputed in a `GainCurvePlan`.
##
## ```nim
## let curve = knotGain([GainKnot(tMs: 0, dB: 0), GainKnot(tMs: 3000, dB: 18)])
## let plan = initGainCurvePlan(float32, curve, dtMs = 2, samples = 2001)
## samples.applyGainCurve(meta, plan)          # in place, live rows, each at its t0
## ```
##
## **Curves.** The gain at time `t` (ms, the trace's own time axis):
## - `knotGain(knots)`: linear in dB between knots `(tMs, dB)`, flat before the
##   first knot and after the last;
## - `powerGain(power, refMs = 1000, minMs = 1)`: spherical-divergence style
##   `(t / refMs)^power`, unity at `refMs`, with `t` taken as at least `minMs`
##   (so `t ≤ 0` does not give 0 or Inf);
## - `exponentialGain(dbPerS)`: `dbPerS · t / 1000` dB, `10^(dbPerS·t/20000)`.
##
## **Time axis.** Sample `j` of row `r` is at
## `t0Ms(r) + (firstSample + j)·dtMs` (`ChunkMeta.sampleTimeMs`), so a row
## that starts late (a SEG-D delay, a zoom window) gets the gain of its own
## times. A row whose `t0Ms` is NaN is left as it is: processSEG never
## guesses a time.
##
## **The table.** The plan holds the gain factor at `t = tMinMs + k·dtMs`,
## `k = 0 ..< len`, covering `tMinMs .. tMaxMs` (by default `0` to the
## record length, `(samples − 1)·dtMs`). A row whose times fall on the grid
## takes the factors as they are (one vectorized multiply); a row offset by
## a fraction of `dtMs` takes them linearly interpolated between grid points
## (two loads and a multiply-add). The relative error of the factor is then
## at most `(ΔdB · ln 10 / 20)² / 8`, with `ΔdB` the gain change over one
## sample: 2.7e-6 (2e-5 dB) for 20 dB per second at 2 ms, and 0 on the grid
## or where the curve is flat. Times outside the table use the curve itself
## (exact, but scalar and slower): widen `tMinMs .. tMaxMs` for rows that
## start before 0 or for windows past the record length.
##
## **Rows.** As everywhere: live rows, each to its live length; dead,
## auxiliary and failed rows and the padding are left untouched. Nothing
## allocates or raises once the plan is built (the block forms raise once,
## on a shape mismatch).

import std/math
import parseSEG
import ../core/[rows, types]

type
  GainKnot* = object
    ## A point of a `knotGain` curve.
    tMs*: float64 ## Time, ms.
    dB*: float64  ## Gain, dB.

  GainCurveKind* = enum
    ## The shape of a `GainCurve`.
    gcKnots       ## linear in dB between knots
    gcPower       ## `(t / refMs)^power`
    gcExponential ## `dbPerS` dB per second

  GainCurve* = object
    ## A gain against time; build it with `knotGain`, `powerGain` or
    ## `exponentialGain`, which validate it.
    case kind: GainCurveKind
    of gcKnots:
      knots: seq[GainKnot]
    of gcPower:
      power, refMs, minMs: float64
    of gcExponential:
      dbPerS: float64

  GainCurvePlan*[T: SampleType] = object
    ## A gain curve tabulated for traces of `samples` samples at `dtMs`; see
    ## the module documentation. Build it with `initGainCurvePlan`.
    curve: GainCurve
    table: seq[T]
    dtMs: float64
    tMinMs: float64
    samples: int

proc badCurve(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func finite(x: float64): bool {.inline.} =
  x.classify notin {fcNan, fcInf, fcNegInf}

proc knotGain*(knots: openArray[GainKnot]): GainCurve =
  ## A gain linear in dB between `knots`, flat outside them. Raises
  ## `ValueError` unless there is at least one knot, every value is finite,
  ## and the times strictly increase.
  if knots.len == 0: badCurve("a knot gain needs at least one knot")
  for i, k in knots:
    if not (k.tMs.finite and k.dB.finite):
      badCurve("knot " & $i & " is not finite: " & $k)
    if i > 0 and not (k.tMs > knots[i - 1].tMs):
      badCurve("knot times must strictly increase: " & $knots[i - 1].tMs &
        " ms is followed by " & $k.tMs & " ms")
  GainCurve(kind: gcKnots, knots: @knots)

proc powerGain*(power: float64, refMs = 1000.0, minMs = 1.0): GainCurve =
  ## A gain `(max(t, minMs) / refMs)^power`: unity at `refMs`, `power = 2`
  ## for the usual spherical divergence correction. Raises `ValueError` for a
  ## `power` that is not finite, or a `refMs` or `minMs` that is not finite
  ## and positive.
  if not power.finite: badCurve("a power must be finite, got " & $power)
  if not (refMs.finite and refMs > 0):
    badCurve("refMs must be finite and positive, got " & $refMs)
  if not (minMs.finite and minMs > 0):
    badCurve("minMs must be finite and positive, got " & $minMs)
  GainCurve(kind: gcPower, power: power, refMs: refMs, minMs: minMs)

proc exponentialGain*(dbPerS: float64): GainCurve =
  ## A gain of `dbPerS` dB per second of time, 0 dB at `t = 0`. Raises
  ## `ValueError` for a `dbPerS` that is not finite.
  if not dbPerS.finite: badCurve("dbPerS must be finite, got " & $dbPerS)
  GainCurve(kind: gcExponential, dbPerS: dbPerS)

func kind*(c: GainCurve): GainCurveKind {.inline, raises: [].} =
  ## The curve's shape.
  c.kind

func gainDbAt*(c: GainCurve, tMs: float64): float64 {.raises: [].} =
  ## The gain at `tMs`, in dB. NaN for a NaN time.
  if tMs.isNaN: return NaN
  case c.kind
  of gcKnots:
    let k = c.knots
    if tMs <= k[0].tMs: return k[0].dB
    if tMs >= k[^1].tMs: return k[^1].dB
    var lo = 0
    var hi = k.len - 1
    while hi - lo > 1: # k[lo].tMs < tMs < k[hi].tMs
      let mid = (lo + hi) shr 1
      if k[mid].tMs <= tMs: lo = mid else: hi = mid
    let f = (tMs - k[lo].tMs) / (k[hi].tMs - k[lo].tMs)
    k[lo].dB + f * (k[hi].dB - k[lo].dB)
  of gcPower:
    20 * c.power * log10(max(tMs, c.minMs) / c.refMs)
  of gcExponential:
    c.dbPerS * tMs / 1000

func factorAt*(c: GainCurve, tMs: float64): float64 {.raises: [].} =
  ## The amplitude factor at `tMs`, `10^(gainDbAt / 20)` (computed directly
  ## for `powerGain`, so it does not round-trip through dB).
  if tMs.isNaN: return NaN
  case c.kind
  of gcPower: pow(max(tMs, c.minMs) / c.refMs, c.power)
  else: pow(10.0, c.gainDbAt(tMs) / 20)

proc initGainCurvePlan*(T: typedesc[SampleType], curve: GainCurve,
    dtMs: float64, samples: int, tMinMs = 0.0, tMaxMs = NaN): GainCurvePlan[T] =
  ## A plan applying `curve` to traces of up to `samples` samples at `dtMs`,
  ## with a table over `tMinMs .. tMaxMs` (`tMaxMs` NaN: the record length,
  ## `tMinMs + (samples − 1)·dtMs`); see the module documentation. Raises
  ## `ValueError` for a `dtMs` that is not finite and positive, `samples < 1`,
  ## non-finite limits, `tMaxMs < tMinMs`, or a table over 2²⁶ samples.
  if not (dtMs.finite and dtMs > 0):
    badCurve("dtMs must be finite and positive, got " & $dtMs)
  if samples < 1: badCurve("a gain curve plan needs samples ≥ 1, got " & $samples)
  let hiMs = (if tMaxMs.isNaN: tMinMs + float64(samples - 1) *
      dtMs else: tMaxMs)
  if not (tMinMs.finite and hiMs.finite):
    badCurve("the table limits must be finite, got " & $tMinMs & " .. " & $hiMs)
  if hiMs < tMinMs:
    badCurve("tMaxMs " & $hiMs & " is before tMinMs " & $tMinMs)
  let span = floor((hiMs - tMinMs) / dtMs + 1e-9)
  if span >= float64(1 shl 26):
    badCurve("a gain table of " & $span & " samples is too long")
  let len = int(span) + 1
  result = GainCurvePlan[T](curve: curve, dtMs: dtMs, tMinMs: tMinMs,
    samples: samples)
  result.table = newSeq[T](len)
  for k in 0..<len:
    result.table[k] = T(curve.factorAt(tMinMs + float64(k) * dtMs))

func samples*[T](p: GainCurvePlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  p.samples

func dtMs*[T](p: GainCurvePlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval the plan was built for.
  p.dtMs

func curve*[T](p: GainCurvePlan[T]): lent GainCurve {.inline, raises: [].} =
  ## The curve the plan applies.
  p.curve

func tableLen*[T](p: GainCurvePlan[T]): int {.inline, raises: [].} =
  ## The number of tabulated times.
  p.table.len

func tableMs*[T](p: GainCurvePlan[T]): Slice[float64] {.raises: [].} =
  ## The times the table covers.
  p.tMinMs .. p.tMinMs + float64(p.table.len - 1) * p.dtMs

# ------------------------------------------------------------------ kernels --

proc mulTable[T](x {.noalias.}: ptr UncheckedArray[T],
    tab {.noalias.}: ptr UncheckedArray[T], n: int) {.inline, raises: [].} =
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0..<n:
    x[j] = x[j] * tab[j]
  {.pop.}

proc mulTableLerp[T](x {.noalias.}: ptr UncheckedArray[T],
    tab {.noalias.}: ptr UncheckedArray[T],

n: int, f: T) {.inline, raises: [].} =
  ## `x[j] *= tab[j] + f·(tab[j + 1] − tab[j])`; reads `tab[0 .. n]`.
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0..<n:
    let a = tab[j]
    x[j] = x[j] * (a + f * (tab[j + 1] - a))
  {.pop.}

proc applyAt[T](plan: GainCurvePlan[T], x: ptr UncheckedArray[T], n: int,
    startMs: float64) {.raises: [].} =
  ## The curve on `x[0 ..< n]`, whose sample 0 is at `startMs`.
  if n == 0 or startMs.isNaN: return
  let u = (startMs - plan.tMinMs) / plan.dtMs # table position of x[0]
  if not u.finite or abs(u) > 1e15: # beyond any table: all direct
    for j in 0..<n:
      x[j] = T(float64(x[j]) * plan.curve.factorAt(startMs + float64(j) * plan.dtMs))
    return
  let r = round(u)
  let onGrid = abs(u - r) <= 1e-9 * max(1.0, abs(u))
  let s = int(if onGrid: r else: floor(u))
  let f = u - float64(s)
  let last = plan.table.len - (if onGrid: 1 else: 2) # the last j with s + j in the table
  let j0 = clamp(-s, 0, n)
  let j1 = clamp(last - s + 1, j0, n)
  for j in 0..<j0:
    x[j] = T(float64(x[j]) * plan.curve.factorAt(startMs + float64(j) * plan.dtMs))
  if j1 > j0:
    let tab = cast[ptr UncheckedArray[T]](unsafeAddr plan.table[s + j0])
    let xs = cast[ptr UncheckedArray[T]](addr x[j0])
    if onGrid: mulTable(xs, tab, j1 - j0)
    else: mulTableLerp(xs, tab, j1 - j0, T(f))
  for j in j1..<n:
    x[j] = T(float64(x[j]) * plan.curve.factorAt(startMs + float64(j) * plan.dtMs))

proc checkPlan[T](plan: GainCurvePlan[T]) {.inline.} =
  doAssert plan.table.len > 0, "a GainCurvePlan must come from initGainCurvePlan"

proc applyGainCurve*[T: SampleType](x: var openArray[T], plan: GainCurvePlan[T],
    startMs: float64) {.raises: [].} =
  ## The curve on one trace, `x` (its live samples, at most `plan.samples`;
  ## an `AssertionDefect` otherwise), whose sample 0 is at `startMs`, in
  ## place. A NaN `startMs` leaves `x` as it is.
  plan.checkPlan
  doAssert x.len <= plan.samples, "a trace longer than the plan's samples"
  if x.len == 0: return
  plan.applyAt(cast[ptr UncheckedArray[T]](addr x[0]), x.len, startMs)

proc applyGainCurve*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    plan: GainCurvePlan[T]) =
  ## The curve on the live rows of `b` in place, each to its live length and
  ## at its own times (`meta.sampleTimeMs(row, 0)` onwards). Raises
  ## `ValueError`, before any sample changes, when `b` does not match `meta`
  ## or has more columns than `plan.samples`.
  meta.checkShape(b)
  plan.checkPlan
  if b.cols > plan.samples:
    badCurve("the block has " & $b.cols & " columns, the plan takes " &
      $plan.samples & " samples")
  for r in 0..<b.len:
    if meta.isLive(r):
      plan.applyAt(b.rowPtr(r), meta.liveSamples(r, b.cols), meta.sampleTimeMs(
          r, 0))

proc applyGainCurve*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    plan: GainCurvePlan[T]) =
  ## `applyGainCurve` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  rowBlock(samples).applyGainCurve(meta, plan)
