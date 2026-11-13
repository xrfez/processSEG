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
## (two loads and a multiply-add). The relative error of a factor is then
## at most `dt² · max|g''/g| / 8` over its grid interval, where the curve is
## smooth there:
## - knots and exponential curves: `(ΔdB · ln 10 / 20)² / 8`, `ΔdB` the gain
##   change over one sample: 2.7e-6 (2e-5 dB) for 20 dB per second at 2 ms;
## - `powerGain`: `|p(p − 1)| · (dt / t)² / 8`: 4e-4 (0.0035 dB) for `t²`
##   at 50 ms and 2 ms;
## - a knot or `minMs` between two grid points (the slope jumps there) adds
##   a first-order error of up to `|Δs| · dt / 4`, with `s = g'/g` the
##   relative slope on either side (`ln 10 / 20` per dB/ms): 0.1 % for a
##   knot where the slope turns by 20 dB/s, at 2 ms. Knots on the dt grid
##   of rows that start on it have none.
##
## It is 0 on the grid. Times outside the table use the curve itself, a
## `pow` per sample (exact, but 65 times the table's cost, measured): widen
## `tMinMs .. tMaxMs` for rows that start before 0 or run past the record
## length. A plan made for a zoom window's width needs the record's extent
## (`tMaxMs`; `Pipeline.compile` takes `recordSamples` for it). The block
## form that takes a `Workspace` softens the fall: it computes the factors
## off the table once per run of rows with one time axis.
##
## **Rows.** As everywhere: live rows, each to its live length; dead,
## auxiliary and failed rows and the padding are left untouched. Nothing
## allocates or raises once the plan is built (the block forms raise once,
## on a shape mismatch).

import std/math
import parseSEG
import ../core/[rows, types, workspace]

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
  ## The gain at `tMs`, in dB. NaN for a NaN time, or for a default
  ## (unbuilt) curve.
  if tMs.isNaN: return NaN
  if c.kind == gcKnots and c.knots.len == 0: return NaN
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
  ## non-finite limits, `tMaxMs < tMinMs`, a table over 2²⁶ samples, or a
  ## gain in the table that overflows `T` (e.g. 800 dB in float32).
  if not (dtMs.finite and dtMs > 0):
    badCurve("dtMs must be finite and positive, got " & $dtMs)
  if samples < 1: badCurve("a gain curve plan needs samples ≥ 1, got " & $samples)
  if curve.kind == gcKnots and curve.knots.len == 0:
    badCurve("the curve has no knots: build it with knotGain, powerGain or " &
      "exponentialGain")
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
    let t = tMinMs + float64(k) * dtMs
    let g = curve.factorAt(t)
    result.table[k] = T(g)
    if result.table[k].float64.classify == fcInf or g.isNaN:
      # 0 · Inf would turn muted zeros into NaN.
      badCurve("the gain at " & $t & " ms (" & $g & ") overflows " & $T)

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

func workspaceSize*[T](p: GainCurvePlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch of the block form that takes a workspace: one row of
  ## factors.
  when T is float32: WorkspaceSize(f32: p.samples)
  else: WorkspaceSize(f64: p.samples)

# ------------------------------------------------------------------ kernels --

type Buf[T] = ptr UncheckedArray[T]

# The two table kernels are `noinline` (one call per row) so that
# tvectorized can check each loop on its own.

proc mulTable[T](x {.noalias.}, tab {.noalias.}: Buf[T], n: int) {.noinline,
    raises: [].} =
  ## `x[j] *= tab[j]`.
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0..<n:
    x[j] = x[j] * tab[j]
  {.pop.}

proc mulTableLerp[T](x {.noalias.}, tab {.noalias.}: Buf[T], n: int,
    f: T) {.noinline, raises: [].} =
  ## `x[j] *= tab[j] + f·(tab[j + 1] − tab[j])`; reads `tab[0 .. n]`.
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0..<n:
    let a = tab[j]
    x[j] = x[j] * (a + f * (tab[j + 1] - a))
  {.pop.}

proc mulOffTable[T](x {.noalias.}, fac {.noalias.}: Buf[T], n: int) {.noinline,
    raises: [].} =
  ## `x[j] *= fac[j]`, a zero staying zero whatever the factor (no 0 · Inf).
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0..<n:
    let v = x[j]
    x[j] = (if v == 0: v else: v * fac[j])
  {.pop.}

type TableSpan = object
  ## Where a row lies on a plan's table: `x[j0 ..< j1]` multiplies by the
  ## table from entry `s + j0` on (interpolated by `f` unless `onGrid`), the
  ## rest of the row is off the table.
  s, j0, j1: int
  f: float64
  onGrid: bool

func spanOf[T](plan: GainCurvePlan[T], n: int, startMs: float64): TableSpan {.
    raises: [].} =
  ## The table span of a row of `n` samples whose sample 0 is at `startMs`.
  let u = (startMs - plan.tMinMs) / plan.dtMs # table position of x[0]
  if not u.finite or abs(u) > 1e15: # beyond any table
    return TableSpan(j0: n, j1: n)
  let r = round(u)
  result.onGrid = abs(u - r) <= 1e-9 * max(1.0, abs(u))
  result.s = int(if result.onGrid: r else: floor(u))
  result.f = u - float64(result.s)
  # The last j with s + j in the table.
  let last = plan.table.len - (if result.onGrid: 1 else: 2)
  result.j0 = clamp(-result.s, 0, n)
  result.j1 = clamp(last - result.s + 1, result.j0, n)

proc mulOnTable[T](plan: GainCurvePlan[T], x: ptr UncheckedArray[T],
    sp: TableSpan) {.inline, raises: [].} =
  if sp.j1 > sp.j0:
    let tab = cast[ptr UncheckedArray[T]](addr plan.table[sp.s + sp.j0])
    let xs = cast[ptr UncheckedArray[T]](addr x[sp.j0])
    if sp.onGrid: mulTable(xs, tab, sp.j1 - sp.j0)
    else: mulTableLerp(xs, tab, sp.j1 - sp.j0, T(sp.f))

proc applyAt[T](plan: GainCurvePlan[T], x: ptr UncheckedArray[T], n: int,
    startMs: float64) {.raises: [].} =
  ## The curve on `x[0 ..< n]`, whose sample 0 is at `startMs`. Off the
  ## table, a zero stays zero whatever the factor (no 0 · Inf).
  template direct(j: int) =
    if x[j] != 0:
      x[j] = T(float64(x[j]) * plan.curve.factorAt(startMs + float64(j) * plan.dtMs))
  if n == 0 or startMs.isNaN: return
  let sp = plan.spanOf(n, startMs)
  for j in 0..<sp.j0: direct(j)
  plan.mulOnTable(x, sp)
  for j in sp.j1..<n: direct(j)

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
  if not sameInterval(meta.dtMs, plan.dtMs):
    badCurve("the plan is for dtMs = " & $plan.dtMs & ", the rows have " & $meta.dtMs)
  if b.cols > plan.samples:
    badCurve("the block has " & $b.cols & " columns, the plan takes " &
      $plan.samples & " samples")
  for r in 0..<b.len:
    if meta.isLive(r):
      plan.applyAt(b.rowPtr(r), meta.liveSamples(r, b.cols), meta.sampleTimeMs(
          r, 0))

proc applyGainCurve*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    plan: GainCurvePlan[T], ws: var Workspace) =
  ## As the form without a workspace, for blocks that may lie off the
  ## plan's table (a zoom window under a plan made for its width, traces
  ## with a recording delay): the factors off the table are computed once
  ## per run of rows that share a time axis, into `ws`
  ## (`plan.workspaceSize`), and multiplied in, instead of once per sample
  ## of every row. On the table it is the same code; off it, the factor is
  ## rounded to the sample type before the multiply, so results differ from
  ## the other form by a rounding. Raises as that form.
  meta.checkShape(b)
  plan.checkPlan
  if not sameInterval(meta.dtMs, plan.dtMs):
    badCurve("the plan is for dtMs = " & $plan.dtMs & ", the rows have " & $meta.dtMs)
  if b.cols > plan.samples:
    badCurve("the block has " & $b.cols & " columns, the plan takes " &
      $plan.samples & " samples")
  let fac = ws.scratch(T, b.cols)
  var facStart = NaN # the start time `fac` holds the factors of
  for r in 0..<b.len:
    let n = (if meta.isLive(r): meta.liveSamples(r, b.cols) else: 0)
    let startMs = meta.sampleTimeMs(r, 0)
    if n > 0 and not startMs.isNaN:
      let x = b.rowPtr(r)
      let sp = plan.spanOf(n, startMs)
      if sp.j0 > 0 or sp.j1 < n:
        if facStart != startMs: # every column off the table of a full row
          let full = plan.spanOf(b.cols, startMs)
          for j in 0..<full.j0:
            fac[j] = T(plan.curve.factorAt(startMs + float64(j) * plan.dtMs))
          for j in full.j1..<b.cols:
            fac[j] = T(plan.curve.factorAt(startMs + float64(j) * plan.dtMs))
          facStart = startMs
        mulOffTable(x, fac, sp.j0)
        mulOffTable(cast[Buf[T]](addr x[sp.j1]), cast[Buf[T]](addr fac[sp.j1]), n - sp.j1)
      plan.mulOnTable(x, sp)

proc applyGainCurve*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    plan: GainCurvePlan[T]) =
  ## `applyGainCurve` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  rowBlock(samples).applyGainCurve(meta, plan)

proc applyGainCurve*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    plan: GainCurvePlan[T], ws: var Workspace) =
  ## The workspace form on a `[rows, cols]` tensor.
  meta.checkShape(samples)
  rowBlock(samples).applyGainCurve(meta, plan, ws)
