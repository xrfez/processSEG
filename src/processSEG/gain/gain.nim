## Amplitude gain: a fixed gain in dB, normalization per trace or per panel,
## and automatic gain control (AGC).
##
## ```nim
## samples.scaleDb(meta, 6)                              # × 2, live rows only
##
## let norm = initNormalizePlan(normPercentile, samples = 2001, scope = perPanel)
## let auto = initAgcPlan(windowMs = 500, dtMs = 2, samples = 2001)
## var ws = initWorkspace()
## ws.reserve(max(norm.workspaceSize, auto.workspaceSize))
## norm.normalize(samples, meta, ws)
## auto.agc(samples, meta, ws)                           # in place
## auto.agc(samples, meta, gains, ws)                    # and keep the gain
## samples.removeGain(meta, gains)                       # undo it later
## ```
##
## **Rows.** As everywhere: the block forms process the live rows, each up to
## its live length (`ChunkMeta.liveSamples`); dead, auxiliary and failed rows
## and the padding past a trace's end are left untouched. A NaN or Inf sample
## stays in its own row: normalization and AGC leave such a row as it is.
##
## **Normalization** scales a trace (`perTrace`) or every live trace of the
## block by one factor (`perPanel`) so that its maximum `|x|`, RMS or a
## percentile of `|x|` (`NormStat`) becomes `target`. A trace or panel whose
## statistic is 0 (all zero) or not finite, or whose factor would not fit
## the sample type (Inf in float32), is left as it is. A panel's
## statistic leaves out rows with a NaN or Inf (`panelStats`), and its factor
## is then applied to every live row. The panel percentile comes from
## `panelStats`'s histogram (within 0.4 %), a trace percentile is exact.
##
## **AGC.** Each sample is divided by the RMS (`agcRms`) or the mean `|x|`
## (`agcMeanAbs`) of a window of `2h + 1` samples centred on it,
## `h = round(windowMs / (2·dtMs))`, and multiplied by `target`:
##
##     g[i] = target / (level(i) + ε),   y[i] = g[i] · x[i]
##
## - near the ends of the trace's live part the window is cut, and the level
##   is over the samples it really holds (no zeros counted in);
## - `ε = epsilon · rms(trace)`, so a muted (zero) stretch stays zero and a
##   trace's quiet part is not blown up past `1/epsilon` of its RMS. A trace
##   that is all zero is left as it is (gain 1);
## - **precision:** the window sums are built without subtraction. The trace
##   is cut into blocks of `2h + 1` samples; each window spans at most two
##   blocks, so its sum is a suffix sum of one block plus a prefix sum of the
##   next, both in float64 over non-negative terms. Each window sum is then
##   accurate to a few ulps whatever the dynamic range (a running sum that
##   adds and subtracts keeps the rounding error of the loudest window it
##   passed, which swamps a quiet window 120 dB down). The cost is O(n),
##   independent of the window length (about 1.6 ns a sample on one thread,
##   bench/README "Gain");
## - the gain step, `g` from the window mean, runs in the sample type (8
##   float32 lanes for float32 data) on means scaled by the trace's RMS, so
##   it is near 1 and cannot overflow float32 whatever the data's scale. A
##   trace whose largest gain, `target / (rms·epsilon)`, does not fit the
##   sample type (float32 data near 1e-30 with the defaults) is left as it
##   is, with gain 1, rather than turning its silent samples into NaN, and
##   so is a trace with a NaN or Inf sample (its RMS is not finite);
## - `requiredMargin` is `h`: a zoom window decoded with `h` extra samples
##   on each side matches the whole trace's AGC over the window, except that
##   ε comes from the decoded window's RMS.
##
## **Gain functions.** The `gains` forms of `agc` also store `g` (1 for every
## sample left untouched), so `removeGain` can undo the AGC later, for
## example before a stack or an export.

import std/math
import parseSEG
import ../core/[basic, rows, types, workspace]
import stats

type
  NormStat* = enum
    ## The statistic `normalize` brings to its target.
    normMax        ## maximum |x|
    normRms        ## RMS
    normPercentile ## a percentile of |x| (a robust maximum)

  NormScope* = enum
    ## Whether each trace gets its own factor, or the whole block one.
    perTrace
    perPanel

  NormalizePlan* = object
    ## A normalization; see the module documentation. Build it with
    ## `initNormalizePlan`.
    stat: NormStat
    scope: NormScope
    target: float64
    percentile: float64
    samples: int

  AgcMode* = enum
    ## The window level that AGC divides by.
    agcRms     ## the RMS of the window
    agcMeanAbs ## the mean |x| of the window

  AgcPlan* = object
    ## An AGC for traces of up to `samples` samples; see the module
    ## documentation. Build it with `initAgcPlan`.
    mode: AgcMode
    half: int
    samples: int
    dtMs: float64
    target: float64
    epsilon: float64
    inv: seq[float64] # 1/c for the window counts c = 0 .. 2h + 1 (inv[0] unused)
    rev: seq[float64] # 1/(2h + 1 − m), m = 0 .. 2h: the tail's counts, ascending in i

proc badArg(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func finitePositive(x: float64): bool {.inline.} =
  x.classify in {fcNormal, fcSubnormal} and x > 0

func fitsIn(T: typedesc, x: float64): bool {.inline.} =
  ## `x` is finite and positive in the sample type (a float64 factor of 1e40
  ## is Inf in float32, and would make zeros NaN).
  x.finitePositive and T(x).float64.finitePositive

# ----------------------------------------------------------------- fixed dB --

func dbToFactor*(dB: float64): float64 {.inline, raises: [].} =
  ## The amplitude factor of a gain of `dB`: `10^(dB / 20)`.
  pow(10.0, dB / 20)

proc scaleDb*[T: SampleType](x: var openArray[T], dB: float64) {.raises: [].} =
  ## Multiplies every sample by `10^(dB / 20)` (6.02 dB doubles the
  ## amplitude). The caller checked `dB` to be finite (the block forms raise
  ## otherwise).
  scale(x, dbToFactor(dB))

proc checkDb(T: typedesc, dB: float64) =
  if dB.classify in {fcNan, fcInf, fcNegInf}:
    badArg("a gain in dB must be finite, got " & $dB)
  if T(dbToFactor(dB)).float64.classify == fcInf: # 0 · Inf: zeros would become NaN
    badArg("a gain of " & $dB & " dB overflows " & $T)

proc scaleDb*[T: SampleType](b: RowBlock[T], meta: ChunkMeta, dB: float64) =
  ## `scaleDb` on the live samples of each live row of `b`. Raises
  ## `ValueError` for a `dB` that is not finite or whose factor overflows `T`
  ## (800 dB in float32), or a block that does not match `meta`.
  checkDb(T, dB)
  scale(b, meta, dbToFactor(dB))

proc scaleDb*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta, dB: float64) =
  ## `scaleDb` on the live part of a `[rows, cols]` tensor (C-contiguous;
  ## raises `ValueError` otherwise, or as the block form).
  checkDb(T, dB)
  scale(samples, meta, dbToFactor(dB))

# ------------------------------------------------------------ normalization --

proc initNormalizePlan*(stat: NormStat, samples: int, scope = perTrace,
    target = 1.0, percentile = 99.0): NormalizePlan =
  ## A normalization bringing `stat` of each trace (`perTrace`) or of the
  ## block (`perPanel`) to `target`, for traces of up to `samples` samples.
  ## `percentile` (0 .. 100) is used by `normPercentile` only. Raises
  ## `ValueError` for `samples < 1`, a `target` that is not finite and
  ## positive, or a percentile outside `0 .. 100`.
  if samples < 1: badArg("a normalize plan needs samples ≥ 1, got " & $samples)
  if not target.finitePositive:
    badArg("a normalize target must be finite and positive, got " & $target)
  if not (percentile >= 0 and percentile <= 100):
    badArg("a percentile must be in 0 .. 100, got " & $percentile)
  NormalizePlan(stat: stat, scope: scope, target: target,
    percentile: percentile,
    samples: samples)

func stat*(p: NormalizePlan): NormStat {.inline, raises: [].} =
  ## The statistic brought to the target.
  p.stat

func scope*(p: NormalizePlan): NormScope {.inline, raises: [].} =
  ## Per trace or per panel.
  p.scope

func target*(p: NormalizePlan): float64 {.inline, raises: [].} =
  ## The value the statistic is brought to.
  p.target

func percentile*(p: NormalizePlan): float64 {.inline, raises: [].} =
  ## The percentile `normPercentile` uses.
  p.percentile

func samples*(p: NormalizePlan): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  p.samples

func workspaceSize*(p: NormalizePlan): WorkspaceSize {.raises: [].} =
  ## The scratch of `normalize` (none unless it uses a percentile).
  if p.stat != normPercentile: WorkspaceSize()
  elif p.scope == perTrace: percentileWorkspace(p.samples)
  else: panelStatsWorkspace

proc checkPlan(p: NormalizePlan) {.inline.} =
  doAssert p.samples > 0, "a NormalizePlan must come from initNormalizePlan"

proc traceStatOf[T](x: openArray[T], p: NormalizePlan,
    ws: var Workspace): float64 {.
    raises: [].} =
  case p.stat
  of normMax: traceStats(x).maxAbs
  of normRms: traceStats(x).rms
  of normPercentile:
    try: percentileAbs(x, p.percentile, ws)
    except ValueError: NaN # the plan checked the percentile

proc scaleToTarget[T](x: var openArray[T], level, target: float64) {.inline.} =
  if level.finitePositive and fitsIn(T, target / level):
    scale(x, target / level)

proc normalize*[T: SampleType](plan: NormalizePlan, x: var openArray[T],
    ws: var Workspace) {.raises: [].} =
  ## Normalizes one trace, `x` (its live samples, at most `plan.samples`; an
  ## `AssertionDefect` otherwise), whatever the plan's scope. `ws` must hold
  ## `plan.workspaceSize`. An all-zero or non-finite trace is left as it is.
  plan.checkPlan
  doAssert x.len <= plan.samples, "a trace longer than the plan's samples"
  scaleToTarget(x, traceStatOf(x, plan, ws), plan.target)

proc panelFactor*[T: SampleType](plan: NormalizePlan, b: RowBlock[T], meta: ChunkMeta,
    ws: var Workspace): float64 =
  ## The one factor a `perPanel` normalization of `b` multiplies its live
  ## rows by (whatever `plan.scope` says), or NaN when it would leave the
  ## block alone (a statistic of 0 or not finite, or a factor that does not
  ## fit `T`). The pipeline uses it to fuse the factor into later steps; a
  ## viewer can use it as a display-gain uniform. `ws` must hold
  ## `panelStatsWorkspace` for `normPercentile`. Raises `ValueError` when `b`
  ## does not match `meta`.
  meta.checkShape(b)
  plan.checkPlan
  let clipP = (if plan.stat == normPercentile: plan.percentile else: NaN)
  let s = panelStats(b, meta, ws, clipP)
  let level =
    case plan.stat
    of normMax: s.maxAbs
    of normRms: s.rms
    of normPercentile: s.clip
  if level.finitePositive and fitsIn(T, plan.target / level): plan.target / level
  else: NaN

proc normalize*[T: SampleType](plan: NormalizePlan, b: RowBlock[T], meta: ChunkMeta,
    ws: var Workspace) =
  ## Normalizes the live rows of `b` in place, per trace or with one factor
  ## for the block (module documentation). `ws` must hold
  ## `plan.workspaceSize`. Raises `ValueError`, before any sample changes,
  ## when `b` does not match `meta` or has more columns than `plan.samples`.
  meta.checkShape(b)
  plan.checkPlan
  if b.cols > plan.samples:
    badArg("the block has " & $b.cols & " columns, the plan takes " &
      $plan.samples & " samples")
  case plan.scope
  of perTrace:
    for r in 0..<b.len:
      if meta.isLive(r):
        normalize(plan, b.row(r, meta.liveSamples(r, b.cols)), ws)
  of perPanel:
    let f = plan.panelFactor(b, meta, ws)
    if not f.isNaN:
      scale(b, meta, f)

proc normalize*[T: SampleType](plan: NormalizePlan, samples: var Tensor[T],
    meta: ChunkMeta, ws: var Workspace) =
  ## `normalize` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  plan.normalize(rowBlock(samples), meta, ws)

# ---------------------------------------------------------------------- AGC --

proc initAgcPlan*(windowMs, dtMs: float64, samples: int, mode = agcRms,
    target = 1.0, epsilon = 1e-9): AgcPlan =
  ## An AGC over a centred window of about `windowMs` (`2h + 1` samples,
  ## `h = round(windowMs / (2·dtMs))`) for traces of up to `samples` samples
  ## at `dtMs`, bringing the window level (`mode`) to `target`, with
  ## `ε = epsilon · rms(trace)`. Raises `ValueError` for a `dtMs`, `target`
  ## or `epsilon` that is not finite and positive, `samples < 1`, or a window
  ## shorter than 3 samples (`windowMs < 3·dtMs`, about).
  if not dtMs.finitePositive:
    badArg("dtMs must be finite and positive, got " & $dtMs)
  if samples < 1 or samples > int32.high:
    badArg("an AGC plan needs samples in 1 .. int32.high, got " & $samples)
  if not windowMs.finitePositive:
    badArg("an AGC window must be finite and positive, got " & $windowMs & " ms")
  let h = round(windowMs / (2 * dtMs))
  if h < 1:
    badArg("an AGC window of " & $windowMs & " ms is under 3 samples at " &
      $dtMs & " ms")
  if h > float64(1 shl 26): # its two tables hold 2h + 2 values each
    badArg("an AGC window of " & $windowMs & " ms is too long (" & $h &
      " samples each side; at most 2²⁶)")
  if not target.finitePositive:
    badArg("an AGC target must be finite and positive, got " & $target)
  if not epsilon.finitePositive:
    badArg("an AGC epsilon must be finite and positive, got " & $epsilon)
  result = AgcPlan(mode: mode, half: int(h), samples: samples, dtMs: dtMs,
    target: target, epsilon: epsilon)
  result.inv = newSeq[float64](2 * int(h) + 2)
  for c in 1..<result.inv.len: result.inv[c] = 1 / float64(c)
  result.rev = newSeq[float64](2 * int(h) + 1)
  for m in 0..<result.rev.len: result.rev[m] = 1 / float64(2 * int(h) + 1 - m)

func requiredMargin*(p: AgcPlan): int {.inline, raises: [].} =
  ## The samples a zoom window needs on each side: half the window, `h`.
  p.half

func windowSamples*(p: AgcPlan): int {.inline, raises: [].} =
  ## The window length, `2h + 1` samples.
  2 * p.half + 1

func windowMs*(p: AgcPlan): float64 {.inline, raises: [].} =
  ## The window length in ms, `(2h + 1)·dtMs`.
  float64(p.windowSamples) * p.dtMs

func mode*(p: AgcPlan): AgcMode {.inline, raises: [].} =
  ## The window level divided by.
  p.mode

func samples*(p: AgcPlan): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  p.samples

func target*(p: AgcPlan): float64 {.inline, raises: [].} =
  ## The level each window is brought to.
  p.target

func epsilon*(p: AgcPlan): float64 {.inline, raises: [].} =
  ## ε relative to the trace RMS.
  p.epsilon

func dtMs*(p: AgcPlan): float64 {.inline, raises: [].} =
  ## The sample interval the plan was built for.
  p.dtMs

func workspaceSize*(p: AgcPlan): WorkspaceSize {.raises: [].} =
  ## The scratch of `agc`: two float64 sums per sample.
  WorkspaceSize(f64: 2 * p.samples + 1)

proc checkPlan(p: AgcPlan) {.inline.} =
  doAssert p.half > 0 and p.inv.len > 0, "an AgcPlan must come from initAgcPlan"

type Buf = ptr UncheckedArray[float64]

template termOf(mode: static AgcMode, v: float64): float64 =
  (when mode == agcRms: v * v else: abs(v))

template levelOf(mode: static AgcMode, mean: untyped): untyped =
  (when mode == agcRms: sqrt(mean) else: mean)

proc blockSums[T](x: ptr UncheckedArray[T], n, w: int, mode: static AgcMode,
    s, p: Buf) {.raises: [].} =
  ## Per block of `w` samples: `s[i]` the sum of the terms from `i` to the
  ## block's end (or `n − 1`), `p[e]` the sum from the block's start to
  ## `e − 1` (0 at a block start), for `e in 0 .. n`.
  ##
  ## Each sum is a serial add chain, and this loop is over half of an AGC's
  ## time. A block's prefix and suffix are two independent chains; two
  ## blocks at a time make four, which a core runs side by side: 8.8 → 6.9
  ## cycles per sample at 500 ms and 2 ms, the same sums in the same order
  ## (three and four blocks at a time were slower).
  {.push boundChecks: off, overflowChecks: off.}
  var bs = 0
  while bs + 2 * w <= n: # two whole blocks
    let mid = bs + w
    var fwd0, bwd0, fwd1, bwd1 = 0.0
    p[bs] = 0
    for k in 0..<w:
      fwd0 += termOf(mode, float64(x[bs + k]))
      p[bs + k + 1] = fwd0
      bwd0 += termOf(mode, float64(x[mid - 1 - k]))
      s[mid - 1 - k] = bwd0
      fwd1 += termOf(mode, float64(x[mid + k]))
      p[mid + k + 1] = fwd1
      bwd1 += termOf(mode, float64(x[mid + w - 1 - k]))
      s[mid + w - 1 - k] = bwd1
    p[mid] = 0 # the second block's start: the loop left the first one's sum there
    bs += 2 * w
  while bs < n:
    let be = min(bs + w, n)
    var fwd, bwd = 0.0
    p[bs] = 0
    for k in 0..<be - bs:
      fwd += termOf(mode, float64(x[bs + k]))
      p[bs + k + 1] = fwd
      bwd += termOf(mode, float64(x[be - 1 - k]))
      s[be - 1 - k] = bwd
    bs = be
  if n mod w == 0: p[n] = 0 # `n` starts a block
  {.pop.}

template agcRange(T, G: typedesc, x, g: ptr UncheckedArray[T], first, last: int,
    mode: static AgcMode, k, eps: G, keep: static bool, meanOf: untyped) =
  ## `y[i] = g[i]·x[i]` for `i in first ..< last`, `meanOf` the window mean
  ## of the terms (an expression of `i`) scaled by `1/rms²` (rms mode) or
  ## `1/rms` (mean-abs mode), so it is near 1 and safe in `G`.
  {.push boundChecks: off, overflowChecks: off.}
  for i {.inject.} in first..<last:
    let gi = k / (levelOf(mode, G(meanOf)) + eps)
    when keep: g[i] = T(gi)
    x[i] = T(x[i] * T(gi))
  {.pop.}

proc agcImpl[T](x: ptr UncheckedArray[T], n: int, plan: AgcPlan,
    ws: var Workspace, g: ptr UncheckedArray[T], mode: static AgcMode,
        keep: static bool) {.raises: [].} =
  if n == 0: return
  doAssert n <= int32.high, "a trace longer than int32.high samples"
  let rms = traceStats(x.toOpenArray(0, n - 1)).rms
  if rms == 0: # all zero: nothing to scale
    when keep:
      for i in 0..<n: g[i] = T(1)
    return
  let h = plan.half
  let w = 2 * h + 1
  let buf = ws.scratch(float64, 2 * n + 1)
  let s = buf
  let p = cast[Buf](addr buf[n])
  blockSums(x, n, w, mode, s, p)
  # The gain step runs in the sample type (float32: 8 lanes, not 4), on
  # means scaled to the trace level: g = (target/rms) / (level/rms + epsilon).
  # The largest gain, target / (rms·epsilon), at a silent sample, must fit
  # the sample type, or 0·Inf would make the silence NaN: otherwise the row
  # is left as it is.
  if not fitsIn(T, plan.target / rms / plan.epsilon):
    when keep:
      for i in 0..<n: g[i] = T(1)
    return
  type G = T # the type of the gain step (see the module documentation)
  let unit = (when mode == agcRms: 1 / (rms * rms) else: 1 / rms)
  let k = G(plan.target / rms)
  let eps = G(plan.epsilon)
  # The ranges of i (module documentation), each with affine, ascending
  # indices so that its loop vectorizes:
  # - the head, i < h: the window is 0 ..< e, e = min(n, i + h + 1) ≤ 2h < w,
  #   inside block 0, so its sum is p[e] (e = n only for a trace under 2h);
  # - whole windows, h ≤ i < n − h: i − h ..< i + h + 1 spans a block end,
  #   so s[i − h] + p[i + h + 1] (p is 0 at a block start);
  # - the tail, i ≥ max(h, n − h): the window is lo ..< n, lo = i − h, with
  #   n − lo = 2h + 1 − m samples, m = i + h + 1 − n: s[lo], plus p[n] while
  #   lo lies before the last block.
  let first = min(h, n)
  let last = max(first, n - h)
  let headCut = clamp(n - h, 0, first)
  let lastBlock = (n - 1) div w * w
  let tailCut = clamp(lastBlock + h, last, n)
  let pn = p[n]
  let inv = cast[Buf](addr plan.inv[0])
  let rev = cast[Buf](addr plan.rev[0])
  let unitW = unit / float64(w)
  # The whole trace, when it is under 2h + 1 samples (then n ≤ 2h).
  let headWhole = (if headCut < first: pn * unit * inv[n] else: 0.0)
  agcRange(T, G, x, g, 0, headCut, mode, k, eps, keep,
    p[i + h + 1] * (unit * inv[i + h + 1]))
  agcRange(T, G, x, g, headCut, first, mode, k, eps, keep, headWhole)
  agcRange(T, G, x, g, first, last, mode, k, eps, keep,
    (s[i - h] + p[i + h + 1]) * unitW)
  agcRange(T, G, x, g, last, tailCut, mode, k, eps, keep,
    (s[i - h] + pn) * (unit * rev[i + h + 1 - n]))
  agcRange(T, G, x, g, tailCut, n, mode, k, eps, keep,
    s[i - h] * (unit * rev[i + h + 1 - n]))

proc agcDispatch[T](x: ptr UncheckedArray[T], n: int, plan: AgcPlan,
    ws: var Workspace, g: ptr UncheckedArray[T], keep: static bool) {.raises: [].} =
  case plan.mode
  of agcRms: agcImpl(x, n, plan, ws, g, agcRms, keep)
  of agcMeanAbs: agcImpl(x, n, plan, ws, g, agcMeanAbs, keep)

proc agc*[T: SampleType](plan: AgcPlan, x: var openArray[T],
    ws: var Workspace) {.raises: [].} =
  ## AGC on one trace, `x` (its live samples, at most `plan.samples`; an
  ## `AssertionDefect` otherwise), in place. `ws` must hold
  ## `plan.workspaceSize`. An all-zero trace is left as it is.
  plan.checkPlan
  doAssert x.len <= plan.samples, "a trace longer than the plan's samples"
  if x.len == 0: return
  agcDispatch(cast[ptr UncheckedArray[T]](addr x[0]), x.len, plan, ws, nil, keep = false)

proc agc*[T: SampleType](plan: AgcPlan, x: var openArray[T], gain: var openArray[T],
    ws: var Workspace) {.raises: [].} =
  ## `agc` on one trace, also storing the gain of each sample in `gain`
  ## (`gain.len == x.len`, an `AssertionDefect` otherwise; 1 everywhere for
  ## an all-zero trace).
  plan.checkPlan
  doAssert x.len <= plan.samples, "a trace longer than the plan's samples"
  doAssert gain.len == x.len, "gain needs one value per sample"
  if x.len == 0: return
  agcDispatch(cast[ptr UncheckedArray[T]](addr x[0]), x.len, plan, ws,
    cast[ptr UncheckedArray[T]](addr gain[0]), keep = true)

proc checkBlock[T](b: RowBlock[T], meta: ChunkMeta, plan: AgcPlan) =
  meta.checkShape(b)
  plan.checkPlan
  if not sameInterval(meta.dtMs, plan.dtMs):
    badArg("the plan is for dtMs = " & $plan.dtMs & ", the rows have " & $meta.dtMs)
  if b.cols > plan.samples:
    badArg("the block has " & $b.cols & " columns, the plan takes " &
      $plan.samples & " samples")

proc agc*[T: SampleType](plan: AgcPlan, b: RowBlock[T], meta: ChunkMeta,
    ws: var Workspace) =
  ## AGC on the live rows of `b` in place, each to its live length. `ws` must
  ## hold `plan.workspaceSize`. Raises `ValueError`, before any sample
  ## changes, when `b` does not match `meta`, has more columns than
  ## `plan.samples`, or the rows' `dtMs` is not the plan's.
  checkBlock(b, meta, plan)
  for r in 0..<b.len:
    if meta.isLive(r):
      agcDispatch(b.rowPtr(r), meta.liveSamples(r, b.cols), plan, ws, nil, keep = false)

proc agc*[T: SampleType](plan: AgcPlan, b: RowBlock[T], meta: ChunkMeta,
    gains: RowBlock[T], ws: var Workspace) =
  ## `agc` on the live rows of `b`, also storing each sample's gain in
  ## `gains` (a block of the same shape): 1 for every sample left untouched
  ## (rows that are not live, padding past a trace's end, all-zero traces).
  ## Raises `ValueError` as the other form, and when `gains` differs in shape.
  checkBlock(b, meta, plan)
  if gains.len != b.len or gains.cols != b.cols:
    badArg("gains is " & $gains.len & " × " & $gains.cols & ", the block " &
      $b.len & " × " & $b.cols)
  for r in 0..<b.len:
    let g = gains.rowPtr(r)
    let n = if meta.isLive(r): meta.liveSamples(r, b.cols) else: 0
    if n > 0:
      agcDispatch(b.rowPtr(r), n, plan, ws, g, keep = true)
    for j in n..<b.cols: g[j] = T(1)

proc agc*[T: SampleType](plan: AgcPlan, samples: var Tensor[T], meta: ChunkMeta,
    ws: var Workspace) =
  ## `agc` on a `[rows, cols]` tensor (C-contiguous; raises `ValueError`
  ## otherwise).
  meta.checkShape(samples)
  plan.agc(rowBlock(samples), meta, ws)

proc agc*[T: SampleType](plan: AgcPlan, samples: var Tensor[T], meta: ChunkMeta,
    gains: var Tensor[T], ws: var Workspace) =
  ## `agc` on a tensor, keeping the gain in `gains`, a tensor of the same
  ## shape (C-contiguous; raises `ValueError` otherwise).
  meta.checkShape(samples)
  plan.agc(rowBlock(samples), meta, rowBlock(gains), ws)

# ------------------------------------------------------------ gain removal --

proc removeGain*[T: SampleType](x: var openArray[T], gain: openArray[
    T]) {.raises: [].} =
  ## Divides each sample by its gain (`gain.len == x.len`, an
  ## `AssertionDefect` otherwise), undoing `agc` with its kept gain.
  doAssert gain.len == x.len, "gain needs one value per sample"
  {.push boundChecks: off.}
  for i in 0..<x.len:
    x[i] = x[i] / gain[i]
  {.pop.}

proc removeGain*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    gains: RowBlock[T]) =
  ## `removeGain` on the live samples of each live row of `b`, with `gains`
  ## from `agc` (same shape). Raises `ValueError` when the shapes differ or
  ## `b` does not match `meta`.
  meta.checkShape(b)
  if gains.len != b.len or gains.cols != b.cols:
    badArg("gains is " & $gains.len & " × " & $gains.cols & ", the block " &
      $b.len & " × " & $b.cols)
  for r in 0..<b.len:
    if meta.isLive(r):
      let n = meta.liveSamples(r, b.cols)
      removeGain(b.row(r, n), gains.row(r, n))

proc removeGain*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    gains: Tensor[T]) =
  ## `removeGain` on a tensor, with `gains` of the same shape (C-contiguous;
  ## raises `ValueError` otherwise).
  meta.checkShape(samples)
  rowBlock(samples).removeGain(meta, rowBlock(gains))
