## Processing flows: `Op`, `Pipeline`, `compile` and `PipelinePlan`
## (decisions D8 and D11).
##
## A `Pipeline` is an ordered list of operations held as plain data, which a
## viewer edits when the user moves a slider. `compile` validates every op
## for one sample interval and trace length, and precomputes what each needs
## (filter responses and FFT plans, AGC tables, gain curves), giving an
## immutable `PipelinePlan`. Applying a plan allocates nothing once a
## `Workspace` holds `plan.workspaceSize`, and never raises for data.
##
## ```nim
## var flow = initPipeline([demeanOp(),
##                          bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
##                          agcOp(windowMs = 500)])
## let plan = flow.compile(float32, dtMs = 2, samples = 2001) # ValueError if invalid
## var ws = initWorkspace()
## ws.reserve(plan)                       # grow-only; a no-op when it fits
## plan.apply(raw, processed, meta, ws)   # src → dst (raw kept), copy fused
## plan.apply(samples, meta, ws)          # in place
## let m = plan.requiredMargin            # samples each side, for zoom windows
##
## flow[2].agc.windowMs = 300             # a slider moved: edit, recompile
## let plan2 = flow.compile(float32, dtMs = 2, samples = 2001)
## assert plan2.generation > plan.generation
## ```
##
## **Ops and builders.** Each op kind has a builder named after it with an
## `Op` suffix (`demeanOp()`, while `demean(x)` is the kernel). An op's
## parameters are public fields, grouped per kind (`op.agc.windowMs`,
## `op.filter.spec`), so a flow can be edited in place; nothing is checked
## until `compile` (the specs and curves an op holds, such as `bandPass` or
## `lmoMute`, check themselves when they are built).
##
## **Stale buffers.** A processed buffer is stale when the plan changed (its
## `generation`) **or the keys did**: mutes read offsets, gain curves read
## `t0Ms`, and every op reads the trace status. Tag buffers with both the
## plan's generation and the file's `keysRevision`.
##
## **Rows.** As everywhere: every op processes the live rows, each up to its
## live length (`ChunkMeta`); dead, auxiliary and failed rows and the
## padding past a trace's end are left as they are (the `src → dst` form
## copies them unchanged).
##
## **Order and cache.** `apply` runs the ops on tiles of rows (about 256 KB
## of samples, a multiple of 8 rows): every op runs on one tile while it is
## in cache, then the next tile. For `src → dst`, a tile is copied into
## `dst` just before its first op, so the copy is not a separate pass over
## the panel. A per-panel normalization needs the whole panel first: it
## splits the flow there, and its factor is applied, fused, at the start of
## the next part.
##
## **Margins.** `requiredMargin` is the sum of the ops' margins (filter
## impulse lengths, half AGC windows): each op widens the context the next
## one sees.
##
## **Generations.** Every compiled plan gets a new `generation`, increasing
## across all plans of the process (an atomic counter, the one global here;
## it only numbers plans). A viewer tags each processed buffer with the
## generation that made it and reprocesses buffers whose tag is stale.
##
## **Threads.** A plan is immutable: any number of threads may apply one
## plan at once, each with its own `Workspace` and disjoint output rows (see
## `ProcessPool` for a pool that does this). A plan holds seqs: share it by
## reference or pointer, never copy it into a thread (a copy allocates on
## that thread).

import std/[atomics, math]
import parseSEG
import core/[basic, mute, rows, types, workspace]
import fft/plan
import filter/[apply, design, iir]
import gain/[curves, gain]

type
  OpKind* = enum
    ## The kind of an `Op`.
    opDemean     ## remove the mean (`demeanOp`)
    opDetrend    ## remove a least-squares line (`detrendOp`)
    opScale      ## multiply by a factor (`scaleOp`)
    opGainDb     ## a fixed gain in dB (`gainDbOp`)
    opClip       ## clip to `lo .. hi` (`clipOp`)
    opPolarity   ## reverse the polarity (`polarityOp`)
    opMuteTop    ## a top mute (`muteTopOp`)
    opMuteBottom ## a bottom mute (`muteBottomOp`)
    opFilter     ## a zero-phase FFT filter (`filterOp`, `bandPassOp`)
    opIir        ## a Butterworth biquad filter (`iirOp`)
    opAgc        ## automatic gain control (`agcOp`)
    opNormalize  ## normalization per trace or panel (`normalizeOp`)
    opGainCurve  ## a time-variant gain (`gainCurveOp`)

  ClipParams* = object
    ## The limits of an `opClip`.
    lo*, hi*: float64

  MuteParams* = object
    ## An `opMuteTop` or `opMuteBottom`: the mute time per offset and the
    ## taper length on the muted side.
    curve*: MuteCurve
    taperMs*: float64

  FilterParams* = object
    ## An `opFilter`: the response and the trend removed before filtering.
    spec*: FilterSpec
    trend*: TrendRemoval

  IirParams* = object
    ## An `opIir`: a Butterworth spec, the phase and the trend removal.
    spec*: FilterSpec
    phase*: IirPhase
    trend*: TrendRemoval

  AgcParams* = object
    ## An `opAgc`; see `initAgcPlan`.
    windowMs*: float64
    mode*: AgcMode
    target*: float64
    epsilon*: float64

  NormalizeParams* = object
    ## An `opNormalize`; see `initNormalizePlan`.
    stat*: NormStat
    scope*: NormScope
    target*: float64
    percentile*: float64

  GainCurveParams* = object
    ## An `opGainCurve`; see `initGainCurvePlan` (`tMaxMs = NaN`: the end of
    ## the record).
    curve*: GainCurve
    tMinMs*, tMaxMs*: float64

  Op* = object
    ## One operation of a `Pipeline`, as plain data. Build it with the
    ## `xOp` builders; its parameters may be edited in place.
    case kind*: OpKind
    of opDemean, opDetrend, opPolarity: discard
    of opScale: factor*: float64
    of opGainDb: dB*: float64
    of opClip: clip*: ClipParams
    of opMuteTop, opMuteBottom: mute*: MuteParams
    of opFilter: filter*: FilterParams
    of opIir: iir*: IirParams
    of opAgc: agc*: AgcParams
    of opNormalize: norm*: NormalizeParams
    of opGainCurve: gainCurve*: GainCurveParams

  Pipeline* = object
    ## An ordered list of `Op`s; see the module documentation.
    ops: seq[Op]

  StepKind = enum
    stDemean, stDetrend, stScale, stClip, stPolarity, stMuteTop, stMuteBottom,
    stFilter, stIir, stAgc, stNormTrace, stNormPanel, stGainCurve

  Step[T: SampleType] = object
    case kind: StepKind
    of stDemean, stDetrend, stPolarity: discard
    of stScale: factor: float64
    of stClip: lo, hi: float64
    of stMuteTop, stMuteBottom:
      mute: MuteCurve
      taperMs: float64
    of stFilter: filter: FilterPlan[T]
    of stIir: iir: IirPlan[T]
    of stAgc: agc: AgcPlan
    of stNormTrace, stNormPanel: norm: NormalizePlan
    of stGainCurve: curve: GainCurvePlan[T]

  PipelinePlan*[T: SampleType] = object
    ## A compiled `Pipeline` for one `dtMs` and trace length; see the module
    ## documentation. Build it with `compile`.
    steps: seq[Step[T]]
    parts: seq[Slice[int]] # step ranges; a stNormPanel only ever starts one
    dtMs: float64
    samples: int
    margin: int
    ws: WorkspaceSize
    generation: uint64

var lastGeneration: Atomic[uint64] # numbers plans; see "Generations"

# ----------------------------------------------------------------- builders --

func demeanOp*(): Op =
  ## Removes each trace's mean.
  Op(kind: opDemean)

func detrendOp*(): Op =
  ## Removes each trace's least-squares line.
  Op(kind: opDetrend)

func scaleOp*(factor: float64): Op =
  ## Multiplies by `factor` (finite).
  Op(kind: opScale, factor: factor)

func gainDbOp*(dB: float64): Op =
  ## A fixed gain of `dB` (finite).
  Op(kind: opGainDb, dB: dB)

func clipOp*(lo, hi: float64): Op =
  ## Clips to `lo .. hi` (`lo ≤ hi`, not NaN).
  Op(kind: opClip, clip: ClipParams(lo: lo, hi: hi))

func polarityOp*(): Op =
  ## Reverses the polarity.
  Op(kind: opPolarity)

func muteTopOp*(curve: MuteCurve, taperMs = 0.0): Op =
  ## Zeroes each trace before its `curve` time, with a taper of `taperMs`
  ## on the muted side (`muteTop`).
  Op(kind: opMuteTop, mute: MuteParams(curve: curve, taperMs: taperMs))

func muteBottomOp*(curve: MuteCurve, taperMs = 0.0): Op =
  ## Zeroes each trace after its `curve` time (`muteBottom`).
  Op(kind: opMuteBottom, mute: MuteParams(curve: curve, taperMs: taperMs))

func filterOp*(spec: FilterSpec, trend = trLinear): Op =
  ## A zero-phase FFT filter with the D4 edge handling (`FilterPlan`).
  Op(kind: opFilter, filter: FilterParams(spec: spec, trend: trend))

proc bandPassOp*(corners: Corners, trend = trLinear): Op =
  ## `filterOp(bandPass(corners))`. Raises `ValueError` for corners that
  ## `bandPass` rejects.
  filterOp(bandPass(corners), trend)

func iirOp*(spec: FilterSpec, phase = ipZero, trend = trLinear): Op =
  ## A Butterworth biquad filter (`IirPlan`; `spec` from `butterBandPass`,
  ## `butterLowCut` or `butterHighCut`).
  Op(kind: opIir, iir: IirParams(spec: spec, phase: phase, trend: trend))

func agcOp*(windowMs: float64, mode = agcRms, target = 1.0,
    epsilon = 1e-9): Op =
  ## AGC over a centred window of about `windowMs` (`AgcPlan`).
  Op(kind: opAgc, agc: AgcParams(windowMs: windowMs, mode: mode,
    target: target, epsilon: epsilon))

func normalizeOp*(stat: NormStat, scope = perTrace, target = 1.0,
    percentile = 99.0): Op =
  ## Brings `stat` of each trace, or of the panel, to `target`
  ## (`NormalizePlan`).
  Op(kind: opNormalize, norm: NormalizeParams(stat: stat, scope: scope,
    target: target, percentile: percentile))

func gainCurveOp*(curve: GainCurve, tMinMs = 0.0, tMaxMs = NaN): Op =
  ## A time-variant gain (`GainCurvePlan`), tabulated over `tMinMs ..
  ## tMaxMs` (NaN: to the end of the record).
  Op(kind: opGainCurve, gainCurve: GainCurveParams(curve: curve,
    tMinMs: tMinMs, tMaxMs: tMaxMs))

# ----------------------------------------------------------------- pipeline --

func initPipeline*(ops: openArray[Op] = []): Pipeline =
  ## A pipeline running `ops` in order.
  Pipeline(ops: @ops)

func len*(p: Pipeline): int {.inline.} =
  ## The number of ops.
  p.ops.len

func `[]`*(p: Pipeline, i: int): lent Op {.inline.} =
  ## Op `i`.
  p.ops[i]

func `[]`*(p: var Pipeline, i: int): var Op {.inline.} =
  ## Op `i`, to edit in place (`flow[2].agc.windowMs = 300`).
  p.ops[i]

func `[]=`*(p: var Pipeline, i: int, op: sink Op) {.inline.} =
  ## Replaces op `i`.
  p.ops[i] = op

func add*(p: var Pipeline, op: sink Op) =
  ## Appends `op`.
  p.ops.add op

func insert*(p: var Pipeline, op: sink Op, i: int) =
  ## Inserts `op` before op `i` (`i = len` appends).
  p.ops.insert(op, i)

func delete*(p: var Pipeline, i: int) =
  ## Removes op `i`, keeping the order of the rest.
  p.ops.delete(i)

func clear*(p: var Pipeline) =
  ## Removes every op.
  p.ops.setLen(0)

iterator items*(p: Pipeline): lent Op =
  ## The ops, in order.
  for op in p.ops: yield op

iterator pairs*(p: Pipeline): tuple[key: int, val: lent Op] =
  ## The ops with their indices.
  for i in 0 ..< p.ops.len: yield (i, p.ops[i])

# ------------------------------------------------------------------ compile --

proc badOp(i: int, kind: OpKind, msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, "op " & $i & " (" & $kind & "): " & msg)

proc badApply(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func finite(x: float64): bool {.inline.} = x.classify notin {fcNan, fcInf, fcNegInf}

proc compileOp[T](op: Op, dtMs: float64, samples: int): Step[T] =
  case op.kind
  of opDemean: Step[T](kind: stDemean)
  of opDetrend: Step[T](kind: stDetrend)
  of opPolarity: Step[T](kind: stPolarity)
  of opScale:
    if not op.factor.finite:
      raise newException(ValueError, "the factor must be finite, got " & $op.factor)
    if not float64(T(op.factor)).finite:
      raise newException(ValueError, "a factor of " & $op.factor &
        " overflows " & $T)
    Step[T](kind: stScale, factor: op.factor)
  of opGainDb:
    if not op.dB.finite:
      raise newException(ValueError, "the gain must be finite, got " & $op.dB & " dB")
    let f = dbToFactor(op.dB)
    if not f.finite or not float64(T(f)).finite:
      raise newException(ValueError, "a gain of " & $op.dB & " dB overflows " & $T)
    Step[T](kind: stScale, factor: f)
  of opClip:
    let c = op.clip
    if c.lo.isNaN or c.hi.isNaN or c.lo > c.hi:
      raise newException(ValueError, "clip limits must satisfy lo ≤ hi, got " &
        $c.lo & " .. " & $c.hi)
    Step[T](kind: stClip, lo: c.lo, hi: c.hi)
  of opMuteTop, opMuteBottom:
    let m = op.mute
    if not m.taperMs.finite or m.taperMs < 0:
      raise newException(ValueError, "a mute taper must be finite and ≥ 0 ms, got " &
        $m.taperMs)
    if op.kind == opMuteTop: Step[T](kind: stMuteTop, mute: m.curve,
        taperMs: m.taperMs)
    else: Step[T](kind: stMuteBottom, mute: m.curve, taperMs: m.taperMs)
  of opFilter:
    Step[T](kind: stFilter, filter: initFilterPlan(T, op.filter.spec, dtMs,
      samples, op.filter.trend))
  of opIir:
    Step[T](kind: stIir, iir: initIirPlan(T, op.iir.spec, dtMs, samples,
      op.iir.phase, op.iir.trend))
  of opAgc:
    let a = op.agc
    Step[T](kind: stAgc, agc: initAgcPlan(a.windowMs, dtMs, samples, a.mode,
      a.target, a.epsilon))
  of opNormalize:
    let n = op.norm
    let plan = initNormalizePlan(n.stat, samples, n.scope, n.target, n.percentile)
    if n.scope == perPanel: Step[T](kind: stNormPanel, norm: plan)
    else: Step[T](kind: stNormTrace, norm: plan)
  of opGainCurve:
    let g = op.gainCurve
    Step[T](kind: stGainCurve, curve: initGainCurvePlan(T, g.curve, dtMs,
      samples, g.tMinMs, g.tMaxMs))

func margin[T](s: Step[T]): int =
  case s.kind
  of stFilter: s.filter.requiredMargin
  of stIir: s.iir.requiredMargin
  of stAgc: s.agc.requiredMargin
  else: 0

func workspaceSize[T](s: Step[T]): WorkspaceSize =
  case s.kind
  of stFilter: s.filter.workspaceSize
  of stIir: s.iir.workspaceSize
  of stAgc: s.agc.workspaceSize
  of stNormTrace: s.norm.workspaceSize
  of stNormPanel: s.norm.workspaceSize # panelStatsWorkspace for a percentile
  else: WorkspaceSize()

proc compile*(p: Pipeline, T: typedesc[SampleType], dtMs: float64,
    samples: int): PipelinePlan[T] =
  ## Validates every op of `p` for traces of up to `samples` samples at
  ## `dtMs` and precomputes them, giving a plan with a new `generation`.
  ## Raises `ValueError` for `dtMs` not finite and positive, `samples < 1`,
  ## or any invalid op; the message starts with the op's index and kind
  ## (`op 1 (opFilter): ...`). An empty pipeline compiles: it copies
  ## (`src → dst`) or does nothing (in place).
  if not dtMs.finite or dtMs <= 0:
    raise newException(ValueError, "dtMs must be finite and positive, got " & $dtMs)
  if samples < 1:
    raise newException(ValueError, "a pipeline needs samples ≥ 1, got " & $samples)
  result.dtMs = dtMs
  result.samples = samples
  result.steps = newSeqOfCap[Step[T]](p.ops.len)
  for i, op in p.ops:
    try:
      result.steps.add compileOp[T](op, dtMs, samples)
    except ValueError as e:
      badOp(i, op.kind, e.msg)
  var start = 0
  for i, s in result.steps:
    result.margin += s.margin
    result.ws = max(result.ws, s.workspaceSize)
    if s.kind == stNormPanel and i > start:
      result.parts.add start ..< i
      start = i
  result.parts.add start ..< result.steps.len # one part even when empty
  result.generation = lastGeneration.fetchAdd(1) + 1

proc requiredMargin*(p: Pipeline, dtMs: float64): int =
  ## The `requiredMargin` of `p` compiled at `dtMs`, without compiling it:
  ## it does not depend on the trace length, so a viewer can size a zoom
  ## window's decode before building the plan for its width. Raises
  ## `ValueError` as `compile` for an op that sets a margin (a filter, an
  ## IIR filter or an AGC); other ops are not checked here.
  if not dtMs.finite or dtMs <= 0:
    raise newException(ValueError, "dtMs must be finite and positive, got " & $dtMs)
  for i, op in p.ops:
    try:
      case op.kind
      of opFilter:
        op.filter.spec.validate(dtMs)
        result += op.filter.spec.impulseLength(dtMs)
      of opIir:
        result += initIirPlan(float64, op.iir.spec, dtMs, 1, op.iir.phase,
          op.iir.trend).requiredMargin
      of opAgc:
        let a = op.agc
        result += initAgcPlan(a.windowMs, dtMs, 1, a.mode, a.target,
          a.epsilon).requiredMargin
      else: discard
    except ValueError as e:
      badOp(i, op.kind, e.msg)

func decodeWindow*(shown: Slice[int], margin, traceLen: int): Slice[int] {.
    raises: [].} =
  ## The samples to decode for a zoom that shows samples `shown` of traces
  ## `traceLen` samples long: `shown` widened by `margin` on each side and
  ## clamped to `0 ..< traceLen`. Decode it (as the prefetcher's
  ## `firstSample` and `cols`, or `readTracesInto`'s sample window), process
  ## it, and draw columns `shown.a − result.a .. shown.b − result.a`.
  ## `shown` must lie in `0 ..< traceLen` (else the result does not hold
  ## it); a negative `margin` counts as 0.
  let m = clamp(margin, 0, max(traceLen, 0)) # no overflow for a huge margin
  max(0, shown.a - m) .. min(traceLen - 1, shown.b + m)

func generation*[T](plan: PipelinePlan[T]): uint64 {.inline, raises: [].} =
  ## The plan's id: never 0, and larger for every later `compile` in the
  ## process, so 0 can mean "never processed".
  plan.generation

func dtMs*[T](plan: PipelinePlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval the plan was compiled for.
  plan.dtMs

func samples*[T](plan: PipelinePlan[T]): int {.inline, raises: [].} =
  ## The widest block the plan takes. Any narrower block works too (a zoom
  ## window with its margin): there is no need to recompile for it, only
  ## the cost of FFTs sized for `samples`.
  plan.samples

func len*[T](plan: PipelinePlan[T]): int {.inline, raises: [].} =
  ## The number of compiled ops.
  plan.steps.len

func requiredMargin*[T](plan: PipelinePlan[T]): int {.inline, raises: [].} =
  ## The samples a zoom window needs on each side: the sum of the ops'
  ## margins.
  plan.margin

func workspaceSize*[T](plan: PipelinePlan[T]): WorkspaceSize {.inline,
    raises: [].} =
  ## The scratch `apply` needs.
  plan.ws

proc reserve*[T](ws: var Workspace, plan: PipelinePlan[T]) =
  ## `ws.reserve(plan.workspaceSize)`: grows `ws` for `plan`, a no-op when
  ## it already fits.
  ws.reserve(plan.ws)

# -------------------------------------------------------------------- apply --
#
# The parts API below is what `ProcessPool` uses, and what a caller that splits
# the rows over its own threads uses to keep per-panel normalizations right:
# `checkApply` once on the caller's thread, then per part: `panelFactorOf` over
# the whole block (one thread) if `partNeedsPanel`, then `runPart` on each
# thread's rows (`RowBlock.sub`, `ChunkMeta.sub`).

const tileBytes = 256 * 1024

func tileRows(cols, size: int): int =
  ## Rows per tile: about `tileBytes` of samples, a multiple of `fftLanes`.
  let r = tileBytes div max(1, cols * size)
  max(fftLanes, r div fftLanes * fftLanes)

proc checkApply*[T](plan: PipelinePlan[T], src, dst: RowBlock[T],
    meta: ChunkMeta) =
  ## Checks that `plan` can be applied from `src` (no rows: in place) into
  ## `dst` with `meta`, as `apply` does before touching a sample: the shapes,
  ## `dtMs`, the width, and that `src` and `dst` are the same block or do
  ## not overlap. Raises `ValueError` otherwise.
  if plan.parts.len == 0:
    badApply("a PipelinePlan must come from compile")
  meta.checkShape(dst)
  if src.len > 0:
    if src.len != dst.len or src.cols != dst.cols:
      badApply("src is " & $src.len & " × " & $src.cols & ", dst " & $dst.len &
        " × " & $dst.cols)
    let bytes = dst.len * dst.cols * sizeof(T)
    let s0 = cast[uint](src.rowPtr(0))
    let d0 = cast[uint](dst.rowPtr(0))
    if s0 != d0 and s0 < d0 + uint(bytes) and d0 < s0 + uint(bytes):
      badApply("src and dst overlap without being the same block")
  if abs(meta.dtMs - plan.dtMs) > 1e-9 * plan.dtMs:
    badApply("the plan is for dtMs = " & $plan.dtMs & ", the rows have " & $meta.dtMs)
  if dst.cols > plan.samples:
    badApply("the block has " & $dst.cols & " columns, the plan takes " &
      $plan.samples & " samples")

func partCount*[T](plan: PipelinePlan[T]): int {.inline, raises: [].} =
  ## The parts `apply` runs one after the other: 1, plus 1 per per-panel
  ## normalization after the first op. Each part but the first starts with
  ## one.
  plan.parts.len

func partNeedsPanel*[T](plan: PipelinePlan[T], part: int): bool {.inline,
    raises: [].} =
  ## Part `part` starts with a per-panel normalization, whose factor
  ## (`panelFactorOf`) must be computed over the whole block first.
  let r = plan.parts[part]
  r.len > 0 and plan.steps[r.a].kind == stNormPanel

proc panelFactorOf*[T](plan: PipelinePlan[T], part: int, b: RowBlock[T],
    meta: ChunkMeta, ws: var Workspace): float64 {.raises: [].} =
  ## The factor of part `part`'s per-panel normalization over the whole
  ## block `b` (the data as the previous parts left it), or NaN when it
  ## leaves the block alone. Needs `checkApply` first; an
  ## `AssertionDefect` for a part that does not start with a per-panel
  ## normalization, or a `ws` smaller than `plan.workspaceSize`.
  doAssert part >= 0 and part < plan.parts.len and plan.partNeedsPanel(part),
    "part " & $part & " has no per-panel normalization"
  doAssert ws.fits(plan.ws), "reserve the workspace for the plan first"
  try:
    panelFactor(b, meta, plan.steps[plan.parts[part].a].norm, ws)
  except ValueError as e:
    raiseAssert "checked by checkApply: " & e.msg

proc runStep[T](s: Step[T], b: RowBlock[T], meta: ChunkMeta,
    ws: var Workspace) {.raises: [ValueError].} =
  case s.kind
  of stDemean: demean(b, meta)
  of stDetrend: detrend(b, meta)
  of stPolarity: reversePolarity(b, meta)
  of stScale: scale(b, meta, s.factor)
  of stClip: clip(b, meta, s.lo, s.hi)
  of stMuteTop: muteTop(b, meta, s.mute, s.taperMs)
  of stMuteBottom: muteBottom(b, meta, s.mute, s.taperMs)
  of stFilter: applyFilter(b, meta, s.filter, ws)
  of stIir: applyFilter(b, meta, s.iir, ws)
  of stAgc: agc(b, meta, s.agc, ws)
  of stNormTrace: normalize(b, meta, s.norm, ws)
  of stNormPanel: discard # its factor is applied by runPart
  of stGainCurve: applyGainCurve(b, meta, s.curve)

proc runPart*[T](plan: PipelinePlan[T], part: int, src, dst: RowBlock[T],
    meta: ChunkMeta, factor: float64, ws: var Workspace) {.raises: [].} =
  ## Runs part `part` of `plan` on the rows of `dst`, tile by tile. When
  ## `src` has rows, each tile is first copied from `src` (the same shape as
  ## `dst`), so pass it for the first part of a `src → dst` apply only. A
  ## finite `factor` (from `panelFactorOf`) scales the live rows first. `ws`
  ## must hold `plan.workspaceSize` (an `AssertionDefect` otherwise, as for
  ## a part outside `0 ..< partCount`). Needs `checkApply` first.
  doAssert part >= 0 and part < plan.parts.len, "no part " & $part
  doAssert ws.fits(plan.ws), "reserve the workspace for the plan first"
  let r = plan.parts[part]
  let first = (if plan.partNeedsPanel(part): r.a + 1 else: r.a)
  let copy = src.len > 0
  let n = dst.len
  if n == 0: return
  let step = tileRows(dst.cols, sizeof(T))
  try:
    var a = 0
    while a < n:
      let rows = a .. min(n, a + step) - 1
      let tb = dst.sub(rows)
      let tm = meta.sub(rows)
      if copy:
        let sp = src.rowPtr(a)
        let dp = dst.rowPtr(a)
        if sp != dp:
          copyMem(dp, sp, rows.len * dst.cols * sizeof(T))
      if not factor.isNaN:
        scale(tb, tm, factor)
      for i in first .. r.b:
        runStep(plan.steps[i], tb, tm, ws)
      a += step
  except ValueError as e:
    raiseAssert "checked by checkApply: " & e.msg

proc applyBlocks[T](plan: PipelinePlan[T], src, dst: RowBlock[T],
    meta: ChunkMeta, ws: var Workspace) =
  plan.checkApply(src, dst, meta)
  doAssert ws.fits(plan.ws), "reserve the workspace for the plan first"
  for part in 0 ..< plan.parts.len:
    let from0 = part == 0 and src.len > 0
    var factor = NaN
    if plan.partNeedsPanel(part):
      factor = plan.panelFactorOf(part, (if from0: src else: dst), meta, ws)
    plan.runPart(part, (if from0: src else: RowBlock[T]()), dst, meta, factor, ws)

proc apply*[T: SampleType](plan: PipelinePlan[T], b: RowBlock[T],
    meta: ChunkMeta, ws: var Workspace) =
  ## Runs `plan` on the block `b` in place. `ws` must hold
  ## `plan.workspaceSize` (`ws.reserve(plan)`; an `AssertionDefect`
  ## otherwise). Raises `ValueError`, before any sample changes, when `b`
  ## does not match `meta`, is wider than `plan.samples`, or the rows'
  ## `dtMs` is not the plan's. Allocates nothing.
  plan.applyBlocks(RowBlock[T](), b, meta, ws)

proc apply*[T: SampleType](plan: PipelinePlan[T], src, dst: RowBlock[T],
    meta: ChunkMeta, ws: var Workspace) =
  ## Runs `plan` on `src` into `dst` (the same shape; raises `ValueError`
  ## otherwise), leaving `src` as it is. Rows that are not live are copied
  ## unchanged. The copy is fused into the first op, tile by tile. `src` and
  ## `dst` must not overlap, unless they are the same block (then it runs in
  ## place: raises `ValueError` for a partial overlap). Raises as the
  ## in-place form. The caller reserves `ws`: `apply` never grows it.
  if src.len != dst.len or src.cols != dst.cols:
    badApply("src is " & $src.len & " × " & $src.cols & ", dst " & $dst.len &
      " × " & $dst.cols)
  plan.applyBlocks(src, dst, meta, ws)

proc apply*[T: SampleType](plan: PipelinePlan[T], samples: var Tensor[T],
    meta: ChunkMeta, ws: var Workspace) =
  ## `apply` in place on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  plan.apply(rowBlock(samples), meta, ws)

proc apply*[T: SampleType](plan: PipelinePlan[T], src: Tensor[T],
    dst: var Tensor[T], meta: ChunkMeta, ws: var Workspace) =
  ## `apply` from the tensor `src` into `dst`, a tensor of the same shape
  ## (both C-contiguous; raises `ValueError` otherwise). `dst` is not
  ## resized: allocate it once and reuse it.
  meta.checkShape(src)
  meta.checkShape(dst)
  plan.apply(rowBlock(src), rowBlock(dst), meta, ws)
