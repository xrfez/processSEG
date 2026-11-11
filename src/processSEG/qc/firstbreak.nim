## First-break picking: an optional minimum-phase band-pass, then one of three pickers
## on the trace energy, a search window (fixed, or guided by a linear
## moveout), and a confidence per pick.
##
## ```nim
## let params = FirstBreakParams(pickMethod: pmEnergyRatio, windowMs: 20,
##   lowCutHz: 8, highCutHz: 60,
##   guideVelocity: 2000, beforeMs: 60, afterMs: 120)   # LMO-guided window
## let plan = initFirstBreakPlan(float32, params, meta.dtMs, samples.shape[1])
## var ws = initWorkspace()
## ws.reserve(plan.workspaceSize)
## var picks = newSeq[FirstBreakPick](meta.len)
## plan.pickInto(samples, meta, ws, picks)     # picks[row].timeMs, .confidence
## ```
##
## **Pickers** (on the energy `e = x²` of the filtered trace, with windows
## of `w = windowMs` samples; the pick is in the search window):
##
## - `pmStaLta`: the mean energy of the `windowMs` ending at the sample
##   (STA) over the mean of the `ltaMs` before that window (LTA); the pick
##   is the largest rise of that ratio (smoothed over `w / 2` samples),
##   where the break's energy enters the STA (a ratio's maximum sits on a
##   plateau up to a window after the onset). This is the energy arrival,
##   about a quarter period after the onset: a bias that is the same on
##   every trace of one wavelet, so relative times keep their accuracy;
## - `pmEnergyRatio`: the modified energy ratio (Wong et al., 2009): the
##   energy of the `w` samples from the sample over that of the `w` before
##   it (equal windows), times the amplitude there (the RMS of the `w / 4`
##   samples from it); the pick is the maximum.
##   The plain ratio has a plateau up to a window before the onset, where
##   noise sets the maximum; the weight moves it onto the break's first
##   strong sample, a bias that is the same on every trace;
## - `pmCoppens`: modified Coppens (Sabbione and Velis, 2010): the energy
##   of the `w` samples ending at the sample over the energy from the start
##   of the trace to it, plus a stabilization term `β = stabilization ·
##   (mean energy) · w`; the pick is the largest rise of that ratio
##   (smoothed over `w / 2` samples).
##
## The ratios' denominators get `1e-6` of the trace's mean energy, so a
## zero (muted) stretch does not divide by zero. A pick needs data before
## it: one and a half windows (STA/LTA: the STA and an LTA cut by the
## trace's start to no less than half a window), half a window (energy
## ratio, whose windows are cut by both trace ends), or one sample
## (Coppens).
##
## **Windows and noise.** Short windows follow the break closely; at low
## SNR the noise in a short "before" window sets the maximum, so lengthen
## `windowMs` (e.g. to 1–2 periods of the break) and high-cut the noise.
##
## **Search window.** The columns between `searchStartMs` and `searchEndMs`
## (trace times, `t0Ms` based; NaN: the trace's start or end) and, when
## `guideVelocity` is set, `[tLmo − beforeMs, tLmo + afterMs]` with `tLmo =
## guideT0Ms + 1000·|offset| / guideVelocity`. A guided pick needs the
## offset: a NaN offset gives no pick.
##
## **Confidence.** `snrDb` is 10·log₁₀ of the mean energy of the
## `snrWindowMs` after the pick over that of the `snrWindowMs` ending
## `windowMs / 2` before it (a guard gap, so a pick a little late does not
## count the break as noise; each window cut by the trace's ends; a pick
## with nothing before the gap has a NaN `snrDb` and confidence 0), and `confidence` maps it linearly from `lowDb` (0)
## to `highDb` (1), clamped. The same measure serves every picker, so
## confidences compare across methods. Noise alone gives about 0 dB, a
## clear break well above `highDb`.
##
## **No pick** (`timeMs`, `col`, `snrDb` NaN, `confidence` 0): a non-live
## row, an empty search window or a trace without energy there. A trace
## whose `t0Ms` is NaN takes no time window (no guide, no search bounds:
## else no pick) and its pick has a `col` but a NaN `timeMs`, as
## `ChunkMeta.sampleTimeMs` gives NaN there.
##
## **Band-pass.** A Butterworth minimum-phase (causal) band-pass
## (`filter/iir`, one pass): a zero-phase filter would put a precursor
## before the break, and picks with it. Its group delay moves the picks
## later by a few ms, the same on every trace.
##
## The input samples are not changed: the trace is filtered in the
## workspace. A plan is read-only and may be shared by threads, each with
## its own `Workspace`.

import std/math
import parseSEG
import ../core/[rows, types, workspace]
import ../filter/[design, iir]
import ../fft/plan as fftplan

type
  PickMethod* = enum
    ## The picker (see the module documentation).
    pmStaLta      ## Leading STA over trailing LTA.
    pmEnergyRatio ## Modified energy ratio: after over before, times |x|.
    pmCoppens     ## Modified Coppens: trailing window over cumulative energy.

  FirstBreakParams* = object
    ## What `initFirstBreakPlan` builds a plan from. Times in ms, velocities
    ## in offset units per second.
    pickMethod*: PickMethod = pmEnergyRatio
    windowMs*: float64 = 20      ## STA, energy-ratio and Coppens window.
    ltaMs*: float64 = 100        ## `pmStaLta`: the long window.
    stabilization*: float64 = 0.2 ## `pmCoppens`: β, relative to the mean window energy.
    lowCutHz*: float64 = 0      ## Band-pass low cut before picking; 0: none.
    highCutHz*: float64 = Inf   ## Band-pass high cut before picking; Inf: none.
    slopeDbPerOctave*: float64 = 24 ## Each cut's slope (a multiple of 6).
    searchStartMs*: float64 = NaN ## Earliest pick time; NaN: the trace's start.
    searchEndMs*: float64 = NaN   ## Latest pick time; NaN: the trace's end.
    guideVelocity*: float64 = NaN ## LMO guide velocity; NaN: no guide.
    guideT0Ms*: float64 = 0       ## LMO guide intercept time.
    beforeMs*: float64 = 50       ## Guided window before the LMO time.
    afterMs*: float64 = 100       ## Guided window after the LMO time.
    snrWindowMs*: float64 = 40    ## Confidence windows.
    lowDb*: float64 = 6           ## SNR of confidence 0.
    highDb*: float64 = 20         ## SNR of confidence 1.

  FirstBreakPick* = object
    ## One trace's pick.
    timeMs*: float64     ## Pick time (`t0Ms` based); NaN: no pick, or no `t0Ms`.
    col*: float64        ## The pick's column in the block; NaN: no pick.
    snrDb*: float64      ## Energy after / before the pick, in dB; NaN: none.
    confidence*: float64 ## 0 .. 1 from `snrDb` (`lowDb` .. `highDb`); 0 without a pick.

  FirstBreakPlan*[T: SampleType] = object
    ## A compiled picker for traces of up to `samples` samples at `dtMs`.
    params: FirstBreakParams
    dtMs: float64
    samples: int
    filtered: bool
    filter: IirPlan[T]
    w, lta, snr: int ## Windows, in samples (≥ 1).

func noPick*(): FirstBreakPick {.inline, raises: [].} =
  ## A trace without a pick.
  FirstBreakPick(timeMs: NaN, col: NaN, snrDb: NaN, confidence: 0)

proc badParams(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func positive(x: float64): bool {.inline.} = x > 0 and x < Inf

func samplesIn(ms, dtMs: float64, samples: int): int {.inline.} =
  # A window of `ms` in samples, 1 .. samples + 1 (rounded as a float first:
  # a long window does not fit an int).
  int(clamp(round(ms / dtMs), 1.0, float64(samples + 1)))

proc initFirstBreakPlan*(T: typedesc[SampleType], params: FirstBreakParams,
    dtMs: float64, samples: int): FirstBreakPlan[T] =
  ## A plan picking with `params` on traces of up to `samples` samples at
  ## `dtMs`. Raises `ValueError` for windows that are not finite and
  ## positive, `ltaMs < windowMs`, a negative or NaN stabilization, `lowDb ≥
  ## highDb`, a search window with `searchEndMs < searchStartMs`, a guide
  ## velocity that is not positive, negative or NaN guide margins, cuts or
  ## a slope the Butterworth design rejects at `dtMs` (`butterBandPass`;
  ## the slope a multiple of 6 dB/octave), `samples < 1`, or a `dtMs` that
  ## is not finite and positive.
  let p = params
  if not positive(dtMs): badParams("dtMs must be finite and positive, got " & $dtMs)
  if samples < 1: badParams("a first-break plan needs samples ≥ 1, got " & $samples)
  if not (positive(p.windowMs) and positive(p.ltaMs) and positive(p.snrWindowMs)):
    badParams("first-break windows must be finite and positive")
  if p.ltaMs < p.windowMs: badParams("ltaMs must be ≥ windowMs")
  if not (p.stabilization >= 0 and p.stabilization < Inf):
    badParams("stabilization must be finite and ≥ 0, got " & $p.stabilization)
  if not (p.lowDb < p.highDb) or p.lowDb.classify == fcNegInf or p.highDb == Inf:
    badParams("need finite lowDb < highDb, got " & $p.lowDb & " and " & $p.highDb)
  if p.searchEndMs < p.searchStartMs: badParams("searchEndMs < searchStartMs")
  if p.searchStartMs == Inf or p.searchEndMs.classify == fcNegInf:
    badParams("the search window is empty")
  if not p.guideVelocity.isNaN:
    if not positive(p.guideVelocity):
      badParams("guideVelocity must be finite and positive (or NaN), got " &
        $p.guideVelocity)
    if not (p.beforeMs >= 0 and p.afterMs >= 0 and p.beforeMs < Inf and p.afterMs < Inf):
      badParams("beforeMs and afterMs must be finite and ≥ 0")
    if p.guideT0Ms.classify in {fcNan, fcInf, fcNegInf}:
      badParams("guideT0Ms must be finite")
  result = FirstBreakPlan[T](params: p, dtMs: dtMs, samples: samples,
    w: p.windowMs.samplesIn(dtMs, samples), lta: p.ltaMs.samplesIn(dtMs, samples),
    snr: p.snrWindowMs.samplesIn(dtMs, samples))
  if not (p.lowCutHz >= 0 and p.highCutHz > p.lowCutHz):
    badParams("need 0 ≤ lowCutHz < highCutHz, got " & $p.lowCutHz & " and " & $p.highCutHz)
  result.filtered = p.lowCutHz > 0 or p.highCutHz < Inf
  if result.filtered:
    let spec = if p.highCutHz == Inf: butterLowCut(p.lowCutHz, p.slopeDbPerOctave)
               elif p.lowCutHz == 0: butterHighCut(p.highCutHz, p.slopeDbPerOctave)
               else: butterBandPass(p.lowCutHz, p.highCutHz, p.slopeDbPerOctave)
    result.filter = initIirPlan(T, spec, dtMs, samples, ipMinimum)

func params*[T](plan: FirstBreakPlan[T]): lent FirstBreakParams {.inline, raises: [].} =
  ## What the plan was built from.
  plan.params

func dtMs*[T](plan: FirstBreakPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval the plan was built for.
  plan.dtMs

func samples*[T](plan: FirstBreakPlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  plan.samples

func filterSize[T](plan: FirstBreakPlan[T]): WorkspaceSize {.inline, raises: [].} =
  if plan.filtered: plan.filter.workspaceSize else: WorkspaceSize()

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func copyLen[T](plan: FirstBreakPlan[T]): int {.inline, raises: [].} =
  # Filtered traces are picked from copies: `fftLanes` rows, filtered as a
  # batch (`pickInto`), or one trace (`pick`). Unfiltered ones are read in
  # place.
  if plan.filtered: fftLanes * plan.samples else: 0

func workspaceSize*[T](plan: FirstBreakPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch of `pick` and `pickInto`: the filter's and `fftLanes`
  ## filtered copies of the traces (for a plan with a band-pass), and two
  ## float64 arrays of `samples + 1`.
  result = plan.filterSize
  when T is float32: result.f32 += plan.copyLen
  else: result.f64 += plan.copyLen
  result.f64 += 2 * (plan.samples + 1)

func copyBase[T](plan: FirstBreakPlan[T]): int {.inline, raises: [].} =
  # Where the copies start, in T elements: after the filter's scratch.
  plan.filterSize.elems(T)

func energyBase[T](plan: FirstBreakPlan[T]): int {.inline, raises: [].} =
  # Where the float64 arrays start: after the filter's scratch and, for
  # float64 traces, the copies.
  plan.filterSize.f64 + (when T is float64: plan.copyLen else: 0)

# ------------------------------------------------------------------ kernels --

{.push boundChecks: off, overflowChecks: off.}

proc prefixEnergy[T](x: ptr UncheckedArray[T], n: int, p: ptr UncheckedArray[float64]) {.
    inline, raises: [].} =
  # p[i] = Σ x[0 ..< i]², p[0] = 0.
  var s = 0.0
  p[0] = 0
  for i in 0 ..< n:
    let v = float64(x[i])
    s += v * v
    p[i + 1] = s

proc argmaxAttribute(m: PickMethod, p, a: ptr UncheckedArray[float64], n, lo, hi, w,
    lta: int, beta, eps: float64): int {.raises: [].} =
  # The column in lo ..< hi with the largest attribute; -1 when none is
  # positive. `a` holds n + 1 float64 of scratch (STA/LTA, Coppens).
  result = -1
  var best = 0.0
  template energy(a0, b0: int): float64 = p[b0] - p[a0] # samples a0 ..< b0
  template risePick(first, span: int) =
    # The largest mean rise of a[] over the span steps from i, i in lo ..< hi.
    let s = max(1, span)
    for i in max(lo, first) ..< hi:
      let j = min(n - 1, i - 1 + s)
      if j <= i - 1: continue
      let r = (a[j] - a[i - 1]) / float64(j - i + 1)
      if r > best:
        best = r
        result = i
  case m
  of pmStaLta:
    # Trailing STA (w samples ending at i) over the LTA before it; a[i] is
    # 0 until the LTA holds w / 2 samples (an LTA of a few samples is noise).
    let minLta = max(1, w div 2)
    for i in 0 ..< n:
      let s0 = max(0, i + 1 - w)
      let m0 = max(0, s0 - lta)
      a[i] = if s0 - m0 < minLta: 0.0
             else: (energy(s0, i + 1) / float64(i + 1 - s0)) /
               (energy(m0, s0) / float64(s0 - m0) + eps)
    risePick(w + minLta, w div 2)
  of pmEnergyRatio:
    let q = max(1, w div 4)
    for i in lo ..< hi:
      let k = min(w, min(i, n - i)) # equal windows, cut by both ends
      if k < max(1, w div 2): continue
      # MER: the ratio weighted by the amplitude at i (cubed in the paper;
      # the maximum is the same), here the RMS of q samples from i, so one
      # noise sample does not set it.
      let r = energy(i, i + k) / (energy(i - k, i) + eps * float64(k)) *
        sqrt(energy(i, min(n, i + q)))
      if r > best:
        best = r
        result = i
  of pmCoppens:
    # CR(i): the w samples ending at i over the energy up to i.
    for i in 0 ..< n:
      a[i] = energy(max(0, i + 1 - w), i + 1) / (p[i + 1] + beta + eps)
    risePick(1, w div 2)

{.pop.}

proc pickRow[T](plan: FirstBreakPlan[T], src: ptr UncheckedArray[T], n: int,
    offset, t0Ms, startCol: float64, ws: var Workspace,
    filtered: bool): FirstBreakPick {.raises: [].} =
  # One trace of n live samples whose column 0 is at `t0Ms + startCol·dt`;
  # `filtered`: `src` already went through the plan's band-pass (else it is
  # filtered here, into a copy, when the plan has one).
  result = noPick()
  if n < 2: return
  let pr = plan.params
  let dt = plan.dtMs
  # The search window, in columns.
  var tLo = if pr.searchStartMs.isNaN: -Inf else: pr.searchStartMs
  var tHi = if pr.searchEndMs.isNaN: Inf else: pr.searchEndMs
  if not pr.guideVelocity.isNaN:
    if offset.isNaN: return
    let tl = pr.guideT0Ms + 1000 * abs(offset) / pr.guideVelocity
    tLo = max(tLo, tl - pr.beforeMs)
    tHi = min(tHi, tl + pr.afterMs)
  if t0Ms.isNaN and (tLo > -Inf or tHi < Inf): return
  # Time of column 0 (NaN with t0: then no time window applies, above, and
  # the pick has a column but no time).
  let origin = t0Ms + startCol * dt
  let cLo = if origin.isNaN: -Inf else: (tLo - origin) / dt
  let cHi = if origin.isNaN: Inf else: (tHi - origin) / dt
  let lo = if cLo <= 0: 0 elif cLo >= float64(n): n else: int(ceil(cLo))
  let hi = if cHi < 0: 0 elif cHi >= float64(n - 1): n else: int(floor(cHi)) + 1
  if lo >= hi: return
  # The trace, filtered in the workspace (when not done by the caller).
  var x = src
  if plan.filtered and not filtered:
    x = ws.scratch(T, n, plan.copyBase)
    copyMem(x, src, n * sizeof(T))
    applyFilter(x.toOpenArray(0, n - 1), plan.filter, ws)
  let base = plan.energyBase
  let p = ws.scratch(float64, n + 1, base)
  let a = ws.scratch(float64, n + 1, base + plan.samples + 1)
  prefixEnergy(x, n, p)
  let total = p[n]
  if not (total > 0) or total == Inf: return
  let mean = total / float64(n)
  let eps = 1e-6 * mean
  let beta = pr.stabilization * mean * float64(plan.w)
  let col = argmaxAttribute(pr.pickMethod, p, a, n, lo, hi, plan.w, plan.lta, beta, eps)
  if col < 0: return
  let k1 = min(plan.snr, n - col)
  let gapEnd = max(0, col - plan.w div 2) # guard gap: a pick a little late
  let k0 = min(plan.snr, gapEnd)
  let after = (p[col + k1] - p[col]) / float64(max(k1, 1))
  let before = (p[gapEnd] - p[gapEnd - k0]) / float64(max(k0, 1))
  result.col = float64(col)
  result.timeMs = origin + float64(col) * dt
  if k0 > 0: # else nothing before the gap: no SNR, confidence 0
    let snr = 10 * log10((after + eps) / (before + eps))
    result.snrDb = snr
    result.confidence = clamp((snr - pr.lowDb) / (pr.highDb - pr.lowDb), 0.0, 1.0)

proc pick*[T: SampleType](plan: FirstBreakPlan[T], x: openArray[T], offset = NaN,
    t0Ms = 0.0, ws: var Workspace): FirstBreakPick {.raises: [].} =
  ## The pick on one trace `x` (its live samples, at most `plan.samples`;
  ## an `AssertionDefect` otherwise) whose sample 0 is at `t0Ms`, with
  ## source–receiver `offset` for a guided window. `ws` must hold
  ## `plan.workspaceSize`. `x` is not changed.
  doAssert x.len <= plan.samples, "a trace longer than the plan's samples"
  if x.len == 0: return noPick()
  plan.pickRow(cast[ptr UncheckedArray[T]](unsafeAddr x[0]), x.len, offset, t0Ms, 0,
    ws, filtered = false)

proc pickInto*[T: SampleType](plan: FirstBreakPlan[T], b: RowBlock[T], meta: ChunkMeta,
    ws: var Workspace, dst: var openArray[FirstBreakPick]) =
  ## The pick of every live row of `b` into `dst[row]` (each row to its live
  ## length, on `meta`'s time axis; non-live rows get `noPick`). `ws` must
  ## hold `plan.workspaceSize`. Raises `ValueError` when `b` does not match
  ## `meta`, has more columns than `plan.samples`, its interval is not the
  ## plan's, or `dst` is not `meta.len` long.
  meta.checkShape(b)
  if b.cols > plan.samples:
    raise newException(ValueError, "the block has " & $b.cols & " columns, the plan " &
      "takes " & $plan.samples)
  if meta.dtMs != plan.dtMs:
    raise newException(ValueError, "the block's interval " & $meta.dtMs &
      " ms is not the plan's " & $plan.dtMs & " ms")
  if dst.len != meta.len:
    raise newException(ValueError, "dst holds " & $dst.len & " picks, the meta " &
      $meta.len & " rows")
  template pickOne(row: int, src: ptr UncheckedArray[T], filtered: bool) =
    dst[row] = noPick()
    if meta.isLive(row):
      let k = meta.keys(row)
      dst[row] = plan.pickRow(src, meta.liveSamples(row, b.cols), k.offset,
        k.t0Ms, float64(meta.firstSample), ws, filtered)
  if not plan.filtered or b.cols == 0:
    for row in 0 ..< meta.len: pickOne(row, b.rowPtr(row), false)
    return
  # Band-pass `fftLanes` rows at a time, as one batch of the IIR filter's
  # lanes, into copies, then pick each copy.
  let c = ws.scratch(T, fftLanes * b.cols, plan.copyBase)
  var g = 0
  while g < meta.len:
    let rows = g .. min(g + fftLanes, meta.len) - 1
    for i in rows:
      copyMem(addr c[(i - g) * b.cols], b.rowPtr(i), b.cols * sizeof(T))
    let cb = rowBlock(c.toOpenArray(0, rows.len * b.cols - 1), rows.len, b.cols)
    applyFilter(cb, meta.sub(rows), plan.filter, ws)
    for i in rows: pickOne(i, cb.rowPtr(i - g), true)
    g += fftLanes

proc pickInto*[T: SampleType](plan: FirstBreakPlan[T], samples: Tensor[T],
    meta: ChunkMeta, ws: var Workspace, dst: var openArray[FirstBreakPick]) =
  ## `pickInto` on a row-major `[rows, samples]` tensor.
  plan.pickInto(rowBlock(samples), meta, ws, dst)
