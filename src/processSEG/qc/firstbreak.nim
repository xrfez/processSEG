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
##   every trace of one wavelet and one SNR, so relative times keep their
##   accuracy (see "Band-pass" for what the SNR does to it);
## - `pmEnergyRatio`: the modified energy ratio (Wong et al., 2009): the
##   energy of the `w` samples from the sample over that of the `w` before
##   it (equal windows), times the amplitude there (the root of the energy
##   of the `w / 4` samples from it, an RMS up to a constant); the pick is
##   the maximum.
##   The plain ratio has a plateau up to a window before the onset, where
##   noise sets the maximum; the weight moves it onto the break's first
##   strong sample, a bias that is the same on every trace of one SNR;
## - `pmCoppens`: Coppens' energy ratio with the stabilization term of the
##   modified Coppens method (Sabbione and Velis, 2010), but without that
##   method's edge-preserving smoothing of the ratio: the energy of the `w`
##   samples ending at the sample over the energy from the start of the
##   trace to it plus `β = stabilization · (mean energy) · w`; the pick is
##   the largest rise of that ratio (a mean rise over `w / 2` samples, in
##   place of the smoothing).
##
## The ratios' denominators get `1e-6` of the mean energy of the samples
## read (below), so a zero (muted) stretch does not divide by zero. A pick
## needs data before it: one and a half windows (STA/LTA: the STA and an
## LTA cut by the trace's start to no less than half a window), half a
## window (energy ratio, whose windows are cut by both trace ends), or one
## window (Coppens: until its numerator window is whole, the ratio rises
## with the window, not with the data).
##
## **What a pick reads.** With a search window (fixed or guided),
## `pmStaLta` and `pmEnergyRatio` read, and band-pass, only the stretch of
## the trace that picks in the window depend on: before the window's first
## column what the attribute and the confidence need there (the STA and the
## LTA, or the energy ratio's window; the noise window and its gap) and the
## band-pass's `requiredMargin` to settle in; after its last column the
## longer of `windowMs` and `snrWindowMs`. The pick is a function of that
## stretch alone: what lies outside it (a later, stronger event, the rest
## of a long record) costs no time and moves no pick. The stretch is
## band-passed as it is; a whole trace has its linear trend taken out
## first. Without a window, and for `pmCoppens` always (its ratio is
## cumulative from the trace's start, and its `β` set by the whole trace),
## the whole trace is read. So a zoom window that holds `requiredMargin`
## samples on each side of the search window picks as the whole trace does.
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
## **Confidence.** `snrDb` is the windowed SNR of `qc/snr` (`windowSnrDb`,
## `seRatio`, no floor) of the filtered trace: 10·log₁₀ of the mean energy
## of the `snrWindowMs` after the pick over that of the `snrWindowMs`
## ending `windowMs / 2` before it (a guard gap, so a pick a little late
## does not count the break as noise; each window cut by the trace's ends,
## each mean energy plus `1e-6` of that of the samples read; a pick with
## nothing before the gap, or with no energy at all there, as after a
## dropout, has a NaN `snrDb` and confidence 0), and `confidence` maps it
## linearly from `lowDb` (0) to `highDb` (1), clamped. The same measure
## serves every picker, so confidences compare across methods. Noise alone
## gives about 0 dB, a clear break well above `highDb`.
##
## **Zeros.** The zeros before a row's first and after its last non-zero
## sample (a recording delay, padding, a mute) are not data: a row is
## picked over its `liveSpan` (`stack/stack`), so the step from zeros to
## data is not a break, the zeros are not its noise, and the data a pick
## needs before it count from the first non-zero sample. The price: a
## noise-free synthetic, whose break is that very step, is not picked on it
## (the pick comes late or not at all, without confidence). Its break is
## `liveSpan(x).a`; to exercise the pickers, add noise.
##
## **No pick** (`timeMs` and `snrDb` NaN, `col` −1, `confidence` 0): a non-live
## row, an empty search window or a trace without energy there. A trace
## whose `t0Ms` is NaN takes no time window (no guide, no search bounds:
## else no pick) and its pick has a `col` but a NaN `timeMs`, as
## `ChunkMeta.sampleTimeMs` gives NaN there.
##
## **Band-pass.** A Butterworth minimum-phase (causal) band-pass
## (`filter/iir`, one pass): a zero-phase filter would put a precursor
## before the break, and picks with it. It moves the picks later, and by
## more at a lower SNR: the filtered break grows out of the noise over a
## few samples (the filter's rise time), and a picker finds it where it
## clears the noise. Median pick minus onset, in samples, of 400 traces
## with a 30 Hz onset at 2 ms in a guided window, default windows, at noise
## of 2.6 %, 6.6 %, 13 % and 26 % of the onset's peak:
##
##     picker          no filter              8–60 Hz, 24 dB/octave
##     pmStaLta        1.9  1.9  1.9  1.8     7.7  8.7  9.5  10.3
##     pmEnergyRatio   0.7  0.9  1.2  1.6     2.9  3.6  4.1   4.8
##     pmCoppens       0.6  1.0  1.4  1.7     2.8  3.6  9.7  10.7
##
## (Coppens with the band-pass jumps a cycle at 13 %: half its picks are on
## either side.) So along a spread whose SNR falls with offset, band-passed
## picks lag by a few samples more at the far traces; without the filter
## the energy ratio and Coppens drift by one sample and STA/LTA by none.
## That is harmless for geometry QC and a bias for a velocity read from
## the picks' slope: filter only when the noise needs it, and compare picks
## of like SNR.
##
## The input samples are not changed: the trace is filtered in the
## workspace. A plan is read-only and may be shared by threads, each with
## its own `Workspace`; `ProcessPool.pickInto` (`core/pool`) picks a block's
## rows on its workers.

import std/math
import parseSEG
import ../core/[rows, types, workspace]
import ../filter/[design, iir]
import ../fft/plan as fftplan
import ../stack/stack
import snr

type
  PickMethod* = enum
    ## The picker (see the module documentation).
    pmStaLta      ## Trailing STA over the LTA before it.
    pmEnergyRatio ## Modified energy ratio: after over before, times |x|.
    pmCoppens     ## Modified Coppens: trailing window over cumulative energy.

  FirstBreakParams* = object
    ## What `initFirstBreakPlan` builds a plan from. Times in ms, velocities
    ## in offset units per second.
    pickMethod*: PickMethod = pmEnergyRatio
    windowMs*: float64 = 20 ## STA, energy-ratio and Coppens window.
    ltaMs*: float64 = 100 ## `pmStaLta`: the long window.
    stabilization*: float64 = 0.2 ## `pmCoppens`: β, relative to the mean window energy.
    lowCutHz*: float64 = 0 ## Band-pass low cut before picking; 0: none.
    highCutHz*: float64 = Inf ## Band-pass high cut before picking; Inf: none.
    slopeDbPerOctave*: float64 = 24 ## Each cut's slope (a multiple of 6).
    searchStartMs*: float64 = NaN ## Earliest pick time; NaN: the trace's start.
    searchEndMs*: float64 = NaN ## Latest pick time; NaN: the trace's end.
    guideVelocity*: float64 = NaN ## LMO guide velocity; NaN: no guide.
    guideT0Ms*: float64 = 0 ## LMO guide intercept time.
    beforeMs*: float64 = 50 ## Guided window before the LMO time.
    afterMs*: float64 = 100 ## Guided window after the LMO time.
    snrWindowMs*: float64 = 40 ## Confidence windows.
    lowDb*: float64 = 6 ## SNR of confidence 0.
    highDb*: float64 = 20 ## SNR of confidence 1.

  FirstBreakPick* = object
    ## One trace's pick; as built, `FirstBreakPick()`, no pick.
    timeMs*: float64 = NaN ## Pick time (`t0Ms` based); NaN: no pick, or no `t0Ms`.
    col*: int = -1         ## The pick's column in the block; −1: no pick.
    snrDb*: float64 = NaN  ## Energy after / before the pick, in dB; NaN: none.
    confidence*: float64   ## 0 .. 1 from `snrDb` (`lowDb` .. `highDb`); 0 without a pick.

  FirstBreakPlan*[T: SampleType] = object
    ## A compiled picker for traces of up to `samples` samples at `dtMs`.
    params: FirstBreakParams
    dtMs: float64
    samples: int
    filtered: bool
    filter: IirPlan[T]
    w, lta, snr: int   ## Windows, in samples (≥ 1).
    whole: bool        ## A pick reads the whole trace, not a stretch about its window.
    before, after: int ## What a pick reads around its search window, in samples.

func noPick*(): FirstBreakPick {.inline, raises: [].} =
  ## A trace without a pick.
  FirstBreakPick(timeMs: NaN, col: -1, snrDb: NaN, confidence: 0)

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
  # Coppens' ratio is cumulative from the trace's start and its β is set by
  # the whole trace, so it reads all of it, as a plan without a window does.
  result.whole = p.pickMethod == pmCoppens or
    (p.guideVelocity.isNaN and p.searchStartMs.isNaN and p.searchEndMs.isNaN)
  if result.filtered:
    let spec = if p.highCutHz == Inf: butterLowCut(p.lowCutHz, p.slopeDbPerOctave)
               elif p.lowCutHz == 0: butterHighCut(p.highCutHz, p.slopeDbPerOctave)
               else: butterBandPass(p.lowCutHz, p.highCutHz, p.slopeDbPerOctave)
    # A whole trace is detrended before the band-pass. A stretch is not: a
    # line fitted to a few windows about the break is set by the break, and
    # taking it out tilts the noise before it (by about the noise's own
    # level, on a 30 Hz onset in 2 % noise: picks moved by up to 7 samples).
    result.filter = initIirPlan(T, spec, dtMs, samples, ipMinimum,
      if result.whole: trLinear else: trNone)
  # Before the window: the attribute's windows at its first column (the STA
  # and the LTA before it; the energy ratio's "before" window) or the
  # confidence's noise window behind its guard gap, whichever reaches
  # further, and the history the causal filter needs to settle. After it:
  # the "after" window or the confidence's signal window.
  let (w, snr) = (result.w, result.snr)
  result.before = max(snr + w div 2, if p.pickMethod == pmStaLta: w + result.lta else: w) +
    (if result.filtered: result.filter.requiredMargin else: 0)
  result.after = max(w, snr)

func params*[T](plan: FirstBreakPlan[T]): lent FirstBreakParams {.inline, raises: [].} =
  ## What the plan was built from.
  plan.params

func dtMs*[T](plan: FirstBreakPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval the plan was built for.
  plan.dtMs

func samples*[T](plan: FirstBreakPlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  plan.samples

func requiredMargin*[T](plan: FirstBreakPlan[T]): int {.raises: [].} =
  ## The samples a block must hold on each side of a row's search window
  ## for the row to pick as the whole trace does, bit for bit (see "What a
  ## pick reads" in the module documentation; fewer are needed after the
  ## window than before it, and the trace's own ends cut the margin):
  ## `decodeWindow(searchColumns, plan.requiredMargin, traceLen)` is such a
  ## zoom window. `plan.samples` for a plan that reads whole traces (no
  ## search window, or `pmCoppens`): no window of the trace picks as all of
  ## it does.
  if plan.whole: plan.samples else: plan.before

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
  # positive. `a` is n float64 of scratch (STA/LTA, Coppens), filled only
  # where the rise reads it.
  result = -1
  var best = 0.0
  let s = max(1, w div 2) # the steps a rise is taken over
  let lastRead = min(n - 1, hi - 2 + s)
  template energy(a0, b0: int): float64 = p[b0] - p[a0] # samples a0 ..< b0
  template risePick(i0: int) =
    # The largest mean rise of a[] over the s steps from i, i in i0 ..< hi
    # (a[] is filled over i0 − 1 .. lastRead).
    for i in i0 ..< hi:
      let j = min(n - 1, i - 1 + s)
      if j > i - 1:
        let r = (a[j] - a[i - 1]) / float64(j - i + 1)
        if r > best:
          best = r
          result = i
  case m
  of pmStaLta:
    # Trailing STA (w samples ending at i) over the LTA before it; a[i] is
    # 0 until the LTA holds w / 2 samples (an LTA of a few samples is noise).
    let i0 = max(lo, w + s)
    for i in i0 - 1 .. lastRead:
      let s0 = max(0, i + 1 - w)
      let m0 = max(0, s0 - lta)
      a[i] = if s0 - m0 < s: 0.0
             else: (energy(s0, i + 1) / float64(i + 1 - s0)) /
               (energy(m0, s0) / float64(s0 - m0) + eps)
    risePick(i0)
  of pmEnergyRatio:
    let q = max(1, w div 4)
    for i in lo ..< hi:
      let k = min(w, min(i, n - i)) # equal windows, cut by both ends
      if k >= s:
        # MER: the ratio weighted by the amplitude at i (cubed in the paper;
        # the maximum is the same), here the RMS of q samples from i, so one
        # noise sample does not set it.
        let r = energy(i, i + k) / (energy(i - k, i) + eps * float64(k)) *
          sqrt(energy(i, min(n, i + q)))
        if r > best:
          best = r
          result = i
  of pmCoppens:
    # CR(i): the w samples ending at i over the energy up to i. Over the
    # first w samples the numerator window is cut by the trace's start, so
    # the ratio climbs from 0 to about 1 whatever the data: no pick there
    # (in noise a tenth of the onset, 81 of 200 traces picked it).
    let i0 = max(lo, w)
    for i in i0 - 1 .. lastRead:
      a[i] = energy(max(0, i + 1 - w), i + 1) / (p[i + 1] + beta + eps)
    risePick(i0)

{.pop.}

type
  Stretch = object
    # What the pick of one row reads: `len` columns from the row's column
    # `first` (0: no pick), with the search window `lo ..< hi` among them.
    first, len, lo, hi: int
    originMs: float64 # The time of the row's column 0; NaN without a time axis.

proc stretchOf[T](plan: FirstBreakPlan[T], src: ptr UncheckedArray[T], n: int,
    offset, t0Ms, startCol: float64): Stretch {.raises: [].} =
  # The stretch of one trace `src` of n live samples whose column 0 is at
  # `t0Ms + startCol·dt`.
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
  # Time of column 0 (NaN with a NaN or infinite t0: then no time window
  # applies, and the pick has a column but no time).
  let origin = if (t0Ms + startCol * dt).classify in {fcInf, fcNegInf}: NaN
               else: t0Ms + startCol * dt
  if origin.isNaN and (tLo > -Inf or tHi < Inf): return
  let cLo = if origin.isNaN: -Inf else: (tLo - origin) / dt
  let cHi = if origin.isNaN: Inf else: (tHi - origin) / dt
  # The zeros before the first and after the last non-zero sample (a
  # recording delay, padding, a mute) are not data: picked as it is, such a
  # row gives the step from zeros to data, with the zeros as its noise.
  let live = liveSpan(src.toOpenArray(0, n - 1))
  if live.len < 2: return
  let lo = max(live.a, if cLo <= 0: 0 elif cLo >= float64(n): n else: int(ceil(cLo)))
  let hi = min(live.b + 1, if cHi < 0: 0 elif cHi >= float64(n - 1): n
    else: int(floor(cHi)) + 1)
  if lo >= hi: return
  let a = if plan.whole: live.a else: max(live.a, lo - plan.before)
  let b = if plan.whole: live.b + 1 else: min(live.b + 1, hi + plan.after)
  Stretch(first: a, len: b - a, lo: lo - a, hi: hi - a, originMs: origin)

proc pickStretch[T](plan: FirstBreakPlan[T], x: ptr UncheckedArray[T], s: Stretch,
    ws: var Workspace): FirstBreakPick {.raises: [].} =
  # The pick on the stretch `s`, whose samples (band-passed, when the plan
  # has a filter) are `x[0 ..< s.len]`.
  result = noPick()
  let pr = plan.params
  let m = s.len
  let base = plan.energyBase
  let p = ws.scratch(float64, m + 1, base)
  let a = ws.scratch(float64, m + 1, base + plan.samples + 1)
  prefixEnergy(x, m, p)
  let total = p[m]
  if not (total > 0) or total == Inf: return
  let mean = total / float64(m)
  let eps = 1e-6 * mean
  let beta = pr.stabilization * mean * float64(plan.w)
  let col = argmaxAttribute(pr.pickMethod, p, a, m, s.lo, s.hi, plan.w, plan.lta, beta, eps)
  if col < 0: return
  let gapEnd = max(0, col - plan.w div 2) # guard gap: a pick a little late
  result.col = s.first + col
  result.timeMs = s.originMs + float64(s.first + col) * plan.dtMs
  # Nothing before the gap (an empty noise window), or no energy at all
  # there (a dropout: zeros are no noise level, and with the stabilization
  # alone they would read +60 dB): NaN, confidence 0.
  let noise = meanPower(x.toOpenArray(0, m - 1), max(0, gapEnd - plan.snr) ..< gapEnd)
  if noise > 0:
    let snr = powerSnrDb(meanPower(x.toOpenArray(0, m - 1), col ..< min(m, col + plan.snr)),
      noise, seRatio, floorDb = -Inf, stabilization = eps)
    if not snr.isNaN:
      result.snrDb = snr
      result.confidence = clamp((snr - pr.lowDb) / (pr.highDb - pr.lowDb), 0.0, 1.0)

proc pick*[T: SampleType](plan: FirstBreakPlan[T], x: openArray[T],
    ws: var Workspace, offset = NaN, t0Ms = 0.0): FirstBreakPick {.raises: [].} =
  ## The pick on one trace `x` (its live samples, at most `plan.samples`;
  ## an `AssertionDefect` otherwise) whose sample 0 is at `t0Ms`, with
  ## source–receiver `offset` for a guided window. `ws` must hold
  ## `plan.workspaceSize`. `x` is not changed.
  doAssert x.len <= plan.samples, "a trace longer than the plan's samples"
  if x.len == 0: return noPick()
  let src = cast[ptr UncheckedArray[T]](addr x[0])
  let s = plan.stretchOf(src, x.len, offset, t0Ms, 0)
  if s.len == 0: return noPick()
  var y = cast[ptr UncheckedArray[T]](addr src[s.first])
  if plan.filtered:
    let c = ws.scratch(T, s.len, plan.copyBase)
    copyMem(c, y, s.len * sizeof(T))
    applyFilter(c.toOpenArray(0, s.len - 1), plan.filter, ws)
    y = c
  plan.pickStretch(y, s, ws)

proc checkPick*[T: SampleType](plan: FirstBreakPlan[T], b: RowBlock[T], meta: ChunkMeta,
    picks: int) =
  ## The checks of `pickInto` on `b`, `meta` and the length `picks` of its
  ## `dst`, without picking: raises `ValueError` when `b` does not match
  ## `meta`, has more columns than `plan.samples`, its interval is not the
  ## plan's (beyond a relative 1e-9), or `picks` is not `meta.len`.
  meta.checkShape(b)
  if b.cols > plan.samples:
    raise newException(ValueError, "the block has " & $b.cols & " columns, the plan " &
      "takes " & $plan.samples)
  if not sameInterval(meta.dtMs, plan.dtMs):
    raise newException(ValueError, "the block's interval " & $meta.dtMs &
      " ms is not the plan's " & $plan.dtMs & " ms")
  if picks != meta.len:
    raise newException(ValueError, "dst holds " & $picks & " picks, the meta " &
      $meta.len & " rows")

proc pickInto*[T: SampleType](plan: FirstBreakPlan[T], b: RowBlock[T], meta: ChunkMeta,
    ws: var Workspace, dst: var openArray[FirstBreakPick]) =
  ## The pick of every live row of `b` into `dst[row]` (each row to its live
  ## length, on `meta`'s time axis; non-live rows get `noPick`). `ws` must
  ## hold `plan.workspaceSize`. Raises `ValueError` as `checkPick`.
  plan.checkPick(b, meta, dst.len)
  template stretchOfRow(row: int): Stretch =
    let k = meta.keys(row)
    plan.stretchOf(b.rowPtr(row), meta.liveSamples(row, b.cols), k.offset, k.t0Ms,
      float64(meta.firstSample))
  if not plan.filtered or b.cols == 0:
    for row in 0 ..< meta.len:
      dst[row] = noPick()
      if meta.isLive(row):
        let s = stretchOfRow(row)
        if s.len > 0:
          dst[row] = plan.pickStretch(cast[ptr UncheckedArray[T]](addr b.rowPtr(row)[s.first]),
            s, ws)
    return
  # Band-pass the stretches of `fftLanes` rows at a time, as one batch of
  # the IIR filter's lanes, in copies (each at the start of its lane's
  # row), then pick each copy. The filter takes its rows' lengths from a
  # meta: one of the lanes, whose sample counts are the stretches'.
  let c = ws.scratch(T, fftLanes * b.cols, plan.copyBase)
  var stretches: array[fftLanes, Stretch]
  var rowOf, counts: array[fftLanes, int]
  var keys: array[fftLanes, TraceKeys]
  for k in keys.mitems:
    k = initTraceKeys()
    k.status = tsLive
  let lanes = initChunkMeta(rowsOf(keys, counts), plan.dtMs)
  var used = 0
  template flush() =
    if used > 0:
      let cb = rowBlock(c.toOpenArray(0, used * b.cols - 1), used, b.cols)
      applyFilter(cb, lanes.sub(0 ..< used), plan.filter, ws)
      for l in 0 ..< used:
        dst[rowOf[l]] = plan.pickStretch(cb.rowPtr(l), stretches[l], ws)
      used = 0
  for row in 0 ..< meta.len:
    dst[row] = noPick()
    if meta.isLive(row):
      let s = stretchOfRow(row)
      if s.len > 0:
        copyMem(addr c[used * b.cols], addr b.rowPtr(row)[s.first], s.len * sizeof(T))
        stretches[used] = s
        rowOf[used] = row
        counts[used] = s.len
        inc used
        if used == fftLanes: flush()
  flush()

proc pickInto*[T: SampleType](plan: FirstBreakPlan[T], samples: Tensor[T],
    meta: ChunkMeta, ws: var Workspace, dst: var openArray[FirstBreakPick]) =
  ## `pickInto` on a row-major `[rows, samples]` tensor.
  plan.pickInto(rowBlock(samples), meta, ws, dst)
