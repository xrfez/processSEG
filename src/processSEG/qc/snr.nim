## Signal-to-noise ratios, in dB: per trace from the power in a signal
## window over the power in a noise window (`windowSnrDb`, `windowSnrInto`),
## and per gather from the coherence of flattened traces, sample by sample
## (`coherenceSnrInto`).
##
## ```nim
## # Per trace: 0 .. 40 ms after each pick over 60 .. 20 ms before it.
## let p = WindowSnrParams(signalStartMs: 0, signalEndMs: 40,
##   noiseStartMs: -60, noiseEndMs: -20, estimator: seExcess)
## var perTrace = newSeq[float64](meta.len)
## windowSnrInto(samples, meta, p, pickTimesMs, perTrace)  # or a `Moveout` guide
## # Per gather, versus time, on NMO- (or LMO-) corrected traces:
## var ws = initWorkspace()
## ws.reserve(coherenceSnrWorkspace(samples.shape[1]))
## var perSample = newSeq[float64](samples.shape[1])
## coherenceSnrInto(samples, meta, windowMs = 100, perSample, ws)
## ```
##
## **The SNR** is the signal power over the noise power, each a mean power
## per sample of one trace (so windows and gathers of different sizes
## compare), in dB: `10·log₁₀(S / N)`.
##
## **Windowed SNR.** For one trace, with `Ps` and `Pn` the mean of `x²`
## over the signal and the noise window:
##
##     seRatio:   Ps / Pn          (signal + noise over noise)
##     seExcess:  (Ps − Pn) / Pn   (signal over noise)
##
## The signal window holds signal and noise, so `seRatio` measures
## `(S + N) / N = SNR + 1`: noise alone gives about 0 dB, but a weak signal
## is lost under that 1 (a true −10 dB reads +0.4 dB). `seExcess` subtracts
## the noise power the noise window measures, so it estimates `S / N`
## without bias for additive noise of the same power in both windows; it is
## right from −10 dB up, but noise alone gives a ratio near 0, often
## negative: then the floor (below). `seRatio` is the classic energy ratio
## (and `qc/firstbreak`'s confidence); `seExcess` the one to read as an
## SNR. Its spread: at an SNR `r` and windows of `L` samples each, the
## standard deviation of `Ps − Pn` is about `2·Pn / √L`, so a ±1 dB
## estimate at −10 dB needs some 10⁵ samples per window, at 0 dB some 10³,
## at +20 dB a few tens.
##
## **Windows.** The windows are times in ms, `[startMs, endMs]` (both
## ends included: the samples whose time falls in it), on the trace's time
## axis (`ChunkMeta.sampleTimeMs`), or relative to a time per row: picks
## (`refMs`, such as `FirstBreakPick.timeMs`) or a linear moveout guide (a
## `Moveout`: its time at each row's offset). A window is cut by the
## trace's live samples. They may overlap; usually the noise window lies
## before the first break and the signal window after it.
##
## **Coherence SNR.** For `N` flattened traces `xᵢ(t) = s(t) + nᵢ(t)` (the
## same signal on each, noise independent from trace to trace, of power
## `σ²`), at each sample `t`:
##
##     mean      m(t) = Σᵢ xᵢ(t) / N
##     coherent  C(t) = N · m(t)²            E[C] = N·s² + σ²
##     residual  R(t) = Σᵢ (xᵢ(t) − m(t))²   E[R] = (N − 1)·σ²
##
## (`C + R = E`, the total energy `Σᵢ xᵢ²`; `C / E` is the semblance). Over
## the window `W` around each output sample:
##
##     noise   σ̂² = Σ_W R / Σ_W (N − 1)
##     signal  Ŝ  = (Σ_W C − |W|·σ̂²) / Σ_W N
##     SNR        = Ŝ / σ̂²
##
## Both estimates are unbiased, so the ratio of their means is the true
## signal power over the noise power, per sample of one trace (the stack's
## SNR is `N` times that). With `N` constant this is the semblance
## relation: `SNR = (N·β − 1) / (N·(1 − β))`, `β = Σ_W C / Σ_W E`. The
## residual is summed as squared deviations from the mean (two passes),
## so the noise estimate keeps its precision at high SNR. Its spread: at
## an SNR `r`, `N` traces and a window of `L` samples, the relative
## standard deviation of `Ŝ` is about `2·√(1 / (r·N·L))` (with noise alone
## `Ŝ / σ̂²` spreads by about `√(2/L) / N`, below −20 dB for `N = 24`,
## `L = 41`, so noise alone reads at most −10 dB).
##
## What it needs: the signal must be flat (NMO or LMO corrected,
## `applyNmo`, `applyLmo`) and of the same amplitude on every trace;
## residual moveout, amplitude variation and coherent noise count as noise.
##
## **Per gather.** `coherenceSnrInto` takes one gather (a `ChunkMeta.sub`
## of its rows). It counts, at each column, the live rows whose
## `liveSpan` (first to last non-zero sample, `stack/stack`) holds it, so
## mutes and zero padding are not data; zeros inside it count. The window
## is `windowMs` rounded to an odd number of samples centred on the output
## sample, cut by the block's ends. Cost: rows·cols twice, plus cols,
## whatever the window (its sums come from block sums, not from a loop
## over each window).
##
## **The floor.** An estimate of no signal (`Ŝ ≤ 0`, `Ps ≤ Pn` with
## `seExcess`, or a zero signal power) has no dB value: it returns
## `floorDb` (default `DefaultSnrFloorDb`, −40 dB), and every result is
## clamped up to it, so plots of SNR against time keep a finite scale.
## `floorDb = -Inf` turns the clamp off (no signal is then −Inf).
##
## **NaN** means the SNR cannot be measured: a non-live row, a NaN pick,
## offset or `t0Ms`, an empty window, a zero noise power (a muted noise
## window), fewer than two traces at every sample of a coherence window,
## or a NaN in the data. Noise-free identical traces give +Inf.
##
## Nothing here changes the samples; nothing raises on data.

import std/math
import parseSEG
import ../core/[rows, types, workspace]
import ../stack/stack
import ../velocity/moveout

type
  SnrEstimator* = enum
    ## How a windowed SNR is read from the two mean powers (see the module
    ## documentation).
    seRatio ## `Ps / Pn`: (signal + noise) over noise; noise alone ≈ 0 dB.
    seExcess ## `(Ps − Pn) / Pn`: signal over noise, unbiased.

  WindowSnrParams* = object
    ## The windows of a windowed SNR, in ms: times on the trace axis, or
    ## relative to a pick or a guide. Both ends are included. The defaults
    ## suit a pick: 0 .. 40 ms after it over 60 .. 20 ms before it.
    signalStartMs*: float64 = 0
    signalEndMs*: float64 = 40
    noiseStartMs*: float64 = -60
    noiseEndMs*: float64 = -20
    estimator*: SnrEstimator = seRatio
    floorDb*: float64 = -40 ## Returned for no signal; results are clamped up to it.

const
  DefaultSnrFloorDb* = -40.0
    ## The default floor of the SNRs, in dB.

proc badSnr(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func finite(x: float64): bool {.inline.} =
  x.classify notin {fcNan, fcInf, fcNegInf}

proc checkFloor(floorDb: float64) =
  if floorDb.isNaN or floorDb == Inf:
    badSnr("floorDb must be below +Inf and not NaN, got " & $floorDb)

proc checkParams(p: WindowSnrParams) =
  for (a, b, what) in [(p.signalStartMs, p.signalEndMs, "signal"),
      (p.noiseStartMs, p.noiseEndMs, "noise")]:
    if not (finite(a) and finite(b)):
      badSnr("the " & what & " window must have finite ends, got " & $a & " .. " & $b)
    if b < a:
      badSnr("the " & what & " window ends before it starts: " & $a & " .. " & $b)
  checkFloor(p.floorDb)

# ----------------------------------------------------------------- kernels --

func meanPower*[T: SampleType](x: openArray[T], cols: Slice[int]): float64 {.raises: [].} =
  ## The mean of `x[j]²` over the columns `cols` within `0 ..< x.len`, summed
  ## in float64; NaN when none is.
  let a = max(cols.a, 0)
  let b = min(cols.b, x.high)
  if a > b: return NaN
  var s = 0.0
  for j in a .. b:
    let v = float64(x[j])
    s += v * v
  s / float64(b - a + 1)

func powerSnrDb*(signal, noise: float64, estimator = seRatio,
    floorDb = DefaultSnrFloorDb, stabilization = 0.0): float64 {.raises: [].} =
  ## The SNR in dB from the mean powers of the signal and the noise window
  ## (`meanPower`), each plus `stabilization` (≥ 0: keeps a silent window
  ## from dividing by zero), read by `estimator`. NaN when either power is
  ## NaN or the noise power is not positive; `floorDb` for a ratio ≤ 0 (no
  ## signal), and every result clamped up to it.
  let s = signal + stabilization
  let n = noise + stabilization
  if s.isNaN or not (n > 0): return NaN
  let r = case estimator
    of seRatio: s / n
    of seExcess: (s - n) / n
  if r.isNaN: NaN
  elif r > 0: max(10 * log10(r), floorDb)
  else: floorDb

func windowSnrDb*[T: SampleType](x: openArray[T], signal, noise: Slice[int],
    estimator = seRatio, floorDb = DefaultSnrFloorDb,
    stabilization = 0.0): float64 {.raises: [].} =
  ## The SNR of one trace, in dB: `powerSnrDb` of the mean powers over the
  ## columns `signal` and `noise` (cut to `0 ..< x.len`; an empty window
  ## gives NaN).
  powerSnrDb(meanPower(x, signal), meanPower(x, noise), estimator, floorDb,
    stabilization)

func windowCols(startMs, endMs, originMs, dtMs: float64, n: int): Slice[int] {.inline.} =
  # The columns c in 0 ..< n whose time originMs + c·dtMs lies in
  # [startMs, endMs] (a 1e-9 column tolerance for rounding); empty for none
  # or NaN.
  const tol = 1e-9
  let lo = max(ceil((startMs - originMs) / dtMs - tol), 0.0)
  let hi = min(floor((endMs - originMs) / dtMs + tol), float64(n - 1))
  if not (lo <= hi): 1 .. 0 else: int(lo) .. int(hi)

# ------------------------------------------------------- windowed, blocks --

type RefKind = enum rkFixed, rkTimes, rkGuide

proc windowSnrImpl[T](b: RowBlock[T], meta: ChunkMeta, p: WindowSnrParams,
    kind: RefKind, refMs: openArray[float64], guide: Moveout,
    dst: var openArray[float64]) =
  meta.checkShape(b)
  checkParams(p)
  if dst.len != meta.len:
    badSnr("dst holds " & $dst.len & " values, the meta " & $meta.len & " rows")
  if kind == rkTimes and refMs.len != meta.len:
    badSnr("refMs holds " & $refMs.len & " times, the meta " & $meta.len & " rows")
  if kind == rkGuide and not (guide.velocity > 0):
    badSnr("the guide is not a valid Moveout (build it with linearMoveout)")
  for r in 0 ..< meta.len:
    dst[r] = NaN
    if meta.isLive(r):
      let n = meta.liveSamples(r, b.cols)
      let origin = meta.sampleTimeMs(r, 0)
      let t = case kind
        of rkFixed: 0.0
        of rkTimes: refMs[r]
        of rkGuide: guide.timeMs(meta.keys(r).offset)
      if n > 0 and finite(origin) and finite(t):
        let sig = windowCols(t + p.signalStartMs, t + p.signalEndMs, origin, meta.dtMs, n)
        let noi = windowCols(t + p.noiseStartMs, t + p.noiseEndMs, origin, meta.dtMs, n)
        dst[r] = windowSnrDb(b.row(r, n), sig, noi, p.estimator, p.floorDb)

proc windowSnrInto*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    params: WindowSnrParams, dst: var openArray[float64]) =
  ## The windowed SNR of every row of `b` into `dst[row]`, in dB, with the
  ## windows at fixed times on the trace axis (`t0Ms` based); NaN for
  ## non-live rows and as the module documentation lists. Raises
  ## `ValueError` when a window is not finite or ends before it starts,
  ## `floorDb` is NaN or +Inf, `b` does not match `meta`, or `dst` is not
  ## `meta.len` long. Allocates nothing.
  windowSnrImpl(b, meta, params, rkFixed, [], Moveout(), dst)

proc windowSnrInto*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    params: WindowSnrParams, refMs: openArray[float64], dst: var openArray[float64]) =
  ## `windowSnrInto` with the windows relative to `refMs[row]`, a time on
  ## the trace axis per row (a pick: `FirstBreakPick.timeMs`); a NaN time
  ## gives NaN. Raises `ValueError` also when `refMs` is not `meta.len` long.
  windowSnrImpl(b, meta, params, rkTimes, refMs, Moveout(), dst)

proc windowSnrInto*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    params: WindowSnrParams, guide: Moveout, dst: var openArray[float64]) =
  ## `windowSnrInto` with the windows relative to the moveout `guide` at
  ## each row's offset (`linearMoveout(velocity, t0Ms)`: the LMO guide of
  ## the first breaks); a NaN offset gives NaN. Raises `ValueError` also for
  ## a default (unbuilt) `Moveout`.
  windowSnrImpl(b, meta, params, rkGuide, [], guide, dst)

proc windowSnrInto*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    params: WindowSnrParams, dst: var openArray[float64]) =
  ## `windowSnrInto` on a row-major `[rows, samples]` tensor.
  windowSnrInto(rowBlock(samples), meta, params, dst)

proc windowSnrInto*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    params: WindowSnrParams, refMs: openArray[float64], dst: var openArray[float64]) =
  ## `windowSnrInto` relative to per-row times, on a tensor.
  windowSnrInto(rowBlock(samples), meta, params, refMs, dst)

proc windowSnrInto*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    params: WindowSnrParams, guide: Moveout, dst: var openArray[float64]) =
  ## `windowSnrInto` relative to a moveout guide, on a tensor.
  windowSnrInto(rowBlock(samples), meta, params, guide, dst)

# --------------------------------------------------------------- coherence --

func coherenceSnrWorkspace*(cols: int): WorkspaceSize {.raises: [].} =
  ## The scratch of `coherenceSnrInto` on a block `cols` samples wide:
  ## `6·cols + 2` float64 (the columns' sums and their window sums).
  WorkspaceSize(f64: 6 * max(cols, 0) + 2)

type F64 = ptr UncheckedArray[float64]

{.push boundChecks: off, overflowChecks: off.}

proc blockSums(v, p: F64, n, w: int) {.raises: [].} =
  ## Turns `v[0 ..< n]` into sums over blocks of `w`: `v[k]` becomes the
  ## sum from `k` through the end of its block (or `n − 1`), `p[k]` the sum
  ## from the block's start to before `k`, and `p[n]` closes the last
  ## block. A window is then one or two of these added, never a
  ## difference: a quiet window after a loud one keeps its precision, and a
  ## NaN stays in the windows that hold it.
  var bs = 0
  var fwd = 0.0
  while bs < n:
    let be = min(bs + w, n)
    fwd = 0
    for k in bs ..< be:
      p[k] = fwd
      fwd += v[k]
    var bwd = 0.0
    for k in countdown(be - 1, bs):
      bwd += v[k]
      v[k] = bwd
    bs = be
  p[n] = (if n mod w == 0: 0.0 else: fwd)

func windowSum(s, p: F64, n, h, j: int): float64 {.inline, raises: [].} =
  ## The sum over `max(0, j − h) .. min(n − 1, j + h)` of what
  ## `blockSums(v, p, n, 2·h + 1)` made `s` and `p` of.
  let a = j - h
  let b = j + h
  if a >= 0 and b < n: s[a] + p[b + 1] # a whole window
  elif a <= 0: p[min(b, n - 1) + 1] # from the start: inside the first block
  else:
    # To the end: the rest of a's block and, if another follows, the last
    # block, whole.
    let last = (n - 1) div (2 * h + 1) * (2 * h + 1)
    if a < last: s[a] + s[last] else: s[a]

{.pop.}

func coherenceDb(residual, coherent, traces, columns, floorDb: float64): float64 {.
    inline, raises: [].} =
  # The SNR of one window from Σ R, Σ C, Σ N and |W| (columns with N ≥ 1).
  let dof = traces - columns
  if not (dof > 0): return NaN
  let noise = residual / dof
  let signal = (coherent - columns * noise) / traces
  if noise.isNaN or signal.isNaN: NaN
  elif noise == 0: (if signal > 0: Inf else: NaN)
  else:
    let r = signal / noise
    if r.isNaN: NaN
    elif r > 0: max(10 * log10(r), floorDb)
    else: floorDb

proc coherenceSnrInto*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    windowMs: float64, dst: var openArray[float64], ws: var Workspace,
    floorDb = DefaultSnrFloorDb) =
  ## The coherence SNR of the gather `b` (flattened: NMO or LMO corrected)
  ## at each column, into `dst[col]`, in dB: the signal power over the
  ## noise power per trace, estimated over `windowMs` centred on the column
  ## from the live rows (see the module documentation for the estimator,
  ## the floor and NaN). `ws` must hold `coherenceSnrWorkspace(b.cols)`.
  ## Raises `ValueError` when `windowMs` is not finite and positive,
  ## `floorDb` is NaN or +Inf, `b` does not match `meta`, `dst` is not
  ## `b.cols` long, or the workspace is too small. Allocates nothing; the
  ## time does not depend on the window.
  meta.checkShape(b)
  if not (finite(windowMs) and windowMs > 0):
    badSnr("windowMs must be finite and positive, got " & $windowMs)
  checkFloor(floorDb)
  if dst.len != b.cols:
    badSnr("dst holds " & $dst.len & " values, the block " & $b.cols & " columns")
  let cols = b.cols
  if cols == 0: return
  if not ws.fits(coherenceSnrWorkspace(cols)):
    badSnr("the workspace holds " & $ws.capacity & "; reserve coherenceSnrWorkspace(" &
      $cols & ")")
  let mean = ws.scratch(float64, cols, 0) # Σ x, then the mean, then C
  let count = ws.scratch(float64, cols, cols) # N, then its running sum
  let resid = ws.scratch(float64, cols, 2 * cols)
  let live = ws.scratch(float64, cols, 3 * cols) # the running count of columns with N > 0
  let meanP = ws.scratch(float64, cols + 1, 4 * cols)
  let residP = ws.scratch(float64, cols + 1, 5 * cols + 1)
  for j in 0 ..< cols:
    mean[j] = 0
    count[j] = 0
    resid[j] = 0
  {.push boundChecks: off, overflowChecks: off.}
  for r in 0 ..< meta.len:
    if meta.isLive(r):
      let x = b.rowPtr(r)
      let span = liveSpan(b.row(r, meta.liveSamples(r, cols)))
      for j in span:
        mean[j] += float64(x[j])
        count[j] += 1
  for j in 0 ..< cols:
    if count[j] > 0: mean[j] /= count[j]
  for r in 0 ..< meta.len:
    if meta.isLive(r):
      let x = b.rowPtr(r)
      let span = liveSpan(b.row(r, meta.liveSamples(r, cols)))
      for j in span:
        let d = float64(x[j]) - mean[j]
        resid[j] += d * d
  for j in 0 ..< cols:
    mean[j] = count[j] * mean[j] * mean[j]
  # The four window sums of each column, in a time that does not grow with
  # the window (it was a loop over the window per column: 21 ms for a 12 s
  # window on 6000 columns). R and C by block sums; N and the columns with
  # N > 0 are whole numbers, exact as differences of running sums. A column
  # without a live row has R = C = N = 0 and adds to none.
  let h = int(min(round(windowMs / meta.dtMs / 2), float64(cols)))
  blockSums(resid, residP, cols, 2 * h + 1)
  blockSums(mean, meanP, cols, 2 * h + 1)
  var n, w = 0.0
  for j in 0 ..< cols:
    w += float64(count[j] > 0)
    n += count[j]
    live[j] = w
    count[j] = n
  for j in 0 ..< cols:
    let lo = max(0, j - h)
    let hi = min(cols - 1, j + h)
    let before = (if lo > 0: (count[lo - 1], live[lo - 1]) else: (0.0, 0.0))
    dst[j] = coherenceDb(windowSum(resid, residP, cols, h, j), windowSum(mean, meanP, cols, h, j),
      count[hi] - before[0], live[hi] - before[1], floorDb)
  {.pop.}

proc coherenceSnrInto*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    windowMs: float64, dst: var openArray[float64], ws: var Workspace,
    floorDb = DefaultSnrFloorDb) =
  ## `coherenceSnrInto` on a row-major `[rows, samples]` tensor.
  coherenceSnrInto(rowBlock(samples), meta, windowMs, dst, ws, floorDb)
