## Vibroseis correlation (`CorrelationPlan`): uncorrelated records
## compressed with a pilot, by cross-correlation (the matched filter) or by
## sweep deconvolution, through the batched FFT.
##
## ```nim
## let pilot = generatePilot(linearSweep(8, 80, 8000, 2, initSweepTaper(250, 250)), float32)
## let plan = initCorrelationPlan(float32, pilot, dtMs = 2, recordSamples = 10001)
## var ws = initWorkspace()
## ws.reserve(plan.workspaceSize)
## var correlated = zeros[float32](raw.shape[0], plan.outSamples)   # 6001: 12 s listen
## plan.correlateInto(raw, correlated, meta, ws)
## ```
##
## **Output.** A new block, since the length changes: `outSamples` columns,
## by default the listen time, `recordSamples − pilot.len + 1` (the lags at
## which the whole pilot lies inside the record). Column `m` is at the
## record's time `t0Ms + m·dtMs`: the pilot's own start time (`Pilot.t0Ms`,
## possibly a fraction of a sample) is removed exactly, by a phase ramp in
## the frequency domain. Lags past the record's end correlate with the
## zeros beyond it (a truncated correlation), never with its start: the FFT
## is long enough (`nfft ≥ recordSamples + pilot.len + 64`) that nothing
## wraps.
##
## **Modes** (`CorrelationMode`), with `X` the record's spectrum and `P` the
## pilot's:
## - `cmCorrelate`: `X · conj(P)`, the cross-correlation. A reflection comes
##   out as the Klauder wavelet (the pilot's autocorrelation): zero phase,
##   with the amplitude spectrum `|P|²`;
## - `cmDeconvolve`: `X · conj(P) / (|P|² + ε)`, sweep deconvolution, with
##   `ε = epsPercent/100 · max |P|²`: a wavelet with a flat amplitude
##   spectrum where `|P|² ≫ ε` (the sweep's band), so sharper and with
##   smaller side lobes, at the cost of the noise outside it. With a
##   ground-force pilot it also removes the vibrator's own distortion.
##
## **Normalization.** The raw output scales with the pilot: by its energy
## `Σp²` for correlation. With `normalize` (the default) the output is
## divided by the peak of its own wavelet (`Σp²` for correlation), so a
## pilot-shaped arrival of amplitude `a` correlates to a wavelet of peak
## `a`, whatever the pilot's amplitude or length: records correlated with
## different sweeps, or with recorded pilots of varying level, compare. The
## factor is folded into the stored pilot spectrum and costs nothing.
##
## **Rows.** Every row except the dead (`tsDead`) and failed ones is
## correlated: live and unknown rows, and auxiliary ones (a recorded pilot
## correlates to the Klauder wavelet, a QC). Dead and failed rows give
## zeros. A record's live length (`liveSamples`) is what is correlated; the
## rest of the row is zero. The rows go through the batched FFT, `fftLanes`
## at a time (one record alone through the one-trace FFT, a third of the
## time of a batch); nothing allocates once the workspace is reserved. A
## NaN in a row makes that row NaN.

import std/[complex, math]
import parseSEG
import ../core/[rows, types, workspace]
import ../fft/plan
import pilot

type
  CorrelationMode* = enum
    ## How records are compressed with the pilot.
    cmCorrelate  ## cross-correlation, `X·conj(P)`
    cmDeconvolve ## sweep deconvolution, `X·conj(P) / (|P|² + ε)`

  CorrelationPlan*[T: SampleType] = object
    ## Correlation of records of up to `recordSamples` samples with one
    ## pilot; see the module documentation.
    fft: FftPlan[T]
    mulRe, mulIm: seq[T] # the multiplier per bin
    shift: int # whole samples of the pilot's start time
    recordSamples, outSamples, pilotSamples: int
    dtMs: float64
    mode: CorrelationMode
    scale: float64

  Buf[T] = ptr UncheckedArray[T]

proc badPlan(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc initCorrelationPlan*[T, P: SampleType](_: typedesc[T], pilot: Pilot[P], dtMs: float64,
    recordSamples: int, outSamples = 0, mode = cmCorrelate,
    epsPercent = 1.0, normalize = true): CorrelationPlan[T] =
  ## A plan correlating records of up to `recordSamples` samples at `dtMs`
  ## with `pilot`, into `outSamples` columns (0: the listen time,
  ## `recordSamples − pilot.len + 1`). Raises `ValueError` for a pilot
  ## `validate` rejects, a pilot sampled at another interval (resample it
  ## with `resamplePilot`), a pilot longer than the record (with the default
  ## `outSamples`), `outSamples < 0` or above `recordSamples`, or an
  ## `epsPercent` that is not finite and positive (deconvolution), or a
  ## pilot start time (`t0Ms`) so far from the time break that no lag
  ## reaches the output.
  pilot.validate
  if not (dtMs > 0 and dtMs < Inf): badPlan("dtMs must be finite and positive, got " & $dtMs)
  if not sameInterval(pilot.dtMs, dtMs):
    badPlan("the pilot is sampled at " & $pilot.dtMs & " ms, the records at " & $dtMs &
      " ms: resample the pilot first (resamplePilot)")
  if recordSamples < 1: badPlan("recordSamples must be ≥ 1, got " & $recordSamples)
  let outN = if outSamples == 0: recordSamples - pilot.len + 1 else: outSamples
  if outN < 1:
    badPlan("the pilot (" & $pilot.len & " samples) is longer than the record (" &
      $recordSamples & "): no listen time; pass outSamples")
  if outSamples < 0 or outN > recordSamples:
    badPlan("outSamples must be in 0 .. recordSamples, got " & $outSamples)
  if mode == cmDeconvolve and not (epsPercent > 0 and epsPercent < Inf):
    badPlan("epsPercent must be finite and positive, got " & $epsPercent)
  # The pilot's start time in samples: a whole part read as an offset, the
  # rest a phase ramp.
  let s = pilot.t0Ms / dtMs
  if abs(s) >= float64(recordSamples + pilot.len):
    badPlan("the pilot starts " & $pilot.t0Ms & " ms from the time break: no lag of a " &
      $recordSamples & "-sample record reaches the output")
  let whole = int(round(s))
  let frac = s - float64(whole)
  let n = nextFastLen(recordSamples + pilot.len + 64 + abs(whole))
  result.fft = initFftPlan(T, n)
  let bins = result.fft.bins
  # The pilot's spectrum, in float64.
  let pf = initFftPlan(float64, n)
  var ws = initWorkspace()
  ws.reserve(pf.workspaceSize(batched = false))
  var x = newSeq[float64](n)
  for j in 0 ..< pilot.len: x[j] = float64(pilot.samples[j])
  var pr, pi = newSeq[float64](bins)
  pf.forward(x, pr, pi, ws)
  var peakPow = 0.0
  for k in 0 ..< bins: peakPow = max(peakPow, pr[k] * pr[k] + pi[k] * pi[k])
  let eps = if mode == cmDeconvolve: epsPercent / 100 * peakPow else: 0.0
  # The wavelet's peak: (1/n)·Σ over the full spectrum of |P|²·W.
  var w = newSeq[float64](bins)
  var peak = 0.0
  for k in 0 ..< bins:
    let p2 = pr[k] * pr[k] + pi[k] * pi[k]
    w[k] = if mode == cmDeconvolve: 1 / (p2 + eps) else: 1.0
    let g = p2 * w[k]
    peak += (if k == 0 or (n mod 2 == 0 and k == bins - 1): g else: 2 * g)
  peak /= float64(n)
  let scale = if normalize: 1 / peak else: 1.0
  result.mulRe = newSeq[T](bins)
  result.mulIm = newSeq[T](bins)
  for k in 0 ..< bins:
    # conj(P)·W·scale·exp(+2πi·k·frac/n): reads the output frac samples later.
    let m = complex(pr[k], -pi[k]) * (w[k] * scale) *
      exp(complex(0.0, 2 * PI * float64(k) * frac / float64(n)))
    result.mulRe[k] = T(m.re)
    result.mulIm[k] = T(m.im)
  if n mod 2 == 0: result.mulIm[bins - 1] = T(0) # inverse takes Nyquist as real
  result.shift = whole
  result.recordSamples = recordSamples
  result.outSamples = outN
  result.pilotSamples = pilot.len
  result.dtMs = dtMs
  result.mode = mode
  result.scale = scale

func recordSamples*[T](plan: CorrelationPlan[T]): int {.inline, raises: [].} =
  ## The longest record the plan takes.
  plan.recordSamples

func outSamples*[T](plan: CorrelationPlan[T]): int {.inline, raises: [].} =
  ## The columns of the output.
  plan.outSamples

func pilotSamples*[T](plan: CorrelationPlan[T]): int {.inline, raises: [].} =
  ## The pilot's length.
  plan.pilotSamples

func dtMs*[T](plan: CorrelationPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval.
  plan.dtMs

func mode*[T](plan: CorrelationPlan[T]): CorrelationMode {.inline, raises: [].} =
  ## Correlation or deconvolution.
  plan.mode

func nfft*[T](plan: CorrelationPlan[T]): int {.inline, raises: [].} =
  ## The FFT length.
  plan.fft.len

func scale*[T](plan: CorrelationPlan[T]): float64 {.inline, raises: [].} =
  ## The normalization factor applied (1 without `normalize`).
  plan.scale

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func workspaceSize*[T](plan: CorrelationPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `correlateInto` needs.
  let f = plan.fft.workspaceSize
  let extra = fftLanes * (plan.fft.len + 2 * plan.fft.bins)
  when T is float32: f + WorkspaceSize(f32: extra)
  else: f + WorkspaceSize(f64: extra)

proc checkPlan[T](plan: CorrelationPlan[T]) {.inline.} =
  doAssert plan.recordSamples > 0, "a CorrelationPlan must come from initCorrelationPlan"

{.push boundChecks: off, overflowChecks: off.}

proc mulKernel[T](re {.noalias.}, im {.noalias.}: Buf[T], mr, mi: T) {.noinline.} =
  ## One bin of `fftLanes` spectra times the plan's multiplier there. Not
  ## inlined: on its own the loop is one vector operation, while inside the
  ## loop over the bins the compiler vectorized across bins, shuffling the
  ## lanes in and out (17.3 against 4.1 ms for 250 batches of 8101 bins).
  for l in 0 ..< fftLanes:
    let a = re[l]
    let b = im[l]
    re[l] = a * mr - b * mi
    im[l] = a * mi + b * mr

proc mulSingle[T](re {.noalias.}, im {.noalias.}: Buf[T], mr, mi: openArray[T]) {.
    raises: [].} =
  ## One trace's spectrum times the plan's multiplier, bin by bin.
  for k in 0 ..< mr.len:
    let a = re[k]
    let b = im[k]
    re[k] = a * mr[k] - b * mi[k]
    im[k] = a * mi[k] + b * mr[k]

template runLanes(plan, x, re, im, ws: untyped) =
  block:
    let n = plan.fft.len
    let bins = plan.fft.bins
    plan.fft.forwardLanes(x.toOpenArray(0, n * fftLanes - 1),
      re.toOpenArray(0, bins * fftLanes - 1), im.toOpenArray(0, bins * fftLanes - 1), ws)
    for k in 0 ..< bins:
      mulKernel(cast[typeof(re)](addr re[k * fftLanes]), cast[typeof(im)](addr im[k * fftLanes]),
        plan.mulRe[k], plan.mulIm[k])
    plan.fft.inverseLanes(re.toOpenArray(0, bins * fftLanes - 1),
      im.toOpenArray(0, bins * fftLanes - 1), x.toOpenArray(0, n * fftLanes - 1), ws)

proc gatherLanes[T](buf: Buf[T], n: int, rows: array[fftLanes, Buf[T]],
    live: array[fftLanes, int], used: int) {.raises: [].} =
  ## The lane buffer of `used` records (`rows[l][0 ..< live[l]]`, zeros
  ## after, to `n`; unused lanes zero): written sample by sample, one pass
  ## over the buffer with a read stream per record, and zeroed only past
  ## the longest record. (Lane by lane after zeroing all of it, the buffer
  ## was passed over nine times: 9.5 against 3.4 ms on 2000 records of
  ## 10001 samples at nfft 16200.)
  var shortest = (if used == fftLanes: int.high else: 0)
  var longest = 0
  for l in 0 ..< used:
    shortest = min(shortest, live[l])
    longest = max(longest, live[l])
  # Locals: through the parameters every store to `buf` would reload them.
  let p = rows
  let len = live
  for j in 0 ..< shortest: # every lane holds a record sample
    for l in 0 ..< fftLanes: buf[j * fftLanes + l] = p[l][j]
  for j in shortest ..< longest:
    for l in 0 ..< fftLanes:
      buf[j * fftLanes + l] = (if j < len[l]: p[l][j] else: T(0))
  if n > longest:
    zeroMem(addr buf[longest * fftLanes], (n - longest) * fftLanes * sizeof(T))

type LagSpans = object
  ## Output columns `0 ..< wrapped` are zero (lags at or before −pilot),
  ## `wrapped ..< direct` hold the negative lags (circular index lag + nfft),
  ## `direct ..< last` the lags from 0 (index lag), the rest are zero.
  wrapped, direct, last: int

func lagSpans(plan: CorrelationPlan, cols: int): LagSpans {.inline.} =
  result.wrapped = clamp(1 - plan.pilotSamples - plan.shift, 0, cols)
  result.direct = clamp(-plan.shift, result.wrapped, cols)
  result.last = clamp(plan.recordSamples - plan.shift, result.direct, cols)

proc scatterLanes[T](plan: CorrelationPlan[T], buf: Buf[T], rows: array[fftLanes, Buf[T]],
    used, cols: int) {.raises: [].} =
  ## The lags of `used` lanes out of the buffer into their output rows,
  ## sample by sample (one pass over the buffer, as the gather: 3.1 against
  ## 6.9 ms lane by lane on 2000 records).
  let s = plan.lagSpans(cols)
  let wrap = plan.shift + plan.fft.len
  let d = rows # a local: through the parameter every store would reload it
  for l in 0 ..< used:
    for m in 0 ..< s.wrapped: d[l][m] = T(0)
    for m in s.last ..< cols: d[l][m] = T(0)
  for m in s.wrapped ..< s.direct:
    for l in 0 ..< used: d[l][m] = buf[(m + wrap) * fftLanes + l]
  for m in s.direct ..< s.last:
    for l in 0 ..< used: d[l][m] = buf[(m + plan.shift) * fftLanes + l]

proc scatterOne[T](plan: CorrelationPlan[T], buf: Buf[T], d: Buf[T], cols: int) {.
    raises: [].} =
  ## One record's lags out of its buffer into `d`.
  let s = plan.lagSpans(cols)
  let wrap = plan.shift + plan.fft.len
  for m in 0 ..< s.wrapped: d[m] = T(0)
  for m in s.wrapped ..< s.direct: d[m] = buf[m + wrap]
  for m in s.direct ..< s.last: d[m] = buf[m + plan.shift]
  for m in s.last ..< cols: d[m] = T(0)

{.pop.}

proc correlateInto*[T: SampleType](plan: CorrelationPlan[T], x: openArray[T],
    y: var openArray[T], ws: var Workspace) {.raises: [].} =
  ## Correlates one record `x` (at most `recordSamples`; zeros after it)
  ## into `y` (`outSamples` values), through the one-trace FFT. `ws` must
  ## hold `plan.workspaceSize`. An `AssertionDefect` for other lengths.
  plan.checkPlan
  doAssert x.len <= plan.recordSamples, "a record longer than the plan's recordSamples"
  doAssert y.len == plan.outSamples, "y must have outSamples values"
  doAssert ws.fits(plan.workspaceSize), "reserve the plan's workspaceSize first"
  let n = plan.fft.len
  let bins = plan.fft.bins
  let at = plan.fft.workspaceSize.elems(T)
  let buf = ws.scratch(T, n + 2 * bins, at)
  let re = cast[ptr UncheckedArray[T]](addr buf[n])
  let im = cast[ptr UncheckedArray[T]](addr buf[n + bins])
  if x.len > 0: copyMem(buf, addr x[0], x.len * sizeof(T))
  zeroMem(addr buf[x.len], (n - x.len) * sizeof(T))
  plan.fft.forward(buf.toOpenArray(0, n - 1), re.toOpenArray(0, bins - 1),
    im.toOpenArray(0, bins - 1), ws)
  mulSingle(re, im, plan.mulRe, plan.mulIm)
  plan.fft.inverse(re.toOpenArray(0, bins - 1), im.toOpenArray(0, bins - 1),
    buf.toOpenArray(0, n - 1), ws)
  if y.len > 0: plan.scatterOne(buf, cast[Buf[T]](addr y[0]), y.len)

proc checkCorrelate*[T: SampleType](plan: CorrelationPlan[T], src, dst: RowBlock[T],
    meta: ChunkMeta) =
  ## The checks of `correlateInto` on `src`, `dst` and `meta`, without
  ## correlating: raises `ValueError` for a block that does not match the
  ## meta, `src` wider than `recordSamples`, `dst` not `outSamples` wide,
  ## another `dtMs` or a windowed meta (`firstSample > 0`).
  plan.checkPlan
  meta.checkShape(src)
  meta.checkShape(dst)
  if src.cols > plan.recordSamples:
    badPlan("the records are " & $src.cols & " samples, the plan takes " & $plan.recordSamples)
  if dst.cols != plan.outSamples:
    badPlan("dst must be " & $plan.outSamples & " samples wide, got " & $dst.cols)
  if not sameInterval(meta.dtMs, plan.dtMs):
    badPlan("the records are at " & $meta.dtMs & " ms, the plan at " & $plan.dtMs & " ms")
  if meta.firstSample != 0:
    badPlan("correlation needs whole records from sample 0, got firstSample " &
      $meta.firstSample)

proc correlateInto*[T: SampleType](plan: CorrelationPlan[T], src, dst: RowBlock[T],
    meta: ChunkMeta, ws: var Workspace) =
  ## Correlates the rows of `src` into `dst` (`outSamples` columns, the same
  ## rows; see the module documentation for which rows). Raises `ValueError`
  ## as `checkCorrelate`, before any sample changes. `ws` must hold
  ## `plan.workspaceSize`.
  plan.checkCorrelate(src, dst, meta)
  doAssert ws.fits(plan.workspaceSize), "reserve the plan's workspaceSize first"
  let n = plan.fft.len
  let bins = plan.fft.bins
  let at = plan.fft.workspaceSize.elems(T)
  let buf = ws.scratch(T, fftLanes * (n + 2 * bins), at)
  let re = cast[ptr UncheckedArray[T]](addr buf[n * fftLanes])
  let im = cast[ptr UncheckedArray[T]](addr buf[n * fftLanes + bins * fftLanes])
  template used(r: int): bool =
    not meta.isFailed(r) and meta.keys(r).status != tsDead
  var r = 0
  while r < src.len:
    # The next fftLanes rows to correlate.
    var lanes, live: array[fftLanes, int]
    var rows: array[fftLanes, Buf[T]]
    var k = 0
    while r < src.len and k < fftLanes:
      if used(r):
        lanes[k] = r
        rows[k] = src.rowPtr(r)
        live[k] = meta.liveSamples(r, src.cols)
        inc k
      else:
        let d = dst.rowPtr(r)
        for m in 0 ..< dst.cols: d[m] = T(0)
      inc r
    if k == 0: break
    gatherLanes(buf, n, rows, live, k)
    plan.runLanes(buf, re, im, ws)
    for l in 0 ..< k: rows[l] = dst.rowPtr(lanes[l])
    plan.scatterLanes(buf, rows, k, dst.cols)

proc correlateInto*[T: SampleType](plan: CorrelationPlan[T], src: Tensor[T],
    dst: var Tensor[T], meta: ChunkMeta, ws: var Workspace) =
  ## The tensor form: `src` is `[rows, ≤ recordSamples]`, `dst`
  ## `[rows, outSamples]`, both C-contiguous. Raises `ValueError` as the
  ## block form.
  meta.checkShape(src)
  meta.checkShape(dst)
  plan.correlateInto(rowBlock(src), rowBlock(dst), meta, ws)

proc correlate*[T: SampleType](plan: CorrelationPlan[T], src: Tensor[T],
    meta: ChunkMeta): Tensor[T] =
  ## A new `[rows, outSamples]` tensor of the correlated records, with its
  ## own workspace: for one-off use (a loop uses `correlateInto`).
  result = zeros[T](src.shape[0], plan.outSamples)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  plan.correlateInto(src, result, meta, ws)
