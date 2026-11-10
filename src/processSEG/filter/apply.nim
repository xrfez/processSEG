## Frequency-domain filtering (`FilterPlan`, decisions D3 and D4): a zero-phase
## `FilterSpec` applied to traces in place, with the edge handling that keeps
## the ends of a trace from swinging after a low-cut.
##
## ```nim
## let plan = initFilterPlan(float32, lowCut(0.5, 1), dtMs = 2, samples = 2001)
## var ws = initWorkspace()
## ws.reserve(plan.workspaceSize)
## samples.applyFilter(plan, meta, ws)      # in place, live rows, each to its live length
## let m = plan.requiredMargin              # samples to decode on each side of a zoom window
## ```
##
## **Edge handling** (D4), per trace, on its live samples only:
## 1. remove the mean or the least-squares line (`trendRemoval`, default
##    `trLinear`);
## 2. pad each end by odd reflection, `2·x₀ − x[k]`, which keeps the value and
##    the slope continuous, for `L` samples, the impulse length of the
##    response (`impulseLength`, 99.9 % of its energy);
## 3. taper the pads only, with a cos² ramp that is 1 with zero slope at the
##    data and falls to 0 at the outer end (the data is never tapered);
## 4. transform with an FFT of `nfft ≥ samples + 4L` (a 2·3·5 length: not a
##    power of two, which only costs time here) and with bins no wider than
##    half the narrowest ramp of the response. After the two pads, at least
##    `2L` zeros separate one end of the trace from the other, so the
##    response does not wrap: the tail past `L` (0.1 % of the energy, but a
##    cos² ramp's impulse decays only as 1/t³) leaks less than −60 dB of the
##    peak, measured with a spike on the last sample, the worst case (with
##    `samples + 2L` it was −40 dB);
## 5. multiply by the response, transform back, and add the removed trend
##    back scaled by the response at 0 Hz (0 for a low-cut, 1 for a high-cut;
##    exact for a zero-phase filter, whose symmetric impulse maps a line to
##    itself times its sum). A high-cut therefore keeps the trend, and a
##    low-cut removes it.
##
## **Margin.** `requiredMargin` is `L`: a zoom window decoded with `L` extra
## samples on each side, filtered, matches the whole trace filtered over the
## window, to the 0.1 % of the impulse energy left out of `L`.
##
## **Rows.** The block forms filter the live rows, each to its live length
## (`ChunkMeta.liveSamples`); dead, auxiliary and failed rows and the padding
## past a trace's end are left untouched. The rows go through the batched FFT,
## `fftLanes` at a time. A NaN or Inf sample makes its own row NaN (its trend
## is NaN) and no other. Nothing allocates once the workspace is reserved.

import std/math
import parseSEG
import ../core/[rows, types, workspace]
import ../fft/plan
import design, edges
export edges.TrendRemoval

type
  FilterPlan*[T: SampleType] = object
    ## A zero-phase filter for traces of up to `samples` samples; see the
    ## module documentation.
    spec: FilterSpec
    fft: FftPlan[T]
    gain: seq[T]   # one value per bin
    fall: seq[T]   # pad weights, k = 1 .. pad (fall[k − 1])
    pad: int
    samples: int
    dtMs: float64
    dcGain: float64
    trend: TrendRemoval

  Buf[T] = ptr UncheckedArray[T]

proc badPlan(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc initFilterPlan*(T: typedesc[SampleType], spec: FilterSpec, dtMs: float64,
    samples: int, trend = trLinear): FilterPlan[T] =
  ## A plan applying `spec` to traces of up to `samples` samples at `dtMs`.
  ## Raises `ValueError` for a `spec` that `validate` rejects at `dtMs`, for
  ## `samples < 1`, or for a response too narrow to design
  ## (`impulseLength`).
  if samples < 1: badPlan("a filter plan needs samples ≥ 1, got " & $samples)
  spec.validate(dtMs)
  let pad = spec.impulseLength(dtMs)
  var n = samples + 4 * pad # pads of L each, then a gap of ≥ 2L of zeros
  let ramp = spec.narrowestRampHz
  if ramp < Inf: # bins no wider than half the narrowest ramp
    n = max(n, int(ceil(2000 / (ramp * dtMs))))
  n = nextFastLen(n)
  result.spec = spec
  result.fft = initFftPlan(T, n)
  result.gain = newSeq[T](result.fft.bins)
  spec.responseInto(dtMs, n, result.gain)
  result.fall = newSeq[T](pad)
  for k in 1..pad:
    result.fall[k - 1] = T(cos(PI / 2 * float64(k) / float64(pad + 1)) ^ 2)
  result.pad = pad
  result.samples = samples
  result.dtMs = dtMs
  result.dcGain = if spec.zeroDc: 0.0 else: spec.response(0, dtMs)
  result.trend = trend

func requiredMargin*[T](plan: FilterPlan[T]): int {.inline, raises: [].} =
  ## The samples a zoom window needs on each side: the impulse length `L`.
  plan.pad

func samples*[T](plan: FilterPlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  plan.samples

func nfft*[T](plan: FilterPlan[T]): int {.inline, raises: [].} =
  ## The FFT length: at least `samples + 4·requiredMargin`.
  plan.fft.len

func spec*[T](plan: FilterPlan[T]): lent FilterSpec {.inline, raises: [].} =
  ## The response the plan applies.
  plan.spec

func dtMs*[T](plan: FilterPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval the plan was built for.
  plan.dtMs

func binHz*[T](plan: FilterPlan[T]): float64 {.inline, raises: [].} =
  ## The bin spacing of the plan's FFT, in Hz.
  1000 / (float64(plan.fft.len) * plan.dtMs)

func gainAt*[T](plan: FilterPlan[T], k: int): float64 {.inline.} =
  ## The gain the plan applies at bin `k` (0 .. nfft div 2).
  float64(plan.gain[k])

func elems(s: WorkspaceSize, T: typedesc): int =
  when T is float32: s.f32 else: s.f64

func workspaceSize*[T](plan: FilterPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch of `applyFilter` (block and single-trace forms).
  let extra = plan.fft.len * fftLanes + 2 * plan.fft.bins * fftLanes
  result = plan.fft.workspaceSize
  when T is float32: result.f32 += extra
  else: result.f64 += extra

# ------------------------------------------------------------------ kernels --

proc packInto[T](plan: FilterPlan[T], x: Buf[T], n: int, a, b: float64,
    dst: Buf[T], stride: int) {.raises: [].} =
  ## Writes the detrended, reflection-padded and pad-tapered `x[0 ..< n]`
  ## (module documentation) to `dst[j·stride]`, j < nfft.
  let p = plan.pad
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0..<n:
    dst[(p + i) * stride] = T(float64(x[i]) - (a + b * float64(i)))
  let d0 = dst[p * stride]
  let dn = dst[(p + n - 1) * stride]
  for k in 1..p:
    let r = dst[(p + min(k, n - 1)) * stride]
    dst[(p - k) * stride] = (d0 + d0 - r) * plan.fall[k - 1]
  for k in 1..p:
    let r = dst[(p + n - 1 - min(k, n - 1)) * stride]
    dst[(p + n - 1 + k) * stride] = (dn + dn - r) * plan.fall[k - 1]
  for j in (2 * p + n)..<plan.fft.len: dst[j * stride] = T(0)
  {.pop.}

proc unpackFrom[T](plan: FilterPlan[T], src: Buf[T], stride: int, a, b: float64,
    x: Buf[T], n: int) {.raises: [].} =
  ## The filtered data back into `x[0 ..< n]`, with the trend times the gain
  ## at 0 Hz added back.
  let p = plan.pad
  let ga = a * plan.dcGain
  let gb = b * plan.dcGain
  {.push boundChecks: off, overflowChecks: off.}
  if ga == 0 and gb == 0:
    for i in 0..<n: x[i] = src[(p + i) * stride]
  else:
    for i in 0..<n: x[i] = T(float64(src[(p + i) * stride]) + (ga + gb * float64(i)))
  {.pop.}

{.push boundChecks: off, overflowChecks: off.} # an inline kernel keeps the
                                              # checks of its definition site
proc gainKernel[T](re {.noalias.}, im {.noalias.}: Buf[T],
    g {.noalias.}: Buf[T], bins: int, L: static int) {.inline.} =
  for k in 0..<bins:
    let w = g[k]
    for l in 0..<L:
      re[k * L + l] = re[k * L + l] * w
      im[k * L + l] = im[k * L + l] * w
{.pop.}

proc gainLanes[T](plan: FilterPlan[T], re, im: Buf[T]) {.noinline, raises: [].} =
  {.push boundChecks: off, overflowChecks: off.}
  gainKernel(re, im, cast[ptr UncheckedArray[T]](unsafeAddr plan.gain[0]), plan.fft.bins, fftLanes)
  {.pop.}

proc gainSingle[T](plan: FilterPlan[T], re, im: Buf[T]) {.noinline, raises: [].} =
  {.push boundChecks: off, overflowChecks: off.}
  gainKernel(re, im, cast[ptr UncheckedArray[T]](unsafeAddr plan.gain[0]), plan.fft.bins, 1)
  {.pop.}

# -------------------------------------------------------------- one trace --

proc checkPlan[T](plan: FilterPlan[T]) {.inline.} =
  doAssert plan.gain.len > 0, "a FilterPlan must come from initFilterPlan"

proc applyFilter*[T: SampleType](x: var openArray[T], plan: FilterPlan[T],
    ws: var Workspace) {.raises: [].} =
  ## Filters one trace, `x` (its live samples, at most `plan.samples`; an
  ## `AssertionDefect` otherwise), in place. `ws` must hold
  ## `plan.workspaceSize`. An empty `x` is left as it is.
  plan.checkPlan
  doAssert x.len <= plan.samples, "a trace longer than the plan's samples"
  let n = x.len
  if n == 0: return
  let nf = plan.fft.len
  let nb = plan.fft.bins
  let at = plan.fft.workspaceSize.elems(T)
  let buf = ws.scratch(T, nf + 2 * nb, at)
  let re = cast[ptr UncheckedArray[T]](addr buf[nf])
  let im = cast[ptr UncheckedArray[T]](addr buf[nf + nb])
  let xp = cast[ptr UncheckedArray[T]](addr x[0])
  let (a, b) = trendOf(plan.trend, xp, n)
  plan.packInto(xp, n, a, b, buf, 1)
  plan.fft.forward(buf.toOpenArray(0, nf - 1), re.toOpenArray(0, nb - 1),
    im.toOpenArray(0, nb - 1), ws)
  plan.gainSingle(re, im)
  plan.fft.inverse(re.toOpenArray(0, nb - 1), im.toOpenArray(0, nb - 1),
    buf.toOpenArray(0, nf - 1), ws)
  plan.unpackFrom(buf, 1, a, b, xp, n)

# -------------------------------------------------------------- block form --

proc applyFilter*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    plan: FilterPlan[T], ws: var Workspace) =
  ## Filters the live rows of `b` in place, each to its live length (module
  ## documentation), `fftLanes` rows at a time. `ws` must hold
  ## `plan.workspaceSize`. Raises `ValueError`, before any sample changes,
  ## when `b` does not match `meta` or has more columns than `plan.samples`.
  meta.checkShape(b)
  plan.checkPlan
  if b.cols > plan.samples:
    badPlan("the block has " & $b.cols & " columns, the plan takes " &
      $plan.samples & " samples")
  let nf = plan.fft.len
  let nb = plan.fft.bins
  let at = plan.fft.workspaceSize.elems(T)
  let buf = ws.scratch(T, nf * fftLanes + 2 * nb * fftLanes, at)
  let re = cast[ptr UncheckedArray[T]](addr buf[nf * fftLanes])
  let im = cast[ptr UncheckedArray[T]](addr buf[nf * fftLanes + nb * fftLanes])
  var lane = 0
  var rowOf: array[fftLanes, int]
  var lenOf: array[fftLanes, int]
  var ta, tb: array[fftLanes, float64]

  template flush() =
    if lane > 0:
      for l in lane..<fftLanes: # unused lanes are zero
        for j in 0..<nf: buf[j * fftLanes + l] = T(0)
      plan.fft.forwardLanes(buf.toOpenArray(0, nf * fftLanes - 1),
        re.toOpenArray(0, nb * fftLanes - 1), im.toOpenArray(0, nb * fftLanes - 1), ws)
      plan.gainLanes(re, im)
      plan.fft.inverseLanes(re.toOpenArray(0, nb * fftLanes - 1),
        im.toOpenArray(0, nb * fftLanes - 1), buf.toOpenArray(0, nf * fftLanes - 1), ws)
      for l in 0..<lane:
        plan.unpackFrom(cast[ptr UncheckedArray[T]](addr buf[l]), fftLanes, ta[l], tb[l],
          b.rowPtr(rowOf[l]), lenOf[l])
      lane = 0

  for r in 0..<b.len:
    if not meta.isLive(r): continue
    let n = meta.liveSamples(r, b.cols)
    if n == 0: continue
    let src = b.rowPtr(r)
    (ta[lane], tb[lane]) = trendOf(plan.trend, src, n)
    plan.packInto(src, n, ta[lane], tb[lane], cast[ptr UncheckedArray[T]](addr buf[lane]), fftLanes)
    rowOf[lane] = r
    lenOf[lane] = n
    inc lane
    if lane == fftLanes: flush()
  flush()

proc applyFilter*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    plan: FilterPlan[T], ws: var Workspace) =
  ## `applyFilter` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  rowBlock(samples).applyFilter(meta, plan, ws)
