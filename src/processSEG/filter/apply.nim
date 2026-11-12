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
## 2. pad each end for `L` samples, the impulse length of the response
##    (`impulseLength`, 99.9 % of its energy):
##    - **with zeros**, when the response rejects 0 Hz (a low-cut or a
##      band-pass: gain under 0.5 there) and the line was removed. Nothing is
##      known about the samples beyond the trace, and with the trend gone
##      zero is the unbiased guess;
##    - **by odd reflection**, `2·x₀ − x[k]`, otherwise (a high-cut or a
##      notch, which pass the level and slope the reflection keeps
##      continuous; and `trMean`/`trNone`, where a drift or an offset left in
##      the data must not meet a step), tapered with a cos² ramp that is 1
##      with zero slope at the data and falls to 0 at the outer end. The
##      data itself is never tapered.
##
##    (An odd reflection under a low-cut pins the end samples: with a
##    symmetric impulse `h`, `y₀ = h₀x₀ + Σ hₖ·(xₖ + 2x₀ − xₖ) = H(0)·x₀ = 0`,
##    whatever the data. Measured on white noise against the same samples
##    inside a longer trace, the first `L/2` samples were off by 41 %
##    (low-cut 2–4 Hz) and 92 % (4-8-60-80 Hz) of the signal there; with
##    zeros 4 % and 15 %.)
## 3. transform with an FFT of `nfft ≥ samples + 2L` (zeros) or
##    `samples + 4L` (reflections: the two pads, then `2L` of zeros), a 2·3·5
##    length, with bins no wider than half the narrowest ramp of the
##    response. At least `2L` zeros separate one end of the trace from the
##    other, so the response does not wrap: its tail past `L` (0.1 % of the
##    energy, but a cos² ramp's impulse decays only as 1/t³) leaks less than
##    −60 dB of the peak (−75 dB with zero pads), measured with a spike on
##    the last sample, the worst case;
## 4. multiply by the response, transform back, and add the removed trend
##    back scaled by the response at 0 Hz (0 for a low-cut, 1 for a high-cut;
##    exact for a zero-phase filter, whose symmetric impulse maps a line to
##    itself times its sum). A high-cut therefore keeps the trend, and a
##    low-cut removes it. With `withZeroDc`, the trend's mean is not added
##    back, only its slope: the output's mean is near zero, not exactly zero
##    (bin 0 is the mean of the padded FFT window, not of the `n` samples; on
##    a trace with a large offset and slope, about 2 % of the signal
##    amplitude was measured).
##
## `trNone` leaves the samples as they are, so the pad taper pulls each end's
## value to 0 over `L` samples. That is fine for data near zero mean at its
## ends; a large offset then leaks through a low-cut's ramp (a DC of 1000
## under a 0.5–1 Hz low-cut moves the first samples by about 1 % of it). The
## default, `trLinear`, has no such leak.
##
## **Margin.** `requiredMargin` is `L`: a zoom window decoded with `L` extra
## samples on each side, filtered, matches the whole trace filtered over the
## window to within about 0.5 % of the trace's peak on broadband data
## (measured on white noise at 2 ms: 0.30 % for 4-8-60-80, 0.43 % for
## 5-10-40-60, 0.06 % for a 0.5–1 Hz low-cut), and about 2.5 % in the worst
## case, a steady sinusoid inside the narrowest ramp, whose tail adds up
## coherently over the 0.1 % of the impulse energy left out of `L` (2.4–2.7 %
## for the same filters). A margin of `1.5·L` brings that case under 0.7 %.
## A high-cut alone has a short `L` and more of the signal inside its ramp:
## 2.8 % on noise for 60–80 Hz (`L` = 20 samples), 0.3 % at `1.5·L`.
##
## **Rows.** The block forms filter the live rows, each to its live length
## (`ChunkMeta.liveSamples`); dead, auxiliary and failed rows and the padding
## past a trace's end are left untouched. The rows go through the batched FFT,
## `fftLanes` at a time; a row's result does not depend on the rows batched
## with it, so any split of the rows across threads gives the same samples. A
## NaN or Inf sample makes its own row NaN (its trend is NaN) and no other.
## Nothing allocates once the workspace is reserved.

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
    gain: seq[T] # one value per bin
    fall: seq[T] # pad weights, k = 1 .. pad (fall[k − 1])
    pad: int
    reflect: bool # the pads hold the odd reflection (else zeros)
    samples: int
    dtMs: float64
    dcGain: float64
    trend: TrendRemoval

  Buf[T] = ptr UncheckedArray[T]

template `+@`[T](b: Buf[T], o: int): Buf[T] =
  cast[Buf[T]](addr b[o])

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
  let dcGain = spec.response(0, dtMs)
  let reflect = dcGain >= 0.5 or trend != trLinear
  # Reflected pads of L each and 2L of zeros between them, or 2L of zeros.
  var n = samples + (if reflect: 4 else: 2) * pad
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
  result.dcGain = dcGain
  result.trend = trend
  result.reflect = reflect

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
  if plan.reflect:
    let d0 = dst[p * stride]
    let dn = dst[(p + n - 1) * stride]
    for k in 1..p:
      let r = dst[(p + min(k, n - 1)) * stride]
      dst[(p - k) * stride] = (d0 + d0 - r) * plan.fall[k - 1]
    for k in 1..p:
      let r = dst[(p + n - 1 - min(k, n - 1)) * stride]
      dst[(p + n - 1 + k) * stride] = (dn + dn - r) * plan.fall[k - 1]
  else:
    for k in 1..p:
      dst[(p - k) * stride] = T(0)
      dst[(p + n - 1 + k) * stride] = T(0)
  for j in (2 * p + n)..<plan.fft.len: dst[j * stride] = T(0)
  {.pop.}

proc unpackFrom[T](plan: FilterPlan[T], src: Buf[T], stride: int, a, b: float64,
    x: Buf[T], n: int) {.raises: [].} =
  ## The filtered data back into `x[0 ..< n]`, with the trend times the gain
  ## at 0 Hz added back (for `withZeroDc`, its slope only, without its mean).
  let p = plan.pad
  let gb = b * plan.dcGain
  let ga = if plan.spec.zeroDc: -gb * float64(n - 1) / 2 else: a * plan.dcGain
  {.push boundChecks: off, overflowChecks: off.}
  if ga == 0 and gb == 0:
    for i in 0..<n: x[i] = src[(p + i) * stride]
  else:
    for i in 0..<n: x[i] = T(float64(src[(p + i) * stride]) + (ga + gb *
        float64(i)))
  {.pop.}

{.push boundChecks: off, overflowChecks: off.}
proc gainRun[T](re {.noalias.}, im {.noalias.}, g {.noalias.}: Buf[T], bins,
    lanes: int) {.noinline, raises: [].} =
  ## Multiplies bin `k` of `lanes` interleaved spectra by `g[k]`. `lanes` is
  ## a run-time value on purpose: with a constant lane count GCC 12
  ## vectorizes the loop over the bins instead (eight bins at a time, through
  ## permutes), four times the instructions of this form.
  for k in 0..<bins:
    let w = g[k]
    let r = cast[Buf[T]](addr re[k * lanes])
    let i = cast[Buf[T]](addr im[k * lanes])
    for l in 0..<lanes:
      r[l] = r[l] * w
      i[l] = i[l] * w
{.pop.}

proc gainLanes[T](plan: FilterPlan[T], re, im: Buf[T]) {.inline, raises: [].} =
  gainRun(re, im, cast[Buf[T]](addr plan.gain[0]), plan.fft.bins, fftLanes)

proc gainSingle[T](plan: FilterPlan[T], re, im: Buf[T]) {.inline, raises: [].} =
  gainRun(re, im, cast[Buf[T]](addr plan.gain[0]), plan.fft.bins, 1)

# A full batch of rows of one length (the usual case) is packed and unpacked
# sample by sample across the lanes, so the lane buffer is written and read
# front to back, one cache line after the other. `packInto` on each row in
# turn writes every eighth value: each of its stores misses L1 (the buffer
# of a batch is larger), 8 times per line. Both forms compute the same
# expression per sample, so a row's result does not depend on its batch.

type
  LaneRows[T] = array[fftLanes, Buf[T]]
  LaneTrend = array[fftLanes, float64]

{.push boundChecks: off, overflowChecks: off.}
proc packSample[T](d {.noalias.}: Buf[T], rows: LaneRows[T], i: int,
    a, b: LaneTrend) {.inline.} =
  let fi = float64(i)
  for l in 0..<fftLanes:
    d[l] = T(float64(rows[l][i]) - (a[l] + b[l] * fi))

proc reflectSample[T](d {.noalias.}, mirror {.noalias.}, edge {.noalias.}: Buf[T],
    fall: T) {.inline.} =
  for l in 0..<fftLanes:
    d[l] = (edge[l] + edge[l] - mirror[l]) * fall

proc packLanes[T](plan: FilterPlan[T], rows: LaneRows[T], n: int,
    a, b: LaneTrend, dst: Buf[T]) {.noinline, raises: [].} =
  ## `packInto` for `fftLanes` rows of `n` samples each, into the lane buffer
  ## `dst` (sample `j` of lane `l` at `dst[j·fftLanes + l]`).
  const L = fftLanes
  let p = plan.pad
  for i in 0..<n:
    packSample(dst +@ ((p + i) * L), rows, i, a, b)
  if plan.reflect:
    let first = dst +@ (p * L)
    let last = dst +@ ((p + n - 1) * L)
    for k in 1..p:
      let m = min(k, n - 1)
      reflectSample(dst +@ ((p - k) * L), dst +@ ((p + m) * L), first, plan.fall[k - 1])
      reflectSample(dst +@ ((p + n - 1 + k) * L), dst +@ ((p + n - 1 - m) * L), last,
        plan.fall[k - 1])
    for j in (2 * p + n) * L..<plan.fft.len * L: dst[j] = T(0)
  else:
    for j in 0..<p * L: dst[j] = T(0)
    for j in (p + n) * L..<plan.fft.len * L: dst[j] = T(0)

proc unpackSample[T](s {.noalias.}: Buf[T], rows: LaneRows[T], i: int,
    ga, gb: LaneTrend) {.inline.} =
  let fi = float64(i)
  for l in 0..<fftLanes:
    rows[l][i] = T(float64(s[l]) + (ga[l] + gb[l] * fi))

proc copySample[T](s {.noalias.}: Buf[T], rows: LaneRows[T], i: int) {.inline.} =
  for l in 0..<fftLanes:
    rows[l][i] = s[l]

proc unpackLanes[T](plan: FilterPlan[T], src: Buf[T], a, b: LaneTrend,
    rows: LaneRows[T], n: int) {.noinline, raises: [].} =
  ## `unpackFrom` for `fftLanes` rows of `n` samples each.
  const L = fftLanes
  let p = plan.pad
  var ga, gb: LaneTrend
  var plain = true
  for l in 0..<L:
    gb[l] = b[l] * plan.dcGain
    ga[l] = (if plan.spec.zeroDc: -gb[l] * float64(n - 1) / 2 else: a[l] * plan.dcGain)
    if ga[l] != 0 or gb[l] != 0:
      plain = false
    else:
      # No trend to add back: `x + (−0 + −0·i)` is `x` bit for bit (adding
      # +0 would turn a −0 sample into +0, unlike `unpackFrom`'s plain copy).
      ga[l] = -0.0
      gb[l] = -0.0
  if plain:
    for i in 0..<n: copySample(src +@ ((p + i) * L), rows, i)
  else:
    for i in 0..<n: unpackSample(src +@ ((p + i) * L), rows, i, ga, gb)
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
  ## when `b` does not match `meta`, has more columns than `plan.samples`, or
  ## its rows' interval (`meta.dtMs`) is not the plan's.
  meta.checkShape(b)
  if abs(meta.dtMs - plan.dtMs) > 1e-9 * plan.dtMs:
    badPlan("the rows' interval " & $meta.dtMs & " ms is not the plan's " &
      $(plan.dtMs) & " ms")
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
  var rowsOf: LaneRows[T]
  var lenOf: array[fftLanes, int]
  var ta, tb: LaneTrend

  template flush() =
    if lane > 0:
      var uniform = lane == fftLanes
      for l in 1..<lane:
        if lenOf[l] != lenOf[0]: uniform = false
      if uniform:
        plan.packLanes(rowsOf, lenOf[0], ta, tb, buf)
      else:
        for l in 0..<lane:
          plan.packInto(rowsOf[l], lenOf[l], ta[l], tb[l], buf +@ l, fftLanes)
        for l in lane..<fftLanes: # unused lanes are zero
          for j in 0..<nf: buf[j * fftLanes + l] = T(0)
      plan.fft.forwardLanes(buf.toOpenArray(0, nf * fftLanes - 1),
        re.toOpenArray(0, nb * fftLanes - 1), im.toOpenArray(0, nb * fftLanes - 1), ws)
      plan.gainLanes(re, im)
      plan.fft.inverseLanes(re.toOpenArray(0, nb * fftLanes - 1),
        im.toOpenArray(0, nb * fftLanes - 1), buf.toOpenArray(0, nf * fftLanes - 1), ws)
      if uniform:
        plan.unpackLanes(buf, ta, tb, rowsOf, lenOf[0])
      else:
        for l in 0..<lane:
          plan.unpackFrom(buf +@ l, fftLanes, ta[l], tb[l], rowsOf[l], lenOf[l])
      lane = 0

  for r in 0..<b.len:
    let n = (if meta.isLive(r): meta.liveSamples(r, b.cols) else: 0)
    if n > 0:
      rowsOf[lane] = b.rowPtr(r)
      lenOf[lane] = n
      (ta[lane], tb[lane]) = trendOf(plan.trend, rowsOf[lane], n)
      inc lane
      if lane == fftLanes: flush()
  flush()

proc applyFilter*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    plan: FilterPlan[T], ws: var Workspace) =
  ## `applyFilter` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  rowBlock(samples).applyFilter(meta, plan, ws)
