## Time-domain Butterworth filters (`IirPlan`, decision D3): cascades of
## biquads, minimum phase (one causal pass) or zero phase (forward-backward),
## started from steady state so that the trace ends do not ring.
##
## ```nim
## let spec = butterBandPass(4, 60, slopeDbPerOctave = 24)   # effective slope
## let zp = initIirPlan(float32, spec, dtMs = 2, samples = 2001)               # zero phase
## let mp = initIirPlan(float32, spec, dtMs = 2, samples = 2001, phase = ipMinimum)
## var ws = initWorkspace()
## ws.reserve(zp.workspaceSize)
## samples.applyFilter(meta, zp, ws)
## ```
##
## **Design.** Each edge of a `butterLowCut`, `butterHighCut` or
## `butterBandPass` spec is a digital Butterworth filter from the bilinear
## transform, prewarped so that its corner is exact; a band-pass is a low-cut
## cascaded with a high-cut. Sections are second order (one first-order
## section for an odd order) and run in float64, in transposed direct form II.
##
## **Slopes and corners** (D5). The spec's slope is the effective one:
## - **zero phase** (`ipZero`): the filter runs forward then backward, so the
##   applied gain is `|H|²`: `slope/12` poles, the corner is −6 dB, and the
##   gain equals the spec's `response` (the same warped Butterworth
##   magnitude as the FFT filter of `filter/apply`). The slope must be a
##   multiple of 12 dB/octave.
## - **minimum phase** (`ipMinimum`): one causal pass, gain `|H|` of
##   `slope/6` poles: the corner is −3 dB. The slope must be a multiple of
##   6 dB/octave. The output is delayed (by the filter's group delay), as a
##   causal filter's is.
##
## **Edge handling.** As for the FFT filters (D4), on each trace's live
## samples: the trend is removed (`trendRemoval`), each end that a pass
## starts from is padded by odd reflection for `requiredMargin` samples, and each pass of each
## section starts in the steady state of its first input sample (the
## `lfilter_zi` method), so a DC offset does not start a transient. The trend
## `a + b·i` is then added back as the filter's steady-state response to it:
## `G(1)·(a + b·i) − b·G′(1)`, with `G(q)` the cascade's transfer function
## in `q = z⁻¹` (for zero phase, `G′(1)` cancels between the passes and the
## gain is `G(1)²`). For minimum phase this includes the filter's delay of
## the trend; without `trLinear`, a drift starts a transient there (a
## steady-state start assumes a constant). The pads are not tapered: the
## steady-state start replaces the taper.
##
## **Rows.** As for `filter/apply`: live rows, each to its live length,
## `fftLanes` traces at a time (the recursion runs across traces, which
## vectorizes); a NaN stays in its row; no allocation once the workspace is
## reserved.

import std/[complex, math]
import parseSEG
import ../core/[rows, types, workspace]
import ../fft/plan
import design, edges
export edges.TrendRemoval

type
  IirPhase* = enum
    ipZero    ## forward-backward: zero phase, gain |H|²
    ipMinimum ## one causal pass: minimum phase, gain |H|

  Biquad = object
    b0, b1, b2, a1, a2: float64 # a0 = 1
    g: float64                  # gain at 0 Hz

  IirPlan*[T: SampleType] = object
    ## A Butterworth biquad cascade for traces of up to `samples` samples;
    ## see the module documentation.
    spec: FilterSpec
    sections: seq[Biquad]
    phase: IirPhase
    pad: int
    samples: int
    dtMs: float64
    dcGain: float64 # G(1), or G(1)² for zero phase
    dcSlope: float64 # G′(1) for minimum phase, 0 for zero phase
    trend: TrendRemoval

  Buf = ptr UncheckedArray[float64]

proc badIir(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

# ------------------------------------------------------------------- design --

func biquad(b0, b1, b2, a1, a2: float64): Biquad =
  Biquad(b0: b0, b1: b1, b2: b2, a1: a1, a2: a2,
    g: (b0 + b1 + b2) / (1 + a1 + a2))

proc butterSections(fc, dtMs: float64, order: int, high: bool): seq[Biquad] =
  ## A digital Butterworth low-pass (`high = false`) or high-pass of `order`
  ## poles with its −3 dB corner at `fc` Hz, by the prewarped bilinear
  ## transform.
  let fs = 1000 / dtMs
  let wc = 2 * fs * tan(PI * fc / fs) # prewarped analog corner, rad/s
  let k2 = 2 * fs
  for k in 0..<order div 2:
    # Analog prototype poles −sin θ ± i·cos θ (θ = π(2k+1)/(2N)); the
    # high-pass poles wc/p are the same set, as |p| = 1.
    let th = PI * float64(2 * k + 1) / float64(2 * order)
    let s = complex(-wc * sin(th), wc * cos(th))
    let z = (complex(k2, 0.0) + s) / (complex(k2, 0.0) - s)
    let a1 = -2 * z.re
    let a2 = z.re * z.re + z.im * z.im
    if high:
      let g = (1 - a1 + a2) / 4 # gain 1 at Nyquist
      result.add biquad(g, -2 * g, g, a1, a2)
    else:
      let g = (1 + a1 + a2) / 4 # gain 1 at 0 Hz
      result.add biquad(g, 2 * g, g, a1, a2)
  if order mod 2 == 1:
    let z = (k2 - wc) / (k2 + wc)
    let a1 = -z
    if high:
      let g = (1 - a1) / 2
      result.add biquad(g, -g, 0, a1, 0)
    else:
      let g = (1 + a1) / 2
      result.add biquad(g, g, 0, a1, 0)

const causalEnergyFraction* = 0.9999
  ## The energy a minimum-phase margin holds. Stricter than the zero-phase
  ## `energyFraction` (0.999): all the error of a causal filter's start comes
  ## from the history before a window, and at 0.999 a zoom window was off by
  ## 2 % of its peak; at 0.9999, under 0.7 %.

proc causalLength(sections: seq[Biquad]): int =
  ## The samples after a spike that hold `causalEnergyFraction` of the
  ## energy of the causal impulse response `h` of `sections`, or of its complement
  ## `δ − h`, whichever is longer (a low-cut's impulse is nearly a spike with
  ## a long, weak tail), run in float64 with a zero start until the response
  ## has died away (the last quarter holds < 1e-12 of the energy).
  var n = 1024
  while true:
    var h = newSeq[float64](n)
    h[0] = 1
    for c in sections: # zero initial state
      var s1, s2 = 0.0
      for j in 0..<n:
        let x = h[j]
        let y = c.b0 * x + s1
        s1 = c.b1 * x - c.a1 * y + s2
        s2 = c.b2 * x - c.a2 * y
        h[j] = y
    var rest, tail = 0.0
    for j in 1..<n:
      rest += h[j] * h[j]
      if j >= n - n div 4: tail += h[j] * h[j]
    if tail <= 1e-12 * max(rest, 1e-300) or n >= 1 shl 24:
      for c0 in [h[0], 1 - h[0]]:
        let total = c0 * c0 + rest
        var e = c0 * c0
        var L = 0
        while e < causalEnergyFraction * total and L < n - 1:
          inc L
          e += h[L] * h[L]
        result = max(result, L)
      return
    n *= 2

proc poles(slope: float64, perPole: float64, side: string): int =
  let n = slope / perPole
  if not (n >= 1 and n <= 32 and n == round(n)):
    badIir("the " & side & " slope of an IIR filter must be a multiple of " &
      $perPole & " dB/octave (" & $int(perPole) &
          " per pole), 1 .. 32 poles; got " &
      $slope)
  int(n)

proc initIirPlan*(T: typedesc[SampleType], spec: FilterSpec, dtMs: float64,
    samples: int, phase = ipZero, trend = trLinear): IirPlan[T] =
  ## A Butterworth cascade for `spec` (a `butter*` spec without notches or
  ## 0 Hz suppression) on traces of up to `samples` samples at `dtMs`.
  ## Raises `ValueError` for another spec, a slope that is not a multiple of
  ## 12 dB/octave (zero phase) or 6 (minimum phase), `samples < 1`, or what
  ## `validate` rejects.
  if samples < 1: badIir("an IIR plan needs samples ≥ 1, got " & $samples)
  spec.validate(dtMs)
  if spec.kind != rkButterworth:
    badIir("an IIR filter needs a Butterworth spec (butterLowCut, " &
      "butterHighCut, butterBandPass)")
  if spec.notches.len > 0 or spec.zeroDc:
    badIir("an IIR filter takes no notches and no 0 Hz suppression")
  let perPole = if phase == ipZero: 12.0 else: 6.0
  var lowSecs, highSecs: seq[Biquad]
  if spec.lowHz > 0:
    lowSecs = butterSections(spec.lowHz, dtMs,
      poles(spec.lowSlope, perPole, "low-cut"), high = true)
  if spec.highHz < Inf:
    highSecs = butterSections(spec.highHz, dtMs,
      poles(spec.highSlope, perPole, "high-cut"), high = false)
  result.sections = lowSecs & highSecs
  result.spec = spec
  result.phase = phase
  if phase == ipZero:
    result.pad = spec.impulseLength(dtMs) # |H|² is the spec's response
  else: # each edge alone, as for impulseLength
    result.pad = max(causalLength(lowSecs), causalLength(highSecs))
  result.samples = samples
  result.dtMs = dtMs
  # G(1) and G′(1) of the cascade, from each section's N(q)/D(q).
  var g = 1.0
  var dg = 0.0
  for c in result.sections:
    let n1 = c.b0 + c.b1 + c.b2
    let d1 = 1 + c.a1 + c.a2
    let gi = n1 / d1
    let dgi = ((c.b1 + 2 * c.b2) * d1 - n1 * (c.a1 + 2 * c.a2)) / (d1 * d1)
    dg = dg * gi + g * dgi
    g *= gi
  result.dcGain = if phase == ipZero: g * g else: g
  result.dcSlope = if phase == ipZero: 0.0 else: dg
  result.trend = trend

func requiredMargin*[T](plan: IirPlan[T]): int {.inline, raises: [].} =
  ## The samples a zoom window needs on each side (before it, for minimum
  ## phase): the impulse length of the spec for zero phase (its `|H|²` is the
  ## spec's response), and for minimum phase the 99.99 %-energy length of the
  ## cascade's own causal impulse (slope/6 poles ring longer).
  plan.pad

func samples*[T](plan: IirPlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  plan.samples

func phase*[T](plan: IirPlan[T]): IirPhase {.inline, raises: [].} =
  ## Zero or minimum phase.
  plan.phase

func order*[T](plan: IirPlan[T]): int {.raises: [].} =
  ## The number of poles of one pass.
  for s in plan.sections: result += (if s.a2 == 0: 1 else: 2)

func spec*[T](plan: IirPlan[T]): lent FilterSpec {.inline, raises: [].} =
  ## The spec the plan was designed from.
  plan.spec

func magnitudeAt*[T](plan: IirPlan[T], f: float64): float64 {.raises: [].} =
  ## The gain the plan applies at `f` Hz, from its sections: `|H|` for
  ## minimum phase, `|H|²` for zero phase.
  let w = 2 * PI * f * plan.dtMs / 1000
  let z1 = complex(cos(w), -sin(w)) # z⁻¹
  var h = complex(1.0, 0.0)
  for s in plan.sections:
    h *= (s.b0 + s.b1 * z1 + s.b2 * z1 * z1) / (1.0 + s.a1 * z1 + s.a2 * z1 * z1)
  if plan.phase == ipZero: abs(h) ^ 2 else: abs(h)

func span[T](plan: IirPlan[T], n: int): int {.inline.} =
  ## The padded length of a trace of `n` samples.
  if plan.phase == ipZero: n + 2 * plan.pad else: n + plan.pad

func workspaceSize*[T](plan: IirPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch of `applyFilter` (block and single-trace forms).
  WorkspaceSize(f64: plan.span(plan.samples) * fftLanes)

# ------------------------------------------------------------------ kernels --

{.push boundChecks: off, overflowChecks: off.} # inline kernels keep the checks
                                              # of their definition site
proc sosKernel(buf {.noalias.}: Buf, m: int, c: Biquad, L: static int,
    backward: static bool) {.inline.} =
  ## One pass of one section over `m` samples of `L` interleaved traces, in
  ## place, started in the steady state of each trace's first input sample.
  let (b0, b1, b2, a1, a2) = (c.b0, c.b1, c.b2, c.a1, c.a2)
  var s1, s2: array[L, float64]
  let first = when backward: m - 1 else: 0
  for l in 0..<L:
    let u = buf[first * L + l]
    let y = c.g * u
    s2[l] = b2 * u - a2 * y
    s1[l] = y - b0 * u
  when backward:
    var j = m - 1
    while j >= 0:
      for l in 0..<L:
        let x = buf[j * L + l]
        let y = b0 * x + s1[l]
        s1[l] = b1 * x - a1 * y + s2[l]
        s2[l] = b2 * x - a2 * y
        buf[j * L + l] = y
      dec j
  else:
    for j in 0..<m:
      for l in 0..<L:
        let x = buf[j * L + l]
        let y = b0 * x + s1[l]
        s1[l] = b1 * x - a1 * y + s2[l]
        s2[l] = b2 * x - a2 * y
        buf[j * L + l] = y
{.pop.}

proc sosLanes(buf: Buf, m: int, c: Biquad, backward: bool) {.noinline, raises: [].} =
  if backward: sosKernel(buf, m, c, fftLanes, true)
  else: sosKernel(buf, m, c, fftLanes, false)

proc sosSingle(buf: Buf, m: int, c: Biquad, backward: bool) {.noinline,
    raises: [].} =
  if backward: sosKernel(buf, m, c, 1, true)
  else: sosKernel(buf, m, c, 1, false)

proc packInto[T](plan: IirPlan[T], x: ptr UncheckedArray[T], n: int,
    a, b: float64, dst: Buf, stride: int) {.raises: [].} =
  ## The detrended `x[0 ..< n]` with odd-reflection pads (both ends for zero
  ## phase, the start only for minimum phase) into `dst[j·stride]`.
  let p = plan.pad
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0..<n:
    dst[(p + i) * stride] = float64(x[i]) - (a + b * float64(i))
  let d0 = dst[p * stride]
  for k in 1..p:
    dst[(p - k) * stride] = 2 * d0 - dst[(p + min(k, n - 1)) * stride]
  if plan.phase == ipZero:
    let dn = dst[(p + n - 1) * stride]
    for k in 1..p:
      dst[(p + n - 1 + k) * stride] = 2 * dn - dst[(p + n - 1 - min(k, n - 1)) * stride]
  {.pop.}

proc unpackFrom[T](plan: IirPlan[T], src: Buf, stride: int, a, b: float64,
    x: ptr UncheckedArray[T], n: int) {.raises: [].} =
  let p = plan.pad
  let ga = a * plan.dcGain - b * plan.dcSlope
  let gb = b * plan.dcGain
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0..<n: x[i] = T(src[(p + i) * stride] + (ga + gb * float64(i)))
  {.pop.}

proc runSingle[T](plan: IirPlan[T], buf: Buf, m: int) {.raises: [].} =
  for c in plan.sections: sosSingle(buf, m, c, false)
  if plan.phase == ipZero:
    for c in plan.sections: sosSingle(buf, m, c, true)

proc runLanes[T](plan: IirPlan[T], buf: Buf, m: int,
    spans: array[fftLanes, int], used: int) {.raises: [].} =
  ## The passes over a batch whose lane `l < used` holds `spans[l]` padded
  ## samples (≤ m). Before the backward passes, each shorter lane holds its
  ## last forward output out to `m`: a constant in steady state reaches the
  ## lane's own end in exactly the start-up state the backward pass would
  ## give it there, so each lane is filtered as if alone.
  for c in plan.sections: sosLanes(buf, m, c, false)
  if plan.phase == ipZero:
    for l in 0..<used:
      let last = buf[(spans[l] - 1) * fftLanes + l]
      for j in spans[l]..<m: buf[j * fftLanes + l] = last
    for c in plan.sections: sosLanes(buf, m, c, true)

proc checkPlan[T](plan: IirPlan[T]) {.inline.} =
  doAssert plan.samples > 0, "an IirPlan must come from initIirPlan"

# ---------------------------------------------------------------- apply --

proc applyFilter*[T: SampleType](x: var openArray[T], plan: IirPlan[T],
    ws: var Workspace) {.raises: [].} =
  ## Filters one trace, `x` (its live samples, at most `plan.samples`; an
  ## `AssertionDefect` otherwise), in place. `ws` must hold
  ## `plan.workspaceSize`. An empty `x` is left as it is.
  plan.checkPlan
  doAssert x.len <= plan.samples, "a trace longer than the plan's samples"
  let n = x.len
  if n == 0: return
  let m = plan.span(n)
  let buf = ws.scratch(float64, m, 0)
  let xp = cast[ptr UncheckedArray[T]](addr x[0])
  let (a, b) = trendOf(plan.trend, xp, n)
  plan.packInto(xp, n, a, b, buf, 1)
  plan.runSingle(buf, m)
  plan.unpackFrom(buf, 1, a, b, xp, n)

proc applyFilter*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    plan: IirPlan[T], ws: var Workspace) =
  ## Filters the live rows of `b` in place, each to its live length,
  ## `fftLanes` rows at a time (module documentation). `ws` must hold
  ## `plan.workspaceSize`. Raises `ValueError`, before any sample changes,
  ## when `b` does not match `meta` or has more columns than `plan.samples`.
  meta.checkShape(b)
  plan.checkPlan
  if b.cols > plan.samples:
    badIir("the block has " & $b.cols & " columns, the plan takes " &
      $plan.samples & " samples")
  let buf = ws.scratch(float64, plan.span(plan.samples) * fftLanes, 0)
  var lane = 0
  var rowOf, lenOf: array[fftLanes, int]
  var ta, tb: array[fftLanes, float64]
  var m = 0 # the longest padded length in the batch

  template flush() =
    if lane > 0:
      var spans: array[fftLanes, int]
      for l in 0..<fftLanes: # shorter traces and unused lanes: zeros after
        spans[l] = if l < lane: plan.span(lenOf[l]) else: 0
        for j in spans[l]..<m: buf[j * fftLanes + l] = 0
      plan.runLanes(buf, m, spans, lane)
      for l in 0..<lane:
        plan.unpackFrom(cast[Buf](addr buf[l]), fftLanes, ta[l], tb[l],
          b.rowPtr(rowOf[l]), lenOf[l])
      lane = 0
      m = 0

  for r in 0..<b.len:
    if not meta.isLive(r): continue
    let n = meta.liveSamples(r, b.cols)
    if n == 0: continue
    let src = b.rowPtr(r)
    (ta[lane], tb[lane]) = trendOf(plan.trend, src, n)
    plan.packInto(src, n, ta[lane], tb[lane], cast[Buf](addr buf[lane]), fftLanes)
    rowOf[lane] = r
    lenOf[lane] = n
    m = max(m, plan.span(n))
    inc lane
    if lane == fftLanes: flush()
  flush()

proc applyFilter*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    plan: IirPlan[T], ws: var Workspace) =
  ## `applyFilter` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  rowBlock(samples).applyFilter(meta, plan, ws)
