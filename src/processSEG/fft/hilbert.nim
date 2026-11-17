## The Hilbert transform and the envelope of a trace (`HilbertPlan`),
## through the real FFT: the analytic signal `z = x + i·H[x]`, whose modulus
## is the envelope and whose argument is the instantaneous phase.
##
## ```nim
## let hp = initHilbertPlan(float64, samples = 4001)
## var ws = initWorkspace()
## ws.reserve(hp.workspaceSize)
## var h, env = newSeq[float64](4001)
## hp.hilbertInto(x, h, ws)            # H[cos] = sin
## hp.envelopeInto(x, env, ws)         # |x + i·H[x]|
## ```
##
## **Definition.** `H` multiplies the spectrum by `−i·sign(f)`: it turns
## `cos(ωt)` into `sin(ωt)` and `sin(ωt)` into `−cos(ωt)`, and sets the 0 Hz
## and Nyquist bins to zero. The analytic signal `x + i·H[x]` then has no
## negative frequencies, and for `x = a(t)·cos(φ(t))` with an envelope `a`
## slower than the carrier it is `a(t)·e^{iφ(t)}`.
##
## **Ends.** The trace is zero padded to `nfft ≥ 2·samples` (by default
## `nextFastLen(2·samples)`), so the transform does not wrap: the result is
## the Hilbert transform of the trace with zeros on both sides. A trace that
## does not start and end near zero (an untapered sweep) makes `H` ring
## near its ends, over a few periods of its lowest frequency; inside, the
## transform of band-limited content is exact to rounding.
##
## **Lengths.** `x` holds up to `samples` values (zeros are assumed after
## it); the output has `x.len` values. Nothing allocates once the workspace is
## reserved, and nothing raises: a NaN in `x` makes the whole output NaN.

import std/math
import parseSEG
import ../core/workspace
import plan

{.localPassC: "-fno-math-errno".} # the envelope's sqrt vectorizes (the C file of this module)

type
  HilbertPlan*[T: SampleType] = object
    ## The Hilbert transform of traces of up to `samples` samples; see the
    ## module documentation.
    fft: FftPlan[T]
    samples: int

  Buf[T] = ptr UncheckedArray[T]

proc initHilbertPlan*(T: typedesc[SampleType], samples: int, nfft = 0): HilbertPlan[T] =
  ## A plan for traces of up to `samples` samples, zero padded to `nfft`
  ## (0: `nextFastLen(2·samples)`). Raises `ValueError` for `samples < 1`, or
  ## an `nfft` below `2·samples` or not 2·3·5-smooth.
  if samples < 1:
    raise newException(ValueError, "a Hilbert plan needs samples ≥ 1, got " & $samples)
  let n = if nfft == 0: nextFastLen(2 * samples) else: nfft
  if n < 2 * samples:
    raise newException(ValueError, "nfft " & $n & " is below twice the samples (" &
      $(2 * samples) & "): the transform would wrap")
  result.fft = initFftPlan(T, n) # raises for a length that is not smooth
  result.samples = samples

func samples*[T](hp: HilbertPlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  hp.samples

func nfft*[T](hp: HilbertPlan[T]): int {.inline, raises: [].} =
  ## The FFT length.
  hp.fft.len

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func workspaceSize*[T](hp: HilbertPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `hilbertInto` and `envelopeInto` need.
  let extra = hp.fft.len + 2 * hp.fft.bins
  result = hp.fft.workspaceSize(batched = false)
  when T is float32: result.f32 += extra
  else: result.f64 += extra

# ------------------------------------------------------------------ kernels --

proc rotateBins[T](re {.noalias.}, im {.noalias.}: Buf[T], n: int) {.raises: [].} =
  ## `X·(−i)` for the bins `1 ..< n`: (re, im) becomes (im, −re).
  {.push boundChecks: off, overflowChecks: off.}
  for k in 1 ..< n:
    let a = re[k]
    re[k] = im[k]
    im[k] = -a
  {.pop.}

proc envelopeKernel(x, h: ptr UncheckedArray[float32], env {.noalias.}: ptr UncheckedArray[
    float32], n: int) {.noinline, raises: [].} =
  ## `env[j] = sqrt(x[j]² + h[j]²)`, float32 (non-generic, so this module's
  ## `-fno-math-errno` applies and the square roots are packed).
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0 ..< n: env[j] = sqrt(x[j] * x[j] + h[j] * h[j])
  {.pop.}

proc envelopeKernel(x, h: ptr UncheckedArray[float64], env {.noalias.}: ptr UncheckedArray[
    float64], n: int) {.noinline, raises: [].} =
  ## The float64 form of `envelopeKernel`.
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0 ..< n: env[j] = sqrt(x[j] * x[j] + h[j] * h[j])
  {.pop.}

proc transform[T](hp: HilbertPlan[T], x: openArray[T], ws: var Workspace): Buf[T] {.
    raises: [].} =
  ## `H[x]` in the scratch's first `x.len` elements.
  let n = hp.fft.len
  let bins = hp.fft.bins
  let at = hp.fft.workspaceSize(batched = false).elems(T)
  let buf = ws.scratch(T, n + 2 * bins, at)
  let re = cast[ptr UncheckedArray[T]](addr buf[n])
  let im = cast[ptr UncheckedArray[T]](addr buf[n + bins])
  if x.len > 0: copyMem(buf, addr x[0], x.len * sizeof(T))
  zeroMem(addr buf[x.len], (n - x.len) * sizeof(T))
  hp.fft.forward(buf.toOpenArray(0, n - 1), re.toOpenArray(0, bins - 1),
    im.toOpenArray(0, bins - 1), ws)
  re[0] = T(0)
  im[0] = T(0)
  let last = if n mod 2 == 0: bins - 1 else: bins # Nyquist, when there is one, is 0
  rotateBins(re, im, last)
  if last < bins:
    re[last] = T(0)
    im[last] = T(0)
  hp.fft.inverse(re.toOpenArray(0, bins - 1), im.toOpenArray(0, bins - 1),
    buf.toOpenArray(0, n - 1), ws)
  buf

proc checkCall[T](hp: HilbertPlan[T], x: openArray[T], outLen: int, ws: Workspace) {.
    inline, raises: [].} =
  doAssert hp.samples > 0, "a HilbertPlan must come from initHilbertPlan"
  doAssert x.len <= hp.samples, "a trace longer than the plan's samples"
  doAssert outLen == x.len, "the output needs x.len values"
  doAssert ws.fits(hp.workspaceSize), "reserve the plan's workspaceSize first"

proc hilbertInto*[T: SampleType](hp: HilbertPlan[T], x: openArray[T],
    h: var openArray[T], ws: var Workspace) {.raises: [].} =
  ## The Hilbert transform of `x` (at most `samples` values) into `h`
  ## (`x.len` values): the imaginary part of the analytic signal. `ws` must
  ## hold `hp.workspaceSize`; other lengths are an `AssertionDefect`.
  hp.checkCall(x, h.len, ws)
  let buf = hp.transform(x, ws)
  if h.len > 0: copyMem(addr h[0], buf, h.len * sizeof(T))

proc envelopeInto*[T: SampleType](hp: HilbertPlan[T], x: openArray[T],
    env: var openArray[T], ws: var Workspace) {.raises: [].} =
  ## The envelope `|x + i·H[x]|` of `x` into `env` (`x.len` values). `ws`
  ## must hold `hp.workspaceSize`; other lengths are an `AssertionDefect`.
  hp.checkCall(x, env.len, ws)
  let buf = hp.transform(x, ws)
  if env.len > 0:
    envelopeKernel(cast[Buf[T]](addr x[0]), buf, cast[Buf[T]](addr env[0]), env.len)

proc wrapPi(d: float64): float64 {.inline.} =
  ## `d` brought into (−π, π].
  const two = 2 * PI
  result = d - two * floor((d + PI) / two)
  if result <= -PI: result += two
  elif result > PI: result -= two

proc analyticInto*[T: SampleType](hp: HilbertPlan[T], x: openArray[T],
    dtMs: float64, env, phaseDeg, hz: var openArray[float64],
    ws: var Workspace, floorDb = -60.0) {.raises: [].} =
  ## The envelope, the instantaneous phase and the instantaneous frequency
  ## of `x` from one Hilbert transform. `env[j] = |x + i·H[x]|`. `phaseDeg`
  ## is `arg(x + i·H[x])` in (−180, 180], NaN where the envelope is 0.
  ## `hz[j]` is the circular mean of the two one-sample phase steps around
  ## sample `j` (`z[j]·conj(z[j-1])` and `z[j+1]·conj(z[j])`, each step taken
  ## in (−π, π]), in Hz (`dtMs` is the sample interval). So it is right for
  ## every frequency from 0 up to Nyquist for a clean tone (a tone at exactly
  ## Nyquist reads +Nyquist). Noise or a second component near Nyquist makes
  ## the two steps differ; the circular mean keeps them on the same side of
  ## the wrap, but the reading is then only as good as the phase is, and for a real signal positive for
  ## `cos(2πft)` with `f > 0` (a negative value is a falling frequency, as in a
  ## down sweep). The value belongs to sample `j` itself; it is NaN at the
  ## first and last samples, and where that sample or either neighbour has
  ## envelope 0 or below `peak·10^(floorDb/20)` (`peak` the largest envelope;
  ## a zero trace is all NaN, never an infinity). All three outputs
  ## have `x.len` values. `ws` must hold `hp.workspaceSize`. A `dtMs` that is
  ## not finite and positive, or another length, is an `AssertionDefect`.
  ## Nothing allocates.
  hp.checkCall(x, env.len, ws)
  doAssert phaseDeg.len == x.len and hz.len == x.len, "env, phase and hz need x.len values"
  doAssert dtMs > 0 and dtMs < Inf, "dtMs must be finite and > 0"
  let n = x.len
  if n == 0: return
  let h = hp.transform(x, ws)
  var peak = 0.0
  for j in 0 ..< n:
    let e = hypot(float64(x[j]), float64(h[j]))
    env[j] = e
    if e > peak: peak = e
    if e > 0:
      var deg = arctan2(float64(h[j]), float64(x[j])) * (180 / PI)
      if deg <= -180: deg += 360
      phaseDeg[j] = deg
    else:
      phaseDeg[j] = NaN
  let thr = peak * pow(10.0, floorDb / 20.0)
  let scale = 1000 / (4 * PI * dtMs) # two steps' sum in radians per sample, to Hz
  hz[0] = NaN
  if n > 1: hz[n - 1] = NaN
  for j in 1 ..< n - 1:
    let e0 = env[j - 1]
    let e1 = env[j]
    let e2 = env[j + 1]
    if e0 <= 0 or e1 <= 0 or e2 <= 0 or e0 < thr or e1 < thr or e2 < thr:
      hz[j] = NaN
    else:
      # Each step is the argument of z[n]·conj(z[n-1]): the phase difference
      # wrapped to (−π, π], so a step up to Nyquist (π per sample) is exact.
      let back = wrapPi((phaseDeg[j] - phaseDeg[j - 1]) * PI / 180)
      let fwd = wrapPi((phaseDeg[j + 1] - phaseDeg[j]) * PI / 180)
      # The circular mean of the two steps: near ±π noise can put them on
      # opposite sides of the wrap, and their plain mean would read about 0.
      hz[j] = wrapPi(back + 0.5 * wrapPi(fwd - back)) * (2 * scale)
