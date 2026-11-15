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
