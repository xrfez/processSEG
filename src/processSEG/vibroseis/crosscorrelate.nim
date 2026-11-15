## A signal against its reference (`CrossCorrelationPlan`): their
## cross-correlation over the lags around zero, its envelope, the spectrum
## and phase of that lag window, and two signals' amplitude ratio and phase
## difference against frequency. For a vibrator's ground force, mass or
## baseplate signal against its pilot, or one unit against another.
##
## ```nim
## let cp = initCrossCorrelationPlan(float64, maxLagMs = 500, dtMs = 1, samples = 8001)
## var ws = initWorkspace()
## ws.reserve(cp.workspaceSize)
## var lags, envelope = newSeq[float64](cp.lags)   # at cp.lagMs(i)
## cp.crossCorrelateInto(groundForce, pilot, lags, envelope, ws)
## var amp, phase = newSeq[float64](cp.bins)        # at cp.binHz(k)
## cp.lagSpectrumInto(lags, amp, phase, ws)
## ```
##
## **Correlation.** `c(m) = Σ_j s[j + m]·r[j]` for the lags `m = −L .. L`
## (`L = maxLagMs / dtMs`, rounded): a signal that is the reference delayed by
## τ peaks at lag +τ. It is computed by FFT with both rows zero padded to at
## least twice the samples, so no lag wraps. `lags[i]` is lag `i − L`, at
## `lagMs(i)`. Normalized (`CorrelationNorm`) as a correlation coefficient by
## default (`cnCoefficient`: divided by `√(Σs²·Σr²)`, so it is 1 at the lag of
## a signal of the reference's shape, whatever the units of either), by the
## reference's energy (`cnReference`: divided by `Σr²`, so a reference scaled
## by `a` peaks at `a`, as `CorrelationPlan` normalizes), or not (`cnNone`). A
## row without energy, or with a NaN, gives NaN lags.
##
## **Envelope.** The modulus of the whole correlation's analytic signal (its
## Hilbert transform from the same spectrum), cut to the lags: it peaks at
## the delay whatever the phase between the two, where the correlation itself
## peaks only for signals in phase.
##
## **Spectrum and phase of the correlation** (`lagSpectrumInto`). The DFT of
## the lag window, tapered at its ends (a Tukey taper, `taperFraction` of the
## window each end), with the phase measured from lag zero, in degrees in
## (−180, 180]: a signal delayed by τ against its reference reads −360·f·τ,
## the reference against itself 0. It is the Blackman-Tukey estimate of the
## cross-spectrum, smoothed over `1 / (2·maxLagMs)` Hz: its phase is the
## signal's phase relative to the reference against frequency, its amplitude
## the product of their amplitude spectra (shown in dB against its peak).
## `nfft` (default the next power of two of `2L + 1`) sets the bins, `binHz`.
##
## **Comparison** (`compareInto`). Two signals of one record, `a` against
## `b`: per bin, `10·log10(Ŝ_aa / Ŝ_bb)` dB (the ratio of their amplitude
## spectra) and `arg Ŝ_ab` degrees (the phase of `a` minus that of `b`), from
## the lag-windowed auto- and cross-spectra above. Outside the band where both
## have energy the values are noise.
##
## **Allocation.** The constructor allocates; the calls do not, once the
## workspace holds `workspaceSize`, and they never raise: other lengths are
## an `AssertionDefect`.

import std/math
import parseSEG
import ../core/workspace
import ../fft/plan

type
  CorrelationNorm* = enum
    ## How `crossCorrelateInto` scales the correlation.
    cnCoefficient = "coefficient" ## Divided by `√(Σs²·Σr²)`: 1 for the reference's shape.
    cnReference = "reference"     ## Divided by `Σr²`: the reference scaled by `a` peaks at `a`.
    cnNone = "none"               ## Not scaled.

  CrossCorrelationPlan*[T: SampleType] = object
    ## Correlation of rows of up to `samples` samples over lags `−L .. L`;
    ## see the module documentation.
    fft: FftPlan[T] # the rows, zero padded to ≥ 2·samples
    lagFft: FftPlan[T] # the lag window
    taper: seq[T] # 2L + 1 weights
    samples, maxLag: int
    dtMs: float64

proc badPlan(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc initCrossCorrelationPlan*(T: typedesc[SampleType], maxLagMs, dtMs: float64,
    samples: int, taperFraction = 0.1, nfft = 0): CrossCorrelationPlan[T] =
  ## A plan correlating rows of up to `samples` samples at `dtMs` over the
  ## lags within `maxLagMs` of zero; the lag window's spectrum has `nfft`
  ## points (0: the next power of two of `2L + 1`). Raises `ValueError` for a
  ## `dtMs` that is not finite and positive, `samples < 2`, a `maxLagMs` under
  ## one sample or of `samples` and more, a `taperFraction` outside 0 .. 0.5,
  ## or an `nfft` below `2L + 1` or not 2·3·5-smooth.
  if not (dtMs > 0 and dtMs < Inf): badPlan("dtMs must be finite and positive, got " & $dtMs)
  if samples < 2: badPlan("a cross-correlation needs samples ≥ 2, got " & $samples)
  if not (maxLagMs >= dtMs and maxLagMs < Inf):
    badPlan("maxLagMs must be at least one sample (" & $dtMs & " ms), got " & $maxLagMs)
  let L = int(round(maxLagMs / dtMs))
  if L >= samples:
    badPlan("maxLagMs = " & $maxLagMs & " reaches past the " & $samples & " samples")
  if not (taperFraction >= 0 and taperFraction <= 0.5):
    badPlan("taperFraction must be in 0 .. 0.5, got " & $taperFraction)
  let lagN = if nfft == 0: nextPowerOfTwo(2 * L + 1) else: nfft
  if lagN < 2 * L + 1:
    badPlan("nfft " & $lagN & " is shorter than the lag window (" & $(2 * L + 1) & ")")
  result.lagFft = initFftPlan(T, lagN) # raises for a length that is not smooth
  result.fft = initFftPlan(T, nextFastLen(2 * samples))
  result.samples = samples
  result.maxLag = L
  result.dtMs = dtMs
  let n = 2 * L + 1
  let ramp = int(round(taperFraction * float64(n)))
  result.taper = newSeq[T](n)
  for i in 0 ..< n:
    let d = min(i, n - 1 - i)
    result.taper[i] = T(if d >= ramp: 1.0 else: sin(PI / 2 * float64(d + 1) /
      float64(ramp + 1)) ^ 2)

func samples*[T](cp: CrossCorrelationPlan[T]): int {.inline, raises: [].} =
  ## The longest row the plan takes.
  cp.samples

func maxLag*[T](cp: CrossCorrelationPlan[T]): int {.inline, raises: [].} =
  ## `L`, the largest lag in samples.
  cp.maxLag

func lags*[T](cp: CrossCorrelationPlan[T]): int {.inline, raises: [].} =
  ## The number of lags, `2L + 1`.
  2 * cp.maxLag + 1

func lagMs*[T](cp: CrossCorrelationPlan[T], i: int): float64 {.inline, raises: [].} =
  ## The lag of `lags[i]`, in ms: `(i − L)·dtMs`.
  float64(i - cp.maxLag) * cp.dtMs

func dtMs*[T](cp: CrossCorrelationPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval.
  cp.dtMs

func nfft*[T](cp: CrossCorrelationPlan[T]): int {.inline, raises: [].} =
  ## The lag window's FFT length.
  cp.lagFft.len

func bins*[T](cp: CrossCorrelationPlan[T]): int {.inline, raises: [].} =
  ## The bins of the lag window's spectrum, `nfft div 2 + 1`.
  cp.lagFft.bins

func binHz*[T](cp: CrossCorrelationPlan[T], k: int): float64 {.inline, raises: [].} =
  ## The frequency of bin `k`, Hz.
  float64(k) * 1000 / (float64(cp.lagFft.len) * cp.dtMs)

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func fftScratch[T](cp: CrossCorrelationPlan[T]): int {.inline.} =
  max(cp.fft.workspaceSize(batched = false).elems(T),
    cp.lagFft.workspaceSize(batched = false).elems(T))

func ownScratch[T](cp: CrossCorrelationPlan[T]): int {.inline.} =
  # A trace, two spectra and a product; the lag window and three spectra.
  cp.fft.len + 6 * cp.fft.bins + cp.lagFft.len + 6 * cp.lagFft.bins

func workspaceSize*[T](cp: CrossCorrelationPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch the calls need.
  let n = cp.fftScratch + cp.ownScratch
  when T is float32: WorkspaceSize(f32: n)
  else: WorkspaceSize(f64: n)

# ------------------------------------------------------------------ kernels --

{.push boundChecks: off, overflowChecks: off.}

proc crossKernel[T](xr {.noalias.}, xi {.noalias.}, yr {.noalias.}, yi {.noalias.},
    cr {.noalias.}, ci {.noalias.}: ptr UncheckedArray[T], n: int) {.raises: [].} =
  ## `c = x·conj(y)`, bin by bin.
  for k in 0 ..< n:
    cr[k] = xr[k] * yr[k] + xi[k] * yi[k]
    ci[k] = xi[k] * yr[k] - xr[k] * yi[k]

{.pop.}

type Scratch[T] = object
  t, ar, ai, br, bi, cr, ci, lag, s1r, s1i, s2r, s2i, s3r, s3i: ptr UncheckedArray[T]

proc scratchOf[T](cp: CrossCorrelationPlan[T], ws: var Workspace): Scratch[T] {.raises: [].} =
  let n = cp.fft.len
  let b = cp.fft.bins
  let ln = cp.lagFft.len
  let lb = cp.lagFft.bins
  let base = ws.scratch(T, cp.ownScratch, cp.fftScratch)
  var at = 0
  proc next(base: ptr UncheckedArray[T], at: var int, count: int): ptr UncheckedArray[T] {.nimcall.} =
    result = cast[ptr UncheckedArray[T]](addr base[at])
    at += count
  result.t = next(base, at, n)
  result.ar = next(base, at, b)
  result.ai = next(base, at, b)
  result.br = next(base, at, b)
  result.bi = next(base, at, b)
  result.cr = next(base, at, b)
  result.ci = next(base, at, b)
  result.lag = next(base, at, ln)
  result.s1r = next(base, at, lb)
  result.s1i = next(base, at, lb)
  result.s2r = next(base, at, lb)
  result.s2i = next(base, at, lb)
  result.s3r = next(base, at, lb)
  result.s3i = next(base, at, lb)

proc spectrumOf[T](cp: CrossCorrelationPlan[T], x: openArray[T], t, re, im: ptr UncheckedArray[T],
    ws: var Workspace) {.raises: [].} =
  ## The spectrum of `x`, zero padded to the plan's FFT, into `re, im`.
  let n = cp.fft.len
  if x.len > 0: copyMem(t, addr x[0], x.len * sizeof(T))
  zeroMem(addr t[x.len], (n - x.len) * sizeof(T))
  cp.fft.forward(t.toOpenArray(0, n - 1), re.toOpenArray(0, cp.fft.bins - 1),
    im.toOpenArray(0, cp.fft.bins - 1), ws)

proc inverseToLags[T](cp: CrossCorrelationPlan[T], s: Scratch[T], dst: ptr UncheckedArray[T],
    ws: var Workspace) {.raises: [].} =
  ## The inverse of the product `cr, ci` into `dst` at the lags −L .. L.
  let n = cp.fft.len
  cp.fft.inverse(s.cr.toOpenArray(0, cp.fft.bins - 1), s.ci.toOpenArray(0, cp.fft.bins - 1),
    s.t.toOpenArray(0, n - 1), ws)
  let L = cp.maxLag
  for i in 0 .. 2 * L:
    let m = i - L
    dst[i] = s.t[(if m < 0: m + n else: m)]

proc lagWindowSpectrum[T](cp: CrossCorrelationPlan[T], lags: ptr UncheckedArray[T], s: Scratch[T],
    re, im: ptr UncheckedArray[T], ws: var Workspace) {.raises: [].} =
  ## The DFT of the tapered lag window, lag zero at index 0.
  let ln = cp.lagFft.len
  let L = cp.maxLag
  zeroMem(s.lag, ln * sizeof(T))
  for i in 0 .. 2 * L:
    let m = i - L
    s.lag[(if m < 0: m + ln else: m)] = lags[i] * cp.taper[i]
  cp.lagFft.forward(s.lag.toOpenArray(0, ln - 1), re.toOpenArray(0, cp.lagFft.bins - 1),
    im.toOpenArray(0, cp.lagFft.bins - 1), ws)

func energy[T](x: openArray[T]): float64 {.inline.} =
  for v in x: result += float64(v) * float64(v)

proc checkRows[T](cp: CrossCorrelationPlan[T], a, b: openArray[T], ws: Workspace) {.
    inline, raises: [].} =
  doAssert cp.samples > 0, "a CrossCorrelationPlan must come from initCrossCorrelationPlan"
  doAssert a.len <= cp.samples and b.len <= cp.samples, "a row longer than the plan's samples"
  doAssert ws.fits(cp.workspaceSize), "reserve the plan's workspaceSize first"

proc correlate[T](cp: CrossCorrelationPlan[T], x, reference: openArray[T],
    lags: var openArray[T], envelope: ptr UncheckedArray[T], ws: var Workspace,
    norm: CorrelationNorm) {.raises: [].} =
  cp.checkRows(x, reference, ws)
  doAssert lags.len == cp.lags, "lags needs 2·maxLag + 1 values"
  let s = cp.scratchOf(ws)
  let nb = cp.fft.bins
  cp.spectrumOf(x, s.t, s.ar, s.ai, ws)
  cp.spectrumOf(reference, s.t, s.br, s.bi, ws)
  crossKernel(s.ar, s.ai, s.br, s.bi, s.cr, s.ci, nb)
  let scale = case norm
    of cnCoefficient: 1 / sqrt(energy(x) * energy(reference))
    of cnReference: 1 / energy(reference)
    of cnNone: 1.0
  let ok = scale > 0 and scale < Inf
  let L = cp.maxLag
  if not ok:
    for i in 0 .. 2 * L:
      lags[i] = T(NaN)
      if envelope != nil: envelope[i] = T(NaN)
    return
  cp.inverseToLags(s, cast[ptr UncheckedArray[T]](addr lags[0]), ws)
  for i in 0 .. 2 * L: lags[i] = T(float64(lags[i]) * scale)
  if envelope != nil:
    # The Hilbert transform of the correlation: its spectrum times −i·sign(f).
    s.cr[0] = T(0)
    s.ci[0] = T(0)
    for k in 1 ..< nb:
      let a = s.cr[k]
      s.cr[k] = s.ci[k]
      s.ci[k] = -a
    if cp.fft.len mod 2 == 0:
      s.cr[nb - 1] = T(0)
      s.ci[nb - 1] = T(0)
    cp.inverseToLags(s, envelope, ws)
    for i in 0 .. 2 * L:
      let h = float64(envelope[i]) * scale
      envelope[i] = T(sqrt(float64(lags[i]) ^ 2 + h * h))

proc crossCorrelateInto*[T: SampleType](cp: CrossCorrelationPlan[T], x,
    reference: openArray[T], lags: var openArray[T], ws: var Workspace,
    norm = cnCoefficient) {.raises: [].} =
  ## The correlation of `x` with `reference` (each at most `samples` values,
  ## zeros after) at the lags `−L .. L` into `lags` (`cp.lags` values; lag of
  ## `lags[i]`: `cp.lagMs(i)`), scaled by `norm`. `ws` holds
  ## `cp.workspaceSize`; other lengths are an `AssertionDefect`.
  cp.correlate(x, reference, lags, nil, ws, norm)

proc crossCorrelateInto*[T: SampleType](cp: CrossCorrelationPlan[T], x,
    reference: openArray[T], lags, envelope: var openArray[T], ws: var Workspace,
    norm = cnCoefficient) {.raises: [].} =
  ## The correlation and its envelope (`cp.lags` values each).
  doAssert envelope.len == cp.lags, "envelope needs 2·maxLag + 1 values"
  cp.correlate(x, reference, lags, cast[ptr UncheckedArray[T]](addr envelope[0]), ws, norm)

proc lagSpectrumInto*[T: SampleType](cp: CrossCorrelationPlan[T], lags: openArray[T],
    amp, phaseDeg: var openArray[float64], ws: var Workspace) {.raises: [].} =
  ## The amplitude and phase (degrees, from lag zero) of the tapered lag
  ## window `lags` (from `crossCorrelateInto`), `cp.bins` values each, at
  ## `cp.binHz(k)`. A NaN lag makes every bin NaN. Other lengths are an
  ## `AssertionDefect`.
  doAssert cp.samples > 0, "a CrossCorrelationPlan must come from initCrossCorrelationPlan"
  doAssert lags.len == cp.lags, "lags needs 2·maxLag + 1 values"
  doAssert amp.len == cp.bins and phaseDeg.len == cp.bins, "amp and phaseDeg need bins values"
  doAssert ws.fits(cp.workspaceSize), "reserve the plan's workspaceSize first"
  let s = cp.scratchOf(ws)
  cp.lagWindowSpectrum(cast[ptr UncheckedArray[T]](addr lags[0]), s, s.s1r, s.s1i, ws)
  for k in 0 ..< cp.bins:
    let re = float64(s.s1r[k])
    let im = float64(s.s1i[k])
    amp[k] = hypot(re, im)
    phaseDeg[k] = radToDeg(arctan2(im, re))
    if phaseDeg[k] == -180: phaseDeg[k] = 180

proc compareInto*[T: SampleType](cp: CrossCorrelationPlan[T], a, b: openArray[T],
    ratioDb, phaseDeg: var openArray[float64], ws: var Workspace) {.raises: [].} =
  ## Signal `a` against signal `b` (each at most `samples` values) per bin of
  ## the lag window's spectrum (`cp.bins` values each, at `cp.binHz(k)`): the
  ## ratio of their amplitude spectra, `10·log10(Ŝ_aa / Ŝ_bb)` dB, and the
  ## phase of `a` minus that of `b`, `arg Ŝ_ab` degrees in (−180, 180] (module
  ## documentation). NaN where either has no energy, or for a NaN in a row.
  ## Other lengths are an `AssertionDefect`.
  cp.checkRows(a, b, ws)
  doAssert ratioDb.len == cp.bins and phaseDeg.len == cp.bins,
    "ratioDb and phaseDeg need bins values"
  let s = cp.scratchOf(ws)
  let nb = cp.fft.bins
  let lb = cp.lagFft.bins
  cp.spectrumOf(a, s.t, s.ar, s.ai, ws)
  cp.spectrumOf(b, s.t, s.br, s.bi, ws)
  # Each product's lag window, then its spectrum: S_ab, S_aa, S_bb.
  # The lags of each product go to the start of `t` (read out through `lag`
  # first: the negative lags sit at its end), where `lagWindowSpectrum`
  # takes them while it fills `lag`.
  let lagBuf = s.t
  template windowed(xr, xi, yr, yi, outR, outI: untyped) =
    crossKernel(xr, xi, yr, yi, s.cr, s.ci, nb)
    cp.inverseToLags(s, s.lag, ws)
    copyMem(lagBuf, s.lag, cp.lags * sizeof(T))
    cp.lagWindowSpectrum(lagBuf, s, outR, outI, ws)
  windowed(s.ar, s.ai, s.br, s.bi, s.s1r, s.s1i)
  windowed(s.ar, s.ai, s.ar, s.ai, s.s2r, s.s2i)
  windowed(s.br, s.bi, s.br, s.bi, s.s3r, s.s3i)
  for k in 0 ..< lb:
    let paa = hypot(float64(s.s2r[k]), float64(s.s2i[k]))
    let pbb = hypot(float64(s.s3r[k]), float64(s.s3i[k]))
    ratioDb[k] = (if paa > 0 and pbb > 0: 10 * log10(paa / pbb) else: NaN)
    let ph = radToDeg(arctan2(float64(s.s1i[k]), float64(s.s1r[k])))
    phaseDeg[k] = (if ratioDb[k].isNaN: NaN elif ph == -180: 180.0 else: ph)
