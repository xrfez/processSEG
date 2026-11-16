## Correlation of a trace with a reference, both sides, placed where the
## caller wants the zero lag (`TraceCorrelationPlan`). Not the vibrator
## correlation (`CorrelationPlan`, `CrossCorrelationPlan`): those keep their
## own windows and scales, and this module does not call them.
##
## ```nim
## let plan = initTraceCorrelationPlan(float64, dtMs = 2, samples = 2001)
## var ws = initWorkspace()
## ws.reserve(plan)
## var dst = newSeq[float64](trace.len)
## plan.correlateAtInto(trace, reference, dst, zeroLagMs = 1000, ws)  # peak at 1000 ms
## ```
##
## **Lags.** `dst` has the trace's length. Sample `j` holds the correlation
## at lag `j − round(zeroLagMs / dtMs)` samples: negative lags before the
## asked time, positive lags after. A positive lag is a signal delayed
## against the reference (`Σ x[i + m]·ref[i]`), so a copy of the reference
## delayed by τ peaks at `zeroLagMs + τ`. Where the lag has no overlap, the
## sample is 0 — not a wrapped value. `zeroLagMs` is rounded to the nearest
## sample; lags themselves are whole samples.
##
## **Scale** (`CorrNorm`). `corrNone` is the raw sum. `corrCoefficient`
## divides by `√(E_x · E_ref)`, so an autocorrelation peaks at 1, and so does
## a cross-correlation of identical traces. A trace with no energy, or a NaN
## or infinity in either, gives NaN under `corrCoefficient`; under `corrNone`
## a non-finite sample gives NaN and a pair of zeros gives 0.
##
## The correlation is an FFT of both traces zero-padded to
## `nextFastLen(2·samples)`, so no lag wraps. The constructor allocates;
## `correlateAtInto` does not.

import std/math
import parseSEG
import ../core/workspace
import plan

type
  CorrNorm* = enum
    ## How `correlateAtInto` scales the correlation.
    corrNone = "none"               ## The raw sum.
    corrCoefficient = "coefficient" ## Divided by `√(E_x·E_ref)`: identical traces peak at 1.

  TraceCorrelationPlan*[T: SampleType] = object
    ## Correlation of traces of up to `samples` samples, both lags, placed
    ## at a caller time. Build it with `initTraceCorrelationPlan`.
    fft: FftPlan[T]
    dtMs: float64
    samples: int

proc badPlan(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc initTraceCorrelationPlan*(T: typedesc[SampleType], dtMs: float64,
    samples: int): TraceCorrelationPlan[T] =
  ## A plan for traces of up to `samples` samples at `dtMs`. Raises
  ## `ValueError` for `samples < 1` or a `dtMs` that is not finite and
  ## positive.
  if samples < 1: badPlan("a correlation needs samples ≥ 1, got " & $samples)
  if not (dtMs > 0 and dtMs < Inf):
    badPlan("dtMs must be finite and > 0, got " & $dtMs)
  result.fft = initFftPlan(T, nextFastLen(2 * samples))
  result.dtMs = dtMs
  result.samples = samples

func samples*[T](p: TraceCorrelationPlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  p.samples

func dtMs*[T](p: TraceCorrelationPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval.
  p.dtMs

func nfft*[T](p: TraceCorrelationPlan[T]): int {.inline, raises: [].} =
  ## The padded FFT length.
  p.fft.len

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func workspaceSize*[T](p: TraceCorrelationPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `correlateAtInto` needs.
  let extra = p.fft.len + 6 * p.fft.bins
  result = p.fft.workspaceSize(batched = false)
  when T is float32: result.f32 += extra
  else: result.f64 += extra

proc energy[T](x: openArray[T]): float64 {.inline.} =
  for v in x:
    let a = float64(v)
    if a.classify in {fcNan, fcInf, fcNegInf}: return NaN
    result += a * a

proc overlap(m, nx, nr: int): int {.inline.} =
  if m >= 0: max(0, min(nr, nx - m)) else: max(0, min(nx, nr + m))

proc correlateAtInto*[T: SampleType](p: TraceCorrelationPlan[T], x, reference: openArray[T],
    dst: var openArray[T], zeroLagMs: float64, ws: var Workspace,
    norm = corrCoefficient) {.raises: [].} =
  ## The correlation of `x` with `reference` into `dst` (`x.len` values),
  ## zero lag at `zeroLagMs` (module documentation). Both traces are at most
  ## `samples`. `ws` holds `workspaceSize`. A `zeroLagMs` that is not finite
  ## is an `AssertionDefect`. Nothing allocates.
  doAssert p.samples > 0, "a TraceCorrelationPlan must come from initTraceCorrelationPlan"
  doAssert x.len <= p.samples and reference.len <= p.samples, "a trace longer than the plan"
  doAssert dst.len == x.len, "dst needs x.len values"
  doAssert zeroLagMs.classify notin {fcNan, fcInf, fcNegInf}, "zeroLagMs must be finite"
  doAssert ws.fits(p.workspaceSize), "reserve the plan's workspaceSize first"
  let n = x.len
  if n == 0: return
  let ex = energy(x)
  let er = energy(reference)
  let bad = ex.isNaN or er.isNaN or (norm == corrCoefficient and not (ex > 0 and er > 0))
  if bad:
    for j in 0 ..< n: dst[j] = T(NaN)
    return
  let scale = if norm == corrCoefficient: 1 / sqrt(ex * er) else: 1.0
  let nfft = p.fft.len
  let bins = p.fft.bins
  let at = p.fft.workspaceSize(batched = false).elems(T)
  let buf = ws.scratch(T, nfft + 6 * bins, at)
  let xr = cast[ptr UncheckedArray[T]](addr buf[nfft])
  let xi = cast[ptr UncheckedArray[T]](addr buf[nfft + bins])
  let rr = cast[ptr UncheckedArray[T]](addr buf[nfft + 2 * bins])
  let ri = cast[ptr UncheckedArray[T]](addr buf[nfft + 3 * bins])
  let cr = cast[ptr UncheckedArray[T]](addr buf[nfft + 4 * bins])
  let ci = cast[ptr UncheckedArray[T]](addr buf[nfft + 5 * bins])
  template spectrum(src: openArray[T], re, im: ptr UncheckedArray[T]) =
    if src.len > 0: copyMem(buf, addr src[0], src.len * sizeof(T))
    zeroMem(addr buf[src.len], (nfft - src.len) * sizeof(T))
    p.fft.forward(buf.toOpenArray(0, nfft - 1), re.toOpenArray(0, bins - 1),
      im.toOpenArray(0, bins - 1), ws)
  spectrum(x, xr, xi)
  spectrum(reference, rr, ri)
  {.push boundChecks: off.}
  for k in 0 ..< bins:
    cr[k] = xr[k] * rr[k] + xi[k] * ri[k]
    ci[k] = xi[k] * rr[k] - xr[k] * ri[k]
  {.pop.}
  p.fft.inverse(cr.toOpenArray(0, bins - 1), ci.toOpenArray(0, bins - 1),
    buf.toOpenArray(0, nfft - 1), ws)
  let z = int(round(zeroLagMs / p.dtMs))
  for j in 0 ..< n:
    let m = j - z
    if overlap(m, n, reference.len) == 0:
      dst[j] = T(0)
    elif m >= 0:
      dst[j] = T(float64(buf[m]) * scale)
    else:
      dst[j] = T(float64(buf[nfft + m]) * scale)
