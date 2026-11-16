## A time shift of one trace by any number of milliseconds (`TimeShiftPlan`),
## the trace keeping its length: samples shifted in are zero, samples shifted
## out are dropped.
##
## ```nim
## let plan = initTimeShiftPlan(float64, dtMs = 2, samples = 2001)
## var ws = initWorkspace()
## ws.reserve(plan)
## plan.shiftMsInto(src, dst, shiftMs = 6, ws)   # 3 samples earlier
## plan.shiftMs(row, shiftMs = -4, ws)           # 2 samples later, in place
## ```
##
## **Sign.** As `shiftSamples`: a positive shift moves events earlier
## (`dst[j]` is the old sample at `j + shift/dt`). A negative shift moves
## them later. A shift of a whole number of samples (within 1e-6 of a sample)
## is a pure move, exact for every value including NaN. Anything else is a
## band-limited shift: the trace is zero-padded to `nextFastLen(2·samples)`
## and the spectrum is multiplied by a phase ramp, so the shift does not
## wrap. A shift of a whole trace or more zeroes it. A NaN or infinite shift
## leaves the samples unchanged.
##
## **Edges.** The pad is zeros, so a fractional shift is the sinc
## interpolant of the trace with zeros on both sides. A trace that does not
## start and end at zero rings near the ends, over a few samples of its
## highest frequency; the interior of a band-limited event is the shifted
## event. The real FFT cannot carry a fractional phase at Nyquist, so a tone
## at Nyquist is not shifted fractionally (its bin stays real). Whole-sample
## shifts have none of this: they move bytes.
##
## `shiftSamples` is the other shift: linear or cubic, used by statics and
## LMO, not band-limited. This one is the viewer's trace shift.

import std/math
import parseSEG
import ../core/workspace
import plan

type
  TimeShiftPlan*[T: SampleType] = object
    ## A band-limited shift of traces of up to `samples` samples. Build it
    ## with `initTimeShiftPlan`.
    fft: FftPlan[T]
    dtMs: float64
    samples: int

proc initTimeShiftPlan*(T: typedesc[SampleType], dtMs: float64, samples: int): TimeShiftPlan[T] =
  ## A plan for traces of up to `samples` samples at `dtMs`. The FFT is
  ## `nextFastLen(2·samples)`, long enough that a shift within the trace does
  ## not wrap. Raises `ValueError` for `samples < 1` or a `dtMs` that is not
  ## finite and positive.
  if samples < 1:
    raise newException(ValueError, "a shift plan needs samples ≥ 1, got " & $samples)
  if not (dtMs > 0 and dtMs < Inf):
    raise newException(ValueError, "dtMs must be finite and > 0, got " & $dtMs)
  result.fft = initFftPlan(T, nextFastLen(2 * samples))
  result.dtMs = dtMs
  result.samples = samples

func samples*[T](p: TimeShiftPlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  p.samples

func dtMs*[T](p: TimeShiftPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval.
  p.dtMs

func nfft*[T](p: TimeShiftPlan[T]): int {.inline, raises: [].} =
  ## The padded FFT length.
  p.fft.len

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func workspaceSize*[T](p: TimeShiftPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `shiftMsInto` needs, including a copy for an in-place call.
  let extra = p.samples + p.fft.len + 2 * p.fft.bins
  result = p.fft.workspaceSize(batched = false)
  when T is float32: result.f32 += extra
  else: result.f64 += extra

proc moveInto[T](src: openArray[T], dst: var openArray[T], k: int) {.raises: [].} =
  ## `dst[j] = src[j + k]`, zero outside. `src` and `dst` may be the same
  ## buffer; a positive `k` moves toward the start.
  let n = dst.len
  let p = cast[ptr UncheckedArray[T]](addr dst[0])
  if k == 0:
    if cast[pointer](addr src[0]) != cast[pointer](p):
      copyMem(p, addr src[0], n * sizeof(T))
    return
  if cast[pointer](addr src[0]) == cast[pointer](p):
    if k > 0:
      moveMem(p, addr p[k], (n - k) * sizeof(T))
      zeroMem(addr p[n - k], k * sizeof(T))
    else:
      moveMem(addr p[-k], p, (n + k) * sizeof(T))
      zeroMem(p, -k * sizeof(T))
    return
  if k > 0:
    copyMem(p, addr src[k], (n - k) * sizeof(T))
    zeroMem(addr p[n - k], k * sizeof(T))
  else:
    copyMem(addr p[-k], addr src[0], (n + k) * sizeof(T))
    zeroMem(p, -k * sizeof(T))

proc rampShift[T](p: TimeShiftPlan[T], src: openArray[T], dst: var openArray[T],
    s: float64, ws: var Workspace) {.raises: [].} =
  ## Fractional shift by `s` samples through a zero-padded phase ramp.
  let n = src.len
  let nfft = p.fft.len
  let bins = p.fft.bins
  let at = p.fft.workspaceSize(batched = false).elems(T)
  let buf = ws.scratch(T, nfft + 2 * bins, at)
  if n > 0: copyMem(buf, addr src[0], n * sizeof(T))
  zeroMem(addr buf[n], (nfft - n) * sizeof(T))
  let re = cast[ptr UncheckedArray[T]](addr buf[nfft])
  let im = cast[ptr UncheckedArray[T]](addr buf[nfft + bins])
  p.fft.forward(buf.toOpenArray(0, nfft - 1), re.toOpenArray(0, bins - 1),
    im.toOpenArray(0, bins - 1), ws)
  {.push boundChecks: off.}
  for k in 0 ..< bins:
    let theta = 2 * PI * float64(k) * s / float64(nfft)
    let c = cos(theta)
    let sn = sin(theta)
    let a = float64(re[k])
    let b = float64(im[k])
    re[k] = T(a * c - b * sn)
    im[k] = T(a * sn + b * c)
  {.pop.}
  p.fft.inverse(re.toOpenArray(0, bins - 1), im.toOpenArray(0, bins - 1),
    buf.toOpenArray(0, nfft - 1), ws)
  if n > 0: copyMem(addr dst[0], buf, n * sizeof(T))

proc shiftMsInto*[T: SampleType](p: TimeShiftPlan[T], src: openArray[T],
    dst: var openArray[T], shiftMs: float64, ws: var Workspace) {.raises: [].} =
  ## `src` shifted by `shiftMs` into `dst` (module documentation). Both have
  ## the same length, at most `samples`. `ws` holds `workspaceSize`. A NaN or
  ## infinite shift copies `src`. Nothing allocates.
  doAssert p.samples > 0, "a TimeShiftPlan must come from initTimeShiftPlan"
  doAssert src.len <= p.samples and dst.len == src.len, "dst needs src.len values, within the plan"
  doAssert ws.fits(p.workspaceSize), "reserve the plan's workspaceSize first"
  let n = src.len
  if n == 0: return
  if shiftMs.classify in {fcNan, fcInf, fcNegInf}:
    if cast[pointer](addr dst[0]) != cast[pointer](addr src[0]):
      copyMem(addr dst[0], addr src[0], n * sizeof(T))
    return
  let s = shiftMs / p.dtMs
  if abs(s) >= float64(n):
    zeroMem(addr dst[0], n * sizeof(T))
    return
  let nearest = round(s)
  if abs(s - nearest) <= 1e-6:
    moveInto(src, dst, int(nearest))
    return
  let same = cast[pointer](addr dst[0]) == cast[pointer](addr src[0])
  if same:
    let at = p.fft.workspaceSize(batched = false).elems(T) + p.fft.len + 2 * p.fft.bins
    let tmp = ws.scratch(T, n, at)
    copyMem(tmp, addr src[0], n * sizeof(T))
    p.rampShift(tmp.toOpenArray(0, n - 1), dst, s, ws)
  else:
    p.rampShift(src, dst, s, ws)

proc shiftMs*[T: SampleType](p: TimeShiftPlan[T], x: var openArray[T],
    shiftMs: float64, ws: var Workspace) {.raises: [].} =
  ## `shiftMsInto` in place.
  p.shiftMsInto(x, x, shiftMs, ws)
