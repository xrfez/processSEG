## The f-k amplitude of a gather (`FkPlan`): a 2-D Fourier modulus, frequency
## fastest, signed wavenumber with zero in the middle and later-on-later
## traces on the positive side. Wavenumber units (cycles/km) stay in the
## viewer; `dkPerTrace` is cycles per trace.
##
## ```nim
## let n = fkRows(meta, cols, firstCol, nCols)
## let plan = initFkPlan(float32, dtMs = 2, traces = n, samples = nCols)
## var ws = initWorkspace()
## ws.reserve(plan)
## var amp = newSeq[float64](plan.nK * plan.nFreq)
## discard plan.fkAmplitudeInto(rowBlock(samples), meta, firstCol, amp, ws)
## ```
##
## **Transform.** Live rows that have a sample in the window are tapered on
## both axes (a Tukey ramp, `taperWeight(tpCos2, (d+0.5)/ramp)`), zero-padded
## in time to `nextFastLen(samples)` and across traces to
## `nextFastLen(2·traces)`, then transformed: a real FFT in time, and the
## complex FFT across traces as two real FFTs. The modulus is unscaled (the
## caller applies `toDbInto`). Dead, auxiliary and failed rows, and live rows
## with no sample in the window, are left out, so `dkPerTrace` is per trace
## that entered, in order.
##
## **Sign.** A event that arrives later on later traces (a positive delay per
## trace) peaks at positive k, to the right of zero (`nK div 2`). The spatial
## FFT's usual sign is flipped so that is so.
##
## **Allocation.** The constructor allocates the plans and the tapers.
## `fkAmplitudeInto` does not, once the workspace holds `workspaceSize`.
##
## **Stop.** `stop` (nil: never) is checked once per trace that enters and
## once per frequency of the spatial pass, never per sample. `Stopped` means
## the job was abandoned; `amp` is then unspecified. The plan and the
## workspace stay usable.

import std/math
import parseSEG
import ../core/[rows, types, workspace]
import plan

type
  FkAxes* = object
    ## The axes of an f-k amplitude. `amp[ik * nFreq + jf]` is wavenumber
    ## index `ik` (zero at `nK div 2`, positive k later) and frequency bin
    ## `jf`.
    nFreq*: int ## Frequency bins, time FFT length div 2 + 1.
    nK*: int ## Wavenumber bins, `nextFastLen(2·traces)`.
    dfHz*: float64 ## Hertz per frequency bin.
    dkPerTrace*: float64 ## Cycles per trace per wavenumber bin.

  FkPlan*[T: SampleType] = object
    ## An f-k transform of `traces` live rows by `samples` columns. Build it
    ## with `initFkPlan`.
    timeFft, spaceFft: FftPlan[T]
    timeW, traceW: seq[T]
    dtMs: float64
    traces, samples: int
    axes: FkAxes

proc badFk(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc tukey[T](n: int, fraction: float64): seq[T] =
  ## Tukey weights: 1 in the middle, a cos² ramp of `fraction` at each end
  ## (`0.5 − 0.5·cos`, the viewer's `taperWeight(tpCos2, (d+0.5)/ramp)`).
  result = newSeq[T](n)
  let ramp = if fraction <= 0: 0 else: int(round(fraction * float64(n)))
  for i in 0 ..< n:
    let d = min(i, n - 1 - i)
    let w = if ramp <= 0 or d >= ramp: 1.0
            else: 0.5 - 0.5 * cos(PI * (float64(d) + 0.5) / float64(ramp))
    result[i] = T(w)

proc initFkPlan*(T: typedesc[SampleType], dtMs: float64, traces, samples: int,
    taperFraction = 0.1): FkPlan[T] =
  ## An f-k plan for `traces` live rows and windows of `samples` columns at
  ## `dtMs`, tapered over `taperFraction` (0 .. 0.5) of each axis. Raises
  ## `ValueError` for `traces < 4`, `samples < 8`, a `dtMs` that is not
  ## finite and positive, or a taper fraction outside 0 .. 0.5.
  if traces < 4: badFk("an f-k gather needs ≥ 4 live rows, got " & $traces)
  if samples < 8: badFk("an f-k window needs ≥ 8 samples, got " & $samples)
  if not (dtMs > 0 and dtMs < Inf):
    badFk("dtMs must be finite and > 0, got " & $dtMs)
  if not (taperFraction >= 0 and taperFraction <= 0.5):
    badFk("taperFraction must be in 0 .. 0.5, got " & $taperFraction)
  let nTime = nextFastLen(samples)
  let nK = nextFastLen(2 * traces)
  result.timeFft = initFftPlan(T, nTime)
  result.spaceFft = initFftPlan(T, nK)
  result.timeW = tukey[T](samples, taperFraction)
  result.traceW = tukey[T](traces, taperFraction)
  result.dtMs = dtMs
  result.traces = traces
  result.samples = samples
  result.axes = FkAxes(nFreq: result.timeFft.bins, nK: nK,
    dfHz: 1000 / (float64(nTime) * dtMs), dkPerTrace: 1 / float64(nK))

func axes*[T](p: FkPlan[T]): FkAxes {.inline, raises: [].} =
  ## The frequency and wavenumber axes.
  p.axes

func nFreq*[T](p: FkPlan[T]): int {.inline, raises: [].} =
  ## Frequency bins.
  p.axes.nFreq

func nK*[T](p: FkPlan[T]): int {.inline, raises: [].} =
  ## Wavenumber bins.
  p.axes.nK

func dfHz*[T](p: FkPlan[T]): float64 {.inline, raises: [].} =
  ## Hertz per frequency bin.
  p.axes.dfHz

func dkPerTrace*[T](p: FkPlan[T]): float64 {.inline, raises: [].} =
  ## Cycles per trace per wavenumber bin.
  p.axes.dkPerTrace

func traces*[T](p: FkPlan[T]): int {.inline, raises: [].} =
  ## The live rows the plan was built for.
  p.traces

func samples*[T](p: FkPlan[T]): int {.inline, raises: [].} =
  ## The window length in columns.
  p.samples

func dtMs*[T](p: FkPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval.
  p.dtMs

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func workspaceSize*[T](p: FkPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `fkAmplitudeInto` needs.
  let fftWs = max(p.timeFft.workspaceSize(batched = false),
    p.spaceFft.workspaceSize(batched = false))
  let extra = p.timeFft.len + 2 * p.timeFft.bins + p.traces * p.timeFft.bins * 2 +
    2 * p.spaceFft.len + 4 * p.spaceFft.bins
  result = fftWs
  when T is float32: result.f32 += extra
  else: result.f64 += extra

proc fkRows*(meta: ChunkMeta, cols, firstCol, nCols: int, rows = 0 .. int.high): int {.
    raises: [].} =
  ## How many live rows of a block `cols` wide enter an f-k window
  ## `firstCol ..< firstCol + nCols` in `rows`: a live row with a sample at
  ## `firstCol` or later. The count `initFkPlan` wants. A negative `firstCol`
  ## counts nothing.
  if firstCol < 0 or nCols < 1: return 0
  let lo = max(rows.a, 0)
  let hi = min(rows.b, meta.len - 1)
  for r in lo .. hi:
    if meta.isLive(r) and meta.liveSamples(r, cols) > firstCol: inc result

proc fkAmplitudeInto*[T: SampleType](p: FkPlan[T], b: RowBlock[T], meta: ChunkMeta,
    firstCol: int, amp: var openArray[float64], ws: var Workspace,
    rows = 0 .. int.high, stop: StopCheck = nil): int =
  ## The f-k modulus of the window `firstCol ..< firstCol + samples` into
  ## `amp` (`nK·nFreq` values, frequency fastest, unscaled). Returns the live
  ## rows used. `stop` (nil: never) is asked once per trace that enters and
  ## once per frequency of the spatial pass, never per sample; true abandons
  ## the job and returns `Stopped`, and `amp` is then unspecified. The plan
  ## is unchanged and the workspace stays usable. Raises `ValueError`
  ## when `b` does not match `meta`, the interval is not the plan's,
  ## `firstCol` is negative, or the live rows in the window are not `traces`
  ## (build the plan for `fkRows`). An `amp` of another length is an
  ## `AssertionDefect`; a short workspace is an `IndexDefect`. Nothing
  ## allocates.
  ##
  ## A wrong `amp` length is an `AssertionDefect`, not the `ValueError` the
  ## viewer audit suggested: every other Into kernel treats a short buffer
  ## as a defect, and a `ValueError` there would be the odd one out.
  meta.checkShape(b)
  if not sameInterval(meta.dtMs, p.dtMs):
    badFk("the rows' interval " & $meta.dtMs & " ms is not the plan's " &
      $(p.dtMs) & " ms")
  if firstCol < 0: badFk("firstCol must be ≥ 0, got " & $firstCol)
  let nFreq = p.timeFft.bins
  let nK = p.spaceFft.len
  doAssert amp.len == nK * nFreq, "amp needs nK·nFreq values"
  doAssert ws.fits(p.workspaceSize), "reserve the plan's workspaceSize first"
  let entered = fkRows(meta, b.cols, firstCol, p.samples, rows)
  if entered != p.traces:
    badFk("the window has " & $entered & " live rows, the plan was built for " &
      $p.traces)
  let nTime = p.timeFft.len
  let sBins = p.spaceFft.bins
  let at = max(p.timeFft.workspaceSize(batched = false),
    p.spaceFft.workspaceSize(batched = false)).elems(T)
  let buf = ws.scratch(T, p.workspaceSize.elems(T) - at, at)
  var o = 0
  template take(n: int): ptr UncheckedArray[T] =
    let p = cast[ptr UncheckedArray[T]](addr buf[o])
    o += n
    p
  let trace = take(nTime)
  let oneRe = take(nFreq)
  let oneIm = take(nFreq)
  let specRe = take(p.traces * nFreq)
  let specIm = take(p.traces * nFreq)
  let spatA = take(nK)
  let spatB = take(nK)
  let ar = take(sBins)
  let ai = take(sBins)
  let br = take(sBins)
  let bi = take(sBins)
  let lo = max(rows.a, 0)
  let hi = min(rows.b, b.len - 1)
  var t = 0
  for r in lo .. hi:
    if not meta.isLive(r): continue
    let live = meta.liveSamples(r, b.cols)
    if live <= firstCol: continue
    if stop != nil and stop(): return Stopped
    let src = b.rowPtr(r)
    let last = min(live, firstCol + p.samples)
    let tw = p.traceW[t]
    zeroMem(trace, nTime * sizeof(T))
    {.push boundChecks: off.}
    for j in 0 ..< last - firstCol:
      trace[j] = src[firstCol + j] * p.timeW[j] * tw
    {.pop.}
    p.timeFft.forward(trace.toOpenArray(0, nTime - 1),
      oneRe.toOpenArray(0, nFreq - 1), oneIm.toOpenArray(0, nFreq - 1), ws)
    for k in 0 ..< nFreq:
      specRe[t * nFreq + k] = oneRe[k]
      specIm[t * nFreq + k] = oneIm[k]
    inc t
  let half = nK div 2
  for jf in 0 ..< nFreq:
    if stop != nil and stop(): return Stopped
    zeroMem(spatA, nK * sizeof(T))
    zeroMem(spatB, nK * sizeof(T))
    for t in 0 ..< p.traces:
      spatA[t] = specRe[t * nFreq + jf]
      spatB[t] = specIm[t * nFreq + jf]
    p.spaceFft.forward(spatA.toOpenArray(0, nK - 1), ar.toOpenArray(0, sBins - 1),
      ai.toOpenArray(0, sBins - 1), ws)
    p.spaceFft.forward(spatB.toOpenArray(0, nK - 1), br.toOpenArray(0, sBins - 1),
      bi.toOpenArray(0, sBins - 1), ws)
    # Full complex DFT of a + i·b, then flipped so later-on-later is +k.
    # Output index ik reads standard bin (nK/2 − ik), wrapped into 0 ..< nK.
    for ik in 0 ..< nK:
      let raw = half - ik
      let sk = if raw >= 0: raw else: raw + nK
      var re, im: float64
      if sk <= half:
        re = float64(ar[sk]) - float64(bi[sk])
        im = float64(ai[sk]) + float64(br[sk])
      else:
        let k = nK - sk
        re = float64(ar[k]) + float64(bi[k])
        im = float64(br[k]) - float64(ai[k])
      amp[ik * nFreq + jf] = hypot(re, im)
  entered

proc fkAmplitudeInto*[T: SampleType](p: FkPlan[T], samples: Tensor[T],
    meta: ChunkMeta, firstCol: int, amp: var openArray[float64],
    ws: var Workspace, rows = 0 .. int.high, stop: StopCheck = nil): int =
  ## `fkAmplitudeInto` on a `[rows, cols]` tensor. Returns the live rows
  ## used, or `Stopped` when `stop` abandons (`amp` then unspecified). The
  ## plan and the workspace stay usable.
  meta.checkShape(samples)
  p.fkAmplitudeInto(rowBlock(samples), meta, firstCol, amp, ws, rows, stop)
