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
## `nextFastLen(2·traces)`, then transformed: a real FFT in time (`fftLanes`
## traces at a time) and the complex FFT across traces as two real FFTs. The
## modulus is unscaled (the caller applies `toDbInto`): a packed
## `sqrt(re² + im²)`, or `hypot` when a float64 sample is extreme.
## Dead, auxiliary and failed rows, and live rows
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

# `sqrt` of a sum of squares never needs errno, and without it GCC packs the
# square roots. This module's C file only: `modulusInto` is not generic, so
# the loop is compiled here (a generic kernel would compile in the caller).
{.localPassC: "-fno-math-errno".}

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

func fftScratch[T](p: FkPlan[T]): WorkspaceSize {.inline.} =
  ## The time FFT runs `fftLanes` traces at a time; the spatial FFT is one
  ## trace. Whichever scratch is larger.
  max(p.timeFft.workspaceSize, p.spaceFft.workspaceSize(batched = false))

func panelScratch[T](p: FkPlan[T]): int {.inline.} =
  ## Lane buffer, stored spectra and the spatial pass, in elements of `T`.
  fftLanes * (p.timeFft.len + 2 * p.timeFft.bins) + p.traces * p.timeFft.bins * 2 +
    2 * p.spaceFft.len + 4 * p.spaceFft.bins

func workspaceSize*[T](p: FkPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `fkAmplitudeInto` needs.
  result = fftScratch(p)
  when T is float32: result.f32 += panelScratch(p)
  else: result.f64 += panelScratch(p)
  result.f64 += 3 * p.spaceFft.len # unit-stride re, im and modulus

func leavesFloat64Range(x: ptr UncheckedArray[float64], n: int): bool {.inline,
    raises: [].} =
  ## A sample is NaN, infinite, a subnormal, below 1e-100 or above 1e140.
  ## The sign bit is cleared first, so the bits compare as signed integers.
  ## A value inside that range cannot overflow a square or flush one, even
  ## after both FFTs (their gain cannot close that gap).
  const
    notSign = 0x7FFF_FFFF_FFFF_FFFF'i64
    tinyBits = 0x2B2B_FF2E_E48E_0530'i64 # 1e-100
    hugeBits = 0x5D00_CB70_D24B_7379'i64 # 1e140
  var bad = 0'i64
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< n:
    let bits = cast[int64](x[i]) and notSign
    let huge = int64(bits > hugeBits)
    let tiny = int64(bits < tinyBits)
    let nz = int64(bits != 0)
    bad = bad or huge or (tiny and nz)
  {.pop.}
  bad != 0

proc modulusInto(re {.noalias.}, im {.noalias.}, mag {.noalias.}: ptr UncheckedArray[
    float64], n: int) {.noinline, raises: [].} =
  ## `mag[i] = sqrt(re[i]² + im[i]²)`, unit stride, no call and no branch.
  ## `-fno-math-errno` on this module packs the square roots. Extreme samples
  ## take `hypotModulus` instead: a call in this proc would align the stack
  ## on the fast path.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< n:
    mag[i] = sqrt(re[i] * re[i] + im[i] * im[i])
  {.pop.}

proc hypotModulus(re {.noalias.}, im {.noalias.}, mag {.noalias.}: ptr UncheckedArray[
    float64], n: int) {.noinline, raises: [].} =
  ## `mag[i] = hypot(re[i], im[i])` for a window whose samples leave range.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< n:
    mag[i] = hypot(re[i], im[i])
  {.pop.}

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
  let at = fftScratch(p).elems(T)
  let buf = ws.scratch(T, panelScratch(p), at)
  var o = 0
  template take(n: int): ptr UncheckedArray[T] =
    let p = cast[ptr UncheckedArray[T]](addr buf[o])
    o += n
    p
  let laneX = take(nTime * fftLanes)
  let laneRe = take(nFreq * fftLanes)
  let laneIm = take(nFreq * fftLanes)
  let specRe = take(p.traces * nFreq)
  let specIm = take(p.traces * nFreq)
  let spatA = take(nK)
  let spatB = take(nK)
  let ar = take(sBins)
  let ai = take(sBins)
  let br = take(sBins)
  let bi = take(sBins)
  let magAt = when T is float32: 0 else: at + panelScratch(p)
  let re64 = ws.scratch(float64, nK, magAt)
  let im64 = ws.scratch(float64, nK, magAt + nK)
  let mag = ws.scratch(float64, nK, magAt + 2 * nK)
  let lo = max(rows.a, 0)
  let hi = min(rows.b, b.len - 1)
  when T is float64:
    var extreme = false
  var t = 0
  var lane = 0
  template flushTime() =
    if lane > 0:
      p.timeFft.forwardLanes(laneX.toOpenArray(0, nTime * fftLanes - 1),
        laneRe.toOpenArray(0, nFreq * fftLanes - 1),
        laneIm.toOpenArray(0, nFreq * fftLanes - 1), ws)
      let base = t - lane
      for l in 0 ..< lane:
        let ti = base + l
        for k in 0 ..< nFreq:
          specRe[ti * nFreq + k] = laneRe[k * fftLanes + l]
          specIm[ti * nFreq + k] = laneIm[k * fftLanes + l]
      lane = 0
  for r in lo .. hi:
    if not meta.isLive(r): continue
    let live = meta.liveSamples(r, b.cols)
    if live <= firstCol: continue
    if stop != nil and stop(): return Stopped
    let src = b.rowPtr(r)
    let last = min(live, firstCol + p.samples)
    when T is float64:
      if not extreme and leavesFloat64Range(cast[ptr UncheckedArray[float64]](
          addr src[firstCol]), last - firstCol):
        extreme = true
    let tw = p.traceW[t]
    if lane == 0: zeroMem(laneX, nTime * fftLanes * sizeof(T))
    {.push boundChecks: off.}
    for j in 0 ..< last - firstCol:
      laneX[j * fftLanes + lane] = src[firstCol + j] * p.timeW[j] * tw
    {.pop.}
    inc lane
    inc t
    if lane == fftLanes: flushTime()
  flushTime()
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
    # The fold is indexed; the modulus is a unit-stride loop so it packs.
    for ik in 0 ..< nK:
      let raw = half - ik
      let sk = if raw >= 0: raw else: raw + nK
      if sk <= half:
        re64[ik] = float64(ar[sk]) - float64(bi[sk])
        im64[ik] = float64(ai[sk]) + float64(br[sk])
      else:
        let k = nK - sk
        re64[ik] = float64(ar[k]) + float64(bi[k])
        im64[ik] = float64(br[k]) - float64(ai[k])
    when T is float64:
      if extreme: hypotModulus(re64, im64, mag, nK)
      else: modulusInto(re64, im64, mag, nK)
    else:
      modulusInto(re64, im64, mag, nK)
    for ik in 0 ..< nK:
      amp[ik * nFreq + jf] = mag[ik]
  entered

proc fkAmplitudeInto*[T: SampleType](p: FkPlan[T], samples: Tensor[T],
    meta: ChunkMeta, firstCol: int, amp: var openArray[float64],
    ws: var Workspace, rows = 0 .. int.high, stop: StopCheck = nil): int =
  ## `fkAmplitudeInto` on a `[rows, cols]` tensor. Returns the live rows
  ## used, or `Stopped` when `stop` abandons (`amp` then unspecified). The
  ## plan and the workspace stay usable.
  meta.checkShape(samples)
  p.fkAmplitudeInto(rowBlock(samples), meta, firstCol, amp, ws, rows, stop)
