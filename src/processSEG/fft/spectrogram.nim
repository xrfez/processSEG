## A time-frequency picture of a trace (`SpectrogramPlan`): the short-time
## Fourier transform's amplitudes, frame by frame, for a time-variant
## spectral analysis (a vibrator's sweep and its harmonics as lines, a
## record's spectrum against time).
##
## ```nim
## let sg = initSpectrogramPlan(float64, windowLen = 257, hop = 10, dtMs = 1,
##   samples = 8001)                              # 257 ms frames every 10 ms
## var ws = initWorkspace()
## ws.reserve(sg.workspaceSize)
## var amp = newSeq[float64](sg.frames * sg.bins)
## sg.spectrogramInto(trace, amp, ws)             # amp[frame·bins + k]
## # frame i at column sg.frameCol(i), bin k at sg.binHz(k); toDbInto for dB
## ```
##
## **Frames.** Frame `i` is centred on column `i·hop` of the trace, for `i` in
## `0 ..< frames` (`frames = (samples − 1) div hop + 1`), and covers the
## `windowLen` samples from `i·hop − windowLen div 2`, weighted by a Hann
## window (`sin²(π·(j + 1)/(windowLen + 1))`) and zero padded to `nfft` (by
## default the next power of two of the window). Samples outside the trace
## count as zero. The frames go through the batched FFT, `lanesFor` at a time.
##
## **Scale.** As `SpectrumPlan`'s: `|X[k]|·2/Σw` (`1/Σw` at 0 Hz and at
## Nyquist), `Σw` the weights of the frame's samples that lie inside the
## trace, so a sinusoid of amplitude `A` on a bin reads `A` in every frame,
## also in the frames the trace's ends cut (to the leakage of its mirror
## image, which a cut window lets through more). A frame with no sample
## inside reads 0.
##
## **Resolution.** A frame of `W = windowLen·dtMs` ms has a main lobe `±2/W`
## kHz wide (Hann), so it separates lines about `2/W` apart (256 ms at 1 ms:
## 7.8 Hz), and smears `W` in time. A sweep's instantaneous frequency moves
## within a frame; its harmonics are the lines `k·f(t)`
## (`SweepTrack.referenceHz` gives `f` to overlay).
##
## **Allocation.** The constructor allocates; `spectrogramInto` does not,
## once the workspace holds `workspaceSize`, and never raises: other lengths
## are an `AssertionDefect`, a NaN makes the frames that hold it NaN.
##
## **Stop.** `averageSpectrogramInto`'s `stop` (nil: never) is checked once
## per live row, never per sample. `Stopped` means the job was abandoned;
## `amp` is then unspecified. The plan and the workspace stay usable.

import std/math
import parseSEG
import ../core/[rows, types, workspace]
import plan

{.localPassC: "-fno-math-errno".} # laneMagnitude's sqrt packs (this module's C file)

type
  SpectrogramPlan*[T: SampleType] = object
    ## The short-time Fourier amplitudes of traces of up to `samples`
    ## samples; see the module documentation.
    fft: FftPlan[T]
    window: seq[T]
    dtMs: float64
    samples, hop, frames: int

proc badPlan(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc initSpectrogramPlan*(T: typedesc[SampleType], windowLen, hop: int,
    dtMs: float64, samples: int, nfft = 0): SpectrogramPlan[T] =
  ## Frames of `windowLen` samples every `hop` samples of traces of up to
  ## `samples` samples at `dtMs`, zero padded to `nfft` (0: the next power of
  ## two of the window). Raises `ValueError` for `windowLen < 2`, `hop < 1`,
  ## `samples < 1`, a `dtMs` that is not finite and positive, or an `nfft`
  ## below the window or not 2·3·5-smooth.
  if windowLen < 2: badPlan("windowLen must be ≥ 2, got " & $windowLen)
  if hop < 1: badPlan("hop must be ≥ 1, got " & $hop)
  if samples < 1: badPlan("samples must be ≥ 1, got " & $samples)
  if not (dtMs > 0 and dtMs < Inf): badPlan(
      "dtMs must be finite and positive, got " & $dtMs)
  let n = if nfft == 0: nextPowerOfTwo(windowLen) else: nfft
  if n < windowLen: badPlan("nfft " & $n & " is shorter than the window (" &
      $windowLen & ")")
  result.fft = initFftPlan(T, n) # raises for a length that is not smooth
  result.dtMs = dtMs
  result.samples = samples
  result.hop = hop
  result.frames = (samples - 1) div hop + 1
  result.window = newSeq[T](windowLen)
  for j in 0 ..< windowLen:
    result.window[j] = T(sin(PI * float64(j + 1) / float64(windowLen + 1)) ^ 2)

func frames*[T](sg: SpectrogramPlan[T]): int {.inline, raises: [].} =
  ## The number of frames.
  sg.frames

func bins*[T](sg: SpectrogramPlan[T]): int {.inline, raises: [].} =
  ## The bins of a frame, `nfft div 2 + 1`.
  sg.fft.bins

func windowLen*[T](sg: SpectrogramPlan[T]): int {.inline, raises: [].} =
  ## The frame length in samples.
  sg.window.len

func hop*[T](sg: SpectrogramPlan[T]): int {.inline, raises: [].} =
  ## The samples between two frames' centres.
  sg.hop

func nfft*[T](sg: SpectrogramPlan[T]): int {.inline, raises: [].} =
  ## The FFT length.
  sg.fft.len

func samples*[T](sg: SpectrogramPlan[T]): int {.inline, raises: [].} =
  ## The longest trace the plan takes.
  sg.samples

func dtMs*[T](sg: SpectrogramPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval.
  sg.dtMs

func frameCol*[T](sg: SpectrogramPlan[T], frame: int): int {.inline, raises: [].} =
  ## The trace column frame `frame` is centred on, `frame·hop`.
  frame * sg.hop

func binHz*[T](sg: SpectrogramPlan[T], k: int): float64 {.inline, raises: [].} =
  ## The frequency of bin `k`, Hz.
  float64(k) * 1000 / (float64(sg.fft.len) * sg.dtMs)

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

proc laneMagnitude(re, im: ptr UncheckedArray[float32],
    mag {.noalias.}: ptr UncheckedArray[float64], n: int) {.noinline, raises: [].} =
  ## `mag[i] = sqrt(re[i]² + im[i]²)` in float64, so a float32 spectrum cannot
  ## overflow or underflow the product. Unit stride, no call, no branch.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< n:
    let r = float64(re[i])
    let m = float64(im[i])
    mag[i] = sqrt(r * r + m * m)
  {.pop.}

func leavesFloat64Range(x: ptr UncheckedArray[float64],
    n: int): bool {.noinline, raises: [].} =
  ## A sample is NaN, infinite, a subnormal, below 1e-100 or above 1e140.
  ## The sign bit is cleared first, so the bits compare as signed integers.
  ## A value inside that range cannot overflow a square or flush one, even
  ## after a frame's window and FFT (the gain cannot close that gap).
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

proc laneMagnitude(re, im: ptr UncheckedArray[float64],
    mag {.noalias.}: ptr UncheckedArray[float64], n: int) {.noinline, raises: [].} =
  ## The float64 form of `laneMagnitude`. No call and no branch, so the
  ## square roots stay packed. A trace whose samples leave range takes
  ## `hypotMagnitude`.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< n:
    mag[i] = sqrt(re[i] * re[i] + im[i] * im[i])
  {.pop.}

proc hypotMagnitude(re, im: ptr UncheckedArray[float64],
    mag {.noalias.}: ptr UncheckedArray[float64], n: int) {.noinline, raises: [].} =
  ## `mag[i] = hypot(re[i], im[i])` for a trace whose samples leave range.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< n:
    mag[i] = hypot(re[i], im[i])
  {.pop.}

func amplitudesFinite*(x: ptr UncheckedArray[float64], n: int): bool {.noinline,
    raises: [].} =
  ## No NaN or infinity in `x[0 ..< n]`: none has every exponent bit set.
  ## An OR of flags, the same rows `classify` dropped per value. Exported
  ## for the pool form of `averageSpectrogramInto`, which calls this on a
  ## row's amplitudes.
  const expBits = 0x7FF0_0000_0000_0000'u64
  var bad = 0'u64
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0 ..< n:
    bad = bad or typeof(bad)((cast[typeof(bad)](x[j]) and expBits) == expBits)
  {.pop.}
  bad == 0

proc spectrogramWidth[T; L: static int](sg: SpectrogramPlan[T], x: openArray[T],
    amp: var openArray[float64], ws: var Workspace, extreme: bool) {.raises: [].} =
  ## `spectrogramInto` at a static width below `fftLanes`. Scratch stays
  ## sized for `fftLanes`; a batch uses the first `n · L` time samples.
  let n = sg.fft.len
  let nb = sg.fft.bins
  let W = sg.window.len
  let at = sg.fft.workspaceSize.elems(T)
  let buf = ws.scratch(T, fftLanes * (n + 2 * nb), at)
  let re = cast[ptr UncheckedArray[T]](addr buf[n * fftLanes])
  let im = cast[ptr UncheckedArray[T]](addr buf[n * fftLanes + nb * fftLanes])
  var first = 0
  while first < sg.frames:
    let used = min(L, sg.frames - first)
    var sw: array[L, float64]
    zeroMem(buf, n * L * sizeof(T))
    for l in 0 ..< used:
      let start = (first + l) * sg.hop - W div 2
      let a = max(0, -start)
      let b = min(W, x.len - start)
      for j in a ..< b:
        buf[j * L + l] = x[start + j] * sg.window[j]
        sw[l] += float64(sg.window[j])
    sg.fft.forwardLanes(L, buf.toOpenArray(0, n * L - 1),
      re.toOpenArray(0, nb * L - 1), im.toOpenArray(0, nb * L - 1), ws)
    let count = nb * L
    let held = min(count, (n * L * sizeof(T)) div sizeof(float64))
    let mag = cast[ptr UncheckedArray[float64]](addr buf[0])
    doAssert held >= 1 and count - held <= fftLanes
    doAssert (nb - 1) * L <= held
    var tail: array[fftLanes, float64]
    when T is float64:
      if extreme: hypotMagnitude(re, im, mag, held)
      else: laneMagnitude(re, im, mag, held)
    else:
      laneMagnitude(re, im, mag, held)
    if held < count:
      when T is float64:
        if extreme:
          hypotMagnitude(cast[ptr UncheckedArray[float64]](addr re[held]),
            cast[ptr UncheckedArray[float64]](addr im[held]),
            cast[ptr UncheckedArray[float64]](addr tail[0]), count - held)
        else:
          laneMagnitude(cast[ptr UncheckedArray[float64]](addr re[held]),
            cast[ptr UncheckedArray[float64]](addr im[held]),
            cast[ptr UncheckedArray[float64]](addr tail[0]), count - held)
      else:
        laneMagnitude(cast[ptr UncheckedArray[T]](addr re[held]),
          cast[ptr UncheckedArray[T]](addr im[held]),
          cast[ptr UncheckedArray[float64]](addr tail[0]), count - held)
    for l in 0 ..< used:
      let row = (first + l) * nb
      if not (sw[l] > 0):
        for k in 0 ..< nb: amp[row + k] = 0
      else:
        let inv = 1.0 / sw[l]
        let two = 2.0 * inv
        amp[row] = mag[l] * inv
        for k in 1 ..< nb - 1:
          amp[row + k] = mag[k * L + l] * two
        let last = (nb - 1) * L + l
        let lastSide = if 2 * (nb - 1) == n: inv else: two
        if last < held: amp[row + nb - 1] = mag[last] * lastSide
        else: amp[row + nb - 1] = tail[last - held] * lastSide
    first += used

func workspaceSize*[T](sg: SpectrogramPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `spectrogramInto` needs.
  let extra = fftLanes * (sg.fft.len + 2 * sg.fft.bins)
  result = sg.fft.workspaceSize
  when T is float32: result.f32 += extra
  else: result.f64 += extra

proc spectrogramInto*[T: SampleType](sg: SpectrogramPlan[T], x: openArray[T],
    amp: var openArray[float64], ws: var Workspace) {.raises: [].} =
  ## The amplitudes of every frame of `x` (at most `samples` values, zeros
  ## after it) into `amp` (`frames·bins` values, frame after frame: bin `k`
  ## of frame `i` is `amp[i·bins + k]`). `ws` holds `sg.workspaceSize`; other
  ## lengths are an `AssertionDefect`.
  doAssert sg.samples > 0, "a SpectrogramPlan must come from initSpectrogramPlan"
  doAssert x.len <= sg.samples, "a trace longer than the plan's samples"
  doAssert amp.len == sg.frames * sg.fft.bins, "amp needs frames·bins values"
  doAssert ws.fits(sg.workspaceSize), "reserve the plan's workspaceSize first"
  let n = sg.fft.len
  let nb = sg.fft.bins
  let W = sg.window.len
  let extreme = when T is float64:
    x.len > 0 and leavesFloat64Range(cast[ptr UncheckedArray[float64]](addr x[
        0]), x.len)
  else:
    false
  let at = sg.fft.workspaceSize.elems(T)
  let width = lanesFor(sg.fft)
  if width != fftLanes:
    case width
    of 4: spectrogramWidth[T, 4](sg, x, amp, ws, extreme)
    of 2: spectrogramWidth[T, 2](sg, x, amp, ws, extreme)
    else: doAssert false, "lanesFor returned a width with no kernel"
    return
  let buf = ws.scratch(T, fftLanes * (n + 2 * nb), at)
  let re = cast[ptr UncheckedArray[T]](addr buf[n * fftLanes])
  let im = cast[ptr UncheckedArray[T]](addr buf[n * fftLanes + nb * fftLanes])
  var first = 0
  while first < sg.frames:
    let used = min(fftLanes, sg.frames - first)
    var sw: array[fftLanes, float64]
    zeroMem(buf, n * fftLanes * sizeof(T))
    for l in 0 ..< used:
      let start = (first + l) * sg.hop - W div 2
      let a = max(0, -start)
      let b = min(W, x.len - start)
      for j in a ..< b:
        buf[j * fftLanes + l] = x[start + j] * sg.window[j]
        sw[l] += float64(sg.window[j])
    sg.fft.forwardLanes(buf.toOpenArray(0, n * fftLanes - 1),
      re.toOpenArray(0, nb * fftLanes - 1), im.toOpenArray(0, nb * fftLanes -
          1), ws)
    # Time samples are dead. Their bytes hold unit-stride magnitudes; a
    # float32 buffer is a few lanes short of one bin, and those sit in `tail`.
    let count = nb * fftLanes
    let held = min(count, (n * fftLanes * sizeof(T)) div sizeof(float64))
    let mag = cast[ptr UncheckedArray[float64]](addr buf[0])
    doAssert held >= fftLanes and count - held <= fftLanes
    doAssert (nb - 1) * fftLanes - 1 < held # bins before the last are in `mag`
    var tail: array[fftLanes, float64]
    when T is float64:
      if extreme: hypotMagnitude(re, im, mag, held)
      else: laneMagnitude(re, im, mag, held)
    else:
      laneMagnitude(re, im, mag, held)
    if held < count:
      when T is float64:
        if extreme:
          hypotMagnitude(cast[ptr UncheckedArray[float64]](addr re[held]),
            cast[ptr UncheckedArray[float64]](addr im[held]),
            cast[ptr UncheckedArray[float64]](addr tail[0]), count - held)
        else:
          laneMagnitude(cast[ptr UncheckedArray[float64]](addr re[held]),
            cast[ptr UncheckedArray[float64]](addr im[held]),
            cast[ptr UncheckedArray[float64]](addr tail[0]), count - held)
      else:
        laneMagnitude(cast[ptr UncheckedArray[T]](addr re[held]),
          cast[ptr UncheckedArray[T]](addr im[held]),
          cast[ptr UncheckedArray[float64]](addr tail[0]), count - held)
    for l in 0 ..< used:
      let row = (first + l) * nb
      if not (sw[l] > 0):
        for k in 0 ..< nb: amp[row + k] = 0
      else:
        let inv = 1.0 / sw[l]
        let two = 2.0 * inv
        amp[row] = mag[l] * inv
        for k in 1 ..< nb - 1:
          amp[row + k] = mag[k * fftLanes + l] * two
        let last = (nb - 1) * fftLanes + l
        let lastSide = if 2 * (nb - 1) == n: inv else: two
        if last < held: amp[row + nb - 1] = mag[last] * lastSide
        else: amp[row + nb - 1] = tail[last - held] * lastSide
    first += used

func averagedRow*[T: SampleType](x: ptr UncheckedArray[T],
    n: int): bool {.inline, raises: [].} =
  ## True when these input samples enter the power average. Exported for
  ## the pool form of `averageSpectrogramInto`, which calls this on the
  ## row's live input samples (`src`, `live`) before the transform. A
  ## float64 row that `leavesFloat64Range` flags is left out (not summed,
  ## not counted, no FFT): a NaN, an infinity, a subnormal, or a nonzero
  ## sample below 1e-100 or above 1e140. A float32 row is kept here; a
  ## non-finite spectrogram is still dropped afterwards.
  when T is float64:
    n <= 0 or not leavesFloat64Range(cast[ptr UncheckedArray[float64]](x), n)
  else:
    true

func averageWorkspace*[T](sg: SpectrogramPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `averageSpectrogramInto` needs: the plan's, plus one frame
  ## of amplitudes.
  result = sg.workspaceSize
  result.f64 += sg.frames * sg.fft.bins

proc averageSpectrogramInto*[T: SampleType](sg: SpectrogramPlan[T], b: RowBlock[
    T], meta: ChunkMeta, amp: var openArray[float64], ws: var Workspace,
    rows = 0 .. int.high, stop: StopCheck = nil): int =
  ## The power average of `spectrogramInto` over the live rows of `b` in
  ## `rows` (clipped to the block), into `amp` (`frames·bins` values, the
  ## same layout). A sinusoid of amplitude A on a bin reads A, as one trace
  ## does. A float64 row whose live input samples leave the safe range (a
  ## NaN, an infinity, a subnormal, or a nonzero sample below 1e-100 or
  ## above 1e140) is left out before the transform, as is a row whose
  ## spectrogram holds a NaN or an infinity, a row that is not live, or a
  ## row with no live sample. Returns the rows used;
  ## with none, `amp` is all zero. `stop` (nil: never) is asked once per live
  ## row that has a sample, never per sample; true abandons the job and
  ## returns `Stopped`, and `amp` is then unspecified. The plan is unchanged
  ## and the workspace stays usable. Raises `ValueError`
  ## when `b` does not match `meta` or the interval is not the plan's. An
  ## `amp` of another length, or a workspace that does not hold
  ## `averageWorkspace`, is an `AssertionDefect` / `IndexDefect`. Nothing
  ## allocates.
  ##
  ## The proposal took `openArray[openArray[T]]`. Nim does not allow a nested
  ## `openArray`, and a block of traces is a `RowBlock` with a `ChunkMeta`
  ## everywhere else, so dead rows and the padding past a short trace stay
  ## out.
  meta.checkShape(b)
  if not sameInterval(meta.dtMs, sg.dtMs):
    badPlan("the rows' interval " & $meta.dtMs & " ms is not the plan's " &
      $(sg.dtMs) & " ms")
  let nb = sg.frames * sg.fft.bins
  doAssert amp.len == nb, "amp needs frames·bins values"
  doAssert ws.fits(sg.averageWorkspace), "reserve averageWorkspace first"
  let at = sg.workspaceSize.elems(T)
  # The plan's scratch is T; the frame amplitudes are float64, from element 0
  # of that region when T is float32, and after the plan's own float64 when
  # T is float64 (workspaceSize puts the FFT scratch in the sample type).
  let tmpAt = when T is float32: 0 else: sg.workspaceSize.f64
  let tmp = ws.scratch(float64, nb, tmpAt)
  for k in 0 ..< nb: amp[k] = 0
  let lo = max(rows.a, 0)
  let hi = min(rows.b, b.len - 1)
  var used = 0
  for r in lo .. hi:
    if not meta.isLive(r): continue
    let live = meta.liveSamples(r, b.cols)
    if live <= 0: continue
    if live > sg.samples:
      badPlan("a live row of " & $live & " samples is longer than the plan's " &
        $sg.samples)
    if stop != nil and stop(): return Stopped
    let src = b.rowPtr(r)
    if averagedRow(src, live):
      sg.spectrogramInto(src.toOpenArray(0, live - 1), tmp.toOpenArray(0, nb -
          1), ws)
      if amplitudesFinite(cast[ptr UncheckedArray[float64]](addr tmp[0]), nb):
        for k in 0 ..< nb: amp[k] += tmp[k] * tmp[k]
        inc used
  if used == 0: return 0
  let u = float64(used)
  for k in 0 ..< nb: amp[k] = sqrt(amp[k] / u)
  used

proc averageSpectrogramInto*[T: SampleType](sg: SpectrogramPlan[T],
    samples: Tensor[T], meta: ChunkMeta, amp: var openArray[float64],
    ws: var Workspace, rows = 0 .. int.high, stop: StopCheck = nil): int =
  ## `averageSpectrogramInto` on a `[rows, cols]` tensor. Returns the rows
  ## used, or `Stopped` when `stop` abandons (`amp` then unspecified). The
  ## plan and the workspace stay usable.
  meta.checkShape(samples)
  sg.averageSpectrogramInto(rowBlock(samples), meta, amp, ws, rows, stop)
