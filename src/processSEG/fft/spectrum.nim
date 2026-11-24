## Amplitude spectra for display (`SpectrumPlan`): per-trace amplitude and
## phase, and an averaged amplitude spectrum over a set of traces and a time
## window, in dB, for a viewer's spectrum panel.
##
## ```nim
## let sp = initSpectrumPlan(float32, dtMs = 2.0, windowLen = 501)  # taper 10 % each end
## var ws = initWorkspace()
## ws.reserve(sp.workspaceSize)
## var amp, db = newSeq[float64](sp.bins)
## let used = sp.averageAmplitudeInto(rowBlock(samples), meta, firstCol = 250, amp, ws)
## toDbInto(amp, db)                             # 0 dB at the peak
## for k in 0 ..< sp.bins: plot(sp.binHz(k), db[k])
## ```
##
## **Window.** A plan covers `windowLen` samples, which are demeaned (unless
## `demean = false`), multiplied by a Tukey taper (a sin² ramp over
## `taperFraction` of the window at each end: 0 is a boxcar, 0.5 a Hann
## window) and zero-padded to the FFT length `nfft`. By default `nfft` is the
## next power of two (`nextPowerOfTwo(windowLen)`, as most viewers do: a 60 s
## record at 1 ms, 60001 samples, gives 65536); any 2·3·5-smooth length ≥ the
## window may be passed instead. Bin `k` is at `k / (nfft·dt)` Hz (`binHz`).
##
## **Short windows.** A trace that ends inside the window (its live length)
## gives a shorter window, of `c` samples: it gets the same taper ramps, the
## end one at its own end (`w[j] = min(taper[j], taper[c−1−j])`), and its own
## `Σw` below, so it is scaled like a full window and does not end in a step.
##
## **Amplitude scale.** Amplitudes are one-sided and corrected for the taper:
## `|X[k]|·2/Σw` (`1/Σw` at 0 Hz and at Nyquist), with `Σw` the sum of the
## taper weights. A sinusoid of amplitude `A` whose frequency falls on a bin
## then reads `A` there, whatever the taper or padding (between bins, the
## taper's scalloping lowers the peak, by up to 1.4 dB for a Hann window).
## The phase is measured from the first sample of the window.
##
## **dB scale** (`toDbInto`): `20·log10(amplitude / ref)`, where `ref` is the
## largest amplitude of the spectrum (`dbPeak`, the default: the peak is 0 dB
## and everything else negative) or `refAmplitude` sample units (`dbAbsolute`,
## `refAmplitude = 1` by default: a sinusoid of amplitude 1 on a bin is 0 dB).
## Values below `floorDb` are clamped to it, so a zero amplitude is never −∞.
##
## **Averaging** (`averageAmplitudeInto`) is over power: the result is
## `sqrt(mean over traces of amplitude²)`, the RMS amplitude, so the dB of
## the average is `10·log10` of the mean power (louder traces weigh more than
## in a mean of amplitudes). Only live rows count (dead, auxiliary and failed
## rows are left out, as are rows with no live sample in the window); a live
## row of zeros counts, and lowers the average. A row with a NaN or infinite
## sample in its window is left out too, so it cannot blank the panel; the
## count of rows averaged is returned. The rows go through the batched FFT,
## `lanesFor` traces at a time.
##
## **NaN.** `amplitudeInto` on a window holding a NaN gives NaN in every bin
## of that trace (its mean and its FFT are NaN); `toDbInto` maps a NaN
## amplitude to `floorDb`, and ignores NaN and +Inf when finding the peak (an
## infinite amplitude is +Inf dB).

import std/math
import parseSEG
import ../core/[rows, sums, types, workspace]
import plan

type
  SpectrumPlan*[T: SampleType] = object
    ## A windowed amplitude spectrum of one window length; see the module
    ## documentation.
    fft: FftPlan[T]
    taper: seq[T] # windowLen weights
    taperSum: float64 # Σ taper: Σw of a full window
    dtMs: float64
    demean: bool

  DbReference* = enum
    ## The 0 dB level of `toDbInto`.
    dbPeak     ## the largest amplitude of the spectrum
    dbAbsolute ## an amplitude of 1 sample unit

proc badSpectrum(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc initSpectrumPlan*(T: typedesc[SampleType], dtMs: float64, windowLen: int,
    taperFraction = 0.1, nfft = 0, demean = true): SpectrumPlan[T] =
  ## A spectrum of windows of `windowLen` samples at `dtMs` ms, tapered over
  ## `taperFraction` (0 .. 0.5) of the window at each end and zero-padded to
  ## `nfft` (0: `nextPowerOfTwo(windowLen)`). Raises `ValueError` for
  ## `windowLen < 1`, a `dtMs` that is not finite and positive, a taper
  ## fraction outside 0 .. 0.5, or an `nfft` shorter than the window or not
  ## 2·3·5-smooth.
  if windowLen < 1:
    badSpectrum("a spectrum window needs ≥ 1 sample, got " & $windowLen)
  if not (dtMs > 0 and dtMs < Inf):
    badSpectrum("dtMs must be finite and > 0, got " & $dtMs)
  if not (taperFraction >= 0 and taperFraction <= 0.5):
    badSpectrum("taperFraction must be in 0 .. 0.5, got " & $taperFraction)
  let n = if nfft == 0: nextPowerOfTwo(windowLen) else: nfft
  if n < windowLen:
    badSpectrum("nfft " & $n & " is shorter than the window (" & $windowLen & ")")
  result.fft = initFftPlan(T, n) # raises for a length that is not smooth
  result.dtMs = dtMs
  result.demean = demean
  result.taper = newSeq[T](windowLen)
  let ramp = int(round(taperFraction * float64(windowLen)))
  for i in 0..<windowLen:
    let d = min(i, windowLen - 1 - i) # distance from the nearer end
    let w =
      if d >= ramp: 1.0
      else: sin(PI / 2 * float64(d + 1) / float64(ramp + 1)) ^ 2
    result.taper[i] = T(w)
    result.taperSum += float64(result.taper[i])

func sides*[T](sp: SpectrumPlan[T], k: int): float64 {.inline.} =
  ## The one-sided factor of bin `k`: 1 at 0 Hz and at Nyquist, else 2.
  if k == 0 or 2 * k == sp.fft.len: 1.0 else: 2.0

func bins*[T](sp: SpectrumPlan[T]): int {.inline, raises: [].} =
  ## The number of frequency bins, `nfft div 2 + 1`.
  sp.fft.bins

func windowLen*[T](sp: SpectrumPlan[T]): int {.inline, raises: [].} =
  ## The window length in samples.
  sp.taper.len

func dtMs*[T](sp: SpectrumPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval the plan was built for.
  sp.dtMs

func nfft*[T](sp: SpectrumPlan[T]): int {.inline, raises: [].} =
  ## The FFT length (the window plus zero padding).
  sp.fft.len

func binHz*[T](sp: SpectrumPlan[T], k: int): float64 {.inline, raises: [].} =
  ## The frequency of bin `k`, in Hz: `k / (nfft·dt)`.
  float64(k) * 1000 / (float64(sp.fft.len) * sp.dtMs)

func elems(s: WorkspaceSize, T: typedesc): int =
  when T is float32: s.f32 else: s.f64

func workspaceSize*[T](sp: SpectrumPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch of `averageAmplitudeInto`, which also covers `amplitudeInto`.
  let extra = sp.fft.len * fftLanes + 2 * sp.fft.bins * fftLanes
  result = sp.fft.workspaceSize
  when T is float32: result.f32 += extra
  else: result.f64 += extra
  result.f64 += sp.fft.bins

# ------------------------------------------------------------------ kernels --

proc amplitudePhaseInto*[T: SampleType](re, im: openArray[T],
    amp, phase: var openArray[T]) {.raises: [].} =
  ## The modulus and the phase (radians, `arctan2(im, re)`, in −π .. π) of
  ## each bin of a spectrum from `forward`, unscaled. All four have the same
  ## length (an `AssertionDefect` otherwise).
  let n = re.len
  doAssert im.len == n and amp.len == n and phase.len == n,
    "re, im, amp and phase need the same length"
  for k in 0..<n:
    amp[k] = T(hypot(float64(re[k]), float64(im[k])))
    phase[k] = T(arctan2(float64(im[k]), float64(re[k])))

proc toDbInto*(amp: openArray[float64], db: var openArray[float64],
    reference = dbPeak, floorDb = -120.0, refAmplitude = 1.0) {.raises: [].} =
  ## `20·log10(amp / ref)` per bin (module documentation), clamped below at
  ## `floorDb`. With `dbAbsolute`, 0 dB is an amplitude of `refAmplitude`
  ## (1 by default, so a call that does not pass it is unchanged). With
  ## `dbPeak` the peak is still 0 dB and `refAmplitude` is not used. A
  ## `refAmplitude` that is not finite and positive clamps every bin to
  ## `floorDb`, as a zero reference would. With `dbPeak` and an all-zero
  ## spectrum, every bin is `floorDb`. `amp` and `db` have the same length
  ## (an `AssertionDefect` otherwise).
  doAssert db.len == amp.len, "amp and db need the same length"
  var r = refAmplitude
  if reference == dbPeak:
    r = 0.0
    for a in amp:
      if a > r and a < Inf: r = a # NaN and Inf are not a peak
  for k in 0..<amp.len:
    let a = amp[k]
    let v = if r > 0 and a > 0: 20 * log10(a / r) else: -Inf # NaN: the floor
    db[k] = max(v, floorDb)

proc wrapStep(d: float64): float64 {.inline.} =
  ## `d` brought into (−180, 180] by steps of 360.
  result = d
  while result > 180: result -= 360
  while result <= -180: result += 360

proc unwrapPhaseInto*(phaseDeg: openArray[float64], shown: openArray[bool],
    dst: var openArray[float64]) {.raises: [].} =
  ## Unwraps `phaseDeg` along the bins `shown` marks, into `dst` (all three
  ## the same length; an `AssertionDefect` otherwise). A bin that is not
  ## shown, or whose phase is not finite, is a gap: `dst` is NaN there, and
  ## the gap is not crossed — the next shown finite bin is unwrapped from the
  ## last drawn value, not from the bins in between. A step of more than 180°
  ## from that value is brought back by ±360° until it is in (−180, 180]. The
  ## first drawn bin is copied as it is. Nothing allocates.
  let n = phaseDeg.len
  doAssert shown.len == n and dst.len == n, "phase, shown and dst need the same length"
  var last = NaN
  for i in 0 ..< n:
    if not shown[i] or phaseDeg[i].classify in {fcNan, fcInf, fcNegInf}:
      dst[i] = NaN
    elif last.isNaN:
      dst[i] = phaseDeg[i]
      last = dst[i]
    else:
      dst[i] = last + wrapStep(phaseDeg[i] - last)
      last = dst[i]

func circularMeanDeg*(angles: openArray[float64]): tuple[meanDeg,
    resultant: float64] {.
    raises: [].} =
  ## The circular mean of `angles` (degrees) and the mean resultant length,
  ## from the sum of their unit vectors: `meanDeg` is its argument in
  ## (−180, 180], and `resultant` is its length over the count, 0 (opposite
  ## angles) to 1 (all equal). A plain mean of wrapped angles is wrong near
  ## ±180°. NaN angles are left out; an angle that is infinite counts as NaN.
  ## With none left (an empty or all-NaN input) the result is `(NaN, 0)`.
  ## When the unit vectors cancel (both sums under `1e-12`, as for opposite
  ## angles) the direction is undefined: `meanDeg` is NaN and `resultant` is
  ## the (near zero) length. Nothing allocates.
  var sx, sy = 0.0
  var n = 0
  for v in angles:
    if v.classify in {fcNormal, fcSubnormal, fcZero, fcNegZero}:
      sx += cos(degToRad(v))
      sy += sin(degToRad(v))
      inc n
  if n == 0: return (NaN, 0.0)
  result.resultant = min(hypot(sx, sy) / float64(n), 1.0)
  if abs(sx) < 1e-12 and abs(sy) < 1e-12:
    result.meanDeg = NaN
  else:
    result.meanDeg = radToDeg(arctan2(sy, sx))
    if result.meanDeg <= -180: result.meanDeg += 360

proc removeDelayInto*(unwrapped, hz: openArray[float64], shown: openArray[bool],
    dst: var openArray[float64]): float64 {.raises: [].} =
  ## Removes the least-squares line `p = a + b·f` from the drawn finite bins
  ## of an unwrapped phase (`unwrapped` degrees, `hz` the frequency of each
  ## bin), into `dst`: `dst = p − (a + b·f)` there and NaN elsewhere. Returns
  ## the delay in ms, `−b·1000/360` (a delay of τ ms is a slope of
  ## `−360·τ/1000` degrees per Hz). NaN, and `dst` all NaN, when fewer than
  ## two bins are drawn and finite, or the normal-equation denominator is
  ## below `1e-12·max(1, Σf²)` (a fit the frequencies do not determine). All
  ## four have the same length (an `AssertionDefect` otherwise). Nothing
  ## allocates.
  let n = unwrapped.len
  doAssert hz.len == n and shown.len == n and dst.len == n,
    "phase, hz, shown and dst need the same length"
  var count = 0
  var sumF, sumF2, sumP, sumPF = 0.0
  for i in 0 ..< n:
    if shown[i] and unwrapped[i].classify notin {fcNan, fcInf, fcNegInf} and
        hz[i].classify notin {fcNan, fcInf, fcNegInf}:
      let f = hz[i]
      let p = unwrapped[i]
      sumF += f
      sumF2 += f * f
      sumP += p
      sumPF += p * f
      inc count
  let den = float64(count) * sumF2 - sumF * sumF
  if count < 2 or abs(den) < 1e-12 * max(1.0, sumF2):
    for i in 0 ..< n: dst[i] = NaN
    return NaN
  let a = (sumP * sumF2 - sumF * sumPF) / den
  let b = (float64(count) * sumPF - sumF * sumP) / den
  for i in 0 ..< n:
    if shown[i] and unwrapped[i].classify notin {fcNan, fcInf, fcNegInf} and
        hz[i].classify notin {fcNan, fcInf, fcNegInf}:
      dst[i] = unwrapped[i] - (a + b * hz[i])
    else:
      dst[i] = NaN
  -b * 1000 / 360

# -------------------------------------------------------------- one trace --

proc windowInto[T](sp: SpectrumPlan[T], x: openArray[T],
    dst: ptr UncheckedArray[T], stride: int): float64 {.raises: [].} =
  ## Demeans and tapers `x` (≤ windowLen samples, the rest zero; module
  ## documentation, "Short windows") into `dst[j·stride]`, j < nfft. Returns
  ## Σw, or NaN when `x` holds a NaN or infinite sample.
  let n = min(x.len, sp.taper.len)
  let sum = sum64(x.toOpenArray(0, n - 1))
  let finite = sum - sum == 0 # false for a NaN or ±Inf sum
  let m = if sp.demean and n > 0 and finite: T(sum / float64(n)) else: T(0)
  var sw = 0.0
  {.push boundChecks: off.}
  if n == sp.taper.len:
    for j in 0..<n: dst[j * stride] = (x[j] - m) * sp.taper[j]
    sw = sp.taperSum
  else:
    for j in 0..<n:
      let w = min(sp.taper[j], sp.taper[n - 1 - j])
      dst[j * stride] = (x[j] - m) * w
      sw += float64(w)
  for j in n..<sp.fft.len: dst[j * stride] = T(0)
  {.pop.}
  if finite: sw else: NaN

proc amplitudeInto*[T: SampleType](sp: SpectrumPlan[T], x: openArray[T],
    amp, phase: var openArray[T], ws: var Workspace) {.raises: [].} =
  ## The scaled amplitude (module documentation) and the phase (radians) of
  ## one window of a trace: `x` holds up to `windowLen` samples (fewer are
  ## zero-padded), `amp` and `phase` have `bins` values (an `AssertionDefect`
  ## otherwise). `ws` must hold `sp.workspaceSize`.
  let nb = sp.fft.bins
  doAssert amp.len == nb and phase.len == nb, "amp and phase need bins values"
  let at = sp.fft.workspaceSize.elems(T)
  let xw = ws.scratch(T, sp.fft.len + 2 * nb, at)
  let sw = sp.windowInto(x, xw, 1)
  let re = cast[ptr UncheckedArray[T]](addr xw[sp.fft.len])
  let im = cast[ptr UncheckedArray[T]](addr xw[sp.fft.len + nb])
  sp.fft.forward(xw.toOpenArray(0, sp.fft.len - 1), re.toOpenArray(0, nb - 1),
    im.toOpenArray(0, nb - 1), ws)
  for k in 0..<nb:
    amp[k] = T(hypot(float64(re[k]), float64(im[k])) * sp.sides(k) / sw)
    phase[k] = T(arctan2(float64(im[k]), float64(re[k])))

# ---------------------------------------------------------- averaged panel --

proc addLanePowers[T](re, im: ptr UncheckedArray[T], nLanes, stride, nb: int,
    scale, acc: ptr UncheckedArray[float64]) {.inline, raises: [].} =
  ## `acc[k] += Σ_l (re² + im²) · scale[l]` for lanes `0 ..< nLanes`.
  ## Bin `k` of lane `l` is element `k·stride + l`.
  {.push boundChecks: off.}
  for k in 0..<nb:
    var s = 0.0
    for l in 0..<nLanes:
      let r = float64(re[k * stride + l])
      let i = float64(im[k * stride + l])
      s += (r * r + i * i) * scale[l]
    acc[k] += s
  {.pop.}

proc addPowerWidth[T; L: static int](sp: SpectrumPlan[T], b: RowBlock[T],
    meta: ChunkMeta, firstCol, row0, row1: int, acc: var openArray[float64],
    ws: var Workspace): int {.raises: [].} =
  ## `addPowerRows` at a static width below `fftLanes`. Scratch stays sized
  ## for `fftLanes`; the live batch uses the first `n · L` samples.
  let nb = sp.fft.bins
  let n = sp.fft.len
  let at = sp.fft.workspaceSize.elems(T)
  let xw = ws.scratch(T, n * fftLanes + 2 * nb * fftLanes, at)
  let re = cast[ptr UncheckedArray[T]](addr xw[n * fftLanes])
  let im = cast[ptr UncheckedArray[T]](addr xw[n * fftLanes + nb * fftLanes])
  var lane = 0
  var used = 0
  var scale: array[L, float64]
  template flush() =
    if lane > 0:
      for l in lane ..< L:
        for j in 0 ..< n: xw[j * L + l] = T(0)
      sp.fft.forwardLanes(L, xw.toOpenArray(0, n * L - 1),
        re.toOpenArray(0, nb * L - 1), im.toOpenArray(0, nb * L - 1), ws)
      addLanePowers(re, im, lane, L, nb,
        cast[ptr UncheckedArray[float64]](addr scale[0]),
        cast[ptr UncheckedArray[float64]](addr acc[0]))
      lane = 0
  for r in row0 .. row1:
    if meta.isLive(r):
      let live = meta.liveSamples(r, b.cols)
      if live > firstCol:
        let last = min(live, firstCol + sp.taper.len)
        let src = b.rowPtr(r)
        let sw = sp.windowInto(src.toOpenArray(firstCol, last - 1),
          cast[ptr UncheckedArray[T]](addr xw[lane]), L)
        if sw > 0:
          scale[lane] = 1 / (sw * sw)
          inc lane
          inc used
          if lane == L: flush()
  flush()
  used

proc addPowerRows*[T: SampleType](sp: SpectrumPlan[T], b: RowBlock[T],
    meta: ChunkMeta, firstCol, row0, row1: int, acc: var openArray[float64],
    ws: var Workspace): int {.raises: [].} =
  ## Adds the scaled power of rows `row0 .. row1` into `acc` (`bins` values),
  ## in row order: each kept row contributes `(re² + im²) / Σw²`, the same
  ## quantity `averageAmplitudeInto` sums. Returns how many were added. A row
  ## that is not live, that has no live sample past `firstCol`, or whose
  ## window holds a NaN or an infinite sample is left out (`acc` unchanged
  ## for that row). `ws` must hold `sp.workspaceSize`. Exported for the pool
  ## form, which sums fixed batches with this and averages them itself.
  let nb = sp.fft.bins
  let n = sp.fft.len
  doAssert acc.len == nb, "acc needs bins values"
  if row1 < row0 or nb == 0: return 0
  let width = lanesFor(sp.fft)
  if width != fftLanes:
    case width
    of 4: return addPowerWidth[T, 4](sp, b, meta, firstCol, row0, row1, acc, ws)
    of 2: return addPowerWidth[T, 2](sp, b, meta, firstCol, row0, row1, acc, ws)
    else: doAssert false, "lanesFor returned a width with no kernel"
  let at = sp.fft.workspaceSize.elems(T)
  let xw = ws.scratch(T, n * fftLanes + 2 * nb * fftLanes, at)
  let re = cast[ptr UncheckedArray[T]](addr xw[n * fftLanes])
  let im = cast[ptr UncheckedArray[T]](addr xw[n * fftLanes + nb * fftLanes])
  var lane = 0
  var used = 0
  var scale: array[fftLanes, float64] # 1/Σw² of each lane's window

  template flush() =
    if lane > 0:
      for l in lane..<fftLanes: # unused lanes are zero
        for j in 0..<n: xw[j * fftLanes + l] = T(0)
      sp.fft.forwardLanes(xw.toOpenArray(0, n * fftLanes - 1), re.toOpenArray(0,
          nb * fftLanes - 1),
        im.toOpenArray(0, nb * fftLanes - 1), ws)
      addLanePowers(re, im, lane, fftLanes, nb,
        cast[ptr UncheckedArray[float64]](addr scale[0]),
        cast[ptr UncheckedArray[float64]](addr acc[0]))
      lane = 0

  for r in row0..row1:
    if meta.isLive(r):
      let live = meta.liveSamples(r, b.cols)
      if live > firstCol:
        let last = min(live, firstCol + sp.taper.len)
        let src = b.rowPtr(r)
        let sw = sp.windowInto(src.toOpenArray(firstCol, last - 1),
          cast[ptr UncheckedArray[T]](addr xw[lane]), fftLanes)
        if sw > 0: # a NaN or Inf sample: the lane is reused
          scale[lane] = 1 / (sw * sw)
          inc lane
          inc used
          if lane == fftLanes: flush()
  flush()
  used

proc averageAmplitudeInto*[T: SampleType](sp: SpectrumPlan[T], b: RowBlock[T],
    meta: ChunkMeta, firstCol: int, amp: var openArray[float64],
    ws: var Workspace, rows = 0 .. int.high): int =
  ## The power-averaged amplitude spectrum (module documentation) of the
  ## window `firstCol ..< firstCol + windowLen` of the live rows of `b` in
  ## `rows` (clipped to the block), into `amp` (`bins` values). Returns the
  ## number of rows averaged; with none, `amp` is all zero. `ws` must hold
  ## `sp.workspaceSize`. Raises `ValueError` when `b` does not match `meta`,
  ## its rows' interval is not the plan's, or for a negative `firstCol`; an `amp.len != bins` is an
  ## `AssertionDefect`, as in the other spectrum procs. No sample of `b`
  ## changes.
  meta.checkShape(b)
  if not sameInterval(meta.dtMs, sp.dtMs):
    badSpectrum("the rows' interval " & $meta.dtMs & " ms is not the plan's " &
      $(sp.dtMs) & " ms")
  let nb = sp.fft.bins
  let n = sp.fft.len
  if firstCol < 0:
    badSpectrum("firstCol must be ≥ 0, got " & $firstCol)
  doAssert amp.len == nb, "amp needs bins values"
  let at = sp.fft.workspaceSize.elems(T)
  let acc = ws.scratch(float64, nb, at = (when T is float32: 0 else: at +
    n * fftLanes + 2 * nb * fftLanes))
  for k in 0..<nb: acc[k] = 0
  let lo = max(rows.a, 0)
  let hi = min(rows.b, b.len - 1)
  let used = sp.addPowerRows(b, meta, firstCol, lo, hi,
    acc.toOpenArray(0, nb - 1), ws)
  for k in 0..<nb:
    amp[k] = if used > 0: sqrt(acc[k] / float64(used)) * sp.sides(k) else: 0.0
  used

proc averageAmplitudeInto*[T: SampleType](sp: SpectrumPlan[T],
    samples: Tensor[T], meta: ChunkMeta, firstCol: int,
    amp: var openArray[float64], ws: var Workspace, rows = 0 .. int.high): int =
  ## `averageAmplitudeInto` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  sp.averageAmplitudeInto(rowBlock(samples), meta, firstCol, amp, ws, rows)

# ---------------------------------------------------- several windows summed --

type
  SpectrumAccumulator*[T: SampleType] = object
    ## A power sum of `averageAmplitudeInto` over several calls, so a spectrum
    ## panel can average windows the caller feeds one block at a time. Build
    ## it with `initSpectrumAccumulator`.
    plan: SpectrumPlan[T]
    power: seq[float64]
    tmp: seq[float64]
    used: int

proc initSpectrumAccumulator*(T: typedesc[SampleType], dtMs: float64,
    windowLen: int,

nfft = 0, taperFraction = 0.1, demean = true): SpectrumAccumulator[T] =
  ## An accumulator of windows of `windowLen` samples at `dtMs`, with the
  ## same taper, padding and demean as `initSpectrumPlan`. Raises `ValueError`
  ## as that does. Allocates the power sum; `add` does not.
  result.plan = initSpectrumPlan(T, dtMs, windowLen, taperFraction, nfft, demean)
  result.power = newSeq[float64](result.plan.bins)
  result.tmp = newSeq[float64](result.plan.bins)

func bins*[T](acc: SpectrumAccumulator[T]): int {.inline, raises: [].} =
  ## The number of frequency bins.
  acc.plan.bins

func windowLen*[T](acc: SpectrumAccumulator[T]): int {.inline, raises: [].} =
  ## The window length in samples.
  acc.plan.windowLen

func nfft*[T](acc: SpectrumAccumulator[T]): int {.inline, raises: [].} =
  ## The FFT length.
  acc.plan.nfft

func dtMs*[T](acc: SpectrumAccumulator[T]): float64 {.inline, raises: [].} =
  ## The sample interval.
  acc.plan.dtMs

func binHz*[T](acc: SpectrumAccumulator[T], k: int): float64 {.inline, raises: [].} =
  ## The frequency of bin `k`, in Hz.
  acc.plan.binHz(k)

func used*[T](acc: SpectrumAccumulator[T]): int {.inline, raises: [].} =
  ## The rows added so far.
  acc.used

func workspaceSize*[T](acc: SpectrumAccumulator[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `add` needs (the plan's).
  acc.plan.workspaceSize

proc clear*[T](acc: var SpectrumAccumulator[T]) {.raises: [].} =
  ## Zeros the power sum. The plan stays.
  for k in 0 ..< acc.power.len: acc.power[k] = 0
  acc.used = 0

proc add*[T: SampleType](acc: var SpectrumAccumulator[T], b: RowBlock[T],
    meta: ChunkMeta, firstCol: int, ws: var Workspace, rows = 0 ..
        int.high): int =
  ## Adds the window `averageAmplitudeInto` would average, as `amp²·rows`, to
  ## the power sum. Returns the rows added (0 when the window has none). The
  ## same `ValueError`s as `averageAmplitudeInto`. `ws` must hold
  ## `acc.workspaceSize`. Nothing allocates.
  let n = acc.plan.averageAmplitudeInto(b, meta, firstCol, acc.tmp, ws, rows)
  if n > 0:
    for k in 0 ..< acc.power.len:
      acc.power[k] += acc.tmp[k] * acc.tmp[k] * float64(n)
    acc.used += n
  n

proc add*[T: SampleType](acc: var SpectrumAccumulator[T], samples: Tensor[T],
    meta: ChunkMeta, firstCol: int, ws: var Workspace, rows = 0 ..
        int.high): int =
  ## `add` on a `[rows, cols]` tensor.
  meta.checkShape(samples)
  acc.add(rowBlock(samples), meta, firstCol, ws, rows)

proc amplitudesInto*[T](acc: SpectrumAccumulator[T], dst: var openArray[float64]) {.
    raises: [].} =
  ## `sqrt(power / used)` into `dst` (`bins` values; an `AssertionDefect`
  ## otherwise): the power-averaged amplitude, on the same scale as
  ## `averageAmplitudeInto` (a sinusoid of amplitude A on a bin reads A).
  ## All zero when nothing has been added.
  doAssert dst.len == acc.power.len, "dst needs bins values"
  if acc.used == 0:
    for k in 0 ..< dst.len: dst[k] = 0
  else:
    let u = float64(acc.used)
    for k in 0 ..< dst.len: dst[k] = sqrt(acc.power[k] / u)
