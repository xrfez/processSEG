## Vibrator signals against sweep time (`SweepTrackPlan`): the phase, the
## amplitude and the frequency of the fundamental, the level of each
## harmonic, the total harmonic distortion and the peak of a signal (a
## ground force, a mass or baseplate acceleration), each measured against
## its reference (the pilot) at points along the sweep.
##
## ```nim
## let tp = initSweepTrackPlan(float64, TrackParams(), dtMs = 1, samples = 8001)
## var ws = initWorkspace()
## ws.reserve(tp.workspaceSize)
## var reference = initSweepReference(tp)
## var track = initSweepTrack(tp)
## tp.referenceInto(pilot, reference, ws)       # once per reference
## tp.trackInto(groundForce, reference, track, ws)
## # track.timeMs: the points; track.phaseDeg, track.amplitude[0] (the
## # fundamental), track.thdPct, track.peak, track.frequencyHz ...
## ```
##
## **The reference's phase.** `referenceInto` takes the reference's analytic
## signal (an FFT Hilbert transform, zero padded to twice its length): its
## phase ψ(t), unwrapped, and its instantaneous frequency `f = ψ'/2π` (a
## central difference over ±2 samples, exact for a linear sweep). Nothing is
## assumed of the sweep's law: linear, dB/octave, exponential or table
## sweeps, and a recorded pilot with its own errors, are handled alike. The
## reference is first demeaned and its content below its band is removed
## (a zero-phase cos² ramp from a quarter to half of the frequency below
## which 1 % of its energy lies, so a small drift does not move it): a DC
## offset, a drift, or
## the low-frequency content of a tapered start, would otherwise add a slowly
## varying offset to its Hilbert transform, so a ripple once per period to ψ,
## which reads as a false second harmonic in every signal tracked against it
## (half the offset: 0.5 % for a 1 % offset). The **sweep** is where the
## reference's envelope is within `minReferenceDb` of its peak; a reference
## with a NaN or no energy has no sweep.
##
## **Tracking.** At point `m` (sample `c = m·step`), harmonic `k` (1: the
## fundamental) has the complex amplitude
##
##   c_k = 2·∫ w·s·e^{−ikψ} dψ / ∫ w dψ,   w = cos⁴((ψ − ψ_c)/2C),
##
## over the `cycles` = C periods of the reference centred on the point (|ψ −
## ψ_c| < πC): a window in phase, so it follows the sweep (4 periods are
## 500 ms at 8 Hz, 50 ms at 80 Hz), the square of a Hann window. For a
## signal `Σ A_k·cos(kψ + θ_k)` (plus a DC offset) it is exactly
## `A_k·e^{iθ_k}`: `w = (3 + 4·cos(x/C) + cos(2x/C))/8` is a sum of cosines
## with fewer than C periods in the window, so over a whole number C ≥ 3 of
## periods it is orthogonal to DC and to every other harmonic, whatever the
## law. It is the tracking filter (synchronous detection) of a vibrator
## controller, as a moving Fourier series. The integral is a sum over the
## samples weighted by `f` (dψ = 2πf·dt); `w` vanishes to the fourth order at
## the window's ends, so the sampled sum keeps that orthogonality to about
## 1e-6 down to 6 samples per period (a Hann window, vanishing to the second
## order, leaked 5e-5 of the fundamental into the 2nd harmonic at 8 samples
## per period). The weight factors into cosines and sines of `ψ/C` and
## `2ψ/C`, so five running sums per harmonic give every window in one pass,
## the harmonics in SIMD lanes: the cost is O(samples × harmonics), whatever
## the window.
##
## **Measures** (`SweepTrack`, one value per point, NaN where not measured):
## - `phaseDeg`: `arg c_1`, the signal's fundamental minus the reference's, in
##   degrees in (−180, 180], positive when the signal leads; the reference
##   delayed by τ reads `−360·f·τ`. NaN where the fundamental is zero;
## - `amplitude[k − 1]`: `|c_k|`, the peak amplitude of harmonic k (not RMS)
##   in the signal's unit: `amplitude[0]` of a ground force is the
##   *fundamental force*;
## - `thdPct`: `100·√(Σ_{k≥2} A_k²)/A_1` over the harmonics measured there
##   (`thdFundamental`), or over `√(Σ_{k≥1} A_k²)` (`thdTotal`); NaN where no
##   harmonic is measured; `distortionInto` gives one harmonic's `100·A_k/A_1`;
## - `peak`: the largest `|s|` within one period of the reference centred on
##   the point, harmonics and noise included (a ground force's *peak force*),
##   of the band-limited signal between the samples: around the largest and
##   the smallest sample (the two lobes) a Kaiser-windowed sinc (8 taps each
##   side) gives the signal every 1/8 sample, and a parabola through the
##   largest of those and its neighbours its vertex. The largest sample alone falls short of a peak by
##   up to `1 − cos(π·f·dt)` (3 % at 80 Hz and 1 ms, 12 % at 2 ms, more with
##   harmonics); this way the error stays under 0.1 % for content up to 0.65
##   of Nyquist;
## - `referenceHz`: the reference's instantaneous frequency (the sweep law as
##   driven) and `frequencyHz`: the signal's fundamental's, `referenceHz +
##   (dθ/dt)/360` from the phase at the neighbouring points.
##
## **Where it is measured.** A point is measured when its window lies inside
## the sweep, so the measures start half a window after the reference does
## and stop half a window before its end. Harmonic `k` is measured where
## `k·f ≤ maxHarmonicFraction · Nyquist` all through the window. Inside the
## sweep's tapers the amplitudes follow the ramps and the phase is measured
## as long as the reference is above `minReferenceDb`.
##
## **Time axis.** Point `m` is at `timeMs[m] = m·step·dtMs` from the rows'
## first sample: the sweep time when the sweep starts at the record's start,
## as on wireline records.
##
## **Allocation.** `initSweepTrackPlan`, `initSweepReference` and
## `initSweepTrack` allocate; `referenceInto` and `trackInto` do not, once
## the workspace holds `workspaceSize`. They never raise: a NaN in the signal
## makes its measures NaN, and lengths that do not fit the plan are an
## `AssertionDefect`.

import std/math
import parseSEG
import ../core/workspace
import ../fft/plan
import kaiser

type
  ThdReference* = enum
    ## What the total harmonic distortion is relative to.
    thdFundamental = "fundamental" ## `√(Σ_{k≥2} A_k²) / A_1`
    thdTotal = "total"             ## `√(Σ_{k≥2} A_k²) / √(Σ_{k≥1} A_k²)`, under 100 %

  TrackParams* = object
    ## What `SweepTrackPlan` measures; see the module documentation.
    harmonics*: int = 5 ## The highest harmonic order tracked, 1 (the fundamental only) .. 16.
    cycles*: int = 4 ## The window: this many periods of the reference (≥ 3), cos⁴ weighted.
    stepMs*: float64 = 10 ## Points every `stepMs` (rounded to whole samples, at least one).
    minReferenceDb*: float64 = -40
      ## The sweep is where the reference's envelope is within this of its peak (< 0).
    maxHarmonicFraction*: float64 = 0.8
      ## Harmonic k is measured where `k·f` stays below this fraction of Nyquist (0 .. 1].
    thd*: ThdReference = thdFundamental ## What THD is relative to.

  SweepTrackPlan*[T: SampleType] = object
    ## Tracking of signals of up to `samples` samples at `dtMs`; see the
    ## module documentation.
    fft: FftPlan[T] # the reference's analytic signal
    peakTaps: seq[float64] # peakPhases × 2·peakHalf windowed-sinc taps, each phase summing to 1
    params: TrackParams
    dtMs: float64
    samples, step, points, lanes: int

  SweepReference* = object
    ## A reference prepared by `referenceInto`: its phase and frequency per
    ## sample and each point's windows. Make it with `initSweepReference`.
    samples, points: int # the plan's (a shape check)
    first, last: int # the sweep's samples (inclusive); first = −1: no sweep
    cosPsi, sinPsi, cosC, sinC, freqHz: seq[float64] # per sample
    lo, hi, lo1, hi1: seq[int] # per point: the C-period and one-period windows; lo = −1: not measured
    weight, refHz, maxHz: seq[float64] # per point: Σ w·f, f at the point, the largest f in the window

  SweepTrack* = object
    ## The measures of one signal against its reference, one value per point
    ## (NaN where not measured); see the module documentation. Make it with
    ## `initSweepTrack`; `trackInto` refills it.
    timeMs*: seq[float64] ## Each point's time from the rows' first sample, ms.
    referenceHz*: seq[float64] ## The reference's instantaneous frequency, Hz.
    frequencyHz*: seq[float64] ## The signal's fundamental's instantaneous frequency, Hz.
    phaseDeg*: seq[float64] ## The fundamental's phase minus the reference's, degrees, (−180, 180].
    amplitude*: seq[seq[float64]] ## `[order − 1][point]`: harmonic `order`'s peak amplitude.
    thdPct*: seq[float64] ## Total harmonic distortion, %.
    peak*: seq[float64] ## The largest `|s|` within one period around the point.

const
  maxHarmonics = 16
  peakPhases = 8 # sub-sample positions of the band-limited peak
  peakHalf = 8 # taps each side of the interpolator
  peakBeta = 8.0 # its Kaiser window
  sumCount = 10 # running sums per harmonic: 5 weights, real and imaginary

proc badTrack(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc validate*(p: TrackParams) =
  ## Checks the parameters (field documentation). Raises `ValueError`.
  if p.harmonics notin 1 .. maxHarmonics:
    badTrack("harmonics must be in 1 .. " & $maxHarmonics & ", got " & $p.harmonics)
  if p.cycles notin 3 .. 1000:
    badTrack("cycles must be a whole number of periods in 3 .. 1000 (over fewer, the " &
      "window is not orthogonal to the neighbouring harmonics), got " & $p.cycles)
  if not (p.stepMs > 0 and p.stepMs < Inf):
    badTrack("stepMs must be finite and positive, got " & $p.stepMs)
  if not (p.minReferenceDb < 0 and p.minReferenceDb > -Inf):
    badTrack("minReferenceDb must be finite and negative, got " & $p.minReferenceDb)
  if not (p.maxHarmonicFraction > 0 and p.maxHarmonicFraction <= 1):
    badTrack("maxHarmonicFraction must be in (0, 1], got " & $p.maxHarmonicFraction)

proc initSweepTrackPlan*(T: typedesc[SampleType], params: TrackParams, dtMs: float64,
    samples: int): SweepTrackPlan[T] =
  ## A plan tracking signals of up to `samples` samples at `dtMs` against a
  ## reference of the same rows. Raises `ValueError` for parameters `validate`
  ## rejects, a `dtMs` that is not finite and positive, or `samples < 2`.
  params.validate
  if not (dtMs > 0 and dtMs < Inf): badTrack("dtMs must be finite and positive, got " & $dtMs)
  if samples < 2: badTrack("a sweep track needs samples ≥ 2, got " & $samples)
  result.fft = initFftPlan(T, nextFastLen(2 * samples))
  result.params = params
  result.dtMs = dtMs
  result.samples = samples
  result.step = max(1, int(round(params.stepMs / dtMs)))
  result.points = (samples - 1) div result.step + 1
  result.lanes = if params.harmonics <= 8: 8 else: 16
  # The interpolator: phase q is q/8 of a sample after a sample, its tap k
  # on the sample k − 7 from that one.
  result.peakTaps = newSeq[float64](peakPhases * 2 * peakHalf)
  let i0b = besselI0(peakBeta)
  for q in 0 ..< peakPhases:
    var sum = 0.0
    for k in 0 ..< 2 * peakHalf:
      let d = float64(k - (peakHalf - 1)) - float64(q) / peakPhases # sample − position
      let u = d / float64(peakHalf)
      let h = (if abs(u) >= 1: 0.0 else: (if d == 0: 1.0 else: sin(PI * d) / (PI * d)) *
        besselI0(peakBeta * sqrt(1 - u * u)) / i0b)
      result.peakTaps[q * 2 * peakHalf + k] = h
      sum += h
    for k in 0 ..< 2 * peakHalf: result.peakTaps[q * 2 * peakHalf + k] /= sum

func params*[T](tp: SweepTrackPlan[T]): TrackParams {.inline, raises: [].} =
  ## The parameters.
  tp.params

func dtMs*[T](tp: SweepTrackPlan[T]): float64 {.inline, raises: [].} =
  ## The sample interval.
  tp.dtMs

func samples*[T](tp: SweepTrackPlan[T]): int {.inline, raises: [].} =
  ## The longest signal the plan takes.
  tp.samples

func points*[T](tp: SweepTrackPlan[T]): int {.inline, raises: [].} =
  ## The analysis points, `(samples − 1) div stepSamples + 1`.
  tp.points

func stepSamples*[T](tp: SweepTrackPlan[T]): int {.inline, raises: [].} =
  ## The samples between two points.
  tp.step

func pointMs*[T](tp: SweepTrackPlan[T], point: int): float64 {.inline, raises: [].} =
  ## Point `point`'s time from the rows' first sample, ms.
  float64(point * tp.step) * tp.dtMs

func elems(s: WorkspaceSize, T: typedesc): int {.inline.} =
  when T is float32: s.f32 else: s.f64

func spectrumAt[T](tp: SweepTrackPlan[T]): int {.inline.} =
  ## Where the reference's buffers start in the sample type's scratch: after
  ## the FFT's own (two traces of `nfft`, two spectra).
  tp.fft.workspaceSize(batched = false).elems(T)

func sumsAt[T](tp: SweepTrackPlan[T]): int {.inline.} =
  ## Where the float64 sums start: after the sample type's scratch when that
  ## is float64.
  when T is float64: tp.spectrumAt + 2 * tp.fft.len + 2 * tp.fft.bins
  else: 0

func workspaceSize*[T](tp: SweepTrackPlan[T]): WorkspaceSize {.raises: [].} =
  ## The scratch `referenceInto` and `trackInto` need.
  let own = tp.spectrumAt + 2 * tp.fft.len + 2 * tp.fft.bins
  # referenceInto: three running sums; trackInto: one snapshot per point.
  let sums = max(5 * (tp.samples + 1), tp.points * sumCount * tp.lanes)
  when T is float32: WorkspaceSize(f32: own, f64: sums)
  else: WorkspaceSize(f64: own + sums)

proc initSweepReference*[T](tp: SweepTrackPlan[T]): SweepReference =
  ## An empty reference sized for `tp` (no sweep until `referenceInto`).
  let n = tp.samples
  let p = tp.points
  result.samples = n
  result.points = p
  result.first = -1
  result.last = -1
  for s in [addr result.cosPsi, addr result.sinPsi, addr result.cosC, addr result.sinC,
      addr result.freqHz]:
    s[] = newSeq[float64](n)
  for s in [addr result.lo, addr result.hi, addr result.lo1, addr result.hi1]:
    s[] = newSeq[int](p)
    for v in s[].mitems: v = -1
  for s in [addr result.weight, addr result.refHz, addr result.maxHz]:
    s[] = newSeq[float64](p)

proc initSweepTrack*[T](tp: SweepTrackPlan[T]): SweepTrack =
  ## A track sized for `tp`, all NaN, with its time axis.
  let p = tp.points
  result.timeMs = newSeq[float64](p)
  for m in 0 ..< p: result.timeMs[m] = tp.pointMs(m)
  for s in [addr result.referenceHz, addr result.frequencyHz, addr result.phaseDeg,
      addr result.thdPct, addr result.peak]:
    s[] = newSeq[float64](p)
    for v in s[].mitems: v = NaN
  result.amplitude = newSeq[seq[float64]](tp.params.harmonics)
  for a in result.amplitude.mitems:
    a = newSeq[float64](p)
    for v in a.mitems: v = NaN

func span*(r: SweepReference): Slice[int] {.inline, raises: [].} =
  ## The samples of the sweep (where the reference's envelope is within
  ## `minReferenceDb` of its peak); empty (`0 .. -1`) when there is none.
  if r.first < 0: 0 .. -1 else: r.first .. r.last

func isMeasured*(r: SweepReference, point: int): bool {.inline, raises: [].} =
  ## Point `point`'s window lies inside the sweep.
  point >= 0 and point < r.lo.len and r.lo[point] >= 0

func len*(t: SweepTrack): int {.inline, raises: [].} =
  ## The number of points.
  t.timeMs.len

func harmonics*(t: SweepTrack): int {.inline, raises: [].} =
  ## The highest harmonic order tracked.
  t.amplitude.len

# ------------------------------------------------------------- reference --

func wrapDeg*(deg: float64): float64 {.raises: [].} =
  ## An angle in degrees wrapped to (−180, 180] (NaN stays NaN).
  let r = floorMod(deg, 360.0)
  if r > 180: r - 360 else: r

proc referenceInto*[T: SampleType](tp: SweepTrackPlan[T], x: openArray[T],
    dst: var SweepReference, ws: var Workspace) {.raises: [].} =
  ## Prepares the reference `x` (at most `samples` values: zeros after it):
  ## its phase and frequency per sample, the sweep's span and each point's
  ## windows. `dst` comes from `initSweepReference(tp)` and `ws` holds
  ## `tp.workspaceSize` (an `AssertionDefect` otherwise).
  doAssert tp.samples > 0, "a SweepTrackPlan must come from initSweepTrackPlan"
  doAssert x.len <= tp.samples, "a reference longer than the plan's samples"
  doAssert dst.samples == tp.samples and dst.points == tp.points,
    "the reference is not sized for this plan"
  doAssert ws.fits(tp.workspaceSize), "reserve the plan's workspaceSize first"
  let n = x.len
  for m in 0 ..< tp.points:
    dst.lo[m] = -1
    dst.hi[m] = -1
    dst.lo1[m] = -1
    dst.hi1[m] = -1
    dst.weight[m] = NaN
    dst.refHz[m] = NaN
    dst.maxHz[m] = NaN
  dst.first = -1
  dst.last = -1
  if n < 5: return
  # The analytic signal xh + i·h of the reference, demeaned and without the
  # content below its band (module documentation).
  let nf = tp.fft.len
  let bins = tp.fft.bins
  let buf = ws.scratch(T, 2 * nf + 2 * bins, tp.spectrumAt)
  let xh = buf
  let h = cast[ptr UncheckedArray[T]](addr buf[nf])
  let re = cast[ptr UncheckedArray[T]](addr buf[2 * nf])
  let im = cast[ptr UncheckedArray[T]](addr buf[2 * nf + bins])
  var mean = 0.0
  for v in x: mean += float64(v)
  mean /= float64(n)
  if not (mean - mean == 0): return # a NaN or an infinity: no sweep
  for j in 0 ..< n: xh[j] = T(float64(x[j]) - mean)
  zeroMem(addr xh[n], (nf - n) * sizeof(T))
  tp.fft.forward(xh.toOpenArray(0, nf - 1), re.toOpenArray(0, bins - 1),
    im.toOpenArray(0, bins - 1), ws)
  var total = 0.0
  for k in 1 ..< bins: total += float64(re[k]) ^ 2 + float64(im[k]) ^ 2
  if not (total > 0 and total < Inf): return
  var ka = 1 # the band's lower edge: 1 % of the energy below it
  var below = float64(re[1]) ^ 2 + float64(im[1]) ^ 2
  while ka + 1 < bins and below < 0.01 * total:
    inc ka
    below += float64(re[ka]) ^ 2 + float64(im[ka]) ^ 2
  let k0 = 0.25 * float64(ka)
  let k1 = 0.5 * float64(ka)
  re[0] = T(0)
  im[0] = T(0)
  for k in 1 ..< bins:
    let kf = float64(k)
    if kf < k1:
      let w = (if kf <= k0: 0.0 else: sin(PI / 2 * (kf - k0) / (k1 - k0)) ^ 2)
      re[k] = T(float64(re[k]) * w)
      im[k] = T(float64(im[k]) * w)
  tp.fft.inverse(re.toOpenArray(0, bins - 1), im.toOpenArray(0, bins - 1),
    xh.toOpenArray(0, nf - 1), ws)
  # H: the spectrum times −i·sign(f), Nyquist (real) to zero.
  for k in 1 ..< bins:
    let a = re[k]
    re[k] = im[k]
    im[k] = -a
  if nf mod 2 == 0:
    re[bins - 1] = T(0)
    im[bins - 1] = T(0)
  tp.fft.inverse(re.toOpenArray(0, bins - 1), im.toOpenArray(0, bins - 1),
    h.toOpenArray(0, nf - 1), ws)
  # The sweep: the envelope within minReferenceDb of its peak.
  var peak = 0.0
  for j in 0 ..< n:
    let e = hypot(float64(xh[j]), float64(h[j]))
    if not (e <= peak): peak = e # a NaN makes the peak NaN
  if not (peak > 0 and peak < Inf): return
  let level = peak * pow(10.0, tp.params.minReferenceDb / 20)
  var first = -1
  var last = -1
  for j in 0 ..< n:
    if hypot(float64(xh[j]), float64(h[j])) >= level:
      if first < 0: first = j
      last = j
  if last - first < 4: return
  dst.first = first
  dst.last = last
  # The unwrapped phase over the sweep, in the scratch (float64), and the
  # unit phasor e^{iψ} from the analytic signal itself.
  let psi = ws.scratch(float64, 5 * (tp.samples + 1), at = tp.sumsAt)
  for j in first .. last:
    let a = float64(xh[j])
    let b = float64(h[j])
    let e = hypot(a, b)
    if e > 0 or j == first:
      dst.cosPsi[j] = a / e
      dst.sinPsi[j] = b / e
    else: # a drop to zero inside the sweep: the phase holds
      dst.cosPsi[j] = dst.cosPsi[j - 1]
      dst.sinPsi[j] = dst.sinPsi[j - 1]
    if j == first: psi[j] = arctan2(b, a)
    else:
      # arg(z_j · conj(z_{j−1})): the step, never a 2π jump.
      let pr = dst.cosPsi[j - 1]
      let pi = dst.sinPsi[j - 1]
      psi[j] = psi[j - 1] + arctan2(dst.sinPsi[j] * pr - dst.cosPsi[j] * pi,
        dst.cosPsi[j] * pr + dst.sinPsi[j] * pi)
  # Frequency: a central difference over ±2 samples (one-sided at the ends).
  let dtS = tp.dtMs / 1000
  for j in first .. last:
    let a = max(first, j - 2)
    let b = min(last, j + 2)
    dst.freqHz[j] = (psi[b] - psi[a]) / (2 * PI * dtS * float64(b - a))
  let C = float64(tp.params.cycles)
  for j in first .. last:
    dst.cosC[j] = cos(psi[j] / C)
    dst.sinC[j] = sin(psi[j] / C)
  # Each point's windows (from ψ), then the weight's running sums, which
  # overwrite ψ.
  let halfC = PI * C
  var lo = first
  var hi = first
  var lo1 = first
  var hi1 = first
  for m in 0 ..< tp.points:
    let c = m * tp.step
    if c < first or c > last: continue
    let pc = psi[c]
    if not (pc - halfC >= psi[first]): continue # the window starts before the sweep
    while lo < c and psi[lo] <= pc - halfC: inc lo
    while hi <= last and psi[hi] < pc + halfC: inc hi
    if hi > last: break # the window ends after the sweep: so do the later points'
    while lo1 < c and psi[lo1] <= pc - PI: inc lo1
    hi1 = max(hi1, c)
    while hi1 <= last and psi[hi1] < pc + PI: inc hi1
    dst.lo[m] = lo
    dst.hi[m] = hi
    dst.lo1[m] = lo1
    dst.hi1[m] = hi1
    dst.refHz[m] = dst.freqHz[c]
    dst.maxHz[m] = max(dst.freqHz[lo], max(dst.freqHz[hi - 1], dst.freqHz[c]))
  # Σ f·(1, cos ψ/C, sin ψ/C, cos 2ψ/C, sin 2ψ/C) from `first`: in run q,
  # index i holds the sum over `first ..< first + i`.
  let stride = tp.samples + 1
  template run(q, i: int): untyped = psi[q * stride + i]
  for q in 0 ..< 5: run(q, 0) = 0
  for j in first .. last:
    let i = j - first
    let f = dst.freqHz[j]
    let (c1, s1) = (dst.cosC[j], dst.sinC[j])
    run(0, i + 1) = run(0, i) + f
    run(1, i + 1) = run(1, i) + f * c1
    run(2, i + 1) = run(2, i) + f * s1
    run(3, i + 1) = run(3, i) + f * (c1 * c1 - s1 * s1)
    run(4, i + 1) = run(4, i) + f * (2 * s1 * c1)
  for m in 0 ..< tp.points:
    let lo = dst.lo[m]
    if lo < 0: continue
    let c = m * tp.step
    let a = lo - first
    let b = dst.hi[m] - first
    let (c1, s1) = (dst.cosC[c], dst.sinC[c])
    template d(q: int): float64 = run(q, b) - run(q, a)
    dst.weight[m] = (3 * d(0) + 4 * (c1 * d(1) + s1 * d(2)) +
      ((c1 * c1 - s1 * s1) * d(3) + 2 * s1 * c1 * d(4))) / 8

# -------------------------------------------------------------- tracking --

type Lanes[L: static int] = object
  ## The running sums of one scan, for each harmonic's phasor P = e^{−ikψ}
  ## (lane k − 1): `s[2q]` and `s[2q + 1]` are the real and imaginary parts
  ## of Σ t·g_q·P, with `t = f·signal` and the weights g = (1, cos ψ/C,
  ## sin ψ/C, cos 2ψ/C, sin 2ψ/C).
  s: array[sumCount, array[L, float64]]

{.push boundChecks: off, overflowChecks: off.}

proc addSample[L: static int](acc: var Lanes[L], t, c1, s1, qr, qi: float64) {.noinline,
    raises: [].} =
  ## Adds one sample: `t = f·s`, `c1, s1 = cos, sin(ψ/C)` and the phasor
  ## `q = e^{−iψ}`, whose powers are the harmonics' phasors. A call of its
  ## own: inlined into the scan, where the sums' address is taken for the
  ## snapshots, GCC kept them in memory and the lane loops scalar.
  var pr, pi: array[L, float64]
  pr[0] = qr
  pi[0] = qi
  for l in 1 ..< L:
    pr[l] = pr[l - 1] * qr - pi[l - 1] * qi
    pi[l] = pr[l - 1] * qi + pi[l - 1] * qr
  let g = [t, t * c1, t * s1, t * (c1 * c1 - s1 * s1), t * (2 * s1 * c1)]
  for q in 0 ..< 5:
    let w = g[q]
    for l in 0 ..< L: # one vector operation per sum
      acc.s[2 * q][l] += w * pr[l]
      acc.s[2 * q + 1][l] += w * pi[l]

{.pop.}

proc finishPoint[T; L: static int](tp: SweepTrackPlan[T], r: SweepReference, m: int,
    now, before: Lanes[L], track: var SweepTrack) {.raises: [].} =
  ## Point `m`'s measures from the sums at its window's end and start.
  let c = m * tp.step
  let c1 = r.cosC[c]
  let s1 = r.sinC[c]
  # The window's weights at the centre: cos⁴ = (3 + 4·cos(x/C) + cos(2x/C))/8.
  let coef = [3.0, 4 * c1, 4 * s1, c1 * c1 - s1 * s1, 2 * s1 * c1]
  let w = r.weight[m]
  let nyq = 500 / tp.dtMs
  let limit = tp.params.maxHarmonicFraction * nyq
  var a1, harm = 0.0
  var measured = 0
  for k in 1 .. tp.params.harmonics:
    let l = k - 1
    var re, im = 0.0
    for q in 0 ..< 5:
      re += coef[q] * (now.s[2 * q][l] - before.s[2 * q][l])
      im += coef[q] * (now.s[2 * q + 1][l] - before.s[2 * q + 1][l])
    # c_k = 2·(Σ…/8)/w, the weight's own /8 included in w.
    let cr = re / (4 * w)
    let ci = im / (4 * w)
    if float64(k) * r.maxHz[m] > limit:
      track.amplitude[l][m] = NaN
      continue
    let a = hypot(cr, ci)
    track.amplitude[l][m] = a
    if k == 1:
      a1 = a
      track.phaseDeg[m] = (if a > 0: wrapDeg(radToDeg(arctan2(ci, cr))) else: NaN)
    else:
      harm += a * a
      inc measured
  if track.amplitude[0][m].isNaN:
    track.phaseDeg[m] = NaN
    track.thdPct[m] = NaN
  elif measured == 0 or not (a1 > 0):
    track.thdPct[m] = NaN
  else:
    track.thdPct[m] = 100 * sqrt(harm) /
      (if tp.params.thd == thdFundamental: a1 else: sqrt(a1 * a1 + harm))

proc scan[T; L: static int](tp: SweepTrackPlan[T], x: openArray[T], r: SweepReference,
    track: var SweepTrack, ws: var Workspace) {.raises: [].} =
  ## One pass over the sweep's samples: the sums are snapshotted at each
  ## measured point's window start, and the point finished at its end.
  let snaps = ws.scratch(float64, tp.points * sumCount * L, tp.sumsAt)
  # The measured points, in order; their windows' starts and ends both
  # increase with the point.
  var mLo = 0 # the next point whose window starts
  var mHi = 0 # the next point whose window ends
  template skip(m: var int) =
    while m < tp.points and r.lo[m] < 0: inc m
  skip(mLo)
  skip(mHi)
  if mLo >= tp.points: return
  let j0 = r.lo[mLo]
  var acc: Lanes[L]
  var j = j0
  let n = x.len
  while mHi < tp.points:
    while mLo < tp.points and r.lo[mLo] == j:
      copyMem(addr snaps[mLo * sumCount * L], addr acc, sizeof(acc))
      inc mLo
      skip(mLo)
    while mHi < tp.points and r.hi[mHi] == j:
      let before = cast[ptr Lanes[L]](addr snaps[mHi * sumCount * L])
      tp.finishPoint(r, mHi, acc, before[], track)
      inc mHi
      skip(mHi)
    if mHi >= tp.points: break
    let t = (if j < n: r.freqHz[j] * float64(x[j]) else: 0.0)
    acc.addSample(t, r.cosC[j], r.sinC[j], r.cosPsi[j], -r.sinPsi[j])
    inc j

proc trackInto*[T: SampleType](tp: SweepTrackPlan[T], x: openArray[T],
    reference: SweepReference, track: var SweepTrack, ws: var Workspace) {.raises: [].} =
  ## Tracks the signal `x` (at most `samples` values: zeros after it)
  ## against `reference` (prepared by `referenceInto` with this plan) into
  ## `track` (from `initSweepTrack(tp)`): every measure at every point, NaN
  ## where not measured. `ws` holds `tp.workspaceSize`; other shapes are an
  ## `AssertionDefect`.
  doAssert tp.samples > 0, "a SweepTrackPlan must come from initSweepTrackPlan"
  doAssert x.len <= tp.samples, "a signal longer than the plan's samples"
  doAssert reference.samples == tp.samples and reference.points == tp.points,
    "the reference is not sized for this plan"
  doAssert track.timeMs.len == tp.points and track.amplitude.len == tp.params.harmonics,
    "the track is not sized for this plan"
  doAssert ws.fits(tp.workspaceSize), "reserve the plan's workspaceSize first"
  for m in 0 ..< tp.points:
    track.timeMs[m] = tp.pointMs(m)
    track.referenceHz[m] = reference.refHz[m]
    track.frequencyHz[m] = NaN
    track.phaseDeg[m] = NaN
    track.thdPct[m] = NaN
    track.peak[m] = NaN
    for a in track.amplitude.mitems: a[m] = NaN
  if reference.first < 0: return
  if tp.lanes == 8: tp.scan[:T, 8](x, reference, track, ws)
  else: tp.scan[:T, 16](x, reference, track, ws)
  # Peaks over one period, and the fundamental's frequency from the phase.
  let n = x.len
  template at(j: int): float64 = (if j >= 0 and j < n: float64(x[j]) else: 0.0)
  for m in 0 ..< tp.points:
    if reference.lo[m] < 0 or track.amplitude[0][m].isNaN: continue
    var p = 0.0
    var hiJ, loJ = -1
    var hiV = -Inf
    var loV = Inf
    for j in reference.lo1[m] ..< min(reference.hi1[m], n):
      let v = float64(x[j])
      if v.isNaN: p = NaN
      if v > hiV: (hiV, hiJ) = (v, j)
      if v < loV: (loV, loJ) = (v, j)
    if not p.isNaN and hiJ >= 0: p = max(abs(hiV), abs(loV))
    for (best, sgn) in [(hiJ, 1.0), (loJ, -1.0)]:
      if best < 0 or p.isNaN: continue
      # The band-limited signal every 1/8 sample within one sample of the
      # lobe's largest sample, signed so that it is a maximum; then the
      # vertex of the parabola through its largest value and neighbours.
      var fine: array[2 * peakPhases + 1, float64]
      for i in 0 .. 2 * peakPhases:
        let pos = (best - 1) * peakPhases + i
        let base = floorDiv(pos, peakPhases)
        let q = pos - base * peakPhases
        var v = 0.0
        for k in 0 ..< 2 * peakHalf:
          v += tp.peakTaps[q * 2 * peakHalf + k] * at(base - (peakHalf - 1) + k)
        fine[i] = sgn * v
      var top = peakPhases
      for i in 0 .. 2 * peakPhases:
        if fine[i] > fine[top]: top = i
      var v = fine[top]
      if top > 0 and top < 2 * peakPhases:
        let a = fine[top - 1]
        let c = fine[top + 1]
        let curve = a - 2 * v + c
        if curve < 0: v -= (a - c) * (a - c) / (8 * curve)
      p = max(p, v)
    track.peak[m] = p
  for m in 0 ..< tp.points:
    let th = track.phaseDeg[m]
    if th.isNaN: continue
    let a = (if m > 0 and not track.phaseDeg[m - 1].isNaN: m - 1 else: m)
    let b = (if m + 1 < tp.points and not track.phaseDeg[m + 1].isNaN: m + 1 else: m)
    if a == b: continue
    let dTheta = wrapDeg(track.phaseDeg[b] - track.phaseDeg[a])
    let dt = (track.timeMs[b] - track.timeMs[a]) / 1000
    track.frequencyHz[m] = reference.refHz[m] + dTheta / (360 * dt)

# ------------------------------------------------------------- derived --

proc distortionInto*(track: SweepTrack, order: int, dst: var openArray[float64]) =
  ## Harmonic `order`'s distortion at each point, `100·A_order / A_1` %, into
  ## `dst` (`track.len` values; NaN where either is not measured or the
  ## fundamental is 0). Raises `ValueError` for an `order` outside
  ## `2 .. track.harmonics`; another `dst` length is an `AssertionDefect`.
  if order notin 2 .. track.harmonics:
    badTrack("order must be in 2 .. " & $track.harmonics & ", got " & $order)
  doAssert dst.len == track.len, "dst needs one value per point"
  for m in 0 ..< track.len:
    let a1 = track.amplitude[0][m]
    dst[m] = (if a1 > 0: 100 * track.amplitude[order - 1][m] / a1 else: NaN)

proc compareTracksInto*(a, b: SweepTrack, phaseDeg, ratioDb: var openArray[float64]) {.
    raises: [].} =
  ## Signal `a` against signal `b` at each point (two tracks of one plan,
  ## against the same or different references): `a`'s fundamental phase
  ## minus `b`'s (degrees, wrapped to (−180, 180]) and the ratio of their
  ## fundamentals, `20·log10(A_a / A_b)` dB. NaN where either is not
  ## measured. Tracks of other lengths, or outputs that are not one value
  ## per point, are an `AssertionDefect`.
  doAssert a.len == b.len and phaseDeg.len == a.len and ratioDb.len == a.len,
    "two tracks of one plan and one value per point"
  for m in 0 ..< a.len:
    phaseDeg[m] = wrapDeg(a.phaseDeg[m] - b.phaseDeg[m])
    let ra = a.amplitude[0][m]
    let rb = b.amplitude[0][m]
    ratioDb[m] = (if ra > 0 and rb > 0: 20 * log10(ra / rb) else: NaN)
