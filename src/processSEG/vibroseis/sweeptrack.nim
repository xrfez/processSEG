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
## sweeps, and a recorded pilot with its own phase errors (its instrument
## filters: the phase is then the signal's against the pilot as recorded),
## are handled alike; not a reference with harmonics of its own (below). The
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
## **The flat part** (`flatSpan`, `extent`): the sweep between its tapers,
## from the reference's envelope averaged over one period around each
## sample (which leaves out the ripple at 2f of an imperfect analytic signal,
## and the slow 1/t tail the Hilbert transform puts after an abrupt start).
## The start taper ends at the first sample, above half the peak, after
## which the average rises over 10 ms by no more than 2e-4 of itself or 2 %
## of its fastest rise before (so a plateau with a slope is flat); the end
## taper starts at the last such sample, mirrored. Measured on cos², linear
## and Blackman tapers of 0.3 and 1 s (8-80 Hz, 0.5-2 ms): within 0.75 of
## a period of the reference of each taper's end, 1.5 with 1 % noise; an
## untapered sweep is flat from 0.8 of a period after its start. Each taper
## is looked for in its half of the sweep: where the envelope never settles
## there (an amplitude changing by more than 2 % of the tapers' steepest
## rise all along), that end of the sweep. `vibqc` judges limits and
## summaries over this part by default.
##
## **Tracking.** At point `m` (sample `c = m·step`), harmonic `k` (1: the
## fundamental) has the complex amplitude
##
##   c_k = 2·∫ w·s·e^{−ikψ} dψ / ∫ w dψ,   w = cos⁴((ψ − ψ_c)/2C),
##
## over C periods of the reference centred on the point (|ψ − ψ_c| < πC):
## a window in phase, so it follows the sweep, the square of a Hann window.
## C is `cycles` (4: 500 ms at 8 Hz), but where those are shorter than
## `minWindowMs` (100 ms), the smallest **even** number of periods that
## reaches it, from the reference's frequency at the point (`cyclesAt`):
## 4 up to 40 Hz, then 6 to 60 Hz, 8 to 80 Hz. The minimum time steadies the
## measures where 4 periods are a few tens of ms (on white noise the phase's
## rms falls by √(C/4), measured 1.24 at C = 6 and 1.42 at C = 8); an even
## count, because an odd one lets a subharmonic at f/2 into the fundamental
## (below).
##
## For a signal `Σ A_k·cos(kψ + θ_k)` (plus a DC offset) the window gives exactly
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
## `2ψ/C`, so five running sums per harmonic give every window of one C in
## one pass, the harmonics in SIMD lanes: one pass per window length the
## sweep passes through (three on an 8-80 Hz sweep at 100 ms, each over its
## points' samples), so the cost is O(samples × harmonics), whatever the
## window.
##
## **What the window lets through.** Only whole harmonics (and DC) are
## rejected exactly. Content between them inside the window's main lobe
## (`±3f/C`) leaks in: a subharmonic at `f/2` or `3f/2` (a decoupling
## plate) reads 1/6 of its level in the fundamental and the 2nd harmonic
## over 4 periods (the weight's `cos(2x/C)` term is at `f/2`), 0.04 over 5
## and 7, nothing over an even number from 6 (hence the even counts a
## minimum time raises the window to); so do noise (THD counts the noise in
## the harmonics' bands) and harmonics folded from above Nyquist (recorded
## data are anti-alias filtered). The reference must be a clean sweep: a
## harmonic of its own at level `a` bends ψ once per period, and every
## signal tracked against it reads false harmonics of about `a/2` (a 5 %
## 2nd harmonic in the reference: 2.9 % in the 2nd harmonic of a clean
## signal; a 5 % 3rd: 1.2° of phase and 1.2 % of fundamental), so a ground
## force is no reference (`compareTracksInto` compares two signals).
##
## **Measures** (`SweepTrack`, one value per point, NaN where not measured):
## - `phaseDeg`: `arg c_1`, the signal's fundamental minus the reference's, in
##   degrees in (−180, 180], positive when the signal leads; the reference
##   delayed by τ reads `−360·f·τ`. NaN where the fundamental is zero;
## - `amplitude[k − 1]`: `|c_k|`, the peak amplitude of harmonic k (not RMS)
##   in the signal's unit: `amplitude[0]` of a ground force is the
##   *fundamental force*;
## - `thdPct`: `100·√(Σ_{k≥2} A_k²)/A_1` (`thdFundamental`), or over
##   `√(Σ_{k≥1} A_k²)` (`thdTotal`), over the harmonics 2 .. `harmonics`: every one of
##   them (`thFixed`, the default: NaN where one is past the harmonic limit,
##   so THD has one meaning all along the sweep), or those measured there
##   (`thAllThatFit`: it steps down where one leaves the band); `thdCount`
##   gives how many it sums at each point (0 where it is NaN). THD's noise
##   floor grows as the square root of that count (1 % white noise: 0.29 %
##   over 2-5, 0.58 % over 2-16, at 0.5 ms). At most 16 harmonics are
##   tracked (the lanes of the sums); `distortionInto` gives one harmonic's
##   `100·A_k/A_1`;
## - `peak`: the largest `|s|` within one period of the reference centred on
##   the point, harmonics and noise included (a ground force's *peak force*),
##   of the band-limited signal between the samples: around the largest and
##   the smallest sample (the two lobes) a Kaiser-windowed sinc (8 taps each
##   side) gives the signal every 1/8 sample, and a parabola through the
##   largest of those and its neighbours its vertex. The largest sample alone falls short of a peak by
##   up to `1 − cos(π·f·dt)` (3 % at 80 Hz and 1 ms, 12 % at 2 ms, more with
##   harmonics); this way the error stays under 0.1 % for content up to 0.65
##   of Nyquist (0.4 % up to 0.8). Samples clipped after the anti-alias
##   filter are not band-limited: their peak overshoots the clip level (3 %
##   at 80 Hz and 1 ms);
## - `referenceHz`: the reference's instantaneous frequency (the sweep law as
##   driven) and `frequencyHz`: the signal's fundamental's, `referenceHz +
##   (dθ/dt)/360` from the phase at the neighbouring points.
##
## **Where it is measured.** A point is measured when its window lies inside
## the sweep, so the measures start half a window after the reference does
## and stop half a window before its end (4 periods of the end's frequency
## at 100 ms and 80 Hz: 50 ms). Harmonic `k` is measured where
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
## the workspace holds `workspaceSize`. They never raise: a NaN or infinite
## sample in the signal makes NaN the points whose window holds it, and only
## those; one in the reference leaves it without a sweep; lengths that do
## not fit the plan are an `AssertionDefect`.

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

  ThdHarmonics* = enum
    ## Which harmonics the total harmonic distortion sums at a point.
    thFixed = "fixed"
      ## Every one of 2 .. `harmonics`: NaN where one of them is past the
      ## harmonic limit, so THD has one meaning along the whole sweep.
    thAllThatFit = "all that fit"
      ## Those of 2 .. `harmonics` measured at the point (below the harmonic
      ## limit): THD steps down where one leaves the band.

  TrackParams* = object
    ## What `SweepTrackPlan` measures; see the module documentation.
    harmonics*: int = 5
      ## The highest harmonic order tracked, 1 (the fundamental only) .. 16;
      ## THD sums the orders 2 .. `harmonics` (`thdHarmonics`).
    cycles*: int = 4
      ## The window: at least this many periods of the reference (≥ 3), cos⁴
      ## weighted.
    minWindowMs*: float64 = 100
      ## ... and at least this long: where `cycles` periods are shorter, the
      ## window is the smallest even number of periods that reaches it (0:
      ## `cycles` everywhere; 0 .. 1e6).
    stepMs*: float64 = 10 ## Points every `stepMs` (rounded to whole samples, at least one).
    minReferenceDb*: float64 = -40
      ## The sweep is where the reference's envelope is within this of its peak (< 0).
    maxHarmonicFraction*: float64 = 0.8
      ## Harmonic k is measured where `k·f` stays below this fraction of Nyquist (0 .. 1].
    thd*: ThdReference = thdFundamental ## What THD is relative to.
    thdHarmonics*: ThdHarmonics = thFixed ## Which harmonics THD sums.

  SweepTrackPlan*[T: SampleType] = object
    ## Tracking of signals of up to `samples` samples at `dtMs`; see the
    ## module documentation.
    fft: FftPlan[T] # the reference's analytic signal
    peakTaps: seq[float64] # peakPhases × 2·peakHalf windowed-sinc taps, each phase summing to 1
    params: TrackParams
    dtMs: float64
    samples, step, points, lanes: int

  SweepExtent* = object
    ## Where a reference's sweep lies, in ms from the rows' first sample (NaN
    ## for a reference without a sweep): the sweep and its flat part, between
    ## the tapers. See the module documentation, "The flat part".
    startMs*: float64 = NaN ## The sweep's first sample (its envelope within `minReferenceDb`).
    endMs*: float64 = NaN ## Its last sample.
    flatStartMs*: float64 = NaN ## Where the start taper ends: the flat part's first sample.
    flatEndMs*: float64 = NaN ## Where the end taper starts: its last sample.

  SweepReference* = object
    ## A reference prepared by `referenceInto`: its phase and frequency per
    ## sample, its extent and each point's windows. Make it with
    ## `initSweepReference`.
    samples, points: int # the plan's (a shape check)
    dtMs: float64
    first, last: int # the sweep's samples (inclusive); first = −1: no sweep
    flatFirst, flatLast: int # its flat part (inclusive)
    psi, cosPsi, sinPsi, freqHz: seq[float64] # per sample: the unwrapped phase, e^{iψ}, f
    lo, hi, lo1, hi1: seq[int] # per point: the C-period and one-period windows; lo = −1: not measured
    cyc: seq[int] # per point: the window's periods C (0: not measured)
    weight, refHz, maxHz: seq[float64] # per point: Σ w·f, f at the point, the largest f in the window
    groups: seq[int] # the distinct window lengths C of the measured points (the first groupCount)
    groupCount: int

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
    thdCount*: seq[int] ## The harmonics THD sums at the point (0 where THD is NaN).
    peak*: seq[float64] ## The largest `|s|` within one period around the point.
    extent*: SweepExtent ## The reference's sweep and flat part (`SweepReference.extent`).

const
  maxHarmonics = 16
  peakPhases = 8      # sub-sample positions of the band-limited peak
  peakHalf = 8        # taps each side of the interpolator
  peakBeta = 8.0      # its Kaiser window
  sumCount = 10       # running sums per harmonic: 5 weights, real and imaginary
  flatLookMs = 10.0   # the flat part: the envelope, averaged over a period, rises (falls)
  flatRise = 2e-4     # by at most this fraction of itself over the next (previous) flatLookMs,
  flatSlowdown = 0.02 # or of its fastest rise (fall) before
  maxMinWindowMs = 1e6

proc badTrack(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc validate*(p: TrackParams) =
  ## Checks the parameters (field documentation). Raises `ValueError`.
  if p.harmonics notin 1 .. maxHarmonics:
    badTrack("harmonics must be in 1 .. " & $maxHarmonics & ", got " & $p.harmonics)
  if p.cycles notin 3 .. 1000:
    badTrack("cycles must be a whole number of periods in 3 .. 1000 (over fewer, the " &
      "window is not orthogonal to the neighbouring harmonics), got " & $p.cycles)
  if not (p.minWindowMs >= 0 and p.minWindowMs <= maxMinWindowMs):
    badTrack("minWindowMs must be in 0 .. " & $maxMinWindowMs & ", got " & $p.minWindowMs)
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
  # In float64 first: a step past the trace (or past an int) is one point.
  let stepF = round(params.stepMs / dtMs)
  result.step = (if stepF >= float64(samples): samples else: max(1, int(stepF)))
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
  let sums = max(5 * (tp.samples + 1), tp.points * (sumCount * tp.lanes + 1))
  when T is float32: WorkspaceSize(f32: own, f64: sums)
  else: WorkspaceSize(f64: own + sums)

proc initSweepReference*[T](tp: SweepTrackPlan[T]): SweepReference =
  ## An empty reference sized for `tp` (no sweep until `referenceInto`).
  let n = tp.samples
  let p = tp.points
  result.samples = n
  result.points = p
  result.dtMs = tp.dtMs
  result.first = -1
  result.last = -1
  result.flatFirst = -1
  result.flatLast = -1
  for s in [addr result.psi, addr result.cosPsi, addr result.sinPsi, addr result.freqHz]:
    s[] = newSeq[float64](n)
  for s in [addr result.lo, addr result.hi, addr result.lo1, addr result.hi1]:
    s[] = newSeq[int](p)
    for v in s[].mitems: v = -1
  result.cyc = newSeq[int](p)
  result.groups = newSeq[int](p)
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
  result.thdCount = newSeq[int](p)
  result.extent = SweepExtent()
  result.amplitude = newSeq[seq[float64]](tp.params.harmonics)
  for a in result.amplitude.mitems:
    a = newSeq[float64](p)
    for v in a.mitems: v = NaN

func span*(r: SweepReference): Slice[int] {.inline, raises: [].} =
  ## The samples of the sweep (where the reference's envelope is within
  ## `minReferenceDb` of its peak); empty (`0 .. -1`) when there is none.
  if r.first < 0: 0 .. -1 else: r.first .. r.last

func flatSpan*(r: SweepReference): Slice[int] {.inline, raises: [].} =
  ## The samples of the sweep's flat part, between its tapers (module
  ## documentation); empty (`0 .. -1`) without a sweep.
  if r.first < 0: 0 .. -1 else: r.flatFirst .. r.flatLast

func extent*(r: SweepReference): SweepExtent {.raises: [].} =
  ## The sweep and its flat part in ms (`span` and `flatSpan` times the
  ## interval); every field NaN without a sweep.
  result = SweepExtent() # (`result` alone would be zeros, not the fields' defaults)
  if r.first >= 0:
    result = SweepExtent(startMs: float64(r.first) * r.dtMs, endMs: float64(r.last) * r.dtMs,
      flatStartMs: float64(r.flatFirst) * r.dtMs, flatEndMs: float64(r.flatLast) * r.dtMs)

func isMeasured*(r: SweepReference, point: int): bool {.inline, raises: [].} =
  ## Point `point`'s window lies inside the sweep.
  point >= 0 and point < r.lo.len and r.lo[point] >= 0

func cyclesAt*(r: SweepReference, point: int): int {.inline, raises: [].} =
  ## The periods of the reference in point `point`'s window (`cycles`, or
  ## more where `minWindowMs` raises it); 0 where it is not measured.
  if r.isMeasured(point): r.cyc[point] else: 0

func len*(t: SweepTrack): int {.inline, raises: [].} =
  ## The number of points.
  t.timeMs.len

func harmonics*(t: SweepTrack): int {.inline, raises: [].} =
  ## The highest harmonic order tracked.
  t.amplitude.len

func sizedFor(t: SweepTrack, points, harmonics: int): bool {.raises: [].} =
  ## Every per-point field of `t` has `points` values (its fields are the
  ## caller's to read, and could have been resized).
  result = t.timeMs.len == points and t.referenceHz.len == points and
    t.frequencyHz.len == points and t.phaseDeg.len == points and
    t.thdPct.len == points and t.thdCount.len == points and t.peak.len == points and
    t.amplitude.len == harmonics
  if result:
    for a in t.amplitude:
      if a.len != points: return false

# ------------------------------------------------------------- reference --

func wrapDeg*(deg: float64): float64 {.raises: [].} =
  ## An angle in degrees wrapped to (−180, 180] (NaN stays NaN).
  let r = floorMod(deg, 360.0)
  if r > 180: r - 360 else: r

func windowCycles(p: TrackParams, f: float64): int {.raises: [].} =
  ## The window's periods at the frequency `f`: `cycles`, or where those
  ## are shorter than `minWindowMs`, the smallest even number that is not
  ## (an odd count lets a subharmonic at f/2 into the fundamental).
  result = p.cycles
  let need = p.minWindowMs * f / 1000 # periods in minWindowMs
  if need > float64(p.cycles) * (1 + 1e-9): # (a rounding past the edge is not)
    result = 2 * int(ceil(need / 2 - 1e-9))

proc flatPart(e: ptr UncheckedArray[float64], first, last, look: int,
    floor: float64): (int, int) {.raises: [].} =
  ## The flat part of the sweep `first .. last` from its envelope averaged
  ## over a period, `e[j − first]`: from the first sample above `floor`
  ## after which it rises over `look` samples by no more than `flatRise` of
  ## itself or `flatSlowdown` of the fastest rise before (a plateau with a
  ## slope is flat), to the last before which it fell by no more. A taper
  ## is looked for in its half of the sweep: without one there (an envelope
  ## that keeps rising), that end of the sweep.
  template at(j: int): float64 = e[j - first]
  let mid = (first + last) div 2
  result = (first, last)
  var fastest = 0.0
  for j in first .. mid:
    var top = at(j)
    for k in j + 1 .. min(last, j + look): top = max(top, at(k))
    let rise = top - at(j)
    fastest = max(fastest, rise)
    if at(j) >= floor and rise <= max(flatRise * at(j), flatSlowdown * fastest):
      result[0] = j
      break
  fastest = 0.0
  for j in countdown(last, mid):
    var top = at(j)
    for k in max(first, j - look) .. j - 1: top = max(top, at(k))
    let fall = top - at(j)
    fastest = max(fastest, fall)
    if at(j) >= floor and fall <= max(flatRise * at(j), flatSlowdown * fastest):
      result[1] = j
      break

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
    dst.cyc[m] = 0
    dst.weight[m] = NaN
    dst.refHz[m] = NaN
    dst.maxHz[m] = NaN
  dst.first = -1
  dst.last = -1
  dst.flatFirst = -1
  dst.flatLast = -1
  dst.groupCount = 0
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
  let bandHz = float64(ka) * 1000 / (float64(nf) * tp.dtMs)       # the band's lower edge
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
  # The unwrapped phase over the sweep and the unit phasor e^{iψ} from the
  # analytic signal itself.
  template psi: untyped = dst.psi
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
  # The flat part, from the envelope averaged over one period of the
  # reference around each sample (which leaves out the ripple an abrupt
  # start puts into the analytic envelope, a 1/t tail of the Hilbert
  # transform): its prefix sums in run 0, the averages in run 1.
  let stride = tp.samples + 1
  let sums = ws.scratch(float64, 5 * stride, at = tp.sumsAt)
  template run(q, i: int): untyped = sums[q * stride + i]
  run(0, 0) = 0
  for j in first .. last:
    run(0, j - first + 1) = run(0, j - first) + hypot(float64(xh[j]), float64(h[j]))
  # The period from the frequency, which is not to be trusted where the
  # envelope is small (it read 1 Hz at the −40 dB start of an 8 Hz sweep):
  # never longer than at half the band's lower edge.
  var top = 0.0
  for j in first .. last:
    var f = dst.freqHz[j]
    if not (f >= 0.5 * bandHz): f = 0.5 * bandHz
    let half = (if f < Inf: min(last - first, int(round(500 / (f * tp.dtMs)))) else: 0)
    let a = max(first, j - half)
    let b = min(last, j + half)
    let e = (run(0, b - first + 1) - run(0, a - first)) / float64(b - a + 1)
    run(1, j - first) = e
    top = max(top, e)
  (dst.flatFirst, dst.flatLast) = flatPart(cast[ptr UncheckedArray[float64]](addr run(1, 0)),
    first, last, max(1, int(round(flatLookMs / tp.dtMs))), 0.5 * top)
  # Each point's window: C periods (`windowCycles`) centred on the point,
  # from ψ; the pointers move both ways, as C changes from point to point.
  var lo = first
  var hi = first
  var lo1 = first
  var hi1 = first
  for m in 0 ..< tp.points:
    let c = m * tp.step
    if c < first or c > last: continue
    let C = windowCycles(tp.params, dst.freqHz[c])
    let halfC = PI * float64(C)
    let pc = psi[c]
    if not (pc - halfC >= psi[first]): continue # the window starts before the sweep
    while lo > first and psi[lo - 1] > pc - halfC: dec lo
    while lo < c and psi[lo] <= pc - halfC: inc lo
    while hi > c + 1 and psi[hi - 1] >= pc + halfC: dec hi
    while hi <= last and psi[hi] < pc + halfC: inc hi
    if hi > last: continue # the window ends after the sweep
    while lo1 < c and psi[lo1] <= pc - PI: inc lo1
    hi1 = max(hi1, c)
    while hi1 <= last and psi[hi1] < pc + PI: inc hi1
    dst.lo[m] = lo
    dst.hi[m] = hi
    dst.lo1[m] = lo1
    dst.hi1[m] = hi1
    dst.cyc[m] = C
    dst.refHz[m] = dst.freqHz[c]
    let g = dst.groupCount # the distinct window lengths, in order of first use
    if g == 0 or dst.groups[g - 1] != C:
      var seen = false
      for i in 0 ..< g:
        if dst.groups[i] == C: seen = true
      if not seen:
        dst.groups[g] = C
        inc dst.groupCount
  # Per window length C: the windows made monotone along the points (they
  # are, unless ψ steps back somewhere, which a scan cannot follow), and
  # the weight from the running sums Σ f·(1, cos ψ/C, sin ψ/C, cos 2ψ/C,
  # sin 2ψ/C) over the group's samples: in run q, index i holds the sum over
  # `g0 ..< g0 + i`.
  for g in 0 ..< dst.groupCount:
    let C = dst.groups[g]
    let Cf = float64(C)
    var g0 = high(int) # the group's samples, g0 ..< g1
    var g1 = -1
    var lastLo, lastHi = -1
    for m in 0 ..< tp.points:
      if dst.lo[m] < 0 or dst.cyc[m] != C: continue
      dst.lo[m] = max(dst.lo[m], lastLo)
      dst.hi[m] = max(dst.hi[m], lastHi)
      (lastLo, lastHi) = (dst.lo[m], dst.hi[m])
      g0 = min(g0, dst.lo[m])
      g1 = max(g1, dst.hi[m])
      let c = m * tp.step
      dst.maxHz[m] = max(dst.freqHz[dst.lo[m]], max(dst.freqHz[dst.hi[m] - 1], dst.freqHz[c]))
    for q in 0 ..< 5: run(q, 0) = 0
    for j in g0 ..< g1:
      let i = j - g0
      let f = dst.freqHz[j]
      let a = psi[j] / Cf
      let (c1, s1) = (cos(a), sin(a))
      run(0, i + 1) = run(0, i) + f
      run(1, i + 1) = run(1, i) + f * c1
      run(2, i + 1) = run(2, i) + f * s1
      run(3, i + 1) = run(3, i) + f * (c1 * c1 - s1 * s1)
      run(4, i + 1) = run(4, i) + f * (2 * s1 * c1)
    for m in 0 ..< tp.points:
      if dst.lo[m] < 0 or dst.cyc[m] != C: continue
      let c = m * tp.step
      let a = dst.lo[m] - g0
      let b = dst.hi[m] - g0
      let ac = psi[c] / Cf
      let (c1, s1) = (cos(ac), sin(ac))
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
  let ac = r.psi[c] / float64(r.cyc[m])
  let c1 = cos(ac)
  let s1 = sin(ac)
  # The window's weights at the centre: cos⁴ = (3 + 4·cos(x/C) + cos(2x/C))/8.
  let coef = [3.0, 4 * c1, 4 * s1, c1 * c1 - s1 * s1, 2 * s1 * c1]
  let w = r.weight[m]
  let nyq = 500 / tp.dtMs
  let limit = tp.params.maxHarmonicFraction * nyq
  var a1, harm = 0.0
  var measured, missing = 0 # harmonics (k ≥ 2) measured, and past the limit
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
      if k > 1: inc missing
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
  elif measured == 0 or not (a1 > 0) or (missing > 0 and tp.params.thdHarmonics == thFixed):
    track.thdPct[m] = NaN
  else:
    track.thdPct[m] = 100 * sqrt(harm) /
      (if tp.params.thd == thdFundamental: a1 else: sqrt(a1 * a1 + harm))
    track.thdCount[m] = measured

proc scan[T; L: static int](tp: SweepTrackPlan[T], x: openArray[T], r: SweepReference,
    track: var SweepTrack, ws: var Workspace, C: int) {.raises: [].} =
  ## One pass over the samples of the points whose window is `C` periods:
  ## the sums are snapshotted at each one's window start, and the point
  ## finished at its end. A NaN or infinite sample is left out of the sums
  ## and counted (the count snapshotted too): the points whose window holds
  ## one stay NaN, the others are measured.
  const stride = sumCount * L + 1 # a snapshot: the sums, then the count
  let snaps = ws.scratch(float64, tp.points * stride, tp.sumsAt)
  let Cf = float64(C)
  # The group's points, in order; their windows' starts and ends both
  # increase with the point (`referenceInto`).
  var mLo = 0 # the next point whose window starts
  var mHi = 0 # the next point whose window ends
  template skip(m: var int) =
    while m < tp.points and (r.lo[m] < 0 or r.cyc[m] != C): inc m
  skip(mLo)
  skip(mHi)
  if mLo >= tp.points: return
  let j0 = r.lo[mLo]
  var acc: Lanes[L]
  var bad = 0 # the non-finite samples so far
  var j = j0
  let n = x.len
  while mHi < tp.points:
    while mLo < tp.points and r.lo[mLo] == j:
      copyMem(addr snaps[mLo * stride], addr acc, sizeof(acc))
      snaps[mLo * stride + stride - 1] = float64(bad)
      inc mLo
      skip(mLo)
    while mHi < tp.points and r.hi[mHi] == j:
      if float64(bad) == snaps[mHi * stride + stride - 1]: # else NaN, as trackInto set it
        let before = cast[ptr Lanes[L]](addr snaps[mHi * stride])
        tp.finishPoint(r, mHi, acc, before[], track)
      inc mHi
      skip(mHi)
    if mHi >= tp.points: break
    let t = (if j < n: r.freqHz[j] * float64(x[j]) else: 0.0)
    if t - t == 0:
      let a = r.psi[j] / Cf
      acc.addSample(t, cos(a), sin(a), r.cosPsi[j], -r.sinPsi[j])
    else: inc bad
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
  doAssert track.sizedFor(tp.points, tp.params.harmonics), "the track is not sized for this plan"
  doAssert ws.fits(tp.workspaceSize), "reserve the plan's workspaceSize first"
  for m in 0 ..< tp.points:
    track.timeMs[m] = tp.pointMs(m)
    track.referenceHz[m] = reference.refHz[m]
    track.frequencyHz[m] = NaN
    track.phaseDeg[m] = NaN
    track.thdPct[m] = NaN
    track.thdCount[m] = 0
    track.peak[m] = NaN
    for a in track.amplitude.mitems: a[m] = NaN
  track.extent = reference.extent
  if reference.first < 0: return
  for g in 0 ..< reference.groupCount: # one scan per window length
    let C = reference.groups[g]
    if tp.lanes == 8: tp.scan[:T, 8](x, reference, track, ws, C)
    else: tp.scan[:T, 16](x, reference, track, ws, C)
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
