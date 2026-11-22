## A reference sweep through a recorder's filter, with the filter's bulk delay
## backed out (`RecordingFilter`, `applyRecordingFilter`).
##
## ```nim
## let p = generatePilot(linearSweep(6, 200, 12000, 2, initSweepTaper(250, 250)), float32)
## # A minimum-phase 207 Hz Butterworth of 6 poles, its delay backed out by a
## # bulk shift of one sample (the recorder's, typed from its datasheet).
## let f = RecordingFilter(kind: rfkMinimumPhase, family: rffButterworth,
##   highCutHz: 207, highOrder: 6, shiftSamples: 1)
## let ref = p.applyRecordingFilter(f)
## doAssert ref.len == p.len and ref.t0Ms == p.t0Ms
## ```
##
## **What is given.** Every parameter is the caller's: nothing is read from
## headers. The filter is one of
## - `rfkNone`: the pilot is returned as it is (the same samples, bit for bit);
## - `rfkLinearPhase`: a symmetric FIR of `taps` coefficients designed from a
##   high-cut and a low-cut (cutoff Hz and roll-off in dB/octave each, 0 Hz =
##   that cut is off): the target gain is a Butterworth-shaped magnitude
##   `1/sqrt(1 + (f/fc)^(slope/3))` (a high-cut; `(fc/f)` for a low-cut; −3 dB
##   at the corner), sampled finely, turned into the impulse response of
##   delay `(taps − 1)/2` samples and cut to `taps` by a Hamming window. A
##   finite filter cannot be steeper than its length allows: the real
##   roll-off is the smaller of the typed one and the window's, and a
##   low-cut below about `2·fs/taps` Hz (`fs` the sampling rate) cannot be
##   made at all. Typed `coefficients` replace the design (a symmetric set is
##   a linear-phase filter; `coefficientsSymmetric` tells);
## - `rfkMinimumPhase`, either by `coefficients` (a causal FIR, as a datasheet
##   lists them, used as they are; the other fields except `shiftSamples` are
##   ignored) or by `family` (`rffButterworth`, or `rffChebyshev` with
##   `rippleDb` of passband ripple): a high-cut of `highOrder` poles at
##   `highCutHz` and a low-cut of `lowOrder` poles at `lowCutHz` (0 = off), by
##   the prewarped bilinear transform, run as a cascade of second-order
##   sections from rest. The corner is −3 dB for Butterworth and the
##   ripple's band edge for Chebyshev.
##
## **The bulk delay.** `shiftSamples` is a whole number of samples that the
## recorder moved its output EARLIER after filtering, to back out the
## filter's delay; negative moves the output later. Linear phase: the delay
## is exactly `(taps − 1)/2` (`linearDelaySamples`); an even tap count gives
## half a sample, which no whole shift removes (`linearShiftSamples`, the
## whole shift suggested, is `floor((taps − 1)/2)`). Minimum phase: the delay
## depends on frequency, one whole shift cannot remove it, and what the
## recorder subtracts is its own number: `bulkDelaySamples` and
## `suggestedShift` estimate it (least squares over a band) and are only a
## suggestion.
##
## **Length.** The result has exactly the pilot's samples and `t0Ms`. The
## filter runs from rest on the pilot followed by zeros, over `shiftSamples`
## samples past the end when positive, so the ring-out that shifting brings
## inside the pilot is in the result; what is still beyond the last sample
## is cut. Samples shifted in at the start (a negative shift) are zeros.
## A shift of `len` samples or more raises `ValueError`.

import std/[complex, math]
import parseSEG
import ../core/types
import pilot

type
  RecFilterKind* = enum
    rfkNone         ## no filter: the pilot is unchanged
    rfkLinearPhase  ## a symmetric FIR designed from cutoffs and roll-offs
    rfkMinimumPhase ## typed coefficients, or a Butterworth/Chebyshev family

  RecFilterFamily* = enum
    rffButterworth ## maximally flat, −3 dB at the cutoff
    rffChebyshev   ## `rippleDb` of ripple in the passband, edge at the cutoff

  RecordingFilter* = object
    ## The recorder's filter and its bulk delay, all typed by the caller.
    kind*: RecFilterKind
    taps*: int                     ## linear phase: the number of coefficients, 3 .. `maxDesignTaps`
    highCutHz*: float64            ## 0 = no high-cut
    highSlopeDbPerOctave*: float64 ## linear phase: the roll-off, above 0
    lowCutHz*: float64             ## 0 = no low-cut
    lowSlopeDbPerOctave*: float64  ## linear phase: the roll-off, above 0
    coefficients*: seq[float64]    ## minimum phase: the FIR taps; empty = use `family`
    family*: RecFilterFamily
    highOrder*: int                ## minimum phase family: poles of the high-cut, 1 .. 32
    lowOrder*: int                 ## minimum phase family: poles of the low-cut, 1 .. 32
    rippleDb*: float64             ## Chebyshev: passband ripple, above 0 up to 12
    shiftSamples*: int             ## whole samples moved earlier after filtering (negative: later)

  Section = object
    b0, b1, b2, a1, a2: float64

const
  minRippleDb* = 0.001     ## the least Chebyshev ripple (below it the poles overflow)
  maxDesignTaps* = 2047    ## the longest linear-phase filter designed from cutoffs
  maxCoefficients* = 16384 ## the most typed coefficients
  maxOrder* = 32           ## the most poles of one cut
  maxRippleDb* = 12.0
  maxSlopeDbPerOctave* = 1000.0

proc badFilter(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func nyquistHz(dtMs: float64): float64 = 500 / dtMs

func linearDelaySamples*(taps: int): float64 {.raises: [].} =
  ## The delay of a symmetric FIR of `taps` coefficients: `(taps − 1)/2`
  ## samples at every frequency.
  (float64(taps) - 1) / 2

func linearShiftSamples*(taps: int): int {.raises: [].} =
  ## The whole-sample shift that backs out a symmetric FIR of `taps`
  ## coefficients: `floor((taps − 1)/2)` (31 for 64 taps, which delay by 31.5:
  ## the half sample is left). 0 below 2 taps.
  if taps <= 1: 0 else: (taps - 1) div 2

func coefficientsSymmetric*(c: openArray[float64]): bool {.raises: [].} =
  ## Whether `c` reads the same backwards (to 1e-9 of its largest value): the
  ## coefficients of a linear-phase filter.
  var big = 0.0
  for x in c: big = max(big, abs(x))
  for i in 0 ..< c.len div 2:
    if abs(c[i] - c[c.high - i]) > 1e-9 * big: return false
  true

# ------------------------------------------------------------------- design --

func badNum(x: float64): bool = x != x or x == Inf or x == -Inf

proc checkCut(hz: float64, what: string, dtMs: float64) =
  if badNum(hz) or hz < 0: badFilter(what & " must be 0 (off) or a frequency above 0")
  if hz >= nyquistHz(dtMs):
    badFilter(what & " " & $hz & " Hz is not below Nyquist (" & $nyquistHz(dtMs) & " Hz)")

proc checkFilter*(f: RecordingFilter, dtMs: float64, samples = 0) =
  ## Raises `ValueError` saying what is wrong: a cutoff not below Nyquist,
  ## a low-cut not below the high-cut, no cut at all, a slope, order, ripple
  ## or tap count out of range, non-finite coefficients (typed `coefficients`
  ## replace the design of either kind), or (with `samples`
  ## above 0) a shift of `samples` or more.
  if f.kind == rfkNone: return
  if samples > 0 and (f.shiftSamples >= samples or f.shiftSamples <= -samples):
    badFilter("a shift of " & $f.shiftSamples & " samples does not fit " & $samples & " samples")
  case f.kind
  of rfkNone: discard
  of rfkLinearPhase, rfkMinimumPhase:
    if f.coefficients.len > 0:
      if f.coefficients.len > maxCoefficients:
        badFilter("at most " & $maxCoefficients & " coefficients")
      var any = false
      for c in f.coefficients:
        if badNum(c): badFilter("a coefficient is not a finite number")
        if c != 0: any = true
      if not any: badFilter("the coefficients are all zero")
    elif f.kind == rfkLinearPhase:
      if f.taps < 3 or f.taps > maxDesignTaps:
        badFilter("the number of taps must be 3 to " & $maxDesignTaps)
      checkCut(f.highCutHz, "the high-cut", dtMs)
      checkCut(f.lowCutHz, "the low-cut", dtMs)
      if f.highCutHz == 0 and f.lowCutHz == 0: badFilter("give a high-cut or a low-cut")
      if f.highCutHz > 0 and f.lowCutHz >= f.highCutHz:
        badFilter("the low-cut must be below the high-cut")
      for (hz, s, what) in [(f.highCutHz, f.highSlopeDbPerOctave, "high-cut"),
          (f.lowCutHz, f.lowSlopeDbPerOctave, "low-cut")]:
        if hz > 0 and (badNum(s) or s <= 0 or s > maxSlopeDbPerOctave):
          badFilter("the " & what & " roll-off must be above 0 and at most " &
            $int(maxSlopeDbPerOctave) & " dB/octave")
    else:
      checkCut(f.highCutHz, "the high-cut", dtMs)
      checkCut(f.lowCutHz, "the low-cut", dtMs)
      if f.highCutHz == 0 and f.lowCutHz == 0: badFilter("give a high-cut or a low-cut")
      if f.highCutHz > 0 and f.lowCutHz >= f.highCutHz:
        badFilter("the low-cut must be below the high-cut")
      if f.highCutHz > 0 and (f.highOrder < 1 or f.highOrder > maxOrder):
        badFilter("the high-cut order must be 1 to " & $maxOrder)
      if f.lowCutHz > 0 and (f.lowOrder < 1 or f.lowOrder > maxOrder):
        badFilter("the low-cut order must be 1 to " & $maxOrder)
      if f.family == rffChebyshev and (badNum(f.rippleDb) or f.rippleDb < minRippleDb or
          f.rippleDb > maxRippleDb):
        badFilter("the ripple must be " & $minRippleDb & " to " & $int(maxRippleDb) & " dB")

proc linearPhaseTaps*(f: RecordingFilter, dtMs: float64): seq[float64] =
  ## The coefficients of a linear-phase filter `f` (see the module
  ## documentation): symmetric, delay `(taps − 1)/2`. `ValueError` as
  ## `checkFilter`.
  doAssert f.kind == rfkLinearPhase and f.coefficients.len == 0
  checkFilter(f, dtMs)
  let n = f.taps
  let c = linearDelaySamples(n)
  let ng = max(4096, 4 * n)
  let hi2 = f.highSlopeDbPerOctave / 3 # exponent 2N of |H| = 1/sqrt(1 + x^2N)
  let lo2 = f.lowSlopeDbPerOctave / 3
  let nyq = nyquistHz(dtMs)
  var mag = newSeq[float64](ng)
  var om = newSeq[float64](ng)
  for i in 0 ..< ng:
    let w = (float64(i) + 0.5) * PI / float64(ng)
    om[i] = w
    let hz = w / PI * nyq
    var m = 1.0
    if f.highCutHz > 0: m *= 1 / sqrt(1 + pow(hz / f.highCutHz, hi2))
    if f.lowCutHz > 0: m *= 1 / sqrt(1 + pow(f.lowCutHz / hz, lo2))
    mag[i] = m
  result = newSeq[float64](n)
  for k in 0 ..< n:
    var s = 0.0
    let d = float64(k) - c
    for i in 0 ..< ng: s += mag[i] * cos(om[i] * d)
    let win = 0.54 - 0.46 * cos(2 * PI * float64(k) / float64(n - 1))
    result[k] = s / float64(ng) * win

func sectionsOf(fc, dtMs: float64, order: int, high: bool, family: RecFilterFamily,
    rippleDb: float64): tuple[sections: seq[Section], scale: float64] =
  ## The cascade of a digital Butterworth/Chebyshev-I low-pass (`high = false`)
  ## or high-pass of `order` poles, corner `fc`, each section of gain 1 at 0 Hz
  ## (Nyquist for a high-pass), and the overall gain of an even-order Chebyshev.
  let fs = 1000 / dtMs
  let wc = 2 * fs * tan(PI * fc / fs)
  let k2 = 2 * fs
  var sigma, omega = 1.0 # ellipse axes of the prototype's poles
  var scale = 1.0
  if family == rffChebyshev:
    let eps = sqrt(pow(10.0, rippleDb / 10) - 1)
    let mu = arcsinh(1 / eps) / float64(order)
    sigma = sinh(mu)
    omega = cosh(mu)
    if order mod 2 == 0: scale = pow(10.0, -rippleDb / 20)
  proc bilinear(s: Complex64): Complex64 =
    (complex(k2, 0.0) + s) / (complex(k2, 0.0) - s)
  var secs: seq[Section] = @[]
  for k in 0 ..< order div 2:
    let th = PI * float64(2 * k + 1) / float64(2 * order)
    var p = complex(-sigma * sin(th), omega * cos(th))
    p = if high: complex(1.0, 0.0) / p * wc else: p * wc
    let z = bilinear(p)
    let a1 = -2 * z.re
    let a2 = z.re * z.re + z.im * z.im
    if high:
      let g = (1 - a1 + a2) / 4
      secs.add Section(b0: g, b1: -2 * g, b2: g, a1: a1, a2: a2)
    else:
      let g = (1 + a1 + a2) / 4
      secs.add Section(b0: g, b1: 2 * g, b2: g, a1: a1, a2: a2)
  if order mod 2 == 1:
    var p = complex(-sigma, 0.0)
    p = if high: complex(1.0, 0.0) / p * wc else: p * wc
    let z0 = bilinear(p).re
    if high:
      # H(z) = g (1 − z⁻¹)/(1 − z0 z⁻¹); at z = −1: g·2/(1 + z0) = 1.
      let gg = (1 + z0) / 2
      secs.add Section(b0: gg, b1: -gg, b2: 0, a1: -z0, a2: 0)
    else:
      let gg = (1 - z0) / 2 # at z = 1: g·2/(1 − z0) = 1
      secs.add Section(b0: gg, b1: gg, b2: 0, a1: -z0, a2: 0)
  (secs, scale)

func cascadeOf(f: RecordingFilter, dtMs: float64, highPart, lowPart: bool):
    tuple[sections: seq[Section], scale: float64] =
  result.scale = 1.0
  if highPart and f.highCutHz > 0:
    let (s, g) = sectionsOf(f.highCutHz, dtMs, f.highOrder, false, f.family, f.rippleDb)
    result.sections.add s
    result.scale *= g
  if lowPart and f.lowCutHz > 0:
    let (s, g) = sectionsOf(f.lowCutHz, dtMs, f.lowOrder, true, f.family, f.rippleDb)
    result.sections.add s
    result.scale *= g

func firOf(f: RecordingFilter, dtMs: float64): seq[float64] =
  ## The coefficients of a FIR filter (`rfkLinearPhase`, or typed taps), else empty.
  case f.kind
  of rfkLinearPhase:
    if f.coefficients.len > 0: f.coefficients else: linearPhaseTaps(f, dtMs)
  of rfkMinimumPhase: f.coefficients
  of rfkNone: @[]

# ----------------------------------------------------------------- response --

func sectionsResponse(c: tuple[sections: seq[Section], scale: float64],
    w: float64): Complex64 =
  let z1 = complex(cos(w), -sin(w))
  result = complex(c.scale, 0.0)
  for s in c.sections:
    result *= (s.b0 + s.b1 * z1 + s.b2 * z1 * z1) / (1.0 + s.a1 * z1 + s.a2 * z1 * z1)

func firResponse(h: openArray[float64], w: float64): Complex64 =
  var re, im = 0.0
  for k, c in h:
    re += c * cos(w * float64(k))
    im -= c * sin(w * float64(k))
  complex(re, im)

type Design = object
  ## A filter's coefficients or sections, made once for many frequencies.
  fir: seq[float64]
  casc: tuple[sections: seq[Section], scale: float64]

proc designOf(f: RecordingFilter, dtMs: float64): Design =
  checkFilter(f, dtMs)
  result.casc.scale = 1.0
  if f.kind == rfkMinimumPhase and f.coefficients.len == 0:
    result.casc = cascadeOf(f, dtMs, true, true)
  else:
    result.fir = firOf(f, dtMs)

func responseOf(d: Design, w: float64): Complex64 =
  if d.fir.len > 0: firResponse(d.fir, w) else: sectionsResponse(d.casc, w)

proc responseAt*(f: RecordingFilter, dtMs, hz: float64): Complex64 =
  ## The complex gain of the whole filter at `hz` Hz (both cuts; the shift
  ## is not included): 1 for `rfkNone`. `ValueError` as `checkFilter`.
  if f.kind == rfkNone: return complex(1.0, 0.0)
  designOf(f, dtMs).responseOf(2 * PI * hz * dtMs / 1000)

proc checkBand(dtMs, fromHz, toHz: float64) =
  if not (fromHz > 0 and fromHz < toHz and toHz < nyquistHz(dtMs)):
    badFilter("the band must satisfy 0 < from < to < Nyquist (" & $nyquistHz(dtMs) & " Hz)")

proc wrapPi(x: float64): float64 =
  result = x - 2 * PI * floor((x + PI) / (2 * PI))

proc bulkDelaySamples*(f: RecordingFilter, dtMs, fromHz, toHz: float64): float64 =
  ## An estimate of the filter's bulk delay in samples over `fromHz .. toHz`:
  ## the least-squares delay of a line through 0 fitted to the unwrapped phase
  ## (`τ = −Σφω / Σω²`, `ω = 2π·f·dt`, 2000 points, phase unwrapped from 0 Hz
  ## in steps of at most 0.1 Hz). Linear phase: exactly `(taps − 1)/2`. A
  ## minimum-phase family: its high-cut alone (a low-cut leads in phase and
  ## has no delay to remove); typed coefficients: all of them. 0 for no
  ## filter. This is a suggestion: the recorder's own shift is what counts.
  ## `ValueError` as `checkFilter`, or for a band not within
  ## `0 < from < to < Nyquist`.
  if f.kind == rfkNone: return 0
  checkFilter(f, dtMs)
  checkBand(dtMs, fromHz, toHz)
  if f.kind == rfkLinearPhase:
    return linearDelaySamples(if f.coefficients.len > 0: f.coefficients.len else: f.taps)
  let fir = if f.kind == rfkMinimumPhase: f.coefficients else: @[]
  let casc = cascadeOf(f, dtMs, true, false)
  if fir.len == 0 and casc.sections.len == 0: return 0
  proc phaseAt(hz: float64): float64 =
    let w = 2 * PI * hz * dtMs / 1000
    let h = if fir.len > 0: firResponse(fir, w) else: sectionsResponse(casc, w)
    arctan2(h.im, h.re)
  var prev = 0.0 # 0 Hz: the gain is real
  var hz = 0.0
  let n0 = int(ceil(fromHz / 0.1))
  let step0 = fromHz / float64(n0)
  for i in 1 .. n0:
    hz = step0 * float64(i)
    prev += wrapPi(phaseAt(hz) - prev)
  var num, den = 0.0
  const points = 2000
  let step = (toHz - fromHz) / float64(points - 1)
  for i in 0 ..< points:
    let fh = fromHz + step * float64(i)
    if i > 0: prev += wrapPi(phaseAt(fh) - prev)
    let w = 2 * PI * fh * dtMs / 1000
    num += prev * w
    den += w * w
  -num / den

proc suggestedShift*(f: RecordingFilter, dtMs, fromHz, toHz: float64): int =
  ## Linear phase: `linearShiftSamples` of the tap count. Else
  ## `bulkDelaySamples` rounded to a whole sample, ties away from zero.
  if f.kind == rfkLinearPhase:
    checkFilter(f, dtMs)
    checkBand(dtMs, fromHz, toHz)
    return linearShiftSamples(if f.coefficients.len > 0: f.coefficients.len else: f.taps)
  let d = bulkDelaySamples(f, dtMs, fromHz, toHz)
  if d >= 0: int(floor(d + 0.5)) else: -int(floor(-d + 0.5))

proc shiftResidualDeg*(f: RecordingFilter, dtMs, fromHz, toHz: float64,
    shiftSamples: int): tuple[meanDeg, peakDeg: float64] =
  ## What the filter leaves in phase over the band after the shift: the
  ## mean and peak of `|phase|` of `H(f)·e^{jω·shift}`, in degrees, 2000
  ## points. It is what a reference WITHOUT the filter would be off by.
  checkBand(dtMs, fromHz, toHz)
  let d = designOf(f, dtMs)
  const points = 2000
  let step = (toHz - fromHz) / float64(points - 1)
  var sum, peak = 0.0
  for i in 0 ..< points:
    let hz = fromHz + step * float64(i)
    let h = d.responseOf(2 * PI * hz * dtMs / 1000)
    let w = 2 * PI * hz * dtMs / 1000 * float64(shiftSamples)
    let ph = abs(wrapPi(arctan2(h.im, h.re) + w)) * 180 / PI
    sum += ph
    peak = max(peak, ph)
  (sum / float64(points), peak)

# -------------------------------------------------------------------- apply --

proc applyRecordingFilter*[T: SampleType](p: Pilot[T], f: RecordingFilter): Pilot[T] =
  ## `p` through the filter `f` from rest, then its output moved by
  ## `f.shiftSamples` (see the module documentation for what is kept, filled
  ## and cut). `rfkNone` returns `p` itself, bit for bit. The result has
  ## `p`'s length, `dtMs`, `t0Ms` and sweep description. `ValueError` as
  ## `checkFilter`, with a shift of `p.len` or more.
  if f.kind == rfkNone: return p
  p.validate
  checkFilter(f, p.dtMs, p.len)
  let n = p.len
  let s = f.shiftSamples
  let total = n + max(s, 0)
  var y = newSeq[float64](total)
  let fir = firOf(f, p.dtMs)
  if fir.len > 0:
    for j in 0 ..< total:
      var acc = 0.0
      # y[j] = sum_k h[k] x[j-k], x = 0 past the last sample
      let kLo = max(0, j - (n - 1))
      for k in kLo .. min(fir.len - 1, j): acc += fir[k] * float64(p.samples[j - k])
      y[j] = acc
  else:
    let (secs, scale) = cascadeOf(f, p.dtMs, true, true)
    var z1 = newSeq[float64](secs.len)
    var z2 = newSeq[float64](secs.len)
    for j in 0 ..< total:
      var x = if j < n: float64(p.samples[j]) * scale else: 0.0
      for i, c in secs:
        let o = c.b0 * x + z1[i]
        z1[i] = c.b1 * x - c.a1 * o + z2[i]
        z2[i] = c.b2 * x - c.a2 * o
        x = o
      y[j] = x
  result = p
  for j in 0 ..< n:
    let src = j + s
    result.samples[j] = if src < 0: T(0) else: T(y[src])
    if badNum(float64(result.samples[j])):
      badFilter("the filter's output is not finite (a coefficient or gain is too large)")
