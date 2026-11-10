## Filter responses (`FilterSpec`, decisions D3 and D5): zero-phase magnitude
## responses for the frequency-domain filters, and the impulse length that
## sets the padding and the margin of a filter (D4).
##
## ```nim
## let bp = bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80))   # Hz, cos² ramps
## let lc = butterLowCut(2, slopeDbPerOctave = 24).withNotch(50, widthHz = 2)
## echo bp.response(6, dtMs = 2)       # 0.5: the middle of a ramp is −6 dB
## echo lc.impulseLength(dtMs = 2)     # samples on each side, 99.9 % energy
## ```
##
## **Responses.** Every response is a real gain `0 .. 1` per frequency, applied
## to the spectrum as it is (zero phase). Cutoffs are always sloped (D5): a
## brick wall cannot be requested.
## - **cos² trapezoid** (`bandPass`, `lowCut`, `highCut`; "Ormsby"): corners
##   `f1 < f2 ≤ f3 < f4` in Hz. The gain is 0 up to `f1`, rises as
##   `sin²(π/2·(f − f1)/(f2 − f1))` to 1 at `f2`, stays 1 to `f3` and falls as
##   `cos²` to 0 at `f4`. The middle of each ramp is −6 dB (gain 0.5). A
##   low-cut is a band-pass without high corners (`f3 = f4 = Inf`), a
##   high-cut one without low corners (`f1 = f2 = 0`).
## - **Butterworth magnitude** (`butterLowCut`, `butterHighCut`,
##   `butterBandPass`): `1/(1 + (w(f)/w(fc))^k)` for a high-cut and
##   `1/(1 + (w(fc)/w(f))^k)` for a low-cut, with `k = slope/6` and
##   `w(f) = tan(π·f·dt)`, the frequency warp of the bilinear transform. This
##   is exactly the magnitude of the digital Butterworth filters of
##   `filter/iir`, so the FFT and IIR forms of a filter agree; below a tenth
##   of the sample rate `w(f)/w(fc)` is `f/fc` within 3 %, and a high-cut is 0
##   at Nyquist. The slope is the
##   **effective** one, in dB/octave, of what is applied: this is `|H|²` of
##   a Butterworth filter of `k/2` poles, as a forward-backward (zero-phase)
##   pass gives, so each pole counts 12 dB/octave, not 6 (D5). The corner
##   `fc` is therefore −6 dB, not −3 dB. A slope that is not a multiple of 12
##   is allowed here (any `k`); the IIR filters (`filter/iir`) need multiples
##   of 12 (zero phase) or 6 (minimum phase, where the corner is −3 dB).
## - **Notches** (`withNotch`): gain 0 at `freqHz`, rising as `sin²` to 1 at
##   `freqHz ± widthHz/2`; −6 dB at `freqHz ± widthHz/4`.
## - **0 Hz suppression** (`withZeroDc`): the gain at 0 Hz is 0, so the
##   filtered trace has zero mean. `filter/apply` sets bin 0 to 0 and adds
##   back only the slope of the removed trend (times the gain at 0 Hz), not
##   its mean; with `trNone` only the mean of the padded window goes. It has
##   no impulse length of its own.
##
## **Impulse length** (`impulseLength`): the zero-phase impulse response is
## symmetric; its length is the number of samples `L` on each side of the
## centre that hold 99.9 % of its energy, taken for the response `H` and for
## its complement `1 − H`, whichever is longer. (A low-cut's impulse is a
## spike minus a narrow low-pass: the spike holds nearly all the energy, and
## the long tail that matters at the trace ends is the low-pass, the
## complement. The same holds for a notch.) Each edge and each notch is
## measured alone and the longest counts: in a band-pass the wide high side
## holds most of the energy and would hide a narrow low ramp. A ramp of width `w` Hz gives
## roughly `L ≈ 1/(w·dt)` samples, so a narrow low-cut ramp means a long
## filter. `filter/apply` pads each trace end by `L` (odd reflection) and
## reports `L` as the margin a zoom window needs.

import std/math
import parseSEG
import ../core/workspace
import ../fft/plan

type
  Corners* = object
    ## The corners of a cos² trapezoid, in Hz: `f1 < f2 ≤ f3 < f4`.
    ## `f2 = 0` means no low cut (`f1 = f2 = 0`), `f3 = Inf` no high cut
    ## (`f3 = f4 = Inf`).
    f1*, f2*, f3*, f4*: float64

  ResponseKind* = enum
    rkTrapezoid   ## cos² ramps between corners
    rkButterworth ## Butterworth magnitude, effective slope in dB/octave

  Notch* = object
    ## A notch: gain 0 at `freqHz`, 1 at `freqHz ± widthHz/2`.
    freqHz*, widthHz*: float64

  FilterSpec* = object
    ## A zero-phase response; see the module documentation. Build it with
    ## `bandPass`, `lowCut`, `highCut` or the `butter*` procs, then
    ## optionally `withNotch` and `withZeroDc`. `initFilterPlan` validates it
    ## against the sample interval.
    notches*: seq[Notch]
    zeroDc*: bool ## set bin 0 to 0 when applying
    case kind*: ResponseKind
    of rkTrapezoid:
      corners*: Corners
    of rkButterworth:
      lowHz*, lowSlope*: float64   ## low-cut corner (−6 dB) and slope; `lowHz = 0`: none
      highHz*, highSlope*: float64 ## high-cut corner and slope; `highHz = Inf`: none

const energyFraction* = 0.999
  ## The fraction of the impulse response's energy `impulseLength` keeps.

proc badFilter(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

# ------------------------------------------------------------- validation --

func hasLow(c: Corners): bool {.inline.} = c.f2 > 0
func hasHigh(c: Corners): bool {.inline.} = c.f3 < Inf

proc checkCorners(c: Corners) =
  for f in [c.f1, c.f2, c.f3, c.f4]:
    if f.isNaN: badFilter("a corner is NaN: " & $c)
  if c.hasLow and not (c.f1 >= 0 and c.f1 < c.f2):
    badFilter("a low cut needs 0 ≤ f1 < f2 (a sloped ramp), got " & $c)
  if not c.hasLow and (c.f1 != 0 or c.f2 != 0):
    badFilter("no low cut needs f1 = f2 = 0, got " & $c)
  if c.hasHigh and not (c.f3 < c.f4 and c.f4 < Inf):
    badFilter("a high cut needs f3 < f4 < Inf (a sloped ramp), got " & $c)
  if not c.hasHigh and c.f4 != Inf:
    badFilter("no high cut needs f3 = f4 = Inf, got " & $c)
  if c.hasLow and c.hasHigh and not (c.f2 <= c.f3):
    badFilter("a band-pass needs f2 ≤ f3, got " & $c)

proc checkSlope(s: float64) =
  if not (s > 0 and s < Inf):
    badFilter("a slope must be finite and > 0 dB/octave, got " & $s)

proc checkButter(lowHz, lowSlope, highHz, highSlope: float64) =
  if not (lowHz >= 0 and lowHz < Inf):
    badFilter("a low-cut corner must be finite and ≥ 0 (0: none), got " & $lowHz)
  if not (highHz > 0):
    badFilter("a high-cut corner must be > 0 (Inf: none), got " & $highHz)
  if lowHz > 0: checkSlope(lowSlope)
  if highHz < Inf: checkSlope(highSlope)
  if not (lowHz < highHz):
    badFilter("the low-cut corner must be below the high-cut one, got " & $lowHz &
      " and " & $highHz)

proc checkNotch(n: Notch) =
  if not (n.freqHz > 0 and n.freqHz < Inf):
    badFilter("a notch frequency must be finite and > 0, got " & $n.freqHz)
  if not (n.widthHz > 0 and n.widthHz < Inf):
    badFilter("a notch width must be finite and > 0, got " & $n.widthHz)

func nyquistHz*(dtMs: float64): float64 {.inline, raises: [].} =
  ## The Nyquist frequency at `dtMs`, in Hz.
  500 / dtMs

proc validate*(spec: FilterSpec, dtMs: float64) =
  ## Checks `spec` (its fields are public, so it may have been built by hand)
  ## and that every corner and notch is below Nyquist at `dtMs`. Raises
  ## `ValueError` otherwise.
  if not (dtMs > 0 and dtMs < Inf):
    badFilter("dtMs must be finite and > 0, got " & $dtMs)
  let nyq = nyquistHz(dtMs)
  case spec.kind
  of rkTrapezoid:
    let c = spec.corners
    checkCorners(c)
    if c.hasHigh and c.f4 > nyq:
      badFilter("f4 = " & $c.f4 & " Hz is above Nyquist (" & $nyq & " Hz at " &
        $dtMs & " ms)")
    if c.hasLow and c.f2 >= nyq:
      badFilter("f2 = " & $c.f2 & " Hz is not below Nyquist (" & $nyq & " Hz)")
  of rkButterworth:
    checkButter(spec.lowHz, spec.lowSlope, spec.highHz, spec.highSlope)
    if spec.highHz < Inf and spec.highHz >= nyq:
      badFilter("the high-cut corner " & $spec.highHz & " Hz is not below Nyquist (" &
        $nyq & " Hz)")
    if spec.lowHz >= nyq:
      badFilter("the low-cut corner " & $spec.lowHz & " Hz is not below Nyquist (" &
        $nyq & " Hz)")
  for n in spec.notches:
    checkNotch(n)
    if n.freqHz >= nyq:
      badFilter("a notch at " & $n.freqHz & " Hz is not below Nyquist (" & $nyq & " Hz)")

# ----------------------------------------------------------------- builders --

proc bandPass*(c: Corners): FilterSpec =
  ## A cos² trapezoid band-pass (or low-cut, or high-cut; see `Corners`).
  ## Raises `ValueError` for corners out of order or without slopes.
  checkCorners(c)
  FilterSpec(kind: rkTrapezoid, corners: c)

proc lowCut*(f1, f2: float64): FilterSpec =
  ## A cos² low-cut: 0 up to `f1`, 1 from `f2` (Hz).
  bandPass(Corners(f1: f1, f2: f2, f3: Inf, f4: Inf))

proc highCut*(f3, f4: float64): FilterSpec =
  ## A cos² high-cut: 1 up to `f3`, 0 from `f4` (Hz).
  bandPass(Corners(f1: 0, f2: 0, f3: f3, f4: f4))

proc butterBandPass*(lowHz, highHz: float64, slopeDbPerOctave = 24.0,
    highSlopeDbPerOctave = NaN): FilterSpec =
  ## A Butterworth band-pass with −6 dB corners `lowHz` and `highHz`, both
  ## sides at the effective `slopeDbPerOctave` unless `highSlopeDbPerOctave`
  ## is given. Raises `ValueError` for corners out of order or a slope that is
  ## not finite and positive.
  let hs = if highSlopeDbPerOctave.isNaN: slopeDbPerOctave else: highSlopeDbPerOctave
  checkButter(lowHz, slopeDbPerOctave, highHz, hs)
  FilterSpec(kind: rkButterworth, lowHz: lowHz, lowSlope: slopeDbPerOctave,
    highHz: highHz, highSlope: hs)

proc butterLowCut*(hz: float64, slopeDbPerOctave = 24.0): FilterSpec =
  ## A Butterworth low-cut with its −6 dB corner at `hz` (> 0).
  if not (hz > 0): badFilter("a low-cut corner must be > 0, got " & $hz)
  butterBandPass(hz, Inf, slopeDbPerOctave, slopeDbPerOctave)

proc butterHighCut*(hz: float64, slopeDbPerOctave = 24.0): FilterSpec =
  ## A Butterworth high-cut with its −6 dB corner at `hz` (finite).
  if not (hz < Inf): badFilter("a high-cut corner must be finite, got " & $hz)
  butterBandPass(0, hz, slopeDbPerOctave, slopeDbPerOctave)

proc withNotch*(spec: sink FilterSpec, freqHz, widthHz: float64): FilterSpec =
  ## `spec` with a notch at `freqHz` (module documentation). Raises
  ## `ValueError` unless both are finite and positive.
  let n = Notch(freqHz: freqHz, widthHz: widthHz)
  checkNotch(n)
  result = spec
  result.notches.add n

func withZeroDc*(spec: sink FilterSpec): FilterSpec =
  ## `spec` with bin 0 set to 0 when applied (module documentation).
  result = spec
  result.zeroDc = true

# ---------------------------------------------------------------- response --

func warp(f, dtMs: float64): float64 {.inline.} =
  ## The bilinear-transform frequency warp `tan(π·f·dt)`, `Inf` from Nyquist.
  let x = PI * f * dtMs / 1000
  if x >= PI / 2: Inf else: tan(x)

func ramp(x: float64): float64 {.inline.} =
  ## sin²(π/2·x) for x in 0 .. 1, clamped outside.
  if x <= 0: 0.0 elif x >= 1: 1.0 else: sin(PI / 2 * x) ^ 2

func response*(spec: FilterSpec, f, dtMs: float64): float64 {.raises: [].} =
  ## The gain of `spec` at `f` Hz (0 .. Nyquist) for samples `dtMs` apart
  ## (the Butterworth forms depend on it, the others do not), without the
  ## 0 Hz suppression of `withZeroDc`, which acts on bin 0 only.
  result = 1.0
  case spec.kind
  of rkTrapezoid:
    let c = spec.corners
    if c.hasLow: result *= ramp((f - c.f1) / (c.f2 - c.f1))
    if c.hasHigh: result *= 1 - ramp((f - c.f3) / (c.f4 - c.f3))
  of rkButterworth:
    let wf = warp(f, dtMs)
    if spec.lowHz > 0:
      result *= (if f <= 0: 0.0 else:
        1 / (1 + pow(warp(spec.lowHz, dtMs) / wf, spec.lowSlope / 6)))
    if spec.highHz < Inf:
      result *= (if wf == Inf: 0.0 else:
        1 / (1 + pow(wf / warp(spec.highHz, dtMs), spec.highSlope / 6)))
  for n in spec.notches:
    result *= ramp(abs(f - n.freqHz) / (n.widthHz / 2))

proc responseInto*[T: SampleType](spec: FilterSpec, dtMs: float64, nfft: int,
    h: var openArray[T]) {.raises: [].} =
  ## The gain of each bin `k = 0 .. nfft div 2` of an FFT of `nfft` samples at
  ## `dtMs`, at `k/(nfft·dt)` Hz, with bin 0 set to 0 for `withZeroDc`. `h`
  ## has `nfft div 2 + 1` values (an `AssertionDefect` otherwise).
  doAssert h.len == nfft div 2 + 1, "h needs nfft div 2 + 1 values"
  let df = 1000 / (float64(nfft) * dtMs)
  for k in 0..<h.len: h[k] = T(spec.response(float64(k) * df, dtMs))
  if spec.zeroDc and h.len > 0: h[0] = T(0)

func narrowestRampHz*(spec: FilterSpec): float64 {.raises: [].} =
  ## The width in Hz of the narrowest feature of `spec`: a cos² ramp, half a
  ## notch, or for a Butterworth edge `fc·min(1, 4/k)/2` (its transition
  ## band). `Inf` for a flat response.
  result = Inf
  case spec.kind
  of rkTrapezoid:
    let c = spec.corners
    if c.hasLow: result = min(result, c.f2 - c.f1)
    if c.hasHigh: result = min(result, c.f4 - c.f3)
  of rkButterworth:
    if spec.lowHz > 0:
      result = min(result, spec.lowHz * min(1.0, 24 / spec.lowSlope) / 2)
    if spec.highHz < Inf:
      result = min(result, spec.highHz * min(1.0, 24 / spec.highSlope) / 2)
  for n in spec.notches: result = min(result, n.widthHz / 2)

# ---------------------------------------------------------- impulse length --

const maxImpulseGrid = 1 shl 24

func energyLength(h: openArray[float64], centre: float64): int =
  ## The half-length holding `energyFraction` of the energy of the symmetric
  ## (circular) impulse `h` with `h[0]` replaced by `centre`.
  let n = h.len
  var total = centre * centre
  for j in 1..<n: total += h[j] * h[j]
  var e = centre * centre
  while e < energyFraction * total and result < n div 2:
    inc result
    e += h[result] * h[result] +
      (if n - result != result: h[n - result] * h[n - result] else: 0.0)

proc featureLength(spec: FilterSpec, dtMs: float64): int =
  ## The energy length of one response, the longer of `H` and `1 − H`.
  let feature = spec.narrowestRampHz
  if feature == Inf: return 0
  let fs = 1000 / dtMs
  var n = nextFastLen(max(64, int(ceil(16 * fs / feature))))
  while true:
    if n > maxImpulseGrid:
      badFilter("the filter's impulse response is longer than " &
        $(maxImpulseGrid div 8) & " samples at " & $dtMs &
        " ms (narrowest ramp " & $feature & " Hz)")
    let p = initFftPlan(float64, n)
    var ws = initWorkspace()
    ws.reserve(p.workspaceSize(batched = false))
    var re = newSeq[float64](p.bins)
    let im = newSeq[float64](p.bins)
    spec.responseInto(dtMs, n, re)
    var h = newSeq[float64](n)
    p.inverse(re, im, h, ws)
    # H and its complement 1 − H (impulse δ − h) share every sample but the
    # centre one.
    let L = max(energyLength(h, h[0]), energyLength(h, 1 - h[0]))
    if L <= n div 8: return L
    n = nextFastLen(2 * n) # the grid was too short to hold the impulse

func features(spec: FilterSpec): seq[FilterSpec] =
  ## Each edge and notch of `spec` as a response of its own.
  let flat = FilterSpec(kind: rkTrapezoid, corners: Corners(f1: 0, f2: 0, f3: Inf, f4: Inf))
  case spec.kind
  of rkTrapezoid:
    let c = spec.corners
    if c.hasLow:
      result.add FilterSpec(kind: rkTrapezoid, corners: Corners(f1: c.f1, f2: c.f2, f3: Inf, f4: Inf))
    if c.hasHigh:
      result.add FilterSpec(kind: rkTrapezoid, corners: Corners(f1: 0, f2: 0, f3: c.f3, f4: c.f4))
  of rkButterworth:
    if spec.lowHz > 0:
      result.add FilterSpec(kind: rkButterworth, lowHz: spec.lowHz,
        lowSlope: spec.lowSlope, highHz: Inf, highSlope: spec.highSlope)
    if spec.highHz < Inf:
      result.add FilterSpec(kind: rkButterworth, lowHz: 0, lowSlope: spec.lowSlope,
        highHz: spec.highHz, highSlope: spec.highSlope)
  for n in spec.notches:
    var f = flat
    f.notches.add n
    result.add f

proc impulseLength*(spec: FilterSpec, dtMs: float64): int =
  ## The number of samples `L` on each side of the centre of the zero-phase
  ## impulse response that hold `energyFraction` (99.9 %) of its energy
  ## (module documentation); 0 for a flat response. Each edge and notch is
  ## measured on its own (a low-cut alone, a high-cut alone, a notch alone),
  ## and `L` is the longest: in a band-pass, the energy of the wide part
  ## would hide the long tail of a narrow low ramp or notch. Each response is
  ## sampled on a grid fine enough to resolve its ramp and long enough to
  ## hold its impulse (up to 2²⁴ points), and inverted with a float64 FFT.
  ## Raises `ValueError` for a `spec` that `validate` rejects, or a response
  ## so narrow that its impulse is longer than 2²¹ samples.
  spec.validate(dtMs)
  for f in spec.features:
    result = max(result, f.featureLength(dtMs))
