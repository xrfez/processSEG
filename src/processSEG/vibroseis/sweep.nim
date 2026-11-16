## Vibroseis sweeps (`SweepSpec`): the pilot signal a vibrator is driven
## with, generated analytically at the sample interval it is needed at.
##
## ```nim
## let spec = linearSweep(f1Hz = 8, f2Hz = 80, lengthMs = 8000, dtMs = 2,
##   taper = initSweepTaper(250, 250, tpCos2))
## let x = spec.sweepSamples(float32)     # 4001 samples, t = 0 .. 8000 ms
## doAssert x.len == 4001 and x[0] == 0
## doAssert abs(spec.frequencyAt(4000) - 44) < 1e-9
## ```
##
## **Signal.** Sample `j`, at `t = j·dtMs`, is
## `amplitude · taper(t) · sin(φ₀ + 2π·∫₀ᵗ f(s) ds)`, for `j` in
## `0 .. lengthMs / dtMs` (both ends included: `sweepLen` samples). The
## phase is the integral of the instantaneous frequency `f(t)`, in float64:
## in closed form for the linear and table laws (exact to rounding), by
## 4-point Gauss–Legendre per sample interval for the others (error below
## 1e-9 rad over a sweep). `φ₀` is `startPhaseDeg`; at 0 a sweep starts at
## zero, as a taper wants.
##
## **Harmonics** (`sweepHarmonics`). The same sweep, plus terms
## `amp · sin(order·φ + phase)`, where `φ` is the fundamental's phase
## (`φ₀` included, the phase `sweepSamples` is built from) and `phase` is
## that term's own. Order 1 with amplitude 1 and phase 0 is the fundamental;
## an empty list is that term alone, so it matches `sweepSamples`. A phase
## error of the ground force against its pilot is the fundamental's phase.
## Where `order·f(t)` is at or above Nyquist the term is left out of that
## sample: cut, not folded back, and not a reason to refuse the call (an
## upsweep's high harmonics are real at the start and would alias later).
## Order 1 is never cut: `validate` keeps the fundamental within Nyquist.
## The cut is a hard zero, so the waveform can step where a harmonic leaves
## the band.
##
## **Laws** (`SweepLaw`), from `f1Hz` at `t = 0` to `f2Hz` at `T = lengthMs`
## (`f2Hz < f1Hz` is a down-sweep):
## - `slLinear`: `f = f1 + (f2 − f1)·t/T`; a flat amplitude spectrum;
## - `slDbPerOctave`: the time spent per Hz grows as `f^n`, with
##   `n = boostDb / 3.0103`, so the amplitude spectrum rises by `boostDb`
##   per octave (the stationary-phase relation `|P(f)|² ∝ dt/df`):
##   `t(f) = T·(f^m − f1^m)/(f2^m − f1^m)`, `m = n + 1`. Dwelling on the
##   high frequencies (`boostDb > 0`) makes up for the earth's absorption;
## - `slDbPerHz`: the same with `dt/df ∝ 10^(boostDb·f/10)`, an amplitude
##   spectrum rising by `boostDb` per Hz. A boost may add up to 300 dB over
##   the band per octave (`|boostDb| · log₂(f2/f1)`) and 150 dB per Hz
##   (`|boostDb| · |f2 − f1|`): steeper, the far end of the law is passed in
##   less time than a float64 resolves;
## - `slExponential`: `f = f1·(f2/f1)^(t/T)`, an equal time per octave
##   (−3 dB/octave: `slDbPerOctave` with `boostDb = −3.0103`);
## - `slTable`: `f(t)` linear between the caller's `(tMs, fHz)` points, which
##   start at 0 and end at `lengthMs` (`tableSweep`). Reading a vendor's
##   table file is the caller's job.
##
## **Validation** (`validate`): frequencies above 0 and at most
## `maxNyquistFraction` (default 0.8) of the Nyquist frequency, so the sweep
## stays below the anti-alias filter of the data it is correlated with, and
## never above Nyquist; a length of a whole number of samples (within 1e-6),
## at least one and below 2³¹; a boost within its bound; tapers that fit
## (`checkFits`).

import std/math
import parseSEG
import taper

type
  SweepLaw* = enum
    ## How the instantaneous frequency runs from `f1Hz` to `f2Hz`.
    slLinear      ## linear in time
    slDbPerOctave ## amplitude spectrum boosted by `boostDb` per octave
    slDbPerHz     ## amplitude spectrum boosted by `boostDb` per Hz
    slExponential ## exponential in time (equal time per octave)
    slTable       ## piecewise linear through `table`

  SweepPoint* = object
    ## A point of a table sweep: the frequency `fHz` at `tMs`.
    tMs*: float64
    fHz*: float64

  SweepSpec* = object
    ## A sweep; make one with `linearSweep`, `dbPerOctaveSweep`,
    ## `dbPerHzSweep`, `exponentialSweep` or `tableSweep`, which validate it.
    law*: SweepLaw
    f1Hz*: float64 ## Frequency at the start (a table's first point).
    f2Hz*: float64 ## Frequency at the end (a table's last point).
    lengthMs*: float64 ## Sweep length `T`.
    dtMs*: float64 ## Sample interval.
    boostDb*: float64 ## dB per octave or per Hz (`slDbPerOctave`, `slDbPerHz`).
    table*: seq[SweepPoint] ## `slTable` only.
    startPhaseDeg*: float64 ## `φ₀`, in degrees.
    amplitude*: float64 = 1 ## Peak amplitude.
    taper*: SweepTaper ## Start and end ramps.
    maxNyquistFraction*: float64 = 0.8 ## Highest frequency allowed, as a fraction of Nyquist (≤ 1).

  SweepHarmonic* = object
    ## One term of a distorted sweep: `amp · sin(order·φ + phaseDeg)`, with
    ## `φ` the fundamental's phase. Order 1, amplitude 1, phase 0 is the
    ## fundamental.
    order*: float64 = 1 ## Harmonic order (1: the fundamental). Must be > 0.
    amp*: float64 = 1 ## Amplitude relative to `SweepSpec.amplitude`.
    phaseDeg*: float64 ## Added phase, degrees.

const
  maxDbPerOctaveTotal = 300.0 # the largest |boostDb| · octaves of a dB/octave sweep
  maxDbPerHzTotal = 150.0     # the largest |boostDb| · bandwidth of a dB/Hz sweep

proc badSweep(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func sweepLen*(spec: SweepSpec): int {.raises: [].} =
  ## The number of samples, `lengthMs / dtMs + 1` (a valid spec's length is
  ## a whole number of samples); 0 for a spec `validate` rejects for its
  ## length or interval (an interval that is not finite and positive, a
  ## length under one interval, not finite, or of 2³¹ samples and more).
  if not (spec.dtMs > 0 and spec.dtMs < Inf): return 0
  let steps = spec.lengthMs / spec.dtMs
  if steps >= 1 and steps <= float64(high(int32)): int(round(steps)) + 1 else: 0

func nyquistOf(dtMs: float64): float64 {.inline.} = 500 / dtMs

proc checkFreq(f, limit, nyq: float64, what: string) =
  if not (f > 0 and f < Inf):
    badSweep(what & " must be > 0 Hz, got " & $f)
  if f > nyq:
    badSweep(what & " = " & $f & " Hz is above Nyquist (" & $nyq & " Hz): it would alias")
  if f > limit * (1 + 1e-12):
    badSweep(what & " = " & $f & " Hz is above the anti-alias limit " & $limit &
      " Hz (maxNyquistFraction · Nyquist); raise maxNyquistFraction to allow it")

proc validate*(spec: SweepSpec) =
  ## Checks a sweep (module documentation). Raises `ValueError`.
  if not (spec.dtMs > 0 and spec.dtMs < Inf):
    badSweep("dtMs must be finite and positive, got " & $spec.dtMs)
  if not (spec.lengthMs >= spec.dtMs and spec.lengthMs < Inf):
    badSweep("lengthMs must be at least one sample interval, got " & $spec.lengthMs)
  let steps = spec.lengthMs / spec.dtMs
  if steps > float64(high(int32)):
    badSweep("a sweep of " & $spec.lengthMs & " ms at " & $spec.dtMs &
      " ms is over 2³¹ samples")
  if abs(steps - round(steps)) > 1e-6 * max(1.0, steps):
    badSweep("lengthMs = " & $spec.lengthMs & " must be a whole number of samples of " &
      $spec.dtMs & " ms")
  if not (spec.maxNyquistFraction > 0 and spec.maxNyquistFraction <= 1):
    badSweep("maxNyquistFraction must be in (0, 1], got " & $spec.maxNyquistFraction)
  if not (spec.amplitude.classify in {fcNormal, fcSubnormal, fcZero}):
    badSweep("the amplitude must be finite, got " & $spec.amplitude)
  if not (spec.startPhaseDeg.classify in {fcNormal, fcSubnormal, fcZero, fcNegZero}):
    badSweep("startPhaseDeg must be finite, got " & $spec.startPhaseDeg)
  let nyq = nyquistOf(spec.dtMs)
  let limit = spec.maxNyquistFraction * nyq
  checkFreq(spec.f1Hz, limit, nyq, "f1Hz")
  checkFreq(spec.f2Hz, limit, nyq, "f2Hz")
  case spec.law
  of slDbPerOctave, slDbPerHz:
    if not (spec.boostDb.classify in {fcNormal, fcSubnormal, fcZero, fcNegZero}):
      badSweep("boostDb must be finite, got " & $spec.boostDb)
    # What the boost adds up to over the band. A dB/Hz law is an exponential
    # of it: past 156 dB (e⁻³⁶) its far end is below a float64's resolution
    # and the end frequency came out infinite. A dB/octave law is a power
    # of f2 / f1 that overflows past 3000 dB; 300 dB is no sweep either.
    let (total, limit) =
      if spec.law == slDbPerHz: (abs(spec.boostDb * (spec.f2Hz - spec.f1Hz)), maxDbPerHzTotal)
      else: (abs(spec.boostDb * log2(spec.f2Hz / spec.f1Hz)), maxDbPerOctaveTotal)
    if total > limit:
      badSweep("a boost of " & $spec.boostDb & (if spec.law == slDbPerHz: " dB/Hz" else:
        " dB/octave") & " from " & $spec.f1Hz & " to " & $spec.f2Hz & " Hz adds up to " &
        $(round(10 * total) / 10) & " dB, over the limit of " & $limit)
  of slTable:
    let t = spec.table
    if t.len < 2: badSweep("a table sweep needs at least two points")
    if t[0].tMs != 0: badSweep("a sweep table must start at 0 ms, got " & $t[0].tMs)
    if abs(t[^1].tMs - spec.lengthMs) > 1e-9 * spec.lengthMs:
      badSweep("a sweep table must end at lengthMs = " & $spec.lengthMs & ", got " & $t[^1].tMs)
    for i, p in t:
      checkFreq(p.fHz, limit, nyq, "the table frequency at " & $p.tMs & " ms")
      if i > 0 and not (p.tMs > t[i - 1].tMs):
        badSweep("sweep table times must increase: " & $t[i - 1].tMs & " then " & $p.tMs)
    if spec.f1Hz != t[0].fHz or spec.f2Hz != t[^1].fHz:
      badSweep("a table sweep's f1Hz and f2Hz must be its first and last frequencies")
  of slLinear, slExponential: discard
  spec.taper.checkFits(spec.lengthMs)

proc linearSweep*(f1Hz, f2Hz, lengthMs, dtMs: float64, taper = noTaper(),
    startPhaseDeg = 0.0, amplitude = 1.0, maxNyquistFraction = 0.8): SweepSpec =
  ## A linear sweep from `f1Hz` to `f2Hz` over `lengthMs`, sampled every
  ## `dtMs`. Raises `ValueError` as `validate`.
  result = SweepSpec(law: slLinear, f1Hz: f1Hz, f2Hz: f2Hz, lengthMs: lengthMs,
    dtMs: dtMs, startPhaseDeg: startPhaseDeg, amplitude: amplitude, taper: taper,
    maxNyquistFraction: maxNyquistFraction)
  result.validate

proc dbPerOctaveSweep*(f1Hz, f2Hz, lengthMs, dtMs, boostDb: float64, taper = noTaper(),
    startPhaseDeg = 0.0, amplitude = 1.0, maxNyquistFraction = 0.8): SweepSpec =
  ## A nonlinear sweep whose amplitude spectrum rises by `boostDb` per
  ## octave (negative: falls). Raises `ValueError` as `validate`.
  result = SweepSpec(law: slDbPerOctave, f1Hz: f1Hz, f2Hz: f2Hz, lengthMs: lengthMs,
    dtMs: dtMs, boostDb: boostDb, startPhaseDeg: startPhaseDeg, amplitude: amplitude,
    taper: taper, maxNyquistFraction: maxNyquistFraction)
  result.validate

proc dbPerHzSweep*(f1Hz, f2Hz, lengthMs, dtMs, boostDb: float64, taper = noTaper(),
    startPhaseDeg = 0.0, amplitude = 1.0, maxNyquistFraction = 0.8): SweepSpec =
  ## A nonlinear sweep whose amplitude spectrum rises by `boostDb` per Hz.
  ## Raises `ValueError` as `validate`.
  result = SweepSpec(law: slDbPerHz, f1Hz: f1Hz, f2Hz: f2Hz, lengthMs: lengthMs,
    dtMs: dtMs, boostDb: boostDb, startPhaseDeg: startPhaseDeg, amplitude: amplitude,
    taper: taper, maxNyquistFraction: maxNyquistFraction)
  result.validate

proc exponentialSweep*(f1Hz, f2Hz, lengthMs, dtMs: float64, taper = noTaper(),
    startPhaseDeg = 0.0, amplitude = 1.0, maxNyquistFraction = 0.8): SweepSpec =
  ## An exponential (logarithmic) sweep, `f = f1·(f2/f1)^(t/T)`. Raises
  ## `ValueError` as `validate`.
  result = SweepSpec(law: slExponential, f1Hz: f1Hz, f2Hz: f2Hz, lengthMs: lengthMs,
    dtMs: dtMs, startPhaseDeg: startPhaseDeg, amplitude: amplitude, taper: taper,
    maxNyquistFraction: maxNyquistFraction)
  result.validate

proc tableSweep*(points: openArray[SweepPoint], dtMs: float64, taper = noTaper(),
    startPhaseDeg = 0.0, amplitude = 1.0, maxNyquistFraction = 0.8): SweepSpec =
  ## A sweep through `points` (from 0 ms to its length, times increasing),
  ## linear in between. Raises `ValueError` as `validate` (also for fewer
  ## than two points).
  if points.len < 2: badSweep("a table sweep needs at least two points")
  result = SweepSpec(law: slTable, f1Hz: points[0].fHz, f2Hz: points[^1].fHz,
    lengthMs: points[^1].tMs, dtMs: dtMs, table: @points, startPhaseDeg: startPhaseDeg,
    amplitude: amplitude, taper: taper, maxNyquistFraction: maxNyquistFraction)
  result.validate

# ------------------------------------------------------------------- laws --

const octaveDb = 20 * log10(2.0) / 2 # 3.0103: dB per octave of dt/df ∝ f

func expm1(x: float64): float64 =
  ## `exp(x) − 1` without cancellation near 0 (Kahan's trick).
  let u = exp(x)
  if u == 1: x
  elif u - 1 == -1: -1.0
  else: (u - 1) * x / ln(u)

func ln1p(x: float64): float64 =
  ## `ln(1 + x)` without cancellation near 0 (Kahan's trick).
  let u = 1 + x
  if u == 1: x else: ln(u) * x / (u - 1)

func frequencyAt*(spec: SweepSpec, tMs: float64): float64 {.raises: [].} =
  ## The instantaneous frequency at `tMs` (clamped to `0 .. lengthMs`), in
  ## Hz: `f1Hz` at 0 and `f2Hz` at `lengthMs`, exactly. The spec is assumed
  ## valid.
  let T = spec.lengthMs
  let x = clamp(tMs, 0.0, T) / T
  let f1 = spec.f1Hz
  let f2 = spec.f2Hz
  # The ends as given: a steep law's formula loses its last step to
  # rounding (1 + (r − 1) with r below 1e-16), which read as an infinite
  # end frequency.
  if x <= 0: return f1
  if x >= 1: return f2
  # The laws from f1 to f2 run one way: rounding does not take them past
  # either (a steep one came out 1e-14 Hz above f2 just before its end).
  template within(f: float64): float64 = clamp(f, min(f1, f2), max(f1, f2))
  case spec.law
  of slLinear: within(f1 + (f2 - f1) * x)
  of slExponential: within(f1 * pow(f2 / f1, x))
  of slDbPerOctave:
    # f^m linear in time, written in f2 / f1 so that no power of a
    # frequency in Hz can overflow.
    let m = spec.boostDb / octaveDb + 1
    if abs(m) < 1e-12: within(f1 * pow(f2 / f1, x))
    else: within(f1 * pow(1 + x * expm1(m * ln(f2 / f1)), 1 / m))
  of slDbPerHz:
    let k = spec.boostDb * ln(10.0) / 10 # dt/df ∝ exp(k·f)
    let e = expm1(k * (f2 - f1)) # exp(k(f2 − f1)) − 1
    if abs(k * (f2 - f1)) < 1e-12: within(f1 + (f2 - f1) * x)
    else: within(f1 + ln1p(x * e) / k)
  of slTable:
    let t = clamp(tMs, 0.0, T)
    var i = 1
    while i < spec.table.len - 1 and spec.table[i].tMs < t: inc i
    let a = spec.table[i - 1]
    let b = spec.table[i]
    a.fHz + (b.fHz - a.fHz) * (t - a.tMs) / (b.tMs - a.tMs)

const
  glNodes = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
    0.8611363115940526]
  glWeights = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
    0.3478548451374538]

proc phasesInto(spec: SweepSpec, phase: var openArray[float64]) =
  ## `2π·∫₀ᵗ f` (radians, without φ₀) at each sample time.
  let dt = spec.dtMs / 1000 # s
  let T = spec.lengthMs / 1000
  case spec.law
  of slLinear:
    let rate = (spec.f2Hz - spec.f1Hz) / T
    for j in 0 ..< phase.len:
      let t = float64(j) * dt
      phase[j] = 2 * PI * (spec.f1Hz * t + 0.5 * rate * t * t)
  of slTable:
    # Piecewise linear f: the phase is piecewise quadratic, exact per piece.
    var acc = 0.0 # cycles at the start of piece i
    var i = 1
    let tab = spec.table
    for j in 0 ..< phase.len:
      let tMs = float64(j) * spec.dtMs
      while i < tab.len - 1 and tab[i].tMs <= tMs:
        let a = tab[i - 1]
        let b = tab[i]
        acc += 0.5 * (a.fHz + b.fHz) * (b.tMs - a.tMs) / 1000
        inc i
      let a = tab[i - 1]
      let b = tab[i]
      let s = (min(tMs, b.tMs) - a.tMs) / 1000
      let slope = (b.fHz - a.fHz) / ((b.tMs - a.tMs) / 1000)
      phase[j] = 2 * PI * (acc + a.fHz * s + 0.5 * slope * s * s)
  else:
    var acc = 0.0
    if phase.len > 0: phase[0] = 0
    for j in 1 ..< phase.len:
      let a = float64(j - 1) * spec.dtMs
      let h = 0.5 * spec.dtMs
      var s = 0.0
      for q in 0 ..< 4:
        s += glWeights[q] * spec.frequencyAt(a + h + h * glNodes[q])
      acc += s * h / 1000
      phase[j] = 2 * PI * acc

proc sweepInto*[T: SampleType](spec: SweepSpec, x: var openArray[T]) =
  ## Fills `x` (`spec.sweepLen` samples) with the sweep. Raises `ValueError`
  ## for an invalid spec or another length.
  spec.validate
  if x.len != spec.sweepLen:
    badSweep("the sweep has " & $spec.sweepLen & " samples, the buffer " & $x.len)
  var phase = newSeq[float64](x.len)
  spec.phasesInto(phase)
  let p0 = degToRad(spec.startPhaseDeg)
  for j in 0 ..< x.len:
    x[j] = T(spec.amplitude * sin(p0 + phase[j]))
  x.applyTaper(spec.dtMs, spec.taper)

proc sweepSamples*(spec: SweepSpec, T: typedesc[SampleType] = float64): seq[T] =
  ## The sweep's samples (`spec.sweepLen` of them). Raises `ValueError` for
  ## an invalid spec, before anything is allocated.
  spec.validate
  result = newSeq[T](spec.sweepLen)
  spec.sweepInto(result)

proc checkHarmonic(h: SweepHarmonic) =
  if not (h.order > 0 and h.order < Inf):
    badSweep("a harmonic order must be finite and > 0, got " & $h.order)
  if h.amp.classify notin {fcNormal, fcSubnormal, fcZero, fcNegZero}:
    badSweep("a harmonic amplitude must be finite, got " & $h.amp)
  if h.phaseDeg.classify notin {fcNormal, fcSubnormal, fcZero, fcNegZero}:
    badSweep("a harmonic phase must be finite, got " & $h.phaseDeg)

proc taperPrefix[T: SampleType](x: var openArray[T], dtMs, sweepMs: float64, t: SweepTaper) =
  ## The start and end ramps of `t` on a prefix of a sweep of `sweepMs`, the
  ## same multiplies as `applyTaper` when the buffer is the whole sweep.
  let s = if t.startMs.isNaN: 0.0 else: t.startMs
  let e = if t.endMs.isNaN: 0.0 else: t.endMs
  if s > 0:
    var j = 0
    while j < x.len and float64(j) * dtMs < s:
      x[j] = T(float64(x[j]) * taperWeight(t.shape, float64(j) * dtMs / s, t.param))
      inc j
  if e > 0:
    var j = x.len - 1
    while j >= 0 and sweepMs - float64(j) * dtMs < e:
      x[j] = T(float64(x[j]) * taperWeight(t.shape, (sweepMs - float64(j) * dtMs) / e, t.param))
      dec j

proc sweepHarmonicsInto*[T: SampleType](spec: SweepSpec, x: var openArray[T],
    harmonics: openArray[SweepHarmonic] = []) =
  ## Fills `x` with the sweep plus `harmonics` (module documentation). Sample
  ## `j` is at `j·dtMs`; samples past `lengthMs` are 0. An empty `harmonics`
  ## on a buffer of `sweepLen` samples is `sweepInto`. Raises `ValueError`
  ## for an invalid spec, an empty buffer, or a harmonic whose order is not
  ## finite and positive or whose amplitude or phase is not finite.
  spec.validate
  if x.len < 1:
    badSweep("a sweep buffer needs at least one sample, got " & $x.len)
  for h in harmonics:
    checkHarmonic(h)
  if harmonics.len == 0 and x.len == spec.sweepLen:
    spec.sweepInto(x)
    return
  let nSweep = spec.sweepLen
  let n = min(x.len, nSweep)
  var phase = newSeq[float64](n)
  spec.phasesInto(phase)
  let p0 = degToRad(spec.startPhaseDeg)
  let nyq = 500.0 / spec.dtMs
  for j in 0 ..< x.len:
    if j >= n:
      x[j] = T(0)
    else:
      let f = spec.frequencyAt(float64(j) * spec.dtMs)
      let p = p0 + phase[j]
      var s = 0.0
      if harmonics.len == 0:
        s = sin(p)
      else:
        for h in harmonics:
          if h.order == 1 or h.order * f < nyq:
            s += h.amp * sin(h.order * p + degToRad(h.phaseDeg))
      x[j] = T(spec.amplitude * s)
  if x.len == nSweep:
    x.applyTaper(spec.dtMs, spec.taper)
  elif x.len > nSweep:
    var head = newSeq[T](nSweep)
    for j in 0 ..< nSweep:
      head[j] = x[j]
    head.applyTaper(spec.dtMs, spec.taper)
    for j in 0 ..< nSweep:
      x[j] = head[j]
  else:
    taperPrefix(x, spec.dtMs, spec.lengthMs, spec.taper)

proc sweepHarmonics*(spec: SweepSpec, T: typedesc[SampleType] = float64,
    harmonics: openArray[SweepHarmonic] = [], samples = 0): seq[T] =
  ## The sweep plus `harmonics`. `samples` 0 is `sweepLen`; a longer buffer
  ## is the sweep followed by zeros (a record with listen time). Raises
  ## `ValueError` as `sweepHarmonicsInto`, before anything is allocated.
  spec.validate
  for h in harmonics:
    checkHarmonic(h)
  let n = if samples == 0: spec.sweepLen else: samples
  if n < 1:
    badSweep("samples must be ≥ 1, got " & $n)
  result = newSeq[T](n)
  spec.sweepHarmonicsInto(result, harmonics)

func phaseAt*(spec: SweepSpec, tMs: float64): float64 {.raises: [].} =
  ## The sweep's phase at `tMs` in radians, `φ₀ + 2π·∫₀ᵗ f`, for the laws
  ## with a closed form (linear, exponential, table; others: NaN). For
  ## tests and plots.
  let t = clamp(tMs, 0.0, spec.lengthMs) / 1000
  let T = spec.lengthMs / 1000
  let p0 = degToRad(spec.startPhaseDeg)
  case spec.law
  of slLinear:
    p0 + 2 * PI * (spec.f1Hz * t + 0.5 * (spec.f2Hz - spec.f1Hz) / T * t * t)
  of slExponential:
    let r = ln(spec.f2Hz / spec.f1Hz)
    if abs(r) < 1e-12: p0 + 2 * PI * spec.f1Hz * t
    else: p0 + 2 * PI * spec.f1Hz * T / r * expm1(r * t / T)
  of slTable:
    var acc = 0.0
    let tab = spec.table
    let tm = t * 1000
    for i in 1 ..< tab.len:
      let a = tab[i - 1]
      let b = tab[i]
      if tm <= b.tMs:
        let s = (tm - a.tMs) / 1000
        let slope = (b.fHz - a.fHz) / ((b.tMs - a.tMs) / 1000)
        return p0 + 2 * PI * (acc + a.fHz * s + 0.5 * slope * s * s)
      acc += 0.5 * (a.fHz + b.fHz) * (b.tMs - a.tMs) / 1000
    p0 + 2 * PI * acc
  else: NaN

func segySweepType*(spec: SweepSpec): int16 {.raises: [].} =
  ## The SEG-Y sweep type code (trace bytes 133-134, binary 3239-3240):
  ## 1 linear, 3 exponential, 4 other (2, parabolic, is not generated).
  case spec.law
  of slLinear: 1
  of slExponential: 3
  of slDbPerOctave:
    if abs(spec.boostDb + octaveDb) < 1e-9: 3 else: 4
  else: 4
