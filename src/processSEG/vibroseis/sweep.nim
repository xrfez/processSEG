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
## **Laws** (`SweepLaw`), from `f1Hz` at `t = 0` to `f2Hz` at `T = lengthMs`
## (`f2Hz < f1Hz` is a down-sweep):
## - `slLinear`: `f = f1 + (f2 − f1)·t/T`; a flat amplitude spectrum;
## - `slDbPerOctave`: the time spent per Hz grows as `f^n`, with
##   `n = boostDb / 3.0103`, so the amplitude spectrum rises by `boostDb`
##   per octave (the stationary-phase relation `|P(f)|² ∝ dt/df`):
##   `t(f) = T·(f^m − f1^m)/(f2^m − f1^m)`, `m = n + 1`. Dwelling on the
##   high frequencies (`boostDb > 0`) makes up for the earth's absorption;
## - `slDbPerHz`: the same with `dt/df ∝ 10^(boostDb·f/10)`, an amplitude
##   spectrum rising by `boostDb` per Hz;
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
## at least one and below 2³¹; tapers that fit (`checkFits`).

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

proc badSweep(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func sweepLen*(spec: SweepSpec): int {.raises: [].} =
  ## The number of samples, `lengthMs / dtMs + 1` (a valid spec's length is
  ## a whole number of samples).
  int(round(spec.lengthMs / spec.dtMs)) + 1

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
    if spec.law == slDbPerHz and abs(spec.boostDb * (spec.f2Hz - spec.f1Hz)) > 600:
      badSweep("a dB/Hz boost of " & $spec.boostDb & " over " & $abs(spec.f2Hz - spec.f1Hz) &
        " Hz exceeds 600 dB")
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
  ## Hz. The spec is assumed valid.
  let T = spec.lengthMs
  let x = clamp(tMs, 0.0, T) / T
  let f1 = spec.f1Hz
  let f2 = spec.f2Hz
  case spec.law
  of slLinear: f1 + (f2 - f1) * x
  of slExponential: f1 * pow(f2 / f1, x)
  of slDbPerOctave:
    let m = spec.boostDb / octaveDb + 1
    if abs(m) < 1e-12: f1 * pow(f2 / f1, x)
    else: pow(pow(f1, m) + x * (pow(f2, m) - pow(f1, m)), 1 / m)
  of slDbPerHz:
    let k = spec.boostDb * ln(10.0) / 10 # dt/df ∝ exp(k·f)
    let e = expm1(k * (f2 - f1)) # exp(k(f2 − f1)) − 1
    if abs(k * (f2 - f1)) < 1e-12: f1 + (f2 - f1) * x
    else: f1 + ln1p(x * e) / k
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
  ## an invalid spec.
  result = newSeq[T](spec.sweepLen)
  spec.sweepInto(result)

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
