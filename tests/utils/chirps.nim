## Synthetic vibrator signals for the tests: a sweep law's phase (the
## library's `phaseAt`, so the linear and exponential laws are not a second
## formula), tapers of their own, harmonics at given levels and phases.
## A viewer builds a distorted sweep with `sweepHarmonics`; this stays for
## the cosine, delay, tilt and Nyquist fade the tests were written against.

import std/[math, random]
import processSEG/vibroseis/sweep

type
  ChirpLaw* = enum clLinear, clExponential
  Ramp* = enum rmCos2, rmLinear, rmBlackman, rmNone
  Chirp* = object
    ## A sweep from `f0` to `f1` Hz over `lengthMs`, starting at sample 0.
    law*: ChirpLaw
    f0*, f1*, lengthMs*: float64
    ramp*: Ramp = rmCos2
    startMs*, endMs*: float64 ## The tapers' lengths.
    tilt*: float64            ## The amplitude rises by this fraction over the sweep.
  Tone* = object
    ## One component: `amp·cos(order·φ(t) + phaseDeg)`.
    order*: float64 = 1
    amp*: float64 = 1
    phaseDeg*: float64

func freqAt*(c: Chirp, tMs: float64): float64 =
  ## The law's frequency at `tMs`, Hz.
  let u = tMs / c.lengthMs
  case c.law
  of clLinear: c.f0 + (c.f1 - c.f0) * u
  of clExponential: c.f0 * exp(ln(c.f1 / c.f0) * u)

func phaseAt*(c: Chirp, tMs: float64): float64 =
  ## 2π∫₀ᵗ f, radians: the library's `phaseAt` for this law, start phase 0.
  let law = if c.law == clLinear: slLinear else: slExponential
  SweepSpec(law: law, f1Hz: c.f0, f2Hz: c.f1, lengthMs: c.lengthMs).phaseAt(tMs)

func rampAt(r: Ramp, u: float64): float64 =
  case r
  of rmCos2: sin(PI / 2 * u) ^ 2
  of rmLinear: u
  of rmBlackman: 0.42 - 0.5 * cos(PI * u) + 0.08 * cos(2 * PI * u)
  of rmNone: 1.0

func weightAt*(c: Chirp, tMs: float64): float64 =
  ## The tapers' weight at `tMs` (0 outside the sweep), times the tilt.
  if tMs < 0 or tMs > c.lengthMs: return 0
  result = 1
  if c.ramp == rmNone: return 1 + c.tilt * (tMs / c.lengthMs - 0.5)
  if c.startMs > 0 and tMs < c.startMs: result = rampAt(c.ramp, tMs / c.startMs)
  if c.endMs > 0 and c.lengthMs - tMs < c.endMs:
    result = min(result, rampAt(c.ramp, (c.lengthMs - tMs) / c.endMs))
  result *= 1 + c.tilt * (tMs / c.lengthMs - 0.5)

proc chirp*(c: Chirp, dtMs: float64, n: int, tones: openArray[Tone] = [Tone()],
    delayMs = 0.0, noise = 0.0, seed = 1, nyquistFade = false): seq[float64] =
  ## `n` samples at `dtMs` of `Σ w(t − τ)·amp·cos(order·φ(t − τ) + phase)`
  ## plus white noise of standard deviation `noise`. With `nyquistFade`, a
  ## component fades out (cos²) between 0.85 and 0.95 of Nyquist, as an
  ## anti-alias filter would take it, so none folds back.
  result = newSeq[float64](n)
  var rng = initRand(seed)
  let nyq = 500 / dtMs
  for j in 0 ..< n:
    let t = float64(j) * dtMs - delayMs
    let w = c.weightAt(t)
    var v = 0.0
    if w > 0:
      let p = c.phaseAt(t)
      let f = c.freqAt(t)
      for tone in tones:
        var a = tone.amp
        if nyquistFade:
          let u = (tone.order * f / nyq - 0.85) / 0.1
          if u >= 1: continue
          if u > 0: a *= cos(PI / 2 * u) ^ 2
        v += a * w * cos(tone.order * p + degToRad(tone.phaseDeg))
    if noise > 0: v += rng.gauss(0, noise)
    result[j] = v
