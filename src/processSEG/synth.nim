## Synthetic traces and gathers: a Ricker wavelet, a sinusoid, seeded noise,
## and a linear or hyperbolic event added to a gather. Nothing here opens a
## file.
##
## ```nim
## let offsets = [-100.0, -50.0, 0.0, 50.0, 100.0]
## var shot = zeros[float64](offsets.len, 501)          # 1 s at 2 ms
## shot.addLinear(offsets, dtMs = 2, t0Ms = 8, velocity = 1800, freqHz = 45)
## shot.addHyperbola(offsets, dtMs = 2, t0Ms = 280, velocity = 2000, freqHz = 30)
## shot.addNoise(0.03, seed = 7)
## doAssert ricker(0, 30) == 1
## ```
##
## **Time.** Column `j` of a row is at `traceT0Ms + (firstSample + j)·dtMs`,
## the same axis as `ChunkMeta.sampleTimeMs` (`traceT0Ms` 0 and `firstSample`
## 0 unless the gather is a window). Travel times are in ms, velocities in
## distance units per second, frequencies in Hz, as the rest of the library.
##
## **Events.** `addLinear` and `addHyperbola` add a Ricker wavelet
## (`addRicker`) on every row at `linearMs` or `hyperbolicMs` of that row's
## offset, plus `shiftsMs` when given. A direct wave, a refraction and ground
## roll are linear events at different velocities and frequencies; a
## reflection is a hyperbola. The wavelet is 1 at its centre and is not added
## more than 3 periods away, so a short gather is not filled with its tails.
## Offsets may be signed: both formulas use `|offset|`. A NaN offset or time
## adds nothing on that row.
##
## **Noise** (`addNoise`) is white Gaussian noise of standard deviation
## `sigma`, drawn with `std/random` (`initRand(seed)` and `gauss`). The same
## seed, length and `sigma` give the same samples on every run on one
## machine, with one Nim version. That is the guarantee. It is not a file
## format: `std/random` can change between Nim versions, and `gauss` uses
## `libm` (`log`, `sqrt`, `sin`), so the last bits can differ across
## platforms and C libraries. A comparison of written samples is valid for
## the same machine and the same Nim version only. `float32` stores each
## sample rounded to float32.

import std/[math, random]
import parseSEG
import core/rows

proc badSynth(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func finite(x: float64): bool {.inline.} =
  x.classify in {fcNormal, fcSubnormal, fcZero, fcNegZero}

# ----------------------------------------------------------------- wavelet --

func ricker*(tauMs, freqHz: float64): float64 {.raises: [].} =
  ## The Ricker wavelet of peak frequency `freqHz`, `tauMs` from its centre.
  ## 1 at the centre, 0 at `±1000 / (π·freqHz·√2)` ms, and odd side lobes
  ## beyond. The amplitude spectrum peaks at `freqHz`.
  let a = (PI * freqHz * tauMs / 1000) ^ 2
  (1 - 2 * a) * exp(-a)

proc addSinusoid*[T: SampleType](x: var openArray[T], dtMs, freqHz: float64,
    amp = 1.0, phaseRad = 0.0, t0Ms = 0.0) =
  ## Adds `amp·sin(2π·f·t + phaseRad)`, `t` in seconds from sample 0 at
  ## `t0Ms`. Raises `ValueError` when `dtMs` is not finite and positive or
  ## `freqHz`, `amp` or `phaseRad` is not finite.
  if not (dtMs > 0 and dtMs < Inf):
    badSynth("dtMs must be finite and positive, got " & $dtMs)
  if not (finite(freqHz) and finite(amp) and finite(phaseRad) and finite(t0Ms)):
    badSynth("freqHz, amp, phaseRad and t0Ms must be finite")
  for j in 0 ..< x.len:
    let t = (t0Ms + j.float64 * dtMs) / 1000
    x[j] += T(amp * sin(2 * PI * freqHz * t + phaseRad))

proc addRicker*[T: SampleType](x: var openArray[T], dtMs, centreMs,
    freqHz: float64, amp = 1.0, t0Ms = 0.0, firstSample = 0) =
  ## Adds a Ricker wavelet centred at `centreMs` (absolute time, like
  ## `t0Ms`). Column `j` is at `t0Ms + (firstSample + j)·dtMs`. Samples
  ## further than 3 periods from the centre are left alone. Raises
  ## `ValueError` when `dtMs` or `freqHz` is not finite and positive, or
  ## `amp` or `t0Ms` is not finite, or `firstSample < 0`.
  if not (dtMs > 0 and dtMs < Inf):
    badSynth("dtMs must be finite and positive, got " & $dtMs)
  if not (freqHz > 0 and freqHz < Inf):
    badSynth("freqHz must be finite and positive, got " & $freqHz)
  if not (finite(amp) and finite(t0Ms) and finite(centreMs)):
    badSynth("amp, t0Ms and centreMs must be finite")
  if firstSample < 0:
    badSynth("firstSample must be ≥ 0, got " & $firstSample)
  let halfMs = 3000.0 / freqHz
  for j in 0 ..< x.len:
    let tau = t0Ms + float64(firstSample + j) * dtMs - centreMs
    if abs(tau) <= halfMs:
      x[j] += T(amp * ricker(tau, freqHz))

# -------------------------------------------------------------------- noise --

func rowSeed(seed: int64, row: int): int64 {.inline.} =
  ## `seed + row`, wrapping in int64, so a row can be drawn on its own.
  cast[int64](uint64(seed) + uint64(row))

proc addNoise*[T: SampleType](x: var openArray[T], sigma: float64, seed: int64) =
  ## Adds white Gaussian noise of standard deviation `sigma` (`std/random`,
  ## `initRand(seed)` then `gauss`; 0 leaves `x` unchanged). The same seed
  ## gives the same samples on one machine and one Nim version; see the
  ## module documentation for what is not guaranteed. Raises `ValueError`
  ## when `sigma` is not finite and ≥ 0.
  if not (sigma >= 0 and sigma < Inf):
    badSynth("sigma must be finite and ≥ 0, got " & $sigma)
  if sigma == 0 or x.len == 0: return
  var rng = initRand(seed)
  for j in 0 ..< x.len:
    x[j] += T(rng.gauss(0, sigma))

proc addNoise*[T: SampleType](samples: var Tensor[T], sigma: float64, seed: int64) =
  ## Adds `addNoise` to every row. Row `r` is seeded with `seed + r`
  ## (wrapping in int64), so the rows differ and one row can be regenerated
  ## without the others. Raises `ValueError` as `addNoise`, and as `rowBlock`
  ## when `samples` is not a rank-2 C-contiguous tensor.
  let b = rowBlock(samples)
  for r in 0 ..< b.len:
    b.row(r).addNoise(sigma, rowSeed(seed, r))

# ------------------------------------------------------------- travel times --

func linearMs*(t0Ms, velocity, offset: float64): float64 {.inline, raises: [].} =
  ## Linear moveout `t0 + |offset| / velocity`, in ms (a direct wave, a
  ## refraction, ground roll). The same value as `lmoTime`.
  t0Ms + abs(offset) / velocity * 1000

func hyperbolicMs*(t0Ms, velocity, offset: float64): float64 {.inline,
    raises: [].} =
  ## Hyperbolic moveout `sqrt(t0² + (|offset| / velocity)²)`, in ms, with the
  ## offset term in ms (a reflection). The same value as `nmoTime`.
  sqrt(t0Ms ^ 2 + (abs(offset) / velocity * 1000) ^ 2)

proc checkEvent[T: SampleType](samples: Tensor[T], offsets: openArray[float64],
    dtMs, t0Ms, velocity, freqHz, amp: float64, shiftsMs, traceT0Ms: openArray[float64],
    firstSample: int, liveCols: openArray[int]): RowBlock[T] =
  if not (dtMs > 0 and dtMs < Inf):
    badSynth("dtMs must be finite and positive, got " & $dtMs)
  if not (freqHz > 0 and freqHz < Inf):
    badSynth("freqHz must be finite and positive, got " & $freqHz)
  if not (velocity > 0 and velocity < Inf):
    badSynth("velocity must be finite and positive, got " & $velocity)
  if not finite(amp):
    badSynth("amp must be finite, got " & $amp)
  if not finite(t0Ms):
    badSynth("t0Ms must be finite, got " & $t0Ms)
  if firstSample < 0:
    badSynth("firstSample must be ≥ 0, got " & $firstSample)
  result = rowBlock(samples)
  if offsets.len != result.len:
    badSynth("need one offset per row, got " & $offsets.len & " for " &
      $result.len & " rows")
  if shiftsMs.len notin [0, result.len]:
    badSynth("shiftsMs must be empty or one per row, got " & $shiftsMs.len)
  if traceT0Ms.len notin [0, result.len]:
    badSynth("traceT0Ms must be empty or one per row, got " & $traceT0Ms.len)
  if liveCols.len notin [0, result.len]:
    badSynth("liveCols must be empty or one per row, got " & $liveCols.len)

proc addEvent[T: SampleType](samples: var Tensor[T], offsets: openArray[float64],
    dtMs, t0Ms, velocity, freqHz, amp: float64, shiftsMs, traceT0Ms: openArray[float64],
    firstSample: int, liveCols: openArray[int], linear: bool) =
  let b = checkEvent(samples, offsets, dtMs, t0Ms, velocity, freqHz, amp, shiftsMs,
    traceT0Ms, firstSample, liveCols)
  for r in 0 ..< b.len:
    let live = if liveCols.len == 0: b.cols else: liveCols[r]
    if live < 0 or live > b.cols:
      badSynth("liveCols[" & $r & "] = " & $live & " is outside 0 .. " & $b.cols)
    if live > 0:
      let shift = if shiftsMs.len == 0: 0.0 else: shiftsMs[r]
      let t0 = if traceT0Ms.len == 0: 0.0 else: traceT0Ms[r]
      let centre =
        if linear: linearMs(t0Ms, velocity, offsets[r]) + shift
        else: hyperbolicMs(t0Ms, velocity, offsets[r]) + shift
      if finite(centre) and finite(t0):
        b.row(r, live).addRicker(dtMs, centre, freqHz, amp, t0, firstSample)

proc addLinear*[T: SampleType](samples: var Tensor[T], offsets: openArray[float64],
    dtMs, t0Ms, velocity, freqHz: float64, amp = 1.0,
    shiftsMs: openArray[float64] = [], traceT0Ms: openArray[float64] = [],
    firstSample = 0, liveCols: openArray[int] = []) =
  ## Adds a linear event at `linearMs(t0Ms, velocity, offsets[row])` on every
  ## row, plus `shiftsMs[row]` when given (a static). Column `j` of a row is
  ## at `traceT0Ms[row] + (firstSample + j)·dtMs` (`traceT0Ms` empty: 0).
  ## Only the first `liveCols[row]` columns are written (`liveCols` empty:
  ## the whole row); columns past that, and a row whose time is not finite,
  ## are left as they are. Raises `ValueError` for a bad `dtMs`, `freqHz`,
  ## `velocity` or `amp`, a gather that is not rank 2 and C-contiguous, or an
  ## array whose length is neither empty nor one per row.
  addEvent(samples, offsets, dtMs, t0Ms, velocity, freqHz, amp, shiftsMs,
    traceT0Ms, firstSample, liveCols, linear = true)

proc addHyperbola*[T: SampleType](samples: var Tensor[T], offsets: openArray[float64],
    dtMs, t0Ms, velocity, freqHz: float64, amp = 1.0,
    shiftsMs: openArray[float64] = [], traceT0Ms: openArray[float64] = [],
    firstSample = 0, liveCols: openArray[int] = []) =
  ## Adds a reflection at `hyperbolicMs(t0Ms, velocity, offsets[row])` on
  ## every row, plus `shiftsMs[row]` when given. The time axis and the
  ## `ValueError`s are `addLinear`'s.
  addEvent(samples, offsets, dtMs, t0Ms, velocity, freqHz, amp, shiftsMs,
    traceT0Ms, firstSample, liveCols, linear = false)
