## Synthetic signals and gathers for the processSEG tests.
##
## Units follow the library (D7): ms, Hz, metres, m/s. Sample `j` of a trace is
## at `t0Ms + j·dtMs`.
##
## **Signals** add into a `var openArray[T]`, so they compose:
##
##   var x = zeroSignal[float32](1001)
##   x.addSinusoid(dtMs = 2, freqHz = 10)
##   x.addDc(500)
##   x.addDrift(dtMs = 2, slopePerS = 30)
##
## **Gathers** (`SynthGather`) hold parseSEG `TraceKeys` with the geometry filled
## in, the live length of each row, and a `[rows, cols]` tensor that is zero past
## each row's live length, as parseSEG's prefetcher delivers it. `rows(g)` is
## the `TraceRows` view the library takes. Events are Ricker wavelets placed at a
## moveout time computed from each trace's |offset|.
##
## Conventions:
## - offset is signed, `rcvX − srcX` (a split spread has both signs); moveout
##   formulas use |offset|;
## - an elevation static is `((srcZ − datum) + (rcvZ − datum)) / vReplacement`:
##   a station above the datum delays the event.

import std/[math, random]
import parseSEG # re-exports the CPU tensor API (not `arraymancer`, whose datasets do not build)

export parseSEG

# ------------------------------------------------------------------ signals --

func zeroSignal*[T: SampleType](n: int): seq[T] =
  ## `n` zero samples.
  newSeq[T](n)

func ricker*(tauMs, freqHz: float64): float64 =
  ## The Ricker wavelet of peak frequency `freqHz`, `tauMs` from its centre;
  ## 1 at the centre.
  let a = (PI * freqHz * tauMs / 1000) ^ 2
  (1 - 2 * a) * exp(-a)

proc addSinusoid*[T: SampleType](x: var openArray[T], dtMs, freqHz: float64,
    amp = 1.0, phaseRad = 0.0, t0Ms = 0.0) =
  ## Adds `amp·sin(2π·f·t + phase)`, `t` in seconds.
  for j in 0..<x.len:
    let t = (t0Ms + j.float64 * dtMs) / 1000
    x[j] += T(amp * sin(2 * PI * freqHz * t + phaseRad))

proc addSpike*[T: SampleType](x: var openArray[T], at: int, amp = 1.0) =
  ## Adds `amp` at sample `at`.
  x[at] += T(amp)

proc addRicker*[T: SampleType](x: var openArray[T], dtMs, centreMs,
    freqHz: float64, amp = 1.0, t0Ms = 0.0) =
  ## Adds a Ricker wavelet centred at `centreMs` (absolute time, like `t0Ms`).
  ## Samples further than 3 periods from the centre are left alone.
  let halfMs = 3000 / freqHz
  for j in 0..<x.len:
    let tau = t0Ms + j.float64 * dtMs - centreMs
    if abs(tau) <= halfMs:
      x[j] += T(amp * ricker(tau, freqHz))

proc addDc*[T: SampleType](x: var openArray[T], level: float64) =
  ## Adds a constant offset.
  for j in 0..<x.len:
    x[j] += T(level)

proc addDrift*[T: SampleType](x: var openArray[T], dtMs, slopePerS: float64,
    start = 0.0) =
  ## Adds the linear drift `start + slopePerS·t`, `t` in seconds from sample 0.
  for j in 0..<x.len:
    x[j] += T(start + slopePerS * j.float64 * dtMs / 1000)

proc addNoise*[T: SampleType](x: var openArray[T], sigma: float64, seed: int64) =
  ## Adds white Gaussian noise of standard deviation `sigma`. The same seed gives
  ## the same noise.
  var rng = initRand(seed)
  for j in 0..<x.len:
    x[j] += T(rng.gauss(0, sigma))

# ----------------------------------------------------------------- geometry --

func splitSpread*(perSide: int, groupInterval, nearOffset: float64): seq[float64] =
  ## Receiver positions relative to the source for a symmetric split spread:
  ## `perSide` receivers each side, the nearest at ±`nearOffset`, from the far
  ## negative end to the far positive end.
  result = newSeqOfCap[float64](2 * perSide)
  for i in countdown(perSide - 1, 0):
    result.add -(nearOffset + i.float64 * groupInterval)
  for i in 0..<perSide:
    result.add nearOffset + i.float64 * groupInterval

func endOn*(channels: int, groupInterval, nearOffset: float64): seq[float64] =
  ## Receiver positions relative to the source for an end-on spread on the
  ## positive side.
  result = newSeqOfCap[float64](channels)
  for i in 0..<channels:
    result.add nearOffset + i.float64 * groupInterval

func stationKeys*(srcX, rcvX, srcZ, rcvZ, dtMs: float64): TraceKeys =
  ## Keys of a live 2D trace along the x axis (y = 0).
  result = initTraceKeys()
  result.srcX = srcX
  result.srcY = 0
  result.rcvX = rcvX
  result.rcvY = 0
  result.srcZ = srcZ
  result.rcvZ = rcvZ
  result.cmpX = (srcX + rcvX) / 2
  result.cmpY = 0
  result.offset = rcvX - srcX
  result.t0Ms = 0
  result.intervalUs = dtMs * 1000
  result.traceCode = 1
  result.status = tsLive

# ------------------------------------------------------------------ gathers --

type
  SynthGather*[T: SampleType] = object
    ## Synthetic traces: keys, live lengths and zero-padded samples.
    keys*: seq[TraceKeys]
    numSamples*: seq[int] ## live samples per row (≤ cols)
    data*: Tensor[T]      ## `[rows, cols]`, zero past each row's live length
    dtMs*: float64

func len*[T](g: SynthGather[T]): int {.inline.} = g.keys.len
func cols*[T](g: SynthGather[T]): int {.inline.} = g.data.shape[1]

func rows*[T](g: SynthGather[T]): TraceRows =
  ## The `TraceRows` view of `g`. It points into `g`: do not resize `g.keys` or
  ## `g.numSamples` while it is in use.
  rowsOf(g.keys, g.numSamples)

proc initSynthGather*[T: SampleType](keys: sink seq[TraceKeys], cols: int,
    dtMs: float64): SynthGather[T] =
  ## A gather of zero traces, each `cols` samples long and fully live.
  let n = keys.len
  result = SynthGather[T](keys: keys, numSamples: newSeq[int](n),
    data: zeros[T](n, cols), dtMs: dtMs)
  for r in 0..<n:
    result.numSamples[r] = cols

proc shotGather*[T: SampleType](srcX: float64, rcvRel: openArray[float64],
    cols: int, dtMs: float64, srcZ = 0.0, rcvZ: openArray[float64] = [],
        record = 1.0): SynthGather[T] =
  ## One shot at `srcX` into receivers at `srcX + rcvRel[i]` (`splitSpread`,
  ## `endOn`). `rcvZ` gives receiver elevations (empty: all 0). `channel` counts
  ## from 1; `ensemble` stays NaN (no binning is assumed).
  doAssert rcvZ.len in [0, rcvRel.len]
  var keys = newSeq[TraceKeys](rcvRel.len)
  for i, rel in rcvRel:
    let z = if rcvZ.len > 0: rcvZ[i] else: 0.0
    keys[i] = stationKeys(srcX, srcX + rel, srcZ, z, dtMs)
    keys[i].record = record
    keys[i].channel = float64(i + 1)
  initSynthGather[T](keys, cols, dtMs)

proc cmpGather*[T: SampleType](cmpX: float64, offsets: openArray[float64],
    cols: int, dtMs: float64, cdp = 1.0): SynthGather[T] =
  ## One CMP at `cmpX`: source at `cmpX − off/2`, receiver at `cmpX + off/2`.
  var keys = newSeq[TraceKeys](offsets.len)
  for i, off in offsets:
    keys[i] = stationKeys(cmpX - off / 2, cmpX + off / 2, 0, 0, dtMs)
    keys[i].ensemble = cdp
    keys[i].channel = float64(i + 1)
  initSynthGather[T](keys, cols, dtMs)

proc setLive*[T](g: var SynthGather[T], row, samples: int) =
  ## Cuts row `row` to `samples` live samples and zeroes the rest.
  doAssert samples in 0..g.cols
  g.numSamples[row] = samples
  for j in samples..<g.cols:
    g.data[row, j] = T(0)

# ------------------------------------------------------------------- events --

func hyperbolicMs*(t0Ms, velocity, offset: float64): float64 =
  ## NMO time `sqrt(t0² + x²/v²)`, in ms, for |offset| `x`.
  sqrt(t0Ms ^ 2 + (abs(offset) / velocity * 1000) ^ 2)

func linearMs*(t0Ms, velocity, offset: float64): float64 =
  ## LMO time `t0 + |x|/v`, in ms.
  t0Ms + abs(offset) / velocity * 1000

func elevationShiftMs*(k: TraceKeys, datumZ, vReplacement: float64): float64 =
  ## The elevation static of a trace: positive when the stations are above the
  ## datum (see the module conventions).
  ((k.srcZ - datumZ) + (k.rcvZ - datumZ)) / vReplacement * 1000

proc addWavelet[T](g: var SynthGather[T], row: int, centreMs, freqHz,
    amp: float64) =
  # Adds a Ricker wavelet to the live part of one row.
  let
    t0 = g.keys[row].t0Ms
    halfMs = 3000 / freqHz
  for j in 0..<g.numSamples[row]:
    let tau = t0 + j.float64 * g.dtMs - centreMs
    if abs(tau) <= halfMs:
      g.data[row, j] = g.data[row, j] + T(amp * ricker(tau, freqHz))

proc addHyperbola*[T](g: var SynthGather[T], t0Ms, velocity, freqHz: float64,
    amp = 1.0, shiftsMs: openArray[float64] = []) =
  ## Adds a reflection at `hyperbolicMs(t0Ms, velocity, offset)` on every row,
  ## plus `shiftsMs[row]` when given (statics).
  doAssert shiftsMs.len in [0, g.len]
  for r in 0..<g.len:
    let shift = if shiftsMs.len > 0: shiftsMs[r] else: 0.0
    g.addWavelet(r, hyperbolicMs(t0Ms, velocity, g.keys[r].offset) + shift,
        freqHz, amp)

proc addLinear*[T](g: var SynthGather[T], t0Ms, velocity, freqHz: float64,
    amp = 1.0, shiftsMs: openArray[float64] = []) =
  ## Adds a linear event (direct wave, refraction) at `linearMs(t0Ms, velocity,
  ## offset)` on every row, plus `shiftsMs[row]` when given.
  doAssert shiftsMs.len in [0, g.len]
  for r in 0..<g.len:
    let shift = if shiftsMs.len > 0: shiftsMs[r] else: 0.0
    g.addWavelet(r, linearMs(t0Ms, velocity, g.keys[r].offset) + shift, freqHz, amp)

proc elevationShiftsMs*[T](g: SynthGather[T], datumZ,
    vReplacement: float64): seq[float64] =
  ## `elevationShiftMs` of every row, for `addHyperbola(shiftsMs = ...)`.
  result = newSeq[float64](g.len)
  for r in 0..<g.len:
    result[r] = elevationShiftMs(g.keys[r], datumZ, vReplacement)
