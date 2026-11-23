## The peak of a correlation envelope: how well two traces match, the time
## shift of that peak, and the phase turn of the correlation there.
##
## The match is the envelope's peak, found between samples by a parabola
## through the peak and its two neighbours, so a constant phase turn does not
## change it. The time shift is that peak's place. The rotation is the
## argument of the correlation's analytic signal at the peak, in degrees,
## taken from the correlation over every lag.
##
## A peak under `NoClearMatch` that does not stand `ClearPeakToSidelobe`
## times above the envelope elsewhere is no clear match: no shift. A peak at
## an end of the lags is no peak: the shift is past them.
##
## ```nim
## let m = matchOf(envelope, lagsMs)
## if m.verdict == mvClear:
##   let rot = rotationOf(correlation, dtMs, m.at, m.offset)
## ```

import std/math
import ../core/workspace
import ../fft/hilbert
import sweeptrack

const
  NoClearMatch* = 0.3
    ## An envelope peak at or above this is a clear match.
  ClearPeakToSidelobe* = 4.5
    ## So is a lower one standing at least this many times above the envelope
    ## elsewhere (a noisy sweep: the shift is good though the coefficient is
    ## small). 4 s sweeps at -20 dB per sample stand 5.0 to 5.3 times above it,
    ## a Ricker at -20 dB 2.4. The envelope of white noise exceeds `x` times its
    ## RMS with probability e^-x², so 4.5 is a false peak in 1 of 10^8 lags.
  SidelobeGapMs* = 20.0
    ## "Elsewhere" is the envelope's RMS beyond this far from the peak.
  MinSidelobeSpanMs* = 40.0
    ## ... when at least this much of the lags computed lie there; otherwise
    ## the coefficient alone decides.
  PreciseShiftPeakToSidelobe* = 20.0
    ## A shift is given to 0.1 ms above this ratio.
  PreciseShiftMatch* = 0.7
    ## ... or at this match (a clean narrow-band trace has a broad envelope and
    ## a low ratio but a coefficient near 1); otherwise to the whole millisecond
    ## (`Match.about`).

type
  MatchVerdict* = enum
    ## Whether a correlation envelope has a usable peak.
    mvNothing ## No finite sample to go by.
    mvNoPeak  ## The peak is at an end of the lags: the shift is past them.
    mvWeak    ## Under `NoClearMatch`, and not `ClearPeakToSidelobe` times the level elsewhere.
    mvClear   ## A clear peak, not at an end of the lags.

  Match* = object
    ## The match, the time shift and the rotation read from one correlation.
    verdict*: MatchVerdict
    match*: float64
      ## The envelope's peak, at most 1; NaN for `mvNothing`.
    shiftMs*: float64
      ## The peak's place, in ms; NaN unless `mvClear`.
    rotationDeg*: float64
      ## -180 to 180. NaN until `rotationOf` sets it, or when it cannot be read.
    at*: int
      ## The sample of the envelope's peak; -1 for none.
    offset*: float64
      ## The peak's place from `at`, in samples, within half a sample.
    sidelobe*: float64
      ## The peak over the envelope's RMS elsewhere; NaN when not known.
    about*: bool
      ## The shift is only good to a millisecond (clear by the ratio alone, or
      ## a ratio under `PreciseShiftPeakToSidelobe`).

proc badParams(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc peakAt*(y: openArray[float64], i: int): tuple[offset, value: float64] =
  ## The vertex of the parabola through `y[i-1]`, `y[i]`, `y[i+1]` (`i` a
  ## maximum inside the array): its place in samples from `i` (within half a
  ## sample) and its height. A flat or noisy top (no downward curve) is the
  ## sample itself. An `i` at either end is that sample, at offset 0.
  ## Raises `ValueError` when `i` is outside `0 ..< y.len` (an empty `y` too).
  if i notin 0 ..< y.len:
    badParams("peakAt: index " & $i & " of " & $y.len)
  result = (0.0, y[i])
  if i < 1 or i + 1 >= y.len: return
  let den = y[i - 1] - 2 * y[i] + y[i + 1]
  if not (den < 0): return
  let d = clamp(0.5 * (y[i - 1] - y[i + 1]) / den, -0.5, 0.5)
  result = (d, y[i] - 0.25 * (y[i - 1] - y[i + 1]) * d)

func sidelobeRatio(envelope, lagsMs: openArray[float64], best: int,
    peak: float64): float64 {.raises: [].} =
  ## `peak` over the RMS of the envelope beyond `SidelobeGapMs` of sample
  ## `best`, within the lags computed; NaN when under `MinSidelobeSpanMs` of
  ## them lie there (or the RMS is 0: Inf).
  result = NaN
  let n = min(envelope.len, lagsMs.len)
  if n < 3: return
  let step = abs(lagsMs[n - 1] - lagsMs[0]) / float64(n - 1)
  var sum = 0.0
  var count = 0
  for i in 0 ..< n:
    if not envelope[i].isNaN and abs(lagsMs[i] - lagsMs[best]) > SidelobeGapMs:
      sum += envelope[i] * envelope[i]
      inc count
  if float64(count) * step < MinSidelobeSpanMs or count == 0: return
  let rms = sqrt(sum / float64(count))
  result = if rms > 0: peak / rms else: Inf

proc matchOf*(envelope, lagsMs: openArray[float64]): Match =
  ## The match and the shift from a correlation's envelope against its lags
  ## (ms, evenly spaced, one per envelope sample). NaN values are not looked
  ## at. A peak is clear at `NoClearMatch`, or at `ClearPeakToSidelobe` times
  ## the envelope elsewhere; that is checked before the peak's place, so noise
  ## peaking on the last lag is no clear match, not no peak.
  ## Raises `ValueError` when `envelope` and `lagsMs` differ in length.
  if envelope.len != lagsMs.len:
    badParams("matchOf: " & $envelope.len & " envelope, " & $lagsMs.len & " lagsMs")
  result = Match(verdict: mvNothing, match: NaN, shiftMs: NaN, rotationDeg: NaN, at: -1,
    sidelobe: NaN)
  var best = -1
  let n = min(envelope.len, lagsMs.len)
  for i in 0 ..< n:
    if not envelope[i].isNaN and (best < 0 or envelope[i] > envelope[best]): best = i
  if best < 0: return
  let (offset, top) = peakAt(envelope, best)
  result.match = min(top, 1.0)
  (result.at, result.offset) = (best, offset)
  result.sidelobe = sidelobeRatio(envelope, lagsMs, best, top)
  let byRatio = not result.sidelobe.isNaN and result.sidelobe >= ClearPeakToSidelobe
  if result.match < NoClearMatch and not byRatio: result.verdict = mvWeak
  elif best == 0 or best == n - 1: result.verdict = mvNoPeak
  else:
    result.verdict = mvClear
    let step = (lagsMs[best + 1] - lagsMs[best - 1]) / 2
    result.shiftMs = lagsMs[best] + offset * step
    result.about = result.match < NoClearMatch or (result.match < PreciseShiftMatch and
      (result.sidelobe.isNaN or result.sidelobe < PreciseShiftPeakToSidelobe))

proc rotationPhase(c: openArray[float32], dtMs: float64, at: int, offset: float64,
    ws: var Workspace): float64 =
  result = NaN
  if c.len < 2 or at notin 0 ..< c.len: return
  let near = if offset < 0: at - 1 elif offset > 0: at + 1 else: at
  if near notin 0 ..< c.len: return
  try:
    let hp = initHilbertPlan(float32, c.len)
    ws.reserve(hp.workspaceSize)
    var env, phase, hz = newSeq[float64](c.len)
    hp.analyticInto(c, dtMs, env, phase, hz, ws, -120.0)
    if phase[at].isNaN or phase[near].isNaN: return
    result = wrapDeg(phase[at] + abs(offset) * wrapDeg(phase[near] - phase[at]))
  except ValueError:
    discard

proc rotationOf*(c: openArray[float32], dtMs: float64, at: int, offset: float64): float64 =
  ## The phase turn (degrees) of the correlation `c` (float32, over every lag,
  ## lag 0 in the middle) at its peak: the argument of its analytic signal at
  ## sample `at`, taken `offset` (within half a sample) towards the next one
  ## by its phase's step. NaN when it cannot be read. Allocates a Hilbert plan,
  ## a workspace and the analytic buffers. `rotationOfInto` takes the caller's
  ## workspace for the transform scratch.
  var ws = initWorkspace()
  result = rotationPhase(c, dtMs, at, offset, ws)

proc rotationOfInto*(c: openArray[float32], dtMs: float64, at: int, offset: float64,
    ws: var Workspace): float64 =
  ## `rotationOf` using `ws` for the Hilbert scratch, grown to the plan's
  ## `workspaceSize` when it does not already fit. The Hilbert plan and the
  ## analytic buffers are still allocated. NaN when the phase cannot be read.
  result = rotationPhase(c, dtMs, at, offset, ws)
