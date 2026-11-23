## Where a trace's signal ends, in milliseconds.
##
## ```nim
## let endMs = signalEndMs(samples, dtMs)
## let (levels, blockMs) = blockLevels(samples, dtMs)
## ```
##
## **Blocks.** `blockLevels` cuts the trace into blocks of about 100 ms (at
## least 2 samples) and returns the largest `|sample|` of each. A NaN sample
## is left out of its block; a block of only NaN is 0, the same as silence.
##
## **The end.** `lastLoudMs` sorts the finite levels and takes the one at
## index `int(n · 0.9)` (the largest when there are 10 or fewer). The signal's
## end is the time of the last level at least a tenth of that, plus `blockMs`.
## NaN when nothing was measured, or that level is not positive and
## `peakFloor` does not see a positive loudest level. NaN levels are not
## measured.
##
## With a `peakFloor` (a fraction of the loudest level) the level must also
## reach it, and the blocks after the last such one stay in the signal while
## they stay at least half of that level: a decaying tail does not end at its
## first dip. `signalEndMs` passes `RippleFraction`.
##
## `signalEndMs` is that detector on the trace's own blocks, and not later
## than the trace's own end (`len · dtMs`). Shorter than 4 samples is NaN,
## too short to say. A longer trace whose blocks measure no end reads as
## ending at that own end (`min` drops a NaN).
##
## `levels` and `timesMs` are parallel: different lengths raise `ValueError`.
## An empty pair is NaN, not an error. Nothing here changes the samples.

import std/[algorithm, math]
import parseSEG

const
  EndFraction = 0.1  # a level below this fraction of the sorted one is silence
  BlockMs = 100.0
  TailFraction = 0.5 # a tail block counts at least this fraction of the level
  RippleFraction* = 1e-3
    ## A block quieter than this fraction of the loudest is not signal
    ## (-60 dB). `signalEndMs` passes it as `peakFloor`.

proc badParams(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func blockLevels*[T: SampleType](x: openArray[T], dtMs: float64): (seq[float64],
    float64) {.raises: [].} =
  ## The largest `|sample|` of each block of about 100 ms, and the block's
  ## length in ms. A NaN sample is left out; a block of only NaN is 0.
  let n = max(2, int(BlockMs / max(dtMs, 1e-9)))
  var levels: seq[float64] = @[]
  var i = 0
  while i < x.len:
    var top = 0.0
    for j in i ..< min(i + n, x.len):
      let v = abs(float64(x[j]))
      if not v.isNaN and v > top: top = v
    levels.add top
    i += n
  (levels, float64(n) * dtMs)

func lastLoud(levels, timesMs: openArray[float64], blockMs, peakFloor: float64): float64 {.
    raises: [].} =
  var sorted: seq[float64] = @[]
  for v in levels:
    if not v.isNaN: sorted.add v
  if sorted.len == 0: return NaN
  sorted.sort()
  let typical = sorted[min(sorted.high, int(float64(sorted.len) * 0.9))]
  if not (typical > 0 or peakFloor > 0 and sorted[^1] > 0): return NaN
  let high = max(EndFraction * typical, peakFloor * sorted[^1])
  let last = min(levels.high, timesMs.high)
  for i in countdown(last, 0):
    if not levels[i].isNaN and levels[i] >= high:
      var k = i
      if peakFloor > 0:
        while k < last and not levels[k + 1].isNaN and levels[k + 1] >= TailFraction * high:
          inc k
      return timesMs[k] + blockMs
  NaN

func lastLoudMs*(levels, timesMs: openArray[float64], blockMs = 0.0,
    peakFloor = 0.0): float64 =
  ## The time (ms) at which the signal ends: that of the last level at least
  ## a tenth of the level at `int(n · 0.9)` of the finite levels, sorted, plus
  ## `blockMs`. NaN when there is no such level. NaN levels are not measured.
  ## With a `peakFloor`, the level must also reach that fraction of the
  ## loudest, and the blocks that follow while they stay at least half of the
  ## level belong to the signal. Raises `ValueError` when `levels` and
  ## `timesMs` differ in length.
  if levels.len != timesMs.len:
    badParams("lastLoudMs: " & $levels.len & " levels, " & $timesMs.len & " timesMs")
  lastLoud(levels, timesMs, blockMs, peakFloor)

func signalEndMs*[T: SampleType](x: openArray[T], dtMs: float64): float64 {.
    raises: [].} =
  ## Where the signal of `x` ends, in ms, from its blocks (`blockLevels`,
  ## then the same detector as `lastLoudMs` with `peakFloor` =
  ## `RippleFraction`), and not later than `len · dtMs`. NaN when `x` is
  ## shorter than 4 samples. A longer trace whose blocks measure no end reads
  ## as ending at `len · dtMs`.
  if x.len < 4: return NaN
  let (levels, length) = blockLevels(x, dtMs)
  var times = newSeq[float64](levels.len)
  for i in 0 ..< levels.len: times[i] = float64(i) * length
  min(lastLoud(levels, times, length, RippleFraction), float64(x.len) * dtMs)
