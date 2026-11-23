## Where a trace's signal ends (qc/signalend).
##
## Equality is `==` on float64 against the source procs kept below under
## other names (NaN only where both sides are NaN). Seeded random traces,
## and the edges named for this detector: empty input, one sample, NaN, a
## flat trace, a peak at either end.
##
## Seen fail: `blockLevels` stepped `i += n + 1`, skipping a sample between
## blocks. `random_traces` then died in `checkLevels` (`got.len == want.len`).
##
## Seen fail: `lastLoudMs` treated equal lengths as the error (`!=` became
## `==`). `empty_input` then raised `ValueError` (`0 levels, 0 timesMs`).
##
## Seen fail: `signalEndMs` dropped the short-trace guard (`x.len < 4` became
## `x.len < 0`). `empty_input` then saw 0, not NaN (`min` drops that end).

import std/[algorithm, math, random]
import processSEG

const
  RefEndFraction = 0.1
  RefBlockMs = 100.0
  RefRippleFraction = 1e-3
  RefTailFraction = 0.5

func blockLevelsRef(x: openArray[float32], dtMs: float64): (seq[float64], float64) =
  let n = max(2, int(RefBlockMs / max(dtMs, 1e-9)))
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

func lastLoudMsRef(levels, timesMs: openArray[float64], blockMs = 0.0,
    peakFloor = 0.0): float64 =
  var sorted: seq[float64] = @[]
  for v in levels:
    if not v.isNaN: sorted.add v
  if sorted.len == 0: return NaN
  sorted.sort()
  let typical = sorted[min(sorted.high, int(float64(sorted.len) * 0.9))]
  if not (typical > 0 or peakFloor > 0 and sorted[^1] > 0): return NaN
  let high = max(RefEndFraction * typical, peakFloor * sorted[^1])
  let last = min(levels.high, timesMs.high)
  for i in countdown(last, 0):
    if not levels[i].isNaN and levels[i] >= high:
      var k = i
      if peakFloor > 0:
        while k < last and not levels[k + 1].isNaN and
            levels[k + 1] >= RefTailFraction * high:
          inc k
      return timesMs[k] + blockMs
  NaN

func signalEndMsRef(x: openArray[float32], dtMs: float64): float64 =
  if x.len < 4: return NaN
  let (levels, length) = blockLevelsRef(x, dtMs)
  var times = newSeq[float64](levels.len)
  for i in 0 ..< levels.len: times[i] = float64(i) * length
  min(lastLoudMsRef(levels, times, length, RefRippleFraction), float64(x.len) * dtMs)

proc sameFloat(a, b: float64): bool =
  a == b or (a.isNaN and b.isNaN)

proc checkEnd(got, want: float64) =
  doAssert sameFloat(got, want)

proc checkLevels(x: openArray[float32], dtMs: float64) =
  let (got, gotLen) = blockLevels(x, dtMs)
  let (want, wantLen) = blockLevelsRef(x, dtMs)
  doAssert gotLen == wantLen
  doAssert got.len == want.len
  for i in 0 ..< got.len:
    doAssert got[i] == want[i]
  var wide = newSeq[float64](x.len)
  for i, v in x: wide[i] = float64(v)
  let (g64, g64Len) = blockLevels(wide, dtMs)
  doAssert g64Len == gotLen
  doAssert g64.len == got.len
  for i in 0 ..< got.len:
    doAssert g64[i] == got[i]

proc checkSignal(x: openArray[float32], dtMs: float64) =
  checkLevels(x, dtMs)
  let got = signalEndMs(x, dtMs)
  checkEnd(got, signalEndMsRef(x, dtMs))
  var wide = newSeq[float64](x.len)
  for i, v in x: wide[i] = float64(v)
  checkEnd(signalEndMs(wide, dtMs), got)

proc timesOf(n: int, step: float64): seq[float64] =
  result = newSeq[float64](n)
  for i in 0 ..< n: result[i] = float64(i) * step

proc main*() =
  block empty_input:
    let none = newSeq[float32]()
    checkSignal(none, 2.0)
    checkSignal(none, 0.0)
    let (levels, length) = blockLevels(none, 1.0)
    doAssert levels.len == 0
    doAssert length == blockLevelsRef(none, 1.0)[1]
    checkEnd(lastLoudMs(newSeq[float64](), newSeq[float64]()), NaN)
    checkEnd(lastLoudMs(newSeq[float64](), newSeq[float64](), 100.0, RippleFraction),
      lastLoudMsRef(newSeq[float64](), newSeq[float64](), 100.0, RefRippleFraction))

  block one_sample:
    let one = @[3.5'f32]
    checkSignal(one, 1.0)
    checkSignal(one, 2.0)
    doAssert signalEndMs(one, 1.0).isNaN
    let (levels, length) = blockLevels(one, 2.0)
    doAssert levels.len == 1
    doAssert levels[0] == 3.5
    doAssert length == float64(max(2, int(100.0 / 2.0))) * 2.0
    checkEnd(lastLoudMs(levels, timesOf(1, length), length),
      lastLoudMsRef(levels, timesOf(1, length), length))

  block nan_samples:
    let all = @[float32(NaN), float32(NaN), float32(NaN), float32(NaN)]
    checkSignal(all, 1.0)
    doAssert signalEndMs(all, 1.0) == 4.0 # min drops the NaN end
    var mixed = newSeq[float32](80)
    for i in 0 ..< mixed.len: mixed[i] = if i mod 5 == 0: float32(NaN) else: 0.2'f32
    mixed[3] = float32(NaN)
    checkSignal(mixed, 1.0)
    let nanLevels = @[NaN, 1.0, NaN, 0.2]
    let t = timesOf(nanLevels.len, 10.0)
    checkEnd(lastLoudMs(nanLevels, t, 10.0), lastLoudMsRef(nanLevels, t, 10.0))
    checkEnd(lastLoudMs(nanLevels, t, 10.0, RippleFraction),
      lastLoudMsRef(nanLevels, t, 10.0, RefRippleFraction))
    checkEnd(lastLoudMs(@[NaN, NaN], timesOf(2, 1.0)), NaN)

  block flat_trace:
    var zeros = newSeq[float32](400)
    checkSignal(zeros, 1.0)
    doAssert signalEndMs(zeros, 1.0) == 400.0
    var flat = newSeq[float32](400)
    for v in flat.mitems: v = 0.25'f32
    checkSignal(flat, 1.0)
    doAssert signalEndMs(flat, 1.0) == 400.0
    checkSignal(flat, 2.0)
    var tiny = newSeq[float32](4)
    for v in tiny.mitems: v = 1e-8'f32
    checkSignal(tiny, 4.0)

  block peak_at_either_end:
    proc peakAt(n, at: int): seq[float32] =
      result = newSeq[float32](n)
      result[at] = 1'f32
    let start = peakAt(400, 0)
    let ending = peakAt(400, 399)
    checkSignal(start, 1.0)
    checkSignal(ending, 1.0)
    doAssert signalEndMs(start, 1.0) == 100.0
    doAssert signalEndMs(ending, 1.0) == 400.0
    doAssert signalEndMs(start, 1.0) != signalEndMs(ending, 1.0)
    checkSignal(peakAt(4, 0), 50.0)
    checkSignal(peakAt(4, 3), 50.0)
    checkSignal(peakAt(7, 0), 25.0)
    checkSignal(peakAt(7, 6), 25.0)

  block hysteresis_and_silence:
    let levels = @[1.0, 0.06, 0.04, 0.0]
    let t = timesOf(levels.len, 100.0)
    let bare = lastLoudMs(levels, t, 100.0, 0.0)
    checkEnd(bare, lastLoudMsRef(levels, t, 100.0, 0.0))
    doAssert bare == 100.0 # no tail without a peak floor
    let withFloor = lastLoudMs(levels, t, 100.0, 0.05)
    checkEnd(withFloor, lastLoudMsRef(levels, t, 100.0, 0.05))
    doAssert withFloor == 200.0 # 0.06 stays; 0.04 does not
    checkEnd(lastLoudMs(@[0.0, 0.0], timesOf(2, 10.0), 10.0, 0.0), NaN)
    checkEnd(lastLoudMs(@[0.0, 0.0], timesOf(2, 10.0), 10.0, RippleFraction), NaN)

  block unequal_lengths:
    doAssertRaises ValueError:
      discard lastLoudMs(@[1.0, 2.0], @[0.0])
    doAssertRaises ValueError:
      discard lastLoudMs(@[1.0], @[0.0, 1.0, 2.0])
    doAssertRaises ValueError:
      discard lastLoudMs(newSeq[float64](), @[0.0])
    checkEnd(lastLoudMs(@[0.4, 0.2], timesOf(2, 25.0), 25.0, 0.0),
      lastLoudMsRef(@[0.4, 0.2], timesOf(2, 25.0), 25.0, 0.0))

  block random_traces:
    var rng = initRand(17)
    let lengths = [0, 1, 2, 3, 4, 5, 7, 8, 16, 50, 99, 100, 101, 250, 400, 1000]
    let dts = [0.0, 0.5, 1.0, 1.25, 2.0, 4.0, 25.0, 200.0]
    for n in lengths:
      for dt in dts:
        var x = newSeq[float32](n)
        for i in 0 ..< n:
          let kind = rng.rand(12)
          if kind == 0: x[i] = float32(NaN)
          elif kind == 1: x[i] = 0'f32
          elif kind == 2: x[i] = 1e-8'f32 * float32(rng.rand(-1.0 .. 1.0))
          else: x[i] = float32(rng.rand(-1.0 .. 1.0))
        checkSignal(x, dt)
    for it in 0 ..< 40:
      let n = rng.rand(0 .. 30)
      var levels = newSeq[float64](n)
      for i in 0 ..< n:
        let kind = rng.rand(10)
        if kind == 0: levels[i] = NaN
        elif kind == 1: levels[i] = 0.0
        elif kind == 2: levels[i] = rng.rand(0.0 .. 1e-4)
        else: levels[i] = rng.rand(0.0 .. 2.0)
      let step = rng.rand(1.0 .. 50.0)
      let blockMs = if it mod 5 == 0: 0.0 else: step
      let floor = case it mod 4
        of 0: 0.0
        of 1: RefRippleFraction
        of 2: 0.05
        else: rng.rand(0.0 .. 0.2)
      let t = timesOf(n, step)
      checkEnd(lastLoudMs(levels, t, blockMs, floor),
        lastLoudMsRef(levels, t, blockMs, floor))

when isMainModule: main()
