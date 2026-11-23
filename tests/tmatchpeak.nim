## Correlation-peak measures against a copy of the source arithmetic: empty
## input, one sample, NaN, a flat trace, a peak at either end, and seeded
## random cases. Float64 results use `==` (both NaN when the source returns NaN).
##
## Fail-first: `peak_ends_flat_and_parabola` failed when `peakAt`'s height line
## dropped the `0.25 *` factor (`got.value == want.value`). `match_empty_one_nan_flat_ends`
## failed when `matchOf` stored `top` instead of `min(top, 1.0)` (1.4 != 1.0).
## `rotation_edges` failed when `rotationOf` passed offset 0 instead of `offset`.
## `peak_out_of_range` failed when `peakAt` read `y[i]` before the range guard
## (IndexDefect, not ValueError) for empty, -1 and `len`.
## `rotation_bad_dt` failed when the `dtMs` guard before `analyticInto` was
## removed (AssertionDefect, not NaN) for 0, -1, Inf and NaN.

import std/[assertions, math, random]
import processSEG

const
  srcNoClearMatch = 0.3
  srcClearPeakToSidelobe = 4.5
  srcSidelobeGapMs = 20.0
  srcMinSidelobeSpanMs = 40.0
  srcPreciseShiftPeakToSidelobe = 20.0
  srcPreciseShiftMatch = 0.7

func srcPeakAt(y: openArray[float64], i: int): tuple[offset, value: float64] =
  ## Source `peakAt`.
  result = (0.0, y[i])
  if i < 1 or i + 1 >= y.len: return
  let den = y[i - 1] - 2 * y[i] + y[i + 1]
  if not (den < 0): return
  let d = clamp(0.5 * (y[i - 1] - y[i + 1]) / den, -0.5, 0.5)
  result = (d, y[i] - 0.25 * (y[i - 1] - y[i + 1]) * d)

func srcSidelobeRatio(envelope, lagsMs: openArray[float64], best: int,
    peak: float64): float64 =
  ## Source `sidelobeRatio`.
  result = NaN
  let n = min(envelope.len, lagsMs.len)
  if n < 3: return
  let step = abs(lagsMs[n - 1] - lagsMs[0]) / float64(n - 1)
  var sum = 0.0
  var count = 0
  for i in 0 ..< n:
    if not envelope[i].isNaN and abs(lagsMs[i] - lagsMs[best]) > srcSidelobeGapMs:
      sum += envelope[i] * envelope[i]
      inc count
  if float64(count) * step < srcMinSidelobeSpanMs or count == 0: return
  let rms = sqrt(sum / float64(count))
  result = if rms > 0: peak / rms else: Inf

func srcMatchOf(envelope, lagsMs: openArray[float64]): Match =
  ## Source `matchOf` (equal lengths; the source truncates a mismatch).
  result = Match(verdict: mvNothing, match: NaN, shiftMs: NaN, rotationDeg: NaN, at: -1,
    sidelobe: NaN)
  var best = -1
  let n = min(envelope.len, lagsMs.len)
  for i in 0 ..< n:
    if not envelope[i].isNaN and (best < 0 or envelope[i] > envelope[best]): best = i
  if best < 0: return
  let (offset, top) = srcPeakAt(envelope, best)
  result.match = min(top, 1.0)
  (result.at, result.offset) = (best, offset)
  result.sidelobe = srcSidelobeRatio(envelope, lagsMs, best, top)
  let byRatio = not result.sidelobe.isNaN and result.sidelobe >= srcClearPeakToSidelobe
  if result.match < srcNoClearMatch and not byRatio: result.verdict = mvWeak
  elif best == 0 or best == n - 1: result.verdict = mvNoPeak
  else:
    result.verdict = mvClear
    let step = (lagsMs[best + 1] - lagsMs[best - 1]) / 2
    result.shiftMs = lagsMs[best] + offset * step
    result.about = result.match < srcNoClearMatch or (result.match < srcPreciseShiftMatch and
      (result.sidelobe.isNaN or result.sidelobe < srcPreciseShiftPeakToSidelobe))

proc srcRotationOf(c: openArray[float32], dtMs: float64, at: int, offset: float64): float64 =
  ## Source `rotationOf`.
  result = NaN
  if c.len < 2 or at notin 0 ..< c.len: return
  let near = if offset < 0: at - 1 elif offset > 0: at + 1 else: at
  if near notin 0 ..< c.len: return
  try:
    let hp = initHilbertPlan(float32, c.len)
    var ws = initWorkspace()
    ws.reserve(hp.workspaceSize)
    var env, phase, hz = newSeq[float64](c.len)
    hp.analyticInto(c, dtMs, env, phase, hz, ws, -120.0)
    if phase[at].isNaN or phase[near].isNaN: return
    result = wrapDeg(phase[at] + abs(offset) * wrapDeg(phase[near] - phase[at]))
  except ValueError:
    discard

func eq(a, b: float64): bool =
  ## `==` on float64; both NaN counts, because NaN is not `==` to itself.
  a == b or (a.isNaN and b.isNaN)

proc sameMatch(got, want: Match) =
  doAssert got.verdict == want.verdict, $got.verdict & " != " & $want.verdict
  doAssert got.at == want.at
  doAssert got.about == want.about
  doAssert eq(got.match, want.match), $got.match & " != " & $want.match
  doAssert eq(got.shiftMs, want.shiftMs), $got.shiftMs & " != " & $want.shiftMs
  doAssert eq(got.rotationDeg, want.rotationDeg)
  doAssert eq(got.offset, want.offset), $got.offset & " != " & $want.offset
  doAssert eq(got.sidelobe, want.sidelobe), $got.sidelobe & " != " & $want.sidelobe

proc lagsOf(n: int, step, origin: float64): seq[float64] =
  result = newSeq[float64](n)
  for i in 0 ..< n: result[i] = origin + float64(i) * step

proc main*() =
  block constants:
    doAssert NoClearMatch == srcNoClearMatch
    doAssert ClearPeakToSidelobe == srcClearPeakToSidelobe
    doAssert SidelobeGapMs == srcSidelobeGapMs
    doAssert MinSidelobeSpanMs == srcMinSidelobeSpanMs
    doAssert PreciseShiftPeakToSidelobe == srcPreciseShiftPeakToSidelobe
    doAssert PreciseShiftMatch == srcPreciseShiftMatch

  block peak_out_of_range:
    # Empty, -1 and len are outside 0 ..< y.len: ValueError, not IndexDefect.
    doAssertRaises ValueError:
      discard peakAt(newSeq[float64](), 0)
    doAssertRaises ValueError:
      discard peakAt(@[0.4, 0.9], -1)
    doAssertRaises ValueError:
      discard peakAt(@[0.4, 0.9], 2)

  block peak_ends_flat_and_parabola:
    # One sample, and a peak at either end: the sample itself.
    doAssert peakAt(@[0.4], 0) == (0.0, 0.4)
    let ends = @[0.2, 0.9, 0.3]
    doAssert peakAt(ends, 0) == (0.0, 0.2)
    doAssert peakAt(ends, 2) == (0.0, 0.3)
    # A flat top (no downward curve) is the sample itself.
    doAssert peakAt(@[0.5, 0.5, 0.5], 1) == (0.0, 0.5)
    # A noisy top (upward curve) is the sample itself.
    doAssert peakAt(@[0.9, 0.1, 0.8], 1) == (0.0, 0.1)
    # A real parabola: offset and height, bit for bit with the source.
    let y = @[0.0, 1.0, 3.0, 2.0, 0.0]
    let got = peakAt(y, 2)
    let want = srcPeakAt(y, 2)
    doAssert got.offset == want.offset
    doAssert got.value == want.value
    doAssert got.offset != 0.0 and got.value != y[2]
    # NaN neighbour: no downward curve, the sample itself.
    doAssert peakAt(@[NaN, 0.7, 0.2], 1) == (0.0, 0.7)
    doAssert peakAt(@[NaN], 0).value.isNaN

  block peak_random:
    var rng = initRand(11)
    for trial in 0 .. 40:
      let n = rng.rand(1 .. 48)
      var y = newSeq[float64](n)
      for i in 0 ..< n:
        if rng.rand(1.0) < 0.1: y[i] = NaN
        else: y[i] = rng.rand(2.0) - 0.4
      for i in 0 ..< n:
        let g = peakAt(y, i)
        let e = srcPeakAt(y, i)
        doAssert g.offset == e.offset, "trial " & $trial & " i " & $i
        doAssert eq(g.value, e.value), $g.value & " != " & $e.value

  block match_empty_one_nan_flat_ends:
    let none = matchOf(newSeq[float64](), newSeq[float64]())
    sameMatch(none, srcMatchOf(newSeq[float64](), newSeq[float64]()))
    doAssert none.verdict == mvNothing and none.at == -1
    doAssert none.match.isNaN and none.shiftMs.isNaN and none.rotationDeg.isNaN
    # One sample: a peak at the only lag. Clear is no peak; weak stays weak.
    let oneClear = matchOf(@[0.8], @[0.0])
    sameMatch(oneClear, srcMatchOf(@[0.8], @[0.0]))
    doAssert oneClear.verdict == mvNoPeak and oneClear.at == 0 and oneClear.shiftMs.isNaN
    let oneWeak = matchOf(@[0.1], @[4.0])
    sameMatch(oneWeak, srcMatchOf(@[0.1], @[4.0]))
    doAssert oneWeak.verdict == mvWeak and oneWeak.match == 0.1
    # All NaN: nothing to go by. A NaN neighbour does not hide a finite peak.
    let allNan = matchOf(@[NaN, NaN, NaN], @[0.0, 1.0, 2.0])
    sameMatch(allNan, srcMatchOf(@[NaN, NaN, NaN], @[0.0, 1.0, 2.0]))
    doAssert allNan.verdict == mvNothing
    let nanSide = @[NaN, 0.6, NaN]
    sameMatch(matchOf(nanSide, @[0.0, 1.0, 2.0]), srcMatchOf(nanSide, @[0.0, 1.0, 2.0]))
    # A flat trace: the first sample wins (`>`), so the peak is at the start.
    var flat = newSeq[float64](80)
    let flatLags = lagsOf(80, 1.0, -40.0)
    for v in flat.mitems: v = 0.8
    let flatClear = matchOf(flat, flatLags)
    sameMatch(flatClear, srcMatchOf(flat, flatLags))
    doAssert flatClear.verdict == mvNoPeak and flatClear.at == 0 and flatClear.offset == 0.0
    for v in flat.mitems: v = 0.05
    let flatWeak = matchOf(flat, flatLags)
    sameMatch(flatWeak, srcMatchOf(flat, flatLags))
    doAssert flatWeak.verdict == mvWeak
    # A clear peak at either end is no peak. A weak one on the last lag is
    # no clear match, not no peak (the clear test comes first).
    let rise = @[0.05, 0.1, 0.2, 0.4, 0.95]
    let fall = @[0.95, 0.4, 0.2, 0.1, 0.05]
    let short = lagsOf(5, 1.0, 0.0)
    let atEnd = matchOf(rise, short)
    let atStart = matchOf(fall, short)
    sameMatch(atEnd, srcMatchOf(rise, short))
    sameMatch(atStart, srcMatchOf(fall, short))
    doAssert atEnd.verdict == mvNoPeak and atEnd.at == 4
    doAssert atStart.verdict == mvNoPeak and atStart.at == 0
    let weakEnd = @[0.01, 0.02, 0.04]
    let weakLags = lagsOf(3, 1.0, 0.0)
    let weakAtEnd = matchOf(weakEnd, weakLags)
    sameMatch(weakAtEnd, srcMatchOf(weakEnd, weakLags))
    doAssert weakAtEnd.verdict == mvWeak and weakAtEnd.at == 2
    # Above 1 is capped.
    let tall = @[0.0, 0.2, 1.4, 0.2, 0.0]
    let tallLags = lagsOf(5, 2.0, -4.0)
    let capped = matchOf(tall, tallLags)
    sameMatch(capped, srcMatchOf(tall, tallLags))
    doAssert capped.verdict == mvClear and capped.match == 1.0
    # An interior peak's shift agrees with the source, including the parabola.
    let spike = @[0.0, 0.0, 0.05, 0.2, 0.9, 0.4, 0.05, 0.0, 0.0]
    let spikeLags = lagsOf(9, 2.0, -8.0)
    let mid = matchOf(spike, spikeLags)
    sameMatch(mid, srcMatchOf(spike, spikeLags))
    doAssert mid.verdict == mvClear and mid.at == 4 and not mid.shiftMs.isNaN

  block match_random:
    var rng = initRand(11)
    for trial in 0 .. 50:
      let n = rng.rand(0 .. 80)
      var env = newSeq[float64](n)
      for i in 0 ..< n:
        let u = rng.rand(1.0)
        if u < 0.08: env[i] = NaN
        elif u < 0.1: env[i] = Inf
        else: env[i] = rng.rand(1.6) - 0.2
      let step = if rng.rand(1.0) < 0.5: rng.rand(3.0) + 0.25 else: -(rng.rand(3.0) + 0.25)
      let lags = lagsOf(n, step, rng.rand(40.0) - 20.0)
      sameMatch(matchOf(env, lags), srcMatchOf(env, lags))

  block lengths_differ:
    try:
      discard matchOf(@[1.0, 0.2], @[0.0])
      doAssert false, "a length mismatch must raise"
    except ValueError as e:
      doAssert e.msg == "matchOf: 2 envelope, 1 lagsMs", e.msg
    doAssertRaises ValueError:
      discard matchOf(newSeq[float64](), @[0.0])
    doAssertRaises ValueError:
      discard matchOf(@[0.4], newSeq[float64]())

  block rotation_edges:
    doAssert rotationOf(newSeq[float32](), 1.0, 0, 0.0).isNaN
    doAssert rotationOf(@[1'f32], 1.0, 0, 0.0).isNaN
    # A zero trace has no phase. A step past the end cannot be read.
    let zeros = newSeq[float32](32)
    doAssert eq(rotationOf(zeros, 2.0, 16, 0.0), srcRotationOf(zeros, 2.0, 16, 0.0))
    doAssert rotationOf(zeros, 2.0, 16, 0.0).isNaN
    var tone = newSeq[float32](64)
    for i in 0 ..< tone.len:
      tone[i] = float32(sin(2 * PI * 40.0 * float64(i) * 0.002))
    let at = 32
    doAssert eq(rotationOf(tone, 2.0, at, 0.0), srcRotationOf(tone, 2.0, at, 0.0))
    doAssert eq(rotationOf(tone, 2.0, at, -0.4), srcRotationOf(tone, 2.0, at, -0.4))
    doAssert eq(rotationOf(tone, 2.0, at, 0.25), srcRotationOf(tone, 2.0, at, 0.25))
    # Negative offset must use abs: dropping it walks the phase the other way.
    doAssert rotationOf(tone, 2.0, at, -0.4) != rotationOf(tone, 2.0, at, 0.4) or
      rotationOf(tone, 2.0, at, -0.4).isNaN
    doAssert rotationOf(tone, 2.0, 0, -0.2).isNaN
    doAssert rotationOf(tone, 2.0, tone.len - 1, 0.2).isNaN

  block rotation_random:
    var rng = initRand(13)
    for trial in 0 .. 12:
      let n = rng.rand(2 .. 96)
      var c = newSeq[float32](n)
      for i in 0 ..< n:
        if rng.rand(1.0) < 0.05: c[i] = 0
        else: c[i] = float32(rng.gauss())
      let at = rng.rand(0 .. n - 1)
      let offset = rng.rand(1.0) - 0.5
      let want = srcRotationOf(c, 2.0, at, offset)
      doAssert eq(rotationOf(c, 2.0, at, offset), want), "trial " & $trial

  block rotation_bad_dt:
    # A dtMs that is not finite and > 0 cannot be read: NaN, not AssertionDefect.
    var tone = newSeq[float32](8)
    for i in 0 ..< tone.len: tone[i] = float32(sin(float64(i)))
    doAssert rotationOf(tone, 0.0, 3, 0.0).isNaN
    doAssert rotationOf(tone, -1.0, 3, 0.0).isNaN
    doAssert rotationOf(tone, Inf, 3, 0.0).isNaN
    doAssert rotationOf(tone, NaN, 3, 0.0).isNaN

when isMainModule: main()
