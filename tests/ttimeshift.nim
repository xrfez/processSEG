## A shift by milliseconds, length kept. What leaves one end does not come
## back at the other.
##
## Fail-first: `shift_whole_samples` failed while the whole-sample path used
## the phase ramp, so +3 samples was not an exact move. `shift_does_not_wrap`
## failed while a 31.5-sample shift was one phase ramp on `nextFastLen(2·n)`:
## the spike that had left the start came back at the far end as −211.8.

import std/[assertions, math]
import processSEG
import utils/allocs

proc main*() =
  block shift_whole_samples:
    let dtMs = 2.0
    let plan = initTimeShiftPlan(float64, dtMs, samples = 32)
    var ws = initWorkspace()
    ws.reserve(plan)
    var src = newSeq[float64](16)
    for j in 0 ..< src.len: src[j] = float64(j + 1)
    src[4] = NaN
    var dst = newSeq[float64](16)
    plan.shiftMsInto(src, dst, shiftMs = 6, ws) # +3 samples
    doAssert dst[1].isNaN, "src[4] moved to index 1"
    for j in 0 ..< 13:
      if src[j + 3].isNaN: doAssert dst[j].isNaN
      else: doAssert dst[j] == src[j + 3]
    for j in 13 ..< 16: doAssert dst[j] == 0, "zeros shifted in"
    plan.shiftMsInto(src, dst, shiftMs = -6, ws)
    for j in 3 ..< 16:
      if src[j - 3].isNaN: doAssert dst[j].isNaN
      else: doAssert dst[j] == src[j - 3]
    for j in 0 ..< 3: doAssert dst[j] == 0
    # Fractional: a Gaussian-windowed tone (quiet at the ends, so the zero
    # pad does not ring into it) matches the same tone generated half a sample
    # later. A hard-truncated tone would ring; that is the documented edge.
    let n = 256
    let sp = initTimeShiftPlan(float64, dtMs, n)
    ws.reserve(sp)
    var tone, want, got = newSeq[float64](n)
    let f = 20.0
    let shift = 0.5 # samples
    for j in 0 ..< n:
      let g = exp(-pow((float64(j) - 128.0) / 16, 2))
      let gs = exp(-pow((float64(j) + shift - 128.0) / 16, 2))
      tone[j] = g * cos(2 * PI * f * float64(j) * dtMs / 1000)
      want[j] = gs * cos(2 * PI * f * (float64(j) + shift) * dtMs / 1000)
    sp.shiftMsInto(tone, got, shiftMs = shift * dtMs, ws)
    var worst = 0.0
    for j in 96 ..< 160: worst = max(worst, abs(got[j] - want[j]))
    doAssert worst < 1e-4, $worst
    var row = tone
    sp.shiftMs(row, shiftMs = 6, ws)
    doAssert row[0] == tone[3] and row[^1] == 0
    let quiet = allocsIn:
      sp.shiftMsInto(tone, got, shiftMs = 1.25, ws)
    doAssert quiet == 0

  func sinc(x: float64): float64 =
    if abs(x) < 1e-12: 1.0 else: sin(PI * x) / (PI * x)

  block shift_does_not_wrap:
    # Spike 1000 at sample 0, n = 32, dt = 2, 63 ms earlier (31.5 samples).
    # A clip-and-zero-pad is ~0 at the far end; a wrap is 1000·sinc(1.5).
    let dtMs = 2.0
    let n = 32
    let plan = initTimeShiftPlan(float64, dtMs, n)
    var ws = initWorkspace()
    ws.reserve(plan)
    var src = newSeq[float64](n)
    src[0] = 1000
    var dst = newSeq[float64](n)
    plan.shiftMsInto(src, dst, shiftMs = 63, ws)
    doAssert abs(dst[31]) < 20, "far end wrapped: " & $dst[31]
    doAssert abs(dst[31] - (-211.8)) > 50, "the periodic image is back: " & $dst[31]
    # A spike at the end, moved 1.5 samples earlier, still has its side lobe
    # in the window: that is the fractional shift, then the plain move.
    var endSpike = newSeq[float64](n)
    endSpike[n - 1] = 1000
    plan.shiftMsInto(endSpike, dst, shiftMs = 1.5 * dtMs, ws)
    doAssert abs(dst[n - 2] - 1000 * sinc(0.5)) < 30, $dst[n - 2]
    # Spikes near each end, shifts of almost the whole length, both signs.
    # The end the spike did not leave through stays quiet.
    for at in [0, 1, n - 2, n - 1]:
      var spike = newSeq[float64](n)
      spike[at] = 1000
      for s in [-(float64(n) - 0.5), float64(n) - 0.5]:
        plan.shiftMsInto(spike, dst, shiftMs = s * dtMs, ws)
        let landed = float64(at) - s
        if landed < -1:
          doAssert abs(dst[n - 1]) < 40, "left through the start, back at the end: s=" &
            $s & " at " & $at & " dst=" & $dst[n - 1]
        if landed > float64(n):
          doAssert abs(dst[0]) < 40, "left through the end, back at the start: s=" &
            $s & " at " & $at & " dst=" & $dst[0]
    # Many fractional shifts: energy that should have been clipped does not
    # appear at the other end.
    var s = 0.25
    while s < float64(n):
      plan.shiftMsInto(src, dst, shiftMs = s * dtMs, ws)
      doAssert abs(dst[n - 1]) < 40, "positive " & $s & " far end " & $dst[n - 1]
      var tail = newSeq[float64](n)
      tail[n - 1] = 1000
      plan.shiftMsInto(tail, dst, shiftMs = -s * dtMs, ws)
      doAssert abs(dst[0]) < 40, "negative " & $s & " start " & $dst[0]
      s += 0.25
    # In-place equals out-of-place, including a shift that used to wrap.
    var inplace = src
    var outplace = newSeq[float64](n)
    plan.shiftMsInto(src, outplace, shiftMs = 63, ws)
    plan.shiftMs(inplace, shiftMs = 63, ws)
    for j in 0 ..< n:
      doAssert inplace[j] == outplace[j], "in-place " & $j
    inplace = src
    plan.shiftMsInto(src, outplace, shiftMs = -63, ws)
    plan.shiftMs(inplace, shiftMs = -63, ws)
    for j in 0 ..< n:
      doAssert inplace[j] == outplace[j], "in-place later " & $j

when isMainModule: main()
