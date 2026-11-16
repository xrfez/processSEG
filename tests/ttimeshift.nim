## A shift by milliseconds, length kept.
##
## Fail-first: `shift_whole_samples` failed while the whole-sample path used
## the phase ramp, so +3 samples was not an exact move.

import std/[assertions, math]
import processSEG
import utils/allocs

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
