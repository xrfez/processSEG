## Both lags of a trace correlation, zero lag at a caller time.
##
## Fail-first: `correlation_peaks_at_asked_time` failed while zero lag was
## written at sample 0, so the asked sample read −0.85 instead of 1.

import std/[assertions, math]
import processSEG
import utils/allocs

func near(a, b, tol: float64): bool = abs(a - b) <= tol

block correlation_peaks_at_asked_time:
  let dtMs = 2.0
  let n = 64
  let plan = initTraceCorrelationPlan(float64, dtMs, n)
  var ws = initWorkspace()
  ws.reserve(plan)
  var x = newSeq[float64](n)
  for j in 0 ..< n: x[j] = cos(2 * PI * 25 * float64(j) * dtMs / 1000)
  var dst = newSeq[float64](n)
  let at = 20.0 # ms, sample 10
  plan.correlateAtInto(x, x, dst, zeroLagMs = at, ws)
  let z = int(round(at / dtMs))
  doAssert near(dst[z], 1.0, 1e-9), $dst[z]
  for k in 1 .. z:
    doAssert near(dst[z + k], dst[z - k], 1e-9), "symmetric about the asked time"
  # A lag with no overlap is 0, not a wrapped value. Eight samples before
  # zero lag, a reference of 8 samples has nothing to multiply.
  var short = newSeq[float64](8)
  for j in 0 ..< 8: short[j] = x[j]
  plan.correlateAtInto(x, short, dst, zeroLagMs = at, ws, norm = corrNone)
  doAssert dst[z - 8] == 0 and dst[z - 9] == 0
  # None: the peak is the energy.
  plan.correlateAtInto(x, x, dst, zeroLagMs = at, ws, norm = corrNone)
  var e = 0.0
  for v in x: e += v * v
  doAssert near(dst[z], e, 1e-6), $dst[z]
  # A delayed copy peaks later than the asked time.
  var delayed = newSeq[float64](n)
  for j in 4 ..< n: delayed[j] = x[j - 4]
  plan.correlateAtInto(delayed, x, dst, zeroLagMs = at, ws)
  var best = 0
  for j in 0 ..< n:
    if abs(dst[j]) > abs(dst[best]): best = j
  doAssert best == z + 4, "a copy delayed by 4 samples peaks 4 later, at " & $best
  let quiet = allocsIn:
    plan.correlateAtInto(x, x, dst, zeroLagMs = at, ws)
  doAssert quiet == 0
