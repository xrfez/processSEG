## FFT throughput (task 2.1): forward + inverse real FFTs of a viewer panel,
## 2000 traces, batched across traces (`forwardLanes`) and one trace at a time.
##
## Target (PLAN.md 2.1): a forward + inverse pass over a 2000-trace float32
## panel at n = 2048 in ≤ 20 ms on one thread (≤ 5 µs per trace and transform),
## so that a band-pass of the panel (3.4) fits a viewer frame with 4 workers.

import std/[math, strutils]
import benchutil
import processSEG

const traces = 2000

proc run(T: typedesc, n: int) =
  let p = initFftPlan(T, n)
  var ws = initWorkspace()
  ws.reserve(p.workspaceSize)
  var x = newSeq[T](n * fftLanes)
  for i in 0..<x.len: x[i] = T(sin(float64(i) * 0.01))
  var re, im = newSeq[T](p.bins * fftLanes)
  let batches = (traces + fftLanes - 1) div fftLanes
  # One "iteration" is the whole panel: every batch forward then inverse.
  let flops = 2.5 * float64(n) * log2(float64(n)) * 2 * traces
  let r = bench($T & " lanes  n=" & $n & " fwd+inv panel", bytesPerIter = 0):
    for _ in 0..<batches:
      p.forwardLanes(x, re, im, ws)
      p.inverseLanes(re, im, x, ws)
  r.report
  echo "    ", formatFloat(r.medianNs / 1e6, ffDecimal, 2), " ms per panel, ",
    formatFloat(r.medianNs / (2 * traces), ffDecimal, 0),
        " ns per trace and transform, ",
    formatFloat(flops / r.medianNs, ffDecimal, 1), " GFlop/s (2.5·n·log2 n per transform)"
  var y = newSeq[T](n)
  for i in 0..<n: y[i] = T(cos(float64(i) * 0.02))
  var r1, i1 = newSeq[T](p.bins)
  let s = bench($T & " single n=" & $n & " fwd+inv 1 trace", bytesPerIter = 0):
    p.forward(y, r1, i1, ws)
    p.inverse(r1, i1, y, ws)
  s.report
  echo "    ", formatFloat(s.medianNs / 2, ffDecimal, 0), " ns per transform"
  echo "# checksum ", x[7], " ", y[3]

proc main() =
  printHeader "real FFT, " & $traces & " traces, forward + inverse"
  for n in [2048, 2025, 4096, 1000]:
    run(float32, n)
  run(float64, 2048)

main()
