## appliedResponseInto: whole call, float32, gain + band-pass + IIR.
##
## The sine/cosine sum this replaces took 31.79 ms at 4001×512 and 509.75 ms
## at 16384×2048 (danger, same machine class, min of 5). The number here is
## the whole call, including compile and the impulse.

import std/[math, strutils]
import benchutil
import processSEG

proc hzOf(n: int, nyq: float64): seq[float64] =
  result = newSeq[float64](n)
  for i in 0 ..< n:
    result[i] = nyq * float64(i) / float64(n - 1)

proc main() =
  printHeader("appliedResponseInto, float32, gainDb + band-pass + IIR")
  let flow = initPipeline([
    gainDbOp(6),
    bandPassOp(Corners(f1: 8, f2: 12, f3: 40, f4: 50)),
    iirOp(butterBandPass(4, 60, 24)),
  ])
  var ws = initWorkspace()
  for (samples, nHz, dtMs) in [(4001, 512, 2.0), (16384, 2048, 2.0)]:
    var hz = hzOf(nHz, 500.0 / dtMs)
    var gain = newSeq[float64](nHz)
    let name = "response " & $samples & "x" & $nHz
    let r = bench(name, bytesPerIter = samples * 4 + nHz * 8):
      flow.appliedResponseInto(float32, dtMs, samples, hz, gain, ws)
    r.report
    echo "    min ", formatFloat(r.minNs / 1e6, ffDecimal, 2), " ms  median ",
      formatFloat(r.medianNs / 1e6, ffDecimal, 2), " ms"

main()
