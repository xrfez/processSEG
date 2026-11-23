## Kaiser-sinc and whole-sample placement of one float32 trace, 16000
## samples, shift 100.3 ms at 1 ms (whole placement at 100 samples).

import benchutil
import processSEG

const n = 16000

proc main() =
  printHeader "placement, " & $n & " float32, shift 100.3 ms at 1 ms"
  var stored = newSeq[float32](n)
  for i in 0 ..< n: stored[i] = float32(i mod 1000) * 0.001'f32
  let nOut = n + 101
  let bytes = (n + nOut) * sizeof(float32)
  var sink = 0.0

  let sinc = bench("placedSinc", bytesPerIter = bytes):
    let y = placedSinc(stored, 100.3, 1.0)
    sink += float64(y[n div 2])
  sinc.report

  let whole = bench("placedWhole", bytesPerIter = bytes):
    let y = placedWhole(stored, 100)
    sink += float64(y[n div 2])
  whole.report

  echo "# checksum ", sink

main()
