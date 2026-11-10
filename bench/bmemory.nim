## Memory-bandwidth baseline for a viewer panel (2000 traces × 2001 samples):
## a plain copy and an in-place scale. Every kernel later is compared with these.

import benchutil

const
  traces = 2000
  samples = 2001
  n = traces * samples

proc scaleInPlace(x: var openArray[float32], k: float32) {.raises: [].} =
  {.push boundChecks: off.}
  for i in 0..<x.len:
    x[i] *= k
  {.pop.}

proc main() =
  printHeader "memory baseline, " & $traces & " x " & $samples & " float32"
  var src = newSeq[float32](n)
  var dst = newSeq[float32](n)
  for i in 0..<n:
    src[i] = float32(i mod 1000)

  let copy = bench("copy (copyMem)", bytesPerIter = 2 * n * sizeof(float32)):
    copyMem(dst[0].addr, src[0].addr, n * sizeof(float32))
  copy.report

  let scale = bench("scale in place", bytesPerIter = 2 * n * sizeof(float32)):
    scaleInPlace(dst, 1.0000001'f32)
  scale.report

  # Use the outputs so neither loop can be dropped.
  var check = 0.0
  for i in countup(0, n - 1, 4099):
    check += dst[i].float64
  echo "# checksum ", check

main()
