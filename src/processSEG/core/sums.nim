## Float64 sums of a row of samples, written so that the compiler vectorizes
## them. Internal to processSEG (the umbrella does not re-export this
## module): `core/basic`, `filter/edges` and `fft/spectrum` share them.
##
## A float sum keeps its IEEE order, so a plain `for` loop is one add per
## sample on a single dependency chain. Eight independent accumulators give
## the compiler eight chains to pack into SIMD registers; the result differs
## from the in-order sum only in rounding.

import parseSEG

func sum64*[T: SampleType](x: openArray[T]): float64 {.raises: [].} =
  ## Σ x in float64, with eight independent accumulators (so it vectorizes).
  {.push boundChecks: off.}
  var acc: array[8, float64]
  let n = x.len div 8 * 8
  var i = 0
  while i < n:
    for j in 0..<8: acc[j] += float64(x[i + j])
    i += 8
  for j in 0..<8: result += acc[j]
  for k in n..<x.len: result += float64(x[k])
  {.pop.}

func sumIndexed64*[T: SampleType](x: openArray[T]): float64 {.raises: [].} =
  ## Σ i·x[i] in float64, vectorized like `sum64` (the index runs in float64
  ## lanes, so there is no integer-to-float conversion per sample).
  {.push boundChecks: off.}
  var acc, idx: array[8, float64]
  for j in 0..<8: idx[j] = float64(j)
  let n = x.len div 8 * 8
  var i = 0
  while i < n:
    for j in 0..<8:
      acc[j] += idx[j] * float64(x[i + j])
      idx[j] += 8
    i += 8
  for j in 0..<8: result += acc[j]
  for k in n..<x.len: result += float64(k) * float64(x[k])
  {.pop.}
