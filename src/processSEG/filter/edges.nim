## Shared edge handling of the filters (decision D4): the trend removed
## before filtering. Internal to `filter/apply` and `filter/iir`, which
## re-export `TrendRemoval`.

import parseSEG

type
  TrendRemoval* = enum
    ## What is removed from a trace before it is padded and filtered (and
    ## added back after, scaled by the gain at 0 Hz).
    trNone ## nothing
    trMean ## the mean
    trLinear ## the least-squares line (also removes the mean)

proc trendOf*[T: SampleType](trend: TrendRemoval, x: ptr UncheckedArray[T],
    n: int): (float64, float64) {.raises: [].} =
  ## `(a, b)` of the trend `a + b·i` of `x[0 ..< n]` (n ≥ 1) to remove.
  case trend
  of trNone: (0.0, 0.0)
  of trMean:
    var s = 0.0
    for i in 0..<n: s += float64(x[i])
    (s / float64(n), 0.0)
  of trLinear:
    var sx, sxi = 0.0
    for i in 0..<n:
      sx += float64(x[i])
      sxi += float64(i) * float64(x[i])
    let nf = float64(n)
    let mean = sx / nf
    if n == 1: return (mean, 0.0)
    let c = (nf - 1) / 2
    let slope = (sxi - c * sx) / (nf * (nf * nf - 1) / 12)
    (mean - slope * c, slope)
