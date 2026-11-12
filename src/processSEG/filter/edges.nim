## Shared edge handling of the filters (decision D4): the trend removed
## before filtering. Internal to `filter/apply` and `filter/iir`, which
## re-export `TrendRemoval`.

import parseSEG
import ../core/sums

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
  of trMean: (sum64(x.toOpenArray(0, n - 1)) / float64(n), 0.0)
  of trLinear:
    let sx = sum64(x.toOpenArray(0, n - 1))
    let nf = float64(n)
    let mean = sx / nf
    if n == 1: return (mean, 0.0)
    let c = (nf - 1) / 2
    let slope = (sumIndexed64(x.toOpenArray(0, n - 1)) - c * sx) /
      (nf * (nf * nf - 1) / 12)
    (mean - slope * c, slope)
