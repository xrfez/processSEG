## The modified Bessel function I₀ for Kaiser windows (internal: the taper
## and the pilot resampler use it; not exported by the umbrella).

func besselI0*(x: float64): float64 =
  ## I₀(x) by its power series: for |x| ≤ 50 about 70 terms, with a relative
  ## error near 1e-16.
  var term = 1.0
  result = 1.0
  let q = x * x / 4
  for k in 1 .. 500:
    term *= q / float64(k * k)
    result += term
    if term < result * 1e-17: break
