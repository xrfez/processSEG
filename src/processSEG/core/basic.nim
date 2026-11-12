## Basic trace kernels: demean, linear detrend, scale, clip, polarity reversal
## and the pad-only taper.
##
## Each operation comes in two forms:
##
## - a **kernel** on one row, `var openArray[T]`: it processes every sample it
##   is given, so the caller passes the live part (`b.row(r, n)`);
## - a **block form** on a `[rows, cols]` tensor (as parseSEG decodes it) or a
##   `RowBlock`, with its `ChunkMeta`: it processes the live rows only, each up
##   to its live length (`ChunkMeta.liveSamples`). Dead, auxiliary and failed
##   rows, and the zero padding past a trace's end, are left untouched.
##
## ```nim
## var samples = f.readTraces(0 ..< 100)        # Tensor[float32], [traces, samples]
## let meta = initChunkMeta(f.rowsOf(0 ..< 100), dtMs = 2)
## samples.detrend(meta)                        # in place, live rows only
## samples.scale(meta, 0.5)
## ```
##
## The kernels do not allocate or raise (except `clip` on bad limits and
## `taperPads` on a ramp that does not fit); the
## block forms raise `ValueError` once, before touching any sample, when the
## tensor or block does not match the meta. Sums run in float64. A NaN or Inf
## sample stays in its own row: a mean or trend over it is NaN, so that row
## becomes NaN, and no other row is affected.

import std/math
import parseSEG
import rows, sums, types

# ------------------------------------------------------------------ helpers --

proc badLimits(lo, hi: float64) {.noinline, noreturn.} =
  raise newException(ValueError, "clip needs lo ≤ hi, both not NaN; got " &
    $lo & " and " & $hi)

proc badTaper(ramp, len: int) {.noinline, noreturn.} =
  raise newException(ValueError, "a taper of " & $ramp &
    " samples on each side does not fit " & $len & " samples")

template eachLiveRow(b: RowBlock, meta: ChunkMeta, x, body: untyped) =
  ## Runs `body` once per live row of `b`, with `x` its live samples.
  meta.checkShape(b)
  for r in 0..<b.len:
    if meta.isLive(r):
      let n = meta.liveSamples(r, b.cols)
      template x: untyped = b.row(r, n)
      body

# ------------------------------------------------------------------ kernels --

proc demean*[T: SampleType](x: var openArray[T]) {.raises: [].} =
  ## Subtracts the mean of `x` from every sample.
  if x.len == 0: return
  let m = T(sum64(x) / float64(x.len))
  {.push boundChecks: off.}
  for i in 0..<x.len:
    x[i] = x[i] - m
  {.pop.}

proc detrend*[T: SampleType](x: var openArray[T]) {.raises: [].} =
  ## Subtracts the least-squares line `a + b·i` from `x`, which also removes
  ## the mean. A row of one sample becomes 0.
  let n = x.len
  if n == 0: return
  doAssert n <= int32.high, "a row longer than int32.high samples"
  # With t = i − c, c = (n − 1)/2: b = Σ t·x / Σ t², a = mean − b·c,
  # and Σ t² = n(n² − 1)/12.
  let
    nf = float64(n)
    c = (nf - 1) / 2
    sx = sum64(x)
    mean = sx / nf
    stt = nf * (nf * nf - 1) / 12
    slope = if n > 1: (sumIndexed64(x) - c * sx) / stt else: 0.0
    a = T(mean - slope * c)
    b = T(slope)
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0'i32..<int32(n):
    x[i] = x[i] - (a + b * T(i))
  {.pop.}

proc scale*[T: SampleType](x: var openArray[T], factor: float64) {.raises: [].} =
  ## Multiplies every sample by `factor`, which the caller checked to be
  ## finite (the block forms raise otherwise).
  let k = T(factor)
  {.push boundChecks: off.}
  for i in 0..<x.len:
    x[i] = x[i] * k
  {.pop.}

proc clip*[T: SampleType](x: var openArray[T], lo, hi: float64) =
  ## Limits every sample to `lo .. hi`. A NaN sample stays NaN. Raises
  ## `ValueError` when `lo > hi` or either is NaN (±Inf is allowed).
  if not (lo <= hi): badLimits(lo, hi)
  let
    l = T(lo)
    h = T(hi)
  {.push boundChecks: off.}
  for i in 0..<x.len:
    let v = x[i]
    x[i] = (if v > h: h elif v < l: l else: v)
  {.pop.}

proc reversePolarity*[T: SampleType](x: var openArray[T]) {.raises: [].} =
  ## Negates every sample.
  {.push boundChecks: off.}
  for i in 0..<x.len:
    x[i] = -x[i]
  {.pop.}

proc cosTaperInto*[T: SampleType](ramp: var openArray[T]) {.raises: [].} =
  ## Fills `ramp` with a rising cos² taper of `n = ramp.len` weights,
  ## `ramp[i] = sin²(π/2 · (i + 1)/(n + 1))`: it starts just above 0 and ends
  ## just below 1, so neither the outermost pad sample nor the data edge gets a
  ## weight of exactly 0 or 1. Computed once per plan, for `taperPads`.
  let n = ramp.len
  for i in 0..<n:
    ramp[i] = T(sin(PI / 2 * float64(i + 1) / float64(n + 1)) ^ 2)

proc taperPads*[T: SampleType](x: var openArray[T], ramp: openArray[T]) =
  ## Tapers the pads of a padded trace, never its data (decision D4): with
  ## `p = ramp.len`, multiplies `x[i]` by `ramp[i]` and `x[^(i + 1)]` by
  ## `ramp[i]` for `i < p`, so both ends fall towards 0 and the middle
  ## `x[p ..< len − p]` (the data) is untouched. `ramp` comes from
  ## `cosTaperInto`. Raises `ValueError` when `2·p > x.len`.
  let p = ramp.len
  let n = x.len
  if 2 * p > n: badTaper(p, n)
  {.push boundChecks: off.}
  for i in 0..<p:
    x[i] = x[i] * ramp[i]
  for i in 0..<p:
    x[n - 1 - i] = x[n - 1 - i] * ramp[i]
  {.pop.}

# -------------------------------------------------------------- block forms --

proc demean*[T: SampleType](b: RowBlock[T], meta: ChunkMeta) =
  ## `demean` on the live samples of each live row of `b`. Raises `ValueError`
  ## when `b` and `meta` have different row counts.
  eachLiveRow(b, meta, x): demean(x)

proc detrend*[T: SampleType](b: RowBlock[T], meta: ChunkMeta) =
  ## `detrend` on the live samples of each live row of `b`; see `demean`.
  eachLiveRow(b, meta, x): detrend(x)

proc scale*[T: SampleType](b: RowBlock[T], meta: ChunkMeta, factor: float64) =
  ## `scale` on the live samples of each live row of `b`; see `demean`. Raises
  ## `ValueError` for a factor that is NaN or infinite, also in `T` (1e39 in
  ## float32: the pipeline's `scaleOp` refuses it too).
  if factor.classify in {fcNan, fcInf, fcNegInf} or
      T(factor).float64.classify in {fcInf, fcNegInf}:
    raise newException(ValueError, "a scale factor must be finite in " & $T &
      ", got " & $factor)
  eachLiveRow(b, meta, x): scale(x, factor)

proc clip*[T: SampleType](b: RowBlock[T], meta: ChunkMeta, lo, hi: float64) =
  ## `clip` on the live samples of each live row of `b`; see `demean`. The
  ## limits are checked before any sample changes.
  if not (lo <= hi): badLimits(lo, hi)
  eachLiveRow(b, meta, x): clip(x, lo, hi)

proc reversePolarity*[T: SampleType](b: RowBlock[T], meta: ChunkMeta) =
  ## `reversePolarity` on the live samples of each live row of `b`; see
  ## `demean`.
  eachLiveRow(b, meta, x): reversePolarity(x)

proc demean*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta) =
  ## `demean` on the live part of a `[rows, cols]` tensor, in place. Raises
  ## `ValueError` unless `samples` is C-contiguous with `meta.len` rows.
  meta.checkShape(samples)
  demean(rowBlock(samples), meta)

proc detrend*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta) =
  ## `detrend` on the live part of a tensor, in place; see the tensor `demean`.
  meta.checkShape(samples)
  detrend(rowBlock(samples), meta)

proc scale*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    factor: float64) =
  ## `scale` on the live part of a tensor, in place; see the tensor `demean`.
  meta.checkShape(samples)
  scale(rowBlock(samples), meta, factor)

proc clip*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta, lo,
    hi: float64) =
  ## `clip` on the live part of a tensor, in place; see the tensor `demean`.
  meta.checkShape(samples)
  clip(rowBlock(samples), meta, lo, hi)

proc reversePolarity*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta) =
  ## `reversePolarity` on the live part of a tensor, in place; see the tensor
  ## `demean`.
  meta.checkShape(samples)
  reversePolarity(rowBlock(samples), meta)
