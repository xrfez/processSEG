## Amplitude statistics of traces and panels: per-trace RMS and maximum
## absolute value, and panel statistics with a percentile clip value, for
## normalization and for a viewer's display gain (a shader uniform instead of
## a pass over the samples).
##
## ```nim
## var stats = newSeq[TraceStats](meta.len)
## traceStatsInto(samples, meta, stats)            # one per row
## var ws = initWorkspace()
## ws.reserve(panelStatsWorkspace)
## let p = panelStats(samples, meta, ws, clipPercentile = 99)
## # display: clip |x| at p.clip, scale by 1 / p.clip
## ```
##
## **Which samples.** The live samples of the live rows: each live row up to
## its live length (`ChunkMeta.liveSamples`). Dead, auxiliary and failed rows
## have no statistics (`traceStatsInto` gives them NaN) and are left out of a
## panel, and so is the zero padding past a trace's end.
##
## **NaN and Inf.** A non-finite sample makes its row's statistics non-finite
## (a NaN anywhere makes both NaN; an Inf without NaN makes both +Inf). A panel
## leaves such rows out, and `PanelStats.rows` counts the rows it used.
##
## **The percentile clip** is the `p`-th percentile of `|x|` over the panel's
## samples, from a histogram over the float32 bit pattern of `|x|` (65536
## bins, each `2⁻⁸` of its value wide, so a relative error under 0.4 %,
## 0.034 dB, over the whole float32 range, whatever the dynamic range of the
## data), interpolated within the bin. `p = 100` gives the exact maximum. The
## percentile of one trace, `percentileAbs`, is exact (selection on a copy).
## Percentiles are linear between order statistics, as numpy's default: with
## `N` samples sorted, rank `h = (N − 1)·p/100`.
##
## Sums run in float64. Nothing allocates; the panel histogram lives in a
## `Workspace` (`panelStatsWorkspace`).

import std/math
import parseSEG
import ../core/[rows, types, workspace]

type
  TraceStats* = object
    ## The amplitude statistics of one trace's live samples.
    rms*: float64    ## `sqrt(Σ x² / n)`; 0 for an empty trace.
    maxAbs*: float64 ## `max |x|`; 0 for an empty trace.

  PanelStats* = object
    ## The amplitude statistics of a panel's live samples (module
    ## documentation). All 0, and `clip` NaN, when no row was used.
    rms*: float64 ## `sqrt(Σ x² / samples)` over the rows used.
    maxAbs*: float64 ## `max |x|` over the rows used.
    clip*: float64 ## The `clipPercentile` of `|x|`; NaN when not asked for.
    rows*: int ## The rows used: live, with finite statistics.
    samples*: int ## The samples used.

const
  histBins = 65536
  histShift = 15  # float32 bits of |x| >> 15: 8 exponent + 8 mantissa bits
  maxKey = 0xFEFF # the last bin of finite float32 values (0x7F7FFFFF shr 15)

  histWays = 4    # interleaved sub-histograms: no store-to-load chain on a hot bin
  panelStatsWorkspace* = WorkspaceSize(u32: histBins * histWays)
    ## The scratch `panelStats` needs (four 256 KB histograms).

func percentileWorkspace*(samples: int): WorkspaceSize {.raises: [].} =
  ## The scratch `percentileAbs` needs for a trace of up to `samples` samples.
  WorkspaceSize(f64: samples)

proc badPercentile(p: float64) {.noinline, noreturn.} =
  raise newException(ValueError, "a percentile must be in 0 .. 100, got " & $p)

proc checkPercentile(p: float64, allowNaN: bool) {.inline.} =
  if p.isNaN:
    if not allowNaN: badPercentile(p)
  elif not (p >= 0 and p <= 100): badPercentile(p)

# ------------------------------------------------------------------ kernels --

func sumSq64[T: SampleType](x: openArray[T]): float64 {.raises: [].} =
  ## Σ x² in float64, with eight accumulators (vectorized).
  {.push boundChecks: off.}
  var acc: array[8, float64]
  let n = x.len div 8 * 8
  var i = 0
  while i < n:
    for j in 0..<8:
      let v = float64(x[i + j])
      acc[j] += v * v
    i += 8
  for j in 0..<8: result += acc[j]
  for k in n..<x.len: result += float64(x[k]) * float64(x[k])
  {.pop.}

func maxAbs[T: SampleType](x: openArray[T]): float64 {.raises: [].} =
  ## max |x|, as the integer maximum of the bit patterns with the sign
  ## cleared: for non-negative floats the integer order is the float order,
  ## and every NaN sorts above +Inf, so a NaN anywhere gives NaN. (A float
  ## maximum with a select stays scalar in `-d:danger`; this vectorizes in
  ## every build.)
  {.push boundChecks: off, overflowChecks: off.}
  when T is float32:
    var m = 0'i32
    for i in 0..<x.len:
      let b = cast[int32](x[i]) and 0x7FFF_FFFF'i32
      m = (if b > m: b else: m)
    {.pop.}
    float64(cast[float32](m))
  else:
    var m = 0'i64
    for i in 0..<x.len:
      let b = cast[int64](x[i]) and 0x7FFF_FFFF_FFFF_FFFF'i64
      m = (if b > m: b else: m)
    {.pop.}
    cast[float64](m)

func sumSqMaxAbs[T: SampleType](x: openArray[T]): (float64, float64) {.inline,
    raises: [].} =
  ## (Σ x², max |x|): two passes over a row that stays in L1.
  (sumSq64(x), maxAbs(x))

func statsOf(sum, mx: float64, n: int): TraceStats {.inline.} =
  if n == 0: return TraceStats()
  result.rms = sqrt(sum / float64(n))
  result.maxAbs = mx

func traceStats*[T: SampleType](x: openArray[T]): TraceStats {.raises: [].} =
  ## The RMS and maximum absolute value of `x` (the live samples of one
  ## trace). See the module documentation for NaN and Inf.
  let (s, m) = sumSqMaxAbs(x)
  statsOf(s, m, x.len)

proc selectK(a: ptr UncheckedArray[float64], lo0, hi0, k: int) {.raises: [].} =
  ## Moves the k-th smallest of `a[lo0 .. hi0]` to `a[k]`, with smaller values
  ## before it and larger after (quickselect, median of three, no NaN).
  var lo = lo0
  var hi = hi0
  while hi > lo:
    let mid = (lo + hi) shr 1
    if a[mid] < a[lo]: swap(a[mid], a[lo])
    if a[hi] < a[lo]: swap(a[hi], a[lo])
    if a[hi] < a[mid]: swap(a[hi], a[mid])
    let pivot = a[mid]
    var i = lo
    var j = hi
    while i <= j:
      while a[i] < pivot: inc i
      while a[j] > pivot: dec j
      if i <= j:
        swap(a[i], a[j])
        inc i
        dec j
    if k <= j: hi = j
    elif k >= i: lo = i
    else: return

proc percentileAbs*[T: SampleType](x: openArray[T], p: float64,
    ws: var Workspace): float64 =
  ## The exact `p`-th percentile of `|x|` (linear between order statistics,
  ## module documentation), by selection on a copy in `ws`, which must hold
  ## `percentileWorkspace(x.len)`. 0 for an empty `x`; NaN when `x` holds a
  ## NaN, +Inf for an Inf at that rank. Raises `ValueError` for `p` outside
  ## `0 .. 100`. O(n) on average.
  checkPercentile(p, allowNaN = false)
  let n = x.len
  if n == 0: return 0
  if maxAbs(x).isNaN: return NaN
  let a = ws.scratch(float64, n)
  {.push boundChecks: off.}
  for i in 0..<n: a[i] = abs(float64(x[i]))
  {.pop.}
  let h = float64(n - 1) * p / 100
  let k = min(int(floor(h)), n - 1)
  selectK(a, 0, n - 1, k)
  let lower = a[k]
  if k + 1 >= n: return lower
  var upper = Inf
  for i in k + 1..<n: upper = min(upper, a[i])
  let f = h - float64(k)
  if f == 0: lower else: lower + f * (upper - lower)

# -------------------------------------------------------------- block forms --

proc traceStatsInto*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    stats: var openArray[TraceStats]) =
  ## The statistics of each row of `b` into `stats` (one per row): of the live
  ## samples for a live row, NaN for a dead, auxiliary or failed row. Raises
  ## `ValueError` when `b` does not match `meta` or `stats.len != b.len`.
  meta.checkShape(b)
  if stats.len != b.len:
    raise newException(ValueError, "stats holds " & $stats.len &
      " values, the block has " & $b.len & " rows")
  for r in 0..<b.len:
    if meta.isLive(r):
      stats[r] = traceStats(b.row(r, meta.liveSamples(r, b.cols)))
    else:
      stats[r] = TraceStats(rms: NaN, maxAbs: NaN)

proc traceStatsInto*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    stats: var openArray[TraceStats]) =
  ## `traceStatsInto` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  rowBlock(samples).traceStatsInto(meta, stats)

func keyOf(a: float32): int {.inline.} =
  ## The histogram bin of a finite, non-negative float32.
  min(int(cast[uint32](a) shr histShift), maxKey)

func binLow(k: int): float64 {.inline.} =
  float64(cast[float32](uint32(k) shl histShift))

proc histogramInto[T](x: openArray[T], h: ptr UncheckedArray[
    uint32]) {.raises: [].} =
  ## Counts `x` into the `histWays` histograms of `h`, sample `i` into
  ## histogram `i mod histWays`.
  {.push boundChecks: off, overflowChecks: off.}
  let n = x.len div histWays * histWays
  var i = 0
  while i < n:
    for w in 0..<histWays:
      let k = w * histBins + keyOf(float32(abs(x[i + w])))
      h[k] = h[k] + 1
    i += histWays
  for j in n..<x.len:
    let k = keyOf(float32(abs(x[j])))
    h[k] = h[k] + 1
  {.pop.}

func orderStat(h: ptr UncheckedArray[uint32], j: int, mx: float64): float64 =
  ## The estimated `j`-th smallest value (0-based) counted in `h`: in its bin,
  ## at its position among the bin's values; never above the maximum `mx`.
  var below = 0
  for k in 0..maxKey:
    let c = int(h[k])
    if below + c > j:
      let lo = binLow(k)
      let hi = min(binLow(k + 1), mx)
      return min(max(lo + (float64(j - below) + 0.5) / float64(c) * (hi - lo),
          lo), mx)
    below += c
  mx

proc panelStats*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    ws: var Workspace, clipPercentile = 99.0, rows = 0 ..
        int.high): PanelStats =
  ## The statistics of the live rows of `b` in `rows` (clipped to the block),
  ## over their live samples, leaving out rows with a NaN or Inf (module
  ## documentation). `clip` is the `clipPercentile` of `|x|`, from a
  ## histogram in `ws` (which must hold `panelStatsWorkspace`); pass
  ## `clipPercentile = NaN` to skip it (one pass over the samples instead of
  ## two; `ws` is then not used). Raises `ValueError` when `b` does not match
  ## `meta`, or for a percentile outside `0 .. 100`.
  meta.checkShape(b)
  checkPercentile(clipPercentile, allowNaN = true)
  let first = max(rows.a, 0)
  let last = min(rows.b, b.len - 1)
  var sum = 0.0
  var mx = 0.0
  for r in first..last:
    if not meta.isLive(r): continue
    let n = meta.liveSamples(r, b.cols)
    let (s, m) = sumSqMaxAbs(b.row(r, n))
    if m.classify in {fcNan, fcInf} or s == Inf: continue
    sum += s
    mx = max(mx, m)
    result.samples += n
    inc result.rows
  result.clip = NaN
  if result.samples == 0: return
  result.rms = sqrt(sum / float64(result.samples))
  result.maxAbs = mx
  if clipPercentile.isNaN: return
  if clipPercentile == 100:
    result.clip = mx
    return
  let h = ws.scratch(uint32, histBins * histWays)
  zeroMem(h, histBins * histWays * sizeof(uint32))
  for r in first..last:
    if not meta.isLive(r): continue
    let n = meta.liveSamples(r, b.cols)
    if maxAbs(b.row(r, n)).classify in {fcNan, fcInf}: continue
    histogramInto(b.row(r, n), h)
  for k in 0..maxKey: # fold the sub-histograms into the first
    for w in 1..<histWays: h[k] = h[k] + h[w * histBins + k]
  # Order statistics k and k + 1 around the rank, each placed in its bin by
  # its position among the bin's count, then linear between them.
  let rank = float64(result.samples - 1) * clipPercentile / 100
  let k = min(int(floor(rank)), result.samples - 1)
  let lower = orderStat(h, k, mx)
  let f = rank - float64(k)
  result.clip =
    if f == 0 or k + 1 >= result.samples: lower
    else: lower + f * (orderStat(h, k + 1, mx) - lower)

proc panelStats*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    ws: var Workspace, clipPercentile = 99.0, rows = 0 ..
        int.high): PanelStats =
  ## `panelStats` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  rowBlock(samples).panelStats(meta, ws, clipPercentile, rows)
