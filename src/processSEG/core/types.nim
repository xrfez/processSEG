## The metadata every processSEG kernel takes alongside a block of samples:
## `ChunkMeta` (decision D12).
##
## A `ChunkMeta` wraps parseSEG's `TraceRows` view of the block plus the three
## things the view does not know: the sample interval, the window origin
## (`firstSample`) and the rows that could not be decoded. It copies no header
## value: per-row keys are read from the view when they are needed.
##
## ```nim
## let rows = rowsOf(keys, numSamples)         # or PrefetchChunk.rows, f.rowsOf(...)
## let meta = initChunkMeta(rows, dtMs = 2.0)  # validates once; raises ValueError
## meta.checkShape(samples)                    # the tensor matches the rows
## for row in 0 ..< meta.len:
##   if meta.isLive(row):
##     let n = meta.liveSamples(row, samples.shape[1])
##     ...                                     # process samples[row, 0 ..< n]
## ```
##
## **Live rows.** A row is live when its trace's `status` is `tsLive` or
## `tsUnknown` (unset SEG-Y trace codes are common) and it is not a failed
## row. Dead and auxiliary rows are passed through untouched by processing.
##
## **Live length.** `numSamples(row)` is the trace's length; the live length
## within the block is that minus `firstSample`, clamped to `0 .. cols`.
## Processing stops there; the rest of the row is zero padding.
##
## **Time axis.** Sample `j` of row `r` is at
## `t0Ms(r) + (firstSample + j) · dtMs` (`sampleTimeMs`).
##
## **NaN.** Keys are reported as parseSEG stores them, NaN where the headers
## cannot give a value (an offset without geometry, the `t0Ms` of an unknown
## SEG-D channel set). Nothing here replaces NaN by a default; derived values,
## such as `sampleTimeMs`, are then NaN too. `dtMs` itself is never NaN: it is
## checked to be finite and positive.
##
## **Lifetime.** A meta holds the view and a pointer to the failed-row list,
## so it inherits their lifetimes: the file (or the synthetic key arrays) and
## the failed-row list must outlive it and keep their size, and a view made
## before `ensureKeys` has no keys and cannot make a meta. Readers must be
## stopped before `applySurvey` or a `HeaderEditor` write.
##
## **Threads.** A `ChunkMeta` has no reference-counted parts: it can be copied
## into worker threads, and any number of threads may read through it at once.

import std/math
import parseSEG
import rows

export SampleType

type
  ChunkMeta* = object
    ## The metadata of a block of rows; see the module documentation. Build it
    ## with `initChunkMeta`, which validates it once, so reads never raise.
    rows: TraceRows
    dtMs: float64
    firstSample: int
    failed: ptr UncheckedArray[int] ## Ascending rows; nil when there are none.
    failedLen: int
    failedBase: int ## Subtracted from `failed[i]` (a `sub` meta shares the list).

proc initChunkMeta*(rows: TraceRows, dtMs: float64, firstSample = 0,
    failed: openArray[int] = []): ChunkMeta =
  ## The metadata of the rows `rows`, sampled every `dtMs` ms, whose column 0
  ## is sample `firstSample` of each trace, with the rows `failed` (0-based in
  ## the block, ascending, such as `PrefetchChunk.failedRows`) not decoded.
  ##
  ## `failed` is not copied: it must outlive the meta and keep its size (never
  ## pass a temporary such as `@[3]`). Raises `ValueError` when the view has no
  ## keys, `dtMs` is not finite and positive, `firstSample` is negative, or
  ## `failed` is not ascending within `0 ..< rows.len`.
  runnableExamples:
    import parseSEG
    var keys = @[initTraceKeys(), initTraceKeys()]
    let counts = @[500, 500]
    let failed = @[1]
    let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 2.0, failed = failed)
    doAssert meta.len == 2 and meta.isFailed(1) and not meta.isLive(1)
  if not rows.hasKeys:
    raise newException(ValueError, "the rows have no keys: open the file with " &
      "keys = true (or call ensureKeys) before making the view")
  if dtMs.classify != fcNormal or dtMs <= 0: # a subnormal interval overflows columns
    raise newException(ValueError, "dtMs must be finite and positive, got " & $dtMs)
  if firstSample < 0:
    raise newException(ValueError, "firstSample must be ≥ 0, got " & $firstSample)
  for i, row in failed:
    if row < 0 or row >= rows.len:
      raise newException(ValueError, "failed row " & $row &
        " is outside 0 ..< " &
        $rows.len)
    if i > 0 and row <= failed[i - 1]:
      raise newException(ValueError, "failed rows must be ascending and unique: " &
        $failed[i - 1] & " is followed by " & $row)
  result = ChunkMeta(rows: rows, dtMs: dtMs, firstSample: firstSample,
    failedLen: failed.len)
  if failed.len > 0:
    result.failed = cast[ptr UncheckedArray[int]](addr failed[0])

func rows*(m: ChunkMeta): TraceRows {.inline, raises: [].} =
  ## The view of the rows.
  m.rows

func len*(m: ChunkMeta): int {.inline, raises: [].} =
  ## The number of rows.
  m.rows.len

func dtMs*(m: ChunkMeta): float64 {.inline, raises: [].} =
  ## The sample interval, in ms (finite and positive).
  m.dtMs

func sameInterval*(dtMs, planDtMs: float64): bool {.inline, raises: [].} =
  ## Whether a block sampled every `dtMs` ms fits a plan (filter, gain,
  ## correlation, picking, stack) made for `planDtMs`: equal within a relative
  ## 1e-9, since an interval read from a header (2000 µs / 1000) and one
  ## computed (1000 / 500 Hz) can differ in the last bit. Every block form
  ## in processSEG checks its block's interval with this test.
  runnableExamples:
    doAssert sameInterval(2000 / 1000, 1000 / 500)
    doAssert not sameInterval(2.0, 4.0)
  abs(dtMs - planDtMs) <= 1e-9 * planDtMs

func firstSample*(m: ChunkMeta): int {.inline, raises: [].} =
  ## The trace sample that column 0 of the block holds.
  m.firstSample

func failedCount*(m: ChunkMeta): int {.inline, raises: [].} =
  ## The number of failed rows.
  m.failedLen

func failedRow*(m: ChunkMeta, i: int): int {.inline, raises: [].} =
  ## The `i`-th failed row (ascending), for `i in 0 ..< failedCount`. An
  ## `IndexDefect` outside that range, with bound checks on.
  when compileOption("boundChecks"):
    if i < 0 or i >= m.failedLen:
      raise newException(IndexDefect, "failed row index " & $i &
        " is outside 0 ..< " &
        $m.failedLen)
  m.failed[i] - m.failedBase

func keys*(m: ChunkMeta, row: int): lent TraceKeys {.inline, raises: [].} =
  ## The keys of row `row`, in place (parseSEG's `keysAt`; the meta checked
  ## `hasKeys`). An `IndexDefect` for a row outside `0 ..< len`, with bound
  ## checks on.
  m.rows.keysAt(row)

func isFailed*(m: ChunkMeta, row: int): bool {.raises: [].} =
  ## Row `row` could not be decoded (its samples are zero). O(log failedCount).
  var lo = 0
  var hi = m.failedLen
  while lo < hi:
    let mid = (lo + hi) shr 1
    if m.failed[mid] - m.failedBase < row: lo = mid + 1 else: hi = mid
  lo < m.failedLen and m.failed[lo] - m.failedBase == row

func isLive*(m: ChunkMeta, row: int): bool {.inline, raises: [].} =
  ## Row `row` holds seismic data to process: its status is `tsLive` or
  ## `tsUnknown`, and it is not a failed row.
  m.keys(row).status in {tsLive, tsUnknown} and not m.isFailed(row)

func liveSamples*(m: ChunkMeta, row, cols: int): int {.inline, raises: [].} =
  ## The live length of row `row` in a block `cols` samples wide: the trace's
  ## `numSamples` minus `firstSample`, clamped to `0 .. cols`. It does not look
  ## at liveness (`isLive`).
  clamp(m.rows.numSamples(row) - m.firstSample, 0, cols)

func t0Ms*(m: ChunkMeta, row: int): float64 {.inline, raises: [].} =
  ## The time of the trace's sample 0, in ms; NaN when the headers cannot give
  ## it.
  m.keys(row).t0Ms

func sampleTimeMs*(m: ChunkMeta, row, col: int): float64 {.inline, raises: [].} =
  ## The time of column `col` of row `row`, in ms:
  ## `t0Ms + (firstSample + col) · dtMs`. NaN when `t0Ms` is.
  m.t0Ms(row) + float64(m.firstSample + col) * m.dtMs

func sub*(m: ChunkMeta, rows: Slice[int]): ChunkMeta {.raises: [ValueError].} =
  ## The meta of the rows `rows` of `m` (row 0 is `m`'s row `rows.a`), for
  ## a `RowBlock.sub` of the same rows: splitting work across threads or
  ## tiles. It shares the view and the failed-row list (no copy, no
  ## allocation) and keeps `dtMs` and `firstSample`. Raises `ValueError` for
  ## rows outside `0 ..< len` (an empty slice may start at `len`).
  result = m
  result.rows = m.rows.sub(rows) # validates the span
  if result.rows.len == 0:
    result.failed = nil
    result.failedLen = 0
    return
  # The failed rows within `rows`: [lo, hi) of the list, by binary search.
  template firstAtLeast(v: int): int =
    var lo = 0
    var hi = m.failedLen
    while lo < hi:
      let mid = (lo + hi) shr 1
      if m.failed[mid] - m.failedBase < v: lo = mid + 1 else: hi = mid
    lo
  let lo = firstAtLeast(rows.a)
  let hi = firstAtLeast(rows.b + 1)
  result.failedLen = hi - lo
  result.failed = (if hi > lo: cast[ptr UncheckedArray[int]](addr m.failed[lo]) else: nil)
  result.failedBase = m.failedBase + rows.a

proc checkShape*[T: SampleType](m: ChunkMeta, samples: Tensor[T]) =
  ## Checks that `samples` is a C-contiguous `[rows, cols]` tensor with one row
  ## per row of `m`. Raises `ValueError` otherwise.
  if samples.rank != 2:
    raise newException(ValueError, "samples must be a [rows, cols] tensor, got rank " &
      $samples.rank)
  if samples.shape[0] != m.len:
    raise newException(ValueError, "samples have " & $samples.shape[0] &
      " rows, the meta " & $m.len)
  if not samples.is_C_contiguous:
    raise newException(ValueError, "samples must be C-contiguous (row-major)")

proc checkShape*[T: SampleType](m: ChunkMeta, b: RowBlock[T]) =
  ## Checks that the block `b` has one row per row of `m`. Raises `ValueError`
  ## otherwise.
  if b.len != m.len:
    raise newException(ValueError, "the block has " & $b.len &
      " rows, the meta " & $m.len)
