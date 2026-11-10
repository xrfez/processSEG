## `ChunkMeta` from what parseSEG hands a reader: a prefetch chunk, or the
## traces of a `readTracesInto` call.
##
## These adapters copy nothing. A meta wraps the rows' `TraceRows` view and
## the failed-row list in place, and takes the sample interval from the rows'
## keys (`TraceKeys.intervalUs`): making one is a single pass over the rows to
## check that their intervals agree, with no allocation.
##
## ```nim
## var p = newPrefetcher[float32](f, 0 ..< f.len, chunkTraces = 256, uniform = true)
## var c: PrefetchChunk[float32]
## while p.next(c):
##   let meta = chunkMeta(c)         # wraps c.rows, c.firstSample, c.failedRows
##   c.samples.detrend(meta)
## ```
##
## ```nim
## var failed: seq[int]               # reused: no allocation once it has grown
## discard f.readTracesInto(first .. last, buf, failed)
## let meta = chunkMeta(f, first .. last, failed = failed)
## ```
##
## **One interval per meta.** SEG-D records can mix sample intervals across
## channel sets. A chunk whose rows disagree raises `ValueError`: prefetch with
## `uniform = true`, or split the traces with `uniformRuns`.
##
## **Lifetimes.** A meta holds the chunk's view and failed rows, so it is
## valid until the chunk is next filled (`next`, `nextInto`) or its
## prefetcher closed, and a meta of a trace list (`rowsOf(f, list)`) needs the
## list, and `failed`, to outlive it. See `core/types` for the rest.

import std/math
import parseSEG
import core/types

proc noInterval(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc requireKeys(rows: TraceRows) =
  if not rows.hasKeys:
    noInterval("the rows have no keys: open the file with keys = true (or call " &
      "ensureKeys) before making the view")

proc intervalMs*(rows: TraceRows): float64 =
  ## The sample interval of the rows, in ms, from their keys' `intervalUs`.
  ## Raises `ValueError` when the view has no keys or no rows, when an
  ## interval is unknown (NaN), or when two rows disagree (a mixed chunk:
  ## prefetch with `uniform = true` or split with `uniformRuns`).
  requireKeys(rows)
  if rows.len == 0:
    noInterval("no rows to take the sample interval from")
  let us = rows.keysAt(0).intervalUs
  if us.isNaN: noInterval("row 0 has no known sample interval")
  for r in 1..<rows.len:
    let x = rows.keysAt(r).intervalUs
    if x.isNaN: noInterval("row " & $r & " has no known sample interval")
    if x != us:
      noInterval("rows 0 and " & $r & " have different sample intervals (" &
        $us & " and " & $x & " µs): prefetch with uniform = true, or split " &
        "the traces with uniformRuns")
  us / 1000

proc chunkMeta*[T: SampleType](c: PrefetchChunk[T]): ChunkMeta =
  ## The meta of a prefetch chunk: its rows, their sample interval, the
  ## chunk's `firstSample` and its `failedRows`, wrapped in place (valid until
  ## `c` is filled again). Raises `ValueError` as `intervalMs` and
  ## `initChunkMeta` do.
  initChunkMeta(c.rows, intervalMs(c.rows), c.firstSample, c.failedRows)

proc chunkMeta*(f: SeismicFile, traces: Slice[int], firstSample = 0,
    failed: openArray[int] = []): ChunkMeta =
  ## The meta of the traces `traces` as `readTracesInto` decodes them:
  ## `firstSample` is the window's first sample (`samples.a`, 0 without a
  ## window) and `failed` the positions it left zero (its `failedRows` form;
  ## not copied, so it must outlive the meta). Raises `ValueError` as
  ## `intervalMs` and `initChunkMeta` do, and for traces outside the file.
  let rows = f.rowsOf(traces)
  initChunkMeta(rows, intervalMs(rows), firstSample, failed)

proc chunkMeta*(f: SeismicFile, traces: openArray[int], firstSample = 0,
    failed: openArray[int] = []): ChunkMeta =
  ## The meta of a list of traces (a gather, a sort order), in list order; see
  ## the `Slice` form. The view keeps a pointer to `traces`: the list must
  ## outlive the meta and keep its size.
  let rows = f.rowsOf(traces)
  initChunkMeta(rows, intervalMs(rows), firstSample, failed)
