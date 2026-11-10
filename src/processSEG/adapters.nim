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
## discard f.readTracesInto(first .. last, 500 .. 999, buf, failed)
## let meta = chunkMeta(f, first .. last, 500 .. 999, failed)  # the same window
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
  ## interval is unknown (NaN, or 0 as in SEG-Y headers that leave it unset:
  ## pass `dtMs` to `chunkMeta` then), or when two rows disagree (a mixed
  ## chunk: prefetch with `uniform = true` or split with `uniformRuns`).
  requireKeys(rows)
  if rows.len == 0:
    noInterval("no rows to take the sample interval from")
  template known(us: float64): bool = us > 0 and us < Inf
  template unknown(r: int) =
    noInterval("row " & $r & " has no sample interval in its headers (" &
      $rows.keysAt(r).intervalUs & " µs): pass dtMs")
  let us = rows.keysAt(0).intervalUs
  if not known(us): unknown(0)
  for r in 1..<rows.len:
    let x = rows.keysAt(r).intervalUs
    if not known(x): unknown(r)
    if x != us:
      noInterval("rows 0 and " & $r & " have different sample intervals (" &
        $us & " and " & $x & " µs): prefetch with uniform = true, or split " &
        "the traces with uniformRuns")
  us / 1000

proc metaOf(rows: TraceRows, dtMs: float64, firstSample: int,
    failed: openArray[int]): ChunkMeta =
  requireKeys(rows)
  initChunkMeta(rows, if dtMs.isNaN: intervalMs(rows) else: dtMs, firstSample,
    failed)

proc chunkMeta*[T: SampleType](c: PrefetchChunk[T], dtMs = NaN): ChunkMeta =
  ## The meta of a prefetch chunk: its rows, their sample interval, the
  ## chunk's `firstSample` and its `failedRows`, wrapped in place (valid until
  ## `c` is filled again). The interval comes from the rows' keys
  ## (`intervalMs`) unless `dtMs` is given, for headers without one. Raises
  ## `ValueError` as `intervalMs` and `initChunkMeta` do.
  metaOf(c.rows, dtMs, c.firstSample, c.failedRows)

proc chunkMeta*(f: SeismicFile, traces: Slice[int], failed: openArray[int] = [],
    dtMs = NaN): ChunkMeta =
  ## The meta of the traces `traces` as `readTracesInto(f, traces, dst,
  ## failed)` decodes them, with `failed` the positions it left zero (not
  ## copied: it must outlive the meta). The interval is from the keys unless
  ## `dtMs` is given. Raises `ValueError` as `intervalMs` and `initChunkMeta`
  ## do, and for traces outside the file.
  metaOf(f.rowsOf(traces), dtMs, 0, failed)

proc chunkMeta*(f: SeismicFile, traces: openArray[int],
    failed: openArray[int] = [], dtMs = NaN): ChunkMeta =
  ## The meta of a list of traces (a gather, a sort order), in list order; see
  ## the `Slice` form. The view keeps a pointer to `traces`: the list must
  ## outlive the meta and keep its size.
  metaOf(f.rowsOf(traces), dtMs, 0, failed)

proc checkWindow(samples: Slice[int]) =
  if samples.a < 0 or samples.b < samples.a:
    raise newException(ValueError, "the sample window " & $samples &
      " must be a..b with 0 <= a <= b")

proc chunkMeta*(f: SeismicFile, traces: Slice[int], samples: Slice[int],
    failed: openArray[int] = [], dtMs = NaN): ChunkMeta =
  ## The meta of the sample window `samples` of `traces`, as
  ## `readTracesInto(f, traces, samples, dst, failed)` decodes it: column 0 is
  ## trace sample `samples.a`. Otherwise as the form without a window.
  checkWindow(samples)
  metaOf(f.rowsOf(traces), dtMs, samples.a, failed)

proc chunkMeta*(f: SeismicFile, traces: openArray[int], samples: Slice[int],
    failed: openArray[int] = [], dtMs = NaN): ChunkMeta =
  ## The meta of the sample window `samples` of a list of traces; see the
  ## `Slice` forms.
  checkWindow(samples)
  metaOf(f.rowsOf(traces), dtMs, samples.a, failed)
