# Proposed parseSEG changes for processSEG and the viewer

Status: proposal, 2026-11-10, written against parseSEG 0.6.1 (`22fb75c`). It replaces
P1–P4 in processSEG's `docs/integration.md`.

## The goal: one per-trace data model

parseSEG's trace index should be the **only** copy of per-trace metadata. The viewer
and processSEG both read it in place. Neither copies headers into objects of its own.

parseSEG is most of the way there:

- `openSeismic(path, keys = true)` builds `f.index.keys: seq[TraceKeys]`, one record per
  trace, parallel to `f.index.entries`.
- `applySurvey` and `applyGeometry` update those keys in place, so an SPS or P1 join is
  already single-source.
- `TraceEntry` holds each trace's sample count, channel set and channel type.

What is missing:

1. **Fields.** processSEG needs a few per-trace values that are not in `TraceKeys`: the
   start time, the sample interval, inline and crossline, and the trace's status
   (live, dead or auxiliary). Today a caller gets them through separate header reads
   (`traceFieldColumn`, `sampleIntervalUs`, a private `segyTimeMs`) and has to keep the
   results in its own arrays. That is the second copy we want to avoid.
2. **A view type.** There is no public, zero-copy, thread-shareable way to say "these
   rows of this chunk are these traces, with these keys". Without one, processSEG would
   copy keys into a `ChunkMeta`.
3. **Consistency.** `HeaderEditor.setTraceField` changes the bytes on disk but not
   `index.keys`, so the keys go stale after an edit. Nothing tells a cache that keys
   changed (after a join or an edit).

With these changes, processSEG's `ChunkMeta` shrinks to a parseSEG view plus `dtMs` and
`firstSample`, and its adapter module mostly disappears.

## Ground rules

- parseSEG stays independent: no processSEG import, no DSP.
- Keep parseSEG's conventions:
  - unknown values are NaN (floats) or 0 (codes);
  - a bad argument raises `ValueError`; data problems never raise and are logged;
  - `xInto` forms reuse caller storage.
- No ref-counted value may need to cross threads. Seqs are fine to *read* from several
  threads while nothing resizes them. Arraymancer tensors are not (their storage is a
  non-atomic `ref`).
- Each change comes with tests, doc comments that `nim doc` picks up, and README/docs
  updates. Release as 0.7.0, since `TraceKeys` grows and the index cache format changes.
- Changes are listed in priority order. C1–C4 unblock processSEG's Phase 1; the others
  can follow.

## C1. Complete `TraceKeys`

Add the fields below, filled wherever keys are built: at open with `keys = true`, by
`traceKeysInto`, and by the per-standard `traceKeys` procs.

| Field | Type | Meaning, per standard |
|---|---|---|
| `t0Ms` | float64 | Time of sample 0 in ms. SEG-Y: delay recording time (bytes 109–110) × the time scalar (bytes 215–216, Rev 1+, 0 means 1), as `convert/toseg2.segyTimeMs` does now. SEG-D: `startTimeMs(rec, trace)`, the channel set start time. SEG-2: DELAY (s) × 1000. 0 when the standard has the field and it is unset, NaN when the header cannot give it. |
| `intervalUs` | float64 | The trace's sample interval, resolved exactly as `sampleIntervalUs(f, trace)` does. |
| `inline`, `crossline` | float64 | SEG-Y: the layout's `iline`/`xline` fields when the Layout stanza (or a vendor map) defines them, else the Rev 2 standard bytes 189 and 193 when the file declares 3D-poststack usage, else NaN. SEG-D/SEG-2: NaN. |
| `traceCode` | int16 | SEG-Y trace identification code (bytes 29–30) as stored; 0 when unknown or not SEG-Y. |
| `status` | `TraceStatus` enum | `tsLive`, `tsDead`, `tsAuxiliary`, `tsUnknown`. Dead: SEG-Y trace codes 2 (dead) and 3 (dummy). Auxiliary: `not isSeismic` (SEG-D channel types, SEG-2 TEST_DATA and UPHOLE), and the SEG-Y auxiliary codes (time break, uphole, sweep, timing, water break, …). |

Notes for whoever implements C1:

- **Memory.** `TraceKeys` is 144 bytes now and ends up around 180. The index cache stores
  keys, so its format version must change. Document the new size in the `keys` field doc
  ("N bytes per trace").
- **Reuse code that already exists:**
  - `isSeismic`'s rules become `status`;
  - `segyTimeMs` moves out of `convert/toseg2` into the key reader;
  - `sampleIntervalUs` and the new `intervalUs` must share one implementation, so they
    cannot disagree.
- **Keep the per-trace start-time accessors too** (the old P1): a public
  `startTimeMs(f: SeismicFile, trace: int): float64` and a column form, for callers
  that open with `keys = false`.
- **Units.** `t0Ms` is in ms even for SEG-Y depth data, where the "time" fields mean
  depth. Document that; processSEG only uses it for time-domain data.

*Accept:*

- every new field has tests on the generated demo files for each standard (SEG-Y Rev 0/1/2,
  SEG-D Rev 1/2/3, SEG-2), including a nonzero delay and a time scalar;
- dead and auxiliary traces are classified correctly;
- inline and crossline come from a Layout stanza;
- an old index cache is rejected cleanly and rebuilt.

## C2. `TraceRows`: a zero-copy view of rows → traces

A small public `object` (no `ref`, no seq inside). It says which trace each row of a
block of samples is, and gives that trace's keys and entry without copying:

```nim
type TraceRows* = object
  ## A view of `count` rows: row i is trace `traceOf(i)`. Valid while the file (or the
  ## arrays it was built from) stays alive and unchanged in size. Not ref-counted:
  ## safe to copy into worker threads.
  count*: int
  # private: ptr UncheckedArray[TraceKeys], ptr UncheckedArray[TraceEntry],
  #          first (range mode) or ptr UncheckedArray[int] (selection mode)

func rowsOf*(f: SeismicFile, traces: Slice[int]): TraceRows        # requires keys
func rowsOf*(f: SeismicFile, traces: openArray[int]): TraceRows    # the list must outlive the view
func rowsOf*(keys: openArray[TraceKeys], numSamples: openArray[int]): TraceRows
  ## For synthetic data and tests: no file needed (processSEG's tests use this).

func traceOf*(r: TraceRows, row: int): int
func keys*(r: TraceRows, row: int): lent TraceKeys    # or a `ptr`, whichever reads best
func numSamples*(r: TraceRows, row: int): int
func channelSet*(r: TraceRows, row: int): int
func sub*(r: TraceRows, rows: Slice[int]): TraceRows   # a sub-view, for splitting work
```

- It raises `ValueError` if the file was opened without keys (see C3).
- The names are suggestions. Run parseSEG's API-design pass (`nim-api-design`) and pick
  final names consistent with the rest of parseSEG.
- Document the lifetime and threading contract plainly. Readers on several threads are
  fine. Anything that changes keys (C4 edits, `applySurvey`) needs the caller to stop
  readers first; the keys revision (C3) lets them notice afterwards.

*Accept:*

- range, selection and synthetic views give the same keys as `traceKeys(f, i)`;
- `sub` is tested at the edges;
- a view is used from 4 threads under ASan;
- no allocation in `rowsOf` or `sub`.

## C3. Keys on demand, and a keys revision

- `ensureKeys*(f: var SeismicFile, progress: Progress = nil)` builds `index.keys` for a
  file opened with `keys = false`, and does nothing if they exist. The viewer can then open
  a large file fast and build keys when the user first needs geometry.
- `keysRevision*(f: SeismicFile): int` goes up whenever keys change: `ensureKeys`,
  `applySurvey`, `applyGeometry`, and editor writes (C4). The viewer and processSEG tag
  derived data (moveout overlays, binning, stack geometry) with it and rebuild when it
  changes.

*Accept:* `ensureKeys` matches `openSeismic(keys = true)` key for key, including through
the index cache, and every key-changing path bumps the revision, each covered by a test.

## C4. Keep keys current under header edits

`HeaderEditor.setTraceField` (and batch edits, `rollbackJournal`, `applyJournal` on an
open file) must refresh the edited trace's `TraceKeys` when the field feeds a key.
Coordinates, scalars, offset, CDP, delay, time scalar, trace code, iline/xline and so on
all feed one.

- **Joined keys.** If the index is `joined` (SPS/P1 applied), an edit to a header that the
  join overrode must not silently undo the join. Choose a rule and document it. Suggested
  rule: re-derive from the headers, then re-apply the stored join for that trace when the
  join filled that field. If keeping join state is too much, refuse to refresh and log a
  diagnostic.
- Bump `keysRevision` once per committed edit or batch.

*Accept:*

- after `setTraceField(17, "trace.sourceX", ...)`, `f.index.keys[17]` matches a fresh
  open;
- the joined-index rule is tested;
- a rollback restores the keys.

## C5. Prefetcher: chunks carry their rows

- **Rows.** `PrefetchChunk` gains `rows: TraceRows` for its rows, in range and selection
  mode alike, so the consumer does not rebuild the mapping with `traceAt`.
- **Failed rows.** Make the rows that could not be decoded (they come back zero) public:
  for example `failedRows(c): openArray[int]`, or `isFailed(c, row)`, backed by the slot's
  existing `fails` array. Today only the count and a `DiagLog` are public.
- **Uniform chunks (optional, a flag).** With `uniform = true`, a chunk never spans two
  channel sets or sample intervals: it is cut short at a boundary. Then every chunk has
  exactly one `dt`.
- **A transform hook (the old P2, optional).** A per-chunk `transform` runs on the worker
  thread right after decoding. It receives the worker index, a raw pointer to the rows
  with their shape, the chunk's `TraceRows`, and the failed rows. It is
  `{.gcsafe, raises: [].}` and may capture only raw pointers to state that outlives the
  prefetcher. It lets batch jobs process each chunk while it is still in cache.

*Accept:*

- `rows` agrees with `traceAt` in both modes;
- failed rows are reported for truncated and undecodable traces;
- uniform chunks never mix intervals, on a SEG-D file with two intervals;
- the transform runs once per chunk on a worker, under ASan.

## C6. Split traces into uniform runs (the old P4)

`iterator uniformRuns*(f: SeismicFile, traces: Slice[int]): Slice[int]` and a selection
form. Each run shares one channel set, sample interval and sample count class. This is
for callers that read without the prefetcher.

*Accept:* run boundaries are tested on mixed SEG-D, and the trivial single run on SEG-Y.

## C7. Write derived traces (the old P3, later)

`writeTrace` with header overrides from a template trace: for example, a stacked trace
written with a source trace's headers but with `cdp`, `fold`/`ensemble`, `offset` and the
coordinates changed. This lets processSEG write a stack or a processed file as SEG-Y.

*Accept:* a round trip of a written stack through `openSeismic` gives the overridden keys
and the samples.

## C8. Statics and datum fields (later, for processSEG Phase 7)

Add the SEG-Y elevation and statics fields to `TraceKeys`, or behind an opt-in: datum
elevations at source and receiver (bytes 53–60), source and receiver depth (49–52),
uphole times (95–98), source/group/total statics (99–104, ms), and the weathering and
subweathering velocities (91–94). All are scaled as the standard says. Lower priority:
processSEG can compute elevation statics from `srcZ`/`rcvZ` and a replacement velocity
without them.

## What processSEG does with this

- `ChunkMeta` becomes `{rows: TraceRows, dtMs, firstSample}`, built in O(1) with no copy.
- Kernels read `rows.keys(i).offset`, `.t0Ms`, `.status`, and `rows.numSamples(i)` for
  the live length.
- Synthetic tests build views with `rowsOf(keys, numSamples)`, so the processSEG core
  still never opens a file.
- Until a change lands, processSEG's adapter works around its absence. Each workaround
  gets deleted when its change lands.

## Workflow notes for the parseSEG agent

- Work in a normal parseSEG checkout. Its 47 tests do not run from processSEG's
  `deps/parseSEG`: processSEG's `config.nims` takes over `nim test`, and `tdocs.nim`
  hard-codes parseSEG's own `deps/`.
- Branch from `origin/main`, with one PR per change or per related group (C1+C2, …).
- processSEG picks up the changes when it moves its requirement to the release
  (`>= 0.7.0`, then `atlas install` and `atlas pin`), or to a merged commit on main.
