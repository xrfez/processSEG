## A viewer simulator: **test code only**, never exported from `src/`.
##
## processSEG is a library, not a viewer. This module acts out what a viewer
## does with it, so the tests can check that the public API supports a
## viewer's loop without allocating or showing stale data:
##
## - prefetch raw chunks (`uniform = true`) into a raw cache that is kept;
## - process each raw chunk into its own processed buffer with a
##   `ProcessPool`, tagging the buffer with the plan's `generation`;
## - "draw": check that every buffer is current instead of drawing it;
## - a slider change: edit the `Pipeline`, recompile, and let the
##   generation mark every buffer stale;
## - a zoom window decoded with the plan's `requiredMargin`;
## - moveout ticks tagged with `keysRevision`, made stale by `applySurvey`.
##
## A plan is compiled per sample interval and row width, because a uniform
## prefetch of a mixed SEG-D file yields chunks at different intervals.

import std/math
import parseSEG, processSEG

type
  Entry* = object
    ## One cached chunk: the raw samples as decoded (kept), the processed
    ## buffer, and the generation of the plan that filled it (0: never).
    raw*: PrefetchChunk[float32]
    processed*: Tensor[float32]
    generation*: uint64
    plan: int ## index into `plans`
    ticks*: seq[float64] ## moveout columns per row
    ticksRevision*: int  ## `keysRevision` the ticks were made from (-1: none)

  PlanKey = object
    dtMs: float64
    cols: int

  ViewerSim* = object
    pool*: ProcessPool
    flow*: Pipeline
    plans: seq[PipelinePlan[float32]]
    keys: seq[PlanKey]
    entries*: seq[Entry]
    reprocessed*: int ## buffers processed by the last `frame`
    drawn*: float64   ## a checksum of what the last `frame` "drew"

proc initViewerSim*(flow: sink Pipeline, workers = 4): ViewerSim =
  ## A viewer with a pool of `workers` and the flow `flow`.
  ViewerSim(pool: newProcessPool(workers), flow: flow)

proc planFor(v: var ViewerSim, dtMs: float64, cols: int): int =
  ## The plan for `dtMs` and `cols`, compiled on first use.
  for i, k in v.keys:
    if k.dtMs == dtMs and k.cols == cols: return i
  v.keys.add PlanKey(dtMs: dtMs, cols: cols)
  v.plans.add v.flow.compile(float32, dtMs, max(1, cols))
  v.plans.len - 1

proc plan*(v: ViewerSim, e: Entry): lent PipelinePlan[float32] =
  ## The current plan of entry `e`.
  v.plans[e.plan]

proc addEntry(v: var ViewerSim, c: sink PrefetchChunk[float32]) =
  let meta = chunkMeta(c)
  let p = v.planFor(meta.dtMs, c.samples.shape[1])
  var e = Entry(processed: zeros[float32](c.samples.shape[0],
    c.samples.shape[1]), plan: p, ticksRevision: -1)
  e.ticks = newSeq[float64](c.count)
  e.raw = c
  v.entries.add e

proc load*(v: var ViewerSim, f: SeismicFile, traces: Slice[int],
    chunkTraces = 64, firstSample = 0, cols = -1, workers = 2) =
  ## Prefetches `traces` (uniform chunks) into the raw cache, appending
  ## entries. The views in the chunks point into `f`'s keys.
  var p = newPrefetcher[float32](f, traces, chunkTraces = chunkTraces,
    firstSample = firstSample, cols = cols, workers = workers, uniform = true)
  defer: p.close()
  var c: PrefetchChunk[float32]
  while p.next(c):
    v.addEntry(move c)

proc loadSynthetic*(v: var ViewerSim, rows: TraceRows, data: Tensor[float32],
    firstSample = 0, failed: seq[int] = @[]) =
  ## Adds a chunk made in memory (synthetic keys), as the prefetcher would
  ## deliver it. `rows` must outlive the entry.
  var c = PrefetchChunk[float32](first: 0, count: rows.len,
    samples: data.clone(), failed: failed.len, failedRows: failed,
    firstSample: firstSample, rows: rows)
  v.addEntry(move c)

proc clearCache*(v: var ViewerSim) =
  ## Drops every entry (after `applySurvey` their views are stale).
  v.entries.setLen 0

proc recompile*(v: var ViewerSim) =
  ## Recompiles every plan from `flow` (after a slider change). Every
  ## processed buffer becomes stale: its generation is older.
  for i, k in v.keys:
    v.plans[i] = v.flow.compile(float32, k.dtMs, max(1, k.cols))

template edit*(v: var ViewerSim, body: untyped) =
  ## A slider change: `body` edits `v.flow`, then the plans are recompiled.
  body
  v.recompile

func stale*(v: ViewerSim, e: Entry): bool =
  ## The processed buffer was not made by the current plan.
  e.generation != v.plans[e.plan].generation

proc frame*(v: var ViewerSim) =
  ## One frame: reprocess every stale buffer from its raw chunk with the
  ## pool, tag it, then "draw" (check that every buffer is current and sum
  ## |processed| over the live rows).
  v.reprocessed = 0
  for e in v.entries.mitems:
    let plan = v.plans[e.plan].addr
    if e.generation != plan[].generation:
      let meta = chunkMeta(e.raw)
      v.pool.apply(plan[], e.raw.samples, e.processed, meta)
      e.generation = plan[].generation
      inc v.reprocessed
  var s = 0.0
  for e in v.entries:
    doAssert not v.stale(e), "drawing a stale buffer"
    let meta = chunkMeta(e.raw)
    let cols = e.processed.shape[1]
    for r in 0 ..< meta.len:
      if meta.isLive(r):
        for j in 0 ..< meta.liveSamples(r, cols):
          s += abs(float64(e.processed[r, j]))
  v.drawn = s

proc updateTicks*(v: var ViewerSim, f: SeismicFile, m: Moveout) =
  ## Fills every entry's moveout columns and tags them with `f`'s
  ## `keysRevision`.
  for e in v.entries.mitems:
    moveoutColumnsInto(chunkMeta(e.raw), m, e.ticks)
    e.ticksRevision = f.keysRevision

func ticksStale*(v: ViewerSim, f: SeismicFile): bool =
  ## Some entry's ticks predate `f`'s current keys.
  for e in v.entries:
    if e.ticksRevision != f.keysRevision: return true
  false

proc zoomWindow*(v: var ViewerSim, f: SeismicFile, traces: Slice[int],
    shown: Slice[int], marginScale = 1.0): Tensor[float32] =
  ## What a zoomed view of samples `shown` of `traces` draws: it decodes the
  ## window widened by the plan's `requiredMargin` on each side (clamped at
  ## sample 0), processes it, and returns the columns of `shown`. The plan
  ## for the widened width is compiled first, so its margin sets the decode.
  ## `marginScale` widens the margin (a viewer's choice; see the test).
  let dtMs = chunkMeta(f, traces).dtMs
  # The margin does not depend on the width; compile once to learn it.
  let m = int(ceil(marginScale * float64(v.flow.compile(float32, dtMs,
    shown.len).requiredMargin)))
  let first = max(0, shown.a - m)
  let cols = shown.b + m + 1 - first
  var p = newPrefetcher[float32](f, traces, chunkTraces = traces.len,
    firstSample = first, cols = cols, uniform = true)
  defer: p.close()
  var c: PrefetchChunk[float32]
  doAssert p.next(c)
  let meta = chunkMeta(c)
  doAssert meta.firstSample == first
  let pi = v.planFor(dtMs, cols)
  var processed = zeros[float32](c.count, cols)
  v.pool.apply(v.plans[pi], c.samples, processed, meta)
  result = processed[_, (shown.a - first) .. (shown.b - first)].clone()

proc peakAbs*(t: Tensor[float32]): float64 =
  ## max |t|.
  for x in t: result = max(result, abs(float64(x)))
