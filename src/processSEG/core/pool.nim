## `ProcessPool`: persistent worker threads that apply a `PipelinePlan`
## across the rows of a block (decision D6, the integration doc's
## "Threading model").
##
## ```nim
## var pool = newProcessPool(workers = 4)        # once: threads start here
## let plan = flow.compile(float32, dtMs = 2, samples = 2001)
## pool.apply(plan, raw, processed, meta)        # per frame or chunk; blocks
## pool.apply(plan, samples, meta)               # in place
## pool.correlateInto(corr, raw, correlated, meta) # vibroseis correlation
## pool.traceAmplitudeInto(raw, meta, AmplitudeParams(), amps) # amplitude QC
## pool.pickInto(picker, raw, meta, picks)       # first breaks
## ```
##
## **Workers.** The threads start once, in `newProcessPool`, sleep on a
## condition variable between jobs and are joined when the pool is
## destroyed, so a job costs a wake-up and a join on a counter, not a thread
## start. Each worker owns its `Workspace` and grows it to the plan's size on
## its first job (`reserve` is a no-op after that): the caller never handles
## workspaces.
##
## **A job.** `apply` validates its arguments on the caller's thread (it
## raises `ValueError` as `PipelinePlan.apply` does, before any sample
## changes) and returns when every worker is done. The workers take the rows
## in runs of about 256 KB of samples, each from a shared counter as it
## finishes its last one, so a block whose rows differ in cost (dead or
## short traces, a busy core) is still shared out evenly. The result is
## identical to the single-thread `apply`, bit for bit. A per-panel
## normalization needs the whole panel: its factor is computed by one
## worker over every row, then the next part of the plan runs on all
## workers again.
##
## **What crosses threads.** No managed value does: a tensor's storage is
## reference counted without atomics, and copying a seq (a plan holds some)
## allocates on the thread that copies it. `apply` passes raw pointers to the
## plan, the rows and the meta, and because it blocks until every worker is
## done, they outlive the work. Each worker's `Workspace` is its own thread's
## variable.
##
## **Callers.** One job runs at a time per pool; a second caller (from
## another thread, through a pointer to the pool) waits for the first to
## finish. A pool cannot be copied; move it. Destroying it stops and joins
## its workers, so it must not be destroyed while another thread is in
## `apply`. A viewer that wants to abandon a frame checks `plan.generation`
## before drawing.

import std/[atomics, locks]
import parseSEG
import rows, types, workspace
import ../pipeline
import ../vibroseis/correlate
import ../qc/[amplitude, firstbreak]
from ../fft/plan import fftLanes
from ../filter/iir import iirLanes

type
  JobFn = proc(ctx: pointer, worker: int, ws: var Workspace) {.nimcall,
      gcsafe, raises: [].}

  WorkerArg = object
    state: ptr PoolState
    index: int

  PoolState = object
    lock: Lock   # guards everything below
    wake: Cond   # a new job or stop, for the workers
    done: Cond   # pending reached 0, for the caller
    call: Lock   # one apply at a time
    jobId: int   # bumped for each job; workers run each id once
    pending: int # workers still on the current job
    stop: bool
    fn: JobFn
    ctx: pointer
    workers: int
    threads: ptr UncheckedArray[Thread[WorkerArg]]

  ProcessPool* = object
    ## Persistent workers that apply plans; see the module documentation.
    ## Build it with `newProcessPool`; it cannot be copied.
    state: ptr PoolState

proc `=destroy`*(p: ProcessPool) =
  ## Stops the workers (they finish no job: the pool is idle when destroyed)
  ## and joins them.
  if p.state != nil:
    let s = p.state
    acquire s.lock
    s.stop = true
    broadcast s.wake
    release s.lock
    for i in 0 ..< s.workers:
      joinThread(s.threads[i])
    deinitCond s.wake
    deinitCond s.done
    deinitLock s.lock
    deinitLock s.call
    deallocShared s.threads
    deallocShared s

proc `=wasMoved`*(p: var ProcessPool) =
  ## Leaves `p` without workers, so destroying it does nothing.
  p.state = nil

proc `=copy`*(dst: var ProcessPool, src: ProcessPool) {.error.}
  ## A pool cannot be copied (a compile error); move it.
proc `=dup`*(src: ProcessPool): ProcessPool {.error.}
  ## A pool cannot be copied (a compile error); move it.

proc workerLoop(a: WorkerArg) {.thread.} =
  let s = a.state
  var ws = initWorkspace()
  var seen = 0
  while true:
    acquire s.lock
    while not s.stop and s.jobId == seen:
      wait s.wake, s.lock
    if s.stop:
      release s.lock
      break
    seen = s.jobId
    let fn = s.fn
    let ctx = s.ctx
    release s.lock
    fn(ctx, a.index, ws)
    acquire s.lock
    dec s.pending
    if s.pending == 0:
      signal s.done
    release s.lock

proc newProcessPool*(workers = 4): ProcessPool =
  ## A pool of `workers` threads, started now. Raises `ValueError` for
  ## `workers < 1`, and `ResourceExhaustedError` when the OS refuses a thread
  ## (the threads already started are joined). A defect on a worker (a
  ## failed assertion, out of memory) ends the process, as on any thread.
  if workers < 1:
    raise newException(ValueError, "a pool needs workers ≥ 1, got " & $workers)
  let s = createShared(PoolState)
  initLock s.lock
  initLock s.call
  initCond s.wake
  initCond s.done
  s.workers = workers
  s.threads = cast[ptr UncheckedArray[Thread[WorkerArg]]](allocShared0(
    sizeof(Thread[WorkerArg]) * workers))
  result.state = s
  for i in 0 ..< workers:
    try:
      createThread(s.threads[i], workerLoop, WorkerArg(state: s, index: i))
    except ResourceExhaustedError:
      s.workers = i # =destroy joins only the threads that started
      raise

func workers*(p: ProcessPool): int {.inline, raises: [].} =
  ## The number of worker threads (0 for a pool moved from or never made).
  if p.state == nil: 0 else: p.state.workers

proc run(s: ptr PoolState, fn: JobFn, ctx: pointer) {.raises: [].} =
  ## Runs `fn` on every worker and waits for all of them. The caller holds
  ## `s.call`.
  acquire s.lock
  s.fn = fn
  s.ctx = ctx
  s.pending = s.workers
  inc s.jobId
  broadcast s.wake
  while s.pending > 0:
    wait s.done, s.lock
  release s.lock

proc runEmpty(ctx: pointer, worker: int, ws: var Workspace) {.nimcall,
    gcsafe, raises: [].} =
  discard

proc ping*(p: ProcessPool) =
  ## Runs an empty job on every worker and waits: the pool's per-job
  ## overhead, for benchmarks; also a way to check the workers are up. An
  ## `AssertionDefect` for a pool moved from or never made.
  doAssert p.state != nil, "a ProcessPool must come from newProcessPool"
  acquire p.state.call
  run(p.state, runEmpty, nil)
  release p.state.call

# ---------------------------------------------------------------- row runs --

type
  RowQueue = object
    ## The rows of a job, handed out in runs of `grain` rows to whichever
    ## worker asks next (review xE P2: one fixed range per worker ran a
    ## block whose first half was dead at half speed).
    next: Atomic[int]
    rows, grain: int

const runBytes = 256 * 1024 # of samples per run: the tile of `runPart`

proc reset(q: var RowQueue, rows, cols, sampleBytes: int, batch = fftLanes) =
  ## A queue of `rows` rows of `cols` samples of `sampleBytes` bytes each,
  ## in runs of whole batches of `batch` rows (an FFT's lanes by default).
  q.rows = rows
  q.grain = max(batch, runBytes div max(1, cols * sampleBytes) div batch * batch)
  q.next.store(0)

template eachRun(q: var RowQueue, span, body: untyped) =
  ## Runs `body` with `span`, a `Slice[int]` of rows, for each run this
  ## thread gets from `q`, until none is left.
  while true:
    let first = q.next.fetchAdd(q.grain)
    if first >= q.rows: break
    let span {.inject.} = first .. min(q.rows, first + q.grain) - 1
    body

# -------------------------------------------------------------------- apply --

type
  ApplyCtx[T: SampleType] = object
    plan: ptr PipelinePlan[T]
    src, dst: RowBlock[T] # src has no rows for an in-place apply
    meta: ChunkMeta
    part: int
    factor: float64
    queue: RowQueue

proc partJob[T](ctx: pointer, worker: int, ws: var Workspace) {.nimcall,
    gcsafe, raises: [].} =
  let c = cast[ptr ApplyCtx[T]](ctx)
  try:
    ws.reserve(c.plan[].workspaceSize)
    let fromSrc = c.part == 0 and c.src.len > 0
    c.queue.eachRun(span):
      let src = (if fromSrc: c.src.sub(span) else: RowBlock[T]())
      c.plan[].runPart(src, c.dst.sub(span), c.meta.sub(span), c.part, c.factor, ws)
  except ValueError as e:
    raiseAssert "checked by apply: " & e.msg

proc panelJob[T](ctx: pointer, worker: int, ws: var Workspace) {.nimcall,
    gcsafe, raises: [].} =
  if worker != 0: return
  let c = cast[ptr ApplyCtx[T]](ctx)
  try:
    ws.reserve(c.plan[].workspaceSize)
  except ValueError as e:
    raiseAssert e.msg
  let b = (if c.part == 0 and c.src.len > 0: c.src else: c.dst)
  c.factor = c.plan[].panelFactorOf(b, c.meta, c.part, ws)

proc applyBlocks[T](p: ProcessPool, plan: PipelinePlan[T], src,
    dst: RowBlock[T], meta: ChunkMeta) =
  doAssert p.state != nil, "a ProcessPool must come from newProcessPool"
  plan.checkApply(src, dst, meta)
  var ctx = ApplyCtx[T](plan: addr plan, src: src, dst: dst, meta: meta)
  acquire p.state.call
  for part in 0 ..< plan.partCount:
    ctx.part = part
    ctx.factor = NaN
    if plan.partNeedsPanel(part):
      run(p.state, panelJob[T], addr ctx)
    ctx.queue.reset(dst.len, dst.cols, sizeof(T))
    run(p.state, partJob[T], addr ctx)
  release p.state.call

proc apply*[T: SampleType](p: ProcessPool, plan: PipelinePlan[T],
    b: RowBlock[T], meta: ChunkMeta) =
  ## Runs `plan` on the block `b` in place, its rows split across the
  ## workers; blocks until they are done. Raises `ValueError`, before any
  ## sample changes, as `PipelinePlan.apply` does.
  p.applyBlocks(plan, RowBlock[T](), b, meta)

proc apply*[T: SampleType](p: ProcessPool, plan: PipelinePlan[T], src,
    dst: RowBlock[T], meta: ChunkMeta) =
  ## Runs `plan` on `src` into `dst` (the same shape, not overlapping unless
  ## the same block), its rows split across the workers; blocks until they
  ## are done. Raises `ValueError` as the in-place form, and for a shape
  ## mismatch.
  if src.len != dst.len or src.cols != dst.cols:
    raise newException(ValueError, "src is " & $src.len & " × " & $src.cols &
      ", dst " & $dst.len & " × " & $dst.cols)
  p.applyBlocks(plan, src, dst, meta)

proc apply*[T: SampleType](p: ProcessPool, plan: PipelinePlan[T],
    samples: var Tensor[T], meta: ChunkMeta) =
  ## `apply` in place on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  p.apply(plan, rowBlock(samples), meta)

proc apply*[T: SampleType](p: ProcessPool, plan: PipelinePlan[T],
    src: Tensor[T], dst: var Tensor[T], meta: ChunkMeta) =
  ## `apply` from `src` into `dst`, tensors of the same shape (C-contiguous;
  ## raises `ValueError` otherwise). `dst` is not resized.
  meta.checkShape(src)
  meta.checkShape(dst)
  p.apply(plan, rowBlock(src), rowBlock(dst), meta)

# -------------------------------------------------------------- correlation --

type
  CorrCtx[T: SampleType] = object
    plan: ptr CorrelationPlan[T]
    src, dst: RowBlock[T]
    meta: ChunkMeta
    queue: RowQueue

proc corrJob[T](ctx: pointer, worker: int, ws: var Workspace) {.nimcall,
    gcsafe, raises: [].} =
  let c = cast[ptr CorrCtx[T]](ctx)
  try:
    ws.reserve(c.plan[].workspaceSize)
    c.queue.eachRun(span):
      c.plan[].correlateInto(c.src.sub(span), c.dst.sub(span), c.meta.sub(span), ws)
  except ValueError as e:
    raiseAssert "checked by correlateInto: " & e.msg

proc correlateInto*[T: SampleType](p: ProcessPool, plan: CorrelationPlan[T],
    src, dst: RowBlock[T], meta: ChunkMeta) =
  ## Vibroseis correlation of `src` into `dst` with the rows split across the
  ## workers; blocks until they are done. Raises `ValueError` as
  ## `CorrelationPlan.correlateInto` does, before any sample changes.
  doAssert p.state != nil, "a ProcessPool must come from newProcessPool"
  plan.checkCorrelate(src, dst, meta) # on the caller's thread
  var ctx = CorrCtx[T](plan: addr plan, src: src, dst: dst, meta: meta)
  ctx.queue.reset(src.len, src.cols, sizeof(T))
  acquire p.state.call
  run(p.state, corrJob[T], addr ctx)
  release p.state.call

proc correlateInto*[T: SampleType](p: ProcessPool, plan: CorrelationPlan[T],
    src: Tensor[T], dst: var Tensor[T], meta: ChunkMeta) =
  ## The tensor form of the pooled correlation.
  meta.checkShape(src)
  meta.checkShape(dst)
  p.correlateInto(plan, rowBlock(src), rowBlock(dst), meta)

# ------------------------------------------------------------ amplitude QC --

type
  AmpCtx[T: SampleType] = object
    params: AmplitudeParams
    src: RowBlock[T]
    meta: ChunkMeta
    dst: ptr UncheckedArray[TraceAmplitude]
    queue: RowQueue

proc ampJob[T](ctx: pointer, worker: int, ws: var Workspace) {.nimcall,
    gcsafe, raises: [].} =
  let c = cast[ptr AmpCtx[T]](ctx)
  try:
    ws.reserve(traceAmplitudeWorkspace(c.src.cols))
    c.queue.eachRun(span):
      traceAmplitudeInto(c.src.sub(span), c.meta.sub(span), c.params,
        c.dst.toOpenArray(span.a, span.b), ws)
  except ValueError as e:
    raiseAssert "checked by traceAmplitudeInto: " & e.msg

proc traceAmplitudeInto*[T: SampleType](p: ProcessPool, b: RowBlock[T],
    meta: ChunkMeta, params: AmplitudeParams, dst: var openArray[TraceAmplitude]) =
  ## `qc/amplitude.traceAmplitudeInto` with the rows split across the
  ## workers, each with its own workspace (`traceAmplitudeWorkspace`,
  ## reserved by the worker); blocks until they are done. The same results
  ## and errors (raised on the caller's thread, before any work).
  doAssert p.state != nil, "a ProcessPool must come from newProcessPool"
  meta.checkShape(b)
  params.validate
  if dst.len != b.len:
    raise newException(ValueError, "dst holds " & $dst.len & " values, the block has " &
      $b.len & " rows")
  if b.len == 0: return
  var ctx = AmpCtx[T](params: params, src: b, meta: meta,
    dst: cast[ptr UncheckedArray[TraceAmplitude]](addr dst[0]))
  ctx.queue.reset(b.len, b.cols, sizeof(T))
  acquire p.state.call
  run(p.state, ampJob[T], addr ctx)
  release p.state.call

proc traceAmplitudeInto*[T: SampleType](p: ProcessPool, samples: Tensor[T],
    meta: ChunkMeta, params: AmplitudeParams, dst: var openArray[TraceAmplitude]) =
  ## The tensor form of the pooled amplitude QC.
  meta.checkShape(samples)
  p.traceAmplitudeInto(rowBlock(samples), meta, params, dst)

# ------------------------------------------------------------ first breaks --

type
  PickCtx[T: SampleType] = object
    plan: ptr FirstBreakPlan[T]
    src: RowBlock[T]
    meta: ChunkMeta
    dst: ptr UncheckedArray[FirstBreakPick]
    queue: RowQueue

proc pickJob[T](ctx: pointer, worker: int, ws: var Workspace) {.nimcall,
    gcsafe, raises: [].} =
  let c = cast[ptr PickCtx[T]](ctx)
  try:
    ws.reserve(c.plan[].workspaceSize)
    c.queue.eachRun(span):
      c.plan[].pickInto(c.src.sub(span), c.meta.sub(span),
        c.dst.toOpenArray(span.a, span.b), ws)
  except ValueError as e:
    raiseAssert "checked by pickInto: " & e.msg

proc pickInto*[T: SampleType](p: ProcessPool, plan: FirstBreakPlan[T], b: RowBlock[T],
    meta: ChunkMeta, dst: var openArray[FirstBreakPick]) =
  ## `FirstBreakPlan.pickInto` (`qc/firstbreak`) with the rows split across
  ## the workers, each with its own workspace; blocks until they are done.
  ## The same picks, bit for bit, and the same errors (`checkPick`, raised
  ## on the caller's thread, before any work).
  doAssert p.state != nil, "a ProcessPool must come from newProcessPool"
  plan.checkPick(b, meta, dst.len)
  if b.len == 0: return
  var ctx = PickCtx[T](plan: addr plan, src: b, meta: meta,
    dst: cast[ptr UncheckedArray[FirstBreakPick]](addr dst[0]))
  # Runs of whole IIR batches: a run's band-pass then fills every lane
  # (dead rows aside), where 8-row runs filled a third of them.
  ctx.queue.reset(b.len, b.cols, sizeof(T), iirLanes)
  acquire p.state.call
  run(p.state, pickJob[T], addr ctx)
  release p.state.call

proc pickInto*[T: SampleType](p: ProcessPool, plan: FirstBreakPlan[T], samples: Tensor[T],
    meta: ChunkMeta, dst: var openArray[FirstBreakPick]) =
  ## The tensor form of the pooled first-break picking.
  meta.checkShape(samples)
  p.pickInto(plan, rowBlock(samples), meta, dst)
