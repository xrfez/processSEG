## ProcessPool (core/pool): results identical to the single-thread apply for
## 1–8 workers (in place, src → dst, per-panel parts), many back-to-back
## jobs, idle start-up and shutdown, moves, two callers sharing one pool,
## argument errors and no allocation in a steady state.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

proc noisyGather(rows, cols: int, seed: int64): SynthGather[float32] =
  var offs = newSeq[float64](rows)
  for i in 0..<rows: offs[i] = 25.0 * float64(i)
  result = cmpGather[float32](0, offs, cols, dtMs = 2)
  for r in 0..<rows:
    var x = newSeq[float32](cols)
    x.addNoise(1, seed = seed + r)
    x.addSinusoid(2, 25, amp = 3)
    x.addDc(5)
    for j in 0..<cols: result.data[r, j] = x[j] * float32(1 + r mod 5)
  if rows > 40:
    result.keys[3].status = tsDead
    result.setLive(5, 300)
    result.setLive(37, 0)

let failed = @[7, 30]

proc flow(): Pipeline =
  initPipeline([demeanOp(),
                bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                normalizeOp(normRms, perPanel, target = 2),
                agcOp(windowMs = 300),
                topMuteOp(lmoMute(2500), taperMs = 20),
                normalizeOp(normPercentile, perPanel, percentile = 95),
                gainDbOp(-3)])

proc maxDiff(a, b: Tensor[float32]): float64 =
  for r in 0..<a.shape[0]:
    for j in 0..<a.shape[1]:
      result = max(result, abs(float64(a[r, j]) - float64(b[r, j])))

block identical_to_single_thread:
  var g = noisyGather(101, 1001, seed = 3)
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  for f in [flow(), initPipeline([normalizeOp(normMax, perPanel)]), initPipeline()]:
    let plan = f.compile(float32, 2, 1001)
    var ws = initWorkspace()
    ws.reserve(plan)
    var want = zeros[float32](101, 1001)
    plan.apply(g.data, want, meta, ws)
    for workers in [1, 2, 3, 4, 8]:
      var pool = newProcessPool(workers)
      doAssert pool.workers == workers
      var dst = zeros[float32](101, 1001)
      pool.apply(plan, g.data, dst, meta)
      doAssert maxDiff(dst, want) == 0, $workers & " workers"
      var inPlace = g.data.clone()
      pool.apply(plan, inPlace, meta)
      doAssert maxDiff(inPlace, want) == 0, $workers & " workers, in place"

block fewer_rows_than_workers:
  var g = noisyGather(3, 200, seed = 4)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let plan = flow().compile(float32, 2, 200)
  var ws = initWorkspace()
  ws.reserve(plan)
  var want = g.data.clone()
  plan.apply(want, meta, ws)
  var pool = newProcessPool(8)
  var got = g.data.clone()
  pool.apply(plan, got, meta)
  doAssert maxDiff(got, want) == 0
  var none = noisyGather(0, 200, seed = 4)
  var empty = none.data.clone()
  pool.apply(plan, empty, initChunkMeta(none.rows, dtMs = 2))

block many_jobs_back_to_back:
  var g = noisyGather(16, 64, seed = 5)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let plan = initPipeline([scaleOp(1.0001)]).compile(float32, 2, 64)
  var pool = newProcessPool(4)
  var want = g.data.clone()
  var ws = initWorkspace()
  for i in 0..<2000:
    pool.apply(plan, g.data, meta)
    plan.apply(want, meta, ws)
    pool.ping
  doAssert maxDiff(g.data, want) == 0

block idle_start_and_shutdown:
  for i in 0..<50:
    var pool = newProcessPool(1 + i mod 8) # destroyed at once, maybe before
    doAssert pool.workers == 1 + i mod 8   # a worker reached its first wait
  for i in 0..<20:
    var pool = newProcessPool(4)
    pool.ping

block moves:
  var a = newProcessPool(2)
  var b = move a
  b.ping
  var pools: seq[ProcessPool]
  pools.add newProcessPool(2)
  pools.add move b
  for p in pools: p.ping
  pools.setLen(0)

type Caller = object
  pool: ptr ProcessPool
  plan: ptr PipelinePlan[float32]
  src, dst: RowBlock[float32]
  meta: ChunkMeta

proc callerLoop(c: Caller) {.thread.} =
  {.cast(gcsafe).}:
    for i in 0..<50:
      try:
        c.pool[].apply(c.plan[], c.src, c.dst, c.meta)
      except ValueError as e:
        raiseAssert e.msg

block two_callers_share_a_pool:
  var g1 = noisyGather(64, 500, seed = 6)
  var g2 = noisyGather(64, 500, seed = 7)
  let m1 = initChunkMeta(g1.rows, dtMs = 2, failed = failed)
  let m2 = initChunkMeta(g2.rows, dtMs = 2)
  let plan = flow().compile(float32, 2, 500)
  var ws = initWorkspace()
  ws.reserve(plan)
  var w1 = zeros[float32](64, 500)
  var w2 = zeros[float32](64, 500)
  plan.apply(g1.data, w1, m1, ws)
  plan.apply(g2.data, w2, m2, ws)
  var pool = newProcessPool(4)
  var d1 = zeros[float32](64, 500)
  var d2 = zeros[float32](64, 500)
  var threads: array[2, Thread[Caller]]
  createThread(threads[0], callerLoop, Caller(pool: addr pool, plan: addr plan,
    src: rowBlock(g1.data), dst: rowBlock(d1), meta: m1))
  createThread(threads[1], callerLoop, Caller(pool: addr pool, plan: addr plan,
    src: rowBlock(g2.data), dst: rowBlock(d2), meta: m2))
  joinThreads(threads)
  doAssert maxDiff(d1, w1) == 0 and maxDiff(d2, w2) == 0

block argument_errors:
  doAssertRaises ValueError: discard newProcessPool(0)
  var g = noisyGather(10, 100, seed = 8)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var pool = newProcessPool(2)
  let orig = g.data.clone()
  let narrow = initPipeline([demeanOp()]).compile(float32, 2, 50)
  doAssertRaises ValueError: pool.apply(narrow, g.data, meta)
  let otherDt = initPipeline([demeanOp()]).compile(float32, 1, 100)
  doAssertRaises ValueError: pool.apply(otherDt, g.data, meta)
  var wrong = zeros[float32](10, 99)
  let ok = initPipeline([demeanOp()]).compile(float32, 2, 100)
  doAssertRaises ValueError: pool.apply(ok, g.data, wrong, meta)
  doAssert maxDiff(g.data, orig) == 0
  pool.ping # still usable

block no_allocation_in_steady_state:
  var g = noisyGather(101, 1001, seed = 9)
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  let plan = flow().compile(float32, 2, 1001)
  var pool = newProcessPool(4)
  var dst = zeros[float32](101, 1001)
  pool.apply(plan, g.data, dst, meta) # workers grow their workspaces
  doAssert allocsIn(pool.apply(plan, g.data, dst, meta)) == 0
  doAssert allocsIn(pool.apply(plan, dst, meta)) == 0
  doAssert allocsIn(pool.ping) == 0
