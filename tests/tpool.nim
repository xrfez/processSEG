## ProcessPool (core/pool): results identical to the single-thread apply for
## 1–16 workers (in place, src → dst, per-panel parts, rows of uneven cost),
## many back-to-back jobs, idle start-up and shutdown, moves, two callers
## sharing one pool, argument errors and no allocation in a steady state.
##
## Fail-first (averaged spectrogram), then restored:
## `rowFinite` forced true: `pool_average_drops_nan` kept 20 rows (float32,
## 1 worker) instead of 18; `pool_average_all_dropped` kept 18 instead of 0.
## Batch width set to the worker count: `pool_average_same_bits` differed at
## float32, 15 rows, 2 workers, bin 1.
## `pool_average_drops_extreme_float64` failed before the pool called
## `averagedRow`: workers 1 used 4 instead of 2 (the 1e200 and 1e-200 rows
## were kept).
## f-k amplitude matches `FkPlan.fkAmplitudeInto` bit for bit.
##
## Fail-first: `fk_pool_matches` failed while the time pass wrote lane 0 of
## every trace (`laneX[j * fftLanes + 0]`): `bad < 0` workers 1 float32
## traces 7 samples 32 differ at 0. `fk_pool_hypot` failed while pass 2
## always called `modulusInto`: hypot path differs at 17, pool
## 1.0000000000000002e+150 (fcNormal), serial 1e+150 (fcNormal).
## `fk_pool_errors` failed while a 1-live-row window did not raise:
## `serialMsg.len > 0 and serialMsg == poolMsg`.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

proc main*() =
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
                  muteTopOp(lmoMute(2500), taperMs = 20),
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

  block rows_of_uneven_cost:
    # The first half of the block dead and a third of the rest short: the
    # workers take runs of rows as they finish, so every worker count gives the
    # single-thread result, on a block of several runs (600 × 1001 float32 is
    # ten runs of 64 rows).
    var g = noisyGather(600, 1001, seed = 11)
    for r in 0..<300: g.keys[r].status = tsDead
    for r in countup(300, 599, 3): g.setLive(r, 200 + r)
    let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
    let plan = flow().compile(float32, 2, 1001)
    var ws = initWorkspace()
    ws.reserve(plan)
    var want = zeros[float32](600, 1001)
    plan.apply(g.data, want, meta, ws)
    for workers in [2, 5, 16]:
      var pool = newProcessPool(workers)
      var dst = zeros[float32](600, 1001)
      pool.apply(plan, g.data, dst, meta)
      doAssert maxDiff(dst, want) == 0, $workers & " workers"

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
      doAssert pool.workers == 1 + i mod 8 # a worker reached its first wait
    for i in 0..<20:
      var pool = newProcessPool(4)
      pool.ping

  block moves:
    var a = newProcessPool(2)
    var b = move a
    b.ping
    doAssert a.workers == 0 and b.workers == 2
    doAssertRaises AssertionDefect: a.ping
    var never: ProcessPool
    doAssert never.workers == 0
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

  block float64_data:
    var offs = newSeq[float64](37)
    var g = cmpGather[float64](0, offs, 700, dtMs = 2)
    for r in 0..<37:
      var x = newSeq[float64](700)
      x.addNoise(1, seed = 40 + r)
      x.addDc(2)
      for j in 0..<700: g.data[r, j] = x[j] * float64(1 + r)
    let meta = initChunkMeta(g.rows, dtMs = 2)
    let plan = flow().compile(float64, 2, 700)
    var ws = initWorkspace()
    ws.reserve(plan)
    var want = zeros[float64](37, 700)
    plan.apply(g.data, want, meta, ws)
    var pool = newProcessPool(3)
    var got = zeros[float64](37, 700)
    pool.apply(plan, g.data, got, meta)
    for r in 0..<37:
      for j in 0..<700: doAssert got[r, j] == want[r, j]

  proc frameClose(got, want: openArray[float64], frames, bins: int): string =
    ## Empty when `got` is within 1e-12 of `want`'s frame peak on every bin.
    for f in 0 ..< frames:
      var peak = 0.0
      for k in 0 ..< bins:
        peak = max(peak, abs(want[f * bins + k]))
      let tol = 1e-12 * peak
      for k in 0 ..< bins:
        let d = abs(got[f * bins + k] - want[f * bins + k])
        if d > tol:
          return "frame " & $f & " bin " & $k & " diff " & $d & " peak " & $peak
    ""

  proc toneGather[T: SampleType](rows, cols: int): SynthGather[T] =
    var offs = newSeq[float64](rows)
    result = cmpGather[T](0, offs, cols, dtMs = 2)
    for r in 0 ..< rows:
      let a = 0.3 + 0.17 * float64(r mod 11)
      let phase = 0.05 * float64(r)
      for j in 0 ..< cols:
        result.data[r, j] = T(a * sin(2 * PI * (3 + float64(r mod 5)) *
          float64(j) * 0.002 + phase))

  proc checkPoolAverage[T: SampleType]() =
    const cols = 80
    let sg = initSpectrogramPlan(T, windowLen = 32, hop = 8, dtMs = 2.0, samples = cols)
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var pools: array[4, ProcessPool]
    let counts = [1, 2, 3, 8]
    for i, w in counts: pools[i] = newProcessPool(w)
    for rows in [1, 15, 16, 17, 200]:
      var g = toneGather[T](rows, cols)
      if rows >= 17:
        g.keys[2].status = tsDead
        g.setLive(4, 20)
      let meta = initChunkMeta(g.rows, dtMs = 2)
      var want = newSeq[float64](sg.frames * sg.bins)
      let used = sg.averageSpectrogramInto(g.data, meta, want, ws)
      var refGot: seq[float64] = @[]
      var refUsed = 0
      for i, w in counts:
        var got = newSeq[float64](sg.frames * sg.bins)
        let n = pools[i].averageSpectrogramInto(sg, g.data, meta, got)
        doAssert n == used, $T & " rows " & $rows & " workers " & $w &
          " used " & $n & " vs " & $used
        let note = frameClose(got, want, sg.frames, sg.bins)
        doAssert note.len == 0, $T & " rows " & $rows & " w " & $w & " " & note
        if i == 0:
          refGot = got
          refUsed = n
        else:
          doAssert n == refUsed
          for k in 0 ..< got.len:
            doAssert got[k] == refGot[k], $T & " rows " & $rows & " workers " & $w &
              " bin " & $k

  block pool_average_same_bits:
    # Workers 1, 2, 3, 8 match each other bit for bit, and the serial form
    # within 1e-12 of the frame peak. Row counts sit on both sides of the
    # batch of 16, including one wave and several.
    checkPoolAverage[float32]()
    checkPoolAverage[float64]()

  proc checkDropped[T: SampleType]() =
    const cols = 64
    const rows = 20
    let sg = initSpectrogramPlan(T, windowLen = 32, hop = 8, dtMs = 2.0, samples = cols)
    var g = toneGather[T](rows, cols)
    g.data[1, 8] = T(NaN)
    g.data[16, 3] = T(NaN)
    let meta = initChunkMeta(g.rows, dtMs = 2)
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var want = newSeq[float64](sg.frames * sg.bins)
    let used = sg.averageSpectrogramInto(g.data, meta, want, ws)
    doAssert used == rows - 2, $T & " serial kept " & $used
    var refGot: seq[float64] = @[]
    for i, w in [1, 2, 3, 8]:
      var pool = newProcessPool(w)
      var got = newSeq[float64](sg.frames * sg.bins)
      let n = pool.averageSpectrogramInto(sg, rowBlock(g.data), meta, got)
      doAssert n == used, $T & " workers " & $w & " kept " & $n
      let note = frameClose(got, want, sg.frames, sg.bins)
      doAssert note.len == 0, $T & " " & note
      if i == 0: refGot = got
      else:
        for k in 0 ..< got.len:
          doAssert got[k] == refGot[k], $T & " workers " & $w

  block pool_average_drops_nan:
    # Two NaN rows are dropped; the count matches the serial form.
    checkDropped[float32]()
    checkDropped[float64]()

  proc checkAllDropped[T: SampleType]() =
    const cols = 48
    const rows = 18
    let sg = initSpectrogramPlan(T, windowLen = 32, hop = 8, dtMs = 2.0, samples = cols)
    var g = toneGather[T](rows, cols)
    for r in 0 ..< rows: g.data[r, 4] = T(NaN)
    let meta = initChunkMeta(g.rows, dtMs = 2)
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var want = newSeq[float64](sg.frames * sg.bins)
    doAssert sg.averageSpectrogramInto(g.data, meta, want, ws) == 0
    for v in want: doAssert v == 0.0
    for w in [1, 8]:
      var pool = newProcessPool(w)
      var got = newSeq[float64](sg.frames * sg.bins)
      for k in 0 ..< got.len: got[k] = 1.0
      let n = pool.averageSpectrogramInto(sg, g.data, meta, got)
      doAssert n == 0, $T & " workers " & $w & " kept " & $n
      for v in got: doAssert v == 0.0, $T & " " & $w

  block pool_average_all_dropped:
    checkAllDropped[float32]()
    checkAllDropped[float64]()

  block pool_average_errors:
    let sg = initSpectrogramPlan(float32, windowLen = 32, hop = 8, dtMs = 2.0, samples = 64)
    var g = toneGather[float32](6, 64)
    let meta = initChunkMeta(g.rows, dtMs = 2)
    var pool = newProcessPool(2)
    var amp = newSeq[float64](sg.frames * sg.bins)
    amp[0] = 1.5
    let other = initChunkMeta(g.rows, dtMs = 4)
    doAssertRaises ValueError:
      discard pool.averageSpectrogramInto(sg, g.data, other, amp)
    doAssert amp[0] == 1.5
    let short = initSpectrogramPlan(float32, windowLen = 32, hop = 8, dtMs = 2.0, samples = 32)
    var shortAmp = newSeq[float64](short.frames * short.bins)
    shortAmp[0] = 1.5
    doAssertRaises ValueError:
      discard pool.averageSpectrogramInto(short, g.data, meta, shortAmp)
    doAssert shortAmp[0] == 1.5
    var bad = newSeq[float64](3)
    doAssertRaises AssertionDefect:
      discard pool.averageSpectrogramInto(sg, g.data, meta, bad)
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var want = newSeq[float64](sg.frames * sg.bins)
    let used = sg.averageSpectrogramInto(g.data, meta, want, ws, rows = 1 .. 3)
    var got = newSeq[float64](sg.frames * sg.bins)
    let n = pool.averageSpectrogramInto(sg, g.data, meta, got, rows = 1 .. 3)
    doAssert n == used and n > 0
    doAssert frameClose(got, want, sg.frames, sg.bins).len == 0
    pool.ping

  block pool_average_drops_extreme_float64:
    # Float64 rows [1e200, 1.0, 1e-200, 1.0]: the two extremes leave the
    # safe range, so used is 2 and the bits match a pool run of the two
    # 1.0 rows alone. The count matches the serial form.
    const cols = 16
    let sg = initSpectrogramPlan(float64, windowLen = 16, hop = 16, dtMs = 1.0,
      samples = cols, nfft = 16)
    let scales = [1e200, 1.0, 1e-200, 1.0]
    var g = cmpGather[float64](0, newSeq[float64](scales.len), cols, 1.0)
    for r in 0 ..< scales.len:
      for j in 0 ..< cols: g.data[r, j] = scales[r]
    let meta = initChunkMeta(g.rows, dtMs = 1.0)
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var serialAmp = newSeq[float64](sg.frames * sg.bins)
    let serialUsed = sg.averageSpectrogramInto(g.data, meta, serialAmp, ws)
    var alone = cmpGather[float64](0, newSeq[float64](2), cols, 1.0)
    for r in 0 ..< 2:
      for j in 0 ..< cols: alone.data[r, j] = 1.0
    let aloneMeta = initChunkMeta(alone.rows, dtMs = 1.0)
    for w in [1, 2]:
      var pool = newProcessPool(w)
      var got = newSeq[float64](sg.frames * sg.bins)
      let n = pool.averageSpectrogramInto(sg, g.data, meta, got)
      doAssert n == 2, "workers " & $w & " used " & $n
      doAssert n == serialUsed, "workers " & $w & " used " & $n &
        " serial " & $serialUsed
      var refAmp = newSeq[float64](sg.frames * sg.bins)
      let nAlone = pool.averageSpectrogramInto(sg, alone.data, aloneMeta, refAmp)
      doAssert nAlone == 2, "workers " & $w & " alone used " & $nAlone
      doAssert refAmp[0] != 0.0, "workers " & $w & " 1.0 spectrum is zero"
      for k in 0 ..< got.len:
        doAssert got[k] == refAmp[k], "workers " & $w & " bin " & $k &
          " " & $got[k] & " vs " & $refAmp[k]
  proc firstDiff(a, b: openArray[float64]): int =
    result = -1
    for i in 0 ..< a.len:
      if cast[uint64](a[i]) != cast[uint64](b[i]): return i

  proc fillFk[T](g: var SynthGather[T]) =
    let cols = g.data.shape[1]
    for r in 0 ..< g.data.shape[0]:
      for j in 0 ..< cols:
        g.data[r, j] = T(sin(0.17 * float64(j) + 0.31 * float64(r)) +
          0.05 * float64(r - j))

  proc checkFk[T](pool: ProcessPool, traces, samples, firstCol: int,
      rows: Slice[int], deadAt = -1, shortAt = -1) =
    let cols = firstCol + samples
    var g = cmpGather[T](0, newSeq[float64](traces + 1), cols, dtMs = 2)
    fillFk(g)
    g.keys[0].status = tsDead
    if deadAt > 0: g.keys[deadAt].status = tsDead
    if shortAt > 0: g.setLive(shortAt, firstCol)
    let meta = initChunkMeta(g.rows, dtMs = 2)
    let entered = fkRows(meta, cols, firstCol, samples, rows)
    let plan = initFkPlan(T, 2, traces = entered, samples = samples, taperFraction = 0.1)
    var ws = initWorkspace()
    ws.reserve(plan)
    var want = newSeq[float64](plan.nK * plan.nFreq)
    var got = newSeq[float64](want.len)
    for i in 0 ..< want.len:
      want[i] = -1
      got[i] = -1
    let nSer = plan.fkAmplitudeInto(g.data, meta, firstCol, want, ws, rows)
    let nPool = pool.fkAmplitudeInto(plan, g.data, meta, firstCol, got, rows)
    doAssert nSer == nPool and nPool == entered
    let bad = firstDiff(want, got)
    doAssert bad < 0, "workers " & $pool.workers & " " & $T & " traces " &
      $traces & " samples " & $samples & " differ at " & $bad
    for i in 0 ..< got.len: got[i] = -1
    let nBlock = pool.fkAmplitudeInto(plan, rowBlock(g.data), meta, firstCol, got, rows)
    doAssert nBlock == entered
    doAssert firstDiff(want, got) < 0

  block fk_pool_matches:
    for workers in [1, 2, 3, 8]:
      var pool = newProcessPool(workers)
      for samples in [32, 64]:
        for traces in [7, 8, 9, 17]:
          checkFk[float32](pool, traces, samples, 0, 0 .. int.high)
          checkFk[float64](pool, traces, samples, 0, 0 .. int.high)
      checkFk[float64](pool, 9, 32, 4, 0 .. int.high, shortAt = 3)
      checkFk[float32](pool, 9, 40, 0, 1 .. 8, deadAt = 2)
      checkFk[float64](pool, 17, 2048, 0, 0 .. int.high)
      checkFk[float32](pool, 200, 512, 0, 0 .. int.high)
      checkFk[float64](pool, 200, 512, 0, 0 .. int.high)

  block fk_pool_hypot:
    const cols = 32
    var g = cmpGather[float64](0, newSeq[float64](8), cols, dtMs = 2)
    fillFk(g)
    g.data[2, 3] = 1e150
    let meta = initChunkMeta(g.rows, dtMs = 2)
    let plan = initFkPlan(float64, 2, traces = 8, samples = cols, taperFraction = 0)
    var ws = initWorkspace()
    ws.reserve(plan)
    var want = newSeq[float64](plan.nK * plan.nFreq)
    var pool = newProcessPool(2)
    var got = newSeq[float64](want.len)
    doAssert plan.fkAmplitudeInto(g.data, meta, 0, want, ws) == 8
    doAssert pool.fkAmplitudeInto(plan, g.data, meta, 0, got) == 8
    let bad = firstDiff(want, got)
    doAssert bad < 0, "hypot path differs at " & $bad & " pool " & $got[bad] &
      " (" & $classify(got[bad]) & ") serial " & $want[bad] &
      " (" & $classify(want[bad]) & ")"
    for a in got:
      doAssert classify(a) notin {fcInf, fcNegInf, fcNan}
    g.data[2, 3] = 0
    g.data[1, 0] = 1e150
    for i in 0 ..< want.len:
      want[i] = -1
      got[i] = -1
    let win = cols - 8
    let plan2 = initFkPlan(float64, 2, traces = 8, samples = win, taperFraction = 0)
    ws.reserve(plan2)
    want.setLen(plan2.nK * plan2.nFreq)
    got.setLen(want.len)
    doAssert plan2.fkAmplitudeInto(g.data, meta, 8, want, ws) == 8
    doAssert pool.fkAmplitudeInto(plan2, g.data, meta, 8, got) == 8
    doAssert firstDiff(want, got) < 0

  block fk_pool_errors:
    var g = cmpGather[float64](0, newSeq[float64](6), 32, dtMs = 2)
    fillFk(g)
    for r in 1 .. 5: g.keys[r].status = tsDead
    let meta = initChunkMeta(g.rows, dtMs = 2)
    doAssert fkRows(meta, 32, 0, 32) == 1
    let plan = initFkPlan(float64, 2, traces = 4, samples = 32)
    var ws = initWorkspace()
    ws.reserve(plan)
    var amp = newSeq[float64](plan.nK * plan.nFreq)
    var pool = newProcessPool(2)
    var serialMsg, poolMsg = ""
    try: discard plan.fkAmplitudeInto(g.data, meta, 0, amp, ws)
    except ValueError as e: serialMsg = e.msg
    try: discard pool.fkAmplitudeInto(plan, g.data, meta, 0, amp)
    except ValueError as e: poolMsg = e.msg
    doAssert serialMsg.len > 0 and serialMsg == poolMsg
    try: discard plan.fkAmplitudeInto(g.data, meta, -1, amp, ws)
    except ValueError as e: serialMsg = e.msg
    try: discard pool.fkAmplitudeInto(plan, g.data, meta, -1, amp)
    except ValueError as e: poolMsg = e.msg
    doAssert serialMsg.len > 0 and serialMsg == poolMsg
    let other = initChunkMeta(g.rows, dtMs = 4)
    try: discard plan.fkAmplitudeInto(g.data, other, 0, amp, ws)
    except ValueError as e: serialMsg = e.msg
    try: discard pool.fkAmplitudeInto(plan, g.data, other, 0, amp)
    except ValueError as e: poolMsg = e.msg
    doAssert serialMsg.len > 0 and serialMsg == poolMsg
    var short = newSeq[float64](3)
    doAssertRaises AssertionDefect:
      discard pool.fkAmplitudeInto(plan, g.data, meta, 0, short)
    var dead: ProcessPool
    doAssertRaises AssertionDefect:
      discard dead.fkAmplitudeInto(plan, g.data, meta, 0, amp)

when isMainModule: main()
