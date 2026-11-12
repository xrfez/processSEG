## The pipeline (pipeline): builders and editing, compile-time validation,
## apply against the same ops run one by one (tiles, per-panel parts, dead,
## failed and short rows), src → dst, margins, generations, no allocation,
## and one plan shared by pointer across 4 threads.

import std/[assertions, math, strutils]
import processSEG
import utils/[allocs, synth]

proc noisyGather[T](rows, cols: int, seed: int64): SynthGather[T] =
  var offs = newSeq[float64](rows)
  for i in 0..<rows: offs[i] = 25.0 * float64(i)
  result = cmpGather[T](0, offs, cols, dtMs = 2)
  for r in 0..<rows:
    var x = newSeq[T](cols)
    x.addNoise(1, seed = seed + r)
    x.addSinusoid(2, 25, amp = 3)
    x.addDc(5)
    x.addDrift(2, 4)
    for j in 0..<cols: result.data[r, j] = x[j] * T(1 + r mod 7)

proc mark[T](g: var SynthGather[T]) =
  ## Dead, auxiliary and short rows, spread over several tiles.
  g.keys[3].status = tsDead
  g.keys[40].status = tsAuxiliary
  g.setLive(5, 700)
  g.setLive(77, 1)
  g.setLive(90, 0)

let failed = @[7, 64, 65]

proc fullFlow(): Pipeline =
  initPipeline([demeanOp(),
                bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                muteTopOp(lmoMute(2500, padMs = -20), taperMs = 20),
                normalizeOp(normRms, perPanel, target = 2),
                agcOp(windowMs = 300),
                gainCurveOp(knotGain([GainKnot(tMs: 0, dB: 0),
                    GainKnot(tMs: 3000, dB: 12)])),
                iirOp(butterHighCut(90, 24)),
                normalizeOp(normPercentile, perPanel, percentile = 95),
                clipOp(-3, 3),
                scaleOp(0.5),
                gainDbOp(6),
                polarityOp(),
                detrendOp(),
                muteBottomOp(timeMute(3500), taperMs = 40),
                normalizeOp(normMax)])

proc oneByOne[T](flow: Pipeline, x: var Tensor[T], meta: ChunkMeta) =
  ## The reference: each op through its own block form, on the whole block.
  let dt = meta.dtMs
  let n = x.shape[1]
  var ws = initWorkspace()
  for op in flow:
    case op.kind
    of opDemean: x.demean(meta)
    of opDetrend: x.detrend(meta)
    of opPolarity: x.reversePolarity(meta)
    of opScale: x.scale(meta, op.factor)
    of opGainDb: x.scale(meta, dbToFactor(op.dB))
    of opClip: x.clip(meta, op.clip.lo, op.clip.hi)
    of opMuteTop: x.muteTop(meta, op.mute.curve, op.mute.taperMs)
    of opMuteBottom: x.muteBottom(meta, op.mute.curve, op.mute.taperMs)
    of opFilter:
      let p = initFilterPlan(T, op.filter.spec, dt, n, op.filter.trend)
      ws.reserve(p.workspaceSize)
      x.applyFilter(meta, p, ws)
    of opIir:
      let p = initIirPlan(T, op.iir.spec, dt, n, op.iir.phase, op.iir.trend)
      ws.reserve(p.workspaceSize)
      x.applyFilter(meta, p, ws)
    of opAgc:
      let a = op.agc
      let p = initAgcPlan(a.windowMs, dt, n, a.mode, a.target, a.epsilon)
      ws.reserve(p.workspaceSize)
      x.agc(meta, p, ws)
    of opNormalize:
      let o = op.norm
      let p = initNormalizePlan(o.stat, n, o.scope, o.target, o.percentile)
      ws.reserve(p.workspaceSize)
      x.normalize(meta, p, ws)
    of opGainCurve:
      let g = op.gainCurve
      let p = initGainCurvePlan(T, g.curve, dt, n, g.tMinMs, g.tMaxMs)
      x.applyGainCurve(meta, p)

proc maxDiff[T](a, b: Tensor[T]): float64 =
  for r in 0..<a.shape[0]:
    for j in 0..<a.shape[1]:
      result = max(result, abs(float64(a[r, j]) - float64(b[r, j])))

block pipeline_editing:
  var flow = initPipeline()
  doAssert flow.len == 0
  flow.add agcOp(windowMs = 500)
  flow.insert(demeanOp(), 0)
  flow.add scaleOp(2)
  doAssert flow.len == 3 and flow[0].kind == opDemean and flow[1].kind == opAgc
  flow[1].agc.windowMs = 300
  doAssert flow[1].agc.windowMs == 300
  flow[2] = gainDbOp(6)
  doAssert flow[2].dB == 6
  flow.delete(0)
  doAssert flow.len == 2 and flow[0].kind == opAgc
  var kinds: seq[OpKind]
  for op in flow: kinds.add op.kind
  for i, op in flow: doAssert op.kind == kinds[i]
  doAssert kinds == @[opAgc, opGainDb]
  flow.clear
  doAssert flow.len == 0

block compile_validates:
  template rejects(op: Op, needle: string) =
    try:
      discard initPipeline([demeanOp(), op]).compile(float32, 2, 1000)
      doAssert false, "compiled: " & $op.kind
    except ValueError as e:
      doAssert e.msg.startsWith("op 1 (" & $op.kind & "): "), e.msg
      doAssert needle in e.msg, e.msg
  rejects(scaleOp(NaN), "finite")
  rejects(gainDbOp(Inf), "finite")
  rejects(gainDbOp(1e6), "overflows")
  rejects(gainDbOp(900), "overflows float32") # fine in float64, Inf in float32
  rejects(scaleOp(1e50), "overflows float32")
  discard initPipeline([gainDbOp(900), scaleOp(1e50)]).compile(float64, 2, 10)
  rejects(clipOp(2, 1), "lo ≤ hi")
  rejects(clipOp(NaN, 1), "lo ≤ hi")
  rejects(muteTopOp(timeMute(100), taperMs = -1), "taper")
  rejects(filterOp(highCut(300, 400)), "Nyquist")
  rejects(iirOp(bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80))), "Butterworth")
  rejects(agcOp(windowMs = -1), "AGC window")
  rejects(normalizeOp(normPercentile, percentile = 101), "percentile")
  rejects(gainCurveOp(GainCurve()), "no knots")
  doAssertRaises ValueError: discard initPipeline().compile(float32, 0, 10)
  doAssertRaises ValueError: discard initPipeline().compile(float32, NaN, 10)
  doAssertRaises ValueError: discard initPipeline().compile(float64, 2, 0)

block plan_properties:
  let flow = initPipeline([bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                           agcOp(windowMs = 500), iirOp(butterLowCut(5))])
  let plan = flow.compile(float32, dtMs = 2, samples = 2001)
  let f = initFilterPlan(float32, bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)), 2, 2001)
  let a = initAgcPlan(500, 2, 2001)
  let i = initIirPlan(float32, butterLowCut(5), 2, 2001)
  doAssert plan.requiredMargin == f.requiredMargin + a.requiredMargin + i.requiredMargin
  doAssert plan.workspaceSize == max(max(f.workspaceSize, a.workspaceSize), i.workspaceSize)
  doAssert plan.len == 3 and plan.samples == 2001 and plan.dtMs == 2
  doAssert plan.partCount == 1
  let plan2 = flow.compile(float32, dtMs = 2, samples = 2001)
  doAssert plan2.generation > plan.generation
  doAssert fullFlow().compile(float32, 2, 2001).partCount == 3
  var ws = initWorkspace()
  ws.reserve(plan)
  doAssert ws.fits(plan.workspaceSize)

proc equalsOneByOne[T](relTol: float64) =
  # `relTol` is relative to the largest output sample: the plan and the ops run
  # the same maths, but compilers may contract or vectorize the two paths
  # differently (gcc 15 with FMA differs in the last bit from gcc 12).
  var g = noisyGather[T](100, 2001, seed = 11)
  g.mark
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  let flow = fullFlow()
  let plan = flow.compile(T, dtMs = 2, samples = 2001)
  var ws = initWorkspace()
  ws.reserve(plan)
  var want = g.data.clone()
  flow.oneByOne(want, meta)
  let tol = relTol * maxDiff(want, zeros[T](100, 2001))
  # In place.
  var got = g.data.clone()
  plan.apply(got, meta, ws)
  doAssert maxDiff(got, want) <= tol, $T & " in place: " & $maxDiff(got, want)
  # src → dst: src untouched, dead and failed rows copied as they are.
  let src = g.data.clone()
  var dst = zeros[T](100, 2001)
  plan.apply(src, dst, meta, ws)
  doAssert maxDiff(src, g.data) == 0
  doAssert maxDiff(dst, want) <= tol, $T & " src → dst: " & $maxDiff(dst, want)
  for r in [3, 7, 40, 64, 65, 90]:
    for j in 0..<2001: doAssert dst[r, j] == src[r, j]
  # A first op that is a per-panel normalization reads src (fused copy).
  let flow2 = initPipeline([normalizeOp(normRms, perPanel, target = 3),
                            agcOp(windowMs = 200)])
  let plan2 = flow2.compile(T, 2, 2001)
  ws.reserve(plan2)
  var want2 = g.data.clone()
  flow2.oneByOne(want2, meta)
  plan2.apply(src, dst, meta, ws)
  doAssert maxDiff(dst, want2) <= tol, $T & " panel first: " & $maxDiff(dst, want2)
  # Applying the same block as src and dst runs in place.
  var same = g.data.clone()
  plan.apply(rowBlock(same), rowBlock(same), meta, ws)
  doAssert maxDiff(same, want) <= tol

block apply_equals_ops_one_by_one:
  equalsOneByOne[float64](1e-13)
  equalsOneByOne[float32](1e-6)

block empty_pipeline:
  var g = noisyGather[float32](10, 100, seed = 1)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let plan = initPipeline().compile(float32, 2, 100)
  var ws = initWorkspace()
  ws.reserve(plan)
  var dst = zeros[float32](10, 100)
  plan.apply(g.data, dst, meta, ws)
  doAssert maxDiff(dst, g.data) == 0
  let before = g.data.clone()
  plan.apply(g.data, meta, ws)
  doAssert maxDiff(before, g.data) == 0

block bad_apply_changes_nothing:
  var g = noisyGather[float32](20, 500, seed = 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let flow = initPipeline([demeanOp(), agcOp(windowMs = 100)])
  var ws = initWorkspace()
  let orig = g.data.clone()
  let narrow = flow.compile(float32, 2, 400)
  ws.reserve(narrow)
  doAssertRaises ValueError: narrow.apply(g.data, meta, ws)
  let otherDt = flow.compile(float32, 4, 500)
  ws.reserve(otherDt)
  doAssertRaises ValueError: otherDt.apply(g.data, meta, ws)
  let ok = flow.compile(float32, 2, 500)
  var small = zeros[float32](19, 500)
  doAssertRaises ValueError: ok.apply(g.data, small, meta, ws)
  var wrongCols = zeros[float32](20, 499)
  doAssertRaises ValueError: ok.apply(g.data, wrongCols, meta, ws)
  doAssertRaises ValueError: PipelinePlan[float32]().apply(g.data, meta, ws)
  doAssert maxDiff(g.data, orig) == 0
  doAssertRaises AssertionDefect:
    var empty = initWorkspace()
    ok.apply(g.data, meta, empty)

block no_allocation_in_steady_state:
  var g = noisyGather[float32](100, 2001, seed = 5)
  g.mark
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  let plan = fullFlow().compile(float32, 2, 2001)
  var ws = initWorkspace()
  ws.reserve(plan)
  var dst = zeros[float32](100, 2001)
  plan.apply(g.data, dst, meta, ws) # warm
  doAssert allocsIn(plan.apply(g.data, dst, meta, ws)) == 0
  doAssert allocsIn(plan.apply(dst, meta, ws)) == 0
  doAssert allocsIn(ws.reserve(plan)) == 0

# ------------------------------------------------ a plan across 4 threads --

type
  Job = object
    plan: ptr PipelinePlan[float32]
    src, dst: RowBlock[float32]
    meta: ChunkMeta

proc work(job: Job) {.thread.} =
  var ws = initWorkspace()
  {.cast(gcsafe).}:
    ws.reserve(job.plan[])
    try:
      job.plan[].apply(job.src, job.dst, job.meta, ws)
    except ValueError as e:
      raiseAssert e.msg

block plan_shared_across_threads:
  var g = noisyGather[float32](203, 2001, seed = 9)
  g.mark
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  var flow = fullFlow()
  var i = 0
  while i < flow.len: # per-panel factors would differ per thread's rows
    if flow[i].kind == opNormalize and flow[i].norm.scope == perPanel: flow.delete(i)
    else: inc i
  let plan = flow.compile(float32, 2, 2001)
  var ws = initWorkspace()
  ws.reserve(plan)
  var want = zeros[float32](203, 2001)
  plan.apply(g.data, want, meta, ws)
  var got = zeros[float32](203, 2001)
  let src = rowBlock(g.data)
  let dst = rowBlock(got)
  var threads: array[4, Thread[Job]]
  for t in 0..<4:
    let rows = (203 * t div 4) ..< (203 * (t + 1) div 4)
    createThread(threads[t], work, Job(plan: addr plan, src: src.sub(rows),
      dst: dst.sub(rows), meta: meta.sub(rows)))
  joinThreads(threads)
  doAssert maxDiff(got, want) == 0

block margins_and_zoom_windows:
  let flow = initPipeline([demeanOp(),
                           bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                           iirOp(butterLowCut(3), phase = ipMinimum),
                           agcOp(windowMs = 300), gainDbOp(3)])
  for dt in [1.0, 2.0, 4.0]:
    doAssert flow.requiredMargin(dt) == flow.compile(float32, dt, 1500).requiredMargin
  doAssert initPipeline([gainDbOp(3)]).requiredMargin(2) == 0
  try:
    discard initPipeline([demeanOp(), agcOp(windowMs = 1)]).requiredMargin(2)
    doAssert false
  except ValueError as e:
    doAssert e.msg.startsWith("op 1 (opAgc): "), e.msg
  doAssertRaises ValueError: discard flow.requiredMargin(0)
  doAssert decodeWindow(800 .. 1200, 110, 2001) == 690 .. 1310
  doAssert decodeWindow(50 .. 300, 110, 2001) == 0 .. 410
  doAssert decodeWindow(1900 .. 2000, 110, 2001) == 1790 .. 2000
  doAssert decodeWindow(10 .. 20, -5, 100) == 10 .. 20 # a negative margin is 0
  doAssert decodeWindow(10 .. 20, high(int), 100) == 0 .. 99 # no overflow
  # A plan takes any narrower block.
  var g = noisyGather[float32](10, 400, seed = 3)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let plan = flow.compile(float32, 2, 1500)
  var ws = initWorkspace()
  ws.reserve(plan)
  plan.apply(g.data, meta, ws)

block gain_curves_in_a_zoom_window:
  # A plan compiled for a window's width with the record's length beside it
  # tabulates its gain curves over the record: on a window far down the
  # trace it gives the bits of a plan compiled for the whole record (both
  # multiply by the table; off the table the curve is evaluated per sample,
  # 45 times slower, and rounds differently).
  const (ns, first, cols) = (6001, 3700, 1101)
  var g = noisyGather[float32](12, ns, seed = 5)
  var w = g.window(first, cols)
  let meta = initChunkMeta(w.rows, dtMs = 2, firstSample = first)
  for curve in [powerGain(2), exponentialGain(3), knotGain([GainKnot(tMs: 0, dB: 0),
      GainKnot(tMs: 9000, dB: 20)])]:
    let flow = initPipeline([gainCurveOp(curve)])
    let whole = flow.compile(float32, 2, ns)
    let window = flow.compile(float32, 2, cols, recordSamples = ns)
    doAssert window.samples == cols and window.recordSamples == ns
    doAssert whole.recordSamples == ns
    var ws = initWorkspace()
    ws.reserve(whole)
    ws.reserve(window)
    var a = zeros[float32](12, cols)
    var b = zeros[float32](12, cols)
    whole.apply(w.data, a, meta, ws)
    window.apply(w.data, b, meta, ws)
    for r in 0..<12:
      for j in 0..<cols:
        doAssert cast[uint32](a[r, j]) == cast[uint32](b[r, j]), $curve.kind & " row " & $r
  doAssertRaises ValueError: discard initPipeline().compile(float32, 2, 100, recordSamples = 99)
  doAssertRaises ValueError: discard initPipeline().compile(float32, 2, 100, recordSamples = -1)

block overlapping_blocks_and_part_preconditions:
  var buf = newSeq[float32](12 * 50)
  let whole = rowBlock(buf, 12, 50)
  var g = noisyGather[float32](10, 50, seed = 4)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let plan = initPipeline([scaleOp(2)]).compile(float32, 2, 50)
  var ws = initWorkspace()
  ws.reserve(plan)
  let a = whole.sub(0 .. 9)
  let b = whole.sub(2 .. 11) # the same shape, 2 rows further on
  doAssertRaises ValueError: plan.apply(a, b, meta, ws)
  plan.apply(a, a, meta, ws) # the same block: in place
  let flow = initPipeline([agcOp(windowMs = 20),
                           normalizeOp(normRms, perPanel)])
  let p2 = flow.compile(float32, 2, 50)
  doAssert p2.partCount == 2 and not p2.partNeedsPanel(0) and p2.partNeedsPanel(1)
  ws.reserve(p2)
  # A Defect leaving a {.raises: [].} proc skips its cleanup, so catching one
  # leaks its message: ASan builds leave these checks out.
  when not defined(addressSanitizer):
    var small = initWorkspace()
    doAssertRaises AssertionDefect:
      p2.runPart(0, RowBlock[float32](), rowBlock(g.data), meta, NaN, small)
    doAssertRaises AssertionDefect:
      discard p2.panelFactorOf(0, rowBlock(g.data), meta, ws)
    doAssertRaises AssertionDefect:
      p2.runPart(2, RowBlock[float32](), rowBlock(g.data), meta, NaN, ws)
  # The parts API by hand gives what apply gives.
  var want = g.data.clone()
  p2.apply(want, meta, ws)
  var got = g.data.clone()
  let blk = rowBlock(got)
  p2.checkApply(RowBlock[float32](), blk, meta)
  for part in 0 ..< p2.partCount:
    var f = NaN
    if p2.partNeedsPanel(part): f = p2.panelFactorOf(part, blk, meta, ws)
    for rows in [0 .. 4, 5 .. 9]: # as two threads would
      p2.runPart(part, RowBlock[float32](), blk.sub(rows), meta.sub(rows), f, ws)
  doAssert maxDiff(got, want) == 0
