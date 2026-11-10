## The viewer loop (tests/utils/viewersim, test code only) on synthetic data
## and parseSEG's demo files: raw cache → pool → processed buffers tagged by
## generation, stale buffers always reprocessed, no allocation in a
## steady-state frame, slider changes, a zoom window with `requiredMargin`,
## and moveout ticks made stale by `applySurvey`.

import std/[assertions, math, os]
import parseSEG, processSEG
import utils/[allocs, demo, synth, viewersim]

proc flow(): Pipeline =
  initPipeline([demeanOp(),
                bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
                agcOp(windowMs = 500),
                normalizeOp(normRms, perPanel)])

proc maxDiff(a, b: Tensor[float32]): float64 =
  for r in 0..<a.shape[0]:
    for j in 0..<a.shape[1]:
      result = max(result, abs(float64(a[r, j]) - float64(b[r, j])))

proc singleThread(fl: Pipeline, e: Entry): Tensor[float32] =
  ## The reference: the same chunk through `PipelinePlan.apply` on this thread.
  let meta = chunkMeta(e.raw)
  let plan = fl.compile(float32, meta.dtMs, e.raw.samples.shape[1])
  var ws = initWorkspace()
  ws.reserve(plan)
  result = zeros[float32](e.raw.samples.shape[0], e.raw.samples.shape[1])
  plan.apply(e.raw.samples, result, meta, ws)

proc gather(rows, cols: int, dtMs: float64, seed: int64): SynthGather[float32] =
  var offs = newSeq[float64](rows)
  for i in 0..<rows: offs[i] = 50.0 * float64(i)
  result = cmpGather[float32](0, offs, cols, dtMs)
  for r in 0..<rows:
    var x = newSeq[float32](cols)
    x.addNoise(1, seed = seed + r)
    x.addRicker(dtMs, 400 + float64(r), 25, amp = 5)
    for j in 0..<cols: result.data[r, j] = x[j]
  result.keys[2].status = tsDead
  result.setLive(4, cols div 2)

block synthetic_loop:
  var g1 = gather(150, 1001, 2, seed = 1)
  var g2 = gather(90, 501, 4, seed = 2) # another interval: its own plan
  let failed = @[7]
  var v = initViewerSim(flow(), workers = 4)
  v.loadSynthetic(g1.rows, g1.data, failed = failed)
  v.loadSynthetic(g2.rows, g2.data)
  doAssert v.entries.len == 2
  doAssert v.stale(v.entries[0]) and v.stale(v.entries[1])
  v.frame
  doAssert v.reprocessed == 2
  for e in v.entries:
    doAssert maxDiff(e.processed, singleThread(v.flow, e)) == 0
  doAssert maxDiff(v.entries[0].raw.samples, g1.data) == 0 # raw cache kept
  let first = v.drawn
  # Nothing changed: nothing is reprocessed and the picture is the same.
  v.frame
  doAssert v.reprocessed == 0 and v.drawn == first
  # A slider change: every buffer is stale, then reprocessed.
  v.edit:
    v.flow[2].agc.windowMs = 200
  doAssert v.stale(v.entries[0]) and v.stale(v.entries[1])
  v.frame
  doAssert v.reprocessed == 2 and v.drawn != first
  for e in v.entries:
    doAssert maxDiff(e.processed, singleThread(v.flow, e)) == 0
  # A wider filter needs more scratch: the first frame grows the workers'
  # workspaces once. After that, a frame after a recompile allocates nothing.
  v.edit:
    v.flow[1].filter.spec = bandPass(Corners(f1: 2, f2: 5, f3: 40, f4: 60))
  v.frame
  v.edit:
    v.flow[2].agc.windowMs = 300
  doAssert allocsIn(v.frame) == 0
  doAssert v.reprocessed == 2
  doAssert allocsIn(v.frame) == 0

block survey_sgy_loop_and_zoom:
  var f = openSeismic(demoDir() / "survey.sgy", keys = true)
  defer: f.close()
  var v = initViewerSim(flow(), workers = 4)
  v.load(f, 0 ..< f.len, chunkTraces = 64)
  doAssert v.entries.len == 4 # 200 traces in chunks of 64
  v.frame
  doAssert v.reprocessed == 4
  let raw = f.readTraces[:float32](0 ..< f.len)
  var row = 0
  for e in v.entries:
    doAssert maxDiff(e.processed, singleThread(v.flow, e)) == 0
    doAssert maxDiff(e.raw.samples, raw[row ..< row + e.raw.count, _]) == 0
    row += e.raw.count
  v.edit:
    v.flow.delete(3) # drop the per-panel normalization: windows differ in level
    v.flow.delete(0) # and demean: a window has its own mean (no margin fixes it)
  v.frame
  # Zoom windows, decoded with the margin, against the whole traces processed
  # and cut to the same columns. The margin bounds the error relative to the
  # strongest signal near the window (99.9 % of each impulse's energy), so:
  # - samples 300 .. 1000 hold two events: band-pass + AGC within 1 % of the
  #   window's peak;
  # - samples 800 .. 1200 hold only noise, 46 dB below an event just past the
  #   margin: the band-pass is within 1 % of the traces' peak, but not of the
  #   quiet window's own peak, and an AGC after it lifts that leak to the
  #   window's level (integration doc, "Sample windows").
  let traces = 20 ..< 60
  let src = f.readTraces[:float32](traces)
  let meta = chunkMeta(f, traces)
  proc zoomError(v: var ViewerSim, shown: Slice[int], ofTraces: bool): float64 =
    let zoom = v.zoomWindow(f, traces, shown)
    doAssert zoom.shape[0] == traces.len and zoom.shape[1] == shown.len
    var full = zeros[float32](traces.len, src.shape[1])
    let plan = v.flow.compile(float32, meta.dtMs, full.shape[1])
    var ws = initWorkspace()
    ws.reserve(plan)
    plan.apply(src, full, meta, ws)
    let want = full[_, shown.a .. shown.b].clone()
    maxDiff(zoom, want) / peakAbs(if ofTraces: full else: want)
  let e1 = v.zoomError(300 .. 1000, ofTraces = false)
  doAssert e1 <= 0.01, "zoom with events off by " & $e1 & " of its peak"
  var bp = initViewerSim(initPipeline([v.flow[0]]), workers = 2)
  doAssert bp.flow[0].kind == opFilter
  let e2 = bp.zoomError(800 .. 1200, ofTraces = true)
  doAssert e2 <= 0.01, "quiet zoom off by " & $e2 & " of the traces' peak"

block ticks_and_apply_survey:
  let path = getTempDir() / "processseg_viewer_record.sgd"
  copyFile(demoDir() / "record.sgd", path)
  var f = openSeismic(path, keys = true)
  defer: f.close()
  var v = initViewerSim(flow(), workers = 2)
  v.load(f, 0 ..< f.len)
  v.frame
  let lmo = linearMoveout(1800, t0Ms = 20)
  v.updateTicks(f, lmo)
  doAssert not v.ticksStale(f)
  doAssert v.entries[0].ticks[0].isNaN # no coordinates yet: no offset
  # The viewer stops its readers, joins the survey, and the ticks go stale.
  var g: SurveyGeometry
  g.addReceiver(10, 104, 1, 1000, 2000, 50)
  g.addSource(20, 99.5, 1, 1300, 2400, 40)
  doAssert f.applySurvey(g) == 1
  doAssert v.ticksStale(f)
  v.clearCache # views and metas are rebuilt after a join
  v.load(f, 0 ..< f.len)
  v.frame
  doAssert v.reprocessed == 1
  v.updateTicks(f, lmo)
  doAssert not v.ticksStale(f)
  let meta = chunkMeta(v.entries[0].raw)
  doAssert abs(meta.keys(0).offset - 500) < 1e-9
  let want = (20 + 1000 * 500 / 1800 - meta.t0Ms(0)) / meta.dtMs
  doAssert abs(v.entries[0].ticks[0] - want) < 1e-9

block keys_change_marks_buffers_stale:
  # A mute reads offsets: after a join the processed buffers are stale even
  # though the plan is the same, and the raw cache is kept.
  let path = getTempDir() / "processseg_viewer_record2.sgd"
  copyFile(demoDir() / "record.sgd", path)
  var f = openSeismic(path, keys = true)
  defer: f.close()
  var v = initViewerSim(initPipeline([muteTopOp(lmoMute(1800), taperMs = 10),
                                      agcOp(windowMs = 200)]), workers = 2)
  v.load(f, 0 ..< f.len)
  v.frame
  let before = v.entries[0].processed.clone()
  var g: SurveyGeometry
  g.addReceiver(10, 104, 1, 1000, 2000, 50)
  g.addSource(20, 99.5, 1, 1300, 2400, 40)
  doAssert f.applySurvey(g) == 1
  v.refresh(f)
  doAssert v.stale(v.entries[0])
  v.frame
  doAssert v.reprocessed == 1
  doAssert maxDiff(v.entries[0].processed, singleThread(v.flow, v.entries[0])) == 0
  doAssert maxDiff(v.entries[0].processed, before) > 0 # the mute now applies
  v.frame
  doAssert v.reprocessed == 0

block line_sg2_ticks:
  var f = openSeismic(demoDir() / "line.sg2", keys = true)
  defer: f.close()
  var v = initViewerSim(initPipeline([agcOp(windowMs = 50)]), workers = 3)
  v.load(f, 0 ..< f.len, chunkTraces = 10)
  v.frame
  for e in v.entries:
    doAssert maxDiff(e.processed, singleThread(v.flow, e)) == 0
  v.updateTicks(f, hyperbolicMoveout(1500, t0Ms = 100))
  for e in v.entries:
    let meta = chunkMeta(e.raw)
    for r in 0..<meta.len:
      let x = meta.keys(r).offset
      if not x.isNaN:
        doAssert abs(e.ticks[r] - (nmoTime(x, 1500, 100) - meta.t0Ms(r)) /
          meta.dtMs) < 1e-9
