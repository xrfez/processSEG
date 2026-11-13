## Correlation kernels for `tests/tvectorized.nim`, in a module of their
## own: a correlation plan transforms its pilot with the one-trace FFT,
## whose instances (scalar by design) would otherwise sit among the lanes
## instances that `veckernels` checks.

import processSEG
import synth

proc vec_correlate_f32(plan: CorrelationPlan[float32], src, dst: RowBlock[float32],
    meta: ChunkMeta, ws: var Workspace) {.exportc: "vec_correlate_f32", noinline.} =
  plan.correlateInto(src, dst, meta, ws)

proc correlateOnce*() =
  # mulKernel (the lanes path), and dot through resamplePilot.
  let pilot = generatePilot(linearSweep(8, 80, 100, 2), float32)
  let resampled = pilot.resamplePilot(1)
  let plan = initCorrelationPlan(float32, pilot, 2, 200)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var g = cmpGather[float32](cmpX = 0, offsets = [0.0, 100.0], 200, 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var y = zeros[float32](2, plan.outSamples)
  vec_correlate_f32(plan, rowBlock(g.data), rowBlock(y), meta, ws)
  echo y[0, 0], resampled.len
