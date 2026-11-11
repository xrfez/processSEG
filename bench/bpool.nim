## ProcessPool (task 5.2): the per-job overhead (an empty job on every
## worker, and an empty plan through `apply`), and a viewer pipeline on a
## 2000 × 2001 float32 panel at 2 ms (demean, band-pass 4-8-60-80, AGC
## 500 ms, src → dst) on the single-thread `apply` and on pools of 1, 2, 4
## and 8 workers. The pool is created once, outside the timing.

import std/strutils
import benchutil
import processSEG
import ../tests/utils/synth

const
  traces = 2000
  ns = 2001
  dt = 2.0

proc panel(rows: int): SynthGather[float32] =
  result = cmpGather[float32](cmpX = 0, offsets = newSeq[float64](rows), ns, dt)
  for r in 0..<rows:
    var x = newSeq[float32](ns)
    x.addNoise(1, seed = r)
    x.addRicker(dt, 1000 + float64(r), 25, amp = 4)
    x.addDc(3)
    for j in 0..<ns: result.data[r, j] = x[j]

proc ms(ns: float64): string = formatFloat(ns / 1e6, ffDecimal, 2) & " ms"

proc main() =
  printHeader "process pool, " & $traces & " traces × " & $ns & " samples f32 at 2 ms"
  var g = panel(traces)
  let meta = initChunkMeta(g.rows, dtMs = dt)
  var dst = zeros[float32](traces, ns)
  let flow = initPipeline([demeanOp(),
    bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80)), agcOp(windowMs = 500)])
  let plan = flow.compile(float32, dt, ns)
  let empty = initPipeline().compile(float32, dt, ns)
  var one = panel(1)
  let oneMeta = initChunkMeta(one.rows, dtMs = dt)

  for workers in [1, 2, 4, 8]:
    var pool = newProcessPool(workers)
    let r = bench("ping, " & $workers & " workers", bytesPerIter = 0):
      pool.ping
    echo "  ping (empty job), ", workers, " workers: ",
      formatFloat(r.medianNs / 1000, ffDecimal, 2), " µs"
    let e = bench("empty plan 1 row, " & $workers & " workers", bytesPerIter = 0):
      pool.apply(empty, one.data, oneMeta)
    echo "  empty plan on 1 row, ", workers, " workers: ",
      formatFloat(e.medianNs / 1000, ffDecimal, 2), " µs"

  var ws = initWorkspace()
  ws.reserve(plan)
  let s = bench("pipeline, apply on 1 thread", bytesPerIter = 2 * traces * ns * 4):
    plan.apply(g.data, dst, meta, ws)
  echo "  demean + band-pass + AGC, apply (no pool): ", ms(s.medianNs)
  for workers in [1, 2, 4, 8]:
    var pool = newProcessPool(workers)
    let r = bench("pipeline, pool " & $workers, bytesPerIter = 2 * traces * ns * 4):
      pool.apply(plan, g.data, dst, meta)
    echo "  demean + band-pass + AGC, pool of ", workers, ": ", ms(r.medianNs),
      "  (", formatFloat(s.medianNs / r.medianNs, ffDecimal, 2), "× apply)"

main()
