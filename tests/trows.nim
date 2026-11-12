## RowBlock (core/rows) and Workspace (core/workspace): zero-copy rows, raw
## rows in threads, grow-only scratch, and no allocation in a steady state.

import std/[assertions, typedthreads]
import processSEG
import utils/[allocs, synth]

proc addOne[T](x: var openArray[T]) =
  for v in x.mitems: v += T(1)

func total[T](x: openArray[T]): float64 =
  for v in x: result += float64(v)

block rows_are_views_of_the_tensor:
  var t = zeros[float32](3, 4)
  let b = rowBlock(t)
  doAssert b.len == 3 and b.cols == 4
  addOne(b.row(1))
  doAssert t[1, 0] == 1 and t[1, 3] == 1 and t[0, 0] == 0 and t[2, 3] == 0
  t[2, 2] = 5
  doAssert b.row(2)[2] == 5 and total(b.row(2)) == 5
  doAssert b.rowPtr(1)[3] == 1

block float64_and_any_buffer:
  var buf = newSeq[float64](6)
  let b = rowBlock(buf, rows = 2, cols = 3)
  addOne(b.row(1))
  doAssert buf == @[0.0, 0, 0, 1, 1, 1]
  doAssertRaises ValueError:
    discard rowBlock(buf, rows = 4, cols = 2)
  doAssertRaises ValueError:
    discard rowBlock(buf, rows = -2, cols = -3)

block a_slice_of_rows:
  # A tensor that is a contiguous slice of a bigger one: rows 2..3 of 5.
  var big = zeros[float32](5, 4)
  for r in 0..<5:
    for j in 0..<4: big[r, j] = float32(10 * r + j)
  let part = big[2..3, _]
  doAssert part.is_C_contiguous
  let b = rowBlock(part)
  doAssert b.len == 2 and b.row(0)[0] == 20 and b.row(1)[3] == 33,
    "the tensor's offset is honoured"

block rejects_bad_tensors:
  doAssertRaises ValueError:
    discard rowBlock(zeros[float32](12))
  let wide = zeros[float64](4, 3)
  doAssertRaises ValueError:
    discard rowBlock(wide.transpose)
  let cut = zeros[float32](4, 6)
  doAssertRaises ValueError:
    discard rowBlock(cut[_, 1..3]) # a column slice has gaps

block empty_blocks:
  let t = zeros[float32](0, 10)
  let b = rowBlock(t)
  doAssert b.len == 0 and b.cols == 10
  let n = zeros[float32](3, 0)
  let c = rowBlock(n)
  doAssert c.len == 3 and c.cols == 0 and total(c.row(2)) == 0

block sub_blocks:
  var t = zeros[float32](6, 2)
  let b = rowBlock(t)
  let s = b.sub(2..4)
  doAssert s.len == 3 and s.cols == 2
  addOne(s.row(0))
  doAssert t[2, 0] == 1 and t[1, 0] == 0
  doAssert b.sub(6..5).len == 0, "an empty slice may start at len"
  doAssertRaises ValueError:
    discard b.sub(4..6)
  doAssertRaises ValueError:
    discard b.sub(-1..2)
  doAssertRaises ValueError:
    discard b.sub(7..6)

block row_bounds:
  when compileOption("boundChecks"):
    let t = zeros[float32](2, 2)
    let b = rowBlock(t)
    doAssertRaises IndexDefect:
      discard b.rowPtr(2)
    doAssertRaises IndexDefect:
      discard b.rowPtr(-1)

# Raw rows in threads: each worker adds one to its own half of the rows.
proc halfWorker(b: RowBlock[float32]) {.thread.} =
  for r in 0..<b.len: addOne(b.row(r))

block rows_in_threads:
  var t = zeros[float32](64, 100)
  let b = rowBlock(t)
  var threads: array[2, Thread[RowBlock[float32]]]
  createThread(threads[0], halfWorker, b.sub(0..31))
  createThread(threads[1], halfWorker, b.sub(32..63))
  joinThreads(threads)
  for r in 0..<64:
    doAssert total(b.row(r)) == 100

block workspace_grows_only:
  var ws = initWorkspace()
  doAssert ws.capacity == WorkspaceSize()
  ws.reserve(WorkspaceSize(f32: 100, f64: 10))
  doAssert ws.capacity == WorkspaceSize(f32: 100, f64: 10)
  ws.reserve(WorkspaceSize(f32: 50, f64: 20))
  doAssert ws.capacity == WorkspaceSize(f32: 100, f64: 20), "no shrinking"
  doAssert ws.fits(WorkspaceSize(f32: 100, f64: 20))
  doAssert not ws.fits(WorkspaceSize(f32: 101))
  doAssertRaises ValueError:
    ws.reserve(WorkspaceSize(f32: -1))
  let a = WorkspaceSize(f32: 3, f64: 5)
  let b = WorkspaceSize(f32: 4, f64: 1)
  doAssert a + b == WorkspaceSize(f32: 7, f64: 6)
  doAssert max(a, b) == WorkspaceSize(f32: 4, f64: 5)

block reserve_for_any_plan:
  # `ws.reserve(plan)` is `ws.reserve(plan.workspaceSize)` for every plan
  # type, not the pipeline's alone.
  var ws = initWorkspace()
  let fft = initFftPlan(float32, 1024)
  ws.reserve(fft)
  doAssert ws.fits(fft.workspaceSize)
  let filter = initFilterPlan(float32, lowCut(2, 4), 2, 1000)
  let iir = initIirPlan(float32, butterLowCut(3), 2, 1000)
  let agc = initAgcPlan(500, 2, 1000)
  let flow = initPipeline([demeanOp(), agcOp(windowMs = 300)]).compile(float32, 2, 1000)
  ws.reserve(filter)
  ws.reserve(iir)
  ws.reserve(agc)
  ws.reserve(flow)
  doAssert ws.fits(filter.workspaceSize) and ws.fits(iir.workspaceSize) and
    ws.fits(agc.workspaceSize) and ws.fits(flow.workspaceSize)
  doAssert iir.dtMs == 2

block scratch_regions:
  var ws = initWorkspace()
  ws.reserve(WorkspaceSize(f32: 10, f64: 8))
  let x = ws.scratch(float32, 4)
  let y = ws.scratch(float32, 6, at = 4)
  for i in 0..<4: x[i] = 1
  for i in 0..<6: y[i] = 2
  doAssert ws.scratch(float32, 10)[3] == 1 and ws.scratch(float32, 10)[4] == 2
  let z = ws.scratch(float64, 8)
  z[7] = 3.5
  doAssert ws.scratch(float64, 1, at = 7)[0] == 3.5
  # Checked in every build mode, -d:danger included: a kernel handed a
  # workspace reserved for a smaller plan must not write past it.
  doAssertRaises IndexDefect:
    discard ws.scratch(float32, 7, at = 4)
  doAssertRaises IndexDefect:
    discard ws.scratch(float64, 9)
  doAssertRaises IndexDefect:
    discard ws.scratch(float64, 1, at = -1)
  doAssertRaises IndexDefect:
    discard ws.scratch(float32, int.high, at = 4)
  doAssert ws.scratch(float32, 0, at = 10) == nil, "an empty request at the end"
  var empty = initWorkspace()
  doAssert empty.scratch(float32, 0) == nil

block scratch_starts_on_a_cache_line:
  # Element 0 of every region is 64-byte aligned, at any size and after a
  # move (a kernel lays its SIMD buffers out from there), and all of the
  # reserved elements are usable from it.
  for n in [1, 3, 16, 17, 1000, 4097]:
    var ws = initWorkspace()
    ws.reserve(WorkspaceSize(f32: n, f64: n, u32: n))
    doAssert cast[int](ws.scratch(float32, n)) mod 64 == 0, "f32, " & $n
    doAssert cast[int](ws.scratch(float64, n)) mod 64 == 0, "f64, " & $n
    doAssert cast[int](ws.scratch(uint32, n)) mod 64 == 0, "u32, " & $n
    let f = ws.scratch(float32, n)
    let d = ws.scratch(float64, n)
    let u = ws.scratch(uint32, n)
    for i in 0..<n: # every element writable (ASan checks the bounds)
      f[i] = 1
      d[i] = 2
      u[i] = 3
    var moved = move ws
    doAssert moved.scratch(float64, n) == d and moved.scratch(float64, n)[n - 1] == 2

block workspace_used_from_a_thread:
  # As the pool will do: the workspace stays in its owner's storage and the
  # worker uses it through a pointer (createThread copies its argument, and a
  # Workspace cannot be copied).
  proc useWorkspace(ws: ptr Workspace) {.thread.} =
    let p = ws[].scratch(float64, 1000)
    for i in 0..<1000: p[i] = float64(i)
  var ws = initWorkspace()
  ws.reserve(WorkspaceSize(f64: 1000))
  var th: Thread[ptr Workspace]
  createThread(th, useWorkspace, addr ws)
  joinThread(th)
  doAssert ws.scratch(float64, 1000)[999] == 999

# A Workspace cannot be copied: tests/reject/rworkspacecopy.nim.

# The steady state of an apply: rows of a tensor, scratch from a workspace
# reserved again for the same plan. After the first pass nothing allocates.
proc fakeApply(samples: Tensor[float32], meta: ChunkMeta, ws: var Workspace,
    need: WorkspaceSize) =
  ws.reserve(need)
  let b = rowBlock(samples)
  let tmp = ws.scratch(float64, b.cols)
  for r in 0..<b.len:
    if not meta.isLive(r): continue
    let n = meta.liveSamples(r, b.cols)
    let x = b.rowPtr(r)
    for j in 0..<n: tmp[j] = float64(x[j])
    for j in 0..<n: x[j] = float32(tmp[j] + 1)

block no_allocation_in_a_steady_state:
  var g = cmpGather[float32](0, [0.0, 100, 200, 300], cols = 500, dtMs = 2)
  g.setLive(2, 100)
  let failed = @[3]
  let meta = initChunkMeta(g.rows, dtMs = 2, failed = failed)
  var ws = initWorkspace()
  let need = WorkspaceSize(f32: 0, f64: 500)
  fakeApply(g.data, meta, ws, need) # the first pass sizes the workspace
  let n = allocsIn:
    for pass in 0..<100:
      fakeApply(g.data, meta, ws, need)
      fakeApply(g.data, meta, ws, WorkspaceSize(f64: 100)) # a smaller plan
  doAssert n == 0, "steady-state applies allocated " & $n & " times"
  doAssert g.data[0, 499] == 201 and g.data[2, 99] == 201 and g.data[2, 100] == 0
  doAssert g.data[3, 0] == 0, "failed rows are left alone"
  let grow = allocsIn:
    ws.reserve(WorkspaceSize(f64: 501))
  doAssert grow == 1, "a bigger plan reallocates once"

block block_size_overflow:
  # rows · cols beyond int: ValueError (it wrapped to 0 in danger builds).
  let empty: seq[float32] = @[]
  doAssertRaises ValueError: discard rowBlock(empty, 1 shl 32, 1 shl 32)
