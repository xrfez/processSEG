## Mutes (core/mute): curves, exact taper values, the time axis (t0Ms and a
## window), live rows, NaN rows and validation.

import std/[assertions, math, sequtils]
import processSEG
import utils/[allocs, synth]

func near(a, b: float64, tol = 1e-6): bool = abs(a - b) <= tol

func ones(n: int): seq[float64] =
  result = newSeq[float64](n)
  for v in result.mitems: v = 1

block curves:
  doAssert timeMute(100).muteTimeMs(NaN) == 100, "a time mute ignores offsets"
  let lmo = lmoMute(velocity = 2000, t0Ms = 10, padMs = -5)
  doAssert near(lmo.muteTimeMs(1000), 505) and near(lmo.muteTimeMs(-1000), 505)
  doAssert lmo.muteTimeMs(NaN).isNaN
  let tab = tableMute([0.0, 100, 300], [10.0, 50, 250])
  doAssert tab.kind == mkTable
  doAssert tab.muteTimeMs(0) == 10 and tab.muteTimeMs(100) == 50
  doAssert near(tab.muteTimeMs(50), 30) and near(tab.muteTimeMs(-200), 150)
  doAssert tab.muteTimeMs(1000) == 250, "flat beyond the last knot"
  doAssert tableMute([200.0], [80.0]).muteTimeMs(0) == 80
  doAssert tab.muteTimeMs(NaN).isNaN

block validation:
  doAssertRaises ValueError: discard timeMute(NaN)
  doAssertRaises ValueError: discard tableMute([], [])
  doAssertRaises ValueError: discard tableMute([0.0, 1], [1.0])
  doAssertRaises ValueError: discard tableMute([1.0, 1], [1.0, 2])
  doAssertRaises ValueError: discard tableMute([-1.0], [1.0])
  doAssertRaises ValueError: discard tableMute([0.0], [Inf])
  doAssertRaises ValueError: discard lmoMute(0)
  doAssertRaises ValueError: discard lmoMute(-1500)
  doAssertRaises ValueError: discard lmoMute(1500, padMs = NaN)
  var g = cmpGather[float32](0, [100.0], cols = 10, dtMs = 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  doAssertRaises ValueError: g.data.muteTop(meta, timeMute(4), taperMs = -1)
  doAssertRaises ValueError: g.data.muteBottom(meta, timeMute(4), taperMs = NaN)

block top_kernel_exact:
  var x = ones(10)
  muteTop(x, startCol = 6, taperCols = 4)
  doAssert x[0] == 0 and x[1] == 0 and x[2] == 0
  for j in 3..5:
    doAssert near(x[j], sin(PI / 2 * float64(j - 2) / 4) ^ 2, 1e-12)
  for j in 6..9: doAssert x[j] == 1
  var sharp = ones(5)
  muteTop(sharp, startCol = 2, taperCols = 0)
  doAssert sharp == @[0.0, 0, 1, 1, 1], "a sharp mute keeps the sample at t"
  var frac = ones(5)
  muteTop(frac, startCol = 2.5, taperCols = 1)
  doAssert frac[1] == 0 and near(frac[2], sin(PI / 4) ^ 2, 1e-12) and frac[3] == 1
  var all = ones(4)
  muteTop(all, startCol = 99, taperCols = 3)
  doAssert all == @[0.0, 0, 0, 0]
  var none = ones(4)
  muteTop(none, startCol = -10, taperCols = 3)
  doAssert none == ones(4)
  var nan = ones(4)
  muteTop(nan, startCol = NaN, taperCols = 3)
  doAssert nan == ones(4), "a NaN mute time leaves the row alone"

block bottom_kernel_exact:
  var x = ones(10)
  muteBottom(x, endCol = 3, taperCols = 4)
  for j in 0..3: doAssert x[j] == 1
  for j in 4..6:
    doAssert near(x[j], sin(PI / 2 * float64(7 - j) / 4) ^ 2, 1e-12)
  for j in 7..9: doAssert x[j] == 0
  var sharp = ones(5)
  muteBottom(sharp, endCol = 2, taperCols = 0)
  doAssert sharp == @[1.0, 1, 1, 0, 0]
  var all = ones(4)
  muteBottom(all, endCol = -1, taperCols = 0)
  doAssert all == @[0.0, 0, 0, 0]
  var none = ones(4)
  muteBottom(none, endCol = 50, taperCols = 2)
  doAssert none == ones(4)

proc filled(offsets: openArray[float64], cols: int): SynthGather[float32] =
  result = cmpGather[float32](0, offsets, cols, dtMs = 4)
  for r in 0..<result.len:
    for j in 0..<cols: result.data[r, j] = 1

block lmo_top_mute_on_a_gather:
  # Offsets 0, 400, 800 at 2000 m/s: mute times 0, 200, 400 ms = columns
  # 0, 50, 100 at 4 ms; a 20 ms taper is the 5 columns before each.
  var g = filled([0.0, 400, 800], cols = 150)
  let meta = initChunkMeta(g.rows, dtMs = 4)
  g.data.muteTop(meta, lmoMute(velocity = 2000), taperMs = 20)
  for i, col in [0, 50, 100]:
    let r = int(i)
    for j in 0..<col - 5: doAssert g.data[r, j] == 0
    for k in 0..<5:
      if col - 5 + k >= 0:
        doAssert near(float64(g.data[r, col - 5 + k]), sin(PI / 2 * float64(k) /
            5) ^ 2)
    for j in col..<150: doAssert g.data[r, j] == 1, "untouched from the mute time on"

block taper_limit_is_continuous:
  # A vanishing taper converges on the sharp cut: the sample at t is kept.
  for L in [0.0, 1e-3]:
    var top = ones(6)
    muteTop(top, startCol = 3, taperCols = L)
    doAssert top == @[0.0, 0, 0, 1, 1, 1], $L
    var bottom = ones(6)
    muteBottom(bottom, endCol = 2, taperCols = L)
    doAssert bottom == @[1.0, 1, 1, 0, 0, 0], $L

block time_axis_follows_t0_and_the_window:
  # Row 0 starts at t0 = 100 ms: a 140 ms mute is its column 10 at 4 ms;
  # row 1 starts at 0, so column 35. A window from trace sample 20 (80 ms)
  # puts 140 ms at its column 15.
  var g = filled([0.0, 0], cols = 40)
  g.keys[0].t0Ms = 100
  let meta0 = initChunkMeta(g.rows, dtMs = 4)
  g.data.muteTop(meta0, timeMute(140))
  doAssert g.data[0, 9] == 0 and g.data[0, 10] == 1
  doAssert g.data[1, 34] == 0 and g.data[1, 35] == 1
  var h = filled([0.0], cols = 60)
  var win = h.window(firstSample = 20, cols = 30)
  let meta1 = initChunkMeta(win.rows, dtMs = 4, firstSample = 20)
  win.data.muteTop(meta1, timeMute(140))
  doAssert win.data[0, 14] == 0 and win.data[0, 15] == 1
  win.data.muteBottom(meta1, timeMute(180))
  doAssert win.data[0, 25] == 1 and win.data[0, 26] == 0

block table_bottom_mute_and_liveness:
  var g = filled([100.0, 200, 300, 400], cols = 100)
  g.keys[1].status = tsAuxiliary
  g.keys[2].offset = NaN
  g.setLive(3, 60)
  for j in 60..<100: g.data[3, j] = 7 # a sentinel past the live end
  let failed: seq[int] = @[]
  let meta = initChunkMeta(g.rows, dtMs = 4, failed = failed)
  let tab = tableMute([100.0, 400], [200.0, 160])       # columns 50 .. 40
  g.data.muteBottom(meta, tab)
  doAssert g.data[0, 50] == 1 and g.data[0, 51] == 0
  doAssert g.data[1, 99] == 1, "auxiliary rows are untouched"
  doAssert g.data[2, 99] == 1, "a NaN offset leaves the row alone"
  doAssert g.data[3, 40] == 1 and g.data[3, 41] == 0 and g.data[3, 59] == 0
  for j in 60..<100:
    doAssert g.data[3, j] == 7, "nothing past the live end is written"
  g.data.muteTop(meta, timeMute(400))
  doAssert g.data[3, 60] == 7, "a top mute also stops at the live end"

block nan_t0_leaves_the_row_alone:
  var g = filled([100.0], cols = 10)
  g.keys[0].t0Ms = NaN
  let meta = initChunkMeta(g.rows, dtMs = 4)
  g.data.muteTop(meta, timeMute(20))
  doAssert g.data[0, 0] == 1

block float64_and_no_allocation:
  var g = cmpGather[float64](0, [0.0, 500, 1000, 1500], cols = 1000, dtMs = 2)
  for r in 0..<g.len:
    for j in 0..<1000: g.data[r, j] = 1
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let top = lmoMute(1800, padMs = -20)
  let bottom = tableMute([0.0, 1500], [1200.0, 1900])
  let n = allocsIn:
    for pass in 0..<10:
      g.data.muteTop(meta, top, taperMs = 30)
      g.data.muteBottom(meta, bottom, taperMs = 40)
  doAssert n == 0, "mutes allocated " & $n & " times"
  doAssert g.data[3, 0] == 0 and g.data[0, 999] == 0

block infinite_bounds_and_tapers:
  # Opposite infinities made a NaN ramp bound, int(NaN) = int.low, and the
  # boundChecks-off loops ran from it (a heap overflow under ASan). An
  # infinite taper is no taper.
  proc ones(): seq[float32] = newSeq[float32](8).mapIt(1'f32)
  var x = ones()
  x.muteBottom(-Inf, Inf)
  doAssert x.allIt(it == 0)
  x = ones()
  x.muteTop(Inf, Inf)
  doAssert x.allIt(it == 0)
  x = ones()
  x.muteTop(5, Inf)          # hard mute at column 5
  doAssert x[0 ..< 5].allIt(it == 0) and x[5 ..< 8].allIt(it == 1)
  x = ones()
  x.muteBottom(2, Inf)
  doAssert x[0 .. 2].allIt(it == 1) and x[3 ..< 8].allIt(it == 0)
  x = ones()
  x.muteTop(-Inf, 3)
  doAssert x.allIt(it == 1)
  # A subnormal interval is refused (it made the block taper infinite).
  var keys = @[initTraceKeys()]
  let counts = @[8]
  doAssertRaises ValueError: discard initChunkMeta(rowsOf(keys, counts), dtMs = 1e-310)
