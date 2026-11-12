## Datum statics (statics/elevation): the sign and value of a static, the
## synthetic elevation shifts removed (whole samples exactly, fractional
## within the interpolation error), the header columns (own datums, a buried
## source, header statics), NaN, dead rows, validation and no allocation.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

func near(a, b: float64, tol = 1e-12): bool = abs(a - b) <= tol * max(1, abs(b))

block sign_and_value:
  let d = datumStatics(datumZ = 100, vReplacement = 2000)
  doAssert d.datumZ == 100 and d.vReplacement == 2000
  var k = initTraceKeys()
  k.srcZ = 120
  k.rcvZ = 90
  # (20 − 10) m above the datum in all: 10 m at 2000 m/s is 5 ms earlier.
  doAssert near(d.staticMs(k), -5)
  # The synthetic delay is the opposite of the correction.
  doAssert near(d.staticMs(k), -elevationShiftMs(k, 100, 2000))
  k.rcvZ = NaN
  doAssert d.staticMs(k).isNaN

# A split-spread shot over topography, with two reflections delayed by the
# elevation static; `whole` keeps every shift a multiple of dt.
proc topoShot(whole: bool): (SynthGather[float64], SynthGather[float64]) =
  let rel = splitSpread(12, 25, 25)
  var rcvZ = newSeq[float64](rel.len)
  for i in 0..<rel.len:
    rcvZ[i] = if whole: 100 + 8 * float64(i mod 5) - 16 else: 100 + 7.3 * sin(
        0.7 * float64(i))
  var clean = shotGather[float64](1000, rel, 600, dtMs = 2, srcZ = 112, rcvZ = rcvZ)
  var shifted = clean.clone
  let shifts = clean.elevationShiftsMs(datumZ = 100, vReplacement = 2000)
  for (t0, v) in [(400.0, 2000.0), (800.0, 2600.0)]:
    clean.addHyperbola(t0, v, freqHz = 20)
    shifted.addHyperbola(t0, v, freqHz = 20, shiftsMs = shifts)
  (clean, shifted)

block whole_sample_shifts_removed_exactly:
  var (clean, g) = topoShot(whole = true)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let d = datumStatics(100, 2000)
  var st = newSeq[float64](g.len)
  staticsInto(meta, d, st)
  for i in 0..<g.len:
    let s = st[i] / 2
    doAssert s == round(s), "shift " & $st[i] & " ms is not whole"
  for interp in [siLinear, siCubic]:
    var x = g.clone
    applyStatics(x.data, meta, st, interp)
    # Events stay well inside the trace, so the result is the clean gather.
    for r in 0..<g.len:
      for j in 0..<g.cols:
        doAssert abs(x.data[r, j] - clean.data[r, j]) <= 1e-12

block fractional_shifts_removed:
  var (clean, g) = topoShot(whole = false)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let d = datumStatics(100, 2000)
  for (interp, tol) in [(siLinear, 0.015), (siCubic, 0.001)]:
    var x = g.clone
    applyStatics(x.data, meta, d, interp) # the buffer-free form
    var worst = 0.0
    for r in 0..<g.len:
      for j in 0..<g.cols:
        worst = max(worst, abs(x.data[r, j] - clean.data[r, j]))
    # A 20 Hz Ricker at 2 ms, peak 1: the interpolation error.
    doAssert worst <= tol, $interp & " " & $worst
    doAssert worst > 0

block header_columns:
  let d = datumStatics(100, 2000)
  var k = initTraceKeys()
  k.srcZ = 130
  k.rcvZ = 110
  var st = initTraceStatics()
  doAssert near(d.staticMs(k, st), d.staticMs(k)) # unknown columns: d's datum
  st.srcDatum = 120
  st.rcvDatum = 110
  doAssert near(d.staticMs(k, st), -5) # 10 m above its own datum, 0 m
  st.srcDepth = 10
  doAssert near(d.staticMs(k, st), 0) # the buried source sits on its datum
  st.rcvDepth = 4
  doAssert near(d.staticMs(k, st), 2) # the receiver 4 m below its datum
  st.rcvDepth = NaN
  doAssert headerStaticMs(st).isNaN
  st.srcStatic = -4
  doAssert headerStaticMs(st) == -4
  st.rcvStatic = 1.5
  doAssert headerStaticMs(st) == -2.5
  var hs = newSeq[float64](2)
  headerStaticsInto([st, initTraceStatics()], hs)
  doAssert hs[0] == -2.5 and hs[1].isNaN
  doAssertRaises ValueError: headerStaticsInto([st], hs)
  # The column form.
  var g = cmpGather[float32](0, [100.0, 200], 10, dtMs = 2)
  g.keys[0].srcZ = 130
  g.keys[0].rcvZ = 110
  g.keys[1].srcZ = 100
  g.keys[1].rcvZ = 100
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var out1 = newSeq[float64](2)
  staticsInto(meta, d, [st, initTraceStatics()], out1)
  doAssert near(out1[0], d.staticMs(g.keys[0], st)) and out1[1] == 0
  var short = newSeq[float64](1)
  doAssertRaises ValueError: staticsInto(meta, d, short)
  doAssertRaises ValueError: staticsInto(meta, d, [st], out1)

block unset_datums_take_the_requested_datum:
  # Header columns whose datum elevations are unknown (NaN, as parseSEG
  # gives them for SEG-Y headers that leave bytes 53-60 unset, and for SEG-D
  # and SEG-2) give the statics of the requested datum: the same as columns
  # that hold that datum, and as the form without columns.
  let d = datumStatics(datumZ = 100, vReplacement = 2000)
  var g = cmpGather[float32](0, [100.0, 200, 300], 10, dtMs = 2)
  for r, (s, z) in [(100.0, 86.09), (131.5, 120.25), (99.0, NaN)]:
    g.keys[r].srcZ = s
    g.keys[r].rcvZ = z
  let meta = initChunkMeta(g.rows, dtMs = 2)
  var unset = newSeq[TraceStatics](3)
  var same = newSeq[TraceStatics](3)
  for i in 0..<3:
    unset[i] = initTraceStatics()
    doAssert unset[i].srcDatum.isNaN and unset[i].rcvDatum.isNaN
    same[i] = initTraceStatics()
    same[i].srcDatum = d.datumZ
    same[i].rcvDatum = d.datumZ
  var byKeys, byUnset, bySame = newSeq[float64](3)
  staticsInto(meta, d, byKeys)
  staticsInto(meta, d, unset, byUnset)
  staticsInto(meta, d, same, bySame)
  for i in 0..<2:
    doAssert byUnset[i] == byKeys[i] and bySame[i] == byKeys[i], $i
    doAssert d.staticMs(g.keys[i], unset[i]) == d.staticMs(g.keys[i], same[i])
  doAssert near(byKeys[0], 6.955) and near(byKeys[1], -25.875)
  doAssert byKeys[2].isNaN and byUnset[2].isNaN and bySame[2].isNaN
  # One datum unknown, the other given: each end takes its own.
  var mixed = initTraceStatics()
  mixed.srcDatum = 120
  doAssert near(d.staticMs(g.keys[1], mixed), -(131.5 - 120 + 120.25 - 100) / 2)
  # An elevation of 0 is a datum (sea level), not "unset": only NaN is.
  var sea = initTraceStatics()
  sea.srcDatum = 0
  sea.rcvDatum = 0
  doAssert near(d.staticMs(g.keys[0], sea), -(100 + 86.09) / 2)

block nan_and_dead_rows_untouched:
  var g = cmpGather[float32](0, [100.0, 200, 300, 400], 50, dtMs = 2)
  for r in 0..<g.len:
    for j in 0..<g.cols: g.data[r, j] = float32(j + 1)
  g.keys[1].status = tsDead
  g.keys[2].status = tsAuxiliary
  let meta = initChunkMeta(g.rows, dtMs = 2)
  let before = g.data.clone
  applyStatics(g.data, meta, [4.0, 4, 4, NaN])
  for j in 0..<g.cols:
    doAssert g.data[0, j] == (if j >= 2: before[0, j - 2] else: 0)
    for r in 1..3: doAssert g.data[r, j] == before[r, j]

block live_length_respected:
  var g = cmpGather[float32](0, [100.0], 20, dtMs = 2)
  for j in 0..<20: g.data[0, j] = 1
  g.setLive(0, 12)
  g.data[0, 15] = 99 # a sentinel past the live end (never there in real data)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  applyStatics(g.data, meta, [-6.0]) # 3 samples earlier
  for j in 0..<9: doAssert g.data[0, j] == 1
  for j in 9..<12: doAssert g.data[0, j] == 0
  doAssert g.data[0, 15] == 99

block validation_and_allocation:
  for v in [0.0, -1, NaN, Inf]:
    doAssertRaises ValueError: discard datumStatics(0, v)
  doAssertRaises ValueError: discard datumStatics(NaN, 1000)
  var g = cmpGather[float32](0, [100.0, 200], 64, dtMs = 2)
  let meta = initChunkMeta(g.rows, dtMs = 2)
  doAssertRaises ValueError: applyStatics(g.data, meta, [1.0])
  let d = datumStatics(0, 1500)
  var st = newSeq[float64](2)
  var b = rowBlock(g.data)
  doAssert allocsIn(staticsInto(meta, d, st)) == 0
  doAssert allocsIn(applyStatics(b, meta, st, siCubic)) == 0
  doAssert allocsIn(applyStatics(b, meta, d)) == 0
