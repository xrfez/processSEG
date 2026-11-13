## Frequency-domain filtering (filter/apply): the D4 edge handling against
## clean references (large DC, linear drift), rows of varying live length, no
## wraparound, a symmetric impulse response, the zoom margin, and no
## allocation in a steady state.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

const
  dt = 2.0
  ns = 2001

proc clean[T](n = ns): seq[T] =
  ## Events well inside the trace, all energy above 5 Hz.
  result = zeroSignal[T](n)
  result.addRicker(dt, 600, 25)
  result.addRicker(dt, 1800, 30, amp = -0.7)
  result.addRicker(dt, 3300, 20, amp = 0.5)

func maxAbs[T](x: openArray[T]): float64 =
  for v in x: result = max(result, abs(float64(v)))

func maxDiff[T](a, b: openArray[T], lo = 0, hi = -1): float64 =
  let h = if hi < 0: a.len - 1 else: hi
  for i in lo..h: result = max(result, abs(float64(a[i]) - float64(b[i])))

proc filtered[T](x: seq[T], plan: FilterPlan[T]): seq[T] =
  result = x
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  plan.applyFilter(result, ws)

block plan_shape:
  let s = lowCut(0.5, 1)
  let p = initFilterPlan(float32, s, dt, ns)
  doAssert p.requiredMargin == s.impulseLength(dt) and p.requiredMargin > 0
  # A low-cut pads with zeros: 2L between the trace's ends. A high-cut pads
  # with reflections of L each, and 2L of zeros between them.
  doAssert p.nfft >= ns + 2 * p.requiredMargin and p.nfft mod 2 == 0
  doAssert p.nfft < ns + 4 * p.requiredMargin
  let hc = initFilterPlan(float32, highCut(60, 80), dt, ns)
  doAssert hc.nfft >= ns + 4 * hc.requiredMargin
  doAssert isFastLen(p.nfft)
  doAssert p.binHz <= s.narrowestRampHz / 2, "bins resolve the ramp"
  doAssert p.samples == ns and p.dtMs == dt and p.spec.kind == rkTrapezoid
  doAssert p.gainAt(0) == 0 and p.gainAt(p.nfft div 2) == 1
  # A short trace with a narrow ramp: df, not the padding, sets nfft.
  let q = initFilterPlan(float64, highCut(60, 60.2), dt, 10)
  doAssert q.binHz <= 0.1 and q.nfft > 10 + 4 * q.requiredMargin
  doAssertRaises ValueError: discard initFilterPlan(float32, s, dt, 0)
  doAssertRaises ValueError: discard initFilterPlan(float32, highCut(200, 260), dt, ns)
  doAssertRaises ValueError: discard initFilterPlan(float32, s, 0, ns)
  # A flat response: no padding, the trace comes back unchanged.
  let flat = initFilterPlan(float64, bandPass(Corners(f1: 0, f2: 0, f3: Inf,
    f4: Inf)), dt, ns)
  doAssert flat.requiredMargin == 0
  let c = clean[float64]()
  doAssert maxDiff(filtered(c, flat), c) < 1e-12

block an_impulse_longer_than_the_trace:
  # A 0.02-0.04 Hz low-cut at 2 ms is 22 000 samples long. Its zero pads are
  # cut to the trace (nfft about 3 × ns, not 50 000), its margin stays L, and
  # on a trace too short to resolve the ramp it removes the trend and little
  # else. Reflected pads (a high-cut, trMean) keep L.
  let s = lowCut(0.02, 0.04)
  let p = initFilterPlan(float64, s, dt, ns)
  doAssert p.requiredMargin == s.impulseLength(dt) and p.requiredMargin > 10 * ns
  doAssert p.nfft >= 3 * ns and p.nfft < 4 * ns, $p.nfft
  let c = clean[float64]()
  var x = c
  x.addDrift(dt, slopePerS = 50, start = -30)
  doAssert maxDiff(filtered(x, p), c) < 0.01
  let m = initFilterPlan(float64, s, dt, ns, trend = trMean)
  doAssert m.nfft >= ns + 4 * m.requiredMargin

proc edgesMatchReference(T: typedesc, tol: float64) =
  # A 1 Hz low-cut with a 0.5–1 Hz ramp: the signal plus a large DC, and plus
  # a linear drift, match the filtered clean signal, in the first 10 % of the
  # samples and everywhere else.
  let plan = initFilterPlan(T, lowCut(0.5, 1), dt, ns)
  let c = clean[T]()
  let reference = filtered(c, plan)
  doAssert maxDiff(reference, c) < 0.02, "the low-cut barely touches 20–30 Hz"
  var dc = c
  dc.addDc(1000)
  var drift = c
  drift.addDrift(dt, slopePerS = 50, start = -30)
  for (name, x) in [("DC", dc), ("drift", drift)]:
    let y = filtered(x, plan)
    let first = maxDiff(y, reference, 0, ns div 10 - 1)
    doAssert first <= tol, $T & " " & name & ": first 10 % off by " & $first
    doAssert maxDiff(y, reference) <= tol, $T & " " & name
  # The mean alone is enough for a DC.
  let mp = initFilterPlan(T, lowCut(0.5, 1), dt, ns, trend = trMean)
  doAssert maxDiff(filtered(dc, mp), filtered(c, mp), 0, ns div 10 - 1) <= tol

block edges_against_clean_references:
  edgesMatchReference(float64, 1e-9)
  edgesMatchReference(float32, 2e-4) # 1000 + x in float32 keeps ~6e-5

block no_swing_without_trend_removal:
  # With nothing removed, the reflection keeps value and slope continuous, so
  # a small drift through a low-cut does not swing at the ends. A large offset
  # does leak (the pad taper pulls it to 0 through the ramp): about 1 % of it,
  # which is why trLinear is the default.
  let plan = initFilterPlan(float64, lowCut(0.5, 1), dt, ns, trend = trNone)
  var x = clean[float64]()
  x.addDrift(dt, slopePerS = 2, start = 0.5)
  let y = filtered(x, plan)
  let e = maxDiff(y, clean[float64](), 0, ns div 10 - 1)
  doAssert e < 0.1, $e
  var big = clean[float64]()
  big.addDc(1000)
  let leak = maxDiff(filtered(big, plan), filtered(clean[float64](), plan), 0, ns div 10 - 1)
  doAssert leak > 1 and leak < 20, $leak

block trend_back_by_dc_gain:
  # A high-cut keeps the trend (gain 1 at 0 Hz); withZeroDc drops its mean and
  # keeps its slope, so the output has zero mean.
  var x = clean[float64]()
  x.addDrift(dt, slopePerS = 3, start = 2) # 2 .. 14, mean 8
  let hc = filtered(x, initFilterPlan(float64, highCut(60, 80), dt, ns))
  doAssert abs(hc[0] - 2) < 1e-3 and abs(hc[^1] - (2 + 3 * 4.0)) < 1e-3
  for tr in [trMean, trLinear]:
    let z = filtered(x, initFilterPlan(float64, highCut(60, 80).withZeroDc, dt, ns, tr))
    var mean = 0.0
    for v in z: mean += v
    doAssert abs(mean / float64(ns)) < 1e-3, $tr & " mean " & $(mean / float64(ns))
    if tr == trLinear:
      doAssert abs(z[0] + 6) < 1e-3 and abs(z[^1] - 6) < 1e-3, $z[0] & " " & $z[^1]

block no_wraparound:
  # An event at the end of the trace does not leak to its start: the output
  # matches a plan for much longer traces (whose FFT cannot wrap within this
  # one), to −60 dB of the peak, in the worst case of a spike on the last
  # sample (its odd reflection fills the pad).
  # The last three put a narrow feature next to a wide one: the low ramp and the
  # notch set L.
  let specs = [lowCut(0.5, 1), lowCut(1, 2), bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
      butterLowCut(2), highCut(60, 80).withNotch(50, 2),
      bandPass(Corners(f1: 0.5, f2: 1, f3: 60, f4: 80)),
      bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)).withNotch(50, 1),
      butterBandPass(2, 80, 48)]
  for s in specs:
    var x = zeroSignal[float64](ns)
    x.addRicker(dt, float64(ns - 15) * dt, 25, amp = 1)
    x.addSpike(ns - 1, 1)
    let y = filtered(x, initFilterPlan(float64, s, dt, ns))
    let wide = filtered(x, initFilterPlan(float64, s, dt, 6 * ns))
    let peak = maxAbs(wide)
    let e = maxDiff(y, wide)
    doAssert e <= 1e-3 * peak, $s.kind & ": wrap " & $(e / peak)
    doAssert maxDiff(y, wide, 0, ns div 10 - 1) <= 1e-3 * peak

block symmetric_impulse_response:
  for s in [bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)), lowCut(1, 2),
      butterBandPass(5, 60), highCut(60, 80).withNotch(50, 4)]:
    let plan = initFilterPlan(float64, s, dt, ns, trend = trNone)
    var x = zeroSignal[float64](ns)
    x.addSpike(1000)
    let y = filtered(x, plan)
    var e = 0.0
    for k in 1..1000: e = max(e, abs(y[1000 + k] - y[1000 - k]))
    doAssert e < 1e-12, $s.kind & ": " & $e
    # The centre is the response's mean over the band: the sum of the gains.
    doAssert y[1000] > 0
  # float32 too.
  let p32 = initFilterPlan(float32, bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
    dt, ns, trend = trNone)
  var x = zeroSignal[float32](ns)
  x.addSpike(1000)
  let y = filtered(x, p32)
  for k in 1..1000: doAssert abs(y[1000 + k] - y[1000 - k]) < 1e-6

proc rowsMatchSingle(T: typedesc, tol: float64) =
  # 19 rows (two full batches and a partial one) of varying live length, a
  # dead row and a failed one; each live row matches the single-trace filter
  # of its live part, and nothing past the live end changes.
  const rows = 19
  var g = cmpGather[T](cmpX = 0, offsets = newSeq[float64](rows), ns, dt)
  for r in 0..<rows:
    var row = clean[T]()
    row.addDc(float64(r) - 9)
    row.addDrift(dt, slopePerS = float64(r mod 5), start = 0)
    for j in 0..<ns: g.data[r, j] = row[j]
  let lives = [ns, 1500, 700, 2, 1, 0, 1999]
  for i, n in lives: g.setLive(3 * i, n)
  g.keys[4].status = tsDead
  let failed = @[7]
  for j in 0..<ns:
    g.data[2, j] = T(5) # beyond-live samples of row 3·k are zero; mark more
  let meta = initChunkMeta(g.rows, dtMs = dt, failed = failed)
  let plan = initFilterPlan(T, lowCut(1, 2), dt, ns)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  let before = g.data.clone()
  plan.applyFilter(g.data, meta, ws)
  for r in 0..<rows:
    let n = meta.liveSamples(r, ns)
    if not meta.isLive(r):
      for j in 0..<ns: doAssert g.data[r, j] == before[r, j], "row " & $r & " untouched"
      continue
    var want = newSeq[T](n)
    for j in 0..<n: want[j] = before[r, j]
    plan.applyFilter(want, ws)
    var e = 0.0
    for j in 0..<n: e = max(e, abs(float64(g.data[r, j]) - float64(want[j])))
    doAssert e <= tol, $T & " row " & $r & " (" & $n & " live): " & $e
    for j in n..<ns: doAssert g.data[r, j] == before[r, j], "past the live end"
  doAssert g.data[4, 100] == before[4, 100] and g.data[7, 100] == before[7, 100]
  # One sample: its trend is itself, so a low-cut leaves 0.
  doAssert g.data[9, 0] == 0
  let n = allocsIn:
    plan.applyFilter(g.data, meta, ws)
  doAssert n == 0, $n & " allocations"

block rows_of_varying_live_length:
  rowsMatchSingle(float64, 1e-12)
  rowsMatchSingle(float32, 2e-5)

proc sameBits[T](a, b: T): bool =
  when T is float32: cast[uint32](a) == cast[uint32](b)
  else: cast[uint64](a) == cast[uint64](b)

proc rowsIgnoreTheirBatch(T: typedesc, spec: FilterSpec) =
  # A row's result does not depend on the rows filtered with it: the whole
  # block (full batches of one length go through the lane-wise pack, the
  # others row by row), each row alone, and a split at an odd row give the
  # same bits. A pool relies on this to match the single-thread apply.
  const rows = 27
  var g = cmpGather[T](cmpX = 0, offsets = newSeq[float64](rows), ns, dt)
  for r in 0..<rows:
    var row = clean[T]()
    row.addNoise(0.3, seed = r)
    row.addDc(100 * float64(r mod 4) - 150)
    row.addDrift(dt, slopePerS = float64(r mod 5) - 2, start = 0)
    for j in 0..<ns: g.data[r, j] = row[j]
  for j in 0..<ns: g.data[10, j] = T(0) # a zero trend among others
  g.data[10, 5] = T(-0.0)
  g.setLive(3, 1500) # rows 0 ..< 8 are not of one length; 8 ..< 16 are
  g.setLive(20, 1)
  g.keys[17].status = tsDead
  let meta = initChunkMeta(g.rows, dtMs = dt)
  let plan = initFilterPlan(T, spec, dt, ns)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var whole = g.data.clone()
  plan.applyFilter(whole, meta, ws)
  var alone = g.data.clone()
  let ab = rowBlock(alone)
  for r in 0..<rows:
    plan.applyFilter(ab.sub(r .. r), meta.sub(r .. r), ws)
  var split = g.data.clone()
  let sb = rowBlock(split)
  for part in [0 .. 10, 11 .. rows - 1]:
    plan.applyFilter(sb.sub(part), meta.sub(part), ws)
  for r in 0..<rows:
    for j in 0..<ns:
      doAssert sameBits(whole[r, j], alone[r, j]), $T & " " & $spec.kind &
        ": row " & $r & " sample " & $j & " differs alone"
      doAssert sameBits(whole[r, j], split[r, j]), $T & " " & $spec.kind &
        ": row " & $r & " sample " & $j & " differs in a split"

block rows_do_not_depend_on_their_batch:
  # A band-pass drops the trend; the high-cuts add it back (all or its slope).
  let specs = [bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)), highCut(60, 80),
    highCut(60, 80).withZeroDc]
  for spec in specs:
    rowsIgnoreTheirBatch(float32, spec)
    rowsIgnoreTheirBatch(float64, spec)

proc endError(spec: FilterSpec, n: int, seed: int64): float64 =
  ## The RMS error over the first and last L/2 samples of a trace cut out of
  ## a longer one (6L more on each side), against the long trace filtered,
  ## as a fraction of the filtered signal's RMS there.
  let L = spec.impulseLength(dt)
  let ctx = 6 * L
  var long = zeroSignal[float64](n + 2 * ctx)
  long.addNoise(1, seed = seed)
  let truth = filtered(long, initFilterPlan(float64, spec, dt, long.len))
  let got = filtered(long[ctx ..< ctx + n], initFilterPlan(float64, spec, dt, n))
  var err, sig = 0.0
  for i in 0..<max(8, L div 2):
    for j in [i, n - 1 - i]:
      err += (got[j] - truth[ctx + j]) ^ 2
      sig += truth[ctx + j] ^ 2
  sqrt(err / sig)

block trace_ends_match_a_long_context:
  # What a low-cut or a band-pass gives at the ends of a trace is what it
  # gives there inside a longer one, as far as the missing samples allow.
  # (Padding by odd reflection pinned the end samples to 0, whatever the
  # data: 41 % and 92 % error here. Zeros after the trend removal give 4 %
  # and 15 %.) A high-cut keeps the reflection, the better choice for it.
  var lc, bp = 0.0
  for seed in 1'i64..8:
    lc = max(lc, endError(lowCut(2, 4), ns, seed))
    bp = max(bp, endError(bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)), ns, seed))
  doAssert lc < 0.10, "low-cut 2-4: " & $lc
  doAssert bp < 0.30, "band-pass 4-8-60-80: " & $bp
  # The end sample itself is not forced to zero.
  var x = zeroSignal[float64](3 * ns)
  x.addNoise(1, seed = 5)
  let plan = initFilterPlan(float64, lowCut(2, 4), dt, 3 * ns)
  let whole = filtered(x, plan)
  let cut = filtered(x[ns ..< 2 * ns], initFilterPlan(float64, lowCut(2, 4), dt, ns))
  doAssert abs(whole[ns]) > 0.05, "pick another seed: the truth is near 0 here"
  doAssert abs(cut[0] - whole[ns]) < 0.5 * abs(whole[ns]), $cut[0] & " for " & $whole[ns]

block zoom_window_on_noise:
  # A window of white noise decoded with requiredMargin, against the whole
  # trace: within 0.7 % of the peak (0.3 % measured; 2.3 % with the reflected
  # pads). Broadband data is the usual case; a steady sinusoid inside the
  # narrowest ramp is the worst (about 2.5 %, see the module documentation).
  const total = 6001
  let spec = bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80))
  var worst = 0.0
  for seed in 1'i64..6:
    var x = zeroSignal[float64](total)
    x.addNoise(1, seed = seed)
    let whole = filtered(x, initFilterPlan(float64, spec, dt, total))
    let m = spec.impulseLength(dt)
    for a in [1000, 2500, 4200]:
      let b = a + 499
      let win = filtered(x[a - m .. b + m], initFilterPlan(float64, spec, dt, b - a + 1 + 2 * m))
      var e = 0.0
      for j in a..b: e = max(e, abs(win[j - (a - m)] - whole[j]))
      worst = max(worst, e / maxAbs(whole))
  doAssert worst < 0.007, $worst

block zoom_window_with_margin:
  # A window decoded with requiredMargin on each side, filtered, matches the
  # whole trace filtered, over the window.
  const total = 4001
  for s in [lowCut(0.5, 1), bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)),
      butterBandPass(3, 80), bandPass(Corners(f1: 0.5, f2: 1, f3: 60, f4: 80)),
      butterBandPass(4, 60, 48), highCut(100, 120).withNotch(50, 1)]:
    var x = zeroSignal[float64](total)
    x.addNoise(1, seed = 3)
    x.addRicker(dt, 3000, 25, amp = 5)
    x.addDrift(dt, 4, 1)
    let whole = filtered(x, initFilterPlan(float64, s, dt, total))
    let (a, b) = (1500, 2300)
    let plan = initFilterPlan(float64, s, dt, b - a + 1 + 2 * 1500)
    let m = plan.requiredMargin
    let lo = max(0, a - m)
    var win = x[lo .. b + m]
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    plan.applyFilter(win, ws)
    var e = 0.0
    for j in a..b: e = max(e, abs(win[j - lo] - whole[j]))
    let rel = e / maxAbs(whole.toOpenArray(a, b))
    doAssert rel < 0.01, $s.kind & ": " & $rel

block nan_stays_in_its_row:
  var g = cmpGather[float32](cmpX = 0, offsets = newSeq[float64](3), 500, dt)
  for r in 0..<3:
    var row = clean[float32](500)
    for j in 0..<500: g.data[r, j] = row[j]
  g.data[1, 250] = NaN
  let meta = initChunkMeta(g.rows, dtMs = dt)
  let plan = initFilterPlan(float32, lowCut(1, 2), dt, 500)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  plan.applyFilter(g.data, meta, ws)
  doAssert g.data[1, 0].isNaN
  for j in 0..<500:
    doAssert g.data[0, j] == g.data[0, j] and g.data[2, j] == g.data[2, j]

block shape_errors:
  var g = cmpGather[float32](cmpX = 0, offsets = newSeq[float64](2), 600, dt)
  let meta = initChunkMeta(g.rows, dtMs = dt)
  let plan = initFilterPlan(float32, lowCut(1, 2), dt, 500)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  doAssertRaises ValueError: plan.applyFilter(g.data, meta, ws)
  var long = newSeq[float32](501)
  doAssertRaises AssertionDefect: plan.applyFilter(long, ws)
  var empty: seq[float32]
  plan.applyFilter(empty, ws)
  var p0: FilterPlan[float32]
  var one = @[1'f32]
  doAssertRaises AssertionDefect: p0.applyFilter(one, ws)

block interval_must_match_the_plan:
  # A plan at 2 ms on rows at 4 ms would put every corner off by 2×: the
  # block forms refuse it (FFT and IIR filters, the averaged spectrum), as
  # AGC, gain curves and pipelines do.
  var g = cmpGather[float32](0, [100.0, 200], 100, dtMs = 4)
  let meta = initChunkMeta(g.rows, 4)
  var ws = initWorkspace()
  let fp = initFilterPlan(float32, bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)), 2, 100)
  ws.reserve(fp.workspaceSize)
  doAssertRaises ValueError: fp.applyFilter(g.data, meta, ws)
  let ip = initIirPlan(float32, butterBandPass(5, 60), 2, 100)
  ws.reserve(ip.workspaceSize)
  doAssertRaises ValueError: ip.applyFilter(g.data, meta, ws)
  let sp = initSpectrumPlan(float32, dtMs = 2, windowLen = 64)
  ws.reserve(sp.workspaceSize)
  var amp = newSeq[float64](sp.bins)
  doAssertRaises ValueError:
    discard sp.averageAmplitudeInto(rowBlock(g.data), meta, 0, amp, ws)
