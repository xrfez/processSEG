## Signal-to-noise ratios (qc/snr): the windowed SNR per trace and the
## coherence SNR per gather against a known signal in white noise across
## −10 … +30 dB, noise alone, windows fixed and relative to picks and to an
## LMO guide, a flattened shot, mutes, non-live rows, float32 and float64,
## no allocation, NaN cases and errors.
##
## Window sizes of the accuracy checks (a ±1 dB estimate at −10 dB needs
## them; see the module documentation for the spread):
## - windowed: one trace per SNR, a noise window of 200 000 samples, then a
##   signal window (a 37 Hz sinusoid in the noise) of 200 000 samples;
## - coherence: 96 traces of the same Gaussian signal in independent noise,
##   a 401-sample window, every column of 3000.

import std/[assertions, math]
import processSEG
import parseSEG
import utils/[allocs, synth]

const levelsDb = [-10.0, -5, 0, 5, 10, 20, 30]

proc liveKeys(rows: int, dtMs: float64): seq[TraceKeys] =
  for r in 0 ..< rows: result.add stationKeys(0, 25 * float64(r + 1), 0, 0, dtMs)

proc setRow[T](g: var SynthGather[T], r: int, x: openArray[float64]) =
  for j in 0 ..< g.cols: g.data[r, j] = T(x[j])

# ---------------------------------------------------------------- windowed --

const
  winLen = 200_000
  winDt = 1.0

proc windowedGather[T](snrDb: openArray[float64], truth: var seq[float64]): SynthGather[T] =
  ## One row per SNR: noise (σ = 1) over the whole row, a sinusoid in the
  ## second half; `truth` gets the sinusoid's power there over σ², in dB.
  result = initSynthGather[T](liveKeys(snrDb.len, winDt), 2 * winLen, winDt)
  truth.setLen(snrDb.len)
  for r, db in snrDb:
    var x = newSeq[float64](2 * winLen)
    x.addNoise(1.0, 100 + r)
    var s = newSeq[float64](winLen)
    s.addSinusoid(winDt, 37, amp = sqrt(2 * pow(10.0, db / 10)))
    var power = 0.0
    for j in 0 ..< winLen:
      x[winLen + j] += s[j]
      power += s[j] * s[j]
    truth[r] = 10 * log10(power / float64(winLen))
    result.setRow(r, x)

let fixedHalves = WindowSnrParams(noiseStartMs: 0, noiseEndMs: float64(winLen - 1) * winDt,
  signalStartMs: float64(winLen) * winDt, signalEndMs: float64(2 * winLen - 1) * winDt)

proc windowedAccuracy[T]() =
  var truth: seq[float64]
  var g = windowedGather[T](levelsDb, truth)
  let meta = initChunkMeta(g.rows, winDt)
  var est = newSeq[float64](g.len)
  var p = fixedHalves
  p.estimator = seExcess
  windowSnrInto(g.data, meta, p, est)
  for r in 0 ..< g.len:
    doAssert abs(est[r] - truth[r]) < 1, $T & " at " & $levelsDb[r] & " dB: " & $est[r]
  # The ratio reads (S + N) / N: right at high SNR, +0.4 dB at −10 dB.
  p.estimator = seRatio
  windowSnrInto(g.data, meta, p, est)
  for r in 0 ..< g.len:
    let want = 10 * log10(1 + pow(10.0, truth[r] / 10))
    doAssert abs(est[r] - want) < 0.1, $levelsDb[r] & ": " & $est[r]

block windowed_snr_within_1_db:
  windowedAccuracy[float32]()
  windowedAccuracy[float64]()

block windowed_noise_alone:
  # Ratio: about 0 dB. Excess: no signal, at most −10 dB (mostly the floor).
  var truth: seq[float64]
  var g = windowedGather[float32]([-200.0, -200, -200, -200], truth)
  let meta = initChunkMeta(g.rows, winDt)
  var est = newSeq[float64](g.len)
  windowSnrInto(g.data, meta, fixedHalves, est)
  for v in est: doAssert abs(v) < 0.05, $v
  var p = fixedHalves
  p.estimator = seExcess
  windowSnrInto(g.data, meta, p, est)
  for v in est: doAssert v <= -10 and v >= DefaultSnrFloorDb, $v
  # Short windows on many traces: the ratio spreads, about 0 dB on average.
  var n = initSynthGather[float64](liveKeys(200, 2), 100, 2)
  for r in 0 ..< n.len:
    var x = newSeq[float64](100)
    x.addNoise(1.0, 500 + r)
    n.setRow(r, x)
  let short = WindowSnrParams(noiseStartMs: 0, noiseEndMs: 78, signalStartMs: 100,
    signalEndMs: 178)
  var e2 = newSeq[float64](n.len)
  windowSnrInto(n.data, initChunkMeta(n.rows, 2), short, e2)
  var mean = 0.0
  for v in e2: mean += v / float64(e2.len)
  doAssert abs(mean) < 0.5, $mean

# --------------------------------------------------- windows and their axis --

const
  dt = 2.0
  cols = 600
  vel = 2000.0
  t0 = 20.0

proc shot(sigma: float64, amp = 1.0): SynthGather[float32] =
  result = shotGather[float32](1000, splitSpread(24, 25, 25), cols, dt)
  result.addLinear(t0, vel, 30, amp)
  for r in 0 ..< result.len:
    var x = newSeq[float64](cols)
    x.addNoise(sigma, 40 + r)
    for j in 0 ..< cols: result.data[r, j] += float32(x[j])

block relative_windows_pick_and_guide:
  var g = shot(0.05)
  g.keys[5].status = tsDead
  g.keys[9].offset = NaN
  g.setLive(30, 5) # cut before its windows
  let failed = @[12]
  let meta = initChunkMeta(g.rows, dt, failed = failed)
  let guide = linearMoveout(vel, t0)
  let p = WindowSnrParams(signalStartMs: -20, signalEndMs: 20, noiseStartMs: -120,
    noiseEndMs: -60, estimator: seExcess)
  var byGuide = newSeq[float64](g.len)
  windowSnrInto(g.data, meta, p, guide, byGuide)
  var refMs = newSeq[float64](g.len)
  moveoutTimesInto(meta, guide, refMs)
  var byTimes = newSeq[float64](g.len)
  windowSnrInto(g.data, meta, p, refMs, byTimes)
  for r in 0 ..< g.len:
    if r in [5, 9, 12, 30]:
      doAssert byGuide[r].isNaN and byTimes[r].isNaN, $r & ": " & $byGuide[r]
    else:
      doAssert byGuide[r] == byTimes[r] or (byGuide[r].isNaN and byTimes[r].isNaN), $r
      if refMs[r] - 60 < 0: # no noise window before the trace's start
        doAssert byGuide[r].isNaN, $r
      elif refMs[r] - 120 >= 0: # a whole noise window
        doAssert byGuide[r] > 15, $r & ": " & $byGuide[r]
      # The same windows by hand, in columns.
      let c = refMs[r] / dt
      var x = newSeq[float32](cols)
      for j in 0 ..< cols: x[j] = g.data[r, j]
      let want = windowSnrDb(x, int(ceil(c - 10)) .. int(floor(c + 10)),
        int(ceil(c - 60)) .. int(floor(c - 30)), seExcess)
      doAssert abs(byGuide[r] - want) < 1e-9 or (want.isNaN and byGuide[r].isNaN), $r
  # A NaN pick gives NaN; the others are kept.
  refMs[3] = NaN
  windowSnrInto(g.data, meta, p, refMs, byTimes)
  doAssert byTimes[3].isNaN and byTimes[4] == byGuide[4]
  # A weaker event reads lower.
  var weak = shot(0.05, amp = 0.1)
  var w = newSeq[float64](weak.len)
  windowSnrInto(weak.data, initChunkMeta(weak.rows, dt), p, guide, w)
  doAssert abs(w[47] - (byGuide[47] - 20)) < 2, $w[47] & " vs " & $byGuide[47]

block fixed_windows_follow_the_time_axis:
  # A sample window (firstSample) and a t0Ms shift move the columns, not the
  # times: the same fixed windows give the same SNRs.
  var g = shot(0.2)
  let meta = initChunkMeta(g.rows, dt)
  let p = WindowSnrParams(signalStartMs: 300, signalEndMs: 500, noiseStartMs: 520,
    noiseEndMs: 700)
  var whole = newSeq[float64](g.len)
  windowSnrInto(g.data, meta, p, whole)
  var w = g.window(100, 400)
  let wm = initChunkMeta(w.rows, dt, firstSample = 100)
  var part = newSeq[float64](w.len)
  windowSnrInto(w.data, wm, p, part)
  for r in 0 ..< g.len: doAssert abs(part[r] - whole[r]) < 1e-9, $r
  for k in g.keys.mitems: k.t0Ms = 100
  var shifted = newSeq[float64](g.len)
  var p2 = p
  p2.signalStartMs += 100
  p2.signalEndMs += 100
  p2.noiseStartMs += 100
  p2.noiseEndMs += 100
  windowSnrInto(g.data, initChunkMeta(g.rows, dt), p2, shifted)
  for r in 0 ..< g.len: doAssert shifted[r] == whole[r], $r

block window_kernels:
  let x = [1.0'f32, 1, 2, 2]
  doAssert meanPower(x, 2 .. 3) == 4 and meanPower(x, -5 .. 1) == 1
  doAssert meanPower(x, 4 .. 9).isNaN and meanPower(x, 2 .. 1).isNaN
  doAssert abs(windowSnrDb(x, 2 .. 3, 0 .. 1) - 10 * log10(4.0)) < 1e-12
  doAssert abs(windowSnrDb(x, 2 .. 3, 0 .. 1, seExcess) - 10 * log10(3.0)) < 1e-12
  doAssert windowSnrDb(x, 0 .. 1, 2 .. 3, seExcess) == DefaultSnrFloorDb # no signal
  doAssert windowSnrDb(x, 0 .. 1, 2 .. 3, seExcess, floorDb = -Inf) == -Inf
  doAssert windowSnrDb(x, 0 .. 1, 2 .. 3, floorDb = 0) == 0 # clamped up
  doAssert powerSnrDb(1, 0).isNaN and powerSnrDb(0, 0).isNaN # no noise power
  doAssert powerSnrDb(0, 1) == DefaultSnrFloorDb
  doAssert powerSnrDb(1, 0, stabilization = 1) == 10 * log10(2.0)
  doAssert powerSnrDb(NaN, 1).isNaN and powerSnrDb(1, NaN).isNaN
  doAssert windowSnrDb(x, 0 .. 1, 9 .. 9).isNaN # empty noise window

# --------------------------------------------------------------- coherence --

const
  cohTraces = 96
  cohCols = 3000
  cohWin = 401
  cohDt = 1.0

proc signalTrace(n: int, seed: int64): seq[float64] =
  result = newSeq[float64](n)
  result.addNoise(1.0, seed)

proc cohGather[T](signal: openArray[float64], scale: float64, seed: int64,
    traces = cohTraces): SynthGather[T] =
  ## `traces` rows of `scale · signal` plus independent noise (σ = 1).
  result = initSynthGather[T](liveKeys(traces, cohDt), signal.len, cohDt)
  for r in 0 ..< traces:
    var x = newSeq[float64](signal.len)
    x.addNoise(1.0, seed + r)
    for j in 0 ..< x.len: x[j] += scale * signal[j]
    result.setRow(r, x)

proc windowPowerDb(s: openArray[float64], scale: float64, j, h: int): float64 =
  ## The signal's mean power over the window around `j`, in dB (σ² = 1).
  var p = 0.0
  let a = max(0, j - h)
  let b = min(s.high, j + h)
  for t in a .. b: p += (scale * s[t]) ^ 2
  10 * log10(p / float64(b - a + 1))

proc coherenceAccuracy[T]() =
  let s = signalTrace(cohCols, 7)
  var ws = initWorkspace()
  ws.reserve(coherenceSnrWorkspace(cohCols))
  var est = newSeq[float64](cohCols)
  for k, db in levelsDb:
    var g = cohGather[T](s, sqrt(pow(10.0, db / 10)), 1000 * k)
    let meta = initChunkMeta(g.rows, cohDt)
    coherenceSnrInto(g.data, meta, float64(cohWin - 1) * cohDt, est, ws)
    var worst = 0.0
    for j in 0 ..< cohCols:
      let want = windowPowerDb(s, sqrt(pow(10.0, db / 10)), j, cohWin div 2)
      worst = max(worst, abs(est[j] - want))
    doAssert worst < 1, $T & " at " & $db & " dB: off by " & $worst

block coherence_snr_within_1_db:
  coherenceAccuracy[float32]()
  coherenceAccuracy[float64]()

block coherence_noise_alone:
  # At most −10 dB everywhere, with a short window on a small gather too.
  var ws = initWorkspace()
  for (traces, win) in [(cohTraces, 400.0), (24, 40.0), (12, 40.0)]:
    var g = cohGather[float32](newSeq[float64](2000), 0, 77, traces)
    ws.reserve(coherenceSnrWorkspace(g.cols))
    var est = newSeq[float64](g.cols)
    coherenceSnrInto(g.data, initChunkMeta(g.rows, cohDt), win, est, ws)
    for j, v in est:
      doAssert v <= -10 and v >= DefaultSnrFloorDb, $traces & " col " & $j & ": " & $v
    var raw = newSeq[float64](g.cols)
    coherenceSnrInto(g.data, initChunkMeta(g.rows, cohDt), win, raw, ws, floorDb = -Inf)
    var floored = 0
    for v in raw:
      if v == -Inf: inc floored
    doAssert floored > 0 and floored < raw.len # an unbiased estimate: both signs

block coherence_versus_time_mutes_and_dead_rows:
  # Signal in the second half only; a top mute on a third of the rows; a
  # dead and a failed row full of large garbage are left out.
  let s = signalTrace(cohCols, 9)
  var half = s
  for j in 0 ..< cohCols div 2: half[j] = 0
  var g = cohGather[float64](half, 1.0, 300)
  for r in 0 ..< cohTraces div 3:
    for j in 0 ..< 2000: g.data[r, j] = 0
  var clean = g.clone
  for r in [10, 40]:
    for j in 0 ..< cohCols:
      g.data[r, j] = 1e6 * sin(float64(j))
      clean.data[r, j] = 0
  g.keys[10].status = tsDead
  let failed = @[40]
  var ws = initWorkspace()
  ws.reserve(coherenceSnrWorkspace(cohCols))
  var est = newSeq[float64](cohCols)
  coherenceSnrInto(g.data, initChunkMeta(g.rows, cohDt, failed = failed), 400, est, ws)
  var want = newSeq[float64](cohCols)
  coherenceSnrInto(clean.data, initChunkMeta(clean.rows, cohDt), 400, want, ws)
  for j in 0 ..< cohCols: doAssert est[j] == want[j], $j
  for j in 0 ..< cohCols div 2 - 200: doAssert est[j] <= -10, $j & ": " & $est[j]
  for j in cohCols div 2 + 200 ..< cohCols:
    doAssert abs(est[j] - windowPowerDb(s, 1, j, 200)) < 1, $j & ": " & $est[j]

block coherence_on_an_lmo_flattened_shot:
  # A linear event, flattened by LMO, is coherent at its time and nowhere else.
  var g = shot(0.2)
  var flat = g.clone
  let meta = initChunkMeta(flat.rows, dt)
  flat.data.applyLmo(meta, vel, biasMs = 100)
  var ws = initWorkspace()
  ws.reserve(coherenceSnrWorkspace(cols))
  var est = newSeq[float64](cols)
  coherenceSnrInto(flat.data, meta, 20, est, ws)
  let at = int((t0 + 100) / dt)
  doAssert est[at] > 3, $est[at]
  for j in at + 40 ..< cols - 100: doAssert est[j] <= -10, $j & ": " & $est[j]
  var raw = newSeq[float64](cols)
  coherenceSnrInto(g.data, meta, 20, raw, ws)
  doAssert raw[at] < est[at] - 5 # not flattened: no coherence there

block float32_matches_float64_and_no_allocation:
  let s = signalTrace(800, 3)
  let g32 = cohGather[float32](s, 0.5, 60, 24)
  let g64 = cohGather[float64](s, 0.5, 60, 24)
  let meta = initChunkMeta(g32.rows, cohDt)
  var ws = initWorkspace()
  ws.reserve(coherenceSnrWorkspace(800))
  var a = newSeq[float64](800)
  var b = newSeq[float64](800)
  coherenceSnrInto(g32.data, meta, 50, a, ws)
  coherenceSnrInto(g64.data, meta, 50, b, ws)
  for j in 0 ..< 800: doAssert abs(a[j] - b[j]) < 1e-3, $j
  doAssert allocsIn(coherenceSnrInto(g32.data, meta, 50, a, ws)) == 0
  var w = newSeq[float64](24)
  let p = WindowSnrParams(signalStartMs: 0, signalEndMs: 300, noiseStartMs: 400,
    noiseEndMs: 700)
  doAssert allocsIn(windowSnrInto(g32.data, meta, p, w)) == 0
  doAssert allocsIn(windowSnrInto(g32.data, meta, p, linearMoveout(2000), w)) == 0

block coherence_nan_cases:
  var ws = initWorkspace()
  ws.reserve(coherenceSnrWorkspace(50))
  var est = newSeq[float64](50)
  var one = cohGather[float32](signalTrace(50, 1), 1, 5, traces = 1)
  coherenceSnrInto(one.data, initChunkMeta(one.rows, cohDt), 10, est, ws)
  for v in est: doAssert v.isNaN # one trace: no noise estimate
  var same = initSynthGather[float32](liveKeys(4, cohDt), 50, cohDt)
  for r in 0 ..< 4:
    for j in 0 ..< 50: same.data[r, j] = float32(1 + j mod 3)
  coherenceSnrInto(same.data, initChunkMeta(same.rows, cohDt), 10, est, ws)
  for v in est: doAssert v == Inf # noise-free identical traces
  var zero = initSynthGather[float32](liveKeys(4, cohDt), 50, cohDt)
  coherenceSnrInto(zero.data, initChunkMeta(zero.rows, cohDt), 10, est, ws)
  for v in est: doAssert v.isNaN # nothing live to measure
  var bad = cohGather[float32](signalTrace(50, 1), 1, 5, traces = 4)
  bad.data[2, 20] = NaN
  coherenceSnrInto(bad.data, initChunkMeta(bad.rows, cohDt), 10, est, ws)
  doAssert est[20].isNaN and est[15].isNaN and not est[14].isNaN and not est[26].isNaN

proc coherenceByDefinition(d: Tensor[float64], cols, h: int, floorDb: float64): seq[float64] =
  ## The estimator of the module documentation, each column's window summed
  ## on its own (every row live from its first to its last non-zero sample).
  let rows = d.shape[0]
  var c, res, cnt = newSeq[float64](cols)
  for j in 0 ..< cols:
    var sum = 0.0
    for r in 0 ..< rows:
      var first = 0
      while first < cols and d[r, first] == 0: inc first
      var last = cols - 1
      while last >= 0 and d[r, last] == 0: dec last
      if j in first .. last:
        sum += d[r, j]
        cnt[j] += 1
    if cnt[j] > 0:
      let m = sum / cnt[j]
      for r in 0 ..< rows:
        var first = 0
        while first < cols and d[r, first] == 0: inc first
        var last = cols - 1
        while last >= 0 and d[r, last] == 0: dec last
        if j in first .. last: res[j] += (d[r, j] - m) * (d[r, j] - m)
      c[j] = cnt[j] * m * m
  for j in 0 ..< cols:
    var sr, sc, sn, sw = 0.0
    for t in max(0, j - h) .. min(cols - 1, j + h):
      if cnt[t] > 0:
        sr += res[t]
        sc += c[t]
        sn += cnt[t]
        sw += 1
    let dof = sn - sw
    if not (dof > 0): result.add NaN
    else:
      let noise = sr / dof
      let signal = (sc - sw * noise) / sn
      result.add (if noise.isNaN or signal.isNaN: NaN
        elif noise == 0: (if signal > 0: Inf else: NaN)
        elif signal / noise > 0: max(10 * log10(signal / noise), floorDb)
        else: floorDb)

block coherence_window_sums_against_the_definition:
  # The window sums come from block sums, in a time that does not depend
  # on the window. Against a loop over every window: equal within 1e-9 dB
  # for windows of 1 sample to more than the block, with mutes (columns of
  # fewer rows, and of none), a loud stretch before a quiet one, and a NaN,
  # which spoils the windows that hold it and no other.
  const (rows, cols) = (7, 157)
  var keys = liveKeys(rows, cohDt)
  var g = initSynthGather[float64](keys, cols, cohDt)
  for r in 0 ..< rows:
    var x = newSeq[float64](cols)
    x.addNoise(1, 800 + r)
    x.addSinusoid(cohDt, 20, amp = 2)
    for j in 0 ..< cols:
      g.data[r, j] = x[j] * (if j < 40: 1e4 else: 1.0) # loud, then quiet
    for j in 0 ..< 13 * (r mod 3): g.data[r, j] = 0 # top mutes
    for j in cols - 9 * (r mod 2) ..< cols: g.data[r, j] = 0
  for r in 0 ..< rows:
    for j in 100 .. 103: g.data[r, j] = (if r > 0: 0.0 else: g.data[r, j]) # one row only
  let meta = initChunkMeta(g.rows, cohDt)
  var ws = initWorkspace()
  ws.reserve(coherenceSnrWorkspace(cols))
  var est = newSeq[float64](cols)
  for withNaN in [false, true]:
    if withNaN: g.data[3, 70] = NaN
    for h in [0, 1, 2, 5, 20, 78, 79, 156, 157, 400]:
      let windowMs = max(float64(2 * h) * cohDt, 0.5 * cohDt)
      for floorDb in [-40.0, -Inf]:
        coherenceSnrInto(g.data, meta, windowMs, est, ws, floorDb = floorDb)
        let want = coherenceByDefinition(g.data, cols, min(h, cols), floorDb)
        for j in 0 ..< cols:
          doAssert (est[j].isNaN and want[j].isNaN) or est[j] == want[j] or
            abs(est[j] - want[j]) < 1e-9, "h " & $h & " column " & $j & ": " & $est[j] &
            " vs " & $want[j]
          if withNaN: doAssert est[j].isNaN == (abs(j - 70) <= h), $h & " " & $j
  # A workspace reserved for a narrower block is refused.
  var small = initWorkspace()
  small.reserve(coherenceSnrWorkspace(cols - 1))
  doAssertRaises ValueError: coherenceSnrInto(g.data, meta, 20, est, small)

block errors:
  var g = shot(0.1)
  let meta = initChunkMeta(g.rows, dt)
  var ws = initWorkspace()
  ws.reserve(coherenceSnrWorkspace(cols))
  var est = newSeq[float64](cols)
  var short = newSeq[float64](cols - 1)
  doAssertRaises ValueError: coherenceSnrInto(g.data, meta, 0, est, ws)
  doAssertRaises ValueError: coherenceSnrInto(g.data, meta, Inf, est, ws)
  doAssertRaises ValueError: coherenceSnrInto(g.data, meta, 20, est, ws, floorDb = NaN)
  doAssertRaises ValueError: coherenceSnrInto(g.data, meta, 20, short, ws)
  doAssertRaises ValueError:
    coherenceSnrInto(g.data, meta.sub(0 .. 9), 20, est, ws)
  var dst = newSeq[float64](g.len)
  doAssertRaises ValueError:
    windowSnrInto(g.data, meta, WindowSnrParams(signalStartMs: 10, signalEndMs: 0), dst)
  doAssertRaises ValueError:
    windowSnrInto(g.data, meta, WindowSnrParams(noiseStartMs: NaN), dst)
  doAssertRaises ValueError:
    windowSnrInto(g.data, meta, WindowSnrParams(floorDb: Inf), dst)
  doAssertRaises ValueError: windowSnrInto(g.data, meta, WindowSnrParams(), short)
  doAssertRaises ValueError:
    windowSnrInto(g.data, meta, WindowSnrParams(), newSeq[float64](3), dst)
  doAssertRaises ValueError: windowSnrInto(g.data, meta, WindowSnrParams(), Moveout(), dst)
