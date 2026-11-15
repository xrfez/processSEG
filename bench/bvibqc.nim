## Vibrator QC (Phase 15) on a wireline record: one unit of four rows (Ref,
## GF, M, B), an 8-80 Hz sweep of 8 s at 1 ms plus a 2 s listen time
## (10001 samples), float32, one thread. Each call as the viewer makes it
## when a plot opens: the reference prepared once, then one signal tracked,
## correlated, compared, or its time-frequency picture.

import std/[math, strutils]
import benchutil
import parseSEG, processSEG

const
  dt = 1.0
  n = 10001

proc ms(r: BenchResult): string = formatFloat(r.medianNs / 1e6, ffDecimal, 3) & " ms"

proc row(spec: SweepSpec, amp, phaseDeg, h2: float64): seq[float32] =
  ## The sweep scaled, shifted and with a 2nd harmonic, in a record of `n`.
  result = newSeq[float32](n)
  for j in 0 ..< spec.sweepLen:
    let t = float64(j) * dt
    let w = (if t < 300: taperWeight(tpCos2, t / 300) elif t > 7700:
      taperWeight(tpCos2, (8000 - t) / 300) else: 1.0)
    let p = spec.phaseAt(t)
    result[j] = float32(amp * w * (sin(p + phaseDeg * PI / 180) + h2 * sin(2 * p)))

proc main() =
  printHeader "vibrator QC, one unit of 4 rows × " & $n & " samples f32, 1 thread"
  let spec = linearSweep(8, 80, 8000, dt, initSweepTaper(300, 300))
  let rows = [row(spec, 1, 0, 0), row(spec, 4e4, 3, 0.05), row(spec, 20, 5, 0.02),
    row(spec, 15, -175, 0.04)]
  var keys = newSeq[TraceKeys](4)
  var counts = newSeq[int](4)
  var data = newTensor[float32](4, n)
  for r in 0 ..< 4:
    keys[r] = initTraceKeys()
    keys[r].status = tsAuxiliary
    keys[r].t0Ms = 0
    counts[r] = n
    for j in 0 ..< n: data[r, j] = rows[r][j]
  let meta = initChunkMeta(rowsOf(keys, counts), dtMs = dt)
  var sink = 0.0

  let tp = initSweepTrackPlan(float32, TrackParams(), dt, n)
  var ws = initWorkspace()
  ws.reserve(vibratorWorkspace(tp))
  var reference = initSweepReference(tp)
  var track = initSweepTrack(tp)
  let a = bench("referenceInto (the pilot: analytic signal, windows)", 0):
    tp.referenceInto(rows[0], reference, ws)
  a.report
  echo "    ", a.ms
  let b = bench("trackInto, harmonics 1-5 (8 lanes), 1001 points", 0):
    tp.trackInto(rows[1], reference, track, ws)
    sink += track.thdPct[400]
  b.report
  echo "    ", b.ms
  let tp12 = initSweepTrackPlan(float32, TrackParams(harmonics: 12), dt, n)
  var ws12 = initWorkspace()
  ws12.reserve(tp12.workspaceSize)
  var r12 = initSweepReference(tp12)
  var t12 = initSweepTrack(tp12)
  tp12.referenceInto(rows[0], r12, ws12)
  let c = bench("trackInto, harmonics 1-12 (16 lanes)", 0):
    tp12.trackInto(rows[1], r12, t12, ws12)
    sink += t12.thdPct[400]
  c.report
  echo "    ", c.ms
  var units = vibUnitsInOrder(1, reactionMass = mass(2450, muKilogram),
    baseplateMass = mass(1180, muKilogram))
  var results: seq[UnitSignature[float32]]
  tp.analyzeVibrators(data, meta, units, VibLimits(phaseDeg: 5), results, ws)
  let d = bench("analyzeVibrators, one unit (4 rows + GF from M and B)", 0):
    tp.analyzeVibrators(data, meta, units, VibLimits(phaseDeg: 5), results, ws)
    sink += results[0].summaries[vsGroundForce].meanForce
  d.report
  echo "    ", d.ms

  let cp = initCrossCorrelationPlan(float32, maxLagMs = 500, dtMs = dt, samples = n)
  var cws = initWorkspace()
  cws.reserve(cp.workspaceSize)
  var lags, env = newSeq[float32](cp.lags)
  var amp, phase = newSeq[float64](cp.bins)
  let e = bench("crossCorrelateInto, lags ±500 ms, with the envelope", 0):
    cp.crossCorrelateInto(rows[1], rows[0], lags, env, cws)
    sink += float64(env[500])
  e.report
  echo "    ", e.ms
  let f = bench("lagSpectrumInto (1001 lags, 1024 points)", 0):
    cp.lagSpectrumInto(lags, amp, phase, cws)
    sink += amp[40]
  f.report
  echo "    ", f.ms
  let g = bench("compareInto (two signals against frequency)", 0):
    cp.compareInto(rows[1], rows[0], amp, phase, cws)
    sink += phase[40]
  g.report
  echo "    ", g.ms

  let sg = initSpectrogramPlan(float32, windowLen = 255, hop = 10, dtMs = dt, samples = n)
  var sws = initWorkspace()
  sws.reserve(sg.workspaceSize)
  var pic = newSeq[float64](sg.frames * sg.bins)
  let h = bench("spectrogramInto, 255-sample frames every 10 ms (1001 × 129)", 0):
    sg.spectrogramInto(rows[1], pic, sws)
    sink += pic[1000]
  h.report
  echo "    ", h.ms
  let hp = initHilbertPlan(float32, n)
  var hws = initWorkspace()
  hws.reserve(hp.workspaceSize)
  var envelope = newSeq[float32](n)
  let k = bench("envelopeInto (Hilbert, nfft " & $hp.nfft & ")", 0):
    hp.envelopeInto(rows[1], envelope, hws)
    sink += float64(envelope[3000])
  k.report
  echo "    ", k.ms
  doAssert sink == sink

main()
