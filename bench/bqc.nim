## QC and vibroseis throughput, one thread: amplitude QC of a panel, the
## coherence SNR of a gather against its window, vibroseis correlation (a
## block of records, and one record), the per-shot refraction fit and the
## resampling of a pilot.

import std/[math, random, strutils]
import benchutil
import processSEG
import parseSEG

const dt = 2.0

proc ms(r: BenchResult): string = formatFloat(r.medianNs / 1e6, ffDecimal, 3) & " ms"

proc liveMeta(keys: var seq[TraceKeys], counts: var seq[int], rows, cols: int): ChunkMeta =
  keys = newSeq[TraceKeys](rows)
  counts = newSeq[int](rows)
  for r in 0 ..< rows:
    keys[r] = initTraceKeys()
    keys[r].status = tsLive
    keys[r].t0Ms = 0
    keys[r].offset = 10 * float64(r + 1)
    counts[r] = cols
  initChunkMeta(rowsOf(keys, counts), dtMs = dt)

proc amplitude() =
  ## 2000 × 3000 float32: decaying noise, a spike on every fifth trace, every
  ## seventh clipped, every eleventh with a DC offset.
  const (rows, cols) = (2000, 3000)
  var keys: seq[TraceKeys]
  var counts: seq[int]
  let meta = liveMeta(keys, counts, rows, cols)
  var x = zeros[float32](rows, cols)
  var rng = initRand(1)
  for r in 0 ..< rows:
    for j in 0 ..< cols: x[r, j] = float32(rng.gauss(0, 1.0) * exp(-float64(j) / 1500))
    if r mod 5 == 0: x[r, rng.rand(0 ..< cols)] = float32(rng.gauss(0, 300))
    if r mod 7 == 0:
      for j in 0 ..< cols: x[r, j] = clamp(x[r, j], -1.5, 1.5)
    if r mod 11 == 0:
      for j in 0 ..< cols: x[r, j] = x[r, j] + 2
  var amps = newSeq[TraceAmplitude](rows)
  var ws = initWorkspace()
  when declared(traceAmplitudeWorkspace): ws.reserve(traceAmplitudeWorkspace(cols))
  let r = bench("traceAmplitudeInto 2000 × 3000 f32", rows * cols * 4):
    when declared(traceAmplitudeWorkspace):
      traceAmplitudeInto(x, meta, AmplitudeParams(), amps, ws)
    else:
      traceAmplitudeInto(x, meta, AmplitudeParams(), amps)
  r.report
  var spikes, clipped = 0
  for a in amps:
    spikes += a.spikes
    clipped += a.clipped
  echo "    ", r.ms, " per panel (", spikes, " spikes, ", clipped, " clipped samples)"

proc coherence() =
  ## A gather of 24 × 6000: the cost against the window.
  const (rows, cols) = (24, 6000)
  var keys: seq[TraceKeys]
  var counts: seq[int]
  let meta = liveMeta(keys, counts, rows, cols)
  var x = zeros[float32](rows, cols)
  var rng = initRand(1)
  for r in 0 ..< rows:
    for j in 0 ..< cols: x[r, j] = float32(rng.gauss(0, 1))
  var ws = initWorkspace()
  ws.reserve(coherenceSnrWorkspace(cols))
  var dst = newSeq[float64](cols)
  for windowMs in [100.0, 1000.0, 12000.0]:
    let r = bench("coherenceSnrInto 24 × 6000, window " & $int(windowMs) & " ms",
        rows * cols * 4):
      coherenceSnrInto(x, meta, windowMs, dst, ws)
    r.report
    echo "    ", r.ms, " per gather"

proc correlation() =
  ## 200 records of 20 s at 2 ms with a 12 s sweep (nfft 16200), then one.
  const (rows, cols) = (200, 10001)
  let pilot = generatePilot(linearSweep(6, 100, 12000, dt, initSweepTaper(300, 300)), float32)
  var keys: seq[TraceKeys]
  var counts: seq[int]
  let meta = liveMeta(keys, counts, rows, cols)
  var raw = zeros[float32](rows, cols)
  var rng = initRand(3)
  for r in 0 ..< rows:
    for j in 0 ..< cols: raw[r, j] = float32(rng.rand(2.0) - 1.0)
  let plan = initCorrelationPlan(float32, pilot, dt, cols)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  var dst = zeros[float32](rows, plan.outSamples)
  let b = bench("correlateInto 200 × 10001, 12 s sweep (nfft " & $plan.nfft & ")",
      rows * cols * 4):
    plan.correlateInto(raw, dst, meta, ws)
  b.report
  echo "    ", b.ms, " per block, ", formatFloat(b.medianNs / 1e3 / float64(rows), ffDecimal, 1),
    " µs per trace"
  var x = newSeq[float32](cols)
  for j in 0 ..< cols: x[j] = raw[0, j]
  var y = newSeq[float32](plan.outSamples)
  let one = bench("correlateInto, one record of 10001", cols * 4):
    plan.correlateInto(x, y, ws)
  one.report
  echo "    ", formatFloat(one.medianNs / 1e3, ffDecimal, 1), " µs per record"

proc shotFit() =
  ## 2000 traces: the fit on picks, both models.
  const rows = 2000
  var keys: seq[TraceKeys]
  var counts: seq[int]
  let meta = liveMeta(keys, counts, rows, 1000)
  var picks = newSeq[FirstBreakPick](rows)
  var rng = initRand(5)
  for r in 0 ..< rows:
    let d = keys[r].offset
    picks[r] = FirstBreakPick(timeMs: min(d / 1.5, d / 3.0 + 80) + rng.gauss(0, 1), col: 0,
      snrDb: 30, confidence: 1)
  var fit: ShotFit
  var sink = 0
  for model in [rmTwoSegment, rmOffsetBinned]:
    let r = bench("fitShot " & $model & ", 2000 picks", 0):
      fitShot(meta, picks, ShotFitParams(model: model), fit)
      sink += fit.used
    r.report
    echo "    ", r.ms, " per shot"
  if sink == 42: echo ""

proc resampling() =
  ## A 30 s pilot at 0.25 ms to 4 ms and to 0.5 ms.
  let fine = generatePilot(linearSweep(4, 100, 30000, 0.25, initSweepTaper(500, 500),
    maxNyquistFraction = 0.2), float32)
  var sink = 0
  for target in [4.0, 0.5]:
    let r = bench("resamplePilot 30 s, 0.25 → " & $target & " ms", fine.len * 4):
      sink += fine.resamplePilot(target).len
    r.report
    echo "    ", r.ms
  if sink == 42: echo ""

proc main() =
  printHeader "QC and vibroseis, 1 thread"
  amplitude()
  coherence()
  correlation()
  shotFit()
  resampling()

main()
