## Amplitude QC (qc/amplitude): a synthetic record with injected dead,
## constant, clipped, spiky, DC-offset, noisy, weak and NaN traces is
## flagged with exactly those flags; header-dead and failed rows are skipped;
## windows, short traces, the record statistics, the pool, float32 and
## float64, no allocation, and parameter errors.

import std/[assertions, math]
import processSEG
import parseSEG
import utils/[allocs, synth]

const
  traces = 40
  cols = 1001
  dtMs = 2.0

type Injected = enum
  inDead, inConstant, inClipped, inSpiky, inDc, inNoisy, inWeak, inNan,
  inHeaderDead, inFailed

const injectedRow: array[Injected, int] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

func expected(r: int): set[AmplitudeFlag] =
  for k, row in injectedRow:
    if row == r:
      return case k
        of inDead: {afDead}
        of inConstant: {afConstant}
        of inClipped: {afClipped}
        of inSpiky: {afSpiky}
        of inDc: {afDcOffset}
        of inNoisy: {afNoisy}
        of inWeak: {afWeak}
        of inNan: {afNonFinite}
        of inHeaderDead, inFailed: {}
  {}

proc baseTrace(r: int): seq[float64] =
  ## A Ricker wavelet at 200 + 10·r ms, its gain varying ±2 dB from trace
  ## to trace (a field record's natural scatter), in weak white noise.
  result = newSeq[float64](cols)
  result.addRicker(dtMs, centreMs = 200 + 10 * float64(r), freqHz = 25,
    amp = pow(10.0, 2 * sin(1.7 * float64(r)) / 20))
  result.addNoise(0.02, 1000 + r)

proc record[T](): SynthGather[T] =
  var keys = newSeq[TraceKeys](traces)
  for r in 0 ..< traces: keys[r] = stationKeys(0, 25 * float64(r + 1), 0, 0, dtMs)
  keys[injectedRow[inHeaderDead]].status = tsDead
  result = initSynthGather[T](keys, cols, dtMs)
  for r in 0 ..< traces:
    var x = baseTrace(r)
    if r == injectedRow[inDead]:
      for v in x.mitems: v = 0
    elif r == injectedRow[inConstant]:
      for v in x.mitems: v = 0.7
    elif r == injectedRow[inClipped]:
      var m = 0.0
      for v in x: m = max(m, abs(v))
      for v in x.mitems: v = clamp(v, -0.5 * m, 0.5 * m)
    elif r == injectedRow[inSpiky]:
      x[1600 div 2] += 1.0 # 1600 ms, far from the wavelet
    elif r == injectedRow[inDc]:
      var s = 0.0
      for v in x: s += v * v
      x.addDc(sqrt(s / cols)) # one standard deviation
    elif r == injectedRow[inNoisy]:
      x.addNoise(2.0, 77)     # some +25 dB
    elif r == injectedRow[inWeak]:
      for v in x.mitems: v *= 0.01
    elif r == injectedRow[inNan]:
      x[100] = NaN
      x[101] = Inf
      x[900] = NaN
    for j in 0 ..< cols: result.data[r, j] = T(x[j])

func same(a, b: float64): bool = a == b or (a.isNaN and b.isNaN)

func same(a, b: TraceAmplitude): bool =
  ## Equal, NaN equal to NaN.
  a.samples == b.samples and same(a.rmsDb, b.rmsDb) and same(a.maxDb, b.maxDb) and
    same(a.mean, b.mean) and a.nonFinite == b.nonFinite and a.clipped == b.clipped and
    a.spikes == b.spikes and same(a.robustZ, b.robustZ) and a.flags == b.flags

proc checkRecord[T]() =
  var g = record[T]()
  let failed = @[injectedRow[inFailed]]
  let meta = initChunkMeta(g.rows, dtMs, failed = failed)
  var amps = newSeq[TraceAmplitude](traces)
  traceAmplitudeInto(g.data, meta, AmplitudeParams(), amps)
  var ws = initWorkspace()
  ws.reserve(recordAmplitudeWorkspace(traces))
  let rec = recordAmplitude(amps, AmplitudeParams(), ws)
  for r in 0 ..< traces:
    doAssert amps[r].flags == expected(r),
      $T & " row " & $r & ": " & $amps[r].flags & " (z " & $amps[r].robustZ & ")"
  for k in [inHeaderDead, inFailed]:
    let a = amps[injectedRow[k]]
    doAssert a.samples == 0 and a.rmsDb.isNaN and a.robustZ.isNaN
  doAssert amps[injectedRow[inNan]].nonFinite == 3
  doAssert amps[injectedRow[inNan]].samples == cols
  doAssert amps[injectedRow[inSpiky]].spikes == 1
  doAssert amps[injectedRow[inClipped]].clipped >= 3, $amps[injectedRow[inClipped]].clipped
  doAssert amps[injectedRow[inDead]].rmsDb == -Inf
  doAssert abs(amps[injectedRow[inConstant]].rmsDb - 20 * log10(0.7)) < 1e-5
  doAssert abs(amps[injectedRow[inConstant]].mean - 0.7) < 1e-6
  # Statistics: every live trace but the dead, constant, header-dead, failed.
  doAssert rec.traces == traces - 4
  doAssert rec.flagged == 8
  for f in AmplitudeFlag: doAssert rec.counts[f] == 1, $f
  doAssert rec.madDb > 0.5 and rec.madDb < 2.5, $rec.madDb
  doAssert amps[injectedRow[inNoisy]].robustZ > 10
  doAssert amps[injectedRow[inWeak]].robustZ < -10
  # The pool gives the same.
  var pooled = newSeq[TraceAmplitude](traces)
  var pool = newProcessPool(3)
  pool.traceAmplitudeInto(g.data, meta, AmplitudeParams(), pooled)
  discard recordAmplitude(pooled, AmplitudeParams(), ws)
  for r in 0 ..< traces: doAssert same(pooled[r], amps[r]), $r
  # No allocation once reserved.
  doAssert allocsIn(traceAmplitudeInto(g.data, meta, AmplitudeParams(), amps)) == 0
  doAssert allocsIn((discard recordAmplitude(amps, AmplitudeParams(), ws))) == 0

block record_flags_float32:
  checkRecord[float32]()

block record_flags_float64:
  checkRecord[float64]()

block record_flags_recomputed:
  # recordAmplitude clears weak / noisy from an earlier call.
  var amps = @[TraceAmplitude(samples: 10, rmsDb: 0, flags: {afWeak}),
    TraceAmplitude(samples: 10, rmsDb: 0), TraceAmplitude(samples: 10, rmsDb: 0)]
  var ws = initWorkspace()
  ws.reserve(recordAmplitudeWorkspace(3))
  let rec = recordAmplitude(amps, AmplitudeParams(), ws)
  doAssert rec.flagged == 0 and rec.madDb == 0 and amps[0].robustZ == 0
  amps[2].rmsDb = 0.1 # MAD 0: any difference is an outlier
  discard recordAmplitude(amps, AmplitudeParams(), ws)
  doAssert amps[2].flags == {afNoisy} and amps[2].robustZ == Inf

block empty_record:
  var amps: seq[TraceAmplitude]
  var ws = initWorkspace()
  let rec = recordAmplitude(amps, AmplitudeParams(), ws)
  doAssert rec.traces == 0 and rec.medianRmsDb.isNaN and rec.flagged == 0

block windows:
  # A spike outside the window is not seen; window times follow t0Ms.
  var g = record[float32]()
  let meta = initChunkMeta(g.rows, dtMs)
  var amps = newSeq[TraceAmplitude](traces)
  let r = injectedRow[inSpiky]
  traceAmplitudeInto(g.data, meta, AmplitudeParams(startMs: 0, endMs: 1000), amps)
  doAssert amps[r].flags == {} and amps[r].samples == 501
  traceAmplitudeInto(g.data, meta, AmplitudeParams(startMs: 1500, endMs: 1700), amps)
  doAssert amps[r].flags == {afSpiky} and amps[r].samples == 101
  g.keys[r].t0Ms = 1000 # the spike now at 2600 ms
  traceAmplitudeInto(g.data, meta, AmplitudeParams(startMs: 1500, endMs: 1700), amps)
  doAssert amps[r].flags == {} and amps[r].samples == 101
  g.keys[r].t0Ms = NaN # a finite window needs t0
  traceAmplitudeInto(g.data, meta, AmplitudeParams(startMs: 1500, endMs: 1700), amps)
  doAssert amps[r].samples == 0 and amps[r].rmsDb.isNaN
  traceAmplitudeInto(g.data, meta, AmplitudeParams(), amps)
  doAssert amps[r].samples == cols and amps[r].flags == {afSpiky}
  # A short trace: its live samples only.
  g.setLive(r, 300)
  traceAmplitudeInto(g.data, meta, AmplitudeParams(), amps)
  doAssert amps[r].samples == 300 and amps[r].flags == {}

block single_trace:
  var x = newSeq[float32](100)
  doAssert traceAmplitude(x, 1).flags == {afDead}
  x[50] = 1
  let a = traceAmplitude(x, 1)
  doAssert a.flags == {afSpiky} and a.spikes == 1, $a # 1 among zeros
  doAssert a.maxDb == 0
  let b = traceAmplitude(x, 1, AmplitudeParams(spikeFactor: Inf, dcRatio: Inf))
  doAssert b.flags == {}
  # Clip runs: three samples at the maximum, positive and negative.
  var y = newSeq[float64](64)
  y.addSinusoid(1, 37)
  for j in 10 .. 11: y[j] = 2
  y[12] = -2
  let c = traceAmplitude(y, 1, AmplitudeParams(spikeFactor: Inf))
  doAssert c.clipped == 3 and c.flags == {afClipped}
  doAssert traceAmplitude(y, 1, AmplitudeParams(clipRun: 4, spikeFactor: Inf)).flags == {}
  var allNan = [NaN, NaN]
  let d = traceAmplitude(allNan, 1)
  doAssert d.flags == {afNonFinite} and d.rmsDb.isNaN and d.nonFinite == 2
  let e = traceAmplitude(newSeq[float32](0), 1)
  doAssert e.samples == 0 and e.flags == {} and e.rmsDb.isNaN

block huge_spikes:
  # Two opposite spikes of 1e12 on noise: each found once, the sliding sum
  # recovers after each one leaves the window.
  var x = newSeq[float32](2000)
  x.addNoise(1, 5)
  x[100] += 1e12
  x[1000] -= 1e12
  let a = traceAmplitude(x, 1)
  doAssert a.spikes == 2 and afSpiky in a.flags, $a.spikes
  # Two spikes in one window: both seen once the factor allows.
  var y = newSeq[float64](400)
  y.addNoise(1, 6)
  y[200] += 1000
  y[205] += 1000
  doAssert traceAmplitude(y, 1, AmplitudeParams(spikeFactor: 4)).spikes == 2

block inf_is_not_clipped:
  var x = newSeq[float64](100)
  x.addSinusoid(1, 37)
  for j in 50 ..< 55: x[j] = Inf
  doAssert traceAmplitude(x, 1, AmplitudeParams(spikeFactor: Inf)).flags == {afNonFinite}

block errors:
  var x = newSeq[float32](10)
  for p in [AmplitudeParams(startMs: 10, endMs: 0), AmplitudeParams(startMs: NaN),
      AmplitudeParams(clipRun: 1), AmplitudeParams(clipTolerance: 1),
      AmplitudeParams(spikeFactor: 0), AmplitudeParams(spikeWindowMs: Inf),
      AmplitudeParams(dcRatio: NaN), AmplitudeParams(outlierZ: 0)]:
    doAssertRaises(ValueError): discard traceAmplitude(x, 1, p)
  doAssertRaises(ValueError): discard traceAmplitude(x, 0)
  var g = record[float32]()
  let meta = initChunkMeta(g.rows, dtMs)
  var short = newSeq[TraceAmplitude](traces - 1)
  doAssertRaises(ValueError): traceAmplitudeInto(g.data, meta, AmplitudeParams(), short)
  var amps = newSeq[TraceAmplitude](4)
  var small = initWorkspace()
  doAssertRaises(ValueError): discard recordAmplitude(amps, AmplitudeParams(), small)
  var pool = newProcessPool(2)
  doAssertRaises(ValueError):
    pool.traceAmplitudeInto(g.data, meta, AmplitudeParams(), short)
