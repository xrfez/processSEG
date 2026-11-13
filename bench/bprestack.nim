## Pre-stack throughput (Phases 6–8, task 9.2), one thread, a panel of 2000
## traces × 2001 samples of float32 at 2 ms (16 MB): statics, LMO, NMO
## (linear and cubic), stacking into bins, first-break picking (three methods,
## with and without the band-pass) and the per-shot refraction fit. Then a
## brute stack of a shot-sorted line written to a temporary SEG-Y file
## (12 000 traces × 1001 samples, 48 MB), with 1 and 4 workers: a velocity
## function, a 2D field, and datum statics with a band-pass and cubic NMO.

import std/[options, os, strutils]
import benchutil
import parseSEG, processSEG
import ../tests/utils/synth

const
  traces = 2000
  ns = 2001
  dt = 2.0

proc shot(): SynthGather[float32] =
  ## A split-spread shot: a refracted arrival plus a reflection over noise.
  var rel = newSeq[float64](traces)
  for i in 0..<traces: rel[i] = 5.0 * float64(i - traces div 2)
  result = shotGather[float32](0, rel, ns, dt)
  result.addLinear(30, 2500, 25)
  result.addHyperbola(1200, 2200, 25)
  for r in 0..<traces:
    var x = newSeq[float32](ns)
    x.addNoise(0.05, seed = r)
    for j in 0..<ns: result.data[r, j] += x[j]

proc ms(r: BenchResult): string = formatFloat(r.medianNs / 1e6, ffDecimal, 3) & " ms per panel"

const
  lineShots = 60
  lineChannels = 200
  lineNs = 1001

proc writeLine(path: string): int =
  ## A shot-sorted 2D line: split spreads of `lineChannels` at 25 m, shots
  ## every 50 m, CDPs every 12.5 m numbered from 1 in `ensemble`, stations
  ## up to 15 m about a datum of 100 m. Returns the number of CDPs.
  let rel = splitSpread(lineChannels div 2, 25, 12.5)
  var w = openSeismicWriter(path, SegyOutputOptions(traceSorting: 1))
  var x = newSeq[float32](lineNs)
  for s in 0..<lineShots:
    let srcX = 10_000 + 50 * float64(s)
    for i, r in rel:
      var k = stationKeys(srcX, srcX + r, 100 + float64(s mod 7) * 2,
        100 + float64(i mod 11) - 5, dt)
      k.record = float64(s + 1)
      k.channel = float64(i + 1)
      k.ensemble = (k.cmpX - 10_000 + float64(lineChannels div 2) * 25) / 12.5 + 1
      k.cmpY = 0
      result = max(result, int(k.ensemble))
      for j in 0..<lineNs: x[j] = 0
      for t0 in [300.0, 700, 1200]:
        x.addRicker(dt, hyperbolicMs(t0, 2000 + t0 * 0.4, r), 30)
      w.writeTrace(k, x)
  doAssert not w.close().isFatal

proc stackBench() =
  let path = getTempDir() / ("processseg_bprestack_" & $getCurrentProcessId() & ".sgy")
  let cdps = writeLine(path)
  var f = openSeismic(path, keys = true)
  let bytes = f.len * lineNs * 4
  let binning = cdpBinning(1, cdps)
  let velocity = gradientVelocity(1800, 600)
  let field = initVelocityField([1.0, float64(cdps) / 2, float64(cdps)], [velocity,
    gradientVelocity(1900, 600), velocity], dtMs = 4, tMaxMs = 2000)
  var sink = 0
  for workers in [1, 4]:
    var opts = initBruteStackOptions(workers = workers)
    let tag = ", " & $workers & (if workers == 1: " worker" else: " workers")
    let a = bench("bruteStack, a function" & tag, bytes):
      sink += bruteStack(f, float32, binning, velocity, opts).stacked
    a.report
    let b = bench("bruteStack, a 2D field" & tag, bytes):
      sink += bruteStack(f, float32, binning, field, opts).stacked
    b.report
    opts.statics = some(datumStatics(100, 2000))
    opts.flow = initPipeline([bandPassOp(Corners(f1: 4, f2: 8, f3: 60, f4: 80))])
    opts.nmo = initNmoParams(siCubic)
    let c = bench("bruteStack, statics + band-pass + cubic" & tag, bytes):
      sink += bruteStack(f, float32, binning, velocity, opts).stacked
    c.report
  doAssert sink > 0
  f.close()
  removeFile(path)

proc main() =
  printHeader "pre-stack, " & $traces & " traces × " & $ns & " samples f32 at 2 ms, 1 thread"
  let g0 = shot()
  var g = g0.clone
  let meta = initChunkMeta(g.rows, dtMs = dt)
  let bytes = traces * ns * 4
  var ws = initWorkspace()
  var sink: float64

  var statics = newSeq[float64](traces)
  for i in 0..<traces: statics[i] = 0.37 * float64(i mod 17) - 3
  for interp in [siLinear, siCubic]:
    let r = bench("applyStatics " & $interp, 2 * bytes):
      g.data.applyStatics(meta, statics, interp)
    r.report
    echo "    ", r.ms
  let lmo = bench("applyLmo 2500 m/s", 2 * bytes):
    g.data.applyLmo(meta, 2500)
    g.data.removeLmo(meta, 2500)
  lmo.report
  echo "    ", lmo.ms, " (apply + remove)"

  ws.reserve(nmoWorkspace(float32, ns))
  let vf = rmsVelocity([VelocityKnot(tMs: 0, vrms: 1800),
                        VelocityKnot(tMs: 4000, vrms: 3200)])
  for interp in [siLinear, siCubic]:
    let p = initNmoParams(interp, maxStretch = 0.5)
    let r = bench("applyNmo " & $interp, 2 * bytes):
      g.data.applyNmo(meta, vf, p, ws)
    r.report
    echo "    ", r.ms

  var acc = initStackAccumulator[float32](bins = 200, samples = ns, dtMs = dt)
  let src = g0.data.clone
  let st = bench("stack addLive (200 bins)", bytes + 2 * bytes):
    for r in 0..<traces:
      discard acc.addLive(r mod 200,
        toOpenArray(cast[ptr UncheckedArray[float32]](src.unsafe_raw_offset[r * ns].addr), 0, ns - 1))
  st.report
  echo "    ", st.ms

  var picks = newSeq[FirstBreakPick](traces)
  for m in [pmStaLta, pmEnergyRatio, pmCoppens]:
    for filtered in [false, true]:
      var params = FirstBreakParams(pickMethod: m, guideVelocity: 2500,
        guideT0Ms: 30)
      if filtered:
        params.lowCutHz = 5
        params.highCutHz = 60
      let plan = initFirstBreakPlan(float32, params, dt, ns)
      ws.reserve(plan.workspaceSize)
      let r = bench("pickInto " & $m & (if filtered: " + band-pass" else: ""), bytes):
        plan.pickInto(g0.data, meta, picks, ws)
      r.report
      echo "    ", r.ms
  var fit: ShotFit
  for model in [rmTwoSegment, rmOffsetBinned]:
    let params = ShotFitParams(model: model, correlate: false)
    let r = bench("fitShot " & $model & " (picks only)", 0):
      fitShot(meta, picks, params, fit)
      sink += fit.used.float64
    r.report
    echo "    ", r.ms
  let c = bench("fitShot two-segment + correlation", bytes):
    fitShot(g0.data, meta, picks, ShotFitParams(), fit)
    sink += fit.used.float64
  c.report
  echo "    ", c.ms
  if sink == 42: echo ""
  printHeader "brute stack, " & $(lineShots * lineChannels) & " traces × " &
    $lineNs & " samples f32 at 2 ms from a SEG-Y file, shot-sorted"
  stackBench()

main()
