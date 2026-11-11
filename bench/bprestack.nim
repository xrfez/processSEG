## Pre-stack throughput (Phases 6–8, task 9.2), one thread, a panel of 2000
## traces × 2001 samples of float32 at 2 ms (16 MB): statics, LMO, NMO
## (linear and cubic), stacking into bins, first-break picking (three methods,
## with and without the band-pass) and the per-shot refraction fit.

import std/strutils
import benchutil
import processSEG
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
        plan.pickInto(g0.data, meta, ws, picks)
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

main()
