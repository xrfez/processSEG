## The exclusion area for noisy traces (qc/exclusion), on synthetic gathers
## built in the test (tests/utils/noisegather): a shot over three receiver
## lines with a loud zone near the source, planted noise, the same gather as
## a receiver gather, four-component nodes and an uncorrelated record.
##
## Each feature was checked to fail without it, by breaking the module and
## watching a test fail: noisy flags also set inside the area
## (trend_area_multiline); the weak finding limited to outside the area
## (planted_inside); node levels taken per trace (nodes_unrotated); neighbours
## taken across a line change (neighbours_stay_on_line); the early-break note
## never set, and the trend replaced by each trace's own pick (planted_outside);
## steps counted on every line instead of the lines the shot reaches
## (steps_fallback); the partial mark never set, and a node with one usable
## trace judged (nodes_unrotated). Review round: offset 0 taken as known
## (zero_offsets), equal offsets accepted, an edge on the smallest arrival
## excluding (zero_offsets), SEG-2 component codes ignored and unknown classes
## made one group (unknown_sensor_classes), the first trend window 7 instead
## of 2 (trend_area_multiline), weak not judged when the area covers a group
## (area_covers_group), stations without identity all equal (no_station_ids,
## unknown_sensor_classes), the short group's reason dropped
## (few_arrivals_group), the no-offsets reason taken from the XY-derived
## offsets (trend_area_multiline).

import std/[algorithm, assertions, math, sequtils]
import processSEG
import parseSEG
import utils/[allocs, noisegather]

type Run = object
  res: NoiseResult
  amps: seq[TraceAmplitude]
  roles: seq[TraceRole]

proc runOn(g: SynthGather[float32], picks: seq[FirstBreakPick], params = ExclusionParams(),
    domain = gdCommonSource, rotated: seq[bool] = @[]): Run =
  let meta = initChunkMeta(g.rows, g.dtMs)
  result.amps = newSeq[TraceAmplitude](g.len)
  var ws = initWorkspace()
  ws.reserve(traceAmplitudeWorkspace(g.cols) + recordAmplitudeWorkspace(g.len) +
    noiseWorkspace(g.len))
  traceAmplitudeInto(g.data, meta, AmplitudeParams(), result.amps, ws)
  result.roles = newSeq[TraceRole](g.len)
  meta.traceRolesInto(domain, rotated, result.roles)
  noiseQc(meta, result.amps, picks, result.roles, AmplitudeParams(), params, domain, result.res,
    ws)

proc run(ng: NoiseGather, params = ExclusionParams(), domain = gdCommonSource): Run =
  runOn(ng.g, ng.picks, params, domain)

func noisy(a: TraceAmplitude): bool = afNoisy in a.flags
func weak(a: TraceAmplitude): bool = afWeak in a.flags

func alongDist(k: int): float64 = hypot(float64(k) * 220, 330) ## station k from the source

proc median(x: seq[float64]): float64 =
  var s = x
  s.sort
  if s.len mod 2 == 1: s[s.len div 2] else: (s[s.len div 2 - 1] + s[s.len div 2]) / 2

proc insideCount(r: Run, ng: NoiseGather, line: int): int =
  for i in 0 ..< ng.spec.stations:
    if r.res.inside[ng.row(line, i)]: inc result

proc main*() =
  const src = 24 # the source's station on line 0

  block roles_basic:
    # Header lines and points: ids, stations; one sensor class and group.
    let ng = noiseGather()
    let meta = initChunkMeta(ng.g.rows, ng.g.dtMs)
    var roles = newSeq[TraceRole](ng.g.len)
    meta.traceRolesInto(gdCommonSource, [], roles)
    doAssert gatherDomainOf(meta) == gdCommonSource
    for r in 0 ..< ng.g.len:
      doAssert roles[r].sensor == scSingle and roles[r].group == 0 and roles[r].node == -1
      doAssert roles[r].line == int32(ng.line[r])
      doAssert roles[r].station == float64(101 + ng.station[r])
    # Header lines gone: the lines come from the receivers' coordinates; points gone too:
    # the station is the rank along the line.
    var b = noiseGather()
    b.stripLines()
    b.stripPoints()
    let metaB = initChunkMeta(b.g.rows, b.g.dtMs)
    metaB.traceRolesInto(gdCommonSource, [], roles)
    for r in 0 ..< b.g.len:
      doAssert roles[r].station == float64(b.station[r]), $r

  block trend_area_multiline:
    var ng = noiseGather()
    ng.stripOffsets() # no offsets: the model cannot apply, the trend does
    let on = ng.run()
    doAssert on.res.rule == arTrend, $on.res.rule
    doAssert on.res.linesFromHeaders and on.res.lineSource == lsHeaders and on.res.lines == 3
    let edge = on.res.edgeMs
    doAssert edge >= breakMs(alongDist(6)) and edge <= breakMs(alongDist(12)), $edge
    doAssert on.res.groups[0].edge == edge and on.res.offsetOrigin == ooUnknown
    # Why the model did not apply: no header offsets (not "too few picks").
    doAssert rnNoOffsets in on.res.reasons and rnFewPicks notin on.res.reasons
    # The trend follows the breaks near the source (a wide first window clipped the apex
    # by over 100 ms), the second line's near stations are inside, the clean gather has no
    # early-break note.
    for r in 0 ..< ng.g.len:
      doAssert not on.res.earlyBreak[r], $r
      if ng.dist[r] < 1500:
        doAssert abs(on.res.arrival[r] - breakMs(ng.dist[r])) <= 80, $r
    for k in -3 .. 3: doAssert on.res.inside[ng.row(1, src + k)], $k
    # Line 3 (far from the source) has nothing inside; the near stations of line 1 do.
    doAssert on.insideCount(ng, 2) == 0
    for k in -5 .. 5: doAssert on.res.inside[ng.row(0, src + k)]
    # The natural near traces are noisy without the area and not with it.
    let off = ng.run(ExclusionParams(choice: acOff))
    doAssert off.res.rule == arNone and off.res.excluded == 0
    var nearOff, nearOn, allOn, allOff = 0
    for k in -5 .. 5:
      if off.amps[ng.row(0, src + k)].noisy: inc nearOff
      if on.amps[ng.row(0, src + k)].noisy: inc nearOn
    for a in on.amps: (if a.noisy: inc allOn)
    for a in off.amps: (if a.noisy: inc allOff)
    # (The trend smooths the loud zone along each line: some traces of the next line, just
    # outside the edge, stay noisy.)
    doAssert nearOff == 11 and nearOn == 0 and allOn * 2 < allOff,
      $nearOff & " " & $nearOn & " " & $allOn & " " & $allOff

  block lines_from_coordinates:
    var ng = noiseGather()
    ng.stripLines()
    ng.stripPoints()
    let meta = initChunkMeta(ng.g.rows, ng.g.dtMs)
    var roles = newSeq[TraceRole](ng.g.len)
    meta.traceRolesInto(gdCommonSource, [], roles)
    # Never fewer lines than the real 3, and no id spans two real lines.
    var ids: set[uint8]
    var owner: array[256, int]
    for i in 0 ..< 256: owner[i] = -1
    for r in 0 ..< ng.g.len:
      let id = int(roles[r].line)
      doAssert id in 0 .. 255
      ids.incl uint8(id)
      doAssert owner[id] in [-1, ng.line[r]], "a line id spans two real lines"
      owner[id] = ng.line[r]
    doAssert ids.card >= 3
    # Counting steps never continues onto the next line: 3 each side of each line's
    # nearest station, the same 7 on each of the three lines.
    let st = ng.run(ExclusionParams(choice: acSteps, steps: 3, stepsOnAllLines: true))
    doAssert st.res.rule == arSteps and st.res.lineSource == lsCoordinates
    for j in 0 ..< 3: doAssert st.insideCount(ng, j) == 7, $j & ": " & $st.insideCount(ng, j)

  block model_area:
    let ng = noiseGather()
    let m = ng.run()
    doAssert m.res.rule == arModel and m.res.reasons == {}, $m.res.rule & $m.res.reasons
    doAssert m.res.agreement >= 0.9, $m.res.agreement
    for r in 0 ..< ng.g.len: doAssert not m.res.earlyBreak[r], $r
    doAssert m.res.edgeMs >= breakMs(alongDist(6)) and m.res.edgeMs <= breakMs(alongDist(12))
    # The model's arrival is finite for every row; no picks of the trace's own are used.
    for r in 0 ..< ng.g.len: doAssert not m.res.arrival[r].isNaN

  block radius_fallback:
    var ng = noiseGather()
    ng.stripPicks()
    let p = ExclusionParams(offsetOrigin: ooHeaders)
    let r1 = ng.run(p)
    doAssert r1.res.rule == arRadius and rnNoPicks in r1.res.reasons
    doAssert r1.res.radius >= alongDist(6) and r1.res.radius <= alongDist(12), $r1.res.radius
    doAssert r1.res.offsetOrigin == ooHeaders and r1.res.edgeMs.isNaN
    for k in -5 .. 5: doAssert r1.res.inside[ng.row(0, src + k)]
    doAssert r1.insideCount(ng, 2) == 0
    # Offsets from the header coordinates alone (the offset field itself gone).
    ng.stripOffsets()
    let r2 = ng.run(p)
    doAssert r2.res.rule == arRadius and abs(r2.res.radius - r1.res.radius) < 1e-9

  block steps_fallback:
    # No offsets, no coordinates, no picks: line and point numbers, with the source on
    # line 1's grid (its line number is line 1's, its point is station 24's).
    var ng = noiseGather()
    ng.stripPicks()
    ng.stripOffsets()
    ng.stripSourceXY()
    ng.stripReceiverXY()
    for k in ng.g.keys.mitems:
      k.srcLine = 1001
      k.srcPoint = float64(101 + src)
    let s = ng.run()
    doAssert s.res.rule == arSteps and s.res.steps == 8, $s.res.rule
    doAssert {rnNoPicks, rnNoOffsets} <= s.res.reasons
    doAssert s.insideCount(ng, 0) == 17 and s.insideCount(ng, 1) == 0 and
      s.insideCount(ng, 2) == 0
    doAssert s.res.linesReached == 1 and s.res.lines == 3
    for k in -8 .. 8: doAssert s.res.inside[ng.row(0, src + k)]
    # With offsets (no picks) every line has a nearest station; the lines the shot
    # reaches are those whose window is loud: line 3 is not, "every line" counts it.
    var ng2 = noiseGather()
    ng2.stripPicks()
    let reach = ng2.run(ExclusionParams(choice: acSteps))
    doAssert reach.insideCount(ng2, 0) == 17 and reach.insideCount(ng2, 2) == 0
    let every = ng2.run(ExclusionParams(choice: acSteps, stepsOnAllLines: true))
    for j in 0 ..< 3: doAssert every.insideCount(ng2, j) == 17, $j
    doAssert every.res.linesReached == 3

  block steps_receiver_xy_only:
    # Receiver coordinates and picks, no source coordinates and no offsets: the source is
    # fitted to the picks; it lands within 3 stations of the truth.
    var ng = noiseGather()
    ng.stripOffsets()
    ng.stripSourceXY()
    let s = ng.run(ExclusionParams(choice: acSteps, steps: 4))
    doAssert s.res.rule == arSteps
    var lo = 1000
    var hi = -1
    for i in 0 ..< ng.spec.stations:
      if s.res.inside[ng.row(0, i)]:
        lo = min(lo, i)
        hi = max(hi, i)
    doAssert hi >= 0 and abs(float64(lo + hi) / 2 - float64(src)) <= 3, $lo & " " & $hi
    doAssert hi - lo == 8, $lo & " " & $hi

  block overrides:
    var ng = noiseGather()
    let edge = breakMs(alongDist(4)) + 1 # between the 4th and 5th stations
    let ft = ng.run(ExclusionParams(choice: acFixedTime, edgeMs: edge))
    doAssert ft.res.rule == arFixedTime and ft.res.edgeMs == edge
    var count = 0
    for r in 0 ..< ng.g.len:
      doAssert ft.res.inside[r] == (ft.res.arrival[r] < edge)
      if ft.res.inside[r]: inc count
    doAssert count >= 9 and count <= 12, $count
    for k in -3 .. 3: doAssert ft.res.inside[ng.row(0, src + k)]
    doAssert not ft.res.inside[ng.row(0, src + 8)]
    # A fixed radius: exactly the traces nearer than it.
    let fr = ng.run(ExclusionParams(choice: acFixedRadius, radius: 1000))
    doAssert fr.res.rule == arFixedRadius and fr.res.radius == 1000
    var near = 0
    for r in 0 ..< ng.g.len:
      doAssert fr.res.inside[r] == (ng.dist[r] < 1000)
      if ng.dist[r] < 1000: inc near
    doAssert near > 0 and fr.res.excluded == near
    # Three stations each side.
    let st = ng.run(ExclusionParams(choice: acSteps, steps: 3))
    doAssert st.res.rule == arSteps and st.res.steps == 3 and st.insideCount(ng, 0) == 7
    # Needs picks / offsets.
    ng.stripPicks()
    let np = ng.run(ExclusionParams(choice: acFixedTime, edgeMs: 100))
    doAssert np.res.rule == arNone and np.res.reasons == {rnOverrideNeedsPicks}
    doAssert np.res.excluded == 0
    ng.stripOffsets()
    ng.stripSourceXY()
    let nr = ng.run(ExclusionParams(choice: acFixedRadius, radius: 1000))
    doAssert nr.res.rule == arNone and nr.res.reasons == {rnOverrideNeedsOffsets}

  block planted_outside:
    var ng = noiseGather()
    let hum = ng.row(0, 3)
    let bb = ng.row(1, 45)
    let burst = ng.row(0, 44)
    ng.plantHum(hum)
    ng.plantBroadband(bb)
    ng.plantBurst(burst)
    for params in [ExclusionParams(), ExclusionParams(choice: acAuto, steps: 8)]:
      var cmp = ng
      if params.steps == 8: cmp.stripOffsets() # the trend instead of the model
      let r = cmp.run(params)
      for t in [hum, bb, burst]:
        doAssert r.amps[t].noisy and r.res.state[t] == nsJudged and not r.res.inside[t], $t
        doAssert r.res.louder[t] and r.res.neighbourState[t] == nbJudged, $t
      doAssert r.res.earlyBreak[burst], $r.res.rule
      doAssert not r.res.earlyBreak[hum] and not r.res.earlyBreak[bb]

  block planted_inside:
    var ng = noiseGather()
    let nearHum = ng.row(0, src - 2)
    let nearWeak = ng.row(0, src)
    let nearSpiky = ng.row(0, src + 1)
    let nearClipped = ng.row(0, src + 2)
    ng.plantHum(nearHum)
    ng.scaleTrace(nearWeak, -25)
    ng.g.data[nearSpiky, 300] = 60
    block:
      var mx = 0'f32
      for j in 0 ..< ng.g.cols: mx = max(mx, abs(ng.g.data[nearClipped, j]))
      for j in 100 .. 104: ng.g.data[nearClipped, j] = mx
      for j in 200 .. 204: ng.g.data[nearClipped, j] = mx
    let r = ng.run()
    doAssert r.res.inside[nearHum] and r.res.state[nearHum] == nsInArea
    doAssert not r.amps[nearHum].noisy
    doAssert r.amps[nearWeak].weak and r.res.inside[nearWeak] # weak is judged everywhere
    doAssert afSpiky in r.amps[nearSpiky].flags and not r.amps[nearSpiky].noisy
    doAssert afClipped in r.amps[nearClipped].flags
    doAssert r.res.groups[0].weak >= 1 and r.res.groups[0].inArea == r.res.excluded

  block neighbours_stay_on_line:
    var ng = noiseGather()
    let last = ng.row(0, 47) # the end of line 1; line 2 starts at row 48
    ng.scaleTrace(last, 15)
    # Three traces of line 3 form a line piece of their own.
    for i in 20 .. 22: ng.g.keys[ng.row(2, i)].rcvLine = 9999
    let r = ng.run()
    doAssert r.res.lines == 4
    var near: seq[float64]
    for i in 40 .. 46: near.add r.amps[ng.row(0, i)].rmsDb
    doAssert r.res.neighbourState[last] == nbJudged
    doAssert abs(r.res.neighbourDb[last] - (r.amps[last].rmsDb - median(near))) < 1e-9
    doAssert r.res.louder[last]
    for i in 20 .. 22:
      doAssert r.res.neighbourState[ng.row(2, i)] == nbFewNeighbours
      doAssert not r.res.louder[ng.row(2, i)] and r.res.neighbourDb[ng.row(2, i)].isNaN
    # A trace inside the area is not compared at all.
    doAssert r.res.neighbourState[ng.row(0, src)] == nbInArea

  block burst_run_limit:
    # 5 adjacent burst traces beyond the visible breaks (the end of line 3): with the trend
    # they get early picks that agree with each other, fall inside the area and are not
    # judged; with the model, which a few false picks do not move, they are judged.
    var ng = noiseGather()
    for i in 0 .. 4: ng.plantBurst(ng.row(2, i))
    var trendOnly = ng
    trendOnly.stripOffsets()
    let t = trendOnly.run()
    doAssert t.res.rule == arTrend
    var insideT, noisyT = 0
    for i in 0 .. 4:
      if t.res.inside[ng.row(2, i)]: inc insideT
      if t.amps[ng.row(2, i)].noisy: inc noisyT
    doAssert insideT >= 3 and noisyT <= 2, $insideT & " " & $noisyT
    let m = ng.run()
    doAssert m.res.rule == arModel
    var judged = 0
    for i in 0 .. 4:
      if m.res.state[ng.row(2, i)] == nsJudged and m.amps[ng.row(2, i)].noisy: inc judged
    doAssert judged == 5, $judged
    # The area itself held: the near stations are still inside and line 3's far side is not.
    doAssert t.res.edgeMs >= breakMs(alongDist(6))

  block receiver_gather:
    var rg = noiseGather().receiverGather()
    let far = rg.row(0, 45)
    rg.scaleTrace(far, 20) # a noisy shot far from the receiver
    let meta = initChunkMeta(rg.g.rows, rg.g.dtMs)
    doAssert gatherDomainOf(meta) == gdCommonReceiver
    let r = rg.run(domain = gdCommonReceiver)
    doAssert r.res.domain == gdCommonReceiver and r.res.rule in {arModel, arTrend}
    doAssert r.res.lines == 3 and r.res.lineSource == lsHeaders # the source lines
    for k in -4 .. 4: doAssert r.res.inside[rg.row(0, src + k)], "the near shots are excluded"
    doAssert r.insideCount(rg, 2) == 0
    doAssert r.amps[far].noisy and r.res.louder[far] and r.res.state[far] == nsJudged
    # The gather's level is the median of the judged units.
    var judged: seq[float64]
    for i in 0 ..< rg.g.len:
      if not r.res.inside[i]: judged.add r.amps[i].rmsDb
    doAssert abs(r.res.groups[0].medianDb - median(judged)) < 1e-9
    doAssert r.res.groups[0].judged == judged.len

  block nodes_unrotated:
    var nd = nodeGather()
    let n = nd.g.len
    let meta = initChunkMeta(nd.g.rows, nd.g.dtMs)
    doAssert gatherDomainOf(meta) == gdCommonSource
    var roles = newSeq[TraceRole](n)
    meta.traceRolesInto(gdCommonSource, nd.rotated, roles)
    for i in 0 ..< nd.nodes:
      doAssert roles[nodeRow(i, 0)].sensor == scPressure and roles[nodeRow(i, 0)].node == -1
      for c in 1 .. 3:
        let q = roles[nodeRow(i, c)]
        doAssert q.sensor == scMotion and not q.rotated and q.node == roles[nodeRow(i, 1)].node
      doAssert roles[nodeRow(i, 0)].group != roles[nodeRow(i, 1)].group
    # Today's single group flags components of clean nodes; per node it flags nothing.
    var ws = initWorkspace()
    ws.reserve(traceAmplitudeWorkspace(nd.g.cols) + recordAmplitudeWorkspace(n) +
      noiseWorkspace(n))
    var amps = newSeq[TraceAmplitude](n)
    traceAmplitudeInto(nd.g.data, meta, AmplitudeParams(), amps, ws)
    let today = recordAmplitude(amps, AmplitudeParams(), ws)
    doAssert today.flagged > 0, "the fixture would not show the fault"
    let off = ExclusionParams(choice: acOff)
    let clean = runOn(nd.g, @[], off, rotated = nd.rotated)
    doAssert clean.res.groups.len == 2
    doAssert clean.res.groups[clean.roles[nodeRow(0, 0)].group].sensor == scPressure
    for a in clean.amps: doAssert not a.noisy and not a.weak
    # One geophone at +20 dB: its node is noisy, on all three of its geophone traces only.
    var hot = nd
    let node = 5
    for j in 0 ..< hot.g.cols:
      hot.g.data[nodeRow(node, 2), j] = hot.g.data[nodeRow(node, 2), j] * 10
    let h = runOn(hot.g, @[], off, rotated = hot.rotated)
    for c in 1 .. 3: doAssert h.amps[nodeRow(node, c)].noisy, $c
    doAssert not h.amps[nodeRow(node, 0)].noisy
    var count = 0
    for a in h.amps: (if a.noisy: inc count)
    doAssert count == 3, $count
    doAssert h.res.groups[h.roles[nodeRow(node, 1)].group].noisy == 1 # one unit
    # A node missing a component is partial; with one left it is not judged.
    var gap = nd
    for j in 0 ..< gap.g.cols:
      gap.g.data[nodeRow(3, 3), j] = 0
      gap.g.data[nodeRow(4, 2), j] = 0
      gap.g.data[nodeRow(4, 3), j] = 0
    let p = runOn(gap.g, @[], off, rotated = gap.rotated)
    for c in 1 .. 3: doAssert p.res.partial[nodeRow(3, c)]
    doAssert p.res.state[nodeRow(3, 1)] == nsJudged and p.res.state[nodeRow(3, 3)] == nsUnusable
    doAssert not p.res.partial[nodeRow(2, 1)]
    doAssert p.res.state[nodeRow(4, 1)] == nsIncompleteNode
    doAssert p.res.state[nodeRow(4, 2)] == nsUnusable
    doAssert afDead in p.amps[nodeRow(4, 2)].flags

  block nodes_rotated:
    var nd = nodeGather(rotatedToVTR = true)
    let off = ExclusionParams(choice: acOff)
    let c = runOn(nd.g, @[], off, rotated = nd.rotated)
    doAssert c.res.groups.len == 4 # the hydrophone and one per component
    var comps: set[uint8]
    for g in c.res.groups:
      if g.sensor == scMotion:
        comps.incl g.component
        doAssert g.units == nd.nodes
    doAssert comps == {15'u8, 16, 17}
    for q in c.roles:
      if q.sensor == scMotion: doAssert q.rotated and q.node >= 0
    for a in c.amps: doAssert not a.noisy and not a.weak
    # A hot component trace is noisy on its own, not its node.
    let hotRow = nodeRow(7, 1)
    for j in 0 ..< nd.g.cols: nd.g.data[hotRow, j] = nd.g.data[hotRow, j] * 10
    let h = runOn(nd.g, @[], off, rotated = nd.rotated)
    var count = 0
    for r, a in h.amps:
      if a.noisy:
        inc count
        doAssert r == hotRow
    doAssert count == 1
    # Rotation told by the caller alone (codes 12-14) gives the same groups.
    var odd = nodeGather(rotatedToVTR = false)
    let rotatedFlags = newSeqWith(odd.g.len, true)
    let o = runOn(odd.g, @[], off, rotated = rotatedFlags)
    doAssert o.res.groups.len == 4

  block uncorrelated_no_picks:
    var g = uncorrelatedGather()
    for j in 0 ..< g.cols:
      g.data[5, j] = g.data[5, j] * 8 # loud
      g.data[9, j] = g.data[9, j] * 0.05 # weak
    let r = runOn(g, @[])
    doAssert r.res.rule == arNone and r.res.reasons == {rnNoPicks, rnNoOffsets, rnNoPosition},
      $r.res.reasons
    doAssert r.res.lineSource == lsUnknown and r.res.lines == 1 and r.res.excluded == 0
    # The flags are those of recordAmplitude.
    let meta = initChunkMeta(g.rows, g.dtMs)
    var amps = newSeq[TraceAmplitude](g.len)
    var ws = initWorkspace()
    ws.reserve(traceAmplitudeWorkspace(g.cols) + recordAmplitudeWorkspace(g.len))
    traceAmplitudeInto(g.data, meta, AmplitudeParams(), amps, ws)
    discard recordAmplitude(amps, AmplitudeParams(), ws)
    var flagged = 0
    for i in 0 ..< g.len:
      doAssert amps[i].flags == r.amps[i].flags and amps[i].robustZ == r.amps[i].robustZ, $i
      if amps[i].flags != {}: inc flagged
    doAssert flagged >= 2

  block old_recordAmplitude_unchanged:
    # acOff is recordAmplitude: the same flags and z-scores; and recordAmplitude itself
    # still calls the near-source traces noisy.
    let ng = noiseGather()
    let off = ng.run(ExclusionParams(choice: acOff))
    let meta = initChunkMeta(ng.g.rows, ng.g.dtMs)
    var amps = newSeq[TraceAmplitude](ng.g.len)
    var ws = initWorkspace()
    ws.reserve(traceAmplitudeWorkspace(ng.g.cols) + recordAmplitudeWorkspace(ng.g.len))
    traceAmplitudeInto(ng.g.data, meta, AmplitudeParams(), amps, ws)
    let rec = recordAmplitude(amps, AmplitudeParams(), ws)
    for i in 0 ..< ng.g.len:
      doAssert amps[i].flags == off.amps[i].flags and amps[i].robustZ == off.amps[i].robustZ
    doAssert rec.counts[afNoisy] > 15
    doAssert abs(rec.medianRmsDb - off.res.groups[0].medianDb) < 1e-12

  block shared_fit:
    let ng = noiseGather()
    let a = ng.run()
    # The same result when the caller passes the fit it computed itself.
    let g = ng.g
    let meta = initChunkMeta(g.rows, g.dtMs)
    var fit: ShotFit
    fitShot(meta, ng.picks, ShotFitParams(), fit)
    var amps = newSeq[TraceAmplitude](g.len)
    var ws = initWorkspace()
    ws.reserve(traceAmplitudeWorkspace(g.cols) + noiseWorkspace(g.len))
    traceAmplitudeInto(g.data, meta, AmplitudeParams(), amps, ws)
    var roles = newSeq[TraceRole](g.len)
    meta.traceRolesInto(gdCommonSource, [], roles)
    var res: NoiseResult
    noiseQc(meta, amps, ng.picks, fit, roles, AmplitudeParams(), ExclusionParams(),
      gdCommonSource, res, ws)
    doAssert res.rule == a.res.rule and res.inside == a.res.inside and res.state == a.res.state
    doAssert res.louder == a.res.louder and res.arrival == a.res.arrival
    for i in 0 ..< g.len: doAssert amps[i].flags == a.amps[i].flags
    # A fit that is not this gather's is refused.
    var wrong: ShotFit
    doAssertRaises(ValueError):
      noiseQc(meta, amps, ng.picks, wrong, roles, AmplitudeParams(), ExclusionParams(),
        gdCommonSource, res, ws)

  block degenerate_gathers:
    # No rows, one row, every row dead, no row live: nothing raises, nothing is judged.
    var keys: seq[TraceKeys]
    var counts: seq[int]
    let empty = initChunkMeta(rowsOf(keys, counts), 2.0)
    var amps: seq[TraceAmplitude]
    var roles: seq[TraceRole]
    var res: NoiseResult
    var ws = initWorkspace()
    ws.reserve(noiseWorkspace(0))
    noiseQc(empty, amps, [], roles, AmplitudeParams(), ExclusionParams(), gdCommonSource, res,
      ws)
    doAssert res.rule == arNone and res.judged == 0 and res.groups.len == 0
    # one live row
    let one = uncorrelatedGather(1)
    let o = runOn(one, @[])
    doAssert o.res.state[0] == nsJudged and o.res.neighbourState[0] == nbFewNeighbours
    # every row dead
    var dead = uncorrelatedGather(30)
    for r in 0 ..< 30:
      for j in 0 ..< dead.cols: dead.data[r, j] = 0
    let d = runOn(dead, @[])
    for r in 0 ..< 30: doAssert d.res.state[r] == nsUnusable and afDead in d.amps[r].flags
    doAssert d.res.judged == 0 and d.res.groups[0].medianDb.isNaN
    # no row live (auxiliary)
    var aux = uncorrelatedGather(30)
    for k in aux.keys.mitems: k.status = tsAuxiliary
    let a = runOn(aux, @[])
    for r in 0 ..< 30: doAssert a.res.state[r] == nsUnusable and a.roles[r].group == -1
    doAssert a.res.groups.len == 0 and a.res.lines == 0
    # a gather whose picks are all NaN or unconfident still gets an area from its offsets
    var ng = noiseGather()
    for q in ng.picks.mitems: q = FirstBreakPick(timeMs: NaN, col: -1, snrDb: NaN, confidence: 0)
    let np = ng.run()
    doAssert np.res.rule == arRadius and {rnFewPicks} <= np.res.reasons, $np.res.reasons

  block zero_offsets:
    # SEG-Y's unset offset is 0: all zero, no coordinates, no picks tells nothing, and
    # nothing is excluded (once, every trace was).
    var ng = noiseGather()
    ng.stripPicks()
    ng.stripSourceXY()
    ng.stripReceiverXY()
    for k in ng.g.keys.mitems: k.offset = 0
    let z = ng.run()
    doAssert z.res.rule == arNone and z.res.excluded == 0 and rnNoOffsets in z.res.reasons
    doAssert z.res.judged == ng.g.len
    let st = ng.run(ExclusionParams(choice: acSteps))
    doAssert st.res.rule == arNone and rnNoPosition in st.res.reasons and st.res.excluded == 0
    # An edge equal to the smallest arrival excludes nothing: equal offsets via coordinates.
    var one = noiseGather(NoiseSpec(lines: 1))
    one.stripPicks()
    for k in one.g.keys.mitems:
      k.offset = NaN
      k.rcvX = 100
      k.rcvY = 0
      k.srcX = 0
      k.srcY = 0
    let eq = one.run()
    doAssert eq.res.excluded == 0 and eq.res.rule != arRadius

    # An edge that lands on the smallest arrival (8 tied traces, 3 of them loud, nothing earlier)
    # excludes nothing.
    var tie = noiseGather(NoiseSpec(lines: 1, stations: 30, srcStation: -200))
    tie.stripPicks()
    tie.stripSourceXY()
    tie.stripReceiverXY()
    for i in 0 ..< 30:
      tie.g.keys[i].offset = (if i < 8: 100.0 else: 100.0 + 50.0 * float64(i - 7))
      if i < 3: tie.scaleTrace(i, 20)
    let tr = tie.run()
    doAssert tr.res.rule == arRadius and tr.res.excluded == 0, $tr.res.rule & $tr.res.excluded

  block unknown_sensor_classes:
    # Components 1-4 (as a survey join writes them) with trace code 1: never one flat
    # group; judged per station on the summed energy of its traces.
    var nd = nodeGather()
    for i in 0 ..< nd.nodes:
      for c in 0 .. 3:
        nd.g.keys[nodeRow(i, c)].traceCode = 1
        nd.g.keys[nodeRow(i, c)].component = uint8(c + 1)
    let off = ExclusionParams(choice: acOff)
    let u = runOn(nd.g, @[], off)
    doAssert u.res.groups.len == 1 and u.res.groups[0].units == nd.nodes
    doAssert rnUnknownSensors in u.res.reasons
    for q in u.roles: doAssert q.guessed and q.node >= 0
    for a in u.amps: doAssert not a.noisy and not a.weak
    # A hot trace makes its station noisy, on all four of its traces.
    var hot = nd
    for j in 0 ..< hot.g.cols:
      hot.g.data[nodeRow(6, 1), j] = hot.g.data[nodeRow(6, 1), j] * 100
    let h = runOn(hot.g, @[], off)
    for c in 0 .. 3: doAssert h.amps[nodeRow(6, c)].noisy, $c
    # SEG-2 style codes in `component` with no trace code are read: pressure and motion.
    var s2 = nodeGather()
    for i in 0 ..< s2.nodes:
      for c in 0 .. 3:
        s2.g.keys[nodeRow(i, c)].traceCode = 0
        s2.g.keys[nodeRow(i, c)].component = uint8(11 + c)
    let k2 = runOn(s2.g, @[], off)
    doAssert k2.res.groups.len == 2 and rnUnknownSensors notin k2.res.reasons
    doAssert k2.roles[nodeRow(0, 0)].sensor == scPressure
    # No point numbers and no coordinates: nothing to sum over, not judged (never flat).
    var bare = nd
    for k in bare.g.keys.mitems:
      k.rcvPoint = NaN
      k.rcvX = NaN
      k.rcvY = NaN
    let b = runOn(bare.g, @[], off)
    doAssert b.res.groups.len == 0 and b.res.judged == 0
    for r in 0 ..< bare.g.len: doAssert b.roles[r].group == -1 and b.res.state[r] == nsUnusable
    doAssert {rnUnknownSensors, rnNoStationIds} <= b.res.reasons

  block area_covers_group:
    # An area over the whole group leaves no statistics outside: weak is still judged,
    # against all of the group, and the result says so.
    var ng = noiseGather()
    let weakRow = ng.row(0, 3)
    ng.scaleTrace(weakRow, -25)
    let r = ng.run(ExclusionParams(choice: acFixedTime, edgeMs: 1e6))
    doAssert r.res.excluded == ng.g.len and r.res.judged == 0
    doAssert rnAllInside in r.res.reasons and r.amps[weakRow].weak
    for a in r.amps: doAssert not a.noisy
    doAssert not r.res.groups[0].medianDb.isNaN

  block no_station_ids:
    # Header lines but no point numbers and no coordinates: each row is a station of its
    # own; counting steps takes 3 either side of each line's nearest station, not whole lines.
    var ng = noiseGather()
    ng.stripPicks()
    ng.stripPoints()
    ng.stripReceiverXY()
    let s = ng.run(ExclusionParams(choice: acSteps, steps: 3, stepsOnAllLines: true))
    doAssert s.res.rule == arSteps
    for j in 0 ..< 3: doAssert s.insideCount(ng, j) == 7, $j & ": " & $s.insideCount(ng, j)
    # Unrotated geophones with no station identity are not judged per node, and say so.
    var nd = nodeGather()
    for k in nd.g.keys.mitems:
      k.rcvPoint = NaN
      k.rcvX = NaN
      k.rcvY = NaN
    let n = runOn(nd.g, @[], ExclusionParams(choice: acOff), rotated = nd.rotated)
    doAssert rnNoStationIds in n.res.reasons
    for q in n.roles: doAssert q.node == -1

  block few_arrivals_group:
    # The geophone nodes are 19 usable units (5 are dead), too few for an edge; the
    # hydrophones (24) give one. The result says so.
    var nd = nodeGather()
    for i in 0 ..< 5:
      for c in 1 .. 3:
        for j in 0 ..< nd.g.cols: nd.g.data[nodeRow(i, c), j] = 0
    let f = runOn(nd.g, @[], ExclusionParams(), rotated = nd.rotated)
    doAssert f.res.rule == arRadius and rnFewArrivals in f.res.reasons, $f.res.rule
    # Without a short group the reason is absent.
    let c = runOn(nodeGather().g, @[], ExclusionParams(), rotated = nodeGather().rotated)
    doAssert rnFewArrivals notin c.res.reasons

  block params_errors:
    for p in [ExclusionParams(levelDb: NaN), ExclusionParams(levelDb: -1),
        ExclusionParams(minConfidence: 1.5), ExclusionParams(minFraction: -0.1),
        ExclusionParams(trendTraces: 0), ExclusionParams(minArrivals: 0),
        ExclusionParams(neighbours: 0), ExclusionParams(minNeighbours: 0),
        ExclusionParams(steps: -1), ExclusionParams(neighbourZ: NaN),
        ExclusionParams(choice: acFixedTime), ExclusionParams(choice: acFixedRadius),
        ExclusionParams(choice: acFixedRadius, radius: -3)]:
      doAssertRaises(ValueError): validate(p)
      let ng = noiseGather(NoiseSpec(lines: 1))
      doAssertRaises(ValueError): discard ng.run(p)
    validate(ExclusionParams())
    validate(ExclusionParams(neighbourZ: Inf))

  block lengths_errors:
    let ng = noiseGather(NoiseSpec(lines: 1))
    let g = ng.g
    let meta = initChunkMeta(g.rows, g.dtMs)
    var amps = newSeq[TraceAmplitude](g.len)
    var roles = newSeq[TraceRole](g.len)
    var res: NoiseResult
    var ws = initWorkspace()
    ws.reserve(noiseWorkspace(g.len))
    let amp = AmplitudeParams()
    let p = ExclusionParams()
    doAssertRaises(ValueError):
      var short = newSeq[TraceAmplitude](g.len - 1)
      noiseQc(meta, short, ng.picks, roles, amp, p, gdCommonSource, res, ws)
    doAssertRaises(ValueError):
      var shortRoles = newSeq[TraceRole](g.len + 1)
      noiseQc(meta, amps, ng.picks, shortRoles, amp, p, gdCommonSource, res, ws)
    doAssertRaises(ValueError):
      noiseQc(meta, amps, ng.picks[0 .. 3], roles, amp, p, gdCommonSource, res, ws)
    doAssertRaises(ValueError):
      var small = initWorkspace()
      noiseQc(meta, amps, ng.picks, roles, amp, p, gdCommonSource, res, small)
    doAssertRaises(ValueError):
      var bad = roles
      bad[0].group = int32(g.len)
      noiseQc(meta, amps, ng.picks, bad, amp, p, gdCommonSource, res, ws)
    doAssertRaises(ValueError):
      meta.traceRolesInto(gdCommonSource, [true], roles)
    doAssertRaises(ValueError):
      var shortRoles = newSeq[TraceRole](3)
      meta.traceRolesInto(gdCommonSource, [], shortRoles)

  block allocates_nothing_after_first:
    let ng = noiseGather()
    let g = ng.g
    let meta = initChunkMeta(g.rows, g.dtMs)
    var fit: ShotFit
    fitShot(meta, ng.picks, ShotFitParams(), fit)
    var amps = newSeq[TraceAmplitude](g.len)
    var ws = initWorkspace()
    ws.reserve(traceAmplitudeWorkspace(g.cols) + noiseWorkspace(g.len))
    traceAmplitudeInto(g.data, meta, AmplitudeParams(), amps, ws)
    var roles = newSeq[TraceRole](g.len)
    meta.traceRolesInto(gdCommonSource, [], roles)
    var res: NoiseResult
    noiseQc(meta, amps, ng.picks, fit, roles, AmplitudeParams(), ExclusionParams(),
      gdCommonSource, res, ws)
    doAssert res.rule == arModel
    doAssert allocsIn(noiseQc(meta, amps, ng.picks, fit, roles, AmplitudeParams(),
      ExclusionParams(), gdCommonSource, res, ws)) == 0
    # The steps fallback that fits the source XY from the picks allocates nothing either.
    var xy = noiseGather()
    xy.stripOffsets()
    xy.stripSourceXY()
    let gx = xy.g
    let metaX = initChunkMeta(gx.rows, gx.dtMs)
    var fitX: ShotFit
    fitShot(metaX, xy.picks, ShotFitParams(), fitX)
    var ampsX = newSeq[TraceAmplitude](gx.len)
    traceAmplitudeInto(gx.data, metaX, AmplitudeParams(), ampsX, ws)
    var rolesX = newSeq[TraceRole](gx.len)
    metaX.traceRolesInto(gdCommonSource, [], rolesX)
    let stepsP = ExclusionParams(choice: acSteps, steps: 4)
    noiseQc(metaX, ampsX, xy.picks, fitX, rolesX, AmplitudeParams(), stepsP, gdCommonSource,
      res, ws)
    doAssert res.rule == arSteps
    doAssert allocsIn(noiseQc(metaX, ampsX, xy.picks, fitX, rolesX, AmplitudeParams(), stepsP,
      gdCommonSource, res, ws)) == 0
    doAssert allocsIn(noiseQc(meta, amps, [], roles, AmplitudeParams(),
      ExclusionParams(choice: acOff), gdCommonSource, res, ws)) == 0
    doAssert allocsIn(noiseQc(meta, amps, [], roles, AmplitudeParams(), ExclusionParams(),
      gdCommonSource, res, ws)) == 0

when isMainModule: main()
