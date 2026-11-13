## Distance-field patches (`survey/patch`, task 11.9): boxes at the
## receiver-line or a given azimuth, radius, maximum offset, centre
## distances, shift at the edges and the intersection with a template,
## against brute force over the SPS coordinates; and the Morgantown 11.7 box
## against its XPS (skipped when absent).

import std/[algorithm, assertions, math, os, random, sets, strutils]
when defined(release) or defined(danger): import std/times
import parseSEG
import processSEG/survey/patch
import utils/survey

proc close(a, b: float64, tol = 1e-9): bool = abs(a - b) <= tol

proc receiversOf(rs: openArray[SpsRelation]): seq[HashSet[StationKey]] =
  ## Per record (1-based, dense), the receiver stations of generated X records.
  var g = initSurveyGeometry()
  var f: SpsFile
  f.relations = @rs
  g.addSps(f)
  for r in g.records:
    while result.len < int(r): result.add initHashSet[StationKey]()
    for (_, rel) in g.relations(r): result[int(r) - 1].incl rel.receiver

type Brute = object
  az, hwI, hwC, radius, maxOffset, dI, dC: float64
  shift: bool

func frame(az, x, y: float64): (float64, float64) =
  let r = degToRad(az)
  (x * sin(r) + y * cos(r), x * cos(r) - y * sin(r))

proc brute(f: SpsFile, b: Brute): seq[HashSet[StationKey]] =
  ## Independent expectation: the receivers inside the limits, per S record.
  var aMin, cMin = Inf
  var aMax, cMax = -Inf
  for p in f.points:
    if p.kind == pkReceiver:
      let (a, c) = frame(b.az, p.easting, p.northing)
      (aMin, aMax, cMin, cMax) = (min(aMin, a), max(aMax, a), min(cMin, c), max(cMax, c))
  for s in f.points:
    if s.kind != pkSource: continue
    var (ca, cc) = frame(b.az, s.easting, s.northing)
    ca += b.dI
    cc += b.dC
    if b.shift:
      if b.hwI < Inf: ca = clamp(ca, aMin + b.hwI, aMax - b.hwI)
      if b.hwC < Inf: cc = clamp(cc, cMin + b.hwC, cMax - b.hwC)
    let r = degToRad(b.az)
    let (cx, cy) = (ca * sin(r) + cc * cos(r), ca * cos(r) - cc * sin(r))
    var set: HashSet[StationKey]
    for p in f.points:
      if p.kind != pkReceiver: continue
      let (a, c) = frame(b.az, p.easting, p.northing)
      if abs(a - ca) <= b.hwI + 1e-6 and abs(c - cc) <= b.hwC + 1e-6 and
          hypot(p.easting - cx, p.northing - cy) <= b.radius + 1e-6 and
          hypot(p.easting - s.easting, p.northing - s.northing) <= b.maxOffset + 1e-6:
        set.incl stationKey(p.line, p.point, 1)
    result.add set

proc check(f: SpsFile, t: PatchTemplate, b: Brute, label: string) =
  var g = initSurveyGeometry()
  g.addSps(f)
  let got = receiversOf(g.generateRelations(f.patchShots, t))
  let want = brute(f, b)
  for i in 0 ..< want.len:
    let have = if i < got.len: got[i] else: initHashSet[StationKey]()
    doAssert have == want[i], label & ": record " & $(i + 1) & ": " & $have.len & " vs " &
      $want[i].len
  doAssert want.len == 147

proc check(s: SyntheticSurvey, t: PatchTemplate, b: Brute, label: string) =
  check(s.spsFile, t, b, label)

block box_equals_template:
  # Sources lie between lines (±100, ±300 m) and on stations (±250 m): a
  # 260 × 310 box is the 4 × 11 template, clipped the same way.
  for az in [90.0, 30.0, 200.0]:
    var s = initSyntheticSurvey()
    (s.azimuthDeg, s.x0, s.y0) = (az, 5000.0, 9000.0)
    let f = s.spsFile
    var g = initSurveyGeometry()
    g.addSps(f)
    let byBox = g.generateRelations(f.patchShots, initDistancePatch(260, 310))
    doAssert receiversOf(byBox) == receiversOf(f.relations), $az
    doAssert byBox.len == f.relations.len # one run per line, channels in the same order

block limits:
  var s = initSyntheticSurvey()
  (s.azimuthDeg, s.x0, s.y0) = (30.0, 1000.0, 2000.0)
  check(s, initDistancePatch(260, 310), Brute(az: 30, hwI: 260, hwC: 310, radius: Inf,
    maxOffset: Inf), "box")
  var t = initDistancePatch(400, 250, radius = 380, maxOffset = 420)
  t.boxAzimuthDeg = 75
  check(s, t, Brute(az: 75, hwI: 400, hwC: 250, radius: 380, maxOffset: 420), "rotated box")
  t = initDistancePatch(Inf, Inf, radius = 330)
  check(s, t, Brute(az: 30, hwI: Inf, hwC: Inf, radius: 330, maxOffset: Inf), "radius")
  t = initDistancePatch(300, 200)
  (t.inlineDistance, t.crosslineDistance) = (125.0, -60.0)
  t.maxOffset = 500
  check(s, t, Brute(az: 30, hwI: 300, hwC: 200, radius: Inf, maxOffset: 500, dI: 125,
    dC: -60), "off-centre")
  t = initDistancePatch(260, 310, peShift)
  check(s, t, Brute(az: 30, hwI: 260, hwC: 310, radius: Inf, maxOffset: Inf, shift: true),
    "shift")

proc irregular(seed: int): SpsFile =
  ## The synthetic survey at 30° with every receiver moved by up to 20 m, and
  ## two lines whose stations do not run one way along the line: one doubles
  ## back every seventh station, one wanders 300 m across.
  var s = initSyntheticSurvey()
  (s.azimuthDeg, s.x0, s.y0) = (30.0, 1000.0, 2000.0)
  result = s.spsFile
  var rng = initRand(seed)
  let (ux, uy) = (sin(degToRad(30.0)), cos(degToRad(30.0)))
  for p in result.points.mitems:
    if p.kind == pkReceiver:
      p.easting += rng.rand(-20.0 .. 20.0)
      p.northing += rng.rand(-20.0 .. 20.0)
      let k = int(p.point) - 100
      if p.line == 1003 and k mod 7 == 3: # 180 m along the line: past three stations
        p.easting += 180 * ux
        p.northing += 180 * uy
      if p.line == 1006: # across the line, by 0 to 300 m
        p.easting += float64(k mod 5) * 75 * uy
        p.northing -= float64(k mod 5) * 75 * ux

block irregular_layout_against_brute_force:
  # The stations inside the limits are found from the box, not by testing
  # every station: on an irregular layout that search must lose none. (The
  # box azimuths are given: the measured one of a jittered layout is not
  # exactly 30°, and the brute force must use the same frame.)
  proc boxAt(az, hwI, hwC: float64, radius = Inf, maxOffset = Inf): PatchTemplate =
    result = initDistancePatch(hwI, hwC, radius = radius, maxOffset = maxOffset)
    result.boxAzimuthDeg = az
  for seed in [1, 2, 3]:
    let f = irregular(seed)
    check(f, boxAt(30, 260, 310), Brute(az: 30, hwI: 260, hwC: 310, radius: Inf,
      maxOffset: Inf), "box")
    check(f, boxAt(75, 400, 250, radius = 380, maxOffset = 420), Brute(az: 75, hwI: 400,
      hwC: 250, radius: 380, maxOffset: 420), "rotated box")
    # A strip across the lines: no limit along either axis of the layout.
    check(f, boxAt(120, Inf, 180), Brute(az: 120, hwI: Inf, hwC: 180, radius: Inf,
      maxOffset: Inf), "strip")
    check(f, boxAt(30, Inf, Inf, radius = 330), Brute(az: 30, hwI: Inf, hwC: Inf,
      radius: 330, maxOffset: Inf), "radius")
    var t = boxAt(30, 300, 200, maxOffset = 500)
    (t.inlineDistance, t.crosslineDistance) = (125.0, -60.0)
    check(f, t, Brute(az: 30, hwI: 300, hwC: 200, radius: Inf, maxOffset: 500, dI: 125,
      dC: -60), "off-centre")
    check(f, boxAt(30, Inf, Inf, maxOffset = 512.5), Brute(az: 30, hwI: Inf, hwC: Inf,
      radius: Inf, maxOffset: 512.5), "maximum offset")
    # The receiver-line azimuth as measured (not exactly 30°): a box along it.
    var g = initSurveyGeometry()
    g.addSps(f)
    let az = g.receiverLayout.azimuthDeg
    doAssert az != 30 and abs(az - 30) < 2
    check(f, initDistancePatch(260, 310), Brute(az: az, hwI: 260, hwC: 310, radius: Inf,
      maxOffset: Inf), "box along the measured azimuth")

block the_order_of_a_layout_does_not_matter:
  # Which stations lie inside the limits does not depend on the order of the
  # lines in the layout or of the stations on a line: a layout shuffled by
  # hand gives the X records of the measured one.
  let f = irregular(4)
  var g = initSurveyGeometry()
  g.addSps(f)
  let layout = g.receiverLayout
  var shuffled = layout
  var rng = initRand(9)
  for line in shuffled.lines.mitems: rng.shuffle(line.stations)
  rng.shuffle(shuffled.lines)
  var rotated = initDistancePatch(400, 250, radius = 380, maxOffset = 420)
  rotated.boxAzimuthDeg = 75
  for t in [initDistancePatch(260, 310), rotated, initDistancePatch(Inf, Inf, radius = 330),
      initDistancePatch(260, 310, peShift)]:
    let want = g.generateRelations(layout, f.patchShots, t)
    let got = g.generateRelations(shuffled, f.patchShots, t)
    doAssert want.len > 147 and receiversOf(got) == receiversOf(want)

block with_template:
  # 4 × 11 template ∩ 200 m maximum offset.
  var s = initSyntheticSurvey()
  s.azimuthDeg = 30
  let f = s.spsFile
  var g = initSurveyGeometry()
  g.addSps(f)
  var t = initPatchTemplate(4, 11)
  t.maxOffset = 200
  let got = receiversOf(g.generateRelations(f.patchShots, t))
  let tpl = receiversOf(f.relations)
  let near = brute(f, Brute(az: 30, hwI: Inf, hwC: Inf, radius: Inf, maxOffset: 200))
  for i in 0 ..< tpl.len: doAssert got[i] == tpl[i] * near[i], $(i + 1)
  # shift takes one or the other
  t.edge = peShift
  doAssertRaises(ValueError): discard g.generateRelations(f.patchShots, t)
  doAssertRaises(ValueError): discard g.generateRelations(f.patchShots, initDistancePatch(-1, 5))
  doAssertRaises(ValueError): discard g.generateRelations(f.patchShots, PatchTemplate())

block agreement:
  # compareRelations: drop a station from the listed X, add one to the generated.
  var s = initSyntheticSurvey()
  let f = s.spsFile
  var g = initSurveyGeometry()
  g.addSps(f)
  var gen = initSurveyGeometry()
  gen.table = g.table
  var gf: SpsFile
  gf.relations = g.generateRelations(f.patchShots, initPatchTemplate(4, 11))
  gf.relations[0].toReceiver -= 1 # record 1 loses its last station on the first line
  gf.relations[0].toChannel -= 1
  gen.addSps(gf)
  let a = compareRelations(g, gen)
  doAssert a.len == 147 and a[0].record == 1
  doAssert a[0].missing == @[stationKey(1001, 105, 1)] and a[0].extra.len == 0
  doAssert a[0].listed == 18 and a[0].generated == 17 and a[0].common == 17
  doAssert close(a[0].jaccard, 17 / 18) and close(a[0].listedFraction, 17 / 18)
  for e in a[1 .. ^1]: doAssert e.jaccard == 1 and e.missing.len == 0 and e.extra.len == 0

const
  dir = currentSourcePath().parentDir.parentDir / "exampleFiles" / "sps"
  base = dir / "PGI_NNE_West_Morgantown_3D_Post261019_NAD27_SPCS_WV_N_USFeet"

func fmt(s: Spread, decimals = 0): string =
  for v in [s.min, s.q1, s.median, s.q3, s.max]:
    if result.len > 0: result.add "/"
    result.add(if decimals == 0: $int(round(v)) else: formatFloat(v, ffDecimal, decimals))

block morgantown:
  if not (fileExists(base & ".sps") and fileExists(base & ".rps") and fileExists(base & ".xps")):
    echo "  skipped: Morgantown SPS files not present"
  else:
    var g = initSurveyGeometry()
    for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
    let p = describePatches(g)
    var shots: seq[PatchShot]
    for r in p.records: shots.add PatchShot(record: r.record, source: r.source)
    when defined(release) or defined(danger): (let t0 = cpuTime())
    let xs = g.generateRelations(p.layout, shots, initDistancePatch(p.boxInline, p.boxCrossline))
    when defined(release) or defined(danger):
      echo "  Morgantown distance patches: ", formatFloat(cpuTime() - t0, ffDecimal, 2), " s"
    var gen = initSurveyGeometry()
    gen.table = g.table
    var f: SpsFile
    f.relations = xs
    gen.addSps(f)
    let a = compareRelations(g, gen)
    var listed, jac, missing, extra: seq[float64]
    for e in a:
      listed.add e.listedFraction
      jac.add e.jaccard
      missing.add float64(e.missing.len)
      extra.add float64(e.extra.len)
    echo "  Morgantown box ±", int(round(p.boxInline)), " × ±",
      int(round(p.boxCrossline)), " ft vs XPS, min/q1/median/q3/max:"
    echo "    listed generated ", fmt(spreadOf(listed), 4), "; jaccard ", fmt(spreadOf(jac), 4)
    echo "    listed not generated ", fmt(spreadOf(missing)), "; generated not listed ",
      fmt(spreadOf(extra))
    doAssert a.len == 5519
    doAssert spreadOf(listed).median >= 0.99, "the 11.9 acceptance: ≥ 99 % at the median record"
