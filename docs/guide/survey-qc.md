# Survey QC: patches, fold maps and record completeness

This page checks a 3D survey's geometry and its field records: what the pre-plot says
should have been recorded, and what the data hold. The steps:

1. read the SPS files with parseSEG into a `SurveyGeometry`, and join traces to it by
   station;
2. describe the receiver patches of the X records, or generate X records from a template
   or a distance box;
3. bin the survey (`surveyGrid`) and make the pre-plot fold map, then the actual fold of
   the traces and their difference;
4. check each field record against the XPS (completeness) and its traces' amplitudes.

processSEG takes data in and returns values (decision D1, D14): parseSEG reads and writes
the SPS, SEG-Y and SEG-D files. The examples build a small orthogonal survey in code, so
they need no files.

## Reading SPS and joining traces

parseSEG reads SPS 1 and 2.1 (`readSps`, `parseSps`) and merges files into a
`SurveyGeometry` (`addSps`): R and S records become stations with coordinates, X records
become relations (record, channel → source and receiver station). O records, or X records
of an `.ops` file, are receiver-domain relations (`rdReceiver`); parseSEG's
`receiverRelations(g)` builds them from R, S and X.

`applySurvey` fills a trace's coordinates, offset and midpoint from the survey. **The
header stations are the truth:** a trace that carries both stations takes their
coordinates directly; a trace with only a source station is found among the X records of
that source, then by channel; the field record number (FFID) is a fallback only. This
matters on real data. In the Morgantown survey (the reference data of these tests), the
SEG-D file `00000151_12900_11300.sgd` has FFID 151 and source 129/113, but the XPS lists
that source as record 77, and its record 151 is another shot (511/117). Joining by FFID
would give every trace the wrong geometry; joining by station gives the right one.
Recorders also number traces their own way: only 23 of its 1127 XPS channels carry the
station their header has, so traces match by receiver station, not by channel.

```nim skip
var g = initSurveyGeometry()
for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
var f = openSeismic("00000151_12900_11300.sgd", keys = true)
var stats: JoinStats
f.applySurvey(g, stats)        # by station; `overwrite = true` refreshes coordinates only
echo stats                     # traces joined by stations, via the source, by record, ...
```

## Patches: describing and generating X records

`describePatches(g)` walks every X record: its receiver lines and stations, its channels,
and the receivers' extent around the source, inline (along the receiver lines) and
crossline (across them). Over the survey it reports the full patch (median half-widths),
the records clipped at the edges, and, when the unclipped records agree, a regular
`lines × channels` template. Morgantown is not a template survey: its patch is a distance
box of about ±19 340 × ±10 060 ft, clipped at the survey edges.

`generateRelations` goes the other way: X records from R and S points and a
`PatchTemplate`, either `lines × channels` around the source (`initPatchTemplate`), or
every receiver inside a distance box, radius or maximum offset (`initDistancePatch`), or
both. The edge modes are `peRoll` (the patch stays centred and is clipped at the edges),
`peShift` (the full patch is kept, moved inside the survey) and `peStatic` (one spread
for every shot). Stations missing from the R file are not in the X: a gap starts a new
X record. `compareRelations` compares generated relations with the supplied XPS, record
by record.

```nim
import parseSEG, processSEG

proc orthogonalSps(): SpsFile =
  ## 8 receiver lines along x (200 apart, 61 stations every 50) and 10 source
  ## lines across them (300 apart, 28 points every 50).
  for i in 0 ..< 8:
    for k in 0 ..< 61:
      var p = initSpsPoint(pkReceiver)
      (p.line, p.point) = (float64(1001 + i), float64(101 + k))
      (p.easting, p.northing, p.elevation) = (50.0 * float64(k), 200.0 * float64(i), 0.0)
      result.points.add p
  for m in 0 ..< 10:
    for q in 0 ..< 28:
      var p = initSpsPoint(pkSource)
      (p.line, p.point) = (float64(5001 + m), float64(301 + q))
      (p.easting, p.northing) = (25.0 + 300 * float64(m), 25.0 + 50 * float64(q))
      p.elevation = 0
      result.points.add p

var f = orthogonalSps()
var g = initSurveyGeometry()
g.addSps(f)                                    # R and S: stations only
# X records from a 4 lines × 20 channels patch, rolled on and off at the edges ...
f.relations = g.generateRelations(f.patchShots, initPatchTemplate(4, 20))
g.addSps(SpsFile(relations: f.relations))      # ... added to the geometry
# (`f.writeSps(path)` would write them as an XPS)

let p = describePatches(g)
doAssert p.records.len == 280 and p.clippedRecords > 0
doAssert p.templateLines == 4 and p.templateChannels == 20
doAssert p.layout.interval == 50 and p.layout.lineSpacing == 200

# The same records from a distance box: every receiver within ±475 inline and
# ±325 crossline of the source.
let box = g.generateRelations(f.patchShots, initDistancePatch(475, 325))
var boxed = initSurveyGeometry()
boxed.addSps(f)
boxed.addSps(SpsFile(relations: box))
for a in compareRelations(g, boxed):
  doAssert a.listed > 0 and a.missing.len == 0 # every template receiver is in the box
```

## Binning and fold maps

**The grid.** `surveyGrid(g)` gives the bin grid of the survey, automatically: one fixed
lattice over the whole survey, its inlines along the receiver lines (the median bearing
between consecutive receivers), the natural bin of half the receiver interval by half the
source interval, and an extent that covers the midpoint of every source–receiver pair.
Any of the three can be given instead (`binInline`, `binCrossline`, `azimuthDeg`), and a
`Binning` of your own works everywhere a grid does. A grid and a fold map hold at most
`maxBins` bins (2²⁶ unless you pass more): a bin size typed as 0.01 for 10 raises a
`ValueError` that names the bin sizes and the extent, instead of running out of memory.

**Centring.** The grid is placed so that its bin centres fall where the midpoints cluster.
On a regular survey every midpoint then sits at a bin centre. Where the midpoints scatter
evenly (irregular source points), no placement is better than another, and the grid keeps
its minimum corner. `centre = false` always uses the corner. How it is measured: along
each axis, the circular mean of the midpoints' phase in the bin, taken from the sources'
and the receivers' phases separately (O(stations), not O(pairs)), with stray stations
trimmed and a significance test (`qc/fold` documentation).

On Morgantown the defaults give azimuth 90.26° and bins of 111.4 × 83.6 ft. The 83.6 ft is
half the source interval along its short source lines, which run at about 97.5°, nearly
parallel to the receiver lines. So it is nominal, and a crossline bin of your choice may
suit that survey better.

**Fold.** `preplotFold(g, grid, options)` counts every X relation in the bin of its
midpoint, optionally per offset class, per azimuth sector, and with each bin's offset
range. `initFoldMap` + `addFold` count live traces (by their header midpoint, or half-way
between their source and receiver coordinates), chunk by chunk in any order. Both use the
same midpoint maths, so `foldDifference(pre, actual)` (pre-plot − actual) is exact: positive
where traces are missing, negative where traces fall that the pre-plot has fewer of (a
geometry error). The results are tensors plus the `Binning`, ready for a display.

```nim
import parseSEG, processSEG

proc orthogonalSps(): SpsFile =
  for i in 0 ..< 8:
    for k in 0 ..< 61:
      var p = initSpsPoint(pkReceiver)
      (p.line, p.point) = (float64(1001 + i), float64(101 + k))
      (p.easting, p.northing, p.elevation) = (50.0 * float64(k), 200.0 * float64(i), 0.0)
      result.points.add p
  for m in 0 ..< 10:
    for q in 0 ..< 28:
      var p = initSpsPoint(pkSource)
      (p.line, p.point) = (float64(5001 + m), float64(301 + q))
      (p.easting, p.northing) = (25.0 + 300 * float64(m), 25.0 + 50 * float64(q))
      p.elevation = 0
      result.points.add p

var f = orthogonalSps()
var g = initSurveyGeometry()
g.addSps(f)
g.addSps(SpsFile(relations: g.generateRelations(f.patchShots, initPatchTemplate(4, 20))))

proc recordKeys(g: SurveyGeometry, record: int64): seq[TraceKeys] =
  ## The traces of one record, as a recorder with station headers writes
  ## them, with coordinates from the survey.
  for (channel, rel) in g.relations(record):
    var k = initTraceKeys()
    (k.record, k.channel, k.status) = (float64(record), float64(channel), tsLive)
    (k.srcLine, k.srcPoint) = (rel.source.line, rel.source.point)
    (k.rcvLine, k.rcvPoint) = (rel.receiver.line, rel.receiver.point)
    doAssert k.applySurvey(g)
    result.add k

let grid = surveyGrid(g)                       # 25 × 25 bins along x
let opts = FoldOptions(offsetBounds: @[0.0, 250, 500, 1000], azimuthSectors: 4)
let pre = g.preplotFold(grid, opts)
doAssert pre.fold.shape == [grid.inlines, grid.crosslines]
doAssert pre.offsetFold.shape[0] == 3 and pre.outside == 0

var actual = initFoldMap(grid, opts)
for r in g.records:                            # any order, any chunking
  var keys = g.recordKeys(r)
  if r == 100: keys.delete(4)                  # one trace lost in the field
  let samples = newSeq[int](keys.len)
  actual.addFold(rowsOf(keys, samples))
let diff = foldDifference(pre, actual)
var missing = 0
for v in diff: missing += v
doAssert missing == 1 and min(diff.toFlatSeq) == 0
```

## Record completeness

`recordCompleteness(g, meta)` checks one field record against the XPS. The X record is
found by the header's **source station** (`recordsAt`; when a source was shot more than
once, the FFID chooses), or by the FFID with `rmFieldRecord`; the other key is always
checked, and a mismatch is reported (`recordMismatch`, `sourceMismatch`: the Morgantown
case). Traces match by **receiver station** (by channel only when the headers have no
stations). The result counts the expected, present and live stations, the **missing** ones
as station ranges per line, the **unlisted** live stations, duplicates, renumbered
channels (informational), and per receiver line expected / live / missing.
`summarize(g, records)` totals many records, and lists the XPS records no data matched.

On Morgantown, the file above matches XPS 77 by its source station: 1127 expected, 1096
present, 31 missing, 2858 live stations the XPS does not list, 3 auxiliary traces, in
0.65 ms once the keys are read.

## Amplitude QC

`traceAmplitudeInto(samples, meta, params, amps)` measures each trace: RMS and maximum in
dB, the mean, non-finite samples, clipped runs at the maximum, spikes against their
neighbours, and flags the traces that are dead, constant, clipped, spiky or DC-offset by
their samples, whatever their header status says. `recordAmplitude(amps, params, ws)` then
takes the record's median and MAD of the RMS in dB, and flags the weak and noisy outliers
by a robust z-score. Give it one record at a time; amplitudes fall with offset, so the
near traces of a shot read as noisy (112 of Morgantown's 3954, over whole traces): compare
traces of similar offset, or a window below the first breaks, for a sharper test. `ProcessPool.traceAmplitudeInto` runs
the first step across the pool's workers. For an SNR per trace, use `windowSnrInto`
([qc.md](qc.md#signal-to-noise-ratios)).

```nim
import std/[math, random]
import parseSEG, processSEG

const (traces, cols, dtMs) = (48, 1000, 2.0)
var keys = newSeq[TraceKeys](traces)
for r, k in keys.mpairs:
  k = initTraceKeys()
  (k.channel, k.t0Ms, k.status) = (float64(r + 1), 0.0, tsLive)
var counts = newSeq[int](traces)
for c in counts.mitems: c = cols
var samples = zeros[float32](traces, cols)
var rng = initRand(7)
for r in 0 ..< traces:
  let gain = pow(10.0, rng.gauss(0, 1.5) / 20)  # a field record's scatter, in dB
  for j in 0 ..< cols:
    let t = float64(j) * dtMs / 1000
    samples[r, j] = float32(gain * (sin(2 * PI * 30 * t) * exp(-2 * t) + rng.gauss(0, 0.05)))
for j in 0 ..< cols: samples[10, j] = 0          # dead, although live by header
var peak = 0'f32
for j in 0 ..< cols: peak = max(peak, abs(samples[20, j]))
for j in 0 ..< cols: samples[20, j] = clamp(samples[20, j], -peak / 2, peak / 2)  # clipped
for j in 0 ..< cols: samples[30, j] *= 50        # noisy
samples[40, 700] = 5                             # a spike

let meta = initChunkMeta(rowsOf(keys, counts), dtMs)
var amps = newSeq[TraceAmplitude](traces)
traceAmplitudeInto(samples, meta, AmplitudeParams(), amps)
var ws = initWorkspace()
ws.reserve(recordAmplitudeWorkspace(traces))
let rec = recordAmplitude(amps, AmplitudeParams(), ws)
doAssert amps[10].flags == {afDead}
doAssert afClipped in amps[20].flags
doAssert amps[30].flags == {afNoisy}
doAssert amps[40].flags == {afSpiky} and amps[40].spikes == 1
doAssert rec.flagged == 4 and rec.traces == traces - 1
```
