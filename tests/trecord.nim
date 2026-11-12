## Record completeness against the XPS (`qc/record`): a synthetic record
## with dropped, duplicated, extra and mis-stationed channels, dead, aux and
## failed traces and renumbered channels reports exactly those; record match
## by source station and by FFID with the other key checked; channel
## matching; repeated sources; the survey summary; Morgantown (skipped when
## its files are absent).

import std/[assertions, math, os, sets, strutils, times]
import parseSEG
import processSEG/core/types
import processSEG/qc/record
import utils/survey

let s = initSyntheticSurvey()
let g = s.geometry
# a shot with the full 4 × 11 patch: source line 3, station 20
var shot: ShotTemplate
for t in s.shots:
  if t.sourceLine == 3 and t.sourceStation == 20: shot = t
doAssert shot.lines.len == 4 and shot.lastStation - shot.firstStation == 10
let srcLine = s.lineNumber(shot.sourceLine)
let srcPoint = s.pointNumber(shot.sourceStation)

proc recordKeys(): seq[TraceKeys] =
  ## One live trace per X channel, as a recorder writes them.
  for (c, rel) in g.relations(shot.record):
    var k = initTraceKeys()
    k.record = float64(shot.record)
    (k.srcLine, k.srcPoint, k.srcIndex) = (srcLine, srcPoint, 0'i16) # SEG-D: index 0
    (k.rcvLine, k.rcvPoint, k.rcvIndex) = (rel.receiver.line, rel.receiver.point, 0'i16)
    k.channel = float64(c)
    k.status = tsLive
    result.add k

proc find(keys: seq[TraceKeys], line, point: float64): int =
  for i, k in keys:
    if k.rcvLine == line and k.rcvPoint == point: return i
  -1

block clean:
  var keys = recordKeys()
  let c = g.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert c.recordMatch == rmSourceStation and c.traceMatch == tmStation
  doAssert c.xpsRecord == shot.record and c.candidates == @[shot.record]
  doAssert not c.recordMismatch and not c.sourceMismatch
  doAssert c.expected == 44 and c.present == 44 and c.presentLive == 44
  doAssert c.missing.len == 0 and c.unlisted == 0 and c.duplicates == 0
  doAssert c.channelMismatch == 0 and c.noStation == 0
  doAssert c.traces == 44 and c.live == 44 and c.lines.len == 4
  for l in c.lines: doAssert l.expected == 11 and l.live == 11 and l.missing == 0

block no_ffid:
  # No FFID in the headers: matched by source station, no mismatch.
  var keys = recordKeys()
  for k in keys.mitems: k.record = NaN
  let c = g.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert c.xpsRecord == shot.record and not c.recordMismatch
  doAssert summarize(g, [c]).recordMismatches == 0

block station_indices:
  # Headers with indices: index 1 matches X (index 1); a second trace at
  # the same line and point with index 2 is another station, not a
  # duplicate, and unlisted.
  var keys = recordKeys()
  for k in keys.mitems: k.rcvIndex = 1
  var other = keys[0]
  other.rcvIndex = 2
  keys.add other
  let c = g.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert c.present == 44 and c.missing.len == 0, $c
  doAssert c.duplicates == 0 and c.unlisted == 1, $c
  # Index 2 alone where X says 1: missing, and the trace unlisted.
  keys = recordKeys()
  keys[0].rcvIndex = 2
  let d = g.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert d.present == 43 and d.missingStations == 1 and d.unlisted == 1, $d

block faults:
  var keys = recordKeys()
  let l0 = s.lineNumber(shot.lines[0])
  let l2 = s.lineNumber(shot.lines[2])
  let p = proc (k: int): float64 = s.pointNumber(shot.firstStation + k)
  # dropped: stations 3-4 of the first line (one range), 9 of the third
  var drop: seq[int]
  for (l, k) in [(l0, 3), (l0, 4), (l2, 9)]: drop.add keys.find(l, p(k))
  # mis-stationed: station 6 of the third line carries a station of line 1001
  let mis = keys.find(l2, p(6))
  # renumbered: two channels off by 500
  for (l, k) in [(l0, 0), (l0, 1)]: keys[keys.find(l, p(k))].channel += 500
  # a dead trace and a failed row stay present, not live
  let dead = keys.find(l0, p(7))
  keys[dead].status = tsDead
  let failedRow = keys.find(l2, p(0))
  var kept: seq[TraceKeys]
  var failed: seq[int]
  for i, k in keys:
    if i notin drop:
      if i == failedRow: failed.add kept.len
      kept.add k
  let mi = kept.find(l2, p(6))
  kept[mi].rcvLine = 1001
  kept[mi].rcvPoint = p(6)
  # duplicated: line 0, station 10 twice; extra: a live station outside the patch
  var dup = kept[kept.find(l0, p(10))]
  kept.add dup
  var extra = kept[0]
  (extra.rcvLine, extra.rcvPoint, extra.channel) = (1008.0, 140.0, 999.0)
  kept.add extra
  # an aux trace (no station) and a seismic trace without a station
  var aux = initTraceKeys()
  aux.status = tsAuxiliary
  kept.add aux
  var bare = kept[1]
  (bare.rcvLine, bare.rcvPoint) = (NaN, NaN)
  kept.add bare
  let counts = newSeq[int](kept.len)
  let meta = initChunkMeta(rowsOf(kept, counts), dtMs = 2, failed = failed)
  let c = g.recordCompleteness(meta)
  echo "  ", c
  doAssert c.xpsRecord == shot.record and not c.recordMismatch and not c.sourceMismatch
  doAssert c.expected == 44
  doAssert c.present == 44 - 3 - 1, $c.present             # dropped, mis-stationed
  doAssert c.presentLive == c.present - 2                    # dead, failed
  doAssert c.missingStations == 4
  doAssert c.missing.len == 3, $c.missing
  doAssert c.missing[0] == StationRange(line: l0, firstPoint: p(3), lastPoint: p(4), count: 2)
  doAssert c.missing[1] == StationRange(line: l2, firstPoint: p(6), lastPoint: p(6), count: 1)
  doAssert c.missing[2] == StationRange(line: l2, firstPoint: p(9), lastPoint: p(9), count: 1)
  doAssert c.unlisted == 2, $c.unlisted                      # mis-stationed, extra
  doAssert c.duplicates == 1 and c.channelMismatch == 2 and c.noStation == 1
  doAssert c.traces == kept.len and c.aux == 1 and c.dead == 1 and c.failed == 1
  doAssert c.live == kept.len - 3
  for l in c.lines:
    if l.line == l0: doAssert l.expected == 11 and l.missing == 2 and l.live == 8
    elif l.line == l2: doAssert l.missing == 2 and l.live == 8
    else: doAssert l.missing == 0 and l.live == 11

block ffid_mismatch:
  # the header FFID is another shot's
  var keys = recordKeys()
  let other = shot.record + 5
  for k in keys.mitems: k.record = float64(other)
  let rows = rowsOf(keys, newSeq[int](keys.len))
  let bySource = g.recordCompleteness(rows)
  doAssert bySource.xpsRecord == shot.record and bySource.recordMismatch
  doAssert not bySource.sourceMismatch and bySource.presentLive == 44
  let byFfid = g.recordCompleteness(rows, recordMatch = rmFieldRecord)
  doAssert byFfid.recordMatch == rmFieldRecord and byFfid.xpsRecord == other
  doAssert byFfid.sourceMismatch and not byFfid.recordMismatch
  doAssert byFfid.xpsSource != bySource.xpsSource
  doAssert "differs" in $byFfid
  # no source station in the headers: FFID by default
  for k in keys.mitems: (k.srcLine, k.srcPoint) = (NaN, NaN)
  let noSrc = g.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert noSrc.recordMatch == rmFieldRecord and noSrc.xpsRecord == other
  doAssert not noSrc.hasSource and not noSrc.sourceMismatch
  # an FFID with no X record
  for k in keys.mitems: k.record = 999_999
  let none = g.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert none.xpsRecord == -1 and none.expected == 0

block channel_match:
  var keys = recordKeys()
  for k in keys.mitems: (k.rcvLine, k.rcvPoint) = (NaN, NaN)
  keys.delete(5)
  keys.delete(5)
  let c = g.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert c.traceMatch == tmChannel and c.expected == 44 and c.present == 42
  doAssert c.missing.len == 1 and c.missing[0].count == 2 and c.channelMismatch == 0
  # by channel although stations are there: renumbered channels go missing
  var k2 = recordKeys()
  k2[0].channel = 1000
  let byChannel = g.recordCompleteness(rowsOf(k2, newSeq[int](k2.len)), traceMatch = tmChannel)
  doAssert byChannel.present == 43 and byChannel.unlisted == 1

block repeated_source:
  # the shot repeated as a second X record: the FFID picks one
  var f = s.spsFile
  var repeat: seq[SpsRelation]
  for r in f.relations:
    if r.fieldRecord == shot.record:
      var x = r
      x.fieldRecord = 5000
      repeat.add x
  f.relations.add repeat
  var g2 = initSurveyGeometry()
  g2.addSps(f)
  var keys = recordKeys()
  for k in keys.mitems: k.record = 5000
  let c = g2.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert c.candidates == @[shot.record, 5000'i64] and c.xpsRecord == 5000
  doAssert not c.recordMismatch
  for k in keys.mitems: k.record = 7
  let amb = g2.recordCompleteness(rowsOf(keys, newSeq[int](keys.len)))
  doAssert amb.xpsRecord == -1 and amb.candidates.len == 2 and "ambiguous" in $amb
  let sum = g2.summarize([c, amb])
  doAssert sum.records == 2 and sum.matched == 1 and sum.ambiguous == 1
  doAssert sum.expected == 44 and sum.present == 44
  doAssert sum.xpsWithoutData.len == s.shots.len and 5000'i64 notin sum.xpsWithoutData
  doAssert shot.record in sum.xpsWithoutData

const
  dir = currentSourcePath().parentDir.parentDir / "exampleFiles"
  base = dir / "sps" / "PGI_NNE_West_Morgantown_3D_Post261019_NAD27_SPCS_WV_N_USFeet"
  shotFile = dir / "00000151_12900_11300.sgd"

block morgantown:
  if not (fileExists(base & ".sps") and fileExists(base & ".rps") and
      fileExists(base & ".xps") and fileExists(shotFile)):
    echo "  skipped: Morgantown files not present"
  else:
    var g = initSurveyGeometry()
    for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
    var f = openSeismic(shotFile, keys = true)
    let rows = f.rowsOf(0 ..< f.len)
    let t0 = cpuTime()
    let c = g.recordCompleteness(rows)
    let ms = (cpuTime() - t0) * 1000
    echo "  Morgantown: ", c, " (", formatFloat(ms, ffDecimal, 2), " ms)"
    doAssert c.recordMatch == rmSourceStation and c.xpsRecord == 77 and c.recordMismatch
    doAssert c.fieldRecord == 151 and not c.sourceMismatch
    doAssert c.expected == 1127 and c.present == 1096 and c.missingStations == 31
    doAssert c.unlisted == 2858 and c.aux == 3 and c.duplicates == 0
    var missing = initHashSet[(float64, float64)]()
    for r in c.missing:
      var p = r.firstPoint
      while p <= r.lastPoint:
        missing.incl (r.line, p)
        p += 1
    for st in [(100.0, 147.0), (104.0, 227.0), (108.0, 187.0), (108.0, 188.0),
        (124.0, 181.0), (124.0, 184.0)]:
      doAssert st in missing, $st
    let byFfid = g.recordCompleteness(rows, recordMatch = rmFieldRecord)
    echo "  Morgantown by FFID: ", byFfid
    doAssert byFfid.xpsRecord == 151 and byFfid.sourceMismatch
    when defined(release) or defined(danger): doAssert ms < 5, $ms
    f.close()
