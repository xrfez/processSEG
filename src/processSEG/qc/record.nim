## Field records checked against the survey: which receivers the X records
## (XPS) list for a shot, and which the data has.
##
## ```nim
## var g = initSurveyGeometry()                   # parseSEG
## for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
## var f = openSeismic(path, keys = true)
## let c = g.recordCompleteness(f.rowsOf(0 ..< f.len))
## echo c                                          # "XPS 77 (by source 129/113): 1127 expected, ..."
## for r in c.missing: echo r.line, ": ", r.firstPoint, "-", r.lastPoint
## if c.recordMismatch: echo "FFID ", c.fieldRecord, " is XPS record ", c.xpsRecord
## ```
##
## **Record match.** A record's traces (one field record: a `ChunkMeta` or a
## `TraceRows` view) are matched to an X record by their **source station**
## (`rmSourceStation`, the default when the headers carry one): parseSEG's
## `recordsAt` gives the X records of that station; with several, the one
## equal to the header FFID, else none (`candidates` lists them). Or by the
## **FFID** (`rmFieldRecord`, the default without a source station). Either
## way the other key is checked, never ignored: `recordMismatch` when the X
## record's number is not the header FFID, `sourceMismatch` when its source
## is not the header's (index 0 matches any). The header source station and
## FFID are those of the first seismic trace that has them.
##
## **Trace match.** By **receiver station** (`tmStation`, the default when
## the headers carry receiver stations): the data's stations are hashed once
## (O(traces)), then the X record's stations are walked in run order
## (O(expected)). By **channel** (`tmChannel`, the default without receiver
## stations): the header channel against the X channel. Recorders often
## number channels their own way, so a matched station whose header channel
## differs from the X channel only counts in `channelMismatch`.
##
## **Counts.** Traces: `traces`, `live` (D12: live or unknown status, not a
## failed row), `dead`, `aux`, `failed`. Stations (channels with
## `tmChannel`; multi-component channels of one station count once):
## `expected` in the X record, `present` in the data (any seismic trace),
## `presentLive` (a live trace), `missing` as ranges of consecutive X
## stations on a line, `unlisted` (live in the data, not in the X record),
## `duplicates` (stations with two seismic traces of one component),
## `noStation` (seismic traces without the key matched on), and per receiver
## line `lines` (expected, live, missing). Auxiliary traces take no part in
## matching.
##
## **Survey.** `summarize` adds up many records and lists the X records no
## data matched.

import std/[algorithm, math, sets, strutils, tables]
import parseSEG
import ../core/types

type
  RecordMatch* = enum
    ## How a record's traces find their X record.
    rmAuto = "auto"              ## By source station when the headers have one, else FFID.
    rmSourceStation = "source"   ## By the header source station (`recordsAt`).
    rmFieldRecord = "FFID"       ## By the header field record number.

  TraceMatch* = enum
    ## How traces find their X channels.
    tmAuto = "auto"        ## By station when the headers have receiver stations, else channel.
    tmStation = "station"  ## By receiver line and point.
    tmChannel = "channel"  ## By the header channel number.

  StationRange* = object
    ## Consecutive X stations of one receiver line (in run order).
    line*, firstPoint*, lastPoint*: float64
    count*: int

  LineCompleteness* = object
    ## One receiver line of an X record.
    line*: float64
    expected*, live*, missing*: int

  RecordCompleteness* = object
    ## One field record against its X record (see the module documentation).
    fieldRecord*: float64    ## Header FFID (NaN: none).
    source*: StationKey      ## Header source station (`hasSource`).
    hasSource*: bool
    recordMatch*: RecordMatch ## How the X record was found (never `rmAuto`).
    traceMatch*: TraceMatch   ## How traces were matched (never `tmAuto`).
    xpsRecord*: int64        ## The X record; −1 for none (no match, or ambiguous).
    candidates*: seq[int64]  ## The source station's X records (`rmSourceStation`).
    xpsSource*: StationKey   ## The X record's source.
    recordMismatch*: bool    ## The X record's number is not the header FFID.
    sourceMismatch*: bool    ## The X record's source is not the header source.
    traces*, live*, dead*, aux*, failed*: int
    expected*, present*, presentLive*, unlisted*, duplicates*, channelMismatch*: int
    noStation*: int
    missing*: seq[StationRange]
    lines*: seq[LineCompleteness] ## Ascending line.

  CompletenessSummary* = object
    ## Many records against the X records of a survey.
    records*: int            ## Records checked.
    matched*: int            ## With an X record.
    unmatched*, ambiguous*: int ## No X record; several candidates, none chosen.
    recordMismatches*, sourceMismatches*: int
    xpsWithoutData*: seq[int64] ## X records no checked record matched, ascending.
    expected*, present*, presentLive*, missing*, unlisted*, duplicates*: int ## Totals.

func missingStations*(c: RecordCompleteness): int {.inline.} =
  ## Expected stations not in the data (`expected − present`).
  c.expected - c.present

func fmt(x: float64): string =
  if x == floor(x) and abs(x) < 1e15: $int64(x) else: $x

func `$`*(s: StationKey): string =
  fmt(s.line) & "/" & fmt(s.point) & (if s.index > 1: "." & $s.index else: "")

func `$`*(c: RecordCompleteness): string =
  result = if c.xpsRecord < 0:
             (if c.candidates.len > 1: "ambiguous: X records " & c.candidates.join(", ")
              else: "no X record")
           else: "XPS " & $c.xpsRecord
  result.add " (by " & (if c.recordMatch == rmSourceStation: "source " & $c.source
                        else: "FFID " & fmt(c.fieldRecord)) & ", traces by " & $c.traceMatch & ")"
  if c.recordMismatch: result.add "; FFID " & fmt(c.fieldRecord) & " differs"
  if c.sourceMismatch: result.add "; X source " & $c.xpsSource & " differs from header " & $c.source
  result.add ": " & $c.expected & " expected, " & $c.present & " present (" & $c.presentLive &
    " live), " & $c.missingStations & " missing in " & $c.missing.len & " ranges, " &
    $c.unlisted & " unlisted, " & $c.duplicates & " duplicated; traces " & $c.traces & ": " &
    $c.live & " live, " & $c.dead & " dead, " & $c.aux & " aux, " & $c.failed & " failed"
  if c.channelMismatch > 0: result.add "; " & $c.channelMismatch & " channel numbers differ"

type
  Seen = object
    index: int16
    seismic, live: int32
    channel: float64      ## Header channel of the first seismic trace.
    components: set[uint8]
    listed: bool

  Status = enum stLive, stDead, stAux, stFailed

func finite(x: float64): bool {.inline.} = x.classify notin {fcNan, fcInf, fcNegInf}

func anyKey(line, point: float64): StationKey {.inline.} = stationKey(line, point, 0)

func channelKey(channel: float64): StationKey {.inline.} =
  # Channel matching reuses the station table: the channel as the line.
  stationKey(channel, 0, 0)

func statusOf(k: TraceKeys): Status {.inline.} =
  case k.status
  of tsLive, tsUnknown: stLive
  of tsDead: stDead
  of tsAuxiliary: stAux

func keyAt(m: ChunkMeta, i: int): lent TraceKeys {.inline.} = m.keys(i)
func statusAt(m: ChunkMeta, i: int): Status {.inline.} =
  if m.isFailed(i): stFailed else: statusOf(m.keys(i))
func keyAt(r: TraceRows, i: int): lent TraceKeys {.inline.} = r.keysAt(i)
func statusAt(r: TraceRows, i: int): Status {.inline.} = statusOf(r.keysAt(i))

proc check[S: ChunkMeta | TraceRows](g: SurveyGeometry, src: S, recordMatch: RecordMatch,
    traceMatch: TraceMatch): RecordCompleteness =
  result = RecordCompleteness(fieldRecord: NaN, xpsRecord: -1)
  let n = src.len
  template keyAt(i: int): untyped = src.keyAt(i)
  template statusAt(i: int): untyped = src.statusAt(i)
  # header keys: FFID, source station, whether receivers have stations
  var rcvStations = false
  for i in 0 ..< n:
    let st = statusAt(i)
    case st
    of stLive: inc result.live
    of stDead: inc result.dead
    of stAux: inc result.aux
    of stFailed: inc result.failed
    if st == stAux: continue
    let k = keyAt(i)
    if result.fieldRecord.isNaN and finite(k.record): result.fieldRecord = k.record
    if not result.hasSource and finite(k.srcLine) and finite(k.srcPoint):
      result.source = stationKey(k.srcLine, k.srcPoint, k.srcIndex)
      result.hasSource = true
    if finite(k.rcvLine) and finite(k.rcvPoint): rcvStations = true
  result.traces = n
  result.recordMatch =
    if recordMatch != rmAuto: recordMatch
    elif result.hasSource: rmSourceStation
    else: rmFieldRecord
  result.traceMatch =
    if traceMatch != tmAuto: traceMatch
    elif rcvStations: tmStation
    else: tmChannel
  # the X record
  case result.recordMatch
  of rmSourceStation:
    if result.hasSource:
      result.candidates = g.recordsAt(result.source)
      if result.candidates.len == 1: result.xpsRecord = result.candidates[0]
      elif finite(result.fieldRecord) and int64(result.fieldRecord).float64 == result.fieldRecord and
          int64(result.fieldRecord) in result.candidates:
        result.xpsRecord = int64(result.fieldRecord)
  of rmFieldRecord, rmAuto:
    if finite(result.fieldRecord) and abs(result.fieldRecord) < 9e18 and
        float64(int64(result.fieldRecord)) == result.fieldRecord:
      for _ in g.relations(int64(result.fieldRecord)):
        result.xpsRecord = int64(result.fieldRecord)
        break
  if result.xpsRecord < 0: return
  # the data's stations (or channels), once
  var seen = initTable[StationKey, Seen](2 * n)
  let byStation = result.traceMatch == tmStation
  for i in 0 ..< n:
    let st = statusAt(i)
    if st == stAux: continue
    let k = keyAt(i)
    let key =
      if byStation:
        if not (finite(k.rcvLine) and finite(k.rcvPoint)):
          inc result.noStation
          continue
        anyKey(k.rcvLine, k.rcvPoint)
      else:
        if not finite(k.channel):
          inc result.noStation
          continue
        channelKey(k.channel)
    seen.withValue(key, e):
      if k.component in e.components: inc result.duplicates
      e.components.incl k.component
      inc e.seismic
      if st == stLive: inc e.live
    do:
      seen[key] = Seen(index: (if byStation: k.rcvIndex else: 0), seismic: 1,
        live: int32(st == stLive), channel: k.channel, components: {k.component})
  # walk the X record
  var listed: HashSet[StationKey]
  var lineAt: Table[float64, int]
  var lastMissing = false
  var first = true
  for (channel, rel) in g.relations(result.xpsRecord):
    if first:
      first = false
      result.xpsSource = rel.source
      result.sourceMismatch = result.hasSource and not sameStation(rel.source, result.source)
      result.recordMismatch = result.fieldRecord != float64(result.xpsRecord)
    let r = rel.receiver
    let key = if byStation: anyKey(r.line, r.point) else: channelKey(float64(channel))
    if listed.containsOrIncl(if byStation: stationKey(r.line, r.point, r.index) else: key):
      continue # another component of a station already counted
    inc result.expected
    var li = lineAt.getOrDefault(r.line, -1)
    if li < 0:
      li = result.lines.len
      lineAt[r.line] = li
      result.lines.add LineCompleteness(line: r.line)
    inc result.lines[li].expected
    var found = false
    seen.withValue(key, e):
      if not byStation or e.index == 0 or r.index == 0 or e.index == r.index:
        found = true
        e.listed = true
        inc result.present
        if e.live > 0:
          inc result.presentLive
          inc result.lines[li].live
        if byStation and e.channel != float64(channel): inc result.channelMismatch
    if found:
      lastMissing = false
    else:
      inc result.lines[li].missing
      if lastMissing and result.missing[^1].line == r.line:
        result.missing[^1].lastPoint = r.point
        inc result.missing[^1].count
      else:
        result.missing.add StationRange(line: r.line, firstPoint: r.point, lastPoint: r.point,
          count: 1)
      lastMissing = true
  for e in seen.values:
    if not e.listed and e.live > 0: inc result.unlisted
  result.lines.sort(proc (a, b: LineCompleteness): int = cmp(a.line, b.line))

proc recordCompleteness*(g: SurveyGeometry, meta: ChunkMeta, recordMatch = rmAuto,
    traceMatch = tmAuto): RecordCompleteness =
  ## Checks the rows of `meta`, one field record, against their X record
  ## (see the module documentation). Failed rows count as `failed` and,
  ## like dead traces, are present but not live.
  check(g, meta, recordMatch, traceMatch)

proc recordCompleteness*(g: SurveyGeometry, rows: TraceRows, recordMatch = rmAuto,
    traceMatch = tmAuto): RecordCompleteness =
  ## `recordCompleteness` of a view with no failed rows. Raises `ValueError`
  ## when the view has no keys.
  if not rows.hasKeys:
    raise newException(ValueError, "the rows have no keys: open the file with " &
      "keys = true (or call ensureKeys) before making the view")
  check(g, rows, recordMatch, traceMatch)

proc summarize*(g: SurveyGeometry, records: openArray[RecordCompleteness]): CompletenessSummary =
  ## Totals over `records` and the X records of `g` that none of them matched.
  var used: HashSet[int64]
  for c in records:
    inc result.records
    if c.xpsRecord >= 0:
      inc result.matched
      used.incl c.xpsRecord
    elif c.candidates.len > 1: inc result.ambiguous
    else: inc result.unmatched
    if c.recordMismatch: inc result.recordMismatches
    if c.sourceMismatch: inc result.sourceMismatches
    result.expected += c.expected
    result.present += c.present
    result.presentLive += c.presentLive
    result.missing += c.missingStations
    result.unlisted += c.unlisted
    result.duplicates += c.duplicates
  for r in g.records:
    if r notin used: result.xpsWithoutData.add r
