## Cutting continuous recordings into records at source times
## (`SliceStreams`, `SlicePlan`, `extractInto`).
##
## Nodes record continuously; a file stores each receiver's stream as
## consecutive segments, one trace row per segment. `initSliceStreams` works
## out, once per dataset, which receiver each segment belongs to and where it
## sits on that receiver's sample grid; `planSlices` then lays out, for
## every (receiver, shot) pair, which rows and columns make up the window
## `shot time + startMs ..< + lengthMs`; `extractInto` copies and stitches
## them into a new `[traces, samples]` tensor (a window can straddle segment
## boundaries, so this is not in place). Data in, no files: the caller reads
## the segments with parseSEG, and writes the records with `sliceKeys`
## (decision D14).
##
## ```nim
## # keys and samples of the segments, read with parseSEG
## let meta = initChunkMeta(rowsOf(keys, numSamples), dtMs = 2)
## let streams = initSliceStreams(meta)          # once: keep it
## let shots = @[initSliceShot(time = 1.7e9 + 12.5, id = 101)]
## let plan = planSlices(streams, shots, SliceWindow(startMs: 0, lengthMs: 4000))
## # plan.tracesNeeded: the only traces to read (or read them all)
## var dst = zeros[float32](plan.len, plan.samples)
## var report = initSliceReport(plan)
## var ws = initWorkspace()
## ws.reserve(sliceWorkspace(float32, plan))
## extractInto(plan, data, meta, dst, report, ws)
## for trace in 0 ..< plan.len:
##   let k = plan.sliceKeys(trace)           # write with parseSEG
##   for gap in report.gaps(trace): discard   # zero-filled columns
## ```
##
## **Streams.** A `SliceStreams` holds everything about the segments that
## does not depend on the shots: the receivers, each segment's position,
## the overlaps and the timing issues. Making it sorts the rows (130 to
## 260 ms for a million of them); a plan made from it costs only the traces
## it lays out (0.06 ms for a shot on 1000 receivers), so a viewer makes the
## streams when it opens the data and a plan for every shot it shows.
## `planSlices(meta, ...)` does both in one call, for a plan made once. The
## streams copy what they need: they stay valid when the meta is gone, as
## long as the rows and their keys are the same, and are only read by
## `planSlices`, so several threads may plan from one. Everything listed
## below under `plan.` for the whole dataset (`overlaps`, `timingIssues`,
## `untimed`, `otherInterval`) is in the streams too.
##
## **Segments.** A segment is a live row (`ChunkMeta.isLive`) with a live
## length (`liveSamples`) and a time: by default `TraceKeys.time` (UTC POSIX
## seconds, float64: about 0.24 µs resolution today), or a caller array (SEG-Y
## bytes 157-168 hold whole seconds, so better times often come from
## elsewhere). The time is the trace's time zero, as `TraceKeys.time`: column
## 0 of the row is at `time + sampleTimeMs(row, 0) / 1000`. Rows with a NaN
## or infinite time (or `t0Ms`), or one beyond ±1e11 s (year 5000), are not
## used and are listed in `plan.untimed`; dead, auxiliary and failed rows
## are not used either (their span is a gap). Nor is a row whose keys give
## another sample interval than the meta's (`intervalUs`, when known): mixed
## sample rates are a property of the data, so such rows are listed in
## `plan.otherInterval`, and are sliced with a meta of their own interval.
##
## **Receivers.** Segments are grouped per receiver and never stitched
## across receivers, even when their times line up. By default a receiver is
## its station as parseSEG's `stationKey` makes it, the rule of the survey
## join and of record QC: `rcvLine` and `rcvPoint` rounded to 1/100, with the
## point index `rcvIndex` (two deployments at one point are two receivers).
## At one station each `channel` and `component` is a stream of its own, and
## without a station (NaN, equal to NaN) they alone tell the streams apart.
## A caller array of group ids overrides that. Receivers are numbered in
## ascending key (or id) order, so the output does not depend on the order
## of the rows.
##
## **Sample grid.** Within a receiver, segment positions are whole samples
## from its earliest segment: `round((start − first start) / dt)`. Times are
## subtracted as float64 seconds before dividing, never accumulated. A
## segment whose start is off the previous segment's grid by more than the
## tolerance, `max(0.01 sample, 1 µs)`, is a timing inconsistency, listed in
## `plan.timingIssues` (it is still placed at the nearest whole sample). The
## same tolerance decides when a window start counts as on the grid.
##
## **Window start.** Output column 0 is the sample of the receiver's stream
## nearest the window start, `shot.time + startMs / 1000`, measured on the
## grid of the segment that holds the window's first data. The rest,
## `plan.fraction(trace)` samples in `[-0.5, 0.5]` (0 within the tolerance), is
## left as is with `saNearest`, or shifted out (`core/shift`, linear or cubic)
## with `saLinear` and `saCubic`, which read `plan.margin` extra samples on
## each side for the interpolation.
##
## **Gaps.** Window samples no segment covers are zero and reported per output
## trace, as column runs (`report.gaps(trace)`); so are the columns of a row
## that is not live (failed) at extract time. With interpolation the samples
## next to a gap mix in its zeros.
##
## **Overlaps.** Where two segments of one receiver overlap, the later one
## (by start time) wins: the earlier one is trimmed where the later covers it.
## Every overlap is in `plan.overlaps` (rows and samples), so a caller who
## wants to refuse them checks the plan first; there is no option. A
## non-whole overlap is also a timing issue. When both rows are read
## (`rowsNeeded` holds both), `extractInto` compares the overlapping samples
## and reports a mismatch (`report.overlapCheck(i)`), which means a timing
## error, not a duplicate.
##
## **Clock drift.** Segment times are taken as the truth. A receiver whose
## clock drifted is corrected by the caller before planning, with
## `driftCorrectTimes` (linear between two clock checks).
##
## **Output keys** (`sliceKeys`): the receiver's keys (from its earliest
## segment), the shot's record number and source position, offset and
## midpoint derived from the coordinates, and the time convention of a shot
## record: `time` is the shot time and `t0Ms` the window's `startMs`, so
## sample `j` is at `time + (t0Ms + j · dtMs) / 1000` and traveltimes are
## relative to the source, as every other processSEG kernel expects.
##
## **Order.** `soReceiverGathers`: trace `r · shots + s` (one receiver, all
## shots); `soShotRecords`: trace `s · receivers + r`. Shots keep the
## caller's order; every pair is a trace, all gap if no data reaches it.

import std/[algorithm, bitops, math]
import parseSEG
import ../core/[rows, shift, types, workspace]

export ShiftInterp

type
  SliceOrder* = enum
    ## The order of the output traces.
    soReceiverGathers ## Trace `r · shotCount + s`: one receiver, all shots.
    soShotRecords     ## Trace `s · receiverCount + r`: one shot, all receivers.

  SliceAlign* = enum
    ## What happens to the sub-sample part of a window start.
    saNearest ## Start at the nearest stream sample; no interpolation.
    saLinear  ## Shift it out by linear interpolation (one sample margin).
    saCubic   ## Shift it out by cubic (Catmull-Rom) interpolation (two samples).

  SliceShot* = object
    ## A source event. Build it with `initSliceShot` (unknown fields NaN).
    time*: float64   ## UTC POSIX seconds of the source's time zero.
    id*: float64     ## Its record (FFID) number: the output `record`.
    srcLine*, srcPoint*: float64
    srcX*, srcY*, srcZ*: float64
    srcIndex*: int16 ## The source point index; 0: unknown.

  SliceWindow* = object
    ## The window cut at each shot.
    startMs*: float64  ## Start relative to the shot time, ms; may be negative.
    lengthMs*: float64 ## Length, ms: the listen time, or listen + sweep for
                       ## uncorrelated vibroseis data. A whole number of samples.

  SlicePiece* = object
    ## A run of output columns of one trace and where it comes from.
    row*: int   ## The input row (of the planning meta); −1 for a gap.
    first*: int ## The row's first column read (0 for a gap).
    dst*: int   ## The first output column; with `saLinear`/`saCubic` the
                ## pieces reach `margin` columns before 0 and after `samples`.
    count*: int ## The number of columns.

  SliceOverlap* = object
    ## Two segments of one receiver that overlap: the later one wins.
    earlier*, later*: int ## The rows, by start time.
    first*: int           ## The earlier row's first column the later covers.
    samples*: int         ## The overlapping samples (the later row's from 0).

  SliceTimingIssue* = object
    ## A segment whose start is off the previous segment's sample grid.
    earlier*, later*: int ## The rows, consecutive in time on one receiver.
    residual*: float64    ## The misfit in samples, between −0.5 and 0.5.

  SlicePlan* = object
    ## Which rows and columns make up every output trace; see the module
    ## documentation. Built by `planSlices`; read-only.
    dtMs: float64
    samples, margin, rowCount, firstSample, maxColumn: int
    order: SliceOrder
    align: SliceAlign
    window: SliceWindow
    shots: seq[SliceShot]
    receiverKeys: seq[TraceKeys]
    spans: seq[SlicePiece]
    spanStart: seq[int] ## Per trace, into `spans`; `len + 1` entries.
    fraction: seq[float64]
    gapSamples: seq[int]
    rowsNeeded: seq[int]
    tracesNeeded: seq[int] ## `traceOf` of `rowsNeeded`, in that order.
    neededBits: seq[uint64] ## A bit per row: set when it is in `rowsNeeded`.
    neededRank: seq[int] ## Per word of `neededBits`: the needed rows before it.
    overlaps: seq[SliceOverlap]
    issues: seq[SliceTimingIssue]
    untimed: seq[int]
    otherInterval: seq[int]

  SliceGap* = object
    ## A run of zero-filled output columns.
    first*, count*: int

  OverlapStatus* = enum
    ## The outcome of comparing an overlap's samples.
    osNotRead  ## Not compared: a row is not in `rowsNeeded` or not live, or
               ## the block is narrower than the overlap.
    osMatch    ## Both rows hold the same samples.
    osMismatch ## They differ: a timing error.

  SliceOverlapCheck* = object
    ## The comparison of one overlap (`plan.overlaps[i]`).
    status*: OverlapStatus
    differing*: int      ## The samples that differ.
    maxAbsDiff*: float64 ## The largest difference.

  SliceReport* = object
    ## What `extractInto` found: gaps per trace and overlap comparisons.
    ## Sized by `initSliceReport`; reused without allocating.
    gaps: seq[SliceGap]
    gapStart: seq[int] ## Per trace, into `gaps`; `len + 1` entries.
    gapSamples: seq[int]
    checks: seq[SliceOverlapCheck]
    capacity: int

  Segment = object
    recv, row, n, p: int
    start, resid: float64

  Interval = object
    a, n, row, col: int ## Grid start, length, row, row column at `a`.
    resid: float64      ## The row's start off the grid, samples.

  SliceStreams* = object
    ## The receivers of a set of segment rows and where every segment sits on
    ## its receiver's sample grid: all that `planSlices` works out before it
    ## looks at a shot. Built once per dataset by `initSliceStreams`; see the
    ## module documentation. It copies what it needs from the meta (no
    ## pointer into the file or the keys), so it can be kept as long as the
    ## rows stay the same, and read from several threads at once.
    dtMs: float64
    rowCount, firstSample: int
    traces: seq[int] ## Per row: its trace (`TraceRows.traceOf`).
    receiverKeys: seq[TraceKeys]
    firstStart: seq[float64] ## Per receiver: the start of its earliest segment, s.
    ivs: seq[Interval] ## The columns each segment wins, per receiver by position.
    ivStart: seq[int] ## Per receiver, into `ivs`; one more entry.
    overlaps: seq[SliceOverlap]
    issues: seq[SliceTimingIssue]
    untimed: seq[int]
    otherInterval: seq[int]

  ReceiverId = object
    ## What tells two receivers apart: the station as parseSEG keys it, then
    ## the channel and the component at it.
    line, point, channel: float64
    index: int16
    component: uint8

func initSliceShot*(time, id: float64, srcLine = NaN, srcPoint = NaN, srcX = NaN,
    srcY = NaN, srcZ = NaN, srcIndex = 0'i16): SliceShot =
  ## A shot at UTC POSIX seconds `time` with record number `id`, and its
  ## source station and position when known.
  SliceShot(time: time, id: id, srcLine: srcLine, srcPoint: srcPoint, srcX: srcX,
    srcY: srcY, srcZ: srcZ, srcIndex: srcIndex)

proc badSlice(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

const MaxSegmentSamples = 1 shl 48
  # The longest row placed on the sample grid: far beyond any file (SEG-D
  # counts samples in 32 bits), and small enough that a position plus a
  # length stays within an int for any time `planSlices` accepts.

func toleranceSamples(dtMs: float64): float64 =
  # 1 µs in samples (a SEG-D timestamp's resolution, a few float64 ulps of a
  # POSIX time), but never under 0.01 sample.
  max(0.01, 1e-3 / dtMs)

func cmpNaN(a, b: float64): int =
  if a.isNaN: (if b.isNaN: 0 else: 1)
  elif b.isNaN: -1
  else: cmp(a, b)

func receiverId(k: TraceKeys): ReceiverId =
  # The station as parseSEG keys it: line and point to 1/100, with the index.
  let st = stationKey(k.rcvLine, k.rcvPoint, k.rcvIndex)
  ReceiverId(line: st.line, point: st.point, index: st.index, channel: k.channel,
    component: k.component)

func cmpReceiver(a, b: ReceiverId): int =
  result = cmpNaN(a.line, b.line)
  if result == 0: result = cmpNaN(a.point, b.point)
  if result == 0: result = cmp(a.index, b.index)
  if result == 0: result = cmpNaN(a.channel, b.channel)
  if result == 0: result = cmp(a.component, b.component)

func marginOf(align: SliceAlign): int =
  case align
  of saNearest: 0
  of saLinear: 1
  of saCubic: 2

func firstEndingAfter(ivs: seq[Interval], lo, hi, pos: int): int =
  # The first of the sorted, disjoint intervals `ivs[lo ..< hi]` that ends
  # after `pos` (`hi` when none does).
  var a = lo
  var b = hi
  while a < b:
    let mid = (a + b) shr 1
    if ivs[mid].a + ivs[mid].n <= pos: a = mid + 1 else: b = mid
  a

proc checkInterval(dtMs: float64) =
  if dtMs < 1e-4: # keeps sample positions of any POSIX time within an int
    badSlice("dtMs must be at least 1e-4 ms to slice, got " & $dtMs)

proc windowSamples(window: SliceWindow, dtMs: float64): int =
  # The samples of the window; raises for one that is not a whole number.
  if window.startMs.classify notin {fcNormal, fcZero, fcNegZero, fcSubnormal}:
    badSlice("window.startMs must be finite, got " & $window.startMs)
  if window.lengthMs.classify notin {fcNormal, fcSubnormal} or window.lengthMs <= 0:
    badSlice("window.lengthMs must be finite and positive, got " & $window.lengthMs)
  let exactLen = window.lengthMs / dtMs
  if exactLen > float64(high(int32)):
    badSlice("window.lengthMs is too long: " & $window.lengthMs & " ms")
  result = int(round(exactLen))
  if result < 1 or abs(exactLen - float64(result)) > 1e-6 * max(1.0, exactLen):
    badSlice("window.lengthMs (" & $window.lengthMs &
      " ms) must be a whole number of samples of " & $dtMs & " ms")

proc checkShots(shots: openArray[SliceShot]) =
  for i, s in shots:
    if s.time.classify notin {fcNormal, fcZero, fcNegZero, fcSubnormal}:
      badSlice("shot " & $i & " has no finite time: " & $s.time)

proc checkPerRow(count, rows: int, what: string) =
  if count != rows:
    badSlice(what & " has " & $count & " entries, the meta " & $rows & " rows")

proc streamsOf(meta: ChunkMeta, times: openArray[float64],
    recv: openArray[int]): SliceStreams =
  # Everything about the segments that does not depend on the shots: at a
  # million rows this is 130 to 250 ms (two sorts), against 0.05 ms for
  # every shot laid out afterwards.
  let dtMs = meta.dtMs
  checkInterval(dtMs)
  let dtS = dtMs / 1000
  let tol = toleranceSamples(dtMs)
  result = SliceStreams(dtMs: dtMs, rowCount: meta.len, firstSample: meta.firstSample,
    traces: newSeq[int](meta.len))

  # The segments, sorted by receiver, then start (ties: length, row).
  var segs: seq[Segment]
  for row in 0 ..< meta.len:
    result.traces[row] = meta.rows.traceOf(row)
    let n = meta.liveSamples(row, high(int))
    if meta.isLive(row) and n > 0:
      if n > MaxSegmentSamples:
        badSlice("row " & $row & " has " & $n & " samples, more than the " &
          $MaxSegmentSamples & " a segment may have")
      let iu = meta.keys(row).intervalUs
      let start = times[row] + meta.sampleTimeMs(row, 0) / 1000
      if iu.classify in {fcNormal, fcSubnormal} and
          abs(iu - dtMs * 1000) > 1e-6 * dtMs * 1000:
        result.otherInterval.add row
      elif start.classify in {fcNormal, fcZero, fcNegZero, fcSubnormal} and
          abs(start) <= 1e11:
        segs.add Segment(recv: recv[row], row: row, n: n, start: start)
      else:
        result.untimed.add row
  segs.sort do (a, b: Segment) -> int:
    result = cmp(a.recv, b.recv)
    if result == 0: result = cmp(a.start, b.start)
    if result == 0: result = cmp(a.n, b.n)
    if result == 0: result = cmp(a.row, b.row)

  # Per receiver: grid positions, overlaps, and the intervals each segment
  # wins (the later segment wins where two overlap).
  var g0 = 0
  while g0 < segs.len:
    var g1 = g0 + 1
    while g1 < segs.len and segs[g1].recv == segs[g0].recv: inc g1
    result.ivStart.add result.ivs.len
    result.receiverKeys.add meta.keys(segs[g0].row)
    let s0 = segs[g0].start
    result.firstStart.add s0
    for i in g0 ..< g1:
      let exact = (segs[i].start - s0) / dtS
      segs[i].p = int(round(exact))
      segs[i].resid = exact - float64(segs[i].p)
      if i > g0:
        var d = segs[i].resid - segs[i - 1].resid
        d -= round(d) # the misfit modulo whole samples
        if abs(d) > tol:
          result.issues.add SliceTimingIssue(earlier: segs[i - 1].row, later: segs[i].row,
            residual: d)
    let first = result.ivs.len
    for i in g0 ..< g1:
      let pi = segs[i].p
      let ei = pi + segs[i].n
      var cur = pi
      var j = i + 1
      while j < g1 and segs[j].p < ei:
        let pj = segs[j].p
        let ej = pj + segs[j].n
        result.overlaps.add SliceOverlap(earlier: segs[i].row, later: segs[j].row,
          first: pj - pi, samples: min(ei, ej) - pj)
        if pj > cur:
          result.ivs.add Interval(a: cur, n: pj - cur, row: segs[i].row, col: cur - pi,
            resid: segs[i].resid)
        cur = max(cur, ej)
        inc j
      if cur < ei:
        result.ivs.add Interval(a: cur, n: ei - cur, row: segs[i].row, col: cur - pi,
          resid: segs[i].resid)
    sort(toOpenArray(result.ivs, first, result.ivs.high)) do (x, y: Interval) -> int:
      cmp(x.a, y.a)
    g0 = g1
  result.ivStart.add result.ivs.len

proc defaultReceivers(meta: ChunkMeta): seq[int] =
  # Group ids in ascending (station, channel, component) order. The ids are
  # made once per row: the sort compares them some twenty times each.
  var ids = newSeq[ReceiverId](meta.len)
  var idx = newSeq[int](meta.len)
  for row in 0 ..< meta.len:
    ids[row] = receiverId(meta.keys(row))
    idx[row] = row
  idx.sort do (a, b: int) -> int: cmpReceiver(ids[a], ids[b])
  result = newSeq[int](meta.len)
  var id = 0
  for i, row in idx:
    if i > 0 and cmpReceiver(ids[idx[i - 1]], ids[row]) != 0: inc id
    result[row] = id

proc keyTimes(meta: ChunkMeta): seq[float64] =
  result = newSeq[float64](meta.len)
  for row in 0 ..< meta.len: result[row] = meta.keys(row).time

proc initSliceStreams*(meta: ChunkMeta, times: openArray[float64],
    receivers: openArray[int]): SliceStreams =
  ## The streams of the segments of `meta` (rows of continuous data), once
  ## per dataset: `times[row]` is the row's time zero in UTC POSIX seconds
  ## (as `TraceKeys.time`; NaN: not used) and `receivers[row]` its receiver
  ## (any id; equal ids are one stream). `planSlices(streams, shots, ...)`
  ## then lays out any shots at the cost of their traces alone. Make it
  ## again when the rows or their keys change.
  ##
  ## Raises `ValueError` when `times` or `receivers` does not have one entry
  ## per row, `meta.dtMs` is under 1e-4 ms, or a used row has more than 2⁴⁸
  ## samples (no file holds such a trace). Data problems (overlaps, off-grid
  ## segments, rows without a time or of another sample interval) are
  ## reported in the streams.
  checkPerRow(times.len, meta.len, "times")
  checkPerRow(receivers.len, meta.len, "receivers")
  streamsOf(meta, times, receivers)

proc initSliceStreams*(meta: ChunkMeta, times: openArray[float64]): SliceStreams =
  ## `initSliceStreams` with caller times and the receivers from the keys
  ## (the station `rcvLine`, `rcvPoint`, `rcvIndex`, then `channel` and
  ## `component`; see the module documentation).
  checkPerRow(times.len, meta.len, "times")
  streamsOf(meta, times, defaultReceivers(meta))

proc initSliceStreams*(meta: ChunkMeta): SliceStreams =
  ## `initSliceStreams` with the times and receivers from the keys:
  ## `TraceKeys.time` and the station (`rcvLine`, `rcvPoint`, `rcvIndex`),
  ## then `channel` and `component`.
  streamsOf(meta, keyTimes(meta), defaultReceivers(meta))

proc layOut(streams: SliceStreams, shots: openArray[SliceShot], window: SliceWindow,
    samples: int, order: SliceOrder, align: SliceAlign): SlicePlan =
  # The pieces of every trace and the rows they read; the lists a plan
  # shares with its streams (receiver keys, data problems) are the caller's
  # to copy or to move in.
  let dtS = streams.dtMs / 1000
  let tol = toleranceSamples(streams.dtMs)
  let m = marginOf(align)
  result = SlicePlan(dtMs: streams.dtMs, samples: samples, margin: m,
    rowCount: streams.rowCount, firstSample: streams.firstSample, order: order,
    align: align, window: window, shots: @shots)
  let receivers = streams.receiverKeys.len
  let traces = receivers * shots.len
  result.spanStart = newSeqOfCap[int](traces + 1)
  result.fraction = newSeq[float64](traces)
  result.gapSamples = newSeq[int](traces)
  # A bit per row (and a count per word of bits), not a flag and an index
  # per row: a plan of one shot over a million rows then zeroes 250 KB, not
  # 9 MB.
  result.neededBits = newSeq[uint64]((streams.rowCount + 63) shr 6)
  for trace in 0 ..< traces:
    let (r, s) =
      if order == soReceiverGathers: (trace div shots.len, trace mod shots.len)
      else: (trace mod receivers, trace div receivers)
    result.spanStart.add result.spans.len
    let lo = streams.ivStart[r]
    let hi = streams.ivStart[r + 1]
    let u = (shots[s].time + window.startMs / 1000 - streams.firstStart[r]) / dtS
    var gaps = 0
    template gap(a, b: int, k0: int) =
      result.spans.add SlicePiece(row: -1, first: 0, dst: a - k0, count: b - a)
      gaps += max(0, min(b, k0 + samples) - max(a, k0))
    if abs(u) > 1e15: # no data can reach it (and keeps the ints below safe)
      result.spans.add SlicePiece(row: -1, dst: -m, count: samples + 2 * m)
      gaps = samples
    else:
      # The first interval reaching the window, whose segment sets the grid.
      let near = int(round(u))
      var k = firstEndingAfter(streams.ivs, lo, hi, near - m)
      var resid = 0.0
      if k < hi and streams.ivs[k].a < near + samples + m: resid = streams.ivs[k].resid
      let base = u - resid
      let k0 = int(round(base))
      var frac = base - float64(k0)
      if abs(frac) <= tol: frac = 0
      result.fraction[trace] = frac
      let wlo = k0 - m
      let whi = k0 + samples + m
      k = firstEndingAfter(streams.ivs, lo, hi, wlo)
      var pos = wlo
      while pos < whi:
        if k < hi and streams.ivs[k].a < whi:
          let iv = streams.ivs[k]
          if iv.a > pos:
            gap(pos, iv.a, k0)
            pos = iv.a
          let e = min(iv.a + iv.n, whi)
          result.spans.add SlicePiece(row: iv.row, first: iv.col + (pos - iv.a),
            dst: pos - k0, count: e - pos)
          result.maxColumn = max(result.maxColumn, iv.col + (e - iv.a))
          result.neededBits[iv.row shr 6].setBit(iv.row and 63)
          pos = e
          inc k
        else:
          gap(pos, whi, k0)
          pos = whi
    result.gapSamples[trace] = gaps
  result.spanStart.add result.spans.len
  # The rows needed, ascending, straight from the bits; and per word how
  # many come before it, so a row's place among them takes no search.
  result.neededRank = newSeq[int](result.neededBits.len)
  for w, word in result.neededBits:
    result.neededRank[w] = result.rowsNeeded.len
    var bits = word
    while bits != 0:
      let row = w shl 6 + countTrailingZeroBits(bits)
      result.rowsNeeded.add row
      result.tracesNeeded.add streams.traces[row]
      bits = bits and (bits - 1)

proc planSlices*(streams: SliceStreams, shots: openArray[SliceShot], window: SliceWindow,
    order = soReceiverGathers, align = saNearest): SlicePlan =
  ## The plan of cutting the segments of `streams` into one trace per
  ## receiver and shot: `window` at each of `shots`, in `order`, with
  ## sub-sample starts handled by `align`. See the module documentation. It
  ## costs the traces it lays out (some 50 ns each), whatever the number of
  ## segments, plus a copy of the streams' receiver keys and of their lists
  ## of data problems (`overlaps`, `timingIssues`, `untimed`,
  ## `otherInterval`: empty for clean data).
  ##
  ## Raises `ValueError` when a shot time or the window start is not finite,
  ## or the window length is not a positive whole number of samples. Data
  ## problems (gaps, overlaps, off-grid segments, rows without a time or of
  ## another sample interval) are reported in the plan.
  let samples = windowSamples(window, streams.dtMs)
  checkShots(shots)
  result = layOut(streams, shots, window, samples, order, align)
  result.receiverKeys = streams.receiverKeys
  result.overlaps = streams.overlaps
  result.issues = streams.issues
  result.untimed = streams.untimed
  result.otherInterval = streams.otherInterval

proc planOnce(streams: var SliceStreams, shots: openArray[SliceShot], window: SliceWindow,
    samples: int, order: SliceOrder, align: SliceAlign): SlicePlan =
  # A plan of streams made for it alone: their lists move into the plan.
  result = layOut(streams, shots, window, samples, order, align)
  result.receiverKeys = move streams.receiverKeys
  result.overlaps = move streams.overlaps
  result.issues = move streams.issues
  result.untimed = move streams.untimed
  result.otherInterval = move streams.otherInterval

proc planSlices*(meta: ChunkMeta, times: openArray[float64], receivers: openArray[int],
    shots: openArray[SliceShot], window: SliceWindow, order = soReceiverGathers,
    align = saNearest): SlicePlan =
  ## `planSlices(initSliceStreams(meta, times, receivers), shots, window,
  ## order, align)` in one call, for a plan made once: the streams are built
  ## again on every call (130 to 250 ms at a million rows), so keep them
  ## (`initSliceStreams`) when the shots change and the segments do not.
  ##
  ## Raises `ValueError` as `initSliceStreams` and the streams form do.
  checkPerRow(times.len, meta.len, "times")
  checkPerRow(receivers.len, meta.len, "receivers")
  checkInterval(meta.dtMs)
  let samples = windowSamples(window, meta.dtMs)
  checkShots(shots)
  var streams = streamsOf(meta, times, receivers)
  planOnce(streams, shots, window, samples, order, align)

proc planSlices*(meta: ChunkMeta, times: openArray[float64], shots: openArray[SliceShot],
    window: SliceWindow, order = soReceiverGathers, align = saNearest): SlicePlan =
  ## `planSlices` in one call with caller times and the receivers from the
  ## keys (`initSliceStreams(meta, times)`).
  checkPerRow(times.len, meta.len, "times")
  checkInterval(meta.dtMs)
  let samples = windowSamples(window, meta.dtMs)
  checkShots(shots)
  var streams = streamsOf(meta, times, defaultReceivers(meta))
  planOnce(streams, shots, window, samples, order, align)

proc planSlices*(meta: ChunkMeta, shots: openArray[SliceShot], window: SliceWindow,
    order = soReceiverGathers, align = saNearest): SlicePlan =
  ## `planSlices` in one call with the times and receivers from the keys
  ## (`initSliceStreams(meta)`).
  checkInterval(meta.dtMs)
  let samples = windowSamples(window, meta.dtMs)
  checkShots(shots)
  var streams = streamsOf(meta, keyTimes(meta), defaultReceivers(meta))
  planOnce(streams, shots, window, samples, order, align)

func dtMs*(streams: SliceStreams): float64 {.inline, raises: [].} =
  ## The sample interval, ms.
  streams.dtMs

func rowCount*(streams: SliceStreams): int {.inline, raises: [].} =
  ## The rows of the meta the streams were made from.
  streams.rowCount

func receiverCount*(streams: SliceStreams): int {.inline, raises: [].} =
  ## The number of receivers (with at least one used segment).
  streams.receiverKeys.len

func receiverKeys*(streams: SliceStreams, receiver: int): lent TraceKeys {.inline,
    raises: [].} =
  ## The keys of receiver `receiver`'s earliest segment.
  streams.receiverKeys[receiver]

func overlaps*(streams: SliceStreams): lent seq[SliceOverlap] {.inline, raises: [].} =
  ## Every overlap between segments of one receiver.
  streams.overlaps

func timingIssues*(streams: SliceStreams): lent seq[SliceTimingIssue] {.inline,
    raises: [].} =
  ## The segments off the previous segment's sample grid.
  streams.issues

func untimed*(streams: SliceStreams): lent seq[int] {.inline, raises: [].} =
  ## The live rows not used because their time (or `t0Ms`) is not finite.
  streams.untimed

func otherInterval*(streams: SliceStreams): lent seq[int] {.inline, raises: [].} =
  ## The live rows not used because their keys give another sample interval
  ## than the meta's `dtMs`, ascending.
  streams.otherInterval

func len*(plan: SlicePlan): int {.inline, raises: [].} =
  ## The number of output traces: receivers × shots.
  plan.fraction.len

func samples*(plan: SlicePlan): int {.inline, raises: [].} =
  ## The samples of each output trace (`window.lengthMs / dtMs`).
  plan.samples

func dtMs*(plan: SlicePlan): float64 {.inline, raises: [].} =
  ## The sample interval, ms.
  plan.dtMs

func margin*(plan: SlicePlan): int {.inline, raises: [].} =
  ## The extra samples read on each side for the interpolation: 0, 1 or 2.
  plan.margin

func order*(plan: SlicePlan): SliceOrder {.inline, raises: [].} =
  ## The order of the output traces.
  plan.order

func align*(plan: SlicePlan): SliceAlign {.inline, raises: [].} =
  ## How window starts are aligned.
  plan.align

func window*(plan: SlicePlan): SliceWindow {.inline, raises: [].} =
  ## The window.
  plan.window

func shotCount*(plan: SlicePlan): int {.inline, raises: [].} =
  ## The number of shots.
  plan.shots.len

func receiverCount*(plan: SlicePlan): int {.inline, raises: [].} =
  ## The number of receivers (with at least one used segment).
  plan.receiverKeys.len

func shotOf*(plan: SlicePlan, trace: int): int {.inline, raises: [].} =
  ## The shot (index into the shots given) of output trace `trace`.
  if plan.order == soReceiverGathers: trace mod plan.shots.len
  else: trace div plan.receiverKeys.len

func receiverOf*(plan: SlicePlan, trace: int): int {.inline, raises: [].} =
  ## The receiver (0 ..< receiverCount) of output trace `trace`.
  if plan.order == soReceiverGathers: trace div plan.shots.len
  else: trace mod plan.receiverKeys.len

func receiverKeys*(plan: SlicePlan, receiver: int): lent TraceKeys {.inline,
    raises: [].} =
  ## The keys of receiver `receiver`'s earliest segment.
  plan.receiverKeys[receiver]

func fraction*(plan: SlicePlan, trace: int): float64 {.inline, raises: [].} =
  ## How far the window start lies after the output's column 0, in samples,
  ## between −0.5 and 0.5: kept with `saNearest`, shifted out otherwise.
  plan.fraction[trace]

func gapSamples*(plan: SlicePlan, trace: int): int {.inline, raises: [].} =
  ## The output samples of `trace` that no segment covers (known at plan
  ## time; `report.gapSamples` adds rows that fail to read).
  plan.gapSamples[trace]

iterator pieces*(plan: SlicePlan, trace: int): SlicePiece =
  ## The pieces of output trace `trace`, in column order, gaps included
  ## (`row = -1`): which columns of which rows a reader needs.
  for i in plan.spanStart[trace] ..< plan.spanStart[trace + 1]:
    yield plan.spans[i]

func rowsNeeded*(plan: SlicePlan): lent seq[int] {.inline, raises: [].} =
  ## The rows any output trace reads, ascending: rows of the planning meta
  ## (as `SlicePiece.row`), for taking them out of a block already in memory
  ## or out of the caller's own per-row arrays. They are **not** trace
  ## numbers unless the planning view starts at trace 0 of its file: read
  ## from a file with `tracesNeeded`.
  plan.rowsNeeded

func tracesNeeded*(plan: SlicePlan): lent seq[int] {.inline, raises: [].} =
  ## The traces any output trace reads: the trace of each of `rowsNeeded`
  ## (`TraceRows.traceOf`), in that order. Read only these, as a block of
  ## their own (`readTracesInto(f, plan.tracesNeeded, data)` with
  ## `chunkMeta(f, plan.tracesNeeded)`, see `extractInto`), or every row of
  ## the planning meta.
  plan.tracesNeeded

func maxColumn*(plan: SlicePlan): int {.inline, raises: [].} =
  ## One past the highest column read from any row: the narrowest block
  ## `extractInto` accepts.
  plan.maxColumn

func overlaps*(plan: SlicePlan): lent seq[SliceOverlap] {.inline, raises: [].} =
  ## Every overlap between segments of one receiver.
  plan.overlaps

func timingIssues*(plan: SlicePlan): lent seq[SliceTimingIssue] {.inline,
    raises: [].} =
  ## The segments off the previous segment's sample grid (non-whole gaps and
  ## overlaps).
  plan.issues

func untimed*(plan: SlicePlan): lent seq[int] {.inline, raises: [].} =
  ## The live rows not used because their time (or `t0Ms`) is not finite.
  plan.untimed

func otherInterval*(plan: SlicePlan): lent seq[int] {.inline, raises: [].} =
  ## The live rows not used because their keys give another sample interval
  ## (`intervalUs`) than the planning meta's `dtMs`, ascending.
  plan.otherInterval

func sliceKeys*(plan: SlicePlan, trace: int): TraceKeys =
  ## The keys of output trace `trace` (decision D14), for parseSEG's writer:
  ## the receiver's keys; `record`, source line, point, point index and
  ## coordinates from the shot, whatever the segments' headers hold; offset
  ## and midpoint from the coordinates (NaN without them);
  ## `time` the shot time and `t0Ms` the window's `startMs`; `intervalUs`
  ## as the receiver's keys hold it (from `dtMs` when they hold none);
  ## ensemble, inline and crossline NaN; `status` dead when no
  ## segment reaches the trace, else live (`traceCode` 0: the writer derives
  ## it from `status`).
  let shot = plan.shots[plan.shotOf(trace)]
  result = plan.receiverKeys[plan.receiverOf(trace)]
  result.record = shot.id
  result.srcLine = shot.srcLine
  result.srcPoint = shot.srcPoint
  result.srcIndex = shot.srcIndex
  result.srcX = shot.srcX
  result.srcY = shot.srcY
  result.srcZ = shot.srcZ
  result.ensemble = NaN
  result.offset = NaN
  result.cmpX = NaN
  result.cmpY = NaN
  result.inline = NaN
  result.crossline = NaN
  result.time = shot.time
  result.t0Ms = plan.window.startMs
  # The receiver's own interval when its keys hold one: `dtMs * 1000` can
  # differ from it in the last bit (1e6 / 31 µs does), and parseSEG compares
  # intervals exactly.
  if not (result.intervalUs > 0 and result.intervalUs < Inf):
    result.intervalUs = plan.dtMs * 1000
  result.traceCode = 0
  result.status = if plan.gapSamples[trace] >= plan.samples: tsDead else: tsLive
  result.deriveGeometry(overwrite = true)

func neededIndex(plan: SlicePlan, row: int): int {.inline, raises: [].} =
  # The place of `row` in `rowsNeeded`, or −1: the needed rows before its
  # word of bits, plus those below it in the word.
  let word = plan.neededBits[row shr 6]
  let bit = 1'u64 shl (row and 63)
  if (word and bit) == 0: -1
  else: plan.neededRank[row shr 6] + countSetBits(word and (bit - 1))

func sliceWorkspace*(T: typedesc[SampleType], plan: SlicePlan): WorkspaceSize =
  ## The scratch `extractInto` needs: none with `saNearest`, one padded row
  ## otherwise.
  let n = if plan.align == saNearest: 0 else: plan.samples + 2 * plan.margin
  when T is float32: WorkspaceSize(f32: n)
  else: WorkspaceSize(f64: n)

proc initSliceReport*(plan: SlicePlan): SliceReport =
  ## A report sized for `plan`, so `extractInto` fills it without allocating.
  result = SliceReport(capacity: plan.spans.len)
  result.gaps = newSeqOfCap[SliceGap](plan.spans.len)
  result.gapStart = newSeq[int](plan.len + 1)
  result.gapSamples = newSeq[int](plan.len)
  result.checks = newSeq[SliceOverlapCheck](plan.overlaps.len)

func len*(report: SliceReport): int {.inline, raises: [].} =
  ## The number of traces reported on.
  report.gapSamples.len

func gapSamples*(report: SliceReport, trace: int): int {.inline, raises: [].} =
  ## The zero-filled output samples of `trace`.
  report.gapSamples[trace]

func gapCount*(report: SliceReport, trace: int): int {.inline, raises: [].} =
  ## The number of zero-filled column runs of `trace`.
  report.gapStart[trace + 1] - report.gapStart[trace]

iterator gaps*(report: SliceReport, trace: int): SliceGap =
  ## The zero-filled column runs of `trace`, ascending.
  for i in report.gapStart[trace] ..< report.gapStart[trace + 1]:
    yield report.gaps[i]

func overlapCheck*(report: SliceReport, i: int): SliceOverlapCheck {.inline,
    raises: [].} =
  ## The comparison of `plan.overlaps[i]`.
  report.checks[i]

func mismatches*(report: SliceReport): int {.raises: [].} =
  ## The number of overlaps whose samples differ.
  for c in report.checks:
    if c.status == osMismatch: inc result

proc extractInto*[T: SampleType](plan: SlicePlan, data: RowBlock[T], meta: ChunkMeta,
    dst: RowBlock[T], report: var SliceReport, ws: var Workspace) =
  ## Cuts the traces of `plan` from the segments `data` into `dst`
  ## (`plan.len` rows of `plan.samples`), zero-filling and reporting gaps, and
  ## compares overlaps (see the module documentation). `data` and `meta` hold
  ## either every row of the planning meta, or only the rows
  ## `plan.rowsNeeded`, in that order: read from a file, those are the traces
  ## `plan.tracesNeeded` (`chunkMeta(f, plan.tracesNeeded)`). A row not live
  ## in `meta` (failed) is a gap. `ws` must hold `sliceWorkspace(T, plan)`
  ## (an `AssertionDefect` otherwise). Allocates nothing.
  ##
  ## Raises `ValueError` when `meta` and `data` disagree, `meta` has neither
  ## row count, another `dtMs` or `firstSample` than the planning meta,
  ## `data` is narrower than `plan.maxColumn`, `dst` is not `plan.len ×
  ## plan.samples`, or `report` was not sized for the plan.
  if data.len != meta.len:
    badSlice("the block has " & $data.len & " rows, the meta " & $meta.len)
  let compact = meta.len != plan.rowCount
  if compact and meta.len != plan.rowsNeeded.len:
    badSlice("the meta has " & $meta.len & " rows: the plan needs every row (" &
      $plan.rowCount & ") or rowsNeeded (" & $plan.rowsNeeded.len & ")")
  if abs(meta.dtMs - plan.dtMs) > 1e-9 * plan.dtMs:
    badSlice("the meta's dtMs is " & $meta.dtMs & ", the plan's " & $plan.dtMs)
  if meta.firstSample != plan.firstSample:
    badSlice("the meta's firstSample is " & $meta.firstSample & ", the plan's " &
      $plan.firstSample)
  if plan.rowsNeeded.len > 0 and data.cols < plan.maxColumn:
    badSlice("the block is " & $data.cols & " samples wide, the plan reads up to " &
      $plan.maxColumn)
  if dst.len != plan.len or dst.cols != plan.samples:
    badSlice("dst must be " & $plan.len & " × " & $plan.samples & ", got " & $dst.len &
      " × " & $dst.cols)
  if report.gapSamples.len != plan.len or report.checks.len != plan.overlaps.len or
      report.capacity < plan.spans.len:
    badSlice("the report was not sized for this plan (initSliceReport)")
  let n = plan.samples
  let m = plan.margin
  let width = n + 2 * m
  var buf: ptr UncheckedArray[T] = nil
  if plan.align != saNearest:
    doAssert ws.fits(sliceWorkspace(T, plan)), "reserve sliceWorkspace(T, plan) first"
    buf = ws.scratch(T, width)
  let interp = if plan.align == saCubic: siCubic else: siLinear

  template dataRow(row: int): int =
    (if compact: plan.neededIndex(row) else: row)

  report.gaps.setLen(0)
  for trace in 0 ..< plan.len:
    report.gapStart[trace] = report.gaps.len
    var gapTotal = 0
    let outRow = dst.rowPtr(trace)
    # Into the padded scratch row, or straight into dst.
    let target = if buf != nil: buf else: outRow
    let shift = if buf != nil: m else: 0
    for i in plan.spanStart[trace] ..< plan.spanStart[trace + 1]:
      let pc = plan.spans[i]
      let r = if pc.row >= 0: dataRow(pc.row) else: -1
      let a = pc.dst + shift
      # Columns `a ..< a + count` of `target`, as whole runs. A loop over the
      # samples keeps its overflow checks in a `-d:release` build, which
      # keeps the compiler from vectorizing it: 3.8 ms against 2.1 ms for
      # 2000 traces of 3000 float32 (`-d:danger` takes 2.1 ms either way).
      if r >= 0 and meta.isLive(r):
        if pc.count > 0:
          copyMem(addr target[a], addr data.rowPtr(r)[pc.first], pc.count * sizeof(T))
      else:
        if pc.count > 0: zeroMem(addr target[a], pc.count * sizeof(T))
        # The gap's columns inside the window, merged with the previous run.
        let g0 = max(pc.dst, 0)
        let g1 = min(pc.dst + pc.count, n)
        if g1 > g0:
          gapTotal += g1 - g0
          let last = report.gaps.len - 1
          if last >= report.gapStart[trace] and
              report.gaps[last].first + report.gaps[last].count == g0:
            report.gaps[last].count += g1 - g0
          else:
            report.gaps.add SliceGap(first: g0, count: g1 - g0)
    if buf != nil:
      if plan.fraction[trace] != 0:
        shiftSamples(toOpenArray(buf, 0, width - 1), plan.fraction[trace], interp)
      copyMem(outRow, addr buf[m], n * sizeof(T))
    report.gapSamples[trace] = gapTotal
  report.gapStart[plan.len] = report.gaps.len

  for i, o in plan.overlaps:
    var c = SliceOverlapCheck(status: osNotRead)
    if plan.neededIndex(o.earlier) >= 0 and plan.neededIndex(o.later) >= 0:
      let re = dataRow(o.earlier)
      let rl = dataRow(o.later)
      # The earlier row's trimmed tail is in no piece: the block may be
      # narrower than it.
      if meta.isLive(re) and meta.isLive(rl) and o.first + o.samples <= data.cols:
        let a = data.rowPtr(re)
        let b = data.rowPtr(rl)
        c.status = osMatch
        for j in 0 ..< o.samples:
          let x = a[o.first + j]
          let y = b[j]
          if x != y and not (x.isNaN and y.isNaN):
            inc c.differing
            c.maxAbsDiff = max(c.maxAbsDiff, abs(float64(x) - float64(y)))
        if c.differing > 0: c.status = osMismatch
    report.checks[i] = c

proc extractInto*[T: SampleType](plan: SlicePlan, data: Tensor[T], meta: ChunkMeta,
    dst: var Tensor[T], report: var SliceReport, ws: var Workspace) =
  ## `extractInto` on `[rows, cols]` tensors. Raises `ValueError` unless both
  ## are C-contiguous and `data` has `meta.len` rows, or as the block form.
  meta.checkShape(data)
  extractInto(plan, rowBlock(data), meta, rowBlock(dst), report, ws)

proc checkDrift(t0, offset0, t1, offset1: float64) =
  for v in [t0, offset0, t1, offset1]:
    if v.classify notin {fcNormal, fcZero, fcNegZero, fcSubnormal}:
      badSlice("drift parameters must be finite, got " & $v)
  if t0 == t1:
    badSlice("the two clock checks must be at different times, both at " & $t0)

proc driftCorrectTimes*(times: var openArray[float64], t0, offset0, t1,
    offset1: float64) =
  ## Corrects one receiver's segment times for a linear clock drift, in
  ## place, before `planSlices`: its clock was `offset0` seconds ahead of the
  ## true time at `t0` and `offset1` ahead at `t1` (clock times, UTC POSIX
  ## seconds), and each time becomes `t − offset(t)`, with `offset`
  ## interpolated (and extrapolated) linearly. NaN times stay NaN. Raises
  ## `ValueError` for a non-finite parameter or `t0 == t1`.
  checkDrift(t0, offset0, t1, offset1)
  let rate = (offset1 - offset0) / (t1 - t0)
  for t in times.mitems:
    t = t - (offset0 + (t - t0) * rate)

proc driftCorrectedTimes*(times: openArray[float64], t0, offset0, t1,
    offset1: float64): seq[float64] =
  ## A corrected copy of `times`; see `driftCorrectTimes`.
  result = @times
  driftCorrectTimes(result, t0, offset0, t1, offset1)
