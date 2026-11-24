## continuous/slice: cutting segmented continuous recordings into records.
## A synthetic stream with known samples is split into segments (rows), and
## every window is checked against a direct slice of the unsegmented stream.

import std/[algorithm, assertions, math, os, random, strutils]
import parseSEG, processSEG
import utils/allocs

proc main*() =
  const
    T0 = 1.7e9 + 0.123 # the stream's sample 0, UTC POSIX seconds
    Dt = 2.0           # ms

  type
    Segments[T] = object
      keys: seq[TraceKeys]
      counts: seq[int]
      data: Tensor[T]

  func streamValue(k: int): float64 =
    ## Sample `k` of the synthetic stream: distinct, smooth enough to interpolate.
    sin(0.05 * float64(k)) + 0.001 * float64(k)

  func timeOf(k: float64): float64 = T0 + k * Dt / 1000

  proc segmentsOf[T](spans: openArray[(int, int)], line = 1.0, point = 1.0,
      value: proc (k: int): float64 = streamValue): Segments[T] =
    ## One row per `(first, count)` span of the stream, in the given order.
    var width = 0
    for (_, n) in spans: width = max(width, n)
    result.data = zeros[T](spans.len, width)
    for i, (first, n) in spans:
      var k = initTraceKeys()
      k.status = tsLive
      k.time = timeOf(float64(first))
      k.t0Ms = 0
      k.intervalUs = Dt * 1000
      k.rcvLine = line
      k.rcvPoint = point
      k.rcvX = 10 * point
      k.rcvY = 0
      result.keys.add k
      result.counts.add n
      for j in 0 ..< n: result.data[i, j] = T(value(first + j))

  proc join[T](a, b: Segments[T]): Segments[T] =
    ## The rows of `a` then `b`.
    result.keys = a.keys & b.keys
    result.counts = a.counts & b.counts
    let width = max(a.data.shape[1], b.data.shape[1])
    result.data = zeros[T](a.keys.len + b.keys.len, width)
    for i in 0 ..< a.keys.len:
      for j in 0 ..< a.data.shape[1]: result.data[i, j] = a.data[i, j]
    for i in 0 ..< b.keys.len:
      for j in 0 ..< b.data.shape[1]: result.data[a.keys.len + i, j] = b.data[i, j]

  proc cut[T](s: Segments[T], meta: ChunkMeta, plan: SlicePlan,
      report: var SliceReport): Tensor[T] =
    result = zeros[T](plan.len, plan.samples)
    report = initSliceReport(plan)
    var ws = initWorkspace()
    ws.reserve(sliceWorkspace(T, plan))
    extractInto(plan, s.data, result, meta, report, ws)

  proc shotAt(k: float64, id = 1.0): SliceShot = initSliceShot(timeOf(k), id)

  proc window(samples: int, startMs = 0.0): SliceWindow =
    SliceWindow(startMs: startMs, lengthMs: float64(samples) * Dt)

  proc checkDirect[T](dst: Tensor[T], trace, first: int, msg: string) =
    ## Row `trace` of `dst` is the stream from sample `first`.
    for j in 0 ..< dst.shape[1]:
      doAssert dst[trace, j] == T(streamValue(first + j)), msg & ": column " & $j

  proc suite[T]() =
    block windows_equal_a_direct_slice:
      # Boundaries at 100, 140, 170 and 220.
      let s = segmentsOf[T]([(0, 100), (100, 40), (140, 30), (170, 50), (220, 200)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      # Windows of 80 samples inside one row and straddling 1, 2 and 3 boundaries.
      let starts = [10, 60, 135, 95, 300]
      var shots: seq[SliceShot]
      for k in starts: shots.add shotAt(float64(k))
      let plan = planSlices(meta, shots, window(80))
      doAssert plan.len == starts.len and plan.receiverCount == 1 and plan.samples == 80
      doAssert plan.overlaps.len == 0 and plan.timingIssues.len == 0
      var report: SliceReport
      let dst = cut(s, meta, plan, report)
      for t, k in starts:
        checkDirect(dst, t, k, "window at " & $k)
        doAssert report.gapSamples(t) == 0 and plan.fraction(t) == 0
      var rows: array[5, int]
      for t in 0 ..< 5:
        var used = 0
        for p in plan.pieces(t):
          doAssert p.row >= 0
          inc used
        rows[t] = used
      doAssert rows == [1, 2, 3, 4, 1], "straddles 0, 1, 2, 3 boundaries: " & $rows
      doAssert plan.rowsNeeded == @[0, 1, 2, 3, 4]

    block negative_start_and_window_offset:
      let s = segmentsOf[T]([(0, 100), (100, 100)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      # The shot at sample 110, the window from 20 ms (10 samples) before it.
      let plan = planSlices(meta, [shotAt(110)], window(50, startMs = -20))
      var report: SliceReport
      let dst = cut(s, meta, plan, report)
      checkDirect(dst, 0, 100, "a negative start")
      let k = plan.sliceKeys(0)
      doAssert k.time == timeOf(110) and k.t0Ms == -20, "time = shot, t0Ms = start"

    block gap_is_zero_filled_and_reported:
      # Samples 100 ..< 110 were never recorded.
      let s = segmentsOf[T]([(0, 100), (110, 100)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(90), shotAt(0)], window(40))
      doAssert plan.gapSamples(0) == 10 and plan.gapSamples(1) == 0
      var report: SliceReport
      let dst = cut(s, meta, plan, report)
      doAssert report.gapSamples(0) == 10 and report.gapCount(0) == 1
      for g in report.gaps(0):
        doAssert g.first == 10 and g.count == 10
      for j in 0 ..< 40:
        let want = if j in 10 ..< 20: T(0) else: T(streamValue(90 + j))
        doAssert dst[0, j] == want, "column " & $j
      doAssert report.gapCount(1) == 0
      checkDirect(dst, 1, 0, "no gap")

    block window_before_and_after_the_data:
      let s = segmentsOf[T]([(0, 100)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(-30), shotAt(90), shotAt(1000)], window(40))
      var report: SliceReport
      let dst = cut(s, meta, plan, report)
      doAssert report.gapSamples(0) == 30 and report.gapSamples(1) == 30
      doAssert report.gapSamples(2) == 40
      doAssert plan.sliceKeys(2).status == tsDead and plan.sliceKeys(0).status == tsLive
      for j in 0 ..< 10: doAssert dst[0, 30 + j] == T(streamValue(j))
      for j in 0 ..< 10: doAssert dst[1, j] == T(streamValue(90 + j))

    block sub_sample_shot_time:
      # A shot 0.4 sample after a stream sample: nearest keeps the stream's
      # sample, the cubic shift recovers the stream at the shot time.
      let s = segmentsOf[T]([(0, 120), (120, 120)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let shots = [shotAt(100.4)]
      let near = planSlices(meta, shots, window(60))
      doAssert abs(near.fraction(0) - 0.4) < 1e-3
      var report: SliceReport
      checkDirect(cut(s, meta, near, report), 0, 100, "nearest")
      for align in [saLinear, saCubic]:
        let plan = planSlices(meta, shots, window(60), align = align)
        doAssert plan.margin == (if align == saCubic: 2 else: 1)
        let dst = cut(s, meta, plan, report)
        let tolerance = if align == saCubic: 1e-4 else: 1e-3
        for j in 0 ..< 60:
          doAssert abs(float64(dst[0, j]) - streamValue(100 + j) - 0.4 * 0.001 -
            (sin(0.05 * (100.4 + float64(j))) - sin(0.05 * float64(100 + j)))) < tolerance,
            $align & " column " & $j
      # A shot at 0.6 sample rounds to the next sample, -0.4 left over.
      let up = planSlices(meta, [shotAt(100.6)], window(60))
      doAssert abs(up.fraction(0) + 0.4) < 1e-3
      checkDirect(cut(s, meta, up, report), 0, 101, "rounded up")

    block sinc_alignment:
      func band(k: float64): float64 =
        sin(2 * PI * 0.05 * k) + 0.6 * sin(2 * PI * 0.13 * k + 0.7) +
          0.3 * sin(2 * PI * 0.22 * k + 1.3)
      func tone(k: float64): float64 = sin(2 * PI * 0.35 * k)
      let fracs = [0.1, 0.25, 0.5, 0.9]
      var shots: seq[SliceShot]
      for f in fracs: shots.add shotAt(200.0 + f)
      let segs = segmentsOf[T]([(0, 220), (220, 280)], point = 1,
        value = proc (k: int): float64 = band(float64(k))).join(
        segmentsOf[T]([(0, 500)], point = 2,
          value = proc (k: int): float64 = band(float64(k))))
      let meta = initChunkMeta(rowsOf(segs.keys, segs.counts), dtMs = Dt)
      let win = window(64)
      for order in [soReceiverGathers, soShotRecords]:
        let plan = planSlices(meta, shots, win, order, saSinc)
        doAssert plan.margin == SincHalf
        var report: SliceReport
        let dst = cut(segs, meta, plan, report)
        for si in 0 .. fracs.high:
          let f = fracs[si]
          let shot = 200.0 + f
          let covered = if order == soReceiverGathers: plan.shotCount + si
            else: si * plan.receiverCount + 1
          var err = 0.0
          var peak = 0.0
          for j in 0 ..< 64:
            let want = band(shot + float64(j))
            err = max(err, abs(float64(dst[covered, j]) - want))
            peak = max(peak, abs(want))
          doAssert err <= 1e-4 * peak, $order & " frac " & $f & " err " & $err
          let span = if order == soReceiverGathers: si else: si * plan.receiverCount
          if f == 0.25:
            var nData = 0
            for pc in plan.pieces(span):
              if pc.row >= 0: inc nData
            doAssert nData == 2, "a shot spanning two segments"
            err = 0.0
            peak = 0.0
            for j in 0 ..< 64:
              let want = band(shot + float64(j))
              err = max(err, abs(float64(dst[span, j]) - want))
              peak = max(peak, abs(want))
            doAssert err <= 1e-4 * peak, "span " & $order & " err " & $err
      let toneSeg = segmentsOf[T]([(0, 500)],
        value = proc (k: int): float64 = tone(float64(k)))
      let toneMeta = initChunkMeta(rowsOf(toneSeg.keys, toneSeg.counts), dtMs = Dt)
      proc toneErr(align: SliceAlign, shot: float64): float64 =
        let plan = planSlices(toneMeta, [shotAt(shot)], win, align = align)
        var report: SliceReport
        let dst = cut(toneSeg, toneMeta, plan, report)
        for j in SincHalf ..< 64 - SincHalf:
          result = max(result, abs(float64(dst[0, j]) - tone(shot + float64(j))))
      for f in fracs:
        let shot = 200.0 + f
        let eSinc = toneErr(saSinc, shot)
        let eCub = toneErr(saCubic, shot)
        let eLin = toneErr(saLinear, shot)
        doAssert eSinc < eCub and eSinc < eLin,
          "0.7 Nyquist frac " & $f & " sinc " & $eSinc & " cubic " & $eCub &
          " linear " & $eLin
        if f == 0.5:
          let eNear = toneErr(saNearest, shot)
          doAssert eSinc < eCub and eCub < eLin and eLin < eNear,
            "order sinc " & $eSinc & " cubic " & $eCub & " linear " & $eLin &
            " nearest " & $eNear
      let near = planSlices(toneMeta, [shotAt(200.004)], win)
      let sinc = planSlices(toneMeta, [shotAt(200.004)], win, align = saSinc)
      doAssert near.fraction(0) == 0 and sinc.fraction(0) == 0
      var same: SliceReport
      doAssert cut(toneSeg, toneMeta, near, same) == cut(toneSeg, toneMeta, sinc, same)

    block overlap_later_segment_wins:
      # Row 1 starts 3 samples before row 0 ends.
      var s = segmentsOf[T]([(0, 100), (97, 100)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(90)], window(20))
      doAssert plan.overlaps.len == 1
      let o = plan.overlaps[0]
      doAssert o.earlier == 0 and o.later == 1 and o.samples == 3 and o.first == 97
      doAssert plan.timingIssues.len == 0
      var pieces: seq[SlicePiece]
      for p in plan.pieces(0): pieces.add p
      doAssert pieces == @[SlicePiece(row: 0, first: 90, dst: 0, count: 7),
        SlicePiece(row: 1, first: 0, dst: 7, count: 13)], "the earlier row is trimmed"
      var report: SliceReport
      checkDirect(cut(s, meta, plan, report), 0, 90, "matching overlap")
      doAssert report.overlapCheck(0).status == osMatch and report.mismatches == 0
      # The later row's overlapping samples differ: they win, and it is reported.
      for j in 0 ..< 3: s.data[1, j] = T(10 + j)
      let dst = cut(s, meta, plan, report)
      for j in 0 ..< 7: doAssert dst[0, j] == T(streamValue(90 + j))
      for j in 0 ..< 3: doAssert dst[0, 7 + j] == T(10 + j), "the later row wins"
      let c = report.overlapCheck(0)
      doAssert c.status == osMismatch and c.differing == 3 and report.mismatches == 1
      doAssert c.maxAbsDiff > 9
      # A block only as wide as the plan reads cannot hold row 0's trimmed tail.
      doAssert plan.maxColumn == 97
      var narrow = zeros[T](2, 97)
      for i in 0 ..< 2:
        for j in 0 ..< 97: narrow[i, j] = s.data[i, j]
      var ws = initWorkspace()
      var out2 = zeros[T](1, 20)
      extractInto(plan, narrow, out2, meta, report, ws)
      doAssert report.overlapCheck(0).status == osNotRead and out2 == dst

    block overlap_not_read:
      # A window after the overlap does not read row 0: nothing to compare.
      let s = segmentsOf[T]([(0, 100), (97, 100)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(120)], window(20))
      doAssert plan.rowsNeeded == @[1]
      var report: SliceReport
      discard cut(s, meta, plan, report)
      doAssert report.overlapCheck(0).status == osNotRead

    block contained_segment:
      # Row 1 lies inside row 0: row 0 resumes after it.
      let s = segmentsOf[T]([(0, 100), (40, 10)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(30)], window(40))
      var rowsUsed: seq[int]
      for p in plan.pieces(0): rowsUsed.add p.row
      doAssert rowsUsed == @[0, 1, 0]
      doAssert plan.overlaps.len == 1 and plan.overlaps[0].samples == 10
      var report: SliceReport
      checkDirect(cut(s, meta, plan, report), 0, 30, "contained")

    block non_whole_overlap_is_a_timing_issue:
      var s = segmentsOf[T]([(0, 100), (97, 100)])
      s.keys[1].time += 0.3 * Dt / 1000
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(90)], window(20))
      doAssert plan.overlaps.len == 1 and plan.timingIssues.len == 1
      let issue = plan.timingIssues[0]
      doAssert issue.earlier == 0 and issue.later == 1 and abs(issue.residual - 0.3) < 1e-3

    block independent_of_row_order:
      let spans = [(0, 100), (97, 53), (150, 60), (210, 100)]
      let a = segmentsOf[T](spans, point = 1).join(segmentsOf[T](spans, point = 2))
      var shuffled = [(97, 53), (210, 100), (0, 100), (150, 60)]
      let b = segmentsOf[T](shuffled, point = 2).join(segmentsOf[T](
        [(150, 60), (0, 100), (210, 100), (97, 53)], point = 1))
      let metaA = initChunkMeta(rowsOf(a.keys, a.counts), dtMs = Dt)
      let metaB = initChunkMeta(rowsOf(b.keys, b.counts), dtMs = Dt)
      let shots = [shotAt(20), shotAt(90), shotAt(140)]
      for order in [soReceiverGathers, soShotRecords]:
        let pa = planSlices(metaA, shots, window(100), order)
        let pb = planSlices(metaB, shots, window(100), order)
        doAssert pa.len == 6 and pb.len == 6 and pa.receiverCount == 2
        var ra, rb: SliceReport
        let da = cut(a, metaA, pa, ra)
        let db = cut(b, metaB, pb, rb)
        doAssert da == db, $order
        for t in 0 ..< pa.len:
          # NaN fields: no == on whole keys.
          let (ka, kb) = (pa.sliceKeys(t), pb.sliceKeys(t))
          doAssert ka.rcvPoint == kb.rcvPoint and ka.time == kb.time and ka.rcvX == kb.rcvX
          checkDirect(da, t, [20, 90, 140][pa.shotOf(t)], $order & " trace " & $t)
          doAssert pa.sliceKeys(t).rcvPoint == float64(pa.receiverOf(t) + 1)
        doAssert pa.overlaps.len == 2 and pb.overlaps.len == 2

    block receivers_are_kept_apart:
      # Two receivers record the same span with different values: a window on
      # one never reads the other.
      let a = segmentsOf[T]([(0, 50)], point = 5)
      let b = segmentsOf[T]([(50, 50)], point = 3,
        value = proc (k: int): float64 = -streamValue(k))
      let s = a.join(b)
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(30)], window(40), soShotRecords)
      doAssert plan.receiverCount == 2
      var report: SliceReport
      let dst = cut(s, meta, plan, report)
      # Receiver 0 is point 3 (ascending keys): data from 50 on only.
      doAssert plan.receiverKeys(0).rcvPoint == 3 and report.gapSamples(0) == 20
      doAssert report.gapSamples(1) == 20
      for j in 0 ..< 20:
        doAssert dst[0, j] == 0 and dst[0, 20 + j] == T(-streamValue(50 + j))
        doAssert dst[1, j] == T(streamValue(30 + j)) and dst[1, 20 + j] == 0
      # Caller receiver ids override the keys: one stream after all.
      var times = newSeq[float64](2)
      for i in 0 ..< 2: times[i] = s.keys[i].time
      let one = planSlices(meta, times, [7, 7], [shotAt(30)], window(40))
      doAssert one.receiverCount == 1
      doAssert cut(s, meta, one, report)[0, 25] == T(-streamValue(55))

    block caller_times:
      # Whole-second header times; the caller knows the real ones.
      var s = segmentsOf[T]([(0, 100), (100, 100)])
      var times = newSeq[float64](2)
      for i in 0 ..< 2:
        times[i] = s.keys[i].time
        s.keys[i].time = floor(s.keys[i].time)
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, times, [shotAt(80)], window(40))
      var report: SliceReport
      checkDirect(cut(s, meta, plan, report), 0, 80, "caller times")

    block compact_block_of_needed_rows:
      let s = segmentsOf[T]([(0, 100), (100, 100), (200, 100), (300, 100)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(150), shotAt(320)], window(40))
      doAssert plan.rowsNeeded == @[1, 3]
      # Only rows 1 and 3, read as a block of their own.
      let need = plan.rowsNeeded
      var keys = @[s.keys[need[0]], s.keys[need[1]]]
      let counts = @[s.counts[need[0]], s.counts[need[1]]]
      var data = zeros[T](2, s.data.shape[1])
      for i, r in need:
        for j in 0 ..< data.shape[1]: data[i, j] = s.data[r, j]
      let sub = initChunkMeta(rowsOf(keys, counts), dtMs = Dt)
      var dst = zeros[T](plan.len, plan.samples)
      var report = initSliceReport(plan)
      var ws = initWorkspace()
      extractInto(plan, data, dst, sub, report, ws)
      checkDirect(dst, 0, 150, "compact")
      checkDirect(dst, 1, 320, "compact")

    block failed_row_is_not_a_gap:
      let s = segmentsOf[T]([(0, 100), (100, 100)])
      let rows = rowsOf(s.keys, s.counts)
      let plan = planSlices(initChunkMeta(rows, dtMs = Dt), [shotAt(90)], window(20))
      let meta = initChunkMeta(rows, dtMs = Dt, failed = @[1])
      var dst = zeros[T](plan.len, plan.samples)
      var report = initSliceReport(plan)
      var ws = initWorkspace()
      extractInto(plan, s.data, dst, meta, report, ws)
      doAssert report.gapSamples(0) == 0 and report.gapCount(0) == 0
      doAssert report.failedSamples(0) == 10 and report.failedCount(0) == 1
      for g in report.failedPieces(0): doAssert g.first == 10 and g.count == 10
      for j in 10 ..< 20: doAssert dst[0, j] == 0
      doAssert plan.sliceKeys(0).status == tsLive
      # A hole beside a failed row stays two records.
      let one = segmentsOf[T]([(0, 100)])
      let oneRows = rowsOf(one.keys, one.counts)
      let hole = planSlices(initChunkMeta(oneRows, dtMs = Dt), [shotAt(90)], window(20))
      let holeMeta = initChunkMeta(oneRows, dtMs = Dt, failed = @[0])
      var holeDst = zeros[T](hole.len, hole.samples)
      var holeReport = initSliceReport(hole)
      extractInto(hole, one.data, holeDst, holeMeta, holeReport, ws)
      doAssert holeReport.gapSamples(0) == 10 and holeReport.gapCount(0) == 1
      doAssert holeReport.failedSamples(0) == 10 and holeReport.failedCount(0) == 1
      for g in holeReport.gaps(0): doAssert g.first == 10 and g.count == 10
      for g in holeReport.failedPieces(0): doAssert g.first == 0 and g.count == 10

    block dead_and_untimed_rows:
      var s = segmentsOf[T]([(0, 100), (100, 100), (200, 100)])
      s.keys[1].status = tsDead
      s.keys[2].time = NaN
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(90)], window(20))
      doAssert plan.untimed == @[2] and plan.gapSamples(0) == 10

    block no_allocation_after_reserve:
      let s = segmentsOf[T]([(0, 100), (100, 40), (140, 30), (170, 50), (220, 200)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      for align in [saNearest, saCubic, saSinc]:
        let plan = planSlices(meta, [shotAt(95.3), shotAt(135), shotAt(400)], window(80),
          align = align)
        var dst = zeros[T](plan.len, plan.samples)
        var report = initSliceReport(plan)
        var ws = initWorkspace()
        ws.reserve(sliceWorkspace(T, plan))
        let b = rowBlock(s.data)
        let d = rowBlock(dst)
        extractInto(plan, b, d, meta, report, ws)
        doAssert allocsIn(extractInto(plan, b, d, meta, report, ws)) == 0, $align
        doAssert allocsIn(extractInto(plan, s.data, dst, meta, report, ws)) == 0, $align
        doAssert report.gapSamples(2) == 60

    block bad_arguments:
      var s = segmentsOf[T]([(0, 100), (100, 100)])
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      doAssertRaises ValueError: # 3 ms is not a whole number of 2 ms samples
        discard planSlices(meta, [shotAt(0)], SliceWindow(startMs: 0, lengthMs: 3))
      doAssertRaises ValueError:
        discard planSlices(meta, [shotAt(0)], SliceWindow(startMs: 0, lengthMs: 0))
      doAssertRaises ValueError:
        discard planSlices(meta, [shotAt(0)], SliceWindow(startMs: NaN, lengthMs: 4))
      doAssertRaises ValueError:
        discard planSlices(meta, [initSliceShot(NaN, 1)], window(10))
      doAssertRaises ValueError:
        discard planSlices(meta, [1.0], [shotAt(0)], window(10))
      doAssertRaises ValueError:
        discard planSlices(meta, [1.0, 2.0], [1], [shotAt(0)], window(10))
      let plan = planSlices(meta, [shotAt(0)], window(10))
      var report = initSliceReport(plan)
      var ws = initWorkspace()
      var wrong = zeros[T](1, 11)
      doAssertRaises ValueError:
        extractInto(plan, s.data, wrong, meta, report, ws)
      var dst = zeros[T](1, 10)
      var narrow = zeros[T](2, 5)
      doAssertRaises ValueError:
        extractInto(plan, narrow, dst, meta, report, ws)
      let other = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = 4)
      doAssertRaises ValueError:
        extractInto(plan, s.data, dst, other, report, ws)
      var stale = initSliceReport(planSlices(meta, [shotAt(0), shotAt(5)], window(10)))
      doAssertRaises ValueError:
        extractInto(plan, s.data, dst, meta, stale, ws)

    block a_row_of_another_interval_is_reported:
      # Mixed sample rates are a property of the data (a line recorded at
      # 1 ms beside one at 2 ms), not a bad argument: the row is left out and
      # listed, and its span is a gap.
      var s = segmentsOf[T]([(0, 100), (100, 100), (200, 100)])
      s.keys[1].intervalUs = 1000
      s.keys[2].intervalUs = NaN # unknown: taken at the meta's interval
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let plan = planSlices(meta, [shotAt(90), shotAt(190)], window(20))
      doAssert plan.otherInterval == @[1] and plan.untimed.len == 0
      doAssert plan.rowsNeeded == @[0, 2]
      doAssert plan.gapSamples(0) == 10 and plan.gapSamples(1) == 10
      var report: SliceReport
      let dst = cut(s, meta, plan, report)
      for j in 0 ..< 10:
        doAssert dst[0, j] == T(streamValue(90 + j)) and dst[0, 10 + j] == 0
        doAssert dst[1, j] == 0 and dst[1, 10 + j] == T(streamValue(200 + j))

    block piecewise_matches_extract:
      # Receiver 1 overlaps itself by 3 samples; receiver 2 is short. Shots
      # sit before the first sample, on a segment boundary, and past the last.
      let s = segmentsOf[T]([(0, 100), (97, 80)], point = 1).join(
        segmentsOf[T]([(0, 40)], point = 2))
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      let shots = [shotAt(-20), shotAt(0), shotAt(90.4), shotAt(160), shotAt(500)]

      proc traceAt(plan: SlicePlan, recv, shot: int): int =
        if plan.order == soReceiverGathers: recv * plan.shotCount + shot
        else: shot * plan.receiverCount + recv

      proc cutPieces(seg: Segments[T], plan: SlicePlan, meta: ChunkMeta,
          report: var SliceReport, overlaps: bool): Tensor[T] =
        result = zeros[T](plan.len, plan.samples)
        report = initSliceReport(plan)
        var ws = initWorkspace()
        ws.reserve(sliceWorkspace(T, plan))
        var hold = newSeq[T](1)
        let src = rowBlock(seg.data)
        let dstb = rowBlock(result)
        for t in 0 ..< plan.len:
          for pc in plan.pieces(t):
            if pc.count > hold.len: hold.setLen(pc.count)
        doAssert allocsIn(
          block:
            prepareSliceReport(plan, report)
            for trace in 0 ..< plan.len:
              beginSliceTrace(plan, trace, report)
              for pc in plan.pieces(trace):
                let width = if pc.row >= 0: src.cols else: 0
                let (first, count) = pieceSource(plan, pc, width)
                if pc.row >= 0 and pc.count > 0 and pc.first >= 0 and pc.first < width:
                  # The piece already includes the margin: do not subtract it.
                  doAssert first == pc.first, "pieceSource subtracted the margin"
                  doAssert count <= pc.count
                  doAssert first >= pc.first - plan.margin
                  doAssert first + count <= pc.first + pc.count + plan.margin
                  doAssert first + count <= width
                let failed = pc.row >= 0 and not meta.isLive(pc.row)
                if count == 0:
                  addPiece(plan, trace, pc, hold.toOpenArray(0, -1), dstb.row(trace),
                    report, ws, failed = failed)
                else:
                  for j in 0 ..< count: hold[j] = src.rowPtr(pc.row)[first + j]
                  addPiece(plan, trace, pc, hold.toOpenArray(0, count - 1),
                    dstb.row(trace), report, ws, failed = failed)
              finishSliceTrace(plan, trace, dstb.row(trace), report, ws)
            if overlaps:
              for i, o in plan.overlaps:
                if meta.isLive(o.earlier) and meta.isLive(o.later) and o.samples > 0 and
                    o.first + o.samples <= src.cols:
                  checkSliceOverlap(plan, i, toOpenArray(src.rowPtr(o.earlier), o.first,
                    o.first + o.samples - 1), toOpenArray(src.rowPtr(o.later), 0,
                    o.samples - 1), report)
        ) == 0

      proc sameReport(a, b: SliceReport, plan: SlicePlan) =
        for t in 0 ..< plan.len:
          doAssert a.gapSamples(t) == b.gapSamples(t)
          doAssert a.failedSamples(t) == b.failedSamples(t)
          var ga, gb, fa, fb: seq[SliceGap]
          for g in a.gaps(t): ga.add g
          for g in b.gaps(t): gb.add g
          for g in a.failedPieces(t): fa.add g
          for g in b.failedPieces(t): fb.add g
          doAssert ga == gb and fa == fb
        for i in 0 ..< plan.overlaps.len:
          doAssert a.overlapCheck(i) == b.overlapCheck(i)

      for order in [soReceiverGathers, soShotRecords]:
        for align in [saNearest, saLinear, saCubic, saSinc]:
          let plan = planSlices(meta, shots, window(40), order, align)
          let span = traceAt(plan, 0, 2)
          doAssert plan.fraction(span) != 0
          var nData = 0
          for pc in plan.pieces(span):
            if pc.row >= 0: inc nData
          doAssert nData == 2, "a shot spanning two segments"
          var report: SliceReport
          let full = cut(s, meta, plan, report)
          var pieceReport: SliceReport
          let piece = cutPieces(s, plan, meta, pieceReport, overlaps = true)
          doAssert piece == full, $order & " " & $align
          sameReport(pieceReport, report, plan)
          doAssert report.gapSamples(traceAt(plan, 0, 0)) > 0
          doAssert report.gapSamples(traceAt(plan, 0, 4)) == plan.samples

      # A row that fails at extract time is a failed piece on both paths, not a gap.
      let rows = rowsOf(s.keys, s.counts)
      let failedPlan = planSlices(meta, [shotAt(90.4)], window(20), align = saCubic)
      let failedMeta = initChunkMeta(rows, dtMs = Dt, failed = @[1])
      var failedReport: SliceReport
      let failedFull = cut(s, failedMeta, failedPlan, failedReport)
      var failedPiece: SliceReport
      doAssert cutPieces(s, failedPlan, failedMeta, failedPiece, overlaps = true) == failedFull
      sameReport(failedPiece, failedReport, failedPlan)

      # Both overlap runs given: a mismatch agrees. Not given: osNotRead, as a
      # block narrower than the trimmed tail.
      var bad = segmentsOf[T]([(0, 100), (97, 80)])
      bad.data[1, 0] = bad.data[1, 0] + 1
      let badMeta = initChunkMeta(rowsOf(bad.keys, bad.counts), dtMs = Dt)
      let badPlan = planSlices(badMeta, [shotAt(90)], window(20), align = saLinear)
      var badReport: SliceReport
      discard cut(bad, badMeta, badPlan, badReport)
      doAssert badReport.overlapCheck(0).status == osMismatch
      var badPiece: SliceReport
      discard cutPieces(bad, badPlan, badMeta, badPiece, overlaps = true)
      doAssert badPiece.overlapCheck(0) == badReport.overlapCheck(0)
      var unread: SliceReport
      discard cutPieces(bad, badPlan, badMeta, unread, overlaps = false)
      doAssert unread.overlapCheck(0).status == osNotRead
      let narrowPlan = planSlices(badMeta, [shotAt(0)], window(10))
      doAssert narrowPlan.overlaps[0].first + narrowPlan.overlaps[0].samples >
        narrowPlan.maxColumn
      var narrow = zeros[T](bad.data.shape[0], narrowPlan.maxColumn)
      for r in 0 ..< narrow.shape[0]:
        for c in 0 ..< narrow.shape[1]: narrow[r, c] = bad.data[r, c]
      var narrowReport: SliceReport
      discard cut(Segments[T](keys: bad.keys, counts: bad.counts, data: narrow),
        badMeta, narrowPlan, narrowReport)
      doAssert narrowReport.overlapCheck(0).status == osNotRead

  suite[float32]()
  suite[float64]()

  block doc_example:
    # Three 2 s segments at 1 ms; a 1 s window from 1.5 s straddles rows 0 and 1.
    var keys = newSeq[TraceKeys](3)
    for i in 0 ..< 3:
      keys[i] = initTraceKeys()
      keys[i].status = tsLive
      keys[i].time = 1000.0 + 2.0 * float64(i)
      keys[i].t0Ms = 0 # NaN (unknown) would leave the rows untimed
    let counts = @[2000, 2000, 2000]
    let meta = initChunkMeta(rowsOf(keys, counts), dtMs = 1)
    let shots = [initSliceShot(time = 1001.5, id = 7)]
    let plan = planSlices(meta, shots, SliceWindow(startMs: 0, lengthMs: 1000))
    doAssert plan.len == 1 and plan.samples == 1000
    doAssert plan.rowsNeeded == @[0, 1]

  block traces_needed_of_a_view_inside_the_file:
    # Four segments of three receivers on disk (trace = 3 · segment +
    # receiver). The plan is made on a view that skips the first segment, so
    # its rows are not the file's trace numbers.
    const segLen = 500
    let path = getTempDir() / "processseg_tslice_continuous.sgy"
    block:
      var w = openSeismicWriter(path, SegyOutputOptions())
      var x = newSeq[float32](segLen)
      for s in 0 ..< 4:
        for r in 0 ..< 3:
          var k = initTraceKeys()
          k.status = tsLive
          (k.rcvLine, k.rcvPoint, k.channel) = (10.0, float64(100 + r), float64(r + 1))
          k.time = 1.7e9 + float64(s * segLen) * Dt / 1000
          k.t0Ms = 0
          k.intervalUs = Dt * 1000
          for j in 0 ..< segLen: x[j] = float32(r * 100_000 + s * segLen + j)
          w.writeTrace(k, x)
      discard w.close()
    var f = openSeismic(path, keys = true)
    let meta = chunkMeta(f, 3 ..< 12)
    # A window of 100 samples from 30 samples before the end of segment 1.
    let shot = initSliceShot(meta.keys(0).time + float64(segLen - 30) * Dt / 1000, id = 1)
    let plan = planSlices(meta, [shot], window(100), soShotRecords)
    doAssert plan.rowsNeeded == @[0, 1, 2, 3, 4, 5], "rows of the planning view"
    doAssert plan.tracesNeeded == @[3, 4, 5, 6, 7, 8], "their traces in the file"
    # The whole view as one block ...
    var all = zeros[float32](meta.len, segLen)
    doAssert f.readTracesInto(3 ..< 12, all) == 0
    var report = initSliceReport(plan)
    var ws = initWorkspace()
    var want = zeros[float32](plan.len, plan.samples)
    extractInto(plan, all, want, meta, report, ws)
    for r in 0 ..< plan.len:
      for j in 0 ..< plan.samples:
        doAssert want[r, j] == float32(r * 100_000 + 2 * segLen - 30 + j)
    # ... and a read of the traces needed alone give the same records.
    let need = plan.tracesNeeded
    var some = zeros[float32](need.len, segLen)
    doAssert f.readTracesInto(need, some) == 0
    var got = zeros[float32](plan.len, plan.samples)
    extractInto(plan, some, got, chunkMeta(f, need), report, ws)
    doAssert got == want, "the compact read"
    f.close()
    removeFile(path)

  block a_receiver_is_its_station_with_the_index:
    # The receiver of a row is its station as parseSEG's `stationKey` makes
    # it: line and point to 1/100, with the point index.
    # Two deployments at one line and point (index 1 and 2) record the same
    # span: two receivers, not one stream overlapping itself.
    var s = segmentsOf[float32]([(0, 100)], point = 7).join(segmentsOf[float32]([(0, 100)],
      point = 7, value = proc (k: int): float64 = -streamValue(k)))
    (s.keys[0].rcvIndex, s.keys[1].rcvIndex) = (1'i16, 2'i16)
    let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
    let plan = planSlices(meta, [shotAt(10)], window(20), soShotRecords)
    doAssert plan.receiverCount == 2 and plan.overlaps.len == 0
    doAssert plan.receiverKeys(0).rcvIndex == 1 and plan.receiverKeys(1).rcvIndex == 2
    var report: SliceReport
    let dst = cut(s, meta, plan, report)
    doAssert dst[0, 0] == float32(streamValue(10)) and dst[1, 0] == float32(-streamValue(10))
    # A point decoded from a binary fraction (SEG-D: sixteen fraction bits) and
    # the same point in decimal are one station: two segments, one stream.
    var t = segmentsOf[float32]([(0, 100), (100, 100)], line = 12, point = 100.1)
    t.keys[1].rcvPoint = 100.0 + 6554.0 / 65536.0
    doAssert t.keys[0].rcvPoint != t.keys[1].rcvPoint
    let one = planSlices(initChunkMeta(rowsOf(t.keys, t.counts), dtMs = Dt), [shotAt(90)],
      window(20))
    doAssert one.receiverCount == 1 and one.gapSamples(0) == 0
    # Channel and component still tell the streams of one station apart.
    var c = segmentsOf[float32]([(0, 100)]).join(segmentsOf[float32]([(0, 100)]))
    c.keys[1].component = 2
    doAssert planSlices(initChunkMeta(rowsOf(c.keys, c.counts), dtMs = Dt), [shotAt(0)],
      window(10)).receiverCount == 2

  block slice_keys:
    var s = segmentsOf[float32]([(0, 100)], line = 4, point = 2)
    s.keys[0].channel = 3
    let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
    let shot = initSliceShot(timeOf(10), id = 1234, srcLine = 9, srcPoint = 5, srcX = 50,
      srcY = 0)
    let plan = planSlices(meta, [shot], window(20, startMs = -4))
    let k = plan.sliceKeys(0)
    doAssert k.record == 1234 and k.srcLine == 9 and k.srcPoint == 5
    doAssert k.rcvLine == 4 and k.rcvPoint == 2 and k.channel == 3
    doAssert k.offset == 30 and k.cmpX == 35, "offset from the coordinates"
    doAssert k.time == timeOf(10) and k.t0Ms == -4 and k.intervalUs == 2000
    doAssert k.status == tsLive and k.ensemble.isNaN

  block slice_keys_take_the_source_from_the_shot_alone:
    # The segment's own source station (a node file's general header) must
    # not leave its point index beside the shot's line and point.
    var s = segmentsOf[float32]([(0, 100)])
    (s.keys[0].srcLine, s.keys[0].srcPoint, s.keys[0].srcIndex) = (77.0, 88.0, 3'i16)
    let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
    let unknown = initSliceShot(timeOf(10), id = 5, srcLine = 9, srcPoint = 4)
    let second = initSliceShot(timeOf(20), id = 6, srcLine = 9, srcPoint = 4, srcIndex = 2)
    let plan = planSlices(meta, [unknown, second], window(20))
    let k = plan.sliceKeys(0)
    doAssert k.srcLine == 9 and k.srcPoint == 4 and k.srcIndex == 0, $k.srcIndex
    doAssert plan.sliceKeys(1).srcIndex == 2

  block slice_keys_keep_the_source_interval:
    # A 31 Hz stream: 1e6 / 31 µs is not `(that / 1000) · 1000` in float64.
    # The output keys hold the interval the segments' keys hold, bit for bit,
    # so a writer and `uniformRuns` see one interval, not two.
    let us = 1e6 / 31
    doAssert us / 1000 * 1000 != us, "the case this test is about"
    var keys = @[initTraceKeys(), initTraceKeys()]
    for i, k in keys.mpairs:
      k.status = tsLive
      (k.time, k.t0Ms, k.intervalUs) = (1000.0, 0.0, us)
      (k.rcvLine, k.rcvPoint) = (1.0, float64(i))
    keys[1].intervalUs = NaN # unknown in the headers: from the meta
    let counts = @[100, 100]
    let meta = initChunkMeta(rowsOf(keys, counts), dtMs = us / 1000)
    let plan = planSlices(meta, [initSliceShot(1000.0, id = 1)],
      SliceWindow(startMs: 0, lengthMs: 10 * us / 1000))
    doAssert plan.sliceKeys(0).intervalUs == us
    doAssert plan.sliceKeys(1).intervalUs == meta.dtMs * 1000

  block a_sample_count_beyond_any_file:
    # No file holds a trace of 2⁶³ samples: it is the caller's array, so a bad
    # argument, and must not overflow the positions on the sample grid.
    let s = segmentsOf[float32]([(0, 10), (10, 100)])
    let counts = @[10, high(int)]
    let meta = initChunkMeta(rowsOf(s.keys, counts), dtMs = Dt)
    doAssertRaises ValueError:
      discard planSlices(meta, [shotAt(0)], window(10))

  block drift_correction:
    # The clock is 1 ms ahead at t = 1000 s and 3 ms ahead at t = 3000 s.
    let times = [1000.0, 2000.0, 3000.0, 5000.0, NaN]
    let fixed = driftCorrectedTimes(times, 1000, 0.001, 3000, 0.003)
    doAssert abs(fixed[0] - 999.999) < 1e-9 and abs(fixed[1] - 1999.998) < 1e-9
    doAssert abs(fixed[3] - 4999.995) < 1e-9 and fixed[4].isNaN
    var inPlace = @times
    driftCorrectTimes(inPlace, 1000, 0.001, 3000, 0.003)
    doAssert inPlace[2] == fixed[2]
    doAssertRaises ValueError:
      discard driftCorrectedTimes(times, 1000, 0, 1000, 0.001)
    doAssertRaises ValueError:
      discard driftCorrectedTimes(times, NaN, 0, 1000, 0.001)

  proc samePlans(a, b: SlicePlan): bool =
    ## Every public part of two plans agrees.
    if a.len != b.len or a.samples != b.samples or a.receiverCount != b.receiverCount or
        a.maxColumn != b.maxColumn or a.rowsNeeded != b.rowsNeeded or
        a.tracesNeeded != b.tracesNeeded or a.overlaps != b.overlaps or
        a.timingIssues != b.timingIssues or a.untimed != b.untimed or
        a.otherInterval != b.otherInterval:
      return false
    for t in 0 ..< a.len:
      if a.fraction(t) != b.fraction(t) or a.gapSamples(t) != b.gapSamples(t): return false
      var pa, pb: seq[SlicePiece]
      for p in a.pieces(t): pa.add p
      for p in b.pieces(t): pb.add p
      if pa != pb: return false
      let (ka, kb) = (a.sliceKeys(t), b.sliceKeys(t))
      if ka.record != kb.record or ka.status != kb.status or ka.rcvPoint != kb.rcvPoint or
          ka.time != kb.time:
        return false
    true

  block streams_made_once_serve_every_plan:
    # Random segmentations of several receivers, with gaps, overlaps, dead and
    # untimed rows and a row of another interval: plans made from one
    # `SliceStreams`, for any shots, order and alignment, are the plans the
    # one-call form makes from the meta; and a compact read of the rows
    # needed extracts what the full block does.
    var rng = initRand(7)
    for trial in 0 ..< 30:
      var s: Segments[float32]
      for r in 0 ..< rng.rand(1 .. 4):
        var spans: seq[(int, int)]
        var pos = rng.rand(0 .. 20)
        while pos < 400:
          let n = rng.rand(1 .. 60)
          let back = if pos > 0 and rng.rand(3) == 0: rng.rand(0 .. min(5, pos)) else: 0
          spans.add (pos - back, n + back)
          pos += n + (if rng.rand(5) == 0: rng.rand(1 .. 15) else: 0)
        let part = segmentsOf[float32](spans, point = float64(r + 1),
          value = proc (k: int): float64 = streamValue(k + 1000 * r))
        s = if r == 0: part else: s.join(part)
      for i in 0 ..< s.keys.len:
        case rng.rand(25)
        of 0: s.keys[i].status = tsDead
        of 1: s.keys[i].time = NaN
        of 2: s.keys[i].intervalUs = 4 * Dt * 1000
        of 3: s.keys[i].time += 0.3 * Dt / 1000
        else: discard
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      var times = newSeq[float64](meta.len)
      var ids = newSeq[int](meta.len)
      for i in 0 ..< meta.len:
        times[i] = s.keys[i].time
        ids[i] = int(s.keys[i].rcvPoint) mod 2
      let byKeys = initSliceStreams(meta)
      let byTimes = initSliceStreams(meta, times)
      let byIds = initSliceStreams(meta, times, ids)
      doAssert byKeys.rowCount == meta.len and byKeys.dtMs == Dt
      for round in 0 ..< 3:
        var shots: seq[SliceShot]
        for i in 0 ..< rng.rand(0 .. 5): shots.add shotAt(rng.rand(-50.0 .. 450.0), float64(i))
        let w = window(rng.rand(1 .. 90), startMs = float64(rng.rand(-10 .. 10)) * Dt)
        let order = if rng.rand(1) == 0: soReceiverGathers else: soShotRecords
        let align = [saNearest, saLinear, saCubic, saSinc][rng.rand(3)]
        let plan = planSlices(byKeys, shots, w, order, align)
        doAssert samePlans(plan, planSlices(meta, shots, w, order, align)), $trial
        doAssert samePlans(planSlices(byTimes, shots, w, order, align),
          planSlices(meta, times, shots, w, order, align)), $trial
        doAssert samePlans(planSlices(byIds, shots, w, order, align),
          planSlices(meta, times, ids, shots, w, order, align)), $trial
        # The streams list what the plan lists for the whole dataset.
        doAssert plan.overlaps == byKeys.overlaps and plan.untimed == byKeys.untimed
        doAssert plan.timingIssues == byKeys.timingIssues
        doAssert plan.otherInterval == byKeys.otherInterval
        doAssert plan.receiverCount == byKeys.receiverCount
        for r in 0 ..< plan.receiverCount:
          doAssert plan.receiverKeys(r).rcvPoint == byKeys.receiverKeys(r).rcvPoint
        # Rows needed: ascending, each read by a piece, and no other row is.
        var seen: seq[int]
        for t in 0 ..< plan.len:
          for p in plan.pieces(t):
            if p.row >= 0 and p.row notin seen: seen.add p.row
        seen.sort()
        doAssert plan.rowsNeeded == seen, $trial
        # A block of the needed rows alone gives the same records and reports.
        var report: SliceReport
        let full = cut(s, meta, plan, report)
        var sub: Segments[float32]
        sub.data = zeros[float32](max(1, seen.len), s.data.shape[1])
        for i, r in seen:
          sub.keys.add s.keys[r]
          sub.counts.add s.counts[r]
          for j in 0 ..< s.data.shape[1]: sub.data[i, j] = s.data[r, j]
        if seen.len > 0 and seen.len < meta.len:
          var subReport: SliceReport
          let subMeta = initChunkMeta(rowsOf(sub.keys, sub.counts), dtMs = Dt)
          doAssert cut(sub, subMeta, plan, subReport) == full, $trial
          for t in 0 ..< plan.len:
            doAssert subReport.gapSamples(t) == report.gapSamples(t)
          for i in 0 ..< plan.overlaps.len:
            doAssert subReport.overlapCheck(i) == report.overlapCheck(i)

  block streams_arguments:
    let s = segmentsOf[float32]([(0, 100), (100, 100)])
    let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
    doAssertRaises ValueError: discard initSliceStreams(meta, [1.0])
    doAssertRaises ValueError: discard initSliceStreams(meta, [1.0, 2.0], [1])
    let tiny = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = 1e-5)
    doAssertRaises ValueError: discard initSliceStreams(tiny)
    let streams = initSliceStreams(meta)
    doAssert streams.receiverCount == 1 and streams.rowCount == 2
    doAssertRaises ValueError: # 3 ms is not a whole number of 2 ms samples
      discard planSlices(streams, [shotAt(0)], SliceWindow(startMs: 0, lengthMs: 3))
    doAssertRaises ValueError:
      discard planSlices(streams, [initSliceShot(NaN, 1)], window(10))
    # No shots: an empty plan, with the dataset's lists all the same.
    let none = planSlices(streams, newSeq[SliceShot](), window(10))
    doAssert none.len == 0 and none.rowsNeeded.len == 0 and none.receiverCount == 1
    # The streams hold no pointer into the keys: they outlive the meta's arrays.
    var kept: SliceStreams
    block:
      var gone = segmentsOf[float32]([(0, 50), (50, 50)], point = 9)
      kept = initSliceStreams(initChunkMeta(rowsOf(gone.keys, gone.counts), dtMs = Dt))
      gone.keys.setLen(0)
    let later = planSlices(kept, [shotAt(40)], window(20))
    doAssert later.rowsNeeded == @[0, 1] and later.receiverKeys(0).rcvPoint == 9

  block random_windows_match_the_stream:
    # Random segmentations (with overlaps of matching samples) and windows.
    var rng = initRand(42)
    for trial in 0 ..< 20:
      var spans: seq[(int, int)]
      var pos = 0
      while pos < 600:
        let n = rng.rand(1 .. 80)
        let back = if pos > 0 and rng.rand(4) == 0: rng.rand(0 .. min(5, pos)) else: 0
        spans.add (pos - back, n + back)
        pos += n
      rng.shuffle(spans)
      let s = segmentsOf[float64](spans)
      let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
      var starts: seq[int]
      var shots: seq[SliceShot]
      for i in 0 ..< 10:
        starts.add rng.rand(0 .. 500)
        shots.add shotAt(float64(starts[^1]))
      let plan = planSlices(meta, shots, window(100))
      var report: SliceReport
      let dst = cut(s, meta, plan, report)
      for t, k in starts: checkDirect(dst, t, k, "trial " & $trial)
      doAssert report.mismatches == 0

  block piece_length_must_match_pieceSource:
    let s = segmentsOf[float32]([(0, 100)])
    let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
    let plan = planSlices(meta, [shotAt(10)], window(20))
    var pc0: SlicePiece
    for pc in plan.pieces(0): pc0 = pc
    doAssert pc0.row == 0 and pc0.count == 20
    let asked = pieceSource(plan, pc0, 100).count
    doAssert asked == 20
    var report = initSliceReport(plan)
    var ws = initWorkspace()
    ws.reserve(sliceWorkspace(float32, plan))
    var dst = newSeq[float32](plan.samples)
    for i in 0 ..< dst.len: dst[i] = 7
    let hold = newSeq[float32](15)
    prepareSliceReport(plan, report)
    beginSliceTrace(plan, 0, report)
    var msg = ""
    try:
      addPiece(plan, 0, pc0, hold, dst, report, ws)
    except ValueError as e:
      msg = e.msg
    doAssert msg.find("trace 0") >= 0, msg
    doAssert msg.find("15") >= 0 and msg.find("20") >= 0, msg
    doAssert msg.find("row 0") >= 0 and msg.find("first " & $pc0.first) >= 0, msg
    for v in dst: doAssert v == 7, "a short piece must write nothing"
    doAssert report.gapSamples(0) == 0 and report.gapCount(0) == 0
    # Longer than asked already raised; the message names the same facts.
    msg = ""
    try:
      addPiece(plan, 0, pc0, newSeq[float32](21), dst, report, ws)
    except ValueError as e:
      msg = e.msg
    doAssert msg.find("trace 0") >= 0 and msg.find("21") >= 0 and msg.find("20") >= 0, msg
    for v in dst: doAssert v == 7

  block call_order_is_checked:
    let s = segmentsOf[float32]([(0, 100), (100, 100)])
    let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
    let plan = planSlices(meta, [shotAt(90), shotAt(10)], window(20))
    var pieces0: seq[SlicePiece]
    for pc in plan.pieces(0): pieces0.add pc
    doAssert pieces0.len == 2
    var report = initSliceReport(plan)
    var ws = initWorkspace()
    ws.reserve(sliceWorkspace(float32, plan))
    var dst = newSeq[float32](plan.samples)
    for i in 0 ..< dst.len: dst[i] = 3
    proc take(pc: SlicePiece): seq[float32] =
      result = newSeq[float32](pc.count)
      if pc.row >= 0:
        for j in 0 ..< pc.count: result[j] = s.data[pc.row, pc.first + j]
    proc raised(body: proc () {.closure.}): string =
      try: body()
      except ValueError as e: result = e.msg
    prepareSliceReport(plan, report)
    let again = raised do (): beginSliceTrace(plan, 0, report)
    doAssert again.len == 0
    let second = raised do (): beginSliceTrace(plan, 0, report)
    doAssert second.find("still open") >= 0, second
    doAssert report.gapSamples(0) == 0 and report.gapCount(0) == 0
    let other = raised do (): beginSliceTrace(plan, 1, report)
    doAssert other.find("still open") >= 0, other
    # Out of turn: the second piece first. Nothing written.
    let skip = raised do ():
      addPiece(plan, 0, pieces0[1], take(pieces0[1]), dst, report, ws)
    doAssert skip.find("expects piece") >= 0, skip
    for v in dst: doAssert v == 3
    doAssert report.gapCount(0) == 0
    # The first piece, then the same piece again.
    addPiece(plan, 0, pieces0[0], take(pieces0[0]), dst, report, ws)
    let saved = dst
    let dup = raised do ():
      addPiece(plan, 0, pieces0[0], take(pieces0[0]), dst, report, ws)
    doAssert dup.find("expects piece") >= 0, dup
    doAssert dst == saved
    let early = raised do (): finishSliceTrace(plan, 0, dst, report, ws)
    doAssert early.find("still has") >= 0, early
    let wrong = raised do (): finishSliceTrace(plan, 1, dst, report, ws)
    doAssert wrong.find("is open") >= 0, wrong
    addPiece(plan, 0, pieces0[1], take(pieces0[1]), dst, report, ws)
    finishSliceTrace(plan, 0, dst, report, ws)
    let rebegin = raised do (): beginSliceTrace(plan, 0, report)
    doAssert rebegin.find("already finished") >= 0, rebegin
    # Trace 1, never begun.
    var dst1 = newSeq[float32](plan.samples)
    let naked = raised do ():
      addPiece(plan, 1, pieces0[0], take(pieces0[0]), dst1, report, ws)
    doAssert naked.find("not been begun") >= 0, naked
    let bare = raised do (): finishSliceTrace(plan, 1, dst1, report, ws)
    doAssert bare.find("not been begun") >= 0, bare
    # Traces in reverse order still match extractInto.
    var full = zeros[float32](plan.len, plan.samples)
    var fullReport = initSliceReport(plan)
    extractInto(plan, s.data, full, meta, fullReport, ws)
    var rev = zeros[float32](plan.len, plan.samples)
    var revReport = initSliceReport(plan)
    let revRows = rowBlock(rev)
    prepareSliceReport(plan, revReport)
    for trace in countdown(plan.len - 1, 0):
      beginSliceTrace(plan, trace, revReport)
      for pc in plan.pieces(trace):
        let src = take(pc)
        if pc.row < 0:
          addPiece(plan, trace, pc, src.toOpenArray(0, -1), revRows.row(trace), revReport, ws)
        else:
          addPiece(plan, trace, pc, src, revRows.row(trace), revReport, ws)
      finishSliceTrace(plan, trace, revRows.row(trace), revReport, ws)
    doAssert rev == full
    for t in 0 ..< plan.len:
      doAssert revReport.gapSamples(t) == fullReport.gapSamples(t)
      doAssert revReport.gapCount(t) == fullReport.gapCount(t)

  block finish_requires_the_same_dst:
    let s = segmentsOf[float32]([(0, 100), (100, 100)])
    let meta = initChunkMeta(rowsOf(s.keys, s.counts), dtMs = Dt)
    let plan = planSlices(meta, [shotAt(90)], window(20))
    var pieces: seq[SlicePiece]
    for pc in plan.pieces(0): pieces.add pc
    doAssert pieces.len == 2
    var report = initSliceReport(plan)
    var ws = initWorkspace()
    prepareSliceReport(plan, report)
    var dst = newSeq[float32](plan.samples)
    var other = newSeq[float32](plan.samples)
    for i in 0 ..< dst.len: dst[i] = 3
    for i in 0 ..< other.len: other[i] = 9
    proc take(pc: SlicePiece): seq[float32] =
      result = newSeq[float32](pc.count)
      if pc.row >= 0:
        for j in 0 ..< pc.count: result[j] = s.data[pc.row, pc.first + j]
    proc raised(body: proc () {.closure.}): string =
      try: body()
      except ValueError as e: result = e.msg
    beginSliceTrace(plan, 0, report)
    addPiece(plan, 0, pieces[0], take(pieces[0]), dst, report, ws)
    let saved = dst
    let mid = raised do ():
      addPiece(plan, 0, pieces[1], take(pieces[1]), other, report, ws)
    doAssert mid.find("same dst") >= 0, mid
    doAssert dst == saved
    for v in other: doAssert v == 9
    doAssert report.gapCount(0) == 0
    addPiece(plan, 0, pieces[1], take(pieces[1]), dst, report, ws)
    let wrong = raised do (): finishSliceTrace(plan, 0, other, report, ws)
    doAssert wrong.find("same dst") >= 0, wrong
    for v in other: doAssert v == 9
    finishSliceTrace(plan, 0, dst, report, ws)
    var full = zeros[float32](plan.len, plan.samples)
    var fullReport = initSliceReport(plan)
    extractInto(plan, s.data, full, meta, fullReport, ws)
    for j in 0 ..< plan.samples: doAssert dst[j] == full[0, j]
    # A shifted cut copies the scratch row into that same dst, not another.
    let shifted = planSlices(meta, [shotAt(90.4)], window(20), align = saLinear)
    var sp: seq[SlicePiece]
    for pc in shifted.pieces(0): sp.add pc
    var sReport = initSliceReport(shifted)
    var sws = initWorkspace()
    sws.reserve(sliceWorkspace(float32, shifted))
    prepareSliceReport(shifted, sReport)
    var sDst = newSeq[float32](shifted.samples)
    var sOther = newSeq[float32](shifted.samples)
    for i in 0 ..< sOther.len: sOther[i] = 9
    beginSliceTrace(shifted, 0, sReport)
    for pc in sp:
      let src = take(pc)
      if pc.row < 0:
        addPiece(shifted, 0, pc, src.toOpenArray(0, -1), sDst, sReport, sws)
      else:
        addPiece(shifted, 0, pc, src, sDst, sReport, sws)
    let moved = raised do (): finishSliceTrace(shifted, 0, sOther, sReport, sws)
    doAssert moved.find("same dst") >= 0, moved
    for v in sOther: doAssert v == 9
    for v in sDst: doAssert v == 0
    finishSliceTrace(shifted, 0, sDst, sReport, sws)
    var sFull = zeros[float32](shifted.len, shifted.samples)
    var sFullReport = initSliceReport(shifted)
    extractInto(shifted, s.data, sFull, meta, sFullReport, sws)
    for j in 0 ..< shifted.samples: doAssert sDst[j] == sFull[0, j]

  block both_paths_agree_on_seeded_plans:
    var rng = initRand(20241124)
    let aligns = [saNearest, saLinear, saCubic, saSinc]
    for trial in 0 ..< 200:
      var spans: seq[(int, int)]
      var pos = 0
      for _ in 0 ..< rng.rand(1 .. 4):
        let n = rng.rand(8 .. 40)
        let back = if pos > 0 and rng.rand(3) == 0: rng.rand(0 .. min(4, pos)) else: 0
        if pos > 0 and rng.rand(4) == 0: pos += rng.rand(1 .. 6)
        spans.add (pos - back, n + back)
        pos += n
      let seg = segmentsOf[float32](spans)
      var failed: seq[int]
      if spans.len > 1 and rng.rand(3) == 0: failed.add rng.rand(0 .. spans.high)
      let meta = initChunkMeta(rowsOf(seg.keys, seg.counts), dtMs = Dt, failed = failed)
      var shots: seq[SliceShot]
      for _ in 0 ..< rng.rand(1 .. 3):
        let frac = if rng.rand(2) == 0: 0.0 else: 0.3
        shots.add shotAt(float64(rng.rand(-10 .. pos + 10)) + frac)
      let plan = planSlices(meta, shots, window(rng.rand(8 .. 24)),
        align = aligns[trial mod 4])
      var fullReport: SliceReport
      let full = cut(seg, meta, plan, fullReport)
      var piece = zeros[float32](plan.len, plan.samples)
      var pieceReport = initSliceReport(plan)
      var ws = initWorkspace()
      ws.reserve(sliceWorkspace(float32, plan))
      var hold = newSeq[float32](1)
      let src = rowBlock(seg.data)
      let dstb = rowBlock(piece)
      prepareSliceReport(plan, pieceReport)
      for trace in 0 ..< plan.len:
        beginSliceTrace(plan, trace, pieceReport)
        for pc in plan.pieces(trace):
          let width = if pc.row >= 0: src.cols else: 0
          let (first, count) = pieceSource(plan, pc, width)
          let dead = pc.row >= 0 and not meta.isLive(pc.row)
          if count == 0:
            addPiece(plan, trace, pc, hold.toOpenArray(0, -1), dstb.row(trace),
              pieceReport, ws, failed = dead)
          else:
            if count > hold.len: hold.setLen(count)
            for j in 0 ..< count: hold[j] = src.rowPtr(pc.row)[first + j]
            addPiece(plan, trace, pc, hold.toOpenArray(0, count - 1), dstb.row(trace),
              pieceReport, ws, failed = dead)
        finishSliceTrace(plan, trace, dstb.row(trace), pieceReport, ws)
      proc rowNeeded(row: int): bool =
        for r in plan.rowsNeeded:
          if r == row: return true
      for i, o in plan.overlaps:
        if rowNeeded(o.earlier) and rowNeeded(o.later) and
            meta.isLive(o.earlier) and meta.isLive(o.later) and
            o.first + o.samples <= src.cols:
          if o.samples == 0:
            var none: array[0, float32]
            checkSliceOverlap(plan, i, none, none, pieceReport)
          else:
            checkSliceOverlap(plan, i, toOpenArray(src.rowPtr(o.earlier), o.first,
              o.first + o.samples - 1), toOpenArray(src.rowPtr(o.later), 0,
              o.samples - 1), pieceReport)
      doAssert piece == full, "trial " & $trial & " " & $aligns[trial mod 4]
      for t in 0 ..< plan.len:
        doAssert pieceReport.gapSamples(t) == fullReport.gapSamples(t), "trial " & $trial
        doAssert pieceReport.failedSamples(t) == fullReport.failedSamples(t), "trial " & $trial
        doAssert pieceReport.gapCount(t) == fullReport.gapCount(t)
        doAssert pieceReport.failedCount(t) == fullReport.failedCount(t)
        var fa, fb: seq[(int, int)]
        for g in pieceReport.failedPieces(t): fa.add((g.first, g.count))
        for g in fullReport.failedPieces(t): fb.add((g.first, g.count))
        doAssert fa == fb, "trial " & $trial & " trace " & $t & ": failed pieces differ"
        var ga, gb: seq[(int, int)]
        for g in pieceReport.gaps(t): ga.add((g.first, g.count))
        for g in fullReport.gaps(t): gb.add((g.first, g.count))
        doAssert ga == gb, "trial " & $trial & " trace " & $t & ": gap runs differ"
      for i in 0 ..< plan.overlaps.len:
        let oa = pieceReport.overlapCheck(i)
        let ob = fullReport.overlapCheck(i)
        doAssert oa.status == ob.status, "trial " & $trial & " overlap " & $i & ": status"
        doAssert oa.differing == ob.differing, "trial " & $trial & " overlap " & $i &
          ": differing"
        doAssert oa.maxAbsDiff == ob.maxAbsDiff or (oa.maxAbsDiff.isNaN and ob.maxAbsDiff.isNaN),
          "trial " & $trial & " overlap " & $i & ": maxAbsDiff"

when isMainModule: main()
