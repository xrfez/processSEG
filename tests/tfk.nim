## f-k amplitude of a gather.
##
## Fail-first: `fk_dip_sign` failed while the k axis was not sign-flipped,
## so the dip peaked at index 6 instead of 10 (zero at 8). `fk_stop` failed
## while the stop checks were compiled out (`if false and stop`), so a
## never-true check was called 0 times instead of once per trace and frequency,
## and while a finished call returned true instead of the rows used and an
## abandoned call returned false instead of `Stopped`.

import std/[assertions, math]
import processSEG
import utils/[allocs, synth]

proc main*() =
  func near(a, b, tol: float64): bool = abs(a - b) <= tol

  proc gather(rows, cols: int, dtMs = 2.0): SynthGather[float64] =
    cmpGather[float64](cmpX = 0, offsets = newSeq[float64](rows), cols, dtMs)

  block fk_dip_sign:
    # 8 traces, 64 samples, a cosine on time bin 8 delayed by 1 sample per
    # trace. That is 0.125 cycles/trace; nK = nextFastLen(16) = 16, so the
    # peak is 2 bins to the positive side of zero.
    const rows = 8
    const cols = 64
    const dtMs = 2.0
    var g = gather(rows, cols, dtMs)
    let plan = initFkPlan(float64, dtMs, traces = rows, samples = cols, taperFraction = 0)
    doAssert plan.nK == 16 and plan.nFreq == 33
    doAssert near(plan.dfHz, 1000 / (64 * dtMs), 1e-12)
    doAssert near(plan.dkPerTrace, 1 / 16, 1e-15)
    let f = plan.dfHz * 8
    for r in 0 ..< rows:
      for j in 0 ..< cols:
        let t = float64(j - r) # later on later traces
        g.data[r, j] = cos(2 * PI * f * t * dtMs / 1000)
    let meta = initChunkMeta(g.rows, dtMs)
    doAssert fkRows(meta, cols, 0, cols) == rows
    var ws = initWorkspace()
    ws.reserve(plan)
    var amp = newSeq[float64](plan.nK * plan.nFreq)
    let n = allocsIn:
      doAssert plan.fkAmplitudeInto(g.data, meta, 0, amp, ws) == rows
    doAssert n == 0, $n & " allocations"
    let zero = plan.nK div 2
    var peakK, peakF = 0
    var peak = 0.0
    for ik in 0 ..< plan.nK:
      let a = amp[ik * plan.nFreq + 8]
      if a > peak:
        peak = a
        peakK = ik
        peakF = 8
    doAssert peakF == 8
    doAssert peakK == zero + 2, "later-on-later must be +k, peak at " & $peakK &
      " zero at " & $zero
    # The other side is the conjugate image, weaker or absent at this bin.
    doAssert amp[(zero - 2) * plan.nFreq + 8] < peak * 0.05
    doAssertRaises ValueError: discard initFkPlan(float64, dtMs, traces = 3, samples = 64)
    doAssertRaises ValueError: discard initFkPlan(float64, dtMs, traces = 8, samples = 7)

  block fk_stop:
    # Fail-first: failed while the stop checks were compiled out, seen 0.
    const rows = 8
    const cols = 64
    const dtMs = 2.0
    var g = gather(rows, cols, dtMs)
    let plan = initFkPlan(float64, dtMs, traces = rows, samples = cols)
    for r in 0 ..< rows:
      for j in 0 ..< cols:
        g.data[r, j] = cos(0.15 * float64(j) + 0.4 * float64(r))
    let meta = initChunkMeta(g.rows, dtMs)
    var ws = initWorkspace()
    ws.reserve(plan)
    var amp = newSeq[float64](plan.nK * plan.nFreq)
    let used = plan.fkAmplitudeInto(g.data, meta, 0, amp, ws)
    doAssert used == rows, "rows used " & $used
    let full = amp
    var again = newSeq[float64](amp.len)
    doAssert plan.fkAmplitudeInto(g.data, meta, 0, again, ws, stop = nil) == used
    doAssert again == full
    var seen = 0
    let never: StopCheck = proc (): bool {.gcsafe, raises: [].} =
      inc seen
      false
    doAssert plan.fkAmplitudeInto(g.data, meta, 0, again, ws, stop = never) == used
    doAssert again == full
    doAssert seen == rows + plan.nFreq, "checked once per trace and frequency, got " & $seen
    proc stoppedAfter(k: int): int =
      var calls = 0
      let stop: StopCheck = proc (): bool {.gcsafe, raises: [].} =
        inc calls
        calls >= k
      doAssert plan.fkAmplitudeInto(g.data, meta, 0, amp, ws, stop = stop) == Stopped
      doAssert calls <= k, "called " & $calls & " times, at most " & $k
      calls
    doAssert stoppedAfter(1) == 1
    doAssert stoppedAfter(3) == 3
    doAssert stoppedAfter(rows + 1) == rows + 1
    var once = 0
    let stopRow: StopCheck = proc (): bool {.gcsafe, raises: [].} =
      inc once
      true
    doAssert plan.fkAmplitudeInto(rowBlock(g.data), meta, 0, amp, ws, stop = stopRow) ==
      Stopped
    doAssert once == 1
    let quiet = allocsIn:
      discard plan.fkAmplitudeInto(g.data, meta, 0, amp, ws)
    doAssert quiet == 0, $quiet & " allocations"
    doAssert amp == full, "the plan and workspace are reusable after a stop"

when isMainModule: main()
