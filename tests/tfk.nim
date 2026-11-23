## f-k amplitude of a gather.
##
## Fail-first: `fk_dip_sign` failed while the k axis was not sign-flipped,
## so the dip peaked at index 6 instead of 10 (zero at 8), and again while
## `modulusInto` stored 0, so the peak frequency was 0 instead of 8. `fk_stop` failed
## while the stop checks were compiled out (`if false and stop`), so a
## never-true check was called 0 times instead of once per trace and frequency,
## and while a finished call returned true instead of the rows used and an
## abandoned call returned false instead of `Stopped`.
## `fk_squares_overflow` failed while the modulus squared in float64:
## uniform(-1e155, 1e155) was 40 infinities. `fk_squares_underflow` differed
## by 3.062e-5 of the peak. `fk_squares_flush` was 40 zeros.

import std/[assertions, math, random]
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

  proc tukeyW(n: int, fraction: float64): seq[float64] =
    result = newSeq[float64](n)
    let ramp = if fraction <= 0: 0 else: int(round(fraction * float64(n)))
    for i in 0 ..< n:
      let d = min(i, n - 1 - i)
      let w = if ramp <= 0 or d >= ramp: 1.0
              else: 0.5 - 0.5 * cos(PI * (float64(d) + 0.5) / float64(ramp))
      result[i] = w

  proc directFk(data: seq[seq[float64]], taper: float64): seq[float64] =
    ## Slow DFT of a tapered gather, magnitude via `hypot`. Time sign is
    ## `exp(-2πi jk/n)`; output `ik` reads spatial bin `nK/2 - ik`.
    let traces = data.len
    let samples = data[0].len
    let nTime = nextFastLen(samples)
    let nK = nextFastLen(2 * traces)
    let nFreq = nTime div 2 + 1
    let timeW = tukeyW(samples, taper)
    let traceW = tukeyW(traces, taper)
    var specRe = newSeq[seq[float64]](traces)
    var specIm = newSeq[seq[float64]](traces)
    for t in 0 ..< traces:
      specRe[t] = newSeq[float64](nFreq)
      specIm[t] = newSeq[float64](nFreq)
      for k in 0 ..< nFreq:
        var re, im = 0.0
        for j in 0 ..< samples:
          let x = data[t][j] * timeW[j] * traceW[t]
          let ang = -2 * PI * float64(k * j) / float64(nTime)
          re += x * cos(ang)
          im += x * sin(ang)
        specRe[t][k] = re
        specIm[t][k] = im
    result = newSeq[float64](nK * nFreq)
    let half = nK div 2
    for jf in 0 ..< nFreq:
      for ik in 0 ..< nK:
        let raw = half - ik
        let sk = if raw >= 0: raw else: raw + nK
        var re, im = 0.0
        for t in 0 ..< traces:
          let ang = -2 * PI * float64(sk * t) / float64(nK)
          let c = cos(ang)
          let s = sin(ang)
          re += specRe[t][jf] * c - specIm[t][jf] * s
          im += specRe[t][jf] * s + specIm[t][jf] * c
        result[ik * nFreq + jf] = hypot(re, im)

  proc panelVsDft(rows, cols, seed: int, scale, tol: float64) =
    var rng = initRand(seed)
    var g = gather(rows, cols)
    var data = newSeq[seq[float64]](rows)
    for r in 0 ..< rows:
      data[r] = newSeq[float64](cols)
      for j in 0 ..< cols:
        let v = scale * (rng.rand(2.0) - 1.0)
        g.data[r, j] = v
        data[r][j] = v
    let plan = initFkPlan(float64, 2.0, traces = rows, samples = cols,
      taperFraction = 0.1)
    let meta = initChunkMeta(g.rows, 2.0)
    var ws = initWorkspace()
    ws.reserve(plan)
    var amp = newSeq[float64](plan.nK * plan.nFreq)
    doAssert plan.fkAmplitudeInto(g.data, meta, 0, amp, ws) == rows
    let want = directFk(data, 0.1)
    doAssert want.len == amp.len
    var peak = 0.0
    for v in want:
      if v > peak: peak = v
    doAssert peak > 0 and peak < Inf
    var worst = 0.0
    var nInf, nZero = 0
    for i, a in amp:
      if a.classify in {fcInf, fcNegInf}: inc nInf
      if a.classify in {fcZero, fcNegZero}: inc nZero
      if a.classify == fcNormal and want[i].classify == fcNormal:
        worst = max(worst, abs(a - want[i]) / peak)
    doAssert nInf == 0, "scale " & $scale & " has " & $nInf & " infinities"
    doAssert nZero < amp.len, "scale " & $scale & " flushed to " & $nZero & " zeros"
    doAssert worst <= tol, "scale " & $scale & " worst/peak " & $worst

  block fk_squares_overflow:
    # uniform(-1e155, 1e155), 4×8, taper 0.1: squares overflow, modulus was Inf.
    panelVsDft(4, 8, 1, 1e155, 1e-6)

  block fk_squares_underflow:
    # uniform(-1e-160, 1e-160): squares lose digits, was 3e-5 of the peak.
    panelVsDft(4, 8, 1, 1e-160, 1e-6)

  block fk_squares_flush:
    # uniform(-1e-200, 1e-200): squares flush to 0.
    panelVsDft(4, 8, 1, 1e-200, 1e-6)

  block fk_nan_inf_classes:
    # Class strings of a 4×8 taper-0.1 panel, as commit 97632c5 (`hypot`).
    const expect = [
      "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
      "NNNNNNNNNNNNNNNNNNNNINNNNNNNNNNNNNNNNNNN",
      "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
      "INNNINNNNNNNNNNNNNNNINNNINNNNNNNNNNNNNNN",
      "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
      "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"]
    let plan = initFkPlan(float64, 2.0, traces = 4, samples = 8, taperFraction = 0.1)
    var ws = initWorkspace()
    ws.reserve(plan)
    var amp = newSeq[float64](plan.nK * plan.nFreq)
    for kind, want in expect:
      var g = gather(4, 8)
      let meta = initChunkMeta(g.rows, 2.0)
      for r in 0 ..< 4:
        for j in 0 ..< 8:
          g.data[r, j] = case kind
            of 0: NaN
            of 1: Inf
            of 2: (if r == 0: Inf elif r == 1: NaN elif r == 2: -Inf else: 1.0)
            of 3: (if r == 0 and j == 0: Inf else: 1.0)
            of 4: (if r == 0 and j == 0: NaN else: 1.0)
            else: (if j mod 2 == 0: Inf else: NaN)
      discard plan.fkAmplitudeInto(g.data, meta, 0, amp, ws)
      var got = ""
      for a in amp:
        got.add case a.classify
          of fcNan: 'N'
          of fcInf: 'I'
          of fcNegInf: 'i'
          of fcZero, fcNegZero: 'z'
          else: 'f'
      doAssert got == want, "kind " & $kind & " " & got

  block fk_lane_remainder:
    # 7, 9 and 17 traces are not multiples of fftLanes. A dropped tail lane
    # disagrees with the slow DFT of the same panel.
    panelVsDft(7, 16, 2, 1.0, 1e-6)
    panelVsDft(9, 16, 3, 1.0, 1e-6)
    panelVsDft(17, 16, 4, 1.0, 1e-6)

when isMainModule: main()
