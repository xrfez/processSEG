## Time-variant spectra: frames equal a direct DFT of the
## windowed samples, a sinusoid on a bin reads its amplitude in every frame,
## a sweep's ridge follows its law and its 2nd harmonic shows 20 dB down,
## float32, refusals and no allocation.
##
## Fail-first: `average_stop` failed while the stop check was compiled out
## (`if false and stop`), so a never-true check was called 0 times instead
## of once per live row, and while a finished call returned true instead of
## the rows used and an abandoned call returned false instead of `Stopped`.
## `float32_large_stays_finite` failed while the magnitude was squared in
## float32 (`sqrt(re * re + im * im)` with no float64 cast): a 1e30 tone
## read Inf. `float32_magnitude_matches_a_direct_dft` failed while the
## magnitude tail was left unread, so the last bin was 0.
## `average_spectrogram` failed while `amplitudesFinite` always returned
## true: the NaN row was used, so 3 rows were used.
## `float64_extreme_magnitude` failed while the magnitude was squared in
## float64: constant 1e200 read Inf at bin 0 (reference 1e200). It failed
## again while an extreme float64 row was squared into the average: rows
## 1e200 and 1.0 used 2 and the average was Inf, and a single 1e200 row
## was counted.

import std/[assertions, math, random]
import processSEG
import utils/[allocs, synth]

proc main*() =
  block frames_equal_a_direct_dft:
    # Random data; the frames at both ends are cut by the trace's ends.
    var rng = initRand(4)
    let n = 300
    var x = newSeq[float64](n)
    for v in x.mitems: v = rng.gauss(0, 1)
    let sg = initSpectrogramPlan(float64, windowLen = 41, hop = 7, dtMs = 2, samples = n)
    doAssert sg.frames == 43 and sg.nfft == 64 and sg.bins == 33 and sg.frameCol(3) == 21
    var ws = initWorkspace()
    ws.reserve(sg.workspaceSize)
    var amp = newSeq[float64](sg.frames * sg.bins)
    sg.spectrogramInto(x, amp, ws)
    for i in 0 ..< sg.frames:
      let start = i * 7 - 20
      var sw = 0.0
      for j in 0 ..< 41:
        if start + j in 0 ..< n: sw += sin(PI * float64(j + 1) / 42) ^ 2
      for k in 0 ..< sg.bins:
        var re, im = 0.0
        for j in 0 ..< 41:
          let c = start + j
          if c notin 0 ..< n: continue
          let w = sin(PI * float64(j + 1) / 42) ^ 2
          let a = -2 * PI * float64(k * j) / 64
          re += w * x[c] * cos(a)
          im += w * x[c] * sin(a)
        let side = (if k == 0 or k == 32: 1.0 else: 2.0)
        let want = hypot(re, im) * side / sw
        doAssert abs(amp[i * sg.bins + k] - want) < 1e-10 * max(1.0, want), $i & " " & $k

  block sinusoid_on_a_bin:
    # Amplitude 3 at bin 20 of a 128-point frame at 2 ms (78.125 Hz): 3 in the
    # frames inside the trace (to the mirror image's leakage) and close to 3 in
    # the frames its ends cut.
    let n = 2000
    let sg = initSpectrogramPlan(float64, windowLen = 127, hop = 25, dtMs = 2, samples = n,
      nfft = 128)
    let f = sg.binHz(20)
    doAssert abs(f - 78.125) < 1e-12
    var x = newSeq[float64](n)
    for j in 0 ..< n: x[j] = 3 * sin(2 * PI * f * float64(j) * 0.002 + 0.4)
    var ws = initWorkspace()
    ws.reserve(sg.workspaceSize)
    var amp = newSeq[float64](sg.frames * sg.bins)
    sg.spectrogramInto(x, amp, ws)
    for i in 0 ..< sg.frames:
      let a = amp[i * sg.bins + 20]
      let c = sg.frameCol(i)
      if c >= 63 and c + 63 < n: doAssert abs(a - 3) < 3e-4, $i & ": " & $a
      else: doAssert abs(a - 3) < 0.15, $i & ": " & $a

  block sweep_ridge_and_harmonic:
    # A linear sweep at 1 ms plus a 2nd harmonic of 10 %: in each frame inside
    # it, the largest bin is within one bin of the law at the frame's time,
    # and the largest near twice that is 20 dB down (within the Hann window's
    # scalloping, 1.4 dB).
    let spec = linearSweep(10, 90, 6000, 1, initSweepTaper(200, 200))
    let n = spec.sweepLen
    var x = spec.sweepSamples(float64)
    for j in 0 ..< n:
      let t = float64(j)
      x[j] += 0.1 * (if t < 200: taperWeight(tpCos2, t / 200) elif t > 5800:
        taperWeight(tpCos2, (6000 - t) / 200) else: 1.0) * sin(2 * spec.phaseAt(t))
    let sg = initSpectrogramPlan(float64, windowLen = 255, hop = 50, dtMs = 1, samples = n)
    var ws = initWorkspace()
    ws.reserve(sg.workspaceSize)
    var amp = newSeq[float64](sg.frames * sg.bins)
    sg.spectrogramInto(x, amp, ws)
    let binW = sg.binHz(1)
    var seen = 0
    for i in 0 ..< sg.frames:
      let t = float64(sg.frameCol(i))
      if t < 400 or t > 5600: continue
      var best = 0
      for k in 1 ..< sg.bins:
        if amp[i * sg.bins + k] > amp[i * sg.bins + best]: best = k
      let f = spec.frequencyAt(t)
      doAssert abs(sg.binHz(best) - f) <= binW, $t & ": " & $sg.binHz(best) & " vs " & $f
      let k2 = int(round(2 * f / binW))
      var h2 = 0.0
      for k in k2 - 1 .. k2 + 1: h2 = max(h2, amp[i * sg.bins + k])
      let db = 20 * log10(h2 / amp[i * sg.bins + best])
      doAssert abs(db + 20) < 1.5, $t & ": " & $db
      inc seen
    doAssert seen > 90

  block float32_and_short_traces:
    let sg = initSpectrogramPlan(float32, windowLen = 64, hop = 16, dtMs = 1, samples = 1000)
    var ws = initWorkspace()
    ws.reserve(sg.workspaceSize)
    var x = newSeq[float32](600) # shorter than the plan: zeros after it
    for j in 0 ..< x.len: x[j] = float32(sin(2 * PI * 125 * float64(j) / 1000))
    var amp = newSeq[float64](sg.frames * sg.bins)
    sg.spectrogramInto(x, amp, ws)
    doAssert abs(amp[10 * sg.bins + 8] - 1) < 1e-3 # 125 Hz is bin 8 of 64 at 1 ms
    doAssert amp[(sg.frames - 1) * sg.bins + 8] == 0 # a frame past the trace
    var bad = x
    bad[160] = NaN
    sg.spectrogramInto(bad, amp, ws)
    doAssert amp[10 * sg.bins + 8].isNaN and not amp[20 * sg.bins + 8].isNaN

  block refusals_and_shapes:
    doAssertRaises(ValueError): discard initSpectrogramPlan(float64, 1, 10, 1, 100)
    doAssertRaises(ValueError): discard initSpectrogramPlan(float64, 64, 0, 1, 100)
    doAssertRaises(ValueError): discard initSpectrogramPlan(float64, 64, 10, 0, 100)
    doAssertRaises(ValueError): discard initSpectrogramPlan(float64, 64, 10, 1, 0)
    doAssertRaises(ValueError): discard initSpectrogramPlan(float64, 64, 10, 1, 100, nfft = 32)
    doAssertRaises(ValueError): discard initSpectrogramPlan(float64, 64, 10, 1, 100, nfft = 77)
    let sg = initSpectrogramPlan(float64, 64, 10, 1, 100)
    var ws = initWorkspace()
    ws.reserve(sg.workspaceSize)
    var x = newSeq[float64](101)
    var amp = newSeq[float64](sg.frames * sg.bins)
    doAssertRaises(AssertionDefect): sg.spectrogramInto(x, amp, ws)
    var y = newSeq[float64](100)
    var short = newSeq[float64](3)
    doAssertRaises(AssertionDefect): sg.spectrogramInto(y, short, ws)

  block no_allocation:
    let sg = initSpectrogramPlan(float32, windowLen = 255, hop = 10, dtMs = 1, samples = 8000)
    var ws = initWorkspace()
    ws.reserve(sg.workspaceSize)
    var x = newSeq[float32](8000)
    for j in 0 ..< x.len: x[j] = float32(sin(float64(j) * 0.2))
    var amp = newSeq[float64](sg.frames * sg.bins)
    sg.spectrogramInto(x, amp, ws)
    let count = allocsIn:
      sg.spectrogramInto(x, amp, ws)
    doAssert count == 0, $count & " allocations"

  block average_spectrogram:
    # Fail-first: a NaN frame was not dropped, so 3 rows were used.
    let sg = initSpectrogramPlan(float64, windowLen = 32, hop = 16, dtMs = 2, samples = 128,
      nfft = 32)
    var g = cmpGather[float64](0, newSeq[float64](3), 128, 2)
    let f = sg.binHz(4)
    for r in 0 ..< 3:
      for j in 0 ..< 128:
        g.data[r, j] = cos(2 * PI * f * float64(j) * 2 / 1000) * 2
    g.data[2, 10] = NaN
    let meta = initChunkMeta(g.rows, dtMs = 2)
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var amp = newSeq[float64](sg.frames * sg.bins)
    let used = sg.averageSpectrogramInto(g.data, meta, amp, ws)
    doAssert used == 2, "a NaN frame drops the trace, got " & $used
    let mid = sg.frames div 2
    doAssert abs(amp[mid * sg.bins + 4] - 2.0) < 1e-3, $amp[mid * sg.bins + 4]
    let quiet = allocsIn:
      discard sg.averageSpectrogramInto(g.data, meta, amp, ws)
    doAssert quiet == 0

  block average_nonfinite:
    # Inf and -Inf in a covered sample drop that row, as a NaN does.
    # Fail-first: a float64 NaN or Inf between frames was kept, and a
    # subnormal row was used, because only the amplitudes were checked.
    # The live input samples are checked now, so both are left out.
    let sg = initSpectrogramPlan(float64, windowLen = 32, hop = 16, dtMs = 2.0,
      samples = 128, nfft = 32)
    let f = sg.binHz(4)
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var amp = newSeq[float64](sg.frames * sg.bins)
    let mid = sg.frames div 2
    for bad in [Inf, -Inf]:
      var g = cmpGather[float64](0, newSeq[float64](2), 128, 2)
      for r in 0 ..< 2:
        for j in 0 ..< 128:
          g.data[r, j] = cos(2 * PI * f * float64(j) * 2 / 1000) * 2
      g.data[1, 20] = bad
      let meta = initChunkMeta(g.rows, dtMs = 2)
      let used = sg.averageSpectrogramInto(g.data, meta, amp, ws)
      doAssert used == 1, $bad & " used " & $used
      doAssert abs(amp[mid * sg.bins + 4] - 2.0) < 1e-3, $bad & " " &
        $amp[mid * sg.bins + 4]
    let sg32 = initSpectrogramPlan(float32, windowLen = 32, hop = 16,
      dtMs = 2.0, samples = 128, nfft = 32)
    var g32 = cmpGather[float32](0, newSeq[float64](2), 128, 2)
    for r in 0 ..< 2:
      for j in 0 ..< 128:
        g32.data[r, j] = float32(cos(2 * PI * f * float64(j) * 2 / 1000) * 2)
    g32.data[0, 20] = Inf
    var ws32 = initWorkspace()
    ws32.reserve(sg32.averageWorkspace)
    let meta32 = initChunkMeta(g32.rows, dtMs = 2)
    let used32 = sg32.averageSpectrogramInto(g32.data, meta32, amp, ws32)
    doAssert used32 == 1, "float32 Inf used " & $used32
    let gap = initSpectrogramPlan(float64, windowLen = 2, hop = 10, dtMs = 1.0,
      samples = 40, nfft = 4)
    var gg = cmpGather[float64](0, newSeq[float64](1), 40, 1)
    for j in 0 ..< 40: gg.data[0, j] = 0.25
    var wsg = initWorkspace()
    wsg.reserve(gap.averageWorkspace)
    var ag = newSeq[float64](gap.frames * gap.bins)
    let mg = initChunkMeta(gg.rows, dtMs = 1)
    for hole in [NaN, Inf, -Inf]:
      gg.data[0, 5] = hole
      let droppedGap = gap.averageSpectrogramInto(gg.data, mg, ag, wsg)
      doAssert droppedGap == 0, "gap " & $hole & " used " & $droppedGap
      for v in ag: doAssert v == 0.0, "gap " & $hole & " amp " & $v
    gg.data[0, 5] = 0.25
    for covered in [NaN, Inf, -Inf]:
      gg.data[0, 0] = covered
      let dropped = gap.averageSpectrogramInto(gg.data, mg, ag, wsg)
      doAssert dropped == 0, "covered " & $covered & " used " & $dropped
    for j in 0 ..< 40: gg.data[0, j] = 1e-320
    doAssert gap.averageSpectrogramInto(gg.data, mg, ag, wsg) == 0
    for v in ag: doAssert v == 0.0, "subnormal amp " & $v

  block average_stop:
    # Fail-first: failed while the stop check was compiled out, seen 0.
    const rows = 4
    const cols = 128
    let sg = initSpectrogramPlan(float64, windowLen = 32, hop = 16, dtMs = 2, samples = cols,
      nfft = 32)
    var g = cmpGather[float64](0, newSeq[float64](rows), cols, 2)
    let f = sg.binHz(4)
    for r in 0 ..< rows:
      for j in 0 ..< cols:
        g.data[r, j] = cos(2 * PI * f * float64(j) * 2 / 1000) * float64(r + 1)
    let meta = initChunkMeta(g.rows, dtMs = 2)
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var amp = newSeq[float64](sg.frames * sg.bins)
    let used = sg.averageSpectrogramInto(g.data, meta, amp, ws)
    doAssert used == rows, "rows used " & $used
    let full = amp
    var again = newSeq[float64](amp.len)
    doAssert sg.averageSpectrogramInto(g.data, meta, again, ws, stop = nil) == used
    doAssert again == full
    var seen = 0
    let never: StopCheck = proc (): bool {.gcsafe, raises: [].} =
      inc seen
      false
    doAssert sg.averageSpectrogramInto(g.data, meta, again, ws, stop = never) == used
    doAssert again == full
    doAssert seen == rows, "checked once per live row, got " & $seen
    proc stoppedAfter(k: int): int =
      var calls = 0
      let stop: StopCheck = proc (): bool {.gcsafe, raises: [].} =
        inc calls
        calls >= k
      doAssert sg.averageSpectrogramInto(g.data, meta, amp, ws, stop = stop) == Stopped
      doAssert calls <= k, "called " & $calls & " times, at most " & $k
      calls
    doAssert stoppedAfter(1) == 1
    doAssert stoppedAfter(3) == 3
    var once = 0
    let stopRow: StopCheck = proc (): bool {.gcsafe, raises: [].} =
      inc once
      true
    doAssert sg.averageSpectrogramInto(rowBlock(g.data), meta, amp, ws,
      stop = stopRow) == Stopped
    doAssert once == 1
    let quiet = allocsIn:
      discard sg.averageSpectrogramInto(g.data, meta, amp, ws)
    doAssert quiet == 0, $quiet & " allocations"
    doAssert amp == full, "the plan and workspace are reusable after a stop"

  block float32_large_stays_finite:
    # A constant 1e30: the float32 FFT stays finite, and so must the magnitude
    # (the product is float64). Squaring in float32 overflows to Inf.
    let sg = initSpectrogramPlan(float32, windowLen = 32, hop = 16, dtMs = 1.0,
      samples = 128, nfft = 32)
    var x = newSeq[float32](128)
    for j in 0 ..< x.len: x[j] = 1e30
    var ws = initWorkspace()
    ws.reserve(sg.workspaceSize)
    var amp = newSeq[float64](sg.frames * sg.bins)
    sg.spectrogramInto(x, amp, ws)
    doAssert amp[0] > 1e29 and amp[0] < 1e32, $amp[0]
    for v in amp:
      doAssert v.classify notin {fcNan, fcInf, fcNegInf}, $v

  block float32_magnitude_matches_a_direct_dft:
    # Even nfft: the last bin is the tail. Odd nfft: the last bin is split.
    var rng = initRand(11)
    for nfft in [16, 15]:
      let n = 48
      let win = 11
      var x = newSeq[float32](n)
      for v in x.mitems: v = float32(rng.gauss(0, 1))
      let sg = initSpectrogramPlan(float32, windowLen = win, hop = 4, dtMs = 1.0,
        samples = n, nfft = nfft)
      var ws = initWorkspace()
      ws.reserve(sg.workspaceSize)
      var amp = newSeq[float64](sg.frames * sg.bins)
      sg.spectrogramInto(x, amp, ws)
      let half = win div 2
      for i in 0 ..< sg.frames:
        let start = i * 4 - half
        var sw = 0.0
        for j in 0 ..< win:
          if start + j in 0 ..< n:
            sw += sin(PI * float64(j + 1) / float64(win + 1)) ^ 2
        for k in 0 ..< sg.bins:
          var re, im = 0.0
          for j in 0 ..< win:
            let c = start + j
            if c notin 0 ..< n: continue
            let w = sin(PI * float64(j + 1) / float64(win + 1)) ^ 2
            let ang = -2 * PI * float64(k * j) / float64(nfft)
            re += w * float64(x[c]) * cos(ang)
            im += w * float64(x[c]) * sin(ang)
          let side = if k == 0 or 2 * k == nfft: 1.0 else: 2.0
          let want = if sw > 0: hypot(re, im) * side / sw else: 0.0
          let got = amp[i * sg.bins + k]
          doAssert abs(got - want) < 1e-4 * max(1.0, abs(want)),
            $nfft & " frame " & $i & " bin " & $k & ": " & $got & " vs " & $want

  block float64_extreme_magnitude:
    # Fail-first: an extreme float64 row was squared into the average.
    # Rows [1e200, 1.0] used 2 and the average was Inf; [1e-200, 1.0] used
    # 2 and the tiny row flushed to 0; a single 1e200 row was counted.
    # window = nfft = 16, one frame. Squares of 1e200 overflow; squares of
    # 1e-200 flush. spectrogramInto still measures them with hypot.
    let sg = initSpectrogramPlan(float64, windowLen = 16, hop = 16, dtMs = 1.0,
      samples = 16, nfft = 16)
    doAssert sg.frames == 1 and sg.bins == 9
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    for scale in [1e200, 1e-200]:
      var x = newSeq[float64](16)
      for j in 0 ..< 16: x[j] = scale
      var amp = newSeq[float64](sg.bins)
      sg.spectrogramInto(x, amp, ws)
      let start = -8
      var sw = 0.0
      for j in 0 ..< 16:
        if start + j in 0 ..< 16:
          sw += sin(PI * float64(j + 1) / 17) ^ 2
      for k in 0 ..< sg.bins:
        var re, im = 0.0
        for j in 0 ..< 16:
          let c = start + j
          if c notin 0 ..< 16: continue
          let w = sin(PI * float64(j + 1) / 17) ^ 2
          let ang = -2 * PI * float64(k * j) / 16
          re += w * x[c] * cos(ang)
          im += w * x[c] * sin(ang)
        let side = if k == 0 or k == 8: 1.0 else: 2.0
        let want = hypot(re, im) * side / sw
        doAssert abs(amp[k] - want) < 1e-6 * max(abs(want), 1e-300),
          $scale & " bin " & $k & ": " & $amp[k] & " vs " & $want
      doAssert amp[0].classify == fcNormal and abs(amp[0] - scale) < 1e-6 * scale,
        $scale & " amp[0] " & $amp[0]
    proc spectrum(scale: float64): seq[float64] =
      var x = newSeq[float64](16)
      for j in 0 ..< 16: x[j] = scale
      result = newSeq[float64](sg.bins)
      sg.spectrogramInto(x, result, ws)
    proc averaged(scale: float64): tuple[used: int, amp: seq[float64]] =
      var g = cmpGather[float64](0, newSeq[float64](1), 16, 1.0)
      for j in 0 ..< 16: g.data[0, j] = scale
      result.amp = newSeq[float64](sg.bins)
      for v in result.amp.mitems: v = 7.0
      result.used = sg.averageSpectrogramInto(g.data, initChunkMeta(g.rows, 1.0),
        result.amp, ws)
    let one = spectrum(1.0)
    for bad in [1e200, 1e-200]:
      var g = cmpGather[float64](0, newSeq[float64](2), 16, 1.0)
      for j in 0 ..< 16:
        g.data[0, j] = bad
        g.data[1, j] = 1.0
      var avg = newSeq[float64](sg.bins)
      let used = sg.averageSpectrogramInto(g.data, initChunkMeta(g.rows, 1.0), avg, ws)
      doAssert used == 1, $bad & " used " & $used
      doAssert avg == one, $bad & " average is not the 1.0 spectrogram"
    let dropped = averaged(1e200)
    doAssert dropped.used == 0, "single 1e200 used " & $dropped.used
    for v in dropped.amp: doAssert v == 0.0, "all-dropped amp " & $v
    for scale in [1e140, 1e-100]:
      let want = spectrum(scale)
      let got = averaged(scale)
      doAssert got.used == 1, $scale & " used " & $got.used
      for k in 0 ..< sg.bins:
        let tol = 1e-12 * max(abs(want[k]), 1e-300)
        doAssert abs(got.amp[k] - want[k]) <= tol,
          $scale & " bin " & $k & ": " & $got.amp[k] & " vs " & $want[k]
    let zero = averaged(0.0)
    doAssert zero.used == 1, "zero row used " & $zero.used
    doAssert zero.amp == spectrum(0.0)

  block float32_average_bits:
    # Parent commit 44c73f8, FNV-1a of the float64 bits. float32 must stay
    # the same bits: the range check is float64 only.
    proc checksum(amp: openArray[float64]): uint64 =
      result = 0xcbf29ce484222325'u64
      for v in amp:
        result = (result xor cast[uint64](v)) * 0x100000001b3'u64
    let sg = initSpectrogramPlan(float32, windowLen = 32, hop = 16, dtMs = 2.0,
      samples = 128, nfft = 32)
    var g = cmpGather[float32](0, newSeq[float64](3), 128, 2.0)
    let f = sg.binHz(4)
    for r in 0 ..< 3:
      for j in 0 ..< 128:
        g.data[r, j] = float32(cos(2 * PI * f * float64(j) * 2 / 1000) *
          float64(r + 1))
    var ws = initWorkspace()
    ws.reserve(sg.averageWorkspace)
    var amp = newSeq[float64](sg.frames * sg.bins)
    let meta = initChunkMeta(g.rows, 2.0)
    doAssert sg.averageSpectrogramInto(g.data, meta, amp, ws) == 3
    doAssert checksum(amp) == 0x1473953EDF002EB2'u64, $checksum(amp)
    g.data[1, 20] = Inf
    doAssert sg.averageSpectrogramInto(g.data, meta, amp, ws) == 2
    doAssert checksum(amp) == 0xA05F6A671BF1BB29'u64, $checksum(amp)

  block float64_nan_inf_classes:
    # Class strings as commit 97632c5 (`hypot`), window 16, hop 8, 32 samples.
    const expect = [
      "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
      "INNNNNNNNINNNNNNNNINNNNNNNNINNNNNNNN",
      "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
      "INNNNNNNIINNNNNNNIffffffffzffffffffz",
      "NNNNNNNNNNNNNNNNNNffffffffzffffffffz"]
    let sg = initSpectrogramPlan(float64, windowLen = 16, hop = 8, dtMs = 1.0,
      samples = 32, nfft = 16)
    var ws = initWorkspace()
    ws.reserve(sg.workspaceSize)
    var amp = newSeq[float64](sg.frames * sg.bins)
    doAssert amp.len == 36
    for kind, want in expect:
      var x = newSeq[float64](32)
      for j in 0 ..< 32:
        x[j] = case kind
          of 0: NaN
          of 1: Inf
          of 2: (if j mod 2 == 0: Inf else: NaN)
          of 3: (if j == 0: Inf else: 1.0)
          else: (if j == 3: NaN else: 0.5)
      sg.spectrogramInto(x, amp, ws)
      var got = ""
      for a in amp:
        got.add case a.classify
          of fcNan: 'N'
          of fcInf: 'I'
          of fcNegInf: 'i'
          of fcZero, fcNegZero: 'z'
          else: 'f'
      doAssert got == want, "kind " & $kind & " " & got

when isMainModule: main()
