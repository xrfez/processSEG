## Time-variant spectra: frames equal a direct DFT of the
## windowed samples, a sinusoid on a bin reads its amplitude in every frame,
## a sweep's ridge follows its law and its 2nd harmonic shows 20 dB down,
## float32, refusals and no allocation.

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

when isMainModule: main()
