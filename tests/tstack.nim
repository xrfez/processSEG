## Stacking (stack/stack): flattened synthetics with mutes and short traces
## stack to the reflectivity with the exact per-sample fold, in any order;
## NMO-corrected CMP gathers stack to the zero-offset trace; merge equals
## one accumulator; normalizations, NaN traces, bounds, memory and no
## allocation.

import std/[assertions, math, random]
import processSEG
import utils/[allocs, synth]

const dt = 2.0

block flat_traces_stack_to_reflectivity:
  # 3 bins; each trace is the bin's reflectivity (non-zero everywhere, in
  # 1/8 steps so every sum is exact) cut by a top mute and a live length.
  const bins = 3
  const ns = 200
  var refl: array[bins, seq[float64]]
  for b in 0..<bins:
    refl[b] = newSeq[float64](ns)
    for j in 0..<ns: refl[b][j] = float64((j * (b + 3)) mod 17 + 1) / 8
  type Tr = tuple[bin, mute, live: int]
  var traces: seq[Tr]
  var rng = initRand(11)
  for b in 0..<bins:
    for k in 0..<(5 + 3 * b):
      traces.add (b, rng.rand(0..60), rng.rand(120..ns))
  var expectFold = newSeq[array[ns, int]](bins)
  for t in traces:
    for j in t.mute..<t.live: inc expectFold[t.bin][j]
  proc stackOf(order: seq[Tr]): StackAccumulator[float64] =
    result = initStackAccumulator[float64](bins, ns, dt)
    var x = newSeq[float64](ns)
    for t in order:
      for j in 0..<ns:
        x[j] = (if j >= t.mute and j < t.live: refl[t.bin][j] else: 0)
      doAssert result.addLive(t.bin, x)
  let acc = stackOf(traces)
  var shuffled = traces
  rng.shuffle(shuffled)
  let acc2 = stackOf(shuffled)
  var st = newTensor[float64](bins, ns)
  acc.stackInto(st, fnFold)
  var fold = newTensor[int32](bins, ns)
  acc.foldInto(fold)
  for b in 0..<bins:
    for j in 0..<ns:
      doAssert fold[b, j] == expectFold[b][j], $b & " " & $j
      doAssert acc.foldAt(b, j) == expectFold[b][j]
      doAssert st[b, j] == (if expectFold[b][j] > 0: refl[b][j] else: 0.0)
      # Any order: the sums are exact, so bit-identical.
      doAssert acc2.sumAt(b, j) == acc.sumAt(b, j)
      doAssert acc2.foldAt(b, j) == acc.foldAt(b, j)

block nmo_corrected_cmps_stack_to_zero_offset:
  let v = gradientVelocity(1800, 600)
  let offsets = [100.0, 300, 500, 700, 900, 1100, 1300, 1500]
  let t0s = [400.0, 900, 1500]
  const ns = 1000
  var acc = initStackAccumulator[float32](2, ns, dt)
  var ws = initWorkspace()
  ws.reserve(nmoWorkspace(float32, ns))
  let p = initNmoParams(siCubic, maxStretch = 0.3, taperMs = 20)
  var zeroOffset = cmpGather[float32](0, [0.0], ns, dt)
  for t0 in t0s: zeroOffset.addHyperbola(t0, v.vrmsAt(t0), freqHz = 20)
  for cdp in 0..<2:
    var g = cmpGather[float32](0, offsets, ns, dt, cdp = float64(cdp))
    for t0 in t0s: g.addHyperbola(t0, v.vrmsAt(t0), freqHz = 20)
    let meta = initChunkMeta(g.rows, dt)
    applyNmo(g.data, meta, v, p, ws)
    let b = rowBlock(g.data)
    for r in 0..<g.len:
      doAssert acc.addLive(cdp, b.row(r))
  var st = newTensor[float32](2, ns)
  acc.stackInto(st)
  # The far trace's stretch mute ends near 950 ms, so around the deepest
  # event every trace counts and the stack is the zero-offset trace.
  for b in 0..<2:
    doAssert acc.foldAt(b, 1500 div 2) == offsets.len
    var worst = 0.0
    for j in (1300 div 2)..<ns:
      worst = max(worst, abs(float64(st[b, j]) - float64(zeroOffset.data[0, j])))
    # NMO stretches the far traces' wavelets (by ~10 % at 1500 ms), so the
    # stack is not the zero-offset trace exactly; its peaks are.
    doAssert worst <= 0.08, $worst
    for t0 in [900.0, 1500]:
      var best = int(t0 / dt) - 5
      for j in best .. best + 10:
        if st[b, j] > st[b, best]: best = j
      doAssert abs(float64(best) * dt - t0) <= dt
      # 3 % low at 900 ms: a trace in its mute taper counts at full fold.
      doAssert abs(st[b, best] - 1) <= 0.04, $st[b, best]
    # Shallow, only the near traces count (the stretch mute).
    doAssert acc.foldAt(b, 300 div 2) in 1 ..< offsets.len

block merge_equals_one:
  var rng = initRand(2)
  var one = initStackAccumulator[float64](4, 50, dt, startMs = 100)
  var a = initStackAccumulator[float64](4, 50, dt, startMs = 100)
  var c = initStackAccumulator[float64](4, 50, dt, startMs = 100)
  var x = newSeq[float64](50)
  for i in 0..<40:
    for j in 0..<50: x[j] = float64(rng.rand(-64..64)) / 16
    let bin = rng.rand(0..3)
    let first = rng.rand(0..20)
    let last = rng.rand(30..50)
    doAssert one.add(bin, x, first, last)
    if i mod 2 == 0: doAssert a.add(bin, x, first, last)
    else: doAssert c.add(bin, x, first, last)
  a.merge(c)
  for b in 0..<4:
    for j in 0..<50:
      doAssert a.sumAt(b, j) == one.sumAt(b, j) and a.foldAt(b, j) ==
          one.foldAt(b, j)
  var other = initStackAccumulator[float64](4, 50, dt)
  doAssertRaises ValueError: a.merge(other) # another start time
  doAssert a.startMs == 100 and a.dtMs == dt and a.bins == 4 and a.samples == 50
  a.clear
  doAssert a.sumAt(2, 10) == 0 and a.foldAt(2, 10) == 0

block normalizations:
  var acc = initStackAccumulator[float32](1, 3, dt)
  doAssert acc.add(0, [2'f32, 4, 0], 0, 2)
  doAssert acc.add(0, [2'f32, 4, 0], 0, 1)
  doAssert acc.add(0, [2'f32, 0, 0], 0, 1)
  doAssert acc.add(0, [2'f32, 0, 0], 0, 1)
  var st = newTensor[float32](1, 3)
  acc.stackInto(st, fnNone)
  doAssert st[0, 0] == 8 and st[0, 1] == 4 and st[0, 2] == 0
  acc.stackInto(st, fnFold)
  doAssert st[0, 0] == 2 and st[0, 1] == 4 and st[0, 2] == 0
  acc.stackInto(st, fnSqrtFold)
  doAssert st[0, 0] == 4 and st[0, 1] == 4
  var wrong = newTensor[float32](2, 3)
  doAssertRaises ValueError: acc.stackInto(wrong)
  var wrongFold = newTensor[int32](1, 4)
  doAssertRaises ValueError: acc.foldInto(wrongFold)

block finish_without_a_copy:
  # finishInto gives the stack and fold stackInto and foldInto give, from
  # the accumulator's own storage, for every normalization and both types.
  proc check[T](norm: FoldNorm) =
    var rng = initRand(9)
    var acc = initStackAccumulator[T](5, 40, dt)
    var x = newSeq[T](40)
    for i in 0..<60:
      for j in 0..<40: x[j] = T(rng.rand(-64..64)) / 16
      doAssert acc.add(rng.rand(0..3), x, rng.rand(0..15), rng.rand(20..40))
    var want = newTensor[T](5, 40)
    var wantFold = newTensor[int32](5, 40)
    acc.stackInto(want, norm)
    acc.foldInto(wantFold)
    let copy = acc # a copy owns its sums: finishing one leaves the other
    var data = newTensor[T](1, 1) # replaced, whatever it held
    var fold: Tensor[int32]
    doAssert allocsIn(acc.finishInto(data, fold, norm)) == 0
    doAssert data == want and fold == wantFold
    doAssert data.shape == [5, 40] and wantFold[4, 0] == 0 and data[4, 7] == 0
    doAssert acc.bins == 0 and not acc.add(-1, x)
    when not defined(addressSanitizer):
      doAssertRaises AssertionDefect: discard acc.add(0, x) # no bins left
    doAssert copy.bins == 5 and copy.sumAt(1, 20) != 0
    var again = newTensor[T](5, 40)
    copy.stackInto(again, norm)
    doAssert again == want
    var finished = acc # a finished accumulator still copies and clears
    finished.clear
    doAssert finished.bins == 0
  for norm in [fnNone, fnFold, fnSqrtFold]:
    check[float32](norm)
    check[float64](norm)
  # A copy does not share the original's sums.
  var a = initStackAccumulator[float32](1, 3, dt)
  doAssert a.add(0, [1'f32, 2, 3])
  var b = a
  doAssert b.add(0, [1'f32, 1, 1])
  doAssert a.sumAt(0, 0) == 1 and a.foldAt(0, 0) == 1
  doAssert b.sumAt(0, 0) == 2 and b.foldAt(0, 0) == 2
  var none = initStackAccumulator[float32](0, 3, dt)
  var data: Tensor[float32]
  var fold: Tensor[int32]
  none.finishInto(data, fold)
  doAssert data.size == 0 and fold.size == 0

block refused_traces:
  var acc = initStackAccumulator[float32](2, 4, dt)
  doAssert not acc.add(-1, [1'f32, 1, 1, 1]) # no bin
  doAssert not acc.add(0, [1'f32, NaN, 1, 1])
  doAssert not acc.addLive(0, [0'f32, Inf, 1, 0])
  doAssert acc.add(0, [1'f32, NaN, 1, 1], 2, 4) # NaN outside the span
  doAssert acc.foldAt(0, 1) == 0 and acc.foldAt(0, 2) == 1
  doAssert acc.addLive(1, [0'f32, 0, 0, 0]) and acc.foldAt(1, 0) == 0
  doAssert liveSpan([0'f32, 0, 3, 0, 4, 0]) == 2 .. 4
  doAssert liveSpan([0'f32, 0]).len == 0
  # A Defect leaving a {.raises: [].} proc skips its cleanup, so catching one
  # leaks its message: ASan builds leave these checks out.
  when not defined(addressSanitizer):
    doAssertRaises AssertionDefect: discard acc.add(2, [1'f32])
    doAssertRaises AssertionDefect: discard acc.add(0, [1'f32, 1, 1, 1, 1])
    doAssertRaises AssertionDefect: discard acc.add(0, [1'f32, 1], 1, 3)
  doAssert acc.add(0, [5'f32, 5]) # shorter than the stack: from sample 0
  doAssert acc.sumAt(0, 0) == 5 and acc.foldAt(0, 3) == 1
  # A sample past a bin's end is not the next bin's sample: bin and sample
  # are checked each on its own (foldAt(0, 5) of 2 × 4 read bin 1's sample 1).
  when compileOption("boundChecks") and not defined(addressSanitizer):
    doAssert acc.add(1, [0'f32, 7, 0, 0], 1, 2) and acc.foldAt(1, 1) == 1
    doAssertRaises IndexDefect: discard acc.foldAt(0, 5)
    doAssertRaises IndexDefect: discard acc.sumAt(0, 5)
    doAssertRaises IndexDefect: discard acc.foldAt(0, -1)
    doAssertRaises IndexDefect: discard acc.foldAt(2, 0)
    doAssertRaises IndexDefect: discard acc.sumAt(-1, 0)

block non_finite_samples_refused_anywhere:
  # A NaN or an infinity at any position of the span refuses the trace (the
  # check runs in vector blocks: first, last and odd positions, any length);
  # the largest, the smallest and negative zero are finite.
  proc check[T]() =
    let quiet = T(NaN)
    let payload =
      when T is float32: cast[float32](0x7F80_0001'u32) # a signalling NaN
      else: cast[float64](0xFFF0_0000_0000_0001'u64)
    let largest = # finite (`high` of a float type is the infinity)
      when T is float32: 3.4028234e38'f32
      else: 1.7976931348623157e308
    for n in [1, 2, 7, 8, 9, 31, 32, 33, 100]:
      var acc = initStackAccumulator[T](1, n, dt)
      var x = newSeq[T](n)
      for j in 0..<n: x[j] = T(j mod 5) - 2
      x[0] = T(-0.0)
      x[n - 1] = largest
      if n > 2: x[1] = T(1e-40) # float32: subnormal
      doAssert acc.add(0, x), $n
      for at in 0..<n:
        for bad in [quiet, payload, T(Inf), T(-Inf)]:
          var y = x
          y[at] = bad
          doAssert not acc.add(0, y), $n & " " & $at
          # Outside the span it does not count.
          if at > 0: doAssert acc.add(0, y, 0, at)
          if at < n - 1: doAssert acc.add(0, y, at + 1, n)
          acc.clear
  check[float32]()
  check[float64]()

block validation_memory_allocation:
  doAssertRaises ValueError: discard initStackAccumulator[float32](-1, 10, dt)
  doAssertRaises ValueError: discard initStackAccumulator[float32](1, 10, 0)
  doAssertRaises ValueError: discard initStackAccumulator[float32](1, 10, dt, NaN)
  doAssertRaises ValueError: discard initStackAccumulator[float32](high(
      int) div 2, 10, dt)
  doAssertRaises ValueError: discard initStackAccumulator[float32](1, high(
      int) div 4, dt)
  doAssert stackMemory(float32, 10_000, 2001) == 160_080_000
  doAssert stackMemory(float64, 2, 3) == 72
  var acc = initStackAccumulator[float32](8, 500, dt)
  var x = newSeq[float32](500)
  for j in 100..<400: x[j] = 1
  var added: bool
  doAssert allocsIn((added = acc.addLive(3, x))) == 0 and added
  var other = initStackAccumulator[float32](8, 500, dt)
  doAssert allocsIn(acc.merge(other)) == 0
  var st = newTensor[float32](8, 500)
  doAssert allocsIn(acc.stackInto(st)) == 0
