## Record stacking (vibroseis/vstack): identical records stack to the record;
## the vertical stack gains 10·log10(N) dB of SNR on white noise; the
## diversity stack suppresses a noise burst in one record while the signal
## keeps its amplitude; floor and clip; any order; meta, row maps and counts;
## merge; bad arguments; float32 and float64; no allocation after reserve.

import std/[algorithm, assertions, math, options, random, sequtils, strformat]
import processSEG
import utils/[allocs, synth]

const dt = 2.0

template ulp(T: typedesc): float64 =
  ## The machine epsilon of `T`.
  (when T is float32: 1.1920929e-7 else: 2.220446049250313e-16)

proc records[T](n, rows, cols: int, sigma: float64, seed: int,
    events: openArray[float64] = [300.0, 1300.0, 2300.0], amp = 1.0): seq[Tensor[T]] =
  ## `n` records of `rows × cols`: the same Ricker events (25 Hz, amplitude
  ## `amp`) in every record, plus white noise of `sigma`, new for each record
  ## and row.
  for k in 0..<n:
    var rec = zeros[T](rows, cols)
    let b = rowBlock(rec)
    for r in 0..<rows:
      for c in events: b.row(r).addRicker(dt, c, 25, amp)
      if sigma > 0: b.row(r).addNoise(sigma, int64(seed * 100_000 + k * 1000 + r))
    result.add rec

proc signal(cols: int, events: openArray[float64] = [300.0, 1300.0, 2300.0],
    amp = 1.0): seq[float64] =
  ## The events of `records`, without noise.
  result = newSeq[float64](cols)
  for c in events: result.addRicker(dt, c, 25, amp)

proc amplitude[T](x: Tensor[T], s: openArray[float64], centre: int): float64 =
  ## The least-squares scale of the event `s` at sample `centre` in `x`, over
  ## its ±3 periods (±60 samples) in every row, so the noise averages out.
  var xs, ss = 0.0
  for r in 0..<x.shape[0]:
    for j in centre - 60 .. centre + 60:
      xs += float64(x[r, j]) * s[j]
      ss += s[j] * s[j]
  xs / ss

proc energyOf[T](a, b: Tensor[T], first, last: int): float64 =
  ## Σ (a − b)² over every row, columns `first ..< last`.
  for r in 0..<a.shape[0]:
    for j in first..<last:
      result += (float64(a[r, j]) - float64(b[r, j])) ^ 2

proc stackOf[T](recs: openArray[Tensor[T]], params: RecordStackParams,
    norm = fnFold): Tensor[T] =
  let rows = recs[0].shape[0]
  let cols = recs[0].shape[1]
  var acc = initRecordStack[T](rows, cols, dt, params)
  var ws = initWorkspace()
  ws.reserve(acc.workspaceSize)
  for rec in recs:
    doAssert acc.addRecord(rec, ws) == rows
  result = newTensor[T](rows, cols)
  acc.stackInto(result, norm)

proc residualEnergy[T](x: Tensor[T], s: openArray[float64]): float64 =
  ## Σ (x − s)² over every row.
  for r in 0..<x.shape[0]:
    for j in 0..<x.shape[1]:
      result += (float64(x[r, j]) - s[j]) ^ 2

let diversity = RecordStackParams(mode: rsDiversity, windowMs: 100)

block identical_records_stack_to_the_record:
  proc check(T: typedesc) =
    var one = records[T](1, 4, 1200, 0.1, 7)[0]
    let recs = @[one, one, one, one, one]
    # float32: exact (5·x and its fifth are exact in float64); float64:
    # within the rounding of the sum.
    let v = stackOf(recs, RecordStackParams())
    when T is float32:
      doAssert v == one, "the vertical mean of identical records is the record"
    for weight in [none(DiversityWeight), some(dwInversePower), some(dwInversePowerSq)]:
      var p = diversity
      if weight.isSome: p.weight = weight.get
      let d = (if weight.isSome: stackOf(recs, p) else: v)
      for r in 0..<one.shape[0]:
        for j in 0..<one.shape[1]:
          doAssert abs(float64(d[r, j]) - float64(one[r, j])) <=
            4 * ulp(T) * abs(float64(one[r, j])), $T & " " & $weight
  check(float32)
  check(float64)

block vertical_stack_snr_gain:
  proc check(T: typedesc) =
    const n = 16
    const rows = 24
    const cols = 2000
    let s = signal(cols)
    let recs = records[T](n, rows, cols, 0.3, 11)
    let sigE = float64(rows) * s.foldl(a + b * b, 0.0)
    let single = 10 * log10(sigE / residualEnergy(recs[0], s))
    let stacked = 10 * log10(sigE / residualEnergy(stackOf(recs, RecordStackParams()), s))
    let gain = stacked - single
    echo &"  {$T}: vertical stack of {n}: SNR {single:.2f} → {stacked:.2f} dB, " &
      &"gain {gain:.2f} dB (expected {10 * log10(float64(n)):.2f})"
    doAssert abs(gain - 10 * log10(float64(n))) < 0.5
  check(float32)
  check(float64)

block diversity_suppresses_a_burst:
  proc check(T: typedesc) =
    const n = 8
    const rows = 12
    const cols = 1600
    const burst = 500..800 # samples (1000 .. 1600 ms), over the 1300 ms event
    let s = signal(cols)
    let clean = records[T](n, rows, cols, 0.05, 23)
    var recs = clean.mapIt(it.clone)
    let b = rowBlock(recs[3])
    for r in 0..<rows:
      var noise = newSeq[float64](burst.len)
      noise.addNoise(5.0, int64(900 + r))
      for j in burst: b.row(r)[j] += T(noise[j - burst.a])
    # The leak: what the burst changes in each stack, over the burst.
    let leakV = energyOf(stackOf(recs, RecordStackParams()),
      stackOf(clean, RecordStackParams()), burst.a, burst.b + 1)
    for weight in [dwInversePower, dwInversePowerSq]:
      var p = diversity
      p.weight = weight
      let d = stackOf(recs, p)
      let leakD = energyOf(d, stackOf(clean, p), burst.a, burst.b + 1)
      let suppression = 10 * log10(leakV / leakD)
      var worst = 0.0
      for c in [150, 650, 1150]: # 650 is inside the burst
        worst = max(worst, abs(amplitude(d, s, c) - 1))
      echo &"  {$T} {weight}: burst leak {suppression:.1f} dB below the vertical " &
        &"stack's; event amplitudes within {100 * worst:.2f} %"
      doAssert suppression >= 20
      doAssert worst <= 0.01

  check(float32)
  check(float64)

block diversity_matches_a_direct_reference:
  # The weights from window powers summed directly, the median by sorting:
  # traces shorter than the window, a multiple of it and not, with zeros.
  proc reference(xs: seq[seq[float64]], h: int, params: RecordStackParams,
      cols: int): seq[float64] =
    var num, den = newSeq[float64](cols)
    for x in xs:
      let n = x.len
      var pow = newSeq[float64](n)
      for i in 0..<n:
        let a = max(0, i - h)
        let b = min(n - 1, i + h)
        for k in a..b: pow[i] += x[k] * x[k]
        pow[i] /= float64(b - a + 1)
      var nz = pow.filterIt(it > 0)
      if nz.len > 0:
        nz.sort
        let med = nz[(nz.len - 1) div 2]
        let lim = (if params.clipLevel > 0: params.clipLevel * sqrt(med) else: Inf)
        for i in 0..<n:
          if pow[i] > 0:
            let q = max(pow[i], params.powerFloor * med)
            let w = (if params.weight == dwInversePower: 1 / q else: 1 / (q * q))
            num[i] += w * clamp(x[i], -lim, lim)
            den[i] += w
    for i in 0..<cols:
      result.add (if den[i] > 0: num[i] / den[i] else: 0.0)
  var rng = initRand(3)
  for (h, lens) in [(2, @[3, 5, 7, 11, 16]), (5, @[11, 22, 40, 41, 4]),
      (20, @[100, 81, 37, 1])]:
    for params in [RecordStackParams(mode: rsDiversity, windowMs: float64(2 * h) * dt,
        powerFloor: 0.3, clipLevel: 2), RecordStackParams(mode: rsDiversity,
        windowMs: float64(2 * h) * dt, weight: dwInversePowerSq, powerFloor: 0)]:
      let cols = max(lens)
      var acc = initRecordStack[float64](1, cols, dt, params)
      doAssert acc.windowSamples == 2 * h + 1
      var ws = initWorkspace()
      ws.reserve(acc.workspaceSize)
      var xs: seq[seq[float64]]
      for n in lens:
        var x = newSeq[float64](n)
        for j in 0..<n:
          x[j] = (if rng.rand(1.0) < 0.2: 0.0 else: rng.gauss(0, 1) * exp(rng.gauss(0, 2)))
        xs.add x
        doAssert acc.add(0, x, ws)
      var got = newTensor[float64](1, cols)
      acc.stackInto(got)
      let want = reference(xs, h, params, cols)
      for j in 0..<cols:
        doAssert abs(got[0, j] - want[j]) <= 1e-9 * max(1.0, abs(want[j])),
          &"h {h}, sample {j}: {got[0, j]} vs {want[j]}"

block power_floor_and_clip:
  const n = 6
  const rows = 4
  const cols = 1200
  # Raw records: the event (amplitude 0.05) is buried in the noise (0.05).
  let s = signal(cols, [1300.0], 0.05)
  # Floor: record 0 drops out (all but silent) over 500 .. 800. Without a
  # floor its tiny power takes over the stack there and the event is lost;
  # with the floor at its median power it weighs about as much as one
  # record of six (a little more: the others' windows also hold the event,
  # and its median is lowered by the dropout itself).
  var recs = records[float64](n, rows, cols, 0.05, 31, [1300.0], 0.05)
  for r in 0..<rows:
    for j in 500..800: recs[0][r, j] = recs[0][r, j] * 1e-4
  var p = diversity
  p.powerFloor = 0
  let noFloor = amplitude(stackOf(recs, p), s, 650)
  p.powerFloor = 1
  let floored = amplitude(stackOf(recs, p), s, 650)
  echo &"  dropout: event amplitude {noFloor:.3f} without a floor, {floored:.3f} with " &
    &"one at the median ({(n - 1) / n:.3f} if it weighed as one record)"
  doAssert noFloor < 0.1, "without a floor the dropout dominates"
  doAssert floored > 0.6 and floored < (n - 1) / n, "the floor keeps the event"
  # Clip: a spike in record 2, much shorter than the window. The weight
  # alone suppresses a big spike (it dominates its window's power) but not
  # one of a few σ·√window, which the clip at 3·√(median power) cuts.
  let quiet = records[float64](n, rows, cols, 0.05, 37, [1300.0], 0.05)
  var spiky = quiet.mapIt(it.clone)
  spiky[2][1, 300] = 1
  p = diversity
  let base = stackOf(quiet, p)[1, 300]
  let unclipped = stackOf(spiky, p)
  p.clipLevel = 3
  let clipped = stackOf(spiky, p)
  let leakU = abs(unclipped[1, 300] - base)
  let leakC = abs(clipped[1, 300] - base)
  echo &"  spike: leak {leakU:.4f} unclipped, {leakC:.4f} clipped at 3"
  doAssert leakC < 0.5 * leakU, "the spike is clipped"
  # Inf is no clip.
  p.clipLevel = Inf
  doAssert stackOf(spiky, p) == unclipped

block any_order:
  proc check(T: typedesc) =
    var recs = records[T](7, 5, 900, 0.2, 41)
    recs[2][1, 400] = T(50) # something for the weights to do
    for params in [RecordStackParams(), diversity]:
      # Rows added one at a time, records and rows shuffled.
      var acc = initRecordStack[T](5, 900, dt, params)
      var ws = initWorkspace()
      ws.reserve(acc.workspaceSize)
      var adds: seq[(int, int)]
      for k in 0..<recs.len:
        for r in 0..<5: adds.add (k, r)
      var rng = initRand(5)
      rng.shuffle(adds)
      for (k, r) in adds:
        doAssert acc.add(r, rowBlock(recs[k]).row(r), ws)
      var shuffled = newTensor[T](5, 900)
      acc.stackInto(shuffled)
      let inOrder = stackOf(recs, params)
      for r in 0..<5:
        for j in 0..<900:
          doAssert abs(float64(shuffled[r, j]) - float64(inOrder[r, j])) <=
            4 * ulp(T) * max(1.0, abs(float64(inOrder[r, j]))), $params.mode
  check(float32)
  check(float64)

block meta_rows_map_and_counts:
  # Two records of 3 rows × 6 samples. Record A: row 1 dead, row 2 live for 4
  # samples. Record B: rows in reverse channel order, row 0 failed.
  var keysA = @[initTraceKeys(), initTraceKeys(), initTraceKeys()]
  keysA[1].status = tsDead
  let countsA = @[6, 6, 4]
  let metaA = initChunkMeta(rowsOf(keysA, countsA), dt)
  var keysB = @[initTraceKeys(), initTraceKeys(), initTraceKeys()]
  let countsB = @[6, 6, 6]
  let failedB = @[0]
  let metaB = initChunkMeta(rowsOf(keysB, countsB), dt, failed = failedB)
  var a = zeros[float32](3, 6)
  var b = zeros[float32](3, 6)
  for j in 0..<6:
    for r in 0..<3:
      a[r, j] = float32(r + 1)
      b[2 - r, j] = float32(10 * (r + 1)) # b's row 2 − r is channel r
  a[2, 4] = 99 # past row 2's live length: never read
  a[2, 5] = 99
  var acc = initRecordStack[float32](3, 6, dt)
  var ws = initWorkspace()
  doAssert acc.addRecord(a, metaA, ws) == 2
  doAssert acc.addRecord(b, metaB, ws, rowMap = [2, 1, 0]) == 2
  var dst = newTensor[float32](3, 6)
  acc.stackInto(dst, fnNone)
  var fold = newTensor[int32](3, 6)
  acc.foldInto(fold)
  for j in 0..<6: # A's row 0 and B's channel 0 (its row 2)
    doAssert dst[0, j] == 11 and fold[0, j] == 2
  for j in 0..<6:
    doAssert dst[1, j] == 20 and fold[1, j] == 1 # A's row 1 dead; B's channel 1
  for j in 0..<4: # A's row 2 (B's channel 2, its row 0, failed)
    doAssert dst[2, j] == 3 and fold[2, j] == 1
  for j in 4..<6:
    doAssert dst[2, j] == 0 and fold[2, j] == 0
  doAssert acc.countAt(2, 3) == 1 and acc.countAt(2, 4) == 0
  # Without a meta, every row is added: B's channel 0 once more, alone.
  doAssert acc.addRecord(b, ws, rowMap = [-1, -1, 0]) == 1
  doAssert acc.countAt(0, 0) == 3
  acc.stackInto(dst)
  doAssert dst[0, 0] == 7

block normalizations:
  for params in [RecordStackParams(), diversity]:
    var acc = initRecordStack[float64](1, 300, dt, params)
    var ws = initWorkspace()
    ws.reserve(acc.workspaceSize)
    var x = newSeq[float64](300)
    for j in 0..<300: x[j] = 1 + float64(j mod 5)
    for k in 0..<4: doAssert acc.add(0, x, ws)
    var mean, total, root = newTensor[float64](1, 300)
    acc.stackInto(mean, fnFold)
    acc.stackInto(total, fnNone)
    acc.stackInto(root, fnSqrtFold)
    for j in 0..<300:
      doAssert abs(mean[0, j] - x[j]) < 1e-12
      doAssert abs(total[0, j] - 4 * x[j]) < 1e-12
      doAssert abs(root[0, j] - 2 * x[j]) < 1e-12

block data_problems_do_not_raise:
  for params in [RecordStackParams(), diversity]:
    var acc = initRecordStack[float32](2, 100, dt, params)
    var ws = initWorkspace()
    ws.reserve(acc.workspaceSize)
    var x = newSeq[float32](100)
    x.addRicker(dt, 100, 25)
    var bad = x
    bad[40] = NaN
    doAssert not acc.add(0, bad, ws)
    bad[40] = Inf
    doAssert not acc.add(0, bad, ws)
    doAssert not acc.add(-1, x, ws)
    doAssert acc.countAt(0, 40) == 0
    doAssert acc.add(0, newSeq[float32](0), ws) # empty: nothing to add
    doAssert acc.add(1, newSeq[float32](100), ws) # all zero
    doAssert acc.countAt(1, 0) == (if params.mode == rsVertical: 1 else: 0)
    doAssert acc.add(0, x.toOpenArray(0, 59), ws) # a short trace
    doAssert acc.countAt(0, 59) == 1 and acc.countAt(0, 60) == 0

block merge_and_clear:
  for params in [RecordStackParams(), diversity]:
    let recs = records[float64](6, 3, 500, 0.2, 53)
    var whole = initRecordStack[float64](3, 500, dt, params)
    var halves = [initRecordStack[float64](3, 500, dt, params),
      initRecordStack[float64](3, 500, dt, params)]
    var ws = initWorkspace()
    ws.reserve(whole.workspaceSize)
    for k, rec in recs:
      discard whole.addRecord(rec, ws)
      discard halves[k mod 2].addRecord(rec, ws)
    halves[0].merge(halves[1])
    var a, b = newTensor[float64](3, 500)
    whole.stackInto(a)
    halves[0].stackInto(b)
    for r in 0..<3:
      for j in 0..<500:
        doAssert abs(a[r, j] - b[r, j]) <= 1e-12 * max(1.0, abs(a[r, j]))
        doAssert whole.countAt(r, j) == halves[0].countAt(r, j)
    whole.clear
    whole.stackInto(a)
    doAssert a == zeros[float64](3, 500) and whole.countAt(1, 1) == 0

block accessors:
  let acc = initRecordStack[float32](3, 400, 4, diversity)
  doAssert acc.rows == 3 and acc.samples == 400 and acc.dtMs == 4
  doAssert acc.params == diversity
  doAssert acc.windowSamples == 27 # h = round(100 / 8) = 13
  doAssert acc.workspaceSize == WorkspaceSize(f64: 1201)
  let v = initRecordStack[float32](3, 400, 4)
  doAssert v.windowSamples == 0 and v.workspaceSize == WorkspaceSize()
  doAssert recordStackMemory(float32, 3, 400) == 3 * 400 * 12
  doAssert recordStackMemory(float64, 3, 400, rsDiversity) == 3 * 400 * 20

block bad_arguments:
  doAssertRaises ValueError: discard initRecordStack[float32](-1, 10, dt)
  doAssertRaises ValueError: discard initRecordStack[float32](1, -10, dt)
  doAssertRaises ValueError: discard initRecordStack[float32](high(int), 10, dt)
  doAssertRaises ValueError: discard initRecordStack[float32](1, 10, 0)
  doAssertRaises ValueError: discard initRecordStack[float32](1, 10, NaN)
  # A vertical stack ignores the diversity options.
  discard initRecordStack[float32](1, 10, dt, RecordStackParams(windowMs: -1))
  for bad in [RecordStackParams(mode: rsDiversity, windowMs: 1.9),
      RecordStackParams(mode: rsDiversity, windowMs: NaN),
      RecordStackParams(mode: rsDiversity, windowMs: 1e300),
      RecordStackParams(mode: rsDiversity, powerFloor: -1),
      RecordStackParams(mode: rsDiversity, powerFloor: Inf),
      RecordStackParams(mode: rsDiversity, clipLevel: -1),
      RecordStackParams(mode: rsDiversity, clipLevel: NaN)]:
    doAssertRaises ValueError: discard initRecordStack[float32](1, 10, dt, bad)
  var acc = initRecordStack[float32](2, 10, dt)
  var ws = initWorkspace()
  doAssertRaises ValueError: discard acc.addRecord(zeros[float32](3, 10), ws)
  doAssertRaises ValueError: discard acc.addRecord(zeros[float32](2, 11), ws)
  doAssertRaises ValueError:
    discard acc.addRecord(zeros[float32](2, 10), ws, rowMap = [0])
  doAssertRaises ValueError:
    discard acc.addRecord(zeros[float32](2, 10), ws, rowMap = [0, 2])
  var keys = @[initTraceKeys()]
  let counts = @[10]
  let meta = initChunkMeta(rowsOf(keys, counts), dt)
  doAssertRaises ValueError: discard acc.addRecord(zeros[float32](2, 10), meta, ws)
  doAssert acc.countAt(0, 0) == 0, "a refused record adds nothing"
  # A record at another interval, or a window of one (its column 0 is not
  # sample 0), does not line up with the stack's samples.
  var keys2 = @[initTraceKeys(), initTraceKeys()]
  let counts2 = @[310, 310]
  let one = ones[float32](2, 10)
  for wrong in [initChunkMeta(rowsOf(keys2, counts2), 2 * dt),
      initChunkMeta(rowsOf(keys2, counts2), dt, firstSample = 300)]:
    doAssertRaises ValueError: discard acc.addRecord(one, wrong, ws)
    doAssertRaises ValueError: discard acc.addRecord(rowBlock(one), wrong, ws)
  doAssert acc.countAt(0, 0) == 0 and acc.countAt(1, 9) == 0
  # The interval within rounding (1e-9, as the filters take it) is the same.
  doAssert acc.addRecord(one, initChunkMeta(rowsOf(keys2, counts2), dt * (1 + 1e-12)), ws) == 2
  acc.clear
  var dst = newTensor[float32](2, 9)
  doAssertRaises ValueError: acc.stackInto(dst)
  var fold = newTensor[int32](3, 10)
  doAssertRaises ValueError: acc.foldInto(fold)
  doAssertRaises ValueError: acc.merge(initRecordStack[float32](2, 10, dt, diversity))
  doAssertRaises ValueError: acc.merge(initRecordStack[float32](2, 10, 4))

block no_allocation_after_reserve:
  proc check(T: typedesc) =
    let recs = records[T](3, 4, 700, 0.1, 61)
    var keys = @[initTraceKeys(), initTraceKeys(), initTraceKeys(), initTraceKeys()]
    let counts = @[700, 700, 500, 700]
    let meta = initChunkMeta(rowsOf(keys, counts), dt)
    let map = @[3, 2, 1, 0]
    for params in [RecordStackParams(), diversity]:
      var acc = initRecordStack[T](4, 700, dt, params)
      var ws = initWorkspace()
      ws.reserve(acc.workspaceSize)
      var dst = newTensor[T](4, 700)
      let b = rowBlock(recs[0])
      let n = allocsIn:
        for rec in recs:
          discard acc.addRecord(rec, ws)
          discard acc.addRecord(rec, meta, ws, map)
        discard acc.add(1, b.row(1), ws)
        acc.stackInto(dst)
      doAssert n == 0, $T & " " & $params.mode & ": " & $n & " allocations"
  check(float32)
  check(float64)

echo "tvstack: ok"
