## Stacking repeated records (`RecordStack`): the sweeps shot at one source
## point, summed channel to channel into one record, before or after
## correlation. The caller groups them: it adds rows of its record tensors
## to output rows, in any order, and no file is read here.
##
## ```nim
## var acc = initRecordStack[float32](rows = 240, samples = 4001, dtMs = 2,
##   RecordStackParams(mode: rsDiversity, windowMs: 200))
## var ws = initWorkspace()
## ws.reserve(acc.workspaceSize)
## for rec in sweeps:                          # the records, in any order
##   discard acc.addRecord(rec.samples, rec.meta, ws)   # row r → output row r
## var stacked = newTensor[float32](acc.rows, acc.samples)
## acc.stackInto(stacked)                      # the (weighted) mean, per sample
## ```
##
## **Grouping.** `add(outRow, x, ws)` adds one trace to an output row;
## `addRecord` adds the rows of a record, row `r` to output row `rowMap[r]`
## (`-1`: not added; an empty map: row `r` to row `r`), so channels that come
## in another order in some sweeps are matched by a map the caller builds
## from its keys. With a `ChunkMeta`, dead, auxiliary and failed rows are not
## added and a row adds only its live samples (`liveSamples`).
##
## **Count.** Each output sample counts the traces that reached it
## (`countAt`, `foldInto`): a short trace counts only where it has samples, a
## record that was not live not at all.
##
## **Vertical stack** (`rsVertical`): the sum of the traces, sample by
## sample. `stackInto` gives the mean (`fnFold`, the default), the sum
## (`fnNone`) or the sum / √count (`fnSqrtFold`). White noise falls by
## √N against the signal: 10·log10(N) dB of SNR for N records.
##
## **Diversity stack** (`rsDiversity`): each trace is weighted, sample by
## sample, by the inverse of its own power in a centred window of `2h + 1`
## samples, `h = round(windowMs / (2·dtMs))` (cut at the trace's ends, like
## AGC's), and the stack is `Σ w·x / Σ w`. A sweep hit by a noise burst
## (traffic, wind) has a large power there and hardly counts, while the
## signal, the same in every sweep, keeps its amplitude. Options:
## - `weight`: `dwInversePower`, `w = 1/p` (the classic), or
##   `dwInversePowerSq`, `w = 1/p²` (suppresses harder);
## - `powerFloor`: `p` is at least `powerFloor ×` the median of the trace's
##   non-zero window powers, so a quiet stretch (a ramp-up, a near-mute)
##   does not take over the stack. 0: no floor;
## - `clipLevel`: a sample above `clipLevel × √(that median)` is clipped to
##   it before it is added, against spikes shorter than the window; the
##   weights still come from the unclipped samples. 0 or Inf: no clip.
##
## A sample whose window is all zero (a muted stretch) gets weight 0 and is
## not counted. `fnFold` gives the weighted mean, `fnNone` the weighted mean
## × count and `fnSqrtFold` × √count, on the vertical stack's scales.
##
## **Precision.** Sums, weights and window powers are float64, whatever `T`.
## The window sums are built without subtraction (as AGC's: per block of
## `2h + 1` samples, a suffix sum of one block plus a prefix sum of the
## next), so a quiet window next to a loud burst keeps its accuracy. With
## `dwInversePowerSq`, float64 samples must stay within about 1e±75 (`p²`
## must fit float64); every float32 value does.
##
## **Data problems.** A trace with a NaN or infinite sample is not added
## (`add` returns false, `addRecord` does not count it); nothing raises.
##
## **Memory.** `rows × samples × 12` bytes (a float64 sum and an int32
## count), 20 for a diversity stack (also the sum of weights). A
## `Workspace` of `workspaceSize` (3 float64 per sample, diversity only)
## serves every add: after `reserve`, adding allocates nothing.
##
## **Threads.** A stack is not shared: give each thread its own and `merge`
## them at the end.

import std/math
import parseSEG
import ../core/[rows, types, workspace]
import ../stack/stack

type
  RecordStackMode* = enum
    ## How the traces of a group are combined.
    rsVertical  ## the plain sum, every trace weight 1
    rsDiversity ## weighted by the inverse of the trace's window power

  DiversityWeight* = enum
    ## The weight of a diversity stack, from the window power `p`.
    dwInversePower   ## `1/p`
    dwInversePowerSq ## `1/p²`

  RecordStackParams* = object
    ## What a `RecordStack` does; see the module documentation. The window,
    ## weight, floor and clip apply to `rsDiversity` only.
    mode*: RecordStackMode = rsVertical
    windowMs*: float64 = 128 ## The power window, centred on each sample.
    weight*: DiversityWeight = dwInversePower
    powerFloor*: float64 = 0.01 ## Relative to the trace's median window power; 0: none.
    clipLevel*: float64 = 0 ## Relative to √(median window power); 0 or Inf: none.

  RecordStack*[T: SampleType] = object
    ## The sums and per-sample counts of `rows` output traces of `samples`
    ## samples; see the module documentation. Build it with
    ## `initRecordStack`.
    rows, samples: int
    dtMs: float64
    params: RecordStackParams
    half: int # h, diversity only
    num: seq[float64] # Σ w·x
    den: seq[float64] # Σ w (diversity only; empty for a vertical stack)
    count: seq[int32]

  F64 = ptr UncheckedArray[float64]
  I32 = ptr UncheckedArray[int32]

proc badArg(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func recordStackMemory*(rows, samples: int, mode = rsVertical): int =
  ## The bytes a stack of `rows × samples` holds: 12 per sample (20 for
  ## `rsDiversity`), whatever the sample type (the sums are float64).
  rows * samples * (if mode == rsDiversity: 20 else: 12)

proc initRecordStack*[T: SampleType](rows, samples: int, dtMs: float64,
    params = RecordStackParams()): RecordStack[T] =
  ## An empty stack of `rows` traces of `samples` samples at `dtMs` ms.
  ## Allocates `recordStackMemory(rows, samples, params.mode)` bytes,
  ## zeroed. Raises `ValueError` unless `rows` and `samples` are ≥ 0 (and
  ## their product does not overflow) and `dtMs` is finite and positive, and,
  ## for `rsDiversity`, unless `windowMs` is finite and at least about 3
  ## samples (`h ≥ 1`), `powerFloor` finite and ≥ 0 and `clipLevel` ≥ 0.
  if rows < 0 or samples < 0:
    badArg("rows and samples must be ≥ 0, got " & $rows & " × " & $samples)
  if samples > 0 and rows > high(int) div 20 div samples:
    badArg("a stack of " & $rows & " × " & $samples & " samples is too large")
  if dtMs.classify != fcNormal or dtMs <= 0:
    badArg("dtMs must be finite and positive, got " & $dtMs)
  var half = 0
  if params.mode == rsDiversity:
    if params.windowMs.classify notin {fcNormal, fcSubnormal} or params.windowMs <= 0:
      badArg("windowMs must be finite and positive, got " & $params.windowMs)
    let h = round(params.windowMs / (2 * dtMs))
    if h < 1:
      badArg("a diversity window of " & $params.windowMs & " ms is under 3 samples at " &
        $dtMs & " ms")
    if h > float64(high(int32)):
      badArg("a diversity window of " & $params.windowMs & " ms is too long")
    half = int(h)
    if params.powerFloor.classify in {fcNan, fcInf, fcNegInf} or params.powerFloor < 0:
      badArg("powerFloor must be finite and ≥ 0, got " & $params.powerFloor)
    if params.clipLevel.classify == fcNan or params.clipLevel < 0:
      badArg("clipLevel must be ≥ 0 (0 or Inf: no clip), got " & $params.clipLevel)
  let n = rows * samples
  RecordStack[T](rows: rows, samples: samples, dtMs: dtMs, params: params,
    half: half, num: newSeq[float64](n),
    den: (if params.mode == rsDiversity: newSeq[float64](n) else: @[]),
    count: newSeq[int32](n))

func rows*[T](acc: RecordStack[T]): int {.inline, raises: [].} =
  ## The number of output rows.
  acc.rows

func samples*[T](acc: RecordStack[T]): int {.inline, raises: [].} =
  ## The samples of each output row.
  acc.samples

func dtMs*[T](acc: RecordStack[T]): float64 {.inline, raises: [].} =
  ## The sample interval, in ms.
  acc.dtMs

func params*[T](acc: RecordStack[T]): RecordStackParams {.inline, raises: [].} =
  ## What the stack does.
  acc.params

func windowSamples*[T](acc: RecordStack[T]): int {.inline, raises: [].} =
  ## The diversity window, `2h + 1` samples (0 for a vertical stack).
  (if acc.params.mode == rsDiversity: 2 * acc.half + 1 else: 0)

func workspaceSize*[T](acc: RecordStack[T]): WorkspaceSize {.raises: [].} =
  ## The scratch of `add`: `3·samples + 1` float64 for a diversity stack,
  ## none for a vertical one.
  (if acc.params.mode == rsDiversity: WorkspaceSize(f64: 3 * acc.samples + 1)
    else: WorkspaceSize())

func countAt*[T](acc: RecordStack[T], row, sample: int): int {.inline, raises: [].} =
  ## The traces that reached sample `sample` of output row `row`. An
  ## `IndexDefect` outside the stack, with bound checks on.
  int(acc.count[row * acc.samples + sample])

# ------------------------------------------------------------------ kernels --

func allFinite[T](x: ptr UncheckedArray[T], n: int): bool {.inline.} =
  # x − x is 0 for finite x and NaN otherwise; one sum checks the trace.
  var s = 0.0
  {.push boundChecks: off.}
  for j in 0..<n:
    s += float64(x[j]) - float64(x[j])
  {.pop.}
  s == 0

proc addVertical[T](num: F64, cnt: I32, x: ptr UncheckedArray[T],
    n: int) {.noinline, raises: [].} =
  ## `num[j] += x[j]`, `cnt[j] += 1` for `j < n`.
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0..<n:
    num[j] = num[j] + float64(x[j])
    cnt[j] = cnt[j] + 1
  {.pop.}

proc addWeighted[T](num, den: F64, cnt: I32, x: ptr UncheckedArray[T], w: F64,
    lim: float64, n: int) {.noinline, raises: [].} =
  ## `num[j] += w[j]·clamp(x[j], ±lim)`, `den[j] += w[j]`, and `cnt[j]` counts
  ## the samples of non-zero weight, for `j < n`.
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0..<n:
    let v = min(max(float64(x[j]), -lim), lim)
    num[j] = num[j] + w[j] * v
    den[j] = den[j] + w[j]
    cnt[j] = cnt[j] + int32(w[j] > 0)
  {.pop.}

proc windowPowers[T](x: ptr UncheckedArray[T], n, h: int, s, p, pow: F64) {.raises: [].} =
  ## `pow[i]`, the mean of `x²` over `max(0, i − h) .. min(n − 1, i + h)`.
  ## Per block of `w = 2h + 1` samples, `p[k]` is the sum from the block's
  ## start through `k` and `s[k]` from `k` through the block's end (or
  ## `n − 1`): a window spans at most two blocks, so its sum adds two of
  ## them, never subtracting.
  {.push boundChecks: off, overflowChecks: off.}
  let w = 2 * h + 1
  var bs = 0
  while bs < n:
    let be = min(bs + w, n)
    var fwd, bwd = 0.0
    for k in 0..<be - bs:
      let f = float64(x[bs + k])
      let b = float64(x[be - 1 - k])
      fwd += f * f
      p[bs + k] = fwd
      bwd += b * b
      s[be - 1 - k] = bwd
    bs = be
  var aStart, bStart = 0 # the starts of the blocks of a and b
  for i in 0..<n:
    let a = max(0, i - h)
    let b = min(n - 1, i + h)
    while aStart + w <= a: aStart += w
    while bStart + w <= b: bStart += w
    # One block: the window starts it (the head, or a whole block) or runs
    # to its end (the tail, b = n − 1).
    let sum =
      if aStart != bStart: s[a] + p[b]
      elif a == aStart: p[b]
      else: s[a]
    pow[i] = sum / float64(b - a + 1)
  {.pop.}

proc lowerMedian(a: F64, n: int): float64 {.raises: [].} =
  ## The `(n − 1) div 2`-th smallest of `a[0 ..< n]` (`n > 0`), reordering
  ## `a`: a quickselect, median of three pivots.
  {.push boundChecks: off, overflowChecks: off.}
  let k = (n - 1) div 2
  var lo = 0
  var hi = n - 1
  while lo < hi:
    let mid = (lo + hi) shr 1
    # Sort a[lo], a[mid], a[hi]; the pivot is the middle one.
    if a[mid] < a[lo]: swap(a[mid], a[lo])
    if a[hi] < a[lo]: swap(a[hi], a[lo])
    if a[hi] < a[mid]: swap(a[hi], a[mid])
    let pivot = a[mid]
    var i = lo
    var j = hi
    while i <= j:
      while a[i] < pivot: inc i
      while pivot < a[j]: dec j
      if i <= j:
        swap(a[i], a[j])
        inc i
        dec j
    if k <= j: hi = j
    elif k >= i: lo = i
    else: return a[k]
  result = a[k]
  {.pop.}

proc addDiversity[T](acc: var RecordStack[T], at: int, x: ptr UncheckedArray[T], n: int,
    ws: var Workspace) {.raises: [].} =
  let buf = ws.scratch(float64, 3 * n + 1)
  let s = buf
  let p = cast[F64](addr buf[n])
  let w = cast[F64](addr buf[2 * n])
  windowPowers(x, n, acc.half, s, p, w)
  # The median of the non-zero powers, from a copy in `s` (free now).
  var m = 0
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0..<n:
    s[m] = w[i]
    m += int(w[i] > 0)
  {.pop.}
  if m == 0: return # all zero: no weight anywhere
  let med = lowerMedian(s, m)
  let fl = acc.params.powerFloor * med
  let lim =
    if acc.params.clipLevel > 0: acc.params.clipLevel * sqrt(med) else: Inf
  {.push boundChecks: off, overflowChecks: off.}
  case acc.params.weight
  of dwInversePower:
    for i in 0..<n:
      let q = max(w[i], fl)
      w[i] = (if w[i] > 0: 1 / q else: 0.0)
  of dwInversePowerSq:
    for i in 0..<n:
      let q = max(w[i], fl)
      w[i] = (if w[i] > 0: 1 / (q * q) else: 0.0)
  {.pop.}
  addWeighted(cast[F64](addr acc.num[at]), cast[F64](addr acc.den[at]),
    cast[I32](addr acc.count[at]), x, w, lim, n)

# --------------------------------------------------------------------- adds --

proc add*[T](acc: var RecordStack[T], outRow: int, x: openArray[T],
    ws: var Workspace): bool {.raises: [].} =
  ## Adds the trace `x` to output row `outRow`, sample `j` onto sample `j`.
  ## Returns false, adding nothing, for `outRow < 0` or a NaN or infinite
  ## sample in `x`. A diversity stack takes its scratch from `ws`, which
  ## must hold `acc.workspaceSize` (an `AssertionDefect` otherwise, in every
  ## build); a vertical one does not use it. An `AssertionDefect` for a
  ## row past the stack or `x` longer than `samples`. Allocates nothing.
  if outRow < 0: return false
  doAssert outRow < acc.rows, "output row " & $outRow & " is outside 0 ..< " & $acc.rows
  doAssert x.len <= acc.samples, "the trace has " & $x.len &
    " samples, the stack " & $acc.samples
  let n = x.len
  if n == 0: return true
  if acc.params.mode == rsDiversity:
    doAssert ws.fits(acc.workspaceSize), "reserve the stack's workspaceSize first"
  let xp = cast[ptr UncheckedArray[T]](addr x[0])
  if not allFinite(xp, n): return false
  let at = outRow * acc.samples
  case acc.params.mode
  of rsVertical:
    addVertical(cast[F64](addr acc.num[at]), cast[I32](addr acc.count[at]), xp, n)
  of rsDiversity:
    addDiversity(acc, at, xp, n, ws)
  true

proc checkMap(acc: RecordStack, rows: int, rowMap: openArray[int]) =
  if rowMap.len == 0:
    if rows > acc.rows:
      badArg("a record of " & $rows & " rows into a stack of " & $acc.rows &
        " (pass a rowMap)")
  else:
    if rowMap.len != rows:
      badArg("rowMap has " & $rowMap.len & " entries, the record " & $rows & " rows")
    for r, o in rowMap:
      if o >= acc.rows:
        badArg("rowMap[" & $r & "] = " & $o & " is outside the stack's " & $acc.rows &
          " rows")

func outRowOf(rowMap: openArray[int], r: int): int {.inline.} =
  (if rowMap.len == 0: r else: rowMap[r])

proc addRecord*[T](acc: var RecordStack[T], b: RowBlock[T], meta: ChunkMeta,
    ws: var Workspace, rowMap: openArray[int] = []): int =
  ## Adds the live rows of the record `b` (`meta.isLive`), each over its
  ## `liveSamples`, row `r` to output row `rowMap[r]` (`< 0`: skipped; an
  ## empty map: row `r`). Returns the rows added (not counting skipped,
  ## non-live and non-finite rows). Raises `ValueError`, adding nothing,
  ## when `meta` does not match `b`, `b` is wider than `samples`, the
  ## record's interval is not the stack's, the meta is of a window
  ## (`firstSample > 0`: column 0 would be stacked onto sample 0), or the
  ## map does not have one entry per row within the stack's rows.
  meta.checkShape(b)
  if b.cols > acc.samples:
    badArg("the record has " & $b.cols & " samples, the stack " & $acc.samples)
  if abs(meta.dtMs - acc.dtMs) > 1e-9 * acc.dtMs:
    badArg("the record is at " & $meta.dtMs & " ms, the stack at " & $acc.dtMs & " ms")
  if meta.firstSample != 0:
    badArg("stacking needs whole records from sample 0, got firstSample " &
      $meta.firstSample)
  acc.checkMap(b.len, rowMap)
  for r in 0..<b.len:
    let o = rowMap.outRowOf(r)
    if o >= 0 and meta.isLive(r):
      if acc.add(o, b.row(r, meta.liveSamples(r, b.cols)), ws): inc result

proc addRecord*[T](acc: var RecordStack[T], b: RowBlock[T], ws: var Workspace,
    rowMap: openArray[int] = []): int =
  ## Adds every row of the record `b`, in full, as the `ChunkMeta` form does
  ## (no row is dead without a meta, and the caller answers for the
  ## interval and for column 0 being sample 0). Raises `ValueError` as that
  ## form for the width and the map.
  if b.cols > acc.samples:
    badArg("the record has " & $b.cols & " samples, the stack " & $acc.samples)
  acc.checkMap(b.len, rowMap)
  for r in 0..<b.len:
    let o = rowMap.outRowOf(r)
    if o >= 0:
      if acc.add(o, b.row(r), ws): inc result

proc addRecord*[T](acc: var RecordStack[T], samples: Tensor[T], meta: ChunkMeta,
    ws: var Workspace, rowMap: openArray[int] = []): int =
  ## The block form on the `[rows, cols]` tensor `samples`. Also raises
  ## `ValueError` unless it is rank 2 and C-contiguous.
  acc.addRecord(rowBlock(samples), meta, ws, rowMap)

proc addRecord*[T](acc: var RecordStack[T], samples: Tensor[T], ws: var Workspace,
    rowMap: openArray[int] = []): int =
  ## The block form without a meta, on the `[rows, cols]` tensor `samples`.
  acc.addRecord(rowBlock(samples), ws, rowMap)

proc merge*[T](acc: var RecordStack[T], other: RecordStack[T]) =
  ## Adds the sums and counts of `other` (another thread's share of the same
  ## stack) into `acc`. Raises `ValueError` unless they have the same shape,
  ## interval (within a relative 1e-9) and parameters.
  if other.rows != acc.rows or other.samples != acc.samples or
      abs(other.dtMs - acc.dtMs) > 1e-9 * acc.dtMs or other.params != acc.params:
    badArg("merging a " & $other.rows & " × " & $other.samples & " record stack into a " &
      $acc.rows & " × " & $acc.samples & " one, or with another interval or parameters")
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0..<acc.num.len:
    acc.num[i] = acc.num[i] + other.num[i]
    acc.count[i] = acc.count[i] + other.count[i]
  for i in 0..<acc.den.len:
    acc.den[i] = acc.den[i] + other.den[i]
  {.pop.}

proc clear*[T](acc: var RecordStack[T]) {.raises: [].} =
  ## Zeroes every sum and count (the storage is kept).
  if acc.num.len > 0:
    zeroMem(addr acc.num[0], acc.num.len * sizeof(float64))
    zeroMem(addr acc.count[0], acc.count.len * sizeof(int32))
  if acc.den.len > 0:
    zeroMem(addr acc.den[0], acc.den.len * sizeof(float64))

# ------------------------------------------------------------------ results --

proc checkDst(acc: RecordStack, dst: Tensor) =
  if dst.rank != 2 or dst.shape[0] != acc.rows or dst.shape[1] != acc.samples:
    badArg("dst must be a [" & $acc.rows & ", " & $acc.samples & "] tensor, got " &
      $dst.shape)
  if not dst.is_C_contiguous:
    badArg("dst must be C-contiguous (row-major)")

proc stackInto*[T](acc: RecordStack[T], dst: var Tensor[T], norm = fnFold) =
  ## The stack, `[rows, samples]`, into `dst` (allocated by the caller): the
  ## mean (`fnFold`), sum (`fnNone`) or sum / √count (`fnSqrtFold`) of a
  ## vertical stack; for a diversity stack the weighted mean `Σ w·x / Σ w`,
  ## times 1, count or √count. A sample of count 0 is 0. Raises `ValueError`
  ## when `dst` has another shape or is not C-contiguous. Allocates nothing.
  acc.checkDst(dst)
  if acc.num.len == 0: return
  let d = cast[ptr UncheckedArray[T]](dst.unsafe_raw_offset(aligned = false))
  {.push boundChecks: off.}
  case acc.params.mode
  of rsVertical:
    for i in 0..<acc.num.len:
      let c = float64(acc.count[i])
      d[i] =
        if c == 0: T(0)
        else:
          case norm
          of fnNone: T(acc.num[i])
          of fnFold: T(acc.num[i] / c)
          of fnSqrtFold: T(acc.num[i] / sqrt(c))
  of rsDiversity:
    for i in 0..<acc.num.len:
      let c = float64(acc.count[i])
      d[i] =
        if c == 0 or acc.den[i] == 0: T(0)
        else:
          let mean = acc.num[i] / acc.den[i]
          case norm
          of fnNone: T(mean * c)
          of fnFold: T(mean)
          of fnSqrtFold: T(mean * sqrt(c))
  {.pop.}

proc foldInto*[T](acc: RecordStack[T], dst: var Tensor[int32]) =
  ## The per-sample count, `[rows, samples]`, into `dst`. Raises `ValueError`
  ## as `stackInto`.
  acc.checkDst(dst)
  if acc.count.len > 0:
    copyMem(cast[pointer](dst.unsafe_raw_offset(aligned = false)), addr acc.count[0],
      acc.count.len * sizeof(int32))
