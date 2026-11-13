## Stacking: `StackAccumulator` sums NMO-corrected traces into bins, in any
## order, with a **per-sample** fold (decision D10), and normalizes the sums
## into a stack.
##
## ```nim
## var acc = initStackAccumulator[float32](bins = b.len, samples = 2001, dtMs = 2)
## for r in 0 ..< meta.len:                      # traces in any order
##   if meta.isLive(r) and bins[r] >= 0:
##     discard acc.addLive(bins[r], block.row(r))   # block = rowBlock(samples)
## var stack = newTensor[float32](acc.bins, acc.samples)
## acc.stackInto(stack, fnFold)                  # sum / fold, per sample
## # or, without a second copy of a large stack:
## var data: Tensor[float32]
## var fold: Tensor[int32]
## acc.finishInto(data, fold, fnFold)            # acc's storage, handed over
## ```
##
## **Fold.** Each bin counts, per sample, the traces that contributed there.
## `add` counts the columns it is given (`first ..< last`); `addLive` counts
## a trace's span between its first and last non-zero samples, so the
## zeros a top mute, a stretch mute or a bottom mute left (and the zero
## padding past a short trace) are not counted as data. Zeros inside the
## span count, and so do the samples of a mute's taper, at full weight
## (a stack under a tapered mute reads a few % low there). A trace with a
## NaN or infinite sample in its span is not added (it would poison the
## bin) and `add` returns false.
##
## **Per bin.** Each bin also counts the traces added to it (`tracesAt`: a
## fold map without the fold tensor) and, when the caller gives them
## (`addMidpoint`), keeps the mean of their midpoints (`cmpXAt`, `cmpYAt`):
## where a bin of header CDP or inline/crossline numbers lies on the map.
##
## **Normalization.** `fnFold` divides each sample by its fold (the mean),
## `fnSqrtFold` by √fold (balances signal against incoherent noise), and
## `fnNone` keeps the sum. A sample of fold 0 is 0.
##
## **Memory.** A stack costs `bins × (samples × (sizeof(T) + 4) + 24)`
## bytes: per sample the sum in `T` and an `int32` fold, per bin a trace
## count and the midpoint sums (`stackMemory`). A float32 stack of
## 10 000 bins × 2001 samples is 160 MB. Sums of float32 lose about
## log₂(fold) bits; at fold 1000, 10 of float32's 24. `stackInto` and
## `foldInto` copy into the caller's tensors (the stack is then held
## twice); `finishInto` normalizes in place and hands the accumulator's own
## storage over, so the stack is held once. A copy of an accumulator is a
## full copy of its sums.
##
## **Threads.** An accumulator is not shared. Either one thread adds every
## trace (`batch.bruteStack`: its workers correct the traces, the calling
## thread stacks them in file order, so the sums do not depend on the
## number of workers), or each thread has its own and they are `merge`d at
## the end (the order of the float sums, and so their last bits, then
## depends on which thread got which trace).

import std/math
import parseSEG

type
  FoldNorm* = enum
    ## How `stackInto` normalizes the sums.
    fnNone     ## the sum
    fnFold     ## the sum / fold: the mean of the traces
    fnSqrtFold ## the sum / √fold

  StackAccumulator*[T: SampleType] = object
    ## Sums and per-sample folds of `bins` stacked traces of `samples`
    ## samples each, on the time axis `startMs + j·dtMs`; see the module
    ## documentation. Build it with `initStackAccumulator`.
    bins, samples: int
    dtMs, startMs: float64
    # Tensors, `[bins, samples]`, so that `finishInto` can hand them over.
    sum: Tensor[T]
    fold: Tensor[int32]
    # Per bin: the traces added, and the midpoints given for them (their
    # count and coordinate sums).
    traces: seq[int32]
    midpoints: seq[int32]
    midX, midY: seq[float64]

template ownCopy[U](t: Tensor[U], size: int): Tensor[U] =
  # Tensors share their storage when copied: an accumulator's copy must not.
  # (An empty one, or one `finishInto` moved out, has nothing to share.)
  (if size == 0: t else: t.clone())

proc `=copy`*[T](dst: var StackAccumulator[T], src: StackAccumulator[T]) =
  ## A copy owns its sums and folds: it is as large as the original.
  let size = src.bins * src.samples
  dst.sum = ownCopy(src.sum, size)
  dst.fold = ownCopy(src.fold, size)
  dst.traces = src.traces
  dst.midpoints = src.midpoints
  dst.midX = src.midX
  dst.midY = src.midY
  dst.bins = src.bins
  dst.samples = src.samples
  dst.dtMs = src.dtMs
  dst.startMs = src.startMs

proc `=dup`*[T](src: StackAccumulator[T]): StackAccumulator[T] =
  ## As `=copy`.
  let size = src.bins * src.samples
  StackAccumulator[T](bins: src.bins, samples: src.samples, dtMs: src.dtMs,
    startMs: src.startMs, sum: ownCopy(src.sum, size),
    fold: ownCopy(src.fold, size), traces: src.traces,
    midpoints: src.midpoints, midX: src.midX, midY: src.midY)

const perBinBytes = 2 * sizeof(int32) + 2 * sizeof(float64)
  # A bin's trace count, midpoint count and midpoint sums.

func stackMemory*(T: typedesc[SampleType], bins, samples: int): int =
  ## The bytes a stack of `bins × samples` holds: `sizeof(T) + 4` per sample
  ## (the sum and the fold) and 24 per bin (its trace count and midpoint
  ## sums). Raises `ValueError` for a negative argument or a size beyond
  ## `int`.
  if bins < 0 or samples < 0:
    raise newException(ValueError, "bins and samples must be ≥ 0, got " &
      $bins & " × " & $samples)
  # Compared before multiplying: a danger build has no overflow checks.
  let perSample = sizeof(T) + 4
  if bins == 0: return 0
  if samples > (high(int) - perBinBytes) div perSample or
      samples * perSample + perBinBytes > high(int) div bins:
    raise newException(ValueError, "a stack of " & $bins & " × " & $samples &
      " samples is too large")
  bins * (samples * perSample + perBinBytes)

proc initStackAccumulator*[T: SampleType](bins, samples: int, dtMs: float64,
    startMs = 0.0): StackAccumulator[T] =
  ## An empty stack of `bins` traces of `samples` samples at `dtMs` from
  ## `startMs`. Allocates `stackMemory(T, bins, samples)` bytes, zeroed.
  ## Raises `ValueError` unless `bins` and `samples` are ≥ 0 (and their
  ## product does not overflow), `dtMs` is finite and positive and `startMs`
  ## finite.
  discard stackMemory(T, bins, samples) # refuses a negative or too large size
  if dtMs.classify notin {fcNormal, fcSubnormal} or dtMs <= 0:
    raise newException(ValueError, "dtMs must be finite and positive, got " & $dtMs)
  if startMs.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "startMs must be finite, got " & $startMs)
  StackAccumulator[T](bins: bins, samples: samples, dtMs: dtMs,
    startMs: startMs, sum: newTensor[T](bins, samples),
    fold: newTensor[int32](bins, samples), traces: newSeq[int32](bins),
    midpoints: newSeq[int32](bins), midX: newSeq[float64](bins),
    midY: newSeq[float64](bins))

template sumPtr[T](acc: StackAccumulator[T]): ptr UncheckedArray[T] =
  cast[ptr UncheckedArray[T]](acc.sum.unsafe_raw_offset(aligned = false))

template foldPtr[T](acc: StackAccumulator[T]): ptr UncheckedArray[int32] =
  cast[ptr UncheckedArray[int32]](acc.fold.unsafe_raw_offset(aligned = false))

func bins*[T](acc: StackAccumulator[T]): int {.inline, raises: [].} =
  ## The number of bins.
  acc.bins

func samples*[T](acc: StackAccumulator[T]): int {.inline, raises: [].} =
  ## The samples of each stacked trace.
  acc.samples

func dtMs*[T](acc: StackAccumulator[T]): float64 {.inline, raises: [].} =
  ## The sample interval, in ms.
  acc.dtMs

func startMs*[T](acc: StackAccumulator[T]): float64 {.inline, raises: [].} =
  ## The time of sample 0, in ms.
  acc.startMs

proc sampleOutside(bin, sample, bins, samples: int) {.noinline, noreturn.} =
  raise newException(IndexDefect, "bin " & $bin & ", sample " & $sample &
    " is outside the " & $bins & " × " & $samples & " stack")

template checkAt(acc: StackAccumulator, bin, sample: int) =
  # Each index on its own: a sample past a bin's end is inside the flat
  # storage (it is a sample of the next bin).
  when compileOption("boundChecks"):
    if bin < 0 or bin >= acc.bins or sample < 0 or sample >= acc.samples:
      sampleOutside(bin, sample, acc.bins, acc.samples)

func foldAt*[T](acc: StackAccumulator[T], bin, sample: int): int {.inline,
    raises: [].} =
  ## The fold of sample `sample` of bin `bin`. An `IndexDefect` for a bin
  ## outside `0 ..< bins` or a sample outside `0 ..< samples`, with bound
  ## checks on.
  checkAt(acc, bin, sample)
  int(acc.foldPtr[bin * acc.samples + sample])

func sumAt*[T](acc: StackAccumulator[T], bin, sample: int): T {.inline,
    raises: [].} =
  ## The sum at sample `sample` of bin `bin`. An `IndexDefect` as `foldAt`.
  checkAt(acc, bin, sample)
  acc.sumPtr[bin * acc.samples + sample]

proc binOutside(bin, bins: int) {.noinline, noreturn.} =
  raise newException(IndexDefect, "bin " & $bin & " is outside 0 ..< " & $bins)

template checkBin(acc: StackAccumulator, bin: int) =
  when compileOption("boundChecks"):
    if bin < 0 or bin >= acc.bins: binOutside(bin, acc.bins)

func tracesAt*[T](acc: StackAccumulator[T], bin: int): int {.inline,
    raises: [].} =
  ## The traces added to bin `bin`: the calls of `add` or `addLive` that
  ## added at least one sample. An `IndexDefect` for a bin outside
  ## `0 ..< bins`, with bound checks on.
  checkBin(acc, bin)
  {.push boundChecks: off.}
  result = int(acc.traces[bin])
  {.pop.}

func cmpXAt*[T](acc: StackAccumulator[T], bin: int): float64 {.inline,
    raises: [].} =
  ## The mean x of the midpoints given for bin `bin` (`addMidpoint`); NaN
  ## when none was. An `IndexDefect` as `tracesAt`.
  checkBin(acc, bin)
  {.push boundChecks: off.}
  result = if acc.midpoints[bin] > 0: acc.midX[bin] / float64(acc.midpoints[bin])
    else: NaN
  {.pop.}

func cmpYAt*[T](acc: StackAccumulator[T], bin: int): float64 {.inline,
    raises: [].} =
  ## The mean y of the midpoints given for bin `bin`; NaN when none was.
  checkBin(acc, bin)
  {.push boundChecks: off.}
  result = if acc.midpoints[bin] > 0: acc.midY[bin] / float64(acc.midpoints[bin])
    else: NaN
  {.pop.}

proc accumulate[T](sum: ptr UncheckedArray[T], fold: ptr UncheckedArray[int32],
    x: ptr UncheckedArray[T], n: int) {.noinline, raises: [].} =
  ## `sum[j] += x[j]`, `fold[j] += 1` for `j < n`.
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0 ..< n:
    sum[j] = sum[j] + x[j]
    fold[j] = fold[j] + 1
  {.pop.}

func allFinite[T](x: ptr UncheckedArray[T], n: int): bool {.noinline,
    raises: [].} =
  ## No NaN or infinite sample in `x[0 ..< n]`: none has all its exponent
  ## bits set. An OR of flags, which vectorizes (a float64 sum of x − x,
  ## 0 for a finite x and NaN otherwise, is an in-order reduction: 1.07 µs
  ## against 0.07 µs for 1501 float32 samples).
  when T is float32:
    const expBits = 0x7F80_0000'u32
    var bad = 0'u32
  else:
    const expBits = 0x7FF0_0000_0000_0000'u64
    var bad = 0'u64
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0 ..< n:
    bad = bad or typeof(bad)((cast[typeof(bad)](x[j]) and expBits) == expBits)
  {.pop.}
  bad == 0

proc add*[T](acc: var StackAccumulator[T], bin: int, x: openArray[T],
    first = 0, last = -1): bool {.raises: [].} =
  ## Adds samples `first ..< last` of the trace `x` (`last = -1`: `x.len`)
  ## into bin `bin`, sample `j` of `x` onto sample `j` of the bin, counts
  ## them in the fold and, when there is at least one, the trace in the
  ## bin (`tracesAt`). Returns false, adding nothing, for `bin < 0`
  ## (no bin, as `binsInto` reports it) or a NaN or infinite sample in the
  ## span. An `AssertionDefect` for a bin past the stack, `x` longer than
  ## `samples` or a span outside `x`. Allocates nothing.
  let last = (if last < 0: x.len else: last)
  if bin < 0: return false
  doAssert bin < acc.bins, "bin " & $bin & " is outside 0 ..< " & $acc.bins
  doAssert x.len <= acc.samples, "the trace has " & $x.len &
    " samples, the stack " & $acc.samples
  doAssert first >= 0 and first <= last and last <= x.len,
    "the span " & $first & " ..< " & $last & " is outside 0 ..< " & $x.len
  if last > first:
    let span = cast[ptr UncheckedArray[T]](addr x[first])
    if not allFinite(span, last - first): return false
    let at = bin * acc.samples + first
    accumulate(cast[ptr UncheckedArray[T]](addr acc.sumPtr[at]),
      cast[ptr UncheckedArray[int32]](addr acc.foldPtr[at]), span, last - first)
    inc acc.traces[bin]
  true

proc addMidpoint*[T](acc: var StackAccumulator[T], bin: int,
    cmpX, cmpY: float64) {.raises: [].} =
  ## Counts the midpoint `(cmpX, cmpY)` of a trace added to bin `bin`
  ## towards the bin's mean midpoint (`cmpXAt`, `cmpYAt`). A midpoint with a
  ## NaN coordinate, or `bin < 0`, is left out. An `AssertionDefect` for a
  ## bin past the stack. Allocates nothing.
  if bin < 0 or cmpX.isNaN or cmpY.isNaN: return
  doAssert bin < acc.bins, "bin " & $bin & " is outside 0 ..< " & $acc.bins
  inc acc.midpoints[bin]
  acc.midX[bin] += cmpX
  acc.midY[bin] += cmpY

func liveSpan*[T](x: openArray[T]): Slice[int] {.raises: [].} =
  ## The span from the first to the last non-zero sample of `x` (empty, `0
  ## .. -1`, when all are zero): what `addLive` adds.
  var a = 0
  while a < x.len and x[a] == 0: inc a
  if a == x.len: return 0 .. -1
  var b = x.high
  while x[b] == 0: dec b
  a .. b

proc addLive*[T](acc: var StackAccumulator[T], bin: int,
    x: openArray[T]): bool {.raises: [].} =
  ## Adds the trace `x` into bin `bin` over its `liveSpan`, so muted or
  ## padded zeros at either end do not count in the fold. Returns false as
  ## `add` (an all-zero trace adds nothing and returns true).
  let s = liveSpan(x)
  acc.add(bin, x, s.a, s.b + 1)

proc mergeInto[T](sum {.noalias.}: ptr UncheckedArray[T],
    fold {.noalias.}: ptr UncheckedArray[int32], other: ptr UncheckedArray[T],
    otherFold: ptr UncheckedArray[int32], n: int) {.noinline, raises: [].} =
  ## `sum[i] += other[i]`, `fold[i] += otherFold[i]` for `i < n`.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< n:
    sum[i] = sum[i] + other[i]
    fold[i] = fold[i] + otherFold[i]
  {.pop.}

proc merge*[T](acc: var StackAccumulator[T], other: StackAccumulator[T]) =
  ## Adds the sums, folds, trace counts and midpoints of `other` into `acc`
  ## (another thread's share of the same stack). Raises `ValueError` unless
  ## they have the same bins, samples and time axis.
  if other.bins != acc.bins or other.samples != acc.samples or
      other.dtMs != acc.dtMs or other.startMs != acc.startMs:
    raise newException(ValueError, "merging a " & $other.bins & " × " &
      $other.samples & " stack into a " & $acc.bins & " × " & $acc.samples &
      " one, or with another time axis")
  let n = acc.bins * acc.samples
  if n > 0:
    mergeInto(acc.sumPtr, acc.foldPtr, other.sumPtr, other.foldPtr, n)
  for bin in 0 ..< acc.bins:
    acc.traces[bin] += other.traces[bin]
    acc.midpoints[bin] += other.midpoints[bin]
    acc.midX[bin] += other.midX[bin]
    acc.midY[bin] += other.midY[bin]

proc clear*[T](acc: var StackAccumulator[T]) {.raises: [].} =
  ## Zeroes every sum, fold, trace count and midpoint (the storage is kept).
  let n = acc.bins * acc.samples
  if n > 0:
    zeroMem(acc.sumPtr, n * sizeof(T))
    zeroMem(acc.foldPtr, n * sizeof(int32))
  for bin in 0 ..< acc.traces.len:
    acc.traces[bin] = 0
    acc.midpoints[bin] = 0
    acc.midX[bin] = 0
    acc.midY[bin] = 0

proc checkDst(acc: StackAccumulator, dst: Tensor) =
  if dst.rank != 2 or dst.shape[0] != acc.bins or dst.shape[1] != acc.samples:
    raise newException(ValueError, "dst must be a [" & $acc.bins & ", " &
      $acc.samples & "] tensor, got " & $dst.shape)
  if not dst.is_C_contiguous:
    raise newException(ValueError, "dst must be C-contiguous (row-major)")

proc normalize[T](x {.noalias.}: ptr UncheckedArray[T],
    fold: ptr UncheckedArray[int32], n: int, norm: FoldNorm) {.noinline,
    raises: [].} =
  ## The sums `x` into the stack, in place. A sum of fold 0 is 0 (nothing
  ## was added there), so dividing by `max(fold, 1)` needs no branch: the
  ## loops are straight-line and vectorize.
  {.push boundChecks: off, overflowChecks: off.}
  case norm
  of fnNone: discard
  of fnFold:
    for i in 0 ..< n:
      x[i] = T(float64(x[i]) / float64(max(fold[i], 1)))
  of fnSqrtFold:
    for i in 0 ..< n:
      x[i] = T(float64(x[i]) / sqrt(float64(max(fold[i], 1))))
  {.pop.}

proc stackInto*[T](acc: StackAccumulator[T], dst: var Tensor[T],
    norm = fnFold) =
  ## The stack, `[bins, samples]`, into `dst` (allocated by the caller),
  ## normalized by `norm`; `acc` is left as it is, so more traces can be
  ## added. Raises `ValueError` when `dst` has another shape or is not
  ## C-contiguous. Allocates nothing.
  checkDst(acc, dst)
  let n = acc.bins * acc.samples
  if n == 0: return
  let d = cast[ptr UncheckedArray[T]](dst.unsafe_raw_offset(aligned = false))
  copyMem(d, acc.sumPtr, n * sizeof(T))
  normalize(d, acc.foldPtr, n, norm)

proc foldInto*[T](acc: StackAccumulator[T], dst: var Tensor[int32]) =
  ## The per-sample fold, `[bins, samples]`, into `dst`. Raises `ValueError`
  ## as `stackInto`.
  checkDst(acc, dst)
  let n = acc.bins * acc.samples
  if n > 0:
    copyMem(cast[pointer](dst.unsafe_raw_offset(aligned = false)), acc.foldPtr,
      n * sizeof(int32))

proc finishInto*[T](acc: var StackAccumulator[T], data: var Tensor[T],
    fold: var Tensor[int32], norm = fnFold) {.raises: [].} =
  ## Ends the accumulation without a copy: normalizes the sums in place by
  ## `norm` (as `stackInto` does into a copy) and moves the stack into
  ## `data` and its fold into `fold`, both `[bins, samples]`, replacing
  ## what they held. `acc` is left with no bins (`add` then finds no bin to
  ## add to): read its trace counts and midpoints (`tracesAt`, `cmpXAt`,
  ## `cmpYAt`) first. A large stack is thus held once, where `stackInto`
  ## with `foldInto` holds it twice. Allocates nothing.
  let n = acc.bins * acc.samples
  if n > 0:
    normalize(acc.sumPtr, acc.foldPtr, n, norm)
  data = move(acc.sum)
  fold = move(acc.fold)
  acc.bins = 0
