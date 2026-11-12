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
## **Normalization.** `fnFold` divides each sample by its fold (the mean),
## `fnSqrtFold` by √fold (balances signal against incoherent noise), and
## `fnNone` keeps the sum. A sample of fold 0 is 0.
##
## **Memory.** A stack costs `bins × samples × (sizeof(T) + 4)` bytes: the
## sums in `T` plus an `int32` fold (`stackMemory`). A float32 stack of
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

template ownCopy[U](t: Tensor[U], size: int): Tensor[U] =
  # Tensors share their storage when copied: an accumulator's copy must not.
  # (An empty one, or one `finishInto` moved out, has nothing to share.)
  (if size == 0: t else: t.clone())

proc `=copy`*[T](dst: var StackAccumulator[T], src: StackAccumulator[T]) =
  ## A copy owns its sums and folds: it is as large as the original.
  let size = src.bins * src.samples
  dst.sum = ownCopy(src.sum, size)
  dst.fold = ownCopy(src.fold, size)
  dst.bins = src.bins
  dst.samples = src.samples
  dst.dtMs = src.dtMs
  dst.startMs = src.startMs

proc `=dup`*[T](src: StackAccumulator[T]): StackAccumulator[T] =
  ## As `=copy`.
  let size = src.bins * src.samples
  StackAccumulator[T](bins: src.bins, samples: src.samples, dtMs: src.dtMs,
    startMs: src.startMs, sum: ownCopy(src.sum, size),
    fold: ownCopy(src.fold, size))

func stackMemory*(T: typedesc[SampleType], bins, samples: int): int =
  ## The bytes a stack of `bins × samples` holds: `sizeof(T) + 4` per sample.
  bins * samples * (sizeof(T) + 4)

proc initStackAccumulator*[T: SampleType](bins, samples: int, dtMs: float64,
    startMs = 0.0): StackAccumulator[T] =
  ## An empty stack of `bins` traces of `samples` samples at `dtMs` from
  ## `startMs`. Allocates `stackMemory(T, bins, samples)` bytes, zeroed.
  ## Raises `ValueError` unless `bins` and `samples` are ≥ 0 (and their
  ## product does not overflow), `dtMs` is finite and positive and `startMs`
  ## finite.
  if bins < 0 or samples < 0:
    raise newException(ValueError, "bins and samples must be ≥ 0, got " &
      $bins & " × " & $samples)
  if samples > 0 and bins > high(int) div (sizeof(T) + 4) div samples:
    raise newException(ValueError, "a stack of " & $bins & " × " & $samples &
      " samples is too large")
  if dtMs.classify notin {fcNormal, fcSubnormal} or dtMs <= 0:
    raise newException(ValueError, "dtMs must be finite and positive, got " & $dtMs)
  if startMs.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "startMs must be finite, got " & $startMs)
  StackAccumulator[T](bins: bins, samples: samples, dtMs: dtMs,
    startMs: startMs, sum: newTensor[T](bins, samples),
    fold: newTensor[int32](bins, samples))

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

proc accumulate[T](sum: ptr UncheckedArray[T], fold: ptr UncheckedArray[int32],
    x: ptr UncheckedArray[T], n: int) {.noinline, raises: [].} =
  ## `sum[j] += x[j]`, `fold[j] += 1` for `j < n`.
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0 ..< n:
    sum[j] = sum[j] + x[j]
    fold[j] = fold[j] + 1
  {.pop.}

func allFinite[T](x: openArray[T], first, last: int): bool {.inline.} =
  # x − x is 0 for finite x and NaN otherwise; one sum checks the span.
  var s = 0.0
  {.push boundChecks: off.}
  for j in first ..< last:
    s += float64(x[j]) - float64(x[j])
  {.pop.}
  s == 0

proc add*[T](acc: var StackAccumulator[T], bin: int, x: openArray[T],
    first = 0, last = -1): bool {.raises: [].} =
  ## Adds samples `first ..< last` of the trace `x` (`last = -1`: `x.len`)
  ## into bin `bin`, sample `j` of `x` onto sample `j` of the bin, and
  ## counts them in the fold. Returns false, adding nothing, for `bin < 0`
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
  if not allFinite(x, first, last): return false
  if last > first:
    let at = bin * acc.samples + first
    accumulate(cast[ptr UncheckedArray[T]](addr acc.sumPtr[at]),
      cast[ptr UncheckedArray[int32]](addr acc.foldPtr[at]),
      cast[ptr UncheckedArray[T]](addr x[first]), last - first)
  true

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
  ## Adds the sums and folds of `other` into `acc` (another thread's share
  ## of the same stack). Raises `ValueError` unless they have the same bins,
  ## samples and time axis.
  if other.bins != acc.bins or other.samples != acc.samples or
      other.dtMs != acc.dtMs or other.startMs != acc.startMs:
    raise newException(ValueError, "merging a " & $other.bins & " × " &
      $other.samples & " stack into a " & $acc.bins & " × " & $acc.samples &
      " one, or with another time axis")
  let n = acc.bins * acc.samples
  if n > 0:
    mergeInto(acc.sumPtr, acc.foldPtr, other.sumPtr, other.foldPtr, n)

proc clear*[T](acc: var StackAccumulator[T]) {.raises: [].} =
  ## Zeroes every sum and fold (the storage is kept).
  let n = acc.bins * acc.samples
  if n > 0:
    zeroMem(acc.sumPtr, n * sizeof(T))
    zeroMem(acc.foldPtr, n * sizeof(int32))

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
  ## add to). A large stack is thus held once, where `stackInto` with
  ## `foldInto` holds it twice. Allocates nothing.
  let n = acc.bins * acc.samples
  if n > 0:
    normalize(acc.sumPtr, acc.foldPtr, n, norm)
  data = move(acc.sum)
  fold = move(acc.fold)
  acc.bins = 0
