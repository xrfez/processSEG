## Zero-copy row access to a block of samples (`RowBlock`).
##
## Kernels work on one row at a time, as an `openArray`. A `RowBlock` is a
## raw view of a C-contiguous `[rows, cols]` buffer, a tensor or any other
## contiguous storage, that hands out those rows without copying:
##
## ```nim
## var samples = zeros[float32](100, 2001)
## let b = rowBlock(samples)          # checks rank 2 and C-contiguity once
## for r in 0 ..< b.len:
##   demean(b.row(r))                 # a var openArray[float32] over row r
## ```
##
## **Lifetime.** A block holds a raw pointer to the storage. It is valid
## while that storage lives and is not reallocated: never make one from a
## temporary (`rowBlock(zeros[float32](4, 4))` dangles at once), and do not
## resize or reassign the tensor while a block of it is in use.
##
## **Threads.** A block has no reference-counted parts, so it can be copied
## into worker threads, which is how the pool hands rows to its workers
## (`sub` splits a block). Threads must write disjoint rows.
##
## **Mutability.** One type serves the source and the destination of a
## `src → dst` apply. A block made from a `let` tensor must only be read.

import parseSEG

type
  RowBlock*[T: SampleType] = object
    ## A raw view of `len` rows of `cols` samples each, stored row after row.
    ## See the module documentation for its lifetime and threading contract.
    data: ptr UncheckedArray[T]
    rows, cols: int

proc badBlock(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc rowOutside(row, count: int) {.noinline, noreturn.} =
  raise newException(IndexDefect, "row " & $row & " is outside 0 ..< " & $count)

proc rowBlock*[T: SampleType](samples: Tensor[T]): RowBlock[T] =
  ## A view of the rows of the `[rows, cols]` tensor `samples`. Raises
  ## `ValueError` unless it is rank 2 and C-contiguous (row-major with no
  ## gaps, as parseSEG's readers and prefetcher return it).
  if samples.rank != 2:
    badBlock("samples must be a [rows, cols] tensor, got rank " & $samples.rank)
  if not samples.is_C_contiguous:
    badBlock("samples must be C-contiguous (row-major)")
  result.rows = samples.shape[0]
  result.cols = samples.shape[1]
  if samples.size > 0:
    result.data = cast[ptr UncheckedArray[T]](samples.unsafe_raw_offset(
        aligned = false))

proc rowBlock*[T: SampleType](data: openArray[T], rows, cols: int): RowBlock[T] =
  ## A view of `data` as `rows` rows of `cols` samples. Raises `ValueError`
  ## when `rows` or `cols` is negative or `data.len != rows · cols`.
  if rows < 0 or cols < 0:
    badBlock("rows and cols must be ≥ 0, got " & $rows & " × " & $cols)
  if data.len != rows * cols:
    badBlock("need " & $rows & " × " & $cols & " samples, got " & $data.len)
  result.rows = rows
  result.cols = cols
  if data.len > 0:
    result.data = cast[ptr UncheckedArray[T]](unsafeAddr data[0])

func len*[T](b: RowBlock[T]): int {.inline, raises: [].} =
  ## The number of rows.
  b.rows

func cols*[T](b: RowBlock[T]): int {.inline, raises: [].} =
  ## The number of samples in each row.
  b.cols

func rowPtr*[T](b: RowBlock[T], row: int): ptr UncheckedArray[T] {.inline,
    raises: [].} =
  ## The storage of row `row`. An `IndexDefect` for a row outside `0 ..< len`
  ## with bound checks on.
  when compileOption("boundChecks"):
    if row < 0 or row >= b.rows:
      rowOutside(row, b.rows)
  cast[ptr UncheckedArray[T]](addr b.data[row * b.cols])

template row*[T](b: RowBlock[T], r: int): untyped =
  ## Row `r` as an `openArray[T]` of `cols` samples, in place: pass it to a
  ## parameter of type `openArray[T]` or `var openArray[T]`.
  toOpenArray(rowPtr(b, r), 0, cols(b) - 1)

func sub*[T](b: RowBlock[T], rows: Slice[int]): RowBlock[T] {.raises: [ValueError].} =
  ## The rows `rows` of `b` as a block of their own (row 0 is `b`'s row
  ## `rows.a`), for splitting work across threads. Raises `ValueError` for
  ## rows outside `0 ..< len` (an empty slice may start at `len`).
  if rows.a < 0 or rows.a > b.rows or rows.b >= b.rows or rows.b < rows.a - 1:
    raise newException(ValueError, "rows " & $rows.a & " .. " & $rows.b &
      " are outside 0 ..< " & $b.rows)
  result.cols = b.cols
  result.rows = rows.b - rows.a + 1
  if result.rows > 0:
    result.data = cast[ptr UncheckedArray[T]](addr b.data[rows.a * b.cols])
