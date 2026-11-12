## Per-thread scratch memory for the kernels (`Workspace`, decision D11).
##
## Kernels that need temporary buffers (FFT work arrays, padded traces,
## running sums) take them from a `Workspace` instead of allocating. A
## workspace only grows: `reserve` reallocates when a request exceeds what it
## holds and is a no-op otherwise, so a steady stream of applies with the same
## (or smaller) plans allocates nothing.
##
## ```nim
## var ws = initWorkspace()
## ws.reserve(WorkspaceSize(f32: 4096, f64: 2048))  # once, or per plan change
## let tmp = ws.scratch(float64, 2048)              # ptr UncheckedArray[float64]
## ```
##
## Scratch contents are undefined on every call: a kernel initializes what it
## reads. Two buffers of one element type come from one region at different
## offsets (`at`), laid out by the caller. Element 0 of each region lies on a
## cache-line boundary (64 bytes), so a kernel that lays its buffers out in
## multiples of 64 bytes never has a SIMD load or store straddle two lines
## (measured on the batched FFT: 15 % faster than at the allocator's 8 bytes).
##
## **Threads.** One workspace per thread. A workspace cannot be copied (that
## would silently double the memory); move it (`move`, `sink`) into the thread
## or the pool worker that owns it.

type
  WorkspaceSize* = object
    ## Scratch needed by a plan, in elements of each type.
    f32*: int ## float32 elements.
    f64*: int ## float64 elements.
    u32*: int ## uint32 elements (counters, such as histogram bins).

  Region[T] = object
    ## `len` usable elements of `data`, from element `first`: the first one
    ## on a cache-line boundary.
    data: seq[T]
    first, len: int

  Workspace* = object
    ## Grow-only scratch; see the module documentation.
    f32: Region[float32]
    f64: Region[float64]
    u32: Region[uint32]

const lineBytes = 64 # a cache line (and the widest SIMD register)

proc `=copy`(dst: var Workspace, src: Workspace) {.error: "a Workspace cannot be " &
  "copied: each thread owns its own; move it instead".}

func initWorkspace*(): Workspace =
  ## An empty workspace; `reserve` sizes it.
  Workspace()

func `+`*(a, b: WorkspaceSize): WorkspaceSize =
  ## The scratch needed by two buffers used at the same time.
  WorkspaceSize(f32: a.f32 + b.f32, f64: a.f64 + b.f64, u32: a.u32 + b.u32)

func max*(a, b: WorkspaceSize): WorkspaceSize =
  ## The scratch needed by two steps that run one after the other.
  WorkspaceSize(f32: max(a.f32, b.f32), f64: max(a.f64, b.f64),
    u32: max(a.u32, b.u32))

func capacity*(ws: Workspace): WorkspaceSize {.raises: [].} =
  ## What `ws` holds.
  WorkspaceSize(f32: ws.f32.len, f64: ws.f64.len, u32: ws.u32.len)

proc grow[T](r: var Region[T], n: int) =
  ## Reallocates `r` to hold `n` elements from a cache-line boundary. A seq's
  ## storage is aligned for its element type, so the boundary is a whole
  ## number of elements away, less than a line.
  const slack = lineBytes div sizeof(T)
  r.data = newSeqUninit[T](n + slack)
  let past = cast[int](addr r.data[0]) and (lineBytes - 1)
  r.first = ((lineBytes - past) and (lineBytes - 1)) div sizeof(T)
  r.len = n

func fits*(ws: Workspace, size: WorkspaceSize): bool {.raises: [].} =
  ## `ws` holds at least `size`, so `reserve(size)` would not allocate.
  ws.f32.len >= size.f32 and ws.f64.len >= size.f64 and ws.u32.len >= size.u32

proc reserve*(ws: var Workspace, size: WorkspaceSize) =
  ## Grows `ws` to hold at least `size`; a no-op when it already does. The
  ## contents are not kept. Raises `ValueError` for a negative size.
  if size.f32 < 0 or size.f64 < 0 or size.u32 < 0:
    raise newException(ValueError, "a workspace size must be ≥ 0, got " & $size)
  if ws.f32.len < size.f32: ws.f32.grow(size.f32)
  if ws.f64.len < size.f64: ws.f64.grow(size.f64)
  if ws.u32.len < size.u32: ws.u32.grow(size.u32)

proc reserve*[P: not WorkspaceSize](ws: var Workspace, plan: P) =
  ## `ws.reserve(plan.workspaceSize)` for any plan that has a
  ## `workspaceSize` (`PipelinePlan`, `FilterPlan`, `IirPlan`, `AgcPlan`,
  ## `FftPlan`, `SpectrumPlan`, ...): grows `ws` for `plan`, a no-op when it
  ## already fits.
  mixin workspaceSize
  ws.reserve(plan.workspaceSize)

proc scratchOutside(at, n, cap: int) {.noinline, noreturn.} =
  raise newException(IndexDefect, "scratch of " & $n & " elements from element " & $at &
    " is outside the reserved 0 ..< " & $cap & " (reserve the plan first)")

template scratchOf(r: Region, at, n: int): untyped =
  # Checked in every build mode: a kernel writes all it asked for, so a
  # workspace reserved for a smaller plan would be a heap overflow in
  # `-d:danger`. Three compares per kernel call, never per sample.
  if at < 0 or n < 0 or n > r.len - at:
    scratchOutside(at, n, r.len)
  # `at` may equal `len` for an empty request: no element is addressed then.
  cast[ptr UncheckedArray[typeof(r.data[0])]](
    if n == 0 or r.len == 0: nil else: addr r.data[r.first + at])

proc scratch*(ws: var Workspace, T: typedesc[float32], n: int,
    at = 0): ptr UncheckedArray[float32] {.inline, raises: [].} =
  ## `n` float32 elements of scratch starting at element `at`, valid until
  ## `ws` is next reserved or destroyed (nil when `n` is 0). Element 0 is on
  ## a 64-byte boundary. An `IndexDefect` beyond the reserved size, in every
  ## build mode (`-d:danger` too).
  scratchOf(ws.f32, at, n)

proc scratch*(ws: var Workspace, T: typedesc[float64], n: int,
    at = 0): ptr UncheckedArray[float64] {.inline, raises: [].} =
  ## `n` float64 elements of scratch starting at element `at`; see the float32
  ## form.
  scratchOf(ws.f64, at, n)

proc scratch*(ws: var Workspace, T: typedesc[uint32], n: int,
    at = 0): ptr UncheckedArray[uint32] {.inline, raises: [].} =
  ## `n` uint32 elements of scratch starting at element `at`; see the float32
  ## form.
  scratchOf(ws.u32, at, n)
