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
## offsets (`at`), laid out by the caller.
##
## **Threads.** One workspace per thread. A workspace cannot be copied (that
## would silently double the memory); move it (`move`, `sink`) into the thread
## or the pool worker that owns it.

type
  WorkspaceSize* = object
    ## Scratch needed by a plan, in elements of each type.
    f32*: int ## float32 elements.
    f64*: int ## float64 elements.

  Workspace* = object
    ## Grow-only scratch; see the module documentation.
    f32: seq[float32]
    f64: seq[float64]

proc `=copy`(dst: var Workspace, src: Workspace) {.error: "a Workspace cannot be " &
  "copied: each thread owns its own; move it instead".}

func initWorkspace*(): Workspace =
  ## An empty workspace; `reserve` sizes it.
  Workspace()

func `+`*(a, b: WorkspaceSize): WorkspaceSize =
  ## The scratch needed by two buffers used at the same time.
  WorkspaceSize(f32: a.f32 + b.f32, f64: a.f64 + b.f64)

func max*(a, b: WorkspaceSize): WorkspaceSize =
  ## The scratch needed by two steps that run one after the other.
  WorkspaceSize(f32: max(a.f32, b.f32), f64: max(a.f64, b.f64))

func capacity*(ws: Workspace): WorkspaceSize {.raises: [].} =
  ## What `ws` holds.
  WorkspaceSize(f32: ws.f32.len, f64: ws.f64.len)

func fits*(ws: Workspace, size: WorkspaceSize): bool {.raises: [].} =
  ## `ws` holds at least `size`, so `reserve(size)` would not allocate.
  ws.f32.len >= size.f32 and ws.f64.len >= size.f64

proc reserve*(ws: var Workspace, size: WorkspaceSize) =
  ## Grows `ws` to hold at least `size`; a no-op when it already does. The
  ## contents are not kept. Raises `ValueError` for a negative size.
  if size.f32 < 0 or size.f64 < 0:
    raise newException(ValueError, "a workspace size must be ≥ 0, got " & $size)
  if ws.f32.len < size.f32:
    ws.f32 = newSeqUninit[float32](size.f32)
  if ws.f64.len < size.f64:
    ws.f64 = newSeqUninit[float64](size.f64)

proc scratchOutside(at, n, cap: int) {.noinline, noreturn.} =
  raise newException(IndexDefect, "scratch " & $at & " ..< " & $(at + n) &
    " is outside the reserved 0 ..< " & $cap & " (reserve the plan first)")

template scratchOf(s: seq, at, n: int): untyped =
  when compileOption("boundChecks"):
    if at < 0 or n < 0 or at + n > s.len:
      scratchOutside(at, n, s.len)
  # `at` may equal `len` for an empty request: no element is addressed then.
  cast[ptr UncheckedArray[typeof(s[0])]](if n == 0 or s.len ==
      0: nil else: addr s[at])

proc scratch*(ws: var Workspace, T: typedesc[float32], n: int,
    at = 0): ptr UncheckedArray[float32] {.inline, raises: [].} =
  ## `n` float32 elements of scratch starting at element `at`, valid until
  ## `ws` is next reserved or destroyed (nil when `n` is 0). An `IndexDefect` beyond the reserved
  ## size, with bound checks on.
  scratchOf(ws.f32, at, n)

proc scratch*(ws: var Workspace, T: typedesc[float64], n: int,
    at = 0): ptr UncheckedArray[float64] {.inline, raises: [].} =
  ## `n` float64 elements of scratch starting at element `at`; see the float32
  ## form.
  scratchOf(ws.f64, at, n)
