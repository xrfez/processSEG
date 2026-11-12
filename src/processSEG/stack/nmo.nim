## Normal moveout (NMO) correction: every output sample at zero-offset time
## `t₀` takes the input at `t = sqrt(t₀² + (1000·x / Vrms(t₀))²)`, by linear
## or 4-point cubic interpolation (inverse mapping, so no output sample is
## left unfilled), and a stretch mute zeroes the early part where the
## correction stretches the wavelet too much.
##
## ```nim
## let p = initNmoParams(maxStretch = 0.3, taperMs = 20)
## var ws = initWorkspace()
## ws.reserve(nmoWorkspace(float32, samples.shape[1]))
## samples.applyNmo(meta, field, p, ws)          # in place, row by row
## let first = nmoInto(src, srcStartMs, dst, dstStartMs, dtMs, offset, vrms, p)
## ```
##
## **Times.** Times are absolute, in ms: a row's column `j` is at
## `t0Ms + (firstSample + j)·dtMs` (`ChunkMeta.sampleTimeMs`), so each row
## is corrected from its own `t0Ms`. The kernel `nmoInto` takes the start
## times of its input and output separately, so it can also resample a row
## onto another time axis (a stack's). Input outside the row reads as zero;
## output times before 0 are zero.
##
## **Stretch.** NMO maps an input interval `Δt` onto the output interval
## `Δt₀`; the stretch is `Δt₀/Δt − 1`, which is `t/t₀ − 1` at a constant
## velocity (the relative drop in frequency). It is measured between
## neighbouring output samples, so a velocity that changes with time
## counts, and a mapping that folds back (`Δt ≤ 0`, from a strong velocity
## inversion) is infinitely stretched. The mute starts at the first output
## sample from the top whose stretch is at most `maxStretch`: samples from
## there on are untouched, the `taperMs` before it are ramped in with a sin²
## taper (`muteTop`'s convention), and earlier ones are zero. With
## `maxStretch = Inf` nothing is muted.
##
## **Velocities.** The block forms take a `VelocityFunction` (one for every
## row) or a `VelocityField` read at each row's location: the keys'
## `ensemble` (the CDP) for a 2D field, `inline` and `crossline` for a 3D
## one, as `moveoutTimesInto`. The offset's sign is ignored.
##
## **NaN.** The block forms leave a row alone when its offset, its `t0Ms` or
## its location is NaN, as they do dead, auxiliary and failed rows. The
## kernel cannot leave its output alone, so it zeroes it then.

import std/math
import parseSEG
import ../core/[mute, rows, shift, types, workspace]

# `sqrt` never needs errno here (its argument is a sum of squares), and
# without errno GCC vectorizes it: this module's own C code only (the generic
# kernels compile in their callers' modules, so the loop is `moveoutCols`).
{.localPassC: "-fno-math-errno".}
import ../velocity/model

export ShiftInterp

type
  NmoParams* = object
    ## How NMO interpolates and mutes; build it with `initNmoParams`.
    interp: ShiftInterp
    maxStretch: float64
    taperMs: float64

proc initNmoParams*(interp = siLinear, maxStretch = 0.3,
    taperMs = 0.0): NmoParams =
  ## NMO with `interp`olation, muting where the stretch exceeds `maxStretch`
  ## (a fraction: 0.3 is 30 %; `Inf` mutes nothing) with a `taperMs` ramp.
  ## Raises `ValueError` unless `maxStretch` is positive (or `Inf`) and
  ## `taperMs` finite and ≥ 0.
  if maxStretch.isNaN or maxStretch <= 0:
    raise newException(ValueError, "maxStretch must be positive (Inf: no " &
      "stretch mute), got " & $maxStretch)
  if taperMs.classify in {fcNan, fcInf, fcNegInf} or taperMs < 0:
    raise newException(ValueError, "taperMs must be finite and ≥ 0, got " & $taperMs)
  NmoParams(interp: interp, maxStretch: maxStretch, taperMs: taperMs)

func interp*(p: NmoParams): ShiftInterp {.inline, raises: [].} =
  ## Linear or cubic interpolation.
  p.interp

func maxStretch*(p: NmoParams): float64 {.inline, raises: [].} =
  ## The largest stretch kept, as a fraction (`Inf`: no mute).
  p.maxStretch

func taperMs*(p: NmoParams): float64 {.inline, raises: [].} =
  ## The length of the mute's ramp, in ms.
  p.taperMs

func nmoStretch*(offset, vrms, t0Ms: float64): float64 {.inline, raises: [].} =
  ## The stretch `t/t₀ − 1` of NMO at a constant velocity `vrms`, at the
  ## zero-offset time `t0Ms` (Inf at `t0Ms ≤ 0` with a non-zero offset).
  let tx = 1000 * offset / vrms
  if t0Ms <= 0: (if tx == 0: 0.0 else: Inf)
  else: sqrt(t0Ms * t0Ms + tx * tx) / t0Ms - 1

proc moveoutCols(cols: var array[256, float64], len, j0: int,
    dstStartMs, dtMs, srcStartMs, x: float64,
    vrms {.noalias.}: ptr UncheckedArray[float64]) {.raises: [].} =
  # The input column of output samples j0 ..< j0 + len (`vrms` from j0);
  # NaN for a NaN velocity. Straight-line, so it vectorizes: the caller
  # drops the samples before time 0 (nothing maps there).
  # i < 256 converts as an int32 (int64 → float64 has no packed AVX2
  # form); j0 + i stays exact as a float64 sum.
  let base = float64(j0)
  {.push boundChecks: off, overflowChecks: off, rangeChecks: off.} # i < 256
  for i in 0 ..< len:
    let t0 = dstStartMs + (base + float64(int32(i))) * dtMs
    let tx = x / vrms[i]
    cols[i] = (sqrt(t0 * t0 + tx * tx) - srcStartMs) / dtMs
  {.pop.}

proc nmoInto*[T: SampleType](src: openArray[T], srcStartMs: float64,
    dst: var openArray[T], dstStartMs, dtMs, offset: float64,
    vrms: openArray[float64], p: NmoParams): int {.raises: [].} =
  ## NMO-corrects the row `src`, whose sample `i` is at `srcStartMs + i·dtMs`,
  ## into `dst`, whose sample `j` is at the zero-offset time
  ## `dstStartMs + j·dtMs`, with the RMS velocity `vrms[j]` there (`vrms.len
  ## == dst.len`). Returns the first column of `dst` the stretch mute does
  ## not zero (`dst.len` when it zeroes all). A NaN offset or start time, or
  ## a `dtMs` that is not positive, zeroes `dst` (and returns `dst.len`);
  ## so does a default `p`. `src` and `dst` must not overlap. Allocates
  ## nothing. An `AssertionDefect` when `vrms.len != dst.len`.
  doAssert vrms.len == dst.len, "vrms must have one velocity per output sample"
  let n = src.len
  let m = dst.len
  if offset.isNaN or srcStartMs.isNaN or dstStartMs.isNaN or not (dtMs > 0) or
      p.maxStretch.isNaN or p.maxStretch <= 0:
    for v in dst.mitems: v = T(0)
    return m
  let x = 1000 * abs(offset)
  # The input column of output sample j (NaN before time 0, where nothing
  # maps; a NaN velocity gives NaN too).
  template inCol(j: int): float64 =
    let t0 = dstStartMs + float64(j) * dtMs
    if t0 < 0: NaN
    else:
      let tx = x / vrms[j]
      (sqrt(t0 * t0 + tx * tx) - srcStartMs) / dtMs
  # The stretch mute: the first column whose stretch Δt₀/Δt − 1 is at most
  # maxStretch, measured to the next column (to the previous one at the end).
  var start = 0
  if p.maxStretch < Inf and m > 1:
    start = m
    var c = inCol(0)
    for j in 0 ..< m:
      let next = if j + 1 < m: inCol(j + 1) else: NaN
      let d = if j + 1 < m: next - c else: c - inCol(j - 1)
      if d > 0 and 1 / d - 1 <= p.maxStretch:
        start = j
        break
      c = next
  let taper = p.taperMs / dtMs
  # The first column the ramp does not zero (the ramp's start has weight 0).
  let first =
    if start >= m: m
    elif taper > 0:
      let f = floor(float64(start) - taper) # a long taper: below any int
      if f < 0: 0 else: int(f) + 1
    else: start
  template at(i: int): float64 =
    (if i >= 0 and i < n: float64(src[i]) else: 0.0)
  {.push boundChecks: off, overflowChecks: off.}
  for j in 0 ..< first:
    dst[j] = T(0)
  # The input columns in blocks, in a loop of their own (sqrt and division
  # vectorize there), then the interpolation.
  const blockLen = 256
  var cols: array[blockLen, float64]
  var j0 = first
  while j0 < m:
    let len = min(blockLen, m - j0)
    moveoutCols(cols, len, j0, dstStartMs, dtMs, srcStartMs, x,
      cast[ptr UncheckedArray[float64]](addr vrms[j0]))
    for i in 0 ..< len:
      let j = j0 + i
      let c = cols[i]
      if c.isNaN or c <= -2 or c >= float64(n + 1) or
          dstStartMs + float64(j) * dtMs < 0:
        dst[j] = T(0)
        continue
      let k = int(floor(c))
      let f = c - float64(k)
      case p.interp
      of siLinear:
        dst[j] = T((1 - f) * at(k) + f * at(k + 1))
      of siCubic:
        let f2 = f * f
        let f3 = f2 * f
        dst[j] = T((-0.5 * f3 + f2 - 0.5 * f) * at(k - 1) +
          (1.5 * f3 - 2.5 * f2 + 1) * at(k) +
          (-1.5 * f3 + 2 * f2 + 0.5 * f) * at(k + 1) +
          (0.5 * f3 - 0.5 * f2) * at(k + 2))
    j0 += len
  {.pop.}
  if start < m and taper > 0:
    muteTop(dst, float64(start), taper)
  first

# -------------------------------------------------------------- block forms --

func nmoWorkspace*(T: typedesc[SampleType], cols: int): WorkspaceSize =
  ## The scratch the block forms need for rows of `cols` samples: a velocity
  ## column and a copy of the row.
  when T is float32: WorkspaceSize(f64: cols, f32: cols)
  else: WorkspaceSize(f64: 2 * cols)

proc rowScratch[T](ws: var Workspace, cols: int): ptr UncheckedArray[T] {.inline.} =
  when T is float32: ws.scratch(float32, cols)
  else: ws.scratch(float64, cols, at = cols)

proc nmoRow[T](b: RowBlock[T], meta: ChunkMeta, r: int, p: NmoParams,
    vel: ptr UncheckedArray[float64], ws: var Workspace) {.inline.} =
  # Corrects the live part of row r in place, its velocities in `vel`.
  let n = meta.liveSamples(r, b.cols)
  if n == 0: return
  let copy = rowScratch[T](ws, b.cols)
  copyMem(copy, b.rowPtr(r), n * sizeof(T))
  let start = meta.sampleTimeMs(r, 0)
  discard nmoInto(copy.toOpenArray(0, n - 1), start, b.row(r, n), start,
    meta.dtMs, meta.keys(r).offset, vel.toOpenArray(0, n - 1), p)

proc checkNmo[T](b: RowBlock[T], meta: ChunkMeta, p: NmoParams,
    ws: Workspace) =
  meta.checkShape(b)
  if p.maxStretch.isNaN or p.maxStretch <= 0:
    raise newException(ValueError, "the NmoParams must come from initNmoParams")
  doAssert ws.fits(nmoWorkspace(T, b.cols)),
    "reserve nmoWorkspace(T, cols) in the workspace first"

func usable(meta: ChunkMeta, r: int): bool {.inline.} =
  meta.isLive(r) and not meta.keys(r).offset.isNaN and not meta.t0Ms(r).isNaN

proc applyNmo*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    f: VelocityFunction, p: NmoParams, ws: var Workspace) =
  ## NMO-corrects every live row of `b` in place with the RMS velocities of
  ## `f`, each row from its own `t0Ms` (see the module documentation). `ws`
  ## must hold `nmoWorkspace(T, b.cols)` (an `AssertionDefect` otherwise).
  ## Raises `ValueError` when `b` and `meta` disagree or `p` is a default.
  ## Allocates nothing.
  checkNmo(b, meta, p, ws)
  let vel = ws.scratch(float64, b.cols)
  var last = NaN # the start time `vel` was computed for: rows share it
  for r in 0 ..< b.len:
    if meta.usable(r):
      let start = meta.sampleTimeMs(r, 0)
      if start != last:
        f.vrmsInto(start, meta.dtMs, vel.toOpenArray(0, b.cols - 1))
        last = start
      nmoRow(b, meta, r, p, vel, ws)

proc applyNmo*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    field: VelocityField, p: NmoParams, ws: var Workspace) =
  ## NMO with the field's RMS velocities at each row's location (`ensemble`
  ## for a 2D field, `inline` and `crossline` for a 3D one); rows with a NaN
  ## location are left alone. Otherwise as the `VelocityFunction` form.
  checkNmo(b, meta, p, ws)
  let vel = ws.scratch(float64, b.cols)
  template col: untyped = vel.toOpenArray(0, b.cols - 1)
  # The location and start time `vel` was computed for: the rows of a
  # gather share them.
  var lastA, lastB, lastStart = NaN
  for r in 0 ..< b.len:
    if not meta.usable(r): continue
    let k = meta.keys(r)
    let start = meta.sampleTimeMs(r, 0)
    case field.kind
    of fk2D:
      if k.ensemble.isNaN: continue
      if k.ensemble != lastA or start != lastStart:
        field.vrmsInto(k.ensemble, start, meta.dtMs, col)
        (lastA, lastStart) = (k.ensemble, start)
    of fk3D:
      if k.inline.isNaN or k.crossline.isNaN: continue
      if k.inline != lastA or k.crossline != lastB or start != lastStart:
        field.vrmsInto(k.inline, k.crossline, start, meta.dtMs, col)
        (lastA, lastB, lastStart) = (k.inline, k.crossline, start)
    nmoRow(b, meta, r, p, vel, ws)

proc applyNmo*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    f: VelocityFunction, p: NmoParams, ws: var Workspace) =
  ## `applyNmo` on a `[rows, cols]` tensor, in place. Raises `ValueError`
  ## unless it is C-contiguous with `meta.len` rows, or as the block form.
  meta.checkShape(samples)
  applyNmo(rowBlock(samples), meta, f, p, ws)

proc applyNmo*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    field: VelocityField, p: NmoParams, ws: var Workspace) =
  ## `applyNmo` with a field on a `[rows, cols]` tensor, in place.
  meta.checkShape(samples)
  applyNmo(rowBlock(samples), meta, field, p, ws)
