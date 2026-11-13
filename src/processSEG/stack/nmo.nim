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
## samples.applyNmo(meta, field, binning, p, ws) # the field as a stack reads it
## let first = nmoInto(src, srcStartMs, dst, dstStartMs, dtMs, offset, vrms, p)
## vrmsToSloth(vrms)                             # once per velocity column: 1/V²
## discard nmoSlothInto(src, srcStartMs, dst, dstStartMs, dtMs, offset, vrms, p)
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
## one, as `moveoutTimesInto`. With a `Binning` the field is read at the
## location of each row's bin instead (`Binning.locationOf`), which is
## where a stack with that binning reads it: the preview of what
## `bruteStack` will stack, also for bins from midpoints (`lineBinning`,
## `gridBinning`), whose numbers are not the traces' header numbers. The
## offset's sign is ignored. The kernel
## works with the *sloth*, `1/Vrms²` (`t² = t₀² + x²·sloth`): `nmoInto`
## takes velocities and inverts them per call; a caller that corrects many
## traces with one velocity column converts it once (`vrmsToSloth`) and
## calls `nmoSlothInto`, which divides nothing. Both give the same samples.
##
## **Precision.** A float32 row is interpolated in float32 where all its
## taps are inside the input (within 1e-6 of a float64 evaluation, relative
## to the samples' size), a float64 row in float64.
##
## **NaN.** The block forms leave a row alone when its offset, its `t0Ms` or
## its location is NaN, as they do dead, auxiliary and failed rows. The
## kernel cannot leave its output alone, so it zeroes it then.

import std/math
import parseSEG
import ../core/[mute, rows, shift, types, workspace]
import ./binning

# `sqrt` never needs errno here (its argument is a sum of squares), and
# without errno GCC vectorizes it; `floor` it vectorizes only when told that
# floating-point operations do not trap (nothing here reads the exception
# flags). Both for this module's own C code only: the kernels below are not
# generic, or are instantiated here, so their loops are compiled with these
# flags (a generic kernel compiles in its caller's module).
{.localPassC: "-fno-math-errno -fno-trapping-math".}
import ../velocity/[model, moveout]

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

const blockLen = 256 # output samples whose input columns are computed at a time

type
  ColBlock = array[blockLen, float64]
  WholeBlock = array[blockLen, float32] # whole column numbers (exact below 2²⁴)

proc moveoutCols(cols: var ColBlock, len, j0: int,
    dstStartMs, dtMs, srcStartMs, x2, invDt: float64,
    sloth {.noalias.}: ptr UncheckedArray[float64]) {.raises: [].} =
  # The input column of output samples j0 ..< j0 + len (`sloth`, 1/V², from
  # j0); NaN for a NaN sloth. Straight-line, so it vectorizes: the caller
  # drops the samples before time 0 (nothing maps there). No division: x²
  # times the sloth, and the inverse of dt (0.55 ns per sample, against
  # 0.88 with x / V and / dt).
  # i < 256 converts as an int32 (int64 → float64 has no packed AVX2
  # form); j0 + i stays exact as a float64 sum.
  let base = float64(j0)
  {.push boundChecks: off, overflowChecks: off, rangeChecks: off.} # i < 256
  for i in 0 ..< len:
    let t0 = dstStartMs + (base + float64(int32(i))) * dtMs
    cols[i] = (sqrt(t0 * t0 + x2 * sloth[i]) - srcStartMs) * invDt
  {.pop.}

proc slothCols(sloth: var ColBlock, len: int,
    vrms {.noalias.}: ptr UncheckedArray[float64]) {.raises: [].} =
  # 1/V² of `len` velocities: the one division left to the velocity form.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< len:
    sloth[i] = 1 / (vrms[i] * vrms[i])
  {.pop.}

proc insideCount(cols: var ColBlock, len: int, lo, hi: float64): int {.
    raises: [].} =
  # How many of the columns are in [lo, hi) (a NaN is not). The `and` of two
  # flags, not of two conditions: no branch, so it vectorizes.
  var count = 0'i32
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< len:
    count = count +% (int32(cols[i] >= lo) and int32(cols[i] < hi))
  {.pop.}
  int(count)

proc splitCols[F](whole {.noalias.}: ptr WholeBlock,
    frac {.noalias.}: ptr array[blockLen, F], cols {.noalias.}: ptr ColBlock,
    len: int) {.raises: [].} =
  # The whole and fractional parts of columns ≥ 0. `floor`, not a conversion
  # to an integer (float64 → int64 has no packed AVX2 form), and
  # -fno-trapping-math for it (see the top of the module): the loop
  # vectorizes.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< len:
    let c = cols[i]
    let w = floor(c)
    whole[i] = float32(w)
    frac[i] = F(c - w)
  {.pop.}

proc cubicWeights[F](w0 {.noalias.}, w1 {.noalias.}, w2 {.noalias.},
    w3 {.noalias.}, frac {.noalias.}: ptr array[blockLen, F], len: int) {.
    raises: [].} =
  # The Catmull-Rom weights of taps −1, 0, 1, 2 at each fraction.
  {.push boundChecks: off, overflowChecks: off.}
  for i in 0 ..< len:
    let f = frac[i]
    let f2 = f * f
    let f3 = f2 * f
    w0[i] = F(-0.5) * f3 + f2 - F(0.5) * f
    w1[i] = F(1.5) * f3 - F(2.5) * f2 + 1
    w2[i] = F(-1.5) * f3 + 2 * f2 + F(0.5) * f
    w3[i] = F(0.5) * f3 - F(0.5) * f2
  {.pop.}

template defineTapKernels(lerpName, cubicName: untyped, F: typedesc) =
  # The interior interpolation of one block: every column has all its taps
  # inside `src`. The whole parts, fractions and weights come from loops
  # that vectorize; the loads at data-dependent places are left to a loop
  # with nothing else in it. `F` is the type the sums run in.
  proc lerpName(dst {.noalias.}, src {.noalias.}: ptr UncheckedArray[F],
      cols {.noalias.}: ptr ColBlock, len: int) {.raises: [].} =
    var whole {.noinit.}: WholeBlock
    var frac {.noinit.}: array[blockLen, F]
    splitCols(addr whole, addr frac, cols, len)
    {.push boundChecks: off, overflowChecks: off, rangeChecks: off.}
    for i in 0 ..< len:
      let k = int(whole[i])
      when F is float32:
        let a = src[k]
        dst[i] = a + frac[i] * (src[k + 1] - a)
      else:
        dst[i] = (1 - frac[i]) * src[k] + frac[i] * src[k + 1]
    {.pop.}

  proc cubicName(dst {.noalias.}, src {.noalias.}: ptr UncheckedArray[F],
      cols {.noalias.}: ptr ColBlock, len: int) {.raises: [].} =
    var whole {.noinit.}: WholeBlock
    var frac {.noinit.}, w0 {.noinit.}, w1 {.noinit.}, w2 {.noinit.},
      w3 {.noinit.}: array[blockLen, F]
    splitCols(addr whole, addr frac, cols, len)
    cubicWeights(addr w0, addr w1, addr w2, addr w3, addr frac, len)
    {.push boundChecks: off, overflowChecks: off, rangeChecks: off.}
    for i in 0 ..< len:
      let k = int(whole[i])
      dst[i] = w0[i] * src[k - 1] + w1[i] * src[k] + w2[i] * src[k + 1] +
        w3[i] * src[k + 2]
    {.pop.}

defineTapKernels(lerpCols32, cubicCols32, float32)
defineTapKernels(lerpCols64, cubicCols64, float64)

func firstFromZero(startMs, dtMs: float64, m: int): int =
  # The first j in 0 .. m with startMs + j·dtMs ≥ 0 (m when there is none):
  # an estimate, then the very expression the columns use.
  if startMs >= 0: return 0
  let est = -startMs / dtMs
  result = if est >= float64(m): m else: int(est)
  while result > 0 and startMs + float64(result - 1) * dtMs >= 0: dec result
  while result < m and startMs + float64(result) * dtMs < 0: inc result

proc nmoImpl[T: SampleType](src: openArray[T], srcStartMs: float64,
    dst: var openArray[T], dstStartMs, dtMs, offset: float64,
    vel: openArray[float64], isSloth: bool, p: NmoParams): int {.raises: [].} =
  # `vel[j]` is the sloth of output sample j, or its RMS velocity.
  doAssert vel.len == dst.len, "one velocity per output sample is needed"
  let n = src.len
  let m = dst.len
  if offset.isNaN or srcStartMs.isNaN or dstStartMs.isNaN or not (dtMs > 0) or
      p.maxStretch.isNaN or p.maxStretch <= 0:
    for v in dst.mitems: v = T(0)
    return m
  let x = 1000 * abs(offset)
  let x2 = x * x
  let invDt = 1 / dtMs
  var cols {.noinit.}: ColBlock
  var sloth {.noinit.}: ColBlock
  {.push boundChecks: off, overflowChecks: off.}
  # The input columns of output samples j0 ..< j0 + len, into `cols`.
  template columns(j0, len: int) =
    let v = cast[ptr UncheckedArray[float64]](addr vel[j0])
    if isSloth:
      moveoutCols(cols, len, j0, dstStartMs, dtMs, srcStartMs, x2, invDt, v)
    else:
      slothCols(sloth, len, v)
      moveoutCols(cols, len, j0, dstStartMs, dtMs, srcStartMs, x2, invDt,
        cast[ptr UncheckedArray[float64]](addr sloth[0]))
  # Nothing maps to an output time before 0: those columns are zero, and
  # no mute starts there.
  let fromZero = firstFromZero(dstStartMs, dtMs, m)
  # The stretch mute: the first column whose stretch Δt₀/Δt − 1 is at most
  # maxStretch, measured to the next column (so the last column takes the
  # one before's). The scan reads the columns of the vectorized loop.
  var start = 0
  if p.maxStretch < Inf and m > 1:
    start = m
    var before = NaN
    var j0 = fromZero
    block scan:
      while j0 < m:
        let len = min(blockLen, m - j0)
        columns(j0, len)
        for i in 0 ..< len:
          let d = cols[i] - before
          if d > 0 and 1 / d - 1 <= p.maxStretch:
            start = j0 + i - 1
            break scan
          before = cols[i]
        j0 += len
  let taper = p.taperMs / dtMs
  # The first column the ramp does not zero (the ramp's start has weight 0).
  let first =
    if start >= m: m
    elif taper > 0:
      let f = floor(float64(start) - taper) # a long taper: below any int
      if f < 0: 0 else: int(f) + 1
    else: start
  let lo = max(first, fromZero)
  for j in 0 ..< lo:
    dst[j] = T(0)
  template at(i: int): float64 =
    (if i >= 0 and i < n: float64(src[i]) else: 0.0)
  # A block all of whose columns have every tap inside the input takes the
  # branch-free kernels; one that touches the input's ends, or holds a NaN
  # column, the general path. (Whole column numbers are kept as float32:
  # exact below 2²⁴ samples.)
  let (inLo, inHi) =
    if p.interp == siLinear: (0.0, float64(n - 1))
    else: (1.0, float64(n - 2))
  let s = if n > 0: cast[ptr UncheckedArray[T]](addr src[0]) else: nil
  var j0 = lo
  while j0 < m:
    let len = min(blockLen, m - j0)
    columns(j0, len)
    let d = cast[ptr UncheckedArray[T]](addr dst[j0])
    if n <= (1 shl 24) and insideCount(cols, len, inLo, inHi) == len:
      when T is float32:
        if p.interp == siLinear: lerpCols32(d, s, addr cols, len)
        else: cubicCols32(d, s, addr cols, len)
      else:
        if p.interp == siLinear: lerpCols64(d, s, addr cols, len)
        else: cubicCols64(d, s, addr cols, len)
    else:
      for i in 0 ..< len:
        let c = cols[i]
        if c > -2 and c < float64(n + 1): # not NaN, and a tap in the input
          let k = int(floor(c))
          let f = c - float64(k)
          case p.interp
          of siLinear:
            d[i] = T((1 - f) * at(k) + f * at(k + 1))
          of siCubic:
            let f2 = f * f
            let f3 = f2 * f
            d[i] = T((-0.5 * f3 + f2 - 0.5 * f) * at(k - 1) +
              (1.5 * f3 - 2.5 * f2 + 1) * at(k) +
              (-1.5 * f3 + 2 * f2 + 0.5 * f) * at(k + 1) +
              (0.5 * f3 - 0.5 * f2) * at(k + 2))
        else:
          d[i] = T(0)
    j0 += len
  {.pop.}
  if start < m and taper > 0:
    muteTop(dst, float64(start), taper)
  first

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
  ## nothing. An `AssertionDefect` when `vrms.len != dst.len`. For many
  ## rows with one velocity column, `nmoSlothInto` saves the inversion.
  nmoImpl(src, srcStartMs, dst, dstStartMs, dtMs, offset, vrms, false, p)

proc nmoSlothInto*[T: SampleType](src: openArray[T], srcStartMs: float64,
    dst: var openArray[T], dstStartMs, dtMs, offset: float64,
    sloth: openArray[float64], p: NmoParams): int {.raises: [].} =
  ## `nmoInto` with the sloth `1/Vrms²` of each output sample (from
  ## `vrmsToSloth`) in place of its velocity: the same samples and return
  ## value, without a division per sample. An `AssertionDefect` when
  ## `sloth.len != dst.len`.
  nmoImpl(src, srcStartMs, dst, dstStartMs, dtMs, offset, sloth, true, p)

proc vrmsToSloth*(v: var openArray[float64]) {.raises: [].} =
  ## Replaces each RMS velocity of `v` by its sloth, `1/V²`, in place, for
  ## `nmoSlothInto` (a NaN stays NaN; a zero velocity gives Inf, which NMO
  ## zeroes like a NaN). Allocates nothing.
  var j0 = 0
  var sloth {.noinit.}: ColBlock
  while j0 < v.len:
    let len = min(blockLen, v.len - j0)
    slothCols(sloth, len, cast[ptr UncheckedArray[float64]](addr v[j0]))
    copyMem(addr v[j0], addr sloth[0], len * sizeof(float64))
    j0 += len

# -------------------------------------------------------------- block forms --

func nmoWorkspace*(T: typedesc[SampleType], cols: int): WorkspaceSize =
  ## The scratch the block forms need for rows of `cols` samples: a sloth
  ## column and a copy of the row.
  when T is float32: WorkspaceSize(f64: cols, f32: cols)
  else: WorkspaceSize(f64: 2 * cols)

proc rowScratch[T](ws: var Workspace, cols: int): ptr UncheckedArray[T] {.inline.} =
  when T is float32: ws.scratch(float32, cols)
  else: ws.scratch(float64, cols, at = cols)

proc nmoRow[T](b: RowBlock[T], meta: ChunkMeta, r: int, p: NmoParams,
    sloth: ptr UncheckedArray[float64], ws: var Workspace) {.inline.} =
  # Corrects the live part of row r in place, its sloths in `sloth`.
  let n = meta.liveSamples(r, b.cols)
  if n == 0: return
  let copy = rowScratch[T](ws, b.cols)
  copyMem(copy, b.rowPtr(r), n * sizeof(T))
  let start = meta.sampleTimeMs(r, 0)
  discard nmoSlothInto(copy.toOpenArray(0, n - 1), start, b.row(r, n), start,
    meta.dtMs, meta.keys(r).offset, sloth.toOpenArray(0, n - 1), p)

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
  let sloth = ws.scratch(float64, b.cols)
  var last = NaN # the start time `sloth` was computed for: rows share it
  for r in 0 ..< b.len:
    if meta.usable(r):
      let start = meta.sampleTimeMs(r, 0)
      if start != last:
        f.vrmsInto(start, meta.dtMs, sloth.toOpenArray(0, b.cols - 1))
        vrmsToSloth(sloth.toOpenArray(0, b.cols - 1))
        last = start
      nmoRow(b, meta, r, p, sloth, ws)

template fieldNmo(b, meta, field, p, ws: untyped, placeOf: untyped) =
  # NMO of every usable row with the field's velocities at `placeOf(keys)`:
  # (location, 0) for a 2D field, (inline, crossline) for a 3D one; a NaN
  # leaves the row alone.
  let sloth = ws.scratch(float64, b.cols)
  template col: untyped = sloth.toOpenArray(0, b.cols - 1)
  # The place and start time `sloth` was computed for: the rows of a gather
  # share them.
  var lastA, lastB, lastStart = NaN
  for r in 0 ..< b.len:
    if meta.usable(r):
      let start = meta.sampleTimeMs(r, 0)
      let (a, c) = placeOf(meta.keys(r))
      if not (a.isNaN or c.isNaN):
        if a != lastA or c != lastB or start != lastStart:
          case field.kind
          of fk2D: field.vrmsInto(a, start, meta.dtMs, col)
          of fk3D: field.vrmsInto(a, c, start, meta.dtMs, col)
          vrmsToSloth(col)
          (lastA, lastB, lastStart) = (a, c, start)
        nmoRow(b, meta, r, p, sloth, ws)

proc applyNmo*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    field: VelocityField, p: NmoParams, ws: var Workspace) =
  ## NMO with the field's RMS velocities at each row's location (`ensemble`
  ## for a 2D field, `inline` and `crossline` for a 3D one); rows with a NaN
  ## location are left alone. Otherwise as the `VelocityFunction` form.
  checkNmo(b, meta, p, ws)
  template byHeader(k: TraceKeys): (float64, float64) =
    (if field.kind == fk2D: (k.ensemble, 0.0) else: (k.inline, k.crossline))
  fieldNmo(b, meta, field, p, ws, byHeader)

proc applyNmo*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    field: VelocityField, binning: Binning, p: NmoParams, ws: var Workspace) =
  ## NMO with the field's RMS velocities at the location of each row's bin
  ## (`binning.locationOf`: the bin's CDP number for a 2D field, its inline
  ## and crossline for a 3D one), as a stack with `binning` reads them
  ## (`bruteStack`); rows in no bin are left alone. Raises `ValueError` as
  ## the other field form, or when the field does not suit the binning
  ## (`checkSuits`). Allocates nothing.
  checkNmo(b, meta, p, ws)
  checkSuits(field, binning)
  template byBin(k: TraceKeys): (float64, float64) =
    block:
      let loc = binning.locationOf(k)
      if field.kind == fk2D: (loc.cdp, 0.0) else: (loc.inline, loc.crossline)
  fieldNmo(b, meta, field, p, ws, byBin)

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

proc applyNmo*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    field: VelocityField, binning: Binning, p: NmoParams, ws: var Workspace) =
  ## `applyNmo` with a field read at each row's bin, on a `[rows, cols]`
  ## tensor, in place.
  meta.checkShape(samples)
  applyNmo(rowBlock(samples), meta, field, binning, p, ws)
