## Top and bottom mutes, with a cos² taper.
##
## A mute is a time per trace, given by a `MuteCurve`:
##
## - `timeMute(tMs)`: one time for every trace;
## - `tableMute(offsets, timesMs)`: a t(offset) table, linear in `|offset|`
##   between the knots and flat beyond them;
## - `lmoMute(velocity, t0Ms, padMs)`: `t0Ms + |offset|/velocity + padMs`, a
##   mute that follows the first breaks (a negative pad mutes above them).
##
## Times are trace times in ms, on the axis of `ChunkMeta.sampleTimeMs`:
## column `j` of row `r` is at `t0Ms(r) + (firstSample + j) · dtMs`, so a mute
## follows each row's `t0Ms` and the window origin.
##
## **Top mute** at time `t`, taper `L`: samples from `t` on are untouched;
## the taper lies on the muted side, scaling `t − L ≤ s < t` by
## `sin²(π/2 · (s − (t − L))/L)` (0 at `t − L`, rising towards 1 at `t`), and
## earlier samples are zeroed. **Bottom mute** mirrors it: samples up to `t`
## are untouched, the taper falls from 1 at `t` to 0 at `t + L`, and later
## samples are zeroed. So `t` is always the edge of the untouched data, and a
## taper shrinking to 0 gives the sharp cut in the limit.
##
## The block forms process live rows only, up to their live length. A row
## whose mute time is NaN (an offset or `t0Ms` the headers cannot give) is
## left untouched: no default is substituted.

import std/math
import parseSEG
import rows, types

type
  MuteKind* = enum
    ## How a `MuteCurve` gives each trace's time.
    mkTime  ## one time for every trace
    mkTable ## linear in |offset| between knots, flat beyond
    mkLmo   ## `t0 + |offset|/velocity + pad`

  MuteCurve* = object
    ## The mute time of each trace; see the module documentation. Build one
    ## with `timeMute`, `tableMute` or `lmoMute`, which validate it.
    case kind: MuteKind
    of mkTime:
      tMs: float64
    of mkTable:
      offsets: seq[float64] ## |offset| knots, strictly ascending, ≥ 0
      timesMs: seq[float64]
    of mkLmo:
      msPerDist: float64 ## 1000/velocity, in ms per distance unit
      t0Ms, padMs: float64

proc badMute(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func finite(x: float64): bool {.inline.} =
  x.classify notin {fcNan, fcInf, fcNegInf}

proc timeMute*(tMs: float64): MuteCurve =
  ## A mute at `tMs` on every trace. Raises `ValueError` unless `tMs` is
  ## finite.
  if not finite(tMs): badMute("a mute time must be finite, got " & $tMs)
  MuteCurve(kind: mkTime, tMs: tMs)

proc tableMute*(offsets, timesMs: openArray[float64]): MuteCurve =
  ## A mute at a time interpolated linearly in `|offset|` between the knots
  ## `(offsets[i], timesMs[i])`, flat beyond the first and last knot. The
  ## offsets are absolute (≥ 0) and strictly ascending; a trace's offset is
  ## used as `|offset|`. Raises `ValueError` for no knots, a length mismatch, a
  ## negative, NaN or non-ascending offset, or a time that is not finite.
  if offsets.len == 0: badMute("a mute table needs at least one knot")
  if offsets.len != timesMs.len:
    badMute("a mute table has " & $offsets.len & " offsets and " &
      $timesMs.len & " times")
  for i in 0..<offsets.len:
    if not finite(offsets[i]) or offsets[i] < 0:
      badMute("mute table offsets must be finite and ≥ 0, got " & $offsets[i])
    if i > 0 and offsets[i] <= offsets[i - 1]:
      badMute("mute table offsets must be strictly ascending: " &
        $offsets[i - 1] & " is followed by " & $offsets[i])
    if not finite(timesMs[i]):
      badMute("mute table times must be finite, got " & $timesMs[i])
  MuteCurve(kind: mkTable, offsets: @offsets, timesMs: @timesMs)

proc lmoMute*(velocity: float64, t0Ms = 0.0, padMs = 0.0): MuteCurve =
  ## A mute at `t0Ms + |offset|/velocity + padMs`, with `velocity` in distance
  ## units per second (the file's coordinate units). Raises `ValueError`
  ## unless `velocity` is finite and positive and the times are finite.
  if not finite(velocity) or velocity <= 0:
    badMute("an LMO mute velocity must be finite and positive, got " & $velocity)
  if not finite(t0Ms) or not finite(padMs):
    badMute("LMO mute times must be finite, got t0Ms " & $t0Ms & ", padMs " &
      $padMs)
  MuteCurve(kind: mkLmo, msPerDist: 1000 / velocity, t0Ms: t0Ms, padMs: padMs)

func kind*(c: MuteCurve): MuteKind {.inline, raises: [].} =
  ## How the curve gives each trace's time.
  c.kind

func muteTimeMs*(c: MuteCurve, offset: float64): float64 {.raises: [].} =
  ## The mute time, in ms, of a trace at `offset` (signed or not: `|offset|` is
  ## used). NaN when the curve depends on the offset and `offset` is NaN.
  case c.kind
  of mkTime:
    c.tMs
  of mkLmo:
    c.t0Ms + abs(offset) * c.msPerDist + c.padMs
  of mkTable:
    let x = abs(offset)
    if x.isNaN: return NaN
    let last = c.offsets.high
    if x <= c.offsets[0]: return c.timesMs[0]
    if x >= c.offsets[last]: return c.timesMs[last]
    var lo = 0 # the last knot ≤ x (binary search)
    var hi = last
    while hi - lo > 1:
      let mid = (lo + hi) shr 1
      if c.offsets[mid] <= x: lo = mid else: hi = mid
    let u = (x - c.offsets[lo]) / (c.offsets[hi] - c.offsets[lo])
    c.timesMs[lo] + u * (c.timesMs[hi] - c.timesMs[lo])

# ------------------------------------------------------------------ kernels --

func firstAtOrAfter(c: float64, n: int): int {.inline.} =
  ## The first column ≥ `c`, clamped to `0 .. n` (0 for NaN).
  if not (c > 0): 0
  elif c >= float64(n): n
  else: int(ceil(c))

func firstAfter(c: float64, n: int): int {.inline.} =
  ## The first column > `c`, clamped to `0 .. n` (0 for NaN).
  if not (c >= 0): 0
  elif c >= float64(n): n
  else: int(floor(c)) + 1

proc muteTop*[T: SampleType](x: var openArray[T], startCol,
    taperCols: float64) {.raises: [].} =
  ## Top mute of one row in column units: `x[j]` for `j ≥ startCol` is
  ## untouched, `startCol − taperCols ≤ j < startCol` is scaled by
  ## `sin²(π/2 · (j − startCol + taperCols)/taperCols)`, and earlier columns
  ## are zeroed. Columns may be fractional. A NaN `startCol` leaves `x`
  ## untouched; `taperCols` must be ≥ 0, and an infinite one is no taper
  ## (its weights are 0 everywhere before `startCol`).
  if startCol.isNaN: return
  let n = x.len
  let taperCols = if taperCols < Inf: taperCols else: 0.0
  let rampStart = startCol - taperCols
  let zeroEnd = firstAtOrAfter(rampStart, n)
  {.push boundChecks: off.}
  for j in 0..<zeroEnd:
    x[j] = T(0)
  if taperCols > 0:
    for j in zeroEnd..<firstAtOrAfter(startCol, n):
      let w = sin(PI / 2 * (float64(j) - rampStart) / taperCols) ^ 2
      x[j] = T(float64(x[j]) * w)
  {.pop.}

proc muteBottom*[T: SampleType](x: var openArray[T], endCol,
    taperCols: float64) {.raises: [].} =
  ## Bottom mute of one row in column units: `x[j]` for `j ≤ endCol` is
  ## untouched, `endCol < j ≤ endCol + taperCols` is scaled by
  ## `sin²(π/2 · (endCol + taperCols − j)/taperCols)`, and later columns are
  ## zeroed. Columns may be fractional. A NaN `endCol` leaves `x` untouched;
  ## `taperCols` must be ≥ 0, and an infinite one is no taper (its weights
  ## are 0 everywhere after `endCol`).
  if endCol.isNaN: return
  let n = x.len
  let taperCols = if taperCols < Inf: taperCols else: 0.0
  let rampEnd = endCol + taperCols
  let zeroStart = firstAfter(rampEnd, n)
  {.push boundChecks: off.}
  if taperCols > 0:
    for j in firstAfter(endCol, n)..<zeroStart:
      let w = sin(PI / 2 * (rampEnd - float64(j)) / taperCols) ^ 2
      x[j] = T(float64(x[j]) * w)
  for j in zeroStart..<n:
    x[j] = T(0)
  {.pop.}

# -------------------------------------------------------------- block forms --

proc checkTaper(taperMs: float64) =
  if not finite(taperMs) or taperMs < 0:
    badMute("a mute taper must be finite and ≥ 0 ms, got " & $taperMs)

func muteCol(meta: ChunkMeta, row: int, c: MuteCurve): float64 {.inline.} =
  ## The column of row `row` at the curve's time (fractional; NaN when the
  ## time or `t0Ms` is).
  let k = meta.keys(row)
  let t = c.muteTimeMs(k.offset)
  (t - k.t0Ms) / meta.dtMs - float64(meta.firstSample)

proc muteTop*[T: SampleType](b: RowBlock[T], meta: ChunkMeta, curve: MuteCurve,
    taperMs = 0.0) =
  ## Top mute of every live row of `b` at its `curve` time, with a taper of
  ## `taperMs` before it (on the muted side); see the module documentation. Raises `ValueError`
  ## when `b` and `meta` disagree or `taperMs` is negative or not finite.
  checkTaper(taperMs)
  meta.checkShape(b)
  let taperCols = taperMs / meta.dtMs
  for r in 0..<b.len:
    if meta.isLive(r):
      muteTop(b.row(r, meta.liveSamples(r, b.cols)), meta.muteCol(r, curve),
        taperCols)

proc muteBottom*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    curve: MuteCurve, taperMs = 0.0) =
  ## Bottom mute of every live row of `b` at its `curve` time, with a taper
  ## of `taperMs` after it (on the muted side); see `muteTop`.
  checkTaper(taperMs)
  meta.checkShape(b)
  let taperCols = taperMs / meta.dtMs
  for r in 0..<b.len:
    if meta.isLive(r):
      muteBottom(b.row(r, meta.liveSamples(r, b.cols)), meta.muteCol(r, curve),
        taperCols)

proc muteTop*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    curve: MuteCurve, taperMs = 0.0) =
  ## Top mute of a `[rows, cols]` tensor, in place; see the `RowBlock` form.
  ## Raises `ValueError` unless `samples` is C-contiguous with `meta.len` rows.
  meta.checkShape(samples)
  muteTop(rowBlock(samples), meta, curve, taperMs)

proc muteBottom*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    curve: MuteCurve, taperMs = 0.0) =
  ## Bottom mute of a `[rows, cols]` tensor, in place; see `muteTop`.
  meta.checkShape(samples)
  muteBottom(rowBlock(samples), meta, curve, taperMs)
