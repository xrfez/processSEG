## Moveout times for overlays: linear (LMO) and hyperbolic (NMO) moveout at
## a constant velocity, one time per trace from the offsets of a view
## (task 5.3; velocity functions and fields come with Phase 6).
##
## ```nim
## let first = linearMoveout(velocity = 1800, t0Ms = 20)   # first breaks
## let refl = hyperbolicMoveout(velocity = 2200, t0Ms = 800)
## var ticks = newSeq[float64](meta.len)                  # once; reused
## moveoutTimesInto(meta.rows, first, ticks)              # ms, per row
## moveoutColumnsInto(meta, refl, ticks)                  # block columns
## ```
##
## **Formulas,** with `x` the offset, `v` the velocity in distance units per
## second (the file's coordinate units) and times in ms:
##
##     linear:      t = t0 + 1000·|x| / v
##     hyperbolic:  t = sqrt(t0² + (1000·x / v)²)
##
## **Sign convention.** Offsets are read as parseSEG stores them: SEG-Y
## header offsets may be signed (one side of a split spread negative),
## offsets derived from coordinates are distances. The sign is kept in the
## keys and ignored here: both formulas use |x|, so the two sides of a split
## spread get the same times. A NaN offset (a SEG-D file without geometry)
## gives a NaN time; nothing defaults it.
##
## **Staleness.** The times derive from the keys: tag them with the file's
## `keysRevision` and recompute them after `applySurvey` or a header edit.

import std/math
import parseSEG
import ../core/types

type
  MoveoutKind* = enum
    ## The shape of a moveout curve.
    mvLinear     ## `t0 + |x|/v`, for first breaks and refractions
    mvHyperbolic ## `sqrt(t0² + x²/v²)`, for reflections

  Moveout* = object
    ## A moveout curve at a constant velocity; build it with
    ## `linearMoveout` or `hyperbolicMoveout`, which validate it.
    kind: MoveoutKind
    velocity: float64
    t0Ms: float64

func lmoTime*(offset, velocity, t0Ms: float64): float64 {.inline, raises: [].} =
  ## Linear moveout: `t0Ms + 1000·|offset| / velocity`, in ms.
  t0Ms + 1000 * abs(offset) / velocity

func nmoTime*(offset, velocity, t0Ms: float64): float64 {.inline, raises: [].} =
  ## Hyperbolic moveout: `sqrt(t0Ms² + (1000·offset / velocity)²)`, in ms.
  let tx = 1000 * offset / velocity
  sqrt(t0Ms * t0Ms + tx * tx)

proc checkMoveout(velocity, t0Ms: float64) =
  if velocity.classify notin {fcNormal, fcSubnormal} or velocity <= 0:
    raise newException(ValueError, "a moveout velocity must be finite and " &
      "positive, got " & $velocity)
  if t0Ms.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "a moveout t0Ms must be finite, got " & $t0Ms)

proc linearMoveout*(velocity: float64, t0Ms = 0.0): Moveout =
  ## Linear moveout at `velocity` (distance units per second) from
  ## `t0Ms` at zero offset. Raises `ValueError` unless `velocity` is finite
  ## and positive and `t0Ms` finite.
  checkMoveout(velocity, t0Ms)
  Moveout(kind: mvLinear, velocity: velocity, t0Ms: t0Ms)

proc hyperbolicMoveout*(velocity: float64, t0Ms = 0.0): Moveout =
  ## Hyperbolic moveout at `velocity` with the zero-offset time `t0Ms`.
  ## Raises `ValueError` as `linearMoveout`.
  checkMoveout(velocity, t0Ms)
  Moveout(kind: mvHyperbolic, velocity: velocity, t0Ms: t0Ms)

func kind*(m: Moveout): MoveoutKind {.inline, raises: [].} =
  ## Linear or hyperbolic.
  m.kind

func velocity*(m: Moveout): float64 {.inline, raises: [].} =
  ## The velocity, in distance units per second.
  m.velocity

func t0Ms*(m: Moveout): float64 {.inline, raises: [].} =
  ## The zero-offset time, in ms.
  m.t0Ms

func timeMs*(m: Moveout, offset: float64): float64 {.inline, raises: [].} =
  ## The curve's time at `offset`, in ms (NaN for a NaN offset). A default
  ## `Moveout` (velocity 0) gives Inf or NaN.
  case m.kind
  of mvLinear: lmoTime(offset, m.velocity, m.t0Ms)
  of mvHyperbolic: nmoTime(offset, m.velocity, m.t0Ms)

proc checkLen(n, want: int) =
  if n != want:
    raise newException(ValueError, "times has " & $n & " elements, the rows " & $want)

proc moveoutTimesInto*(rows: TraceRows, m: Moveout,
    times: var openArray[float64]) =
  ## Fills `times[i]` with the curve's time at row `i`'s offset, in ms, for
  ## every row (whatever its status). Raises `ValueError` when the view has
  ## no keys or `times.len != rows.len`. Allocates nothing.
  if not rows.hasKeys:
    raise newException(ValueError, "the rows have no keys: open the file with " &
      "keys = true (or call ensureKeys) before making the view")
  checkLen(times.len, rows.len)
  for i in 0 ..< rows.len:
    times[i] = m.timeMs(rows.keysAt(i).offset)

proc moveoutColumnsInto*(meta: ChunkMeta, m: Moveout,
    cols: var openArray[float64]) =
  ## Fills `cols[i]` with the curve's time at row `i` as a fractional column
  ## of the block, `(t − t0Ms(i)) / dtMs − firstSample`, ready to draw over
  ## the samples (NaN when the offset or the row's `t0Ms` is). Raises
  ## `ValueError` when `cols.len != meta.len`.
  checkLen(cols.len, meta.len)
  for i in 0 ..< meta.len:
    let k = meta.keys(i)
    cols[i] = (m.timeMs(k.offset) - k.t0Ms) / meta.dtMs - float64(
        meta.firstSample)
