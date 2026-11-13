## Moveout times for overlays: linear (LMO) and hyperbolic (NMO) moveout at
## a constant velocity or from a velocity model (`velocity/model`), one time
## per trace from the offsets of a view; and LMO applied to and removed from
## the samples (`applyLmo`, `removeLmo`).
##
## ```nim
## let first = linearMoveout(velocity = 1800, t0Ms = 20)   # first breaks
## let refl = hyperbolicMoveout(velocity = 2200, t0Ms = 800)
## var ticks = newSeq[float64](meta.len)                  # once; reused
## moveoutTimesInto(meta.rows, first, ticks)              # ms, per row
## moveoutColumnsInto(meta, refl, ticks)                  # block columns
## let fromModel = hyperbolicMoveout(vfun, t0Ms = 800)    # Vrms(800 ms)
## moveoutTimesInto(meta, field, t0Ms = 800, ticks)       # per row's location
## moveoutTimesInto(meta, field, binning, t0Ms = 800, ticks) # per row's bin
## samples.applyLmo(meta, velocity = 1800, biasMs = 50)   # flatten first breaks
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
## keys and ignored here: both formulas use `|x|`, so the two sides of a split
## spread get the same times. A NaN offset (a SEG-D file without geometry)
## gives a NaN time; nothing defaults it.
##
## **Models.** A hyperbola from a velocity model takes the RMS velocity at
## its zero-offset time, `t = sqrt(t0² + (1000·x / Vrms(t0))²)`. From a
## `VelocityField`, each row takes it at its own location: the keys'
## `ensemble` (the CDP) for a 2D field, `inline` and `crossline` for a 3D
## one, or caller locations (a distance along the line, from binning). A
## NaN location gives a NaN time. A stack does not read its field there
## when its bins come from midpoints (`lineBinning`, `gridBinning`): it
## reads it at each bin's numbers. The forms with a `Binning` do the same
## (each row at the location of its bin, `Binning.locationOf`), so a
## display can show the hyperbolas the stack will correct; a row in no bin
## gets NaN.
##
## **LMO on samples.** `applyLmo` moves each live row earlier by
## `1000·|x|/v − biasMs` (so a linear event at `t0 + 1000·|x|/v` lands at
## `t0 + biasMs`, flat); `removeLmo` moves it back. The shift is fractional
## (`core/shift`, linear or cubic), samples moved in from outside the live
## part are zero, and an apply then remove loses what was moved out of the
## row: use `biasMs` to keep the early part. Rows with a NaN offset, and
## dead, auxiliary and failed rows, are left alone.
##
## **Staleness.** The times derive from the keys: tag them with the file's
## `keysRevision` and recompute them after `applySurvey` or a header edit.

import std/math
import parseSEG
import ../core/[rows, shift, types]
import ../stack/binning
import model

export ShiftInterp

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
  ## Hyperbolic moveout at `velocity` with the zero-offset time `t0Ms`
  ## (≥ 0: a two-way time; a negative one would act as `|t0Ms|`). Raises
  ## `ValueError` as `linearMoveout`, and for `t0Ms < 0`.
  checkMoveout(velocity, t0Ms)
  if t0Ms < 0:
    raise newException(ValueError, "a hyperbolic moveout needs t0Ms ≥ 0, got " & $t0Ms)
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

proc moveoutTimesInto*(meta: ChunkMeta, m: Moveout,
    times: var openArray[float64]) =
  ## `moveoutTimesInto(meta.rows, m, times)`: the times in ms for the rows
  ## of a meta (which always has keys). Raises `ValueError` when
  ## `times.len != meta.len`.
  moveoutTimesInto(meta.rows, m, times)

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

# ------------------------------------------------------- from velocity models --

proc hyperbolicMoveout*(f: VelocityFunction, t0Ms: float64): Moveout =
  ## The hyperbola through `t0Ms` at zero offset with the RMS velocity of
  ## `f` at `t0Ms`. Raises `ValueError` as the constant form (a default `f`
  ## gives a NaN velocity).
  hyperbolicMoveout(f.vrmsAt(t0Ms), t0Ms)

proc checkT0(t0Ms: float64) =
  if t0Ms.classify in {fcNan, fcInf, fcNegInf} or t0Ms < 0:
    raise newException(ValueError, "a hyperbolic moveout needs a finite " &
      "t0Ms ≥ 0, got " & $t0Ms)

func fieldVrms(field: VelocityField, k: TraceKeys,
    t0Ms: float64): float64 {.inline.} =
  case field.kind
  of fk2D: field.vrmsAt(k.ensemble, t0Ms)
  of fk3D: field.vrmsAt(k.inline, k.crossline, t0Ms)

proc moveoutTimesInto*(rows: TraceRows, field: VelocityField, t0Ms: float64,
    times: var openArray[float64]) =
  ## Fills `times[i]` with the hyperbola through `t0Ms` at row `i`'s
  ## offset, with the field's RMS velocity at the row's location (`ensemble`
  ## in 2D, `inline` and `crossline` in 3D) and `t0Ms`, in ms; NaN where the
  ## location or offset is. Raises `ValueError` without keys, for a length
  ## mismatch or a `t0Ms` that is negative or not finite.
  if not rows.hasKeys:
    raise newException(ValueError, "the rows have no keys: open the file with " &
      "keys = true (or call ensureKeys) before making the view")
  checkLen(times.len, rows.len)
  checkT0(t0Ms)
  for i in 0 ..< rows.len:
    let k = rows.keysAt(i)
    times[i] = nmoTime(k.offset, field.fieldVrms(k, t0Ms), t0Ms)

proc moveoutTimesInto*(meta: ChunkMeta, field: VelocityField, t0Ms: float64,
    times: var openArray[float64]) =
  ## `moveoutTimesInto(meta.rows, field, t0Ms, times)`.
  moveoutTimesInto(meta.rows, field, t0Ms, times)

proc moveoutTimesInto*(rows: TraceRows, field: VelocityField,
    locations: openArray[float64], t0Ms: float64, times: var openArray[float64]) =
  ## The 2D form with caller locations: row `i` is at `locations[i]` (a
  ## distance along the line, say). Raises `ValueError` as the other form,
  ## for a 3D field, or when `locations.len != rows.len`.
  if not rows.hasKeys:
    raise newException(ValueError, "the rows have no keys: open the file with " &
      "keys = true (or call ensureKeys) before making the view")
  if field.kind != fk2D:
    raise newException(ValueError, "locations need a 2D field")
  checkLen(times.len, rows.len)
  if locations.len != rows.len:
    raise newException(ValueError, "locations has " & $locations.len &
      " elements, the rows " & $rows.len)
  checkT0(t0Ms)
  for i in 0 ..< rows.len:
    times[i] = nmoTime(rows.keysAt(i).offset, field.vrmsAt(locations[i], t0Ms), t0Ms)

proc toColumns(meta: ChunkMeta, cols: var openArray[float64]) =
  # Times in ms into fractional block columns, row by row.
  for i in 0 ..< meta.len:
    cols[i] = (cols[i] - meta.keys(i).t0Ms) / meta.dtMs - float64(
        meta.firstSample)

proc moveoutColumnsInto*(meta: ChunkMeta, field: VelocityField, t0Ms: float64,
    cols: var openArray[float64]) =
  ## The field's hyperbolas as fractional block columns, as the constant
  ## `moveoutColumnsInto`. Raises `ValueError` as `moveoutTimesInto`.
  moveoutTimesInto(meta, field, t0Ms, cols)
  toColumns(meta, cols)

# ------------------------------------------ a field read as a stack reads it --

proc checkSuits*(field: VelocityField, binning: Binning) =
  ## Raises `ValueError` unless `field` can be read at the bins of
  ## `binning`: it has nodes, and it is 2D for a 2D binning (`cdpBinning`,
  ## `lineBinning`: read at the bin's CDP number) or 3D for a 3D one
  ## (`gridBinning`, `inlineCrosslineBinning`: at its inline and crossline).
  if field.len == 0:
    raise newException(ValueError, "the velocity field is empty")
  if field.kind == fk2D and binning.is3D:
    raise newException(ValueError, "a 2D velocity field needs a 2D binning " &
      "(cdpBinning or lineBinning)")
  if field.kind == fk3D and not binning.is3D:
    raise newException(ValueError, "a 3D velocity field needs a 3D binning " &
      "(gridBinning or inlineCrosslineBinning)")

proc moveoutTimesInto*(rows: TraceRows, field: VelocityField, binning: Binning,
    t0Ms: float64, times: var openArray[float64]) =
  ## Fills `times[i]` with the hyperbola through `t0Ms` at row `i`'s offset,
  ## with the field's RMS velocity at the location of the row's bin
  ## (`binning.locationOf`: the bin's CDP number for a 2D field, its inline
  ## and crossline for a 3D one) and `t0Ms`: the velocity a stack with
  ## `binning` corrects the row with. NaN for a row in no bin or with a NaN
  ## offset. Raises `ValueError` without keys, for a length mismatch, a
  ## `t0Ms` that is negative or not finite, or a field that does not suit
  ## the binning (`checkSuits`). Allocates nothing.
  if not rows.hasKeys:
    raise newException(ValueError, "the rows have no keys: open the file with " &
      "keys = true (or call ensureKeys) before making the view")
  checkLen(times.len, rows.len)
  checkT0(t0Ms)
  checkSuits(field, binning)
  for i in 0 ..< rows.len:
    let k = rows.keysAt(i)
    let loc = binning.locationOf(k)
    let v =
      case field.kind
      of fk2D: field.vrmsAt(loc.cdp, t0Ms)
      of fk3D: field.vrmsAt(loc.inline, loc.crossline, t0Ms)
    times[i] = nmoTime(k.offset, v, t0Ms)

proc moveoutTimesInto*(meta: ChunkMeta, field: VelocityField, binning: Binning,
    t0Ms: float64, times: var openArray[float64]) =
  ## `moveoutTimesInto(meta.rows, field, binning, t0Ms, times)`.
  moveoutTimesInto(meta.rows, field, binning, t0Ms, times)

proc moveoutColumnsInto*(meta: ChunkMeta, field: VelocityField,
    binning: Binning, t0Ms: float64, cols: var openArray[float64]) =
  ## The hyperbolas of the `Binning` form of `moveoutTimesInto` as
  ## fractional block columns, ready to draw. Raises `ValueError` as it
  ## does.
  moveoutTimesInto(meta, field, binning, t0Ms, cols)
  toColumns(meta, cols)

# -------------------------------------------------------------- LMO on samples --

proc checkLmo(velocity, biasMs: float64) =
  checkMoveout(velocity, 0)
  if biasMs.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "biasMs must be finite, got " & $biasMs)

proc shiftLmo[T: SampleType](b: RowBlock[T], meta: ChunkMeta, velocity,
    biasMs: float64, interp: ShiftInterp, sign: float64) =
  checkLmo(velocity, biasMs)
  meta.checkShape(b)
  for r in 0 ..< b.len:
    if meta.isLive(r):
      let s = (1000 * abs(meta.keys(r).offset) / velocity - biasMs) / meta.dtMs
      shiftSamples(b.row(r, meta.liveSamples(r, b.cols)), sign * s, interp)

proc applyLmo*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    velocity: float64, biasMs = 0.0, interp = siLinear) =
  ## Linear moveout correction, in place: every live row moves earlier by
  ## `1000·|offset|/velocity − biasMs` ms (see the module documentation).
  ## Raises `ValueError` when `b` and `meta` disagree, `velocity` is not
  ## finite and positive or `biasMs` not finite. Allocates nothing.
  shiftLmo(b, meta, velocity, biasMs, interp, 1)

proc removeLmo*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    velocity: float64, biasMs = 0.0, interp = siLinear) =
  ## The inverse of `applyLmo`: every live row moves later by the same
  ## shift. Raises `ValueError` as `applyLmo`.
  shiftLmo(b, meta, velocity, biasMs, interp, -1)

proc applyLmo*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    velocity: float64, biasMs = 0.0, interp = siLinear) =
  ## `applyLmo` on a `[rows, cols]` tensor, in place. Raises `ValueError`
  ## unless it is C-contiguous with `meta.len` rows, or as the block form.
  meta.checkShape(samples)
  applyLmo(rowBlock(samples), meta, velocity, biasMs, interp)

proc removeLmo*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    velocity: float64, biasMs = 0.0, interp = siLinear) =
  ## `removeLmo` on a `[rows, cols]` tensor, in place; see `applyLmo`.
  meta.checkShape(samples)
  removeLmo(rowBlock(samples), meta, velocity, biasMs, interp)
