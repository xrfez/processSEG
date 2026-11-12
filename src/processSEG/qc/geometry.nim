## Geometry QC from the trace keys: the header offset against the
## source–receiver distance of the coordinates, zero, negative and missing
## offsets, missing coordinates and a header midpoint that is not halfway
## between the source and the receiver, as a set of flags per trace.
##
## ```nim
## var checks = newSeq[GeometryCheck](meta.len)  # once; reused
## checkGeometryInto(meta, checks)               # one per row
## for row in flaggedRows(checks):               # rows with a `badGeometry` flag
##   echo row, " ", checks[row].flags, " ", checks[row].offsetError
## ```
##
## **What the keys hold.** parseSEG fills a missing or zero SEG-Y header
## offset, and a missing midpoint, from the coordinates when both ends are
## known (`deriveGeometry`), so a zero header offset is seen only without
## coordinates or with coincident ones, and a derived offset always agrees
## with the distance. SEG-D and SEG-2 offsets come from the coordinates.
##
## **Signs.** SEG-Y offsets are signed (negative on one side of a split
## spread): `gfNegativeOffset` reports the sign and is not in `badGeometry`.
## The comparison uses `|offset|`.
##
## **Tolerance.** An offset or midpoint mismatches when it is off by more
## than `max(tolerance, relTolerance · distance)` (in the coordinates'
## units): coordinates are often rounded, and the header offset to whole
## units. An infinite coordinate is not an unknown one (NaN): its distance
## is infinite, which no offset or midpoint matches.
##
## **Units.** The comparison is in the coordinates' units, which must be
## those of the offset: parseSEG does not convert SEG-Y coordinates given
## in arc-seconds or degrees (coordinate units 2-4), so such a file
## mismatches on every trace.
##
## **Midpoints.** The CDP X/Y of a processed or binned file is often the
## bin centre, not the midpoint (up to half a bin diagonal away), so
## `gfMidpointMismatch` is not in `badGeometry`.
##
## Non-live rows (dead, auxiliary, failed) are not checked: their flags are
## empty and their values NaN.

import std/math
import parseSEG
import ../core/types

type
  GeometryFlag* = enum
    ## A finding on one trace's geometry.
    gfNoOffset         ## The offset is unknown (NaN).
    gfZeroOffset       ## The offset is 0, or the source and receiver coincide
                       ## (their distance is within `tolerance`, so a real 0.5 m
                       ## offset counts with the default of 1).
    gfNegativeOffset   ## A signed (SEG-Y) offset below 0: normal on one side of a split spread.
    gfNoCoordinates    ## The source or the receiver coordinates are unknown.
    gfOffsetMismatch   ## `|offset|` differs from the source–receiver distance.
    gfMidpointMismatch ## The header midpoint is not halfway between source and receiver.

  GeometryFlags* = set[GeometryFlag] ## The findings on one trace.

  GeometryQc* = object
    ## The tolerances of `checkGeometry` (see the module documentation).
    tolerance*: float64 = 1.0     ## Absolute, in the coordinates' units.
    relTolerance*: float64 = 0.01 ## Relative to the distance.

  GeometryCheck* = object
    ## One trace's findings.
    flags*: GeometryFlags
    distance*: float64    ## Source–receiver distance from the coordinates; NaN unknown.
    offsetError*: float64 ## `|offset| − distance`; NaN when either is unknown.
    midpointError*: float64
      ## Distance of the header midpoint from the halfway point; NaN unknown.

const badGeometry* = {gfNoOffset, gfZeroOffset, gfOffsetMismatch}
  ## The flags `flaggedRows` selects by default: geometry that is wrong or
  ## missing for processing. A negative offset and missing coordinates
  ## alone are not, nor a midpoint mismatch (processed and 3D SEG-Y often
  ## hold the bin centre as the CDP X/Y: add `gfMidpointMismatch` when the
  ## file holds true midpoints).

func uncheckedGeometry(): GeometryCheck {.inline.} =
  GeometryCheck(distance: NaN, offsetError: NaN, midpointError: NaN)

proc checkTolerances(qc: GeometryQc) =
  if not (qc.tolerance >= 0 and qc.tolerance < Inf and qc.relTolerance >= 0 and
      qc.relTolerance < Inf):
    raise newException(ValueError, "geometry tolerances must be finite and ≥ 0, got " &
      $qc.tolerance & " and " & $qc.relTolerance)

func checkGeometry*(k: TraceKeys, qc = GeometryQc()): GeometryCheck {.raises: [].} =
  ## The findings on the trace with keys `k`, whatever its status. Negative
  ## or NaN tolerances flag every comparable trace (`checkGeometryInto`
  ## validates them).
  result = uncheckedGeometry()
  if k.offset.isNaN: result.flags.incl gfNoOffset
  elif k.offset == 0: result.flags.incl gfZeroOffset
  elif k.offset < 0: result.flags.incl gfNegativeOffset
  if not (k.hasSourceXY and k.hasReceiverXY):
    result.flags.incl gfNoCoordinates
    return
  let d = hypot(k.rcvX - k.srcX, k.rcvY - k.srcY)
  result.distance = d
  # An infinite coordinate makes the distance infinite: the relative part
  # would then accept any offset, so only the absolute one applies.
  let tol = if d < Inf: max(qc.tolerance, qc.relTolerance * d) else: qc.tolerance
  if d <= qc.tolerance: result.flags.incl gfZeroOffset
  if not k.offset.isNaN:
    result.offsetError = abs(k.offset) - d
    if not (abs(result.offsetError) <= tol): result.flags.incl gfOffsetMismatch
  if not (k.cmpX.isNaN or k.cmpY.isNaN):
    result.midpointError = hypot(k.cmpX - (k.srcX + k.rcvX) / 2,
      k.cmpY - (k.srcY + k.rcvY) / 2)
    if not (result.midpointError <= tol): result.flags.incl gfMidpointMismatch

proc checkGeometryInto*(meta: ChunkMeta, dst: var openArray[GeometryCheck],
    qc = GeometryQc()) =
  ## `checkGeometry` of every live row of `meta` into `dst[row]`; non-live
  ## rows get no flags and NaN values. Raises `ValueError` when `dst` is not
  ## `meta.len` long or a tolerance is negative, NaN or infinite.
  qc.checkTolerances
  if dst.len != meta.len:
    raise newException(ValueError, "dst holds " & $dst.len & " checks, the meta " &
      $meta.len & " rows")
  for row in 0 ..< meta.len:
    dst[row] = if meta.isLive(row): checkGeometry(meta.keys(row), qc)
               else: uncheckedGeometry()

proc checkGeometryInto*(rows: TraceRows, dst: var openArray[GeometryCheck],
    qc = GeometryQc()) =
  ## `checkGeometry` of every row of `rows`, whatever its status. Raises
  ## `ValueError` as the `ChunkMeta` form, and when the view has no keys.
  qc.checkTolerances
  if not rows.hasKeys: raise newException(ValueError, "the rows have no keys")
  if dst.len != rows.len:
    raise newException(ValueError, "dst holds " & $dst.len & " checks, the view " &
      $rows.len & " rows")
  for row in 0 ..< rows.len: dst[row] = checkGeometry(rows.keysAt(row), qc)

func flaggedRows*(checks: openArray[GeometryCheck], flags = badGeometry): seq[int] =
  ## The rows whose findings include one of `flags`, ascending.
  for row, c in checks:
    if (c.flags * flags) != {}: result.add row
