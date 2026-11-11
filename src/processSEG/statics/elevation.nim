## Datum (elevation) statics: per-trace time corrections that move the
## source and the receiver of every trace onto a flat datum, at a
## replacement velocity, applied as fractional time shifts (`core/shift`).
##
## ```nim
## let d = datumStatics(datumZ = 100, vReplacement = 1800)
## var st = newSeq[float64](meta.len)        # once; reused
## staticsInto(meta, d, st)                  # ms per row, from srcZ and rcvZ
## applyStatics(samples, meta, st)           # events move by st[row] ms
## ```
##
## **Sign convention.** A static is the time, in ms, **added** to every
## event of the trace: the sample at `t` moves to `t + static`. A station
## above the datum delays the events, so its correction is negative:
##
##     static = −((srcZ − datumZ) + (rcvZ − datumZ)) · 1000 / vReplacement
##
## with elevations in the file's units and `vReplacement` in those units per
## second. A source buried at `srcDepth` below the surface (from parseSEG's
## `traceStatics`) counts at `srcZ − srcDepth`, a buried receiver at
## `rcvZ − rcvDepth`.
##
## **Header columns.** parseSEG reads the SEG-Y datum, depth and statics
## fields on demand (`traceStatics`; NaN where unknown, and for SEG-D and
## SEG-2). The `TraceStatics` forms take a trace's own datum elevations
## (`srcDatum`, `rcvDatum`) and the source depth where the headers give
## them, and the datum of `DatumStatics` otherwise. `headerStaticMs` is the
## source plus group static the headers record (bytes 99-102), for data
## whose statics were computed elsewhere; `totalStatic` (bytes 103-104) is
## what was already applied to the samples, which processSEG never undoes
## on its own.
##
## **NaN.** An unknown elevation or velocity gives a NaN static, and a NaN
## static leaves its row alone: nothing defaults it (`bruteStack` skips
## such traces).
##
## **Batch.** `bruteStack` takes a `DatumStatics` only; per-trace datums
## and header statics are for callers that apply `applyStatics` themselves.
##
## Uphole and refraction (weathering) statics are not computed here.

import std/math
import parseSEG
import ../core/[rows, shift, types]

export ShiftInterp

type
  DatumStatics* = object
    ## A flat datum and a replacement velocity; build it with
    ## `datumStatics`, which validates them.
    datumZ: float64
    vReplacement: float64

proc datumStatics*(datumZ, vReplacement: float64): DatumStatics =
  ## Statics to the elevation `datumZ` at `vReplacement` (distance units per
  ## second). Raises `ValueError` unless `datumZ` is finite and
  ## `vReplacement` finite and positive.
  if datumZ.classify in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "datumZ must be finite, got " & $datumZ)
  if vReplacement.classify notin {fcNormal, fcSubnormal} or vReplacement <= 0:
    raise newException(ValueError, "vReplacement must be finite and positive, got " &
      $vReplacement)
  DatumStatics(datumZ: datumZ, vReplacement: vReplacement)

func datumZ*(d: DatumStatics): float64 {.inline, raises: [].} =
  ## The datum elevation.
  d.datumZ

func vReplacement*(d: DatumStatics): float64 {.inline, raises: [].} =
  ## The replacement velocity, in distance units per second.
  d.vReplacement

func stationStaticMs(z, datum, v: float64): float64 {.inline.} =
  -(z - datum) * 1000 / v

func staticMs*(d: DatumStatics, k: TraceKeys): float64 {.inline, raises: [].} =
  ## The datum static of a trace, in ms (see the module documentation for
  ## the sign): NaN when `srcZ` or `rcvZ` is.
  stationStaticMs(k.srcZ, d.datumZ, d.vReplacement) +
    stationStaticMs(k.rcvZ, d.datumZ, d.vReplacement)

func staticMs*(d: DatumStatics, k: TraceKeys, st: TraceStatics): float64 {.
    raises: [].} =
  ## The datum static of a trace with its header columns: the trace's own
  ## `srcDatum` and `rcvDatum` where they are finite (else `d.datumZ`), and
  ## the source at `srcZ − srcDepth` and the receiver at `rcvZ − rcvDepth`
  ## where those depths are finite.
  let srcDatum = if st.srcDatum.isNaN: d.datumZ else: st.srcDatum
  let rcvDatum = if st.rcvDatum.isNaN: d.datumZ else: st.rcvDatum
  let srcZ = if st.srcDepth.isNaN: k.srcZ else: k.srcZ - st.srcDepth
  let rcvZ = if st.rcvDepth.isNaN: k.rcvZ else: k.rcvZ - st.rcvDepth
  stationStaticMs(srcZ, srcDatum, d.vReplacement) +
    stationStaticMs(rcvZ, rcvDatum, d.vReplacement)

func headerStaticMs*(st: TraceStatics): float64 {.inline, raises: [].} =
  ## The source plus group static the headers record, in ms; NaN when both
  ## are unknown, the known one when only one is.
  if st.srcStatic.isNaN: st.rcvStatic
  elif st.rcvStatic.isNaN: st.srcStatic
  else: st.srcStatic + st.rcvStatic

proc checkLen(n, want: int, what: string) =
  if n != want:
    raise newException(ValueError, what & " has " & $n &
        " elements, the rows " & $want)

proc staticsInto*(meta: ChunkMeta, d: DatumStatics,
    staticsMs: var openArray[float64]) =
  ## Fills `staticsMs[i]` with row `i`'s datum static, in ms, for every row
  ## (whatever its status). Raises `ValueError` when `staticsMs.len !=
  ## meta.len`. Allocates nothing.
  checkLen(staticsMs.len, meta.len, "staticsMs")
  for i in 0 ..< meta.len:
    staticsMs[i] = d.staticMs(meta.keys(i))

proc staticsInto*(meta: ChunkMeta, d: DatumStatics,
    statics: openArray[TraceStatics], staticsMs: var openArray[float64]) =
  ## The form with header columns: `statics[i]` is row `i`'s
  ## (`traceStaticsInto(f, traces, statics)` for the same traces). Raises
  ## `ValueError` when `statics` or `staticsMs` is not `meta.len` long.
  checkLen(statics.len, meta.len, "statics")
  checkLen(staticsMs.len, meta.len, "staticsMs")
  for i in 0 ..< meta.len:
    staticsMs[i] = d.staticMs(meta.keys(i), statics[i])

proc headerStaticsInto*(statics: openArray[TraceStatics],
    staticsMs: var openArray[float64]) =
  ## Fills `staticsMs[i]` with `headerStaticMs(statics[i])`, for
  ## `applyStatics` with the statics the headers record. Raises `ValueError`
  ## when the lengths differ.
  checkLen(staticsMs.len, statics.len, "staticsMs")
  for i in 0 ..< statics.len:
    staticsMs[i] = headerStaticMs(statics[i])

proc applyStatics*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    staticsMs: openArray[float64], interp = siLinear) =
  ## Shifts every live row `r` of `b` by `staticsMs[r]` ms, in place: the
  ## sample at `t` moves to `t + staticsMs[r]` (fractional shifts by linear
  ## or cubic interpolation, zeros moved in at the ends of the live part).
  ## A NaN static, and dead, auxiliary and failed rows, are left alone.
  ## Raises `ValueError` when `b`, `meta` and `staticsMs` disagree.
  ## Allocates nothing.
  meta.checkShape(b)
  checkLen(staticsMs.len, meta.len, "staticsMs")
  for r in 0 ..< b.len:
    if meta.isLive(r):
      shiftSamples(b.row(r, meta.liveSamples(r, b.cols)),
        -staticsMs[r] / meta.dtMs, interp)

proc applyStatics*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    staticsMs: openArray[float64], interp = siLinear) =
  ## `applyStatics` on a `[rows, cols]` tensor, in place. Raises
  ## `ValueError` unless it is C-contiguous with `meta.len` rows, or as the
  ## block form.
  meta.checkShape(samples)
  applyStatics(rowBlock(samples), meta, staticsMs, interp)

proc applyStatics*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    d: DatumStatics, interp = siLinear) =
  ## Datum statics computed and applied row by row, with no statics buffer:
  ## `applyStatics(b, meta, staticsInto(meta, d))`. Raises `ValueError`
  ## when `b` and `meta` disagree. Allocates nothing.
  meta.checkShape(b)
  for r in 0 ..< b.len:
    if meta.isLive(r):
      shiftSamples(b.row(r, meta.liveSamples(r, b.cols)),
        -d.staticMs(meta.keys(r)) / meta.dtMs, interp)

proc applyStatics*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    d: DatumStatics, interp = siLinear) =
  ## The datum form on a `[rows, cols]` tensor, in place.
  meta.checkShape(samples)
  applyStatics(rowBlock(samples), meta, d, interp)
