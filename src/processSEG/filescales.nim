## What a file's own headers say about the scale of its samples (decision
## D15): the per-trace `FileScale` column a calibration reads
## (`fileScalesInto`), a report of the raw header values behind it
## (`scaleReport`), and the SEG-Y header values that state the unit of
## calibrated samples written back (`segyUnitOverrides`). Like `adapters`,
## this module reads a `SeismicFile`, through parseSEG's public API.
##
## ```nim
## for g in f.scaleReport(): echo g          # check a file against a spec sheet
## var scales = newSeq[FileScale](c.rows.len) # once per chunk, reused
## f.fileScalesInto(c.rows, scales)
## let meta = chunkMeta(c, scales = scales)  # the meta points at the column
## ```
##
## **What is read, per standard.**
## - **SEG-D Rev 1/2:** the trace's channel set's MP factor: 2^MP mV per
##   count (MP 0 reads as 1 mV per count: legal, but for integer samples a
##   sign that the recorder left it unset; the report shows the format code
##   beside it). No sensitivity: the standard headers have none.
## - **SEG-D Rev 3:** the channel set's descale multiplier (DSM, mV per count;
##   0 or NaN is unknown here, where parseSEG's `descaleFactor` gives 1), and
##   the trace's Sensor Info block (0x41) sensor sensitivity STY, 0 =
##   unknown, taken as mV per physical unit: SEG-D 3.0 gives the physical
##   value as `sample · DSM / STY`, but its Sensor Info table says only
##   `sample / STY`. This reading is an **assumption**, not yet checked
##   against a recorder's file that has the block. The unit is THE#1 byte
##   31, else the channel set's byte 62 (millibar, bar, mm/s, m/s, mm/s²,
##   m/s², Pa, µPa; others are unknown). A unit without a sensitivity is
##   kept in `FileScale.unit`, so a rule giving a sensitivity of another
##   quantity is flagged. The block is read through parseSEG's
##   `traceHeaders` listing, which allocates: about microseconds per trace,
##   only for traces with extension blocks after THE#1.
## - **SEG-Y:** trace bytes 169-170 (N, the trace weighting factor: one count
##   is 2^−N), 203-204 (the trace value unit), 205-212 (the transduction
##   constant `C`, mantissa × 10^exponent, and its units): the physical value
##   is taken as `sample · 2^−N · C` (N applied before C: the standard does
##   not say, and N is 0 in most files). A count is `2^−N` of the value unit (V or mV); a
##   value unit of Pa, m/s, m/s² or N makes the samples already physical
##   (`FileScale.physical`). A value unit of 0 (unknown) is volts only when
##   N ≠ 0 (Rev 1 defines N as the LSB in volts); the standard reads headers
##   that are all zero as volts, but here they are **unknown**, since that is
##   what an unfilled header looks like. `C` in Pa, m/s, m/s² or N gives the
##   sensitivity `v / C` (V per unit, `v` the volts of one value unit), or,
##   when the volts are unknown, makes the samples physical directly
##   (`2^−N · C` per count); `C` in V or mV gives the descale; `C` with
##   unknown units is ignored (the report shows it).
## - **SEG-2:** DESCALING_FACTOR (mV per count) and STACK: one shot's
##   millivolts are `sample · DESCALING_FACTOR / STACK`.
##
## The report adds the fields that tell whether a gain is inside the descale:
## SEG-Y bytes 119-124 (gain type, instrument gain constant and initial gain
## in dB), SEG-2 FIXED_GAIN, and the SEG-D format code.

import std/[math, options, strutils, tables]
import parseSEG
import core/units

type
  ScaleGroup* = object
    ## Traces whose scale headers hold the same values (`scaleReport`).
    first*: int       ## The first of them.
    count*: int       ## How many of the traces asked about.
    channelSet*: int  ## The 0-based SEG-D channel set; -1 for SEG-Y and SEG-2.
    fields*: seq[(string, float64)]
      ## The header values as read, named as parseSEG names them
      ## (`trace.transductionMantissa`, `ext1.physicalUnit`; channel set
      ## descriptor fields as `cs.<field>`, the Sensor Info block as
      ## `sensorInfo.<field>`); NaN where the trace has no such field.
    scale*: FileScale ## What `fileScalesInto` makes of them.

func physicalUnitOf(code: int): (AmplitudeUnit, float64) =
  # A SEG-D Rev 3 physical unit (CSD byte 62, THE#1 byte 31): the quantity and
  # SI units per unit.
  case code
  of 0x01: (auPressure, 100.0) # millibar
  of 0x02: (auPressure, 1e5) # bar
  of 0x03: (auVelocity, 1e-3) # mm/s
  of 0x04: (auVelocity, 1.0) # m/s
  of 0x05: (auAcceleration, 1e-3) # mm/s²
  of 0x06: (auAcceleration, 1.0) # m/s²
  of 0x12: (auPressure, 1.0) # Pa
  of 0x13: (auPressure, 1e-6) # µPa
  else: (auRecorded, NaN)

func segyQuantity(code: int): AmplitudeUnit =
  # A SEG-Y trace value or transduction unit code that is a physical quantity.
  case code
  of 1: auPressure
  of 6: auVelocity
  of 7: auAcceleration
  of 8: auForce # newton (SEG-Y Rev 2)
  else: auRecorded

func positive(x: float64): bool {.inline.} =
  x.classify notin {fcNan, fcInf, fcNegInf} and x > 0

# ------------------------------------------------------------------ SEG-D --

proc segdScale(f: SeismicFile, t: int, fields: var seq[(string, float64)],
    report: bool): (FileScale, int) =
  # The scale of SEG-D trace `t` and its channel set (-1 if unknown).
  var s = initFileScale()
  let e = f.traceEntry(t)
  let rec = f.index.segdRecords[e.record]
  let st = int(e.scanType)
  let cs = e.channelSetIndex
  if report:
    fields.add ("gh1.formatCode", float64(rec.gh1.formatCode))
  if st < 0 or st >= rec.scanTypes.len or cs < 0:
    return (s, -1)
  if report:
    let sensor = f.findTraceField(t, "ext1.sensorType")
    fields.add ("ext1.sensorType", (if sensor.isSome: sensor.get.toFloat else: NaN))
  case rec.scanTypes[st].layout
  of slRev2:
    if cs >= rec.scanTypes[st].descriptors.len: return (s, -1)
    let mp = rec.scanTypes[st].descriptors[cs].mpFactor
    if report: fields.add ("cs.mpFactor", mp)
    s.descale = pow(2.0, mp) * 1e-3
  of slRev3:
    if cs >= rec.scanTypes[st].descriptorsRev3.len: return (s, -1)
    let d = rec.scanTypes[st].descriptorsRev3[cs]
    let dsm = float64(d.descaleMultiplier)
    if report:
      fields.add ("cs.descaleMultiplier", dsm)
      fields.add ("cs.physicalUnit", float64(d.physicalUnit))
    if dsm.positive: s.descale = dsm * 1e-3
    var unit = NaN
    var sty = NaN
    if int(e.headerBytes) > SegdTraceHeaderSize + SegdBlockSize:
      # Extension blocks after THE#1: the Sensor Info block may be there.
      for row in f.traceHeaders(t):
        if row.blockName == "ext1" and row.field == "physicalUnit":
          unit = row.value.toFloat
        elif row.kind == hbSegdSensorInfo and row.field == "sensorSensitivity" and sty.isNaN:
          sty = row.value.toFloat
    else:
      let u = f.findTraceField(t, "ext1.physicalUnit")
      if u.isSome: unit = u.get.toFloat
    if report:
      fields.add ("ext1.physicalUnit", unit)
      fields.add ("sensorInfo.sensorSensitivity", sty)
    let code = (if unit.positive: int(unit) else: int(d.physicalUnit))
    let (q, si) = physicalUnitOf(code)
    s.unit = q # the quantity the headers name, with or without a sensitivity
    if sty.positive and q != auRecorded:
      s.sensitivity = sty * 1e-3 / si # mV per unit → V per SI unit
  (s, cs)

# ------------------------------------------------------------------ SEG-Y --

func segyScale(n, valueUnit, mantissa, exponent, units: float64): FileScale =
  result = initFileScale()
  if n.isNaN or valueUnit.isNaN: return
  let base = pow(2.0, -n)
  let vu = int(valueUnit)
  let q = segyQuantity(vu)
  if q != auRecorded:
    result.descale = base
    result.unit = q
    result.physical = true
    return
  let volts = case vu
    of 2: 1.0
    of 3: 1e-3
    of 0: (if n != 0: 1.0 else: NaN) # Rev 1: the LSB is 2^−N volts
    else: NaN
  if not volts.isNaN: result.descale = base * volts
  let c = mantissa * pow(10.0, exponent)
  if c.positive:
    let tu = int(units)
    let tq = segyQuantity(tu)
    if tq != auRecorded:
      if volts.isNaN:
        # physical = sample · 2^−N · C, with the volts unknown: already physical.
        (result.descale, result.unit, result.physical) = (base * c, tq, true)
      else:
        # V per unit = (volts of one value unit) / C.
        (result.sensitivity, result.unit) = (volts / c, tq)
    elif tu in [2, 3]:
      # A transduction to volts or millivolts: the descale itself.
      result.descale = base * c * (if tu == 2: 1.0 else: 1e-3)

const
  segyScaleFields = ["trace.traceWeightingFactor", "trace.traceValueUnit",
    "trace.transductionMantissa", "trace.transductionExponent", "trace.transductionUnits"]
  segyGainFields = ["trace.gainType", "trace.instrumentGainConstant",
    "trace.instrumentInitialGain"]
  seg2Fields = ["trace.DESCALING_FACTOR", "trace.STACK"]
  seg2GainFields = ["trace.FIXED_GAIN"]

proc columns(f: SeismicFile, names: openArray[string], traces: openArray[int]): seq[seq[float64]] =
  for name in names:
    var col = newSeq[float64](traces.len)
    f.traceFieldColumnInto(name, traces, col)
    result.add col

func seg2Scale(df, stack: float64): FileScale =
  result = initFileScale()
  if df.positive:
    result.descale = df / (if stack.positive: stack else: 1.0) * 1e-3

# ------------------------------------------------------------------ public --

proc fileScalesInto*(f: SeismicFile, traces: openArray[int], dst: var openArray[FileScale]) =
  ## What the headers of each trace of `traces` say about its scale, into
  ## `dst` (one per trace; see the module documentation for what is read).
  ## Call it once per chunk into a reused buffer and pass the buffer to the
  ## meta (`chunkMeta(..., scales = dst)`). Allocates a few columns of
  ## scratch (and, for SEG-D Rev 3 traces with extension blocks, a header
  ## listing per trace). Raises `ValueError` when `dst.len != traces.len`,
  ## for a trace outside the file or a closed file.
  if dst.len != traces.len:
    raise newException(ValueError, "need one FileScale per trace: " & $traces.len &
      " traces, " & $dst.len & " given")
  for t in traces:
    if t < 0 or t >= f.len:
      raise newException(ValueError, "trace " & $t & " is outside 0 ..< " & $f.len)
  case f.kind
  of skSegd:
    var none: seq[(string, float64)]
    for i, t in traces: dst[i] = f.segdScale(t, none, report = false)[0]
  of skSegy:
    let c = f.columns(segyScaleFields, traces)
    for i in 0 ..< traces.len: dst[i] = segyScale(c[0][i], c[1][i], c[2][i], c[3][i], c[4][i])
  of skSeg2:
    let c = f.columns(seg2Fields, traces)
    for i in 0 ..< traces.len: dst[i] = seg2Scale(c[0][i], c[1][i])
  of skUnknown:
    for i in 0 ..< traces.len: dst[i] = initFileScale()

proc fileScalesInto*(f: SeismicFile, traces: Slice[int], dst: var openArray[FileScale]) =
  ## `fileScalesInto` for a range of traces.
  if traces.a < 0 or traces.b >= f.len:
    if traces.b >= traces.a:
      raise newException(ValueError, "traces " & $traces & " are outside 0 ..< " & $f.len)
  var list = newSeq[int](max(0, traces.len))
  for i in 0 ..< list.len: list[i] = traces.a + i
  f.fileScalesInto(list, dst)

proc fileScalesInto*(f: SeismicFile, rows: TraceRows, dst: var openArray[FileScale]) =
  ## `fileScalesInto` for the traces of a view (a prefetch chunk's `rows`),
  ## one per row.
  var list = newSeq[int](rows.len)
  for r in 0 ..< rows.len: list[r] = rows.traceOf(r)
  f.fileScalesInto(list, dst)

proc scaleReport*(f: SeismicFile, traces: Slice[int]): seq[ScaleGroup] =
  ## What the headers of `traces` say about their scale, grouped: one
  ## `ScaleGroup` per distinct combination of channel set, header values and
  ## resulting `FileScale`, in order of first appearance, with the raw
  ## values (`fields`) to check against a recorder's documentation or a
  ## sensor's spec sheet before calibrating. Costs a header read per trace:
  ## ask about a record, not a whole survey. Raises `ValueError` for traces
  ## outside the file or a closed file.
  if traces.b >= traces.a and (traces.a < 0 or traces.b >= f.len):
    raise newException(ValueError, "traces " & $traces & " are outside 0 ..< " & $f.len)
  var list = newSeq[int](max(0, traces.len))
  for i in 0 ..< list.len: list[i] = traces.a + i
  var groups = initTable[string, int]()
  var cols: seq[seq[float64]]
  var names: seq[string]
  case f.kind
  of skSegy:
    names = @segyScaleFields & @segyGainFields
  of skSeg2:
    names = @seg2Fields & @seg2GainFields
  else: discard
  if names.len > 0: cols = f.columns(names, list)
  for i, t in list:
    var fields: seq[(string, float64)]
    var s = initFileScale()
    var cs = -1
    case f.kind
    of skSegd:
      (s, cs) = f.segdScale(t, fields, report = true)
    of skSegy:
      for k, n in names: fields.add (n, cols[k][i])
      s = segyScale(cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i])
    of skSeg2:
      for k, n in names: fields.add (n, cols[k][i])
      s = seg2Scale(cols[0][i], cols[1][i])
    of skUnknown: discard
    let key = $cs & "|" & $fields & "|" & $s & "|" & $s.physical
    let g = groups.getOrDefault(key, -1)
    if g >= 0:
      inc result[g].count
    else:
      groups[key] = result.len
      result.add ScaleGroup(first: t, count: 1, channelSet: cs, fields: fields, scale: s)

proc scaleReport*(f: SeismicFile): seq[ScaleGroup] =
  ## `scaleReport` of every trace of the file.
  f.scaleReport(0 ..< f.len)

func `$`*(g: ScaleGroup): string =
  ## One line: the traces, the channel set, the header values and the scale
  ## they give, e.g. `"3 traces from 1, channel set 1: cs.mpFactor = -10,
  ## ext1.sensorType = 2 → 9.76562e-07 V/count, no sensitivity"`.
  result = $g.count & (if g.count == 1: " trace" else: " traces") & " from " & $g.first
  if g.channelSet >= 0: result.add ", channel set " & $g.channelSet
  result.add ": "
  var parts: seq[string]
  for (name, v) in g.fields:
    var text = (if v.isNaN: "none" else: formatFloat(v, ffDefault, 7))
    if '.' in text and 'e' notin text:
      text.trimZeros()
      if text.endsWith('.'): text.setLen(text.len - 1)
    parts.add name & " = " & text
  result.add parts.join(", ") & " → " & $g.scale

proc segyUnitOverrides*(unit: AmplitudeUnit): seq[HeaderOverride] =
  ## The SEG-Y trace header values that state samples written in `unit`
  ## (D14: pass them to parseSEG's `writeTrace`): the trace value unit
  ## (bytes 203-204: Pa 1, V 2, m/s 6, m/s² 7, N 8; 0 for `auRecorded`), the trace
  ## weighting factor 0 and no transduction constant (205-212 zero). A file
  ## written so reads back through `fileScalesInto` as physical (or as
  ## volts), so a calibration does not scale it again.
  let code = case unit
    of auRecorded: 0
    of auVolt: 2
    of auVelocity: 6
    of auAcceleration: 7
    of auPressure: 1
    of auForce: 8
  @[headerOverride("trace.traceValueUnit", code),
    headerOverride("trace.traceWeightingFactor", 0),
    headerOverride("trace.transductionMantissa", 0),
    headerOverride("trace.transductionExponent", 0),
    headerOverride("trace.transductionUnits", 0)]
