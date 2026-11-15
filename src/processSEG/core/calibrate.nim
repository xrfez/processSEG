## Recorded values to volts or physical units, per row (decision D15).
##
## Many recorders write samples as quantization values (counts). Two factors
## make them physical: the **descale** (volts at the recorder's input per
## count) and the sensor's **sensitivity** (volts per m/s, per m/s² or per
## Pa), and the second differs per sensor: a hydrophone beside three
## geophones, vertical and horizontal geophones, one unit and the next. Each
## row of a block gets
##
##   factor = descale / (10^(gain/20) · sensitivity)
##
## (`ctPhysical`, in m/s, m/s² or Pa; `ctVolts` leaves out the sensitivity),
## where `gain` is a recorder gain in dB that the descale does not include.
##
## ```nim
## let cal = initCalibration([
##   CalibrationRule(match: RuleMatch(sensorType: 1),          # hydrophones
##                   sensitivity: sensitivity(20, suVPerBar)),
##   CalibrationRule(match: RuleMatch(sensorType: 2),          # vertical geophones
##                   sensitivity: sensitivity(22.5, suVPerMPerS)),
##   CalibrationRule(descale: descale(0.5, duMicroVPerCount))]) # every row
## let plan = initCalibrationPlan(cal)          # validates; ValueError
## var rows = newSeq[RowCalibration](meta.len)
## let summary = plan.resolveInto(meta, rows)   # per row: factor, unit, where from
## plan.calibrate(samples, meta)                # in place
## ```
##
## **Where the factors come from.** A `Calibration` holds the caller's rules;
## each `CalibrationRule` matches rows by any of SEG-D sensor type,
## component, channel set, channel and receiver station (one recording unit),
## and gives any of a descale, a sensitivity and a gain. The three factors
## resolve **independently**, each from the most specific matching rule that
## gives it. Specificity ranks the criteria station > channel > channel set >
## component > sensor type, compared on the set of criteria a rule has (as
## binary digits: a station rule beats every rule without a station; a
## channel-set and component rule beats a channel-set rule). Two rules with
## the same criteria and values are refused. A rule without criteria is the
## **catch-all**, the default.
##
## The file's own headers are the second source: a per-row `FileScale`
## column the meta points at (`fileScalesInto` fills it from a file,
## `initChunkMeta(..., scales = column)`). `fileFactors` orders the two:
##
## - `ffFallback` (default): the specific rules, then the file, then the
##   catch-all. What the caller entered for a row wins over the file, and
##   the file wins over a blanket default.
## - `ffFirst`: the file, then the specific rules, then the catch-all.
## - `ffIgnore`: the rules only.
##
## Neither overrides the other silently: a row where the file and a rule both
## give a factor, more than 0.1 % apart, is a **conflict** (`RowCalibration.
## conflict`, `CalibrationSummary.conflicts`), whichever won. A sensitivity
## whose quantity does not fit the row's SEG-D sensor type (a V/(m/s) value on
## a hydrophone, sensor type 1) is a **mismatch**, reported likewise. A row
## whose headers say it is already physical (`FileScale.physical`) is taken
## as the file says, unless `ffIgnore`; a rule that gives it a factor is a
## conflict.
##
## **Rows.**
## - A **live** row that cannot reach the target (no descale; for
##   `ctPhysical` no sensitivity) is filled with **NaN** over its live length
##   (`csUnresolved`). A row of counts beside rows in m/s would be 10⁶ to 10⁹
##   times too large for everything after it (a panel normalization would be
##   set by it), while NaN is what panel statistics, spectra and stacks
##   already leave out, and it stays in its row. The raw data is untouched in
##   the `calibrateInto` and pipeline `src → dst` forms.
## - A **dead** row is converted when it can be, else kept as recorded.
## - An **auxiliary** row (time break, pilot, uphole) is converted to volts
##   when its descale is known and `scaleAuxiliary` (the default), and never
##   given a sensitivity; otherwise kept.
## - A **failed** row is never touched.
##
## Rows are processed up to their live length (`ChunkMeta.liveSamples`), as
## everywhere.
##
## **Order.** Run this before anything that changes amplitudes (gain, AGC,
## normalization, clip, scale): those, statistics and QC then see physical
## values. The pipeline's `calibrateOp` enforces it.
##
## **The flat band only.** One sensitivity holds above the sensor's natural
## frequency; below it a geophone's response falls off and depends on its
## damping. No response correction is made here.
##
## **Cost and threads.** A plan is immutable: any number of threads may use
## one. Resolving a row is a few hash lookups (one per criteria set used);
## nothing allocates once the plan is built.

import std/[hashes, math, options, strutils, tables]
import parseSEG
import basic, rows, types, units

type
  CalibrationTarget* = enum
    ## What a calibration converts to.
    ctPhysical = "physical" ## m/s, m/s² or Pa: needs a descale and a sensitivity per row.
    ctVolts = "volts"       ## V at the recorder's input: needs a descale per row.

  FileFactors* = enum
    ## How the file's own headers rank against the caller's rules.
    ffFallback = "fallback" ## The specific rules, then the file, then the catch-all rule.
    ffFirst = "first"       ## The file, then the specific rules, then the catch-all rule.
    ffIgnore = "ignore"     ## The rules only.

  RuleMatch* = object
    ## The rows a rule applies to; unset criteria match any row. Build it
    ## with the object constructor (`RuleMatch(sensorType: 1)`), whose
    ## defaults leave every criterion unset.
    sensorType*: int = -1
      ## `TraceKeys.sensorType` (SEG-D THE#1 byte 21: 1 hydrophone, 2 vertical
      ## geophone, 3-5 horizontal geophones, 6-9 accelerometers); -1: any.
    component*: int = -1
      ## `TraceKeys.component` (SEG-D Rev 3 group index or sensor type, SEG-Y
      ## trace identification code 11-17, SEG-2 12-14); -1: any.
    channelSet*: int = -1 ## `TraceRows.channelSetIndex`: the 0-based SEG-D channel set; -1: any.
    channel*: float64 = NaN
      ## `TraceKeys.channel`: the channel (trace number) in the record, so in a
      ## node's record the channel of the unit; NaN: any.
    station*: Option[StationKey]
      ## The receiver station (`stationKey(line, point, index)`, index 0: any),
      ## which identifies one recording unit or sensor position; none: any.

  CalibrationRule* = object
    ## Factors for the rows `match` selects. Each factor is optional: an
    ## unset `descale` or `sensitivity` (the default) and a NaN `gainDb` give
    ## none, and that factor comes from elsewhere.
    match*: RuleMatch
    descale*: Descale         ## Volts per count (`descale(value, unit)`).
    sensitivity*: Sensitivity ## As wired (`sensitivity`, `openCircuitSensitivity`).
    gainDb*: float64 = NaN
      ## A recorder gain the descale does not include, in dB: the samples are
      ## divided by `10^(gainDb/20)`. SEG-D's MP factor and descale
      ## multiplier include the system gain by definition; if a file's
      ## descale does not change between records at two gain settings, it
      ## does not include it, and the gain goes here.

  Calibration* = object
    ## Plain data: the rules and how to apply them. Edit it, then build an
    ## `initCalibrationPlan` (or let the pipeline's `compile` do it).
    rules*: seq[CalibrationRule]
    target*: CalibrationTarget = ctPhysical
    fileFactors*: FileFactors = ffFallback
    scaleAuxiliary*: bool = true ## Convert auxiliary rows to volts when their descale is known.

  FactorSource* = enum
    ## Where a row's factor came from.
    fsNone = "none" ## Nowhere: the factor is unknown.
    fsRule = "rule" ## A rule (`RowCalibration.descaleRule`, ...: its index).
    fsFile = "file" ## The file's headers (`FileScale`).

  CalibrationStatus* = enum
    ## What happened to a row.
    csConverted = "converted"   ## Scaled into `RowCalibration.unit`.
    csUnresolved = "unresolved" ## A live row without the factors the target needs: NaN.
    csKept = "kept"             ## Left as recorded: failed, or dead or auxiliary without its factors.

  RowCalibration* = object
    ## How one row is calibrated (`resolveInto`).
    factor*: float64
      ## The multiplier `calibrate` applies: `unit` per recorded unit; 1 for a
      ## kept row and NaN for an unresolved one, so `scaleRows` with these
      ## factors is `calibrate`.
    unit*: AmplitudeUnit ## The row's unit after it (`auRecorded` when not converted).
    status*: CalibrationStatus
    descaleFrom*, sensitivityFrom*: FactorSource
    descaleRule*, sensitivityRule*, gainRule*: int
      ## The index in `Calibration.rules` of the rule that gave each factor;
      ## -1 when none did.
    conflict*: bool ## The file and a rule both gave a factor, more than 0.1 % apart.
    mismatch*: bool ## The sensitivity's quantity does not fit the SEG-D sensor type.

  CalibrationSummary* = object
    ## Counts over the rows of one `resolveInto` (or `calibrate`) call.
    units*: array[AmplitudeUnit, int] ## Rows by unit after calibration (kept rows: `auRecorded`).
    converted*, unresolved*, kept*: int ## Rows by status.
    noDescale*, noSensitivity*: int
      ## Live rows unresolved for want of a descale, or (`ctPhysical`) of a
      ## sensitivity.
    conflicts*, mismatches*: int ## Rows flagged so.

  RatioRange* = object
    ## The range of `ratioDbInto`'s per-row ratios.
    rows*: int            ## Rows converted, to the same unit, by both plans.
    minDb*, maxDb*: float64 ## NaN when `rows` is 0.

  MatchKey = object
    # A rule's criteria, or a row's values under one criteria set (`mask`).
    mask: uint8
    sensorType, component, channelSet: int
    channel, line, point: float64
    index: int16

  CalibrationPlan* = object
    ## A validated `Calibration` with its rules indexed, ready to apply.
    ## Build it with `initCalibrationPlan`; immutable and shareable.
    target: CalibrationTarget
    fileFactors: FileFactors
    scaleAux: bool
    classes: seq[uint8] # criteria sets of the specific rules, most specific first
    catchAll: int       # the rule without criteria, -1 when there is none
    index: Table[MatchKey, int]
    descale, sens, gain: seq[float64] # per rule; NaN when it gives none (gain: linear)
    sensUnit: seq[AmplitudeUnit]

const
  bSensorType = 1'u8
  bComponent = 2'u8
  bChannelSet = 4'u8
  bChannel = 8'u8
  bStation = 16'u8
  conflictTolerance = 1e-3

func hundredths(x: float64): float64 {.inline.} = round(x * 100) / 100

func hash(k: MatchKey): Hash =
  var h: Hash = hash(k.mask)
  if (k.mask and bSensorType) != 0: h = h !& hash(k.sensorType)
  if (k.mask and bComponent) != 0: h = h !& hash(k.component)
  if (k.mask and bChannelSet) != 0: h = h !& hash(k.channelSet)
  if (k.mask and bChannel) != 0: h = h !& hash(k.channel + 0.0) # -0.0 hashes as 0.0
  if (k.mask and bStation) != 0:
    h = h !& hash(k.line + 0.0)
    h = h !& hash(k.point + 0.0)
    h = h !& hash(k.index)
  !$h

func `==`(a, b: MatchKey): bool =
  if a.mask != b.mask: return false
  let m = a.mask
  ((m and bSensorType) == 0 or a.sensorType == b.sensorType) and
    ((m and bComponent) == 0 or a.component == b.component) and
    ((m and bChannelSet) == 0 or a.channelSet == b.channelSet) and
    ((m and bChannel) == 0 or a.channel == b.channel) and
    ((m and bStation) == 0 or (a.line == b.line and a.point == b.point and
      a.index == b.index))

func maskOf(m: RuleMatch): uint8 =
  if m.sensorType >= 0: result = result or bSensorType
  if m.component >= 0: result = result or bComponent
  if m.channelSet >= 0: result = result or bChannelSet
  if not m.channel.isNaN: result = result or bChannel
  if m.station.isSome: result = result or bStation

func keyOf(m: RuleMatch): MatchKey =
  result = MatchKey(mask: maskOf(m), sensorType: m.sensorType, component: m.component,
    channelSet: m.channelSet, channel: hundredths(m.channel))
  if m.station.isSome:
    let s = m.station.get
    (result.line, result.point, result.index) = (hundredths(s.line), hundredths(s.point),
      s.index)

func finite(x: float64): bool {.inline.} = x.classify notin {fcNan, fcInf, fcNegInf}

func `$`*(m: RuleMatch): string =
  ## The criteria in words: `"sensor type 1, channel set 2"`, `"any row"`.
  var parts: seq[string]
  if m.station.isSome:
    let s = m.station.get
    parts.add "station " & $s.line & "/" & $s.point & (if s.index != 0: "." & $s.index else: "")
  if not m.channel.isNaN: parts.add "channel " & $m.channel
  if m.channelSet >= 0: parts.add "channel set " & $m.channelSet
  if m.component >= 0: parts.add "component " & $m.component
  if m.sensorType >= 0: parts.add "sensor type " & $m.sensorType
  if parts.len == 0: "any row" else: parts.join(", ")

func `$`*(r: CalibrationRule): string =
  ## The rule in words: `"sensor type 2: sensitivity 22.5 V/(m/s)"`.
  var parts: seq[string]
  if r.descale.isSet: parts.add "descale " & $r.descale
  if r.sensitivity.isSet: parts.add "sensitivity " & $r.sensitivity
  if not r.gainDb.isNaN: parts.add "gain " & $r.gainDb & " dB"
  $r.match & ": " & (if parts.len == 0: "no factor" else: parts.join(", "))

func `$`*(c: Calibration): string =
  ## The rules, one per line with their index, then the target and order.
  for i, r in c.rules: result.add "rule " & $i & ", " & $r & "\n"
  result.add "target " & $c.target & ", file factors " & $c.fileFactors &
    (if c.scaleAuxiliary: ", auxiliary rows to volts" else: ", auxiliary rows kept")

func `$`*(s: CalibrationSummary): string =
  ## The counts in words: `"2000 converted (1998 m/s, 2 Pa), 0 unresolved, 0 kept"`.
  result = $s.converted & " converted"
  var parts: seq[string]
  for u in AmplitudeUnit:
    if u != auRecorded and s.units[u] > 0: parts.add $s.units[u] & " " & $u
  if parts.len > 0: result.add " (" & parts.join(", ") & ")"
  result.add ", " & $s.unresolved & " unresolved"
  if s.unresolved > 0:
    result.add " (" & $s.noDescale & " without a descale, " & $s.noSensitivity &
      " without a sensitivity)"
  result.add ", " & $s.kept & " kept"
  if s.conflicts > 0: result.add ", " & $s.conflicts & " conflicts"
  if s.mismatches > 0: result.add ", " & $s.mismatches & " sensor-type mismatches"

func initCalibration*(rules: openArray[CalibrationRule] = [], target = ctPhysical,
    fileFactors = ffFallback, scaleAuxiliary = true): Calibration =
  ## A calibration with `rules`; see the module documentation.
  Calibration(rules: @rules, target: target, fileFactors: fileFactors,
    scaleAuxiliary: scaleAuxiliary)

proc badRule(i: int, msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, "rule " & $i & ": " & msg)

proc initCalibrationPlan*(cal: Calibration): CalibrationPlan =
  ## Validates `cal` and indexes its rules. Raises `ValueError`, naming the
  ## rule, for a rule that gives no factor, a gain that is not finite, a
  ## criterion out of range (sensor type or component above 255, a channel
  ## set below -1, a channel or station that is not finite), or two rules
  ## with the same criteria and values.
  result = CalibrationPlan(target: cal.target, fileFactors: cal.fileFactors,
    scaleAux: cal.scaleAuxiliary, catchAll: -1)
  let n = cal.rules.len
  result.descale = newSeq[float64](n)
  result.sens = newSeq[float64](n)
  result.gain = newSeq[float64](n)
  result.sensUnit = newSeq[AmplitudeUnit](n)
  var masks: set[uint8]
  for i, r in cal.rules:
    let m = r.match
    if not (r.descale.isSet or r.sensitivity.isSet or not r.gainDb.isNaN):
      badRule(i, "gives no factor (set a descale, a sensitivity or gainDb)")
    if not r.gainDb.isNaN and not r.gainDb.finite:
      badRule(i, "gainDb must be finite, got " & $r.gainDb)
    if m.sensorType > 255 or m.sensorType < -1:
      badRule(i, "sensorType must be -1 (any) or 0 .. 255, got " & $m.sensorType)
    if m.component > 255 or m.component < -1:
      badRule(i, "component must be -1 (any) or 0 .. 255, got " & $m.component)
    if m.channelSet < -1:
      badRule(i, "channelSet must be -1 (any) or ≥ 0, got " & $m.channelSet)
    if not m.channel.isNaN and not m.channel.finite:
      badRule(i, "channel must be finite (NaN: any), got " & $m.channel)
    if m.station.isSome and not (m.station.get.line.finite and m.station.get.point.finite):
      badRule(i, "the station's line and point must be finite")
    let key = keyOf(m)
    if key.mask == 0:
      if result.catchAll >= 0:
        badRule(i, "has no criteria, like rule " & $result.catchAll & ": one catch-all only")
      result.catchAll = i
    else:
      let j = result.index.getOrDefault(key, -1)
      if j >= 0: badRule(i, "matches the same rows as rule " & $j & " (" & $m & ")")
      result.index[key] = i
      masks.incl key.mask
    result.descale[i] = r.descale.voltsPerCount
    result.sens[i] = r.sensitivity.voltsPerUnit
    result.sensUnit[i] = r.sensitivity.unit
    result.gain[i] = (if r.gainDb.isNaN: NaN else: pow(10.0, r.gainDb / 20))
  for mask in countdown(31'u8, 1'u8):
    if mask in masks: result.classes.add mask

func target*(p: CalibrationPlan): CalibrationTarget {.inline, raises: [].} =
  ## The plan's target.
  p.target

# ---------------------------------------------------------------- resolve --

type
  Found = object
    d, s, g: int # rule indices, -1: none

func take(p: CalibrationPlan, r: int, f: var Found) {.inline, raises: [].} =
  if f.d < 0 and not p.descale[r].isNaN: f.d = r
  if f.s < 0 and not p.sens[r].isNaN: f.s = r
  if f.g < 0 and not p.gain[r].isNaN: f.g = r

func specificRules(p: CalibrationPlan, meta: ChunkMeta, row: int): Found {.raises: [].} =
  # The most specific rule giving each factor, without the catch-all.
  result = Found(d: -1, s: -1, g: -1)
  if p.classes.len == 0: return
  let k = meta.keys(row)
  let cs = meta.rows.channelSetIndex(row)
  for mask in p.classes:
    var key = MatchKey(mask: mask, sensorType: int(k.sensorType), component: int(k.component),
      channelSet: cs)
    if (mask and bChannelSet) != 0 and cs < 0: continue
    if (mask and bChannel) != 0:
      if k.channel.isNaN: continue
      key.channel = hundredths(k.channel)
    if (mask and bStation) != 0:
      if k.rcvLine.isNaN or k.rcvPoint.isNaN: continue
      (key.line, key.point) = (hundredths(k.rcvLine), hundredths(k.rcvPoint))
      # The exact index, then a rule for any index (0); a trace without an
      # index (0) also matches index 1 (survey files number the first
      # position 1), as parseSEG's station join does.
      let tries = (if k.rcvIndex != 0: [k.rcvIndex, 0'i16] else: [0'i16, 1'i16])
      for t in tries:
        key.index = t
        let r = p.index.getOrDefault(key, -1)
        if r >= 0: p.take(r, result)
    else:
      let r = p.index.getOrDefault(key, -1)
      if r >= 0: p.take(r, result)
    if result.d >= 0 and result.s >= 0 and result.g >= 0: return

func expectedUnit(sensorType: uint8): AmplitudeUnit {.inline.} =
  case sensorType
  of 1: auPressure
  of 2 .. 5: auVelocity
  of 6 .. 9: auAcceleration
  else: auRecorded

func differs(a, b: float64): bool {.inline.} =
  abs(a - b) > conflictTolerance * max(abs(a), abs(b))

func resolveRow(p: CalibrationPlan, meta: ChunkMeta, row: int): RowCalibration {.raises: [].} =
  result = RowCalibration(factor: 1, unit: auRecorded, status: csKept, descaleRule: -1,
    sensitivityRule: -1, gainRule: -1)
  if meta.isFailed(row): return
  let k = meta.keys(row)
  let aux = k.status == tsAuxiliary
  let live = k.status in {tsLive, tsUnknown}
  if aux and not p.scaleAux: return
  let useFile = p.fileFactors != ffIgnore and meta.hasFileScales
  let fs = (if useFile: meta.fileScale(row) else: initFileScale())
  let spec = p.specificRules(meta, row)
  let c = p.catchAll
  # The rule that would give each factor (specific, else the catch-all).
  template ruleFor(field: untyped, idx: int): int =
    (if idx >= 0: idx elif c >= 0 and not p.field[c].isNaN: c else: -1)
  let rd = ruleFor(descale, spec.d)
  let rs = ruleFor(sens, spec.s)
  let rg = ruleFor(gain, spec.g)
  var unit = auRecorded
  var factor = NaN
  if useFile and fs.physical:
    # The headers say the samples are already physical.
    result.conflict = rd >= 0 or rs >= 0 or rg >= 0
    if p.target == ctPhysical and not aux and fs.descale.finite and fs.descale > 0:
      (factor, unit) = (fs.descale, fs.unit)
      (result.descaleFrom, result.sensitivityFrom) = (fsFile, fsFile)
  else:
    # Each factor: (value, source, rule) in the order `fileFactors` sets.
    var d, s: float64 = NaN
    var sUnit = auRecorded
    let fileD = fs.descale.finite and fs.descale > 0
    let fileS = fs.sensitivity.finite and fs.sensitivity > 0 and
      fs.unit in {auVelocity, auAcceleration, auPressure}
    template pick(fileHas: bool, fileV: float64, specIdx, ruleIdx: int, field: untyped,
        v: var float64, src: var FactorSource, ri: var int) =
      if p.fileFactors == ffFirst and fileHas: (v, src) = (fileV, fsFile)
      elif specIdx >= 0: (v, src, ri) = (p.field[specIdx], fsRule, specIdx)
      elif fileHas: (v, src) = (fileV, fsFile)
      elif ruleIdx >= 0: (v, src, ri) = (p.field[ruleIdx], fsRule, ruleIdx)
    pick(fileD, fs.descale, spec.d, rd, descale, d, result.descaleFrom, result.descaleRule)
    let wantS = not aux and p.target == ctPhysical
    if wantS:
      pick(fileS, fs.sensitivity, spec.s, rs, sens, s, result.sensitivityFrom,
        result.sensitivityRule)
      sUnit = (if result.sensitivityFrom == fsFile: fs.unit
               elif result.sensitivityRule >= 0: p.sensUnit[result.sensitivityRule]
               else: auRecorded)
    # A conflict: the file and a rule both give a factor, and they differ.
    if fileD and rd >= 0 and differs(fs.descale, p.descale[rd]): result.conflict = true
    if wantS and fileS and rs >= 0 and
        (fs.unit != p.sensUnit[rs] or differs(fs.sensitivity, p.sens[rs])):
      result.conflict = true
    let g = (if rg >= 0: p.gain[rg] else: 1.0)
    result.gainRule = rg
    if wantS and not s.isNaN:
      let want = expectedUnit(k.sensorType)
      result.mismatch = want != auRecorded and want != sUnit
    if not d.isNaN:
      if aux or p.target == ctVolts:
        (factor, unit) = (d / g, auVolt)
      elif not s.isNaN:
        (factor, unit) = (d / (g * s), sUnit)
  if not factor.isNaN:
    (result.factor, result.unit, result.status) = (factor, unit, csConverted)
  elif live:
    (result.factor, result.status) = (NaN, csUnresolved)

func count(s: var CalibrationSummary, r: RowCalibration, p: CalibrationPlan) {.inline.} =
  case r.status
  of csConverted:
    inc s.converted
    inc s.units[r.unit]
  of csUnresolved:
    inc s.unresolved
    if r.descaleFrom == fsNone: inc s.noDescale
    elif p.target == ctPhysical and r.sensitivityFrom == fsNone: inc s.noSensitivity
  of csKept:
    inc s.kept
    inc s.units[auRecorded]
  if r.conflict: inc s.conflicts
  if r.mismatch: inc s.mismatches

proc resolveInto*(p: CalibrationPlan, meta: ChunkMeta,
    rows: var openArray[RowCalibration]): CalibrationSummary =
  ## How each row of `meta` is calibrated, into `rows` (one per row), and
  ## the counts over them, without touching a sample: a viewer labels its
  ## amplitude readout from `rows[r].unit`, shows the summary's unresolved
  ## rows and conflicts, or applies `rows[r].factor` itself (a GPU uniform,
  ## `scaleRows`). Raises `ValueError` when `rows.len != meta.len`.
  ## Allocates nothing.
  if rows.len != meta.len:
    raise newException(ValueError, "need one RowCalibration per row: " & $meta.len &
      " rows, " & $rows.len & " given")
  for r in 0 ..< meta.len:
    rows[r] = p.resolveRow(meta, r)
    result.count(rows[r], p)

# ---------------------------------------------------------------- kernels --

proc scaleInto*[T: SampleType](src: openArray[T], dst: var openArray[T],
    factor: float64) {.raises: [].} =
  ## `dst[i] = src[i] · factor` for every `i` of `src` (`dst` at least as
  ## long; an `AssertionDefect` otherwise). A NaN factor fills with NaN.
  doAssert dst.len >= src.len, "dst is shorter than src"
  let k = T(factor)
  {.push boundChecks: off.}
  for i in 0 ..< src.len:
    dst[i] = src[i] * k
  {.pop.}

proc badFactors(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc checkFactors(factors: openArray[float64], n: int) =
  if factors.len != n:
    badFactors("need one factor per row: " & $n & " rows, " & $factors.len & " factors")
  for i, f in factors:
    if f == Inf or f == -Inf: badFactors("factor " & $i & " is infinite")

proc checkPair[T](src, dst: RowBlock[T]) =
  if src.len != dst.len or src.cols != dst.cols:
    badFactors("src is " & $src.len & " × " & $src.cols & ", dst " & $dst.len & " × " &
      $dst.cols)
  let bytes = dst.len * dst.cols * sizeof(T)
  if bytes > 0:
    let s0 = cast[uint](src.rowPtr(0))
    let d0 = cast[uint](dst.rowPtr(0))
    if s0 != d0 and s0 < d0 + uint(bytes) and d0 < s0 + uint(bytes):
      badFactors("src and dst overlap without being the same block")

proc scaleRow[T](src, dst: RowBlock[T], meta: ChunkMeta, r: int, k: float64,
    keep: bool) {.inline, raises: [].} =
  # Row `r` of `src` times `k` into `dst` over its live length, the rest
  # copied; `keep`: copied whole.
  let cols = dst.cols
  let n = (if keep: 0 else: meta.liveSamples(r, cols))
  let same = src.rowPtr(r) == dst.rowPtr(r)
  if n > 0:
    if same: scale(dst.row(r, n), k)
    else: scaleInto(src.row(r, n), dst.row(r, n), k)
  if not same and n < cols:
    copyMem(addr dst.rowPtr(r)[n], addr src.rowPtr(r)[n], (cols - n) * sizeof(T))

proc scaleRowsImpl[T](src, dst: RowBlock[T], meta: ChunkMeta,
    factors: openArray[float64]) =
  meta.checkShape(dst)
  checkPair(src, dst)
  checkFactors(factors, meta.len)
  for r in 0 ..< dst.len:
    scaleRow(src, dst, meta, r, factors[r], meta.isFailed(r))

proc scaleRows*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    factors: openArray[float64]) =
  ## Multiplies every row of `b` that is not a failed row by `factors[r]`,
  ## up to its live length (whatever its status: the caller chose the
  ## factor; pass 1 to keep a row). A NaN factor fills the row with NaN.
  ## Raises `ValueError` when `b` does not match `meta` or `factors` has not
  ## one factor per row or holds an infinite one.
  scaleRowsImpl(b, b, meta, factors)

proc scaleRows*[T: SampleType](samples: var Tensor[T], meta: ChunkMeta,
    factors: openArray[float64]) =
  ## `scaleRows` on a `[rows, cols]` tensor (C-contiguous), in place.
  meta.checkShape(samples)
  scaleRows(rowBlock(samples), meta, factors)

proc scaleRowsInto*[T: SampleType](src, dst: RowBlock[T], meta: ChunkMeta,
    factors: openArray[float64]) =
  ## `scaleRows` from `src` into `dst` (the same shape; they must not
  ## overlap unless they are the same block), leaving `src` as it is: rows
  ## and padding not scaled are copied. Raises as `scaleRows`, and for a
  ## shape mismatch or a partial overlap.
  scaleRowsImpl(src, dst, meta, factors)

proc scaleRowsInto*[T: SampleType](src: Tensor[T], dst: var Tensor[T], meta: ChunkMeta,
    factors: openArray[float64]) =
  ## `scaleRowsInto` on tensors (both C-contiguous, of one shape).
  meta.checkShape(src)
  meta.checkShape(dst)
  scaleRowsInto(rowBlock(src), rowBlock(dst), meta, factors)

# ----------------------------------------------------------- calibrate --

proc calibrateImpl[T](p: CalibrationPlan, src, dst: RowBlock[T], meta: ChunkMeta,
    inverse: bool): CalibrationSummary =
  meta.checkShape(dst)
  checkPair(src, dst)
  for r in 0 ..< dst.len:
    let c = p.resolveRow(meta, r)
    result.count(c, p)
    let k = (if c.status == csConverted and inverse: 1 / c.factor else: c.factor)
    scaleRow(src, dst, meta, r, k, c.status == csKept)

proc calibrate*[T: SampleType](p: CalibrationPlan, b: RowBlock[T],
    meta: ChunkMeta): CalibrationSummary {.discardable.} =
  ## Converts each row of `b` in place, as `resolveInto` describes it:
  ## converted rows times their factor, unresolved (live) rows NaN, kept rows
  ## untouched, each up to its live length. Returns the counts. Raises
  ## `ValueError` when `b` does not match `meta`. Allocates nothing.
  calibrateImpl(p, b, b, meta, inverse = false)

proc calibrate*[T: SampleType](p: CalibrationPlan, samples: var Tensor[T],
    meta: ChunkMeta): CalibrationSummary {.discardable.} =
  ## `calibrate` on a `[rows, cols]` tensor (C-contiguous), in place.
  meta.checkShape(samples)
  calibrate(p, rowBlock(samples), meta)

proc calibrateInto*[T: SampleType](p: CalibrationPlan, src, dst: RowBlock[T],
    meta: ChunkMeta): CalibrationSummary {.discardable.} =
  ## `calibrate` from `src` into `dst` (the same shape, not overlapping
  ## unless the same block), leaving the raw rows as they are; kept rows
  ## and padding are copied. Raises `ValueError` for a shape mismatch or a
  ## partial overlap.
  calibrateImpl(p, src, dst, meta, inverse = false)

proc calibrateInto*[T: SampleType](p: CalibrationPlan, src: Tensor[T], dst: var Tensor[T],
    meta: ChunkMeta): CalibrationSummary {.discardable.} =
  ## `calibrateInto` on tensors (both C-contiguous, of one shape).
  meta.checkShape(src)
  meta.checkShape(dst)
  calibrateInto(p, rowBlock(src), rowBlock(dst), meta)

proc uncalibrate*[T: SampleType](p: CalibrationPlan, b: RowBlock[T],
    meta: ChunkMeta): CalibrationSummary {.discardable.} =
  ## The inverse of `calibrate`: rows in the plan's units back to recorded
  ## values (counts), dividing by the factor `resolveInto` gives (to write
  ## an integer format with the input's descale). Unresolved live rows
  ## become NaN, kept rows are untouched. Raises as `calibrate`.
  calibrateImpl(p, b, b, meta, inverse = true)

proc uncalibrate*[T: SampleType](p: CalibrationPlan, samples: var Tensor[T],
    meta: ChunkMeta): CalibrationSummary {.discardable.} =
  ## `uncalibrate` on a tensor, in place.
  meta.checkShape(samples)
  uncalibrate(p, rowBlock(samples), meta)

proc uncalibrateInto*[T: SampleType](p: CalibrationPlan, src, dst: RowBlock[T],
    meta: ChunkMeta): CalibrationSummary {.discardable.} =
  ## `uncalibrate` from `src` into `dst`; see `calibrateInto`.
  calibrateImpl(p, src, dst, meta, inverse = true)

proc uncalibrateInto*[T: SampleType](p: CalibrationPlan, src: Tensor[T],
    dst: var Tensor[T], meta: ChunkMeta): CalibrationSummary {.discardable.} =
  ## `uncalibrateInto` on tensors.
  meta.checkShape(src)
  meta.checkShape(dst)
  uncalibrateInto(p, rowBlock(src), rowBlock(dst), meta)

proc ratioDbInto*(a, b: CalibrationPlan, meta: ChunkMeta,
    dst: var openArray[float64]): RatioRange =
  ## Per row, how much larger data comes out under plan `a` than under plan
  ## `b`, in dB (`20·log10(factorA / factorB)`), into `dst` (one per row):
  ## to compare two candidate tables, such as a spec sheet's damped and
  ## open-circuit readings, before choosing one. NaN for a row that either
  ## does not convert, or that they convert to different units. Returns how
  ## many rows compared and their range. Raises `ValueError` when
  ## `dst.len != meta.len`. Allocates nothing.
  if dst.len != meta.len:
    raise newException(ValueError, "need one value per row: " & $meta.len & " rows, " &
      $dst.len & " given")
  result = RatioRange(minDb: NaN, maxDb: NaN)
  for r in 0 ..< meta.len:
    let x = a.resolveRow(meta, r)
    let y = b.resolveRow(meta, r)
    if x.status == csConverted and y.status == csConverted and x.unit == y.unit:
      let db = 20 * log10(abs(x.factor / y.factor))
      dst[r] = db
      if result.rows == 0: (result.minDb, result.maxDb) = (db, db)
      else: (result.minDb, result.maxDb) = (min(result.minDb, db), max(result.maxDb, db))
      inc result.rows
    else:
      dst[r] = NaN
