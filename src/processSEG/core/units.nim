## Amplitude units: what a sample's value means after calibration
## (`AmplitudeUnit`), and the two factors that get it there from a recorded
## value, entered with their units so that a bare number cannot be read in
## the wrong one (decision D15):
##
## - the **descale** (`Descale`): volts at the recorder's input per recorded
##   unit (count), as a recorder's documentation or a SEG-D channel set gives
##   it (`descale(0.5, duMicroVPerCount)`);
## - the **sensitivity** (`Sensitivity`): volts out of the sensor per unit of
##   ground motion, pressure or force (`sensitivity(22.5, suVPerMPerS)`).
##
## A recorded value `x` is then `x · descale` volts, and `x · descale /
## sensitivity` metres per second (geophone), metres per second squared
## (accelerometer), pascals (hydrophone) or newtons (a vibrator's ground-force
## channel). Units are SI throughout; a viewer prints prefixes (µm/s, mPa, kN)
## itself.
##
## **Mass** (`Mass`) is entered with its unit too (`mass(2450, muKilogram)`):
## a vibrator's ground force is the weighted sum of its reaction-mass and
## baseplate accelerations (`vibroseis/vibqc`), in newtons for accelerations
## in m/s².
##
## ```nim
## let geophone = sensitivity(22.5, suVPerMPerS)        # as wired: the damped figure
## let hydrophone = sensitivity(-194, suDbReVPerMicroPa) # ≈ 20 V/bar
## let lsb = descale(0.5, duMicroVPerCount)
## doAssert geophone.unit == auVelocity and hydrophone.unit == auPressure
## echo 1000 * lsb.voltsPerCount / geophone.voltsPerUnit, " m/s" # 1000 counts
## ```
##
## **Which sensitivity.** Enter the sensitivity that holds as the sensor is
## wired to the recorder, with `sensitivity`. For a geophone that is the
## *damped* sensitivity: the coil works into a shunt (damping) resistor, and
## into the recorder's input impedance, which divide its output. A spec sheet
## that gives the open-circuit (intrinsic) sensitivity with the coil and
## shunt resistances is entered with `openCircuitSensitivity`, which computes
## the damped value:
##
##   S = n·S₀ · R_L / (n·R_c / m + R_L),   R_L = R_s ∥ R_in
##
## for `n` sensors in series per string and `m` strings in parallel, with
## S₀ one sensor's open-circuit sensitivity, R_c its coil resistance, R_s the
## shunt across the string(s) and R_in the recorder's input impedance (large
## and often left out). It is the voltage divider of the coil's EMF (n·S₀ per
## unit of velocity, behind the strings' resistance) into the load. A string
## entered with `sensitivity` is entered as a whole: the sensitivity of the
## string as wired.
##
## **The flat band only.** A single sensitivity holds above the sensor's
## natural frequency. Near and below it a geophone's response falls off (12
## dB per octave below it) and its shape depends on the damping, so amplitudes
## there come out low: this module does not correct the response.

import std/[math, strutils]

type
  AmplitudeUnit* = enum
    ## The unit of a row's samples.
    auRecorded = "recorded"  ## As decoded from the file: counts, or units its headers do not state.
    auVolt = "V"             ## Volts at the recorder's input (the sensor's output).
    auVelocity = "m/s"       ## Particle velocity (geophones).
    auAcceleration = "m/s²" ## Acceleration (accelerometers).
    auPressure = "Pa"        ## Pressure (hydrophones).
    auForce = "N"            ## Force (a vibrator's ground force).

  SensitivityUnit* = enum
    ## The unit a sensitivity is entered in. `$` gives its symbol.
    suVPerMPerS = "V/(m/s)"              ## Volts per metre per second.
    suVPerInPerS = "V/(in/s)"            ## Volts per inch per second.
    suVPerMPerS2 = "V/(m/s²)"           ## Volts per metre per second squared.
    suVPerG = "V/g"                      ## Volts per standard gravity (9.80665 m/s²).
    suVPerBar = "V/bar"                  ## Volts per bar (10⁵ Pa).
    suVPerPa = "V/Pa"                    ## Volts per pascal.
    suMicroVPerPa = "µV/Pa"             ## Microvolts per pascal.
    suDbReVPerMicroPa = "dB re 1 V/µPa" ## Decibels relative to 1 V per micropascal.
    suVPerN = "V/N"                      ## Volts per newton.
    suVPerKN = "V/kN"                    ## Volts per kilonewton.
    suVPerLbf = "V/lbf"                  ## Volts per pound-force (4.4482216152605 N).

  SensitivityForm* = enum
    ## How a `Sensitivity` was entered.
    snNone = "none"                ## Not set (`Sensitivity()`).
    snAsWired = "as wired"         ## `sensitivity`: the value as the sensor is wired.
    snOpenCircuit = "open circuit" ## `openCircuitSensitivity`: computed from the resistances.

  Sensitivity* = object
    ## A sensor's output per unit of what it measures, in SI (volts per m/s,
    ## per m/s² or per Pa), with what was entered. `Sensitivity()` is not set
    ## (`isSet`). Build it with `sensitivity` or `openCircuitSensitivity`.
    form: SensitivityForm
    unit: AmplitudeUnit
    voltsPerUnit: float64 # the effective value, SI
    entered: float64 # the value as entered, in `enteredUnit`
    enteredUnit: SensitivityUnit
    coilOhms, shuntOhms, inputOhms: float64
    inSeries, inParallel: int

  DescaleUnit* = enum
    ## The unit a descale is entered in. `$` gives its symbol.
    duVPerCount = "V/count"
    duMilliVPerCount = "mV/count"
    duMicroVPerCount = "µV/count"
    duNanoVPerCount = "nV/count"

  Descale* = object
    ## Volts at the recorder's input per recorded unit. `Descale()` is not
    ## set (`isSet`). Build it with `descale`.
    voltsPerCount: float64 # 0: not set
    entered: float64
    enteredUnit: DescaleUnit

  FileScale* = object
    ## What a file's own headers say about one trace's scale, in SI, as
    ## `fileScalesInto` reads them (NaN where they say nothing).
    descale*: float64
      ## Volts per recorded unit; when `physical`, units of `unit` per
      ## recorded unit.
    sensitivity*: float64 ## Volts per unit of `unit`.
    unit*: AmplitudeUnit
      ## The quantity the headers name (`auVelocity`, `auAcceleration`,
      ## `auPressure`): the unit of `sensitivity`, or of the samples when
      ## `physical`, or (SEG-D Rev 3) the physical unit of a trace without a
      ## sensitivity; `auRecorded` when they name none.
    physical*: bool
      ## The headers say the samples are already in `unit` (times
      ## `descale`): no sensitivity applies.

  MassUnit* = enum
    ## The unit a mass is entered in. `$` gives its symbol.
    muKilogram = "kg" ## Kilograms.
    muPound = "lb"    ## Pounds (avoirdupois, 0.45359237 kg).
    muTonne = "t"     ## Tonnes (1000 kg).

  Mass* = object
    ## A mass in kilograms, with what was entered. `Mass()` is not set
    ## (`isSet`). Build it with `mass`.
    kilograms: float64 # 0: not set
    entered: float64
    enteredUnit: MassUnit

const
  inch = 0.0254                  # m
  standardG = 9.80665            # m/s²
  bar = 1e5                      # Pa
  pound = 0.45359237             # kg
  poundForce = pound * standardG # N: 4.4482216152605

proc badValue(what: string, v: float64) {.noinline, noreturn.} =
  raise newException(ValueError, what & " must be finite and positive, got " & $v)

func finite(x: float64): bool {.inline.} = x.classify notin {fcNan, fcInf, fcNegInf}

func quantityOf*(u: SensitivityUnit): AmplitudeUnit =
  ## What a sensitivity in `u` measures: `auVelocity`, `auAcceleration`,
  ## `auPressure` or `auForce`.
  case u
  of suVPerMPerS, suVPerInPerS: auVelocity
  of suVPerMPerS2, suVPerG: auAcceleration
  of suVPerBar, suVPerPa, suMicroVPerPa, suDbReVPerMicroPa: auPressure
  of suVPerN, suVPerKN, suVPerLbf: auForce

func toSi(value: float64, u: SensitivityUnit): float64 =
  # Volts per SI unit of the quantity.
  case u
  of suVPerMPerS, suVPerMPerS2, suVPerPa, suVPerN: value
  of suVPerInPerS: value / inch
  of suVPerG: value / standardG
  of suVPerBar: value / bar
  of suMicroVPerPa: value * 1e-6
  of suDbReVPerMicroPa: pow(10.0, value / 20) * 1e6 # V/µPa → V/Pa
  of suVPerKN: value * 1e-3
  of suVPerLbf: value / poundForce

func fromSi(si: float64, u: SensitivityUnit): float64 =
  case u
  of suVPerMPerS, suVPerMPerS2, suVPerPa, suVPerN: si
  of suVPerInPerS: si * inch
  of suVPerG: si * standardG
  of suVPerBar: si * bar
  of suMicroVPerPa: si * 1e6
  of suDbReVPerMicroPa: 20 * log10(si * 1e-6)
  of suVPerKN: si * 1e3
  of suVPerLbf: si * poundForce

proc checkedSi(value: float64, unit: SensitivityUnit): float64 =
  if unit == suDbReVPerMicroPa:
    if not value.finite:
      raise newException(ValueError, "a sensitivity in dB must be finite, got " & $value)
  elif not value.finite or value <= 0:
    badValue("a sensitivity", value)
  result = toSi(value, unit)
  if not result.finite or result <= 0:
    raise newException(ValueError, "a sensitivity of " & $value & " " & $unit &
      " is out of range")

proc sensitivity*(value: float64, unit: SensitivityUnit): Sensitivity =
  ## The sensitivity of a sensor (or a string of them) **as wired to the
  ## recorder**, used as it is: for a geophone the damped figure of its spec
  ## sheet (see the module documentation). Raises `ValueError` unless `value`
  ## is finite and positive (finite in dB).
  runnableExamples:
    let s = sensitivity(0.7, suVPerInPerS)
    doAssert abs(s.voltsPerUnit - 0.7 / 0.0254) < 1e-12 # V/(m/s)
    doAssert sensitivity(20, suVPerBar).voltsPerUnit == 2e-4 # V/Pa
  Sensitivity(form: snAsWired, unit: quantityOf(unit), voltsPerUnit: checkedSi(value, unit),
    entered: value, enteredUnit: unit, coilOhms: NaN, shuntOhms: NaN, inputOhms: NaN,
    inSeries: 1, inParallel: 1)

proc openCircuitSensitivity*(value: float64, unit: SensitivityUnit,
    coilOhms, shuntOhms: float64, inputOhms = Inf, inSeries: Positive = 1,
    inParallel: Positive = 1): Sensitivity =
  ## The sensitivity as wired, computed from one sensor's open-circuit
  ## (intrinsic) sensitivity `value`, its coil resistance and the shunt
  ## (damping) resistor across the string, for `inSeries` sensors per string
  ## and `inParallel` strings:
  ## `S = n·S₀ · R_L / (n·R_c / m + R_L)`, `R_L = shunt ∥ input`.
  ## `inputOhms` is the recorder's input impedance (`Inf`: not counted; leave
  ## it out when it is large against the shunt); `shuntOhms = Inf` means no
  ## shunt, so the input impedance alone loads the coil, and both `Inf` give
  ## the open-circuit value. Raises `ValueError` for a `value` as `sensitivity`
  ## refuses, a negative or non-finite coil resistance, a shunt or input
  ## impedance that is not positive, or a result that is not finite and
  ## positive (an overflow, or an underflow to 0).
  runnableExamples:
    # 20 V/(m/s) open circuit, a 400 Ω coil, a 1600 Ω shunt: 20 · 1600 / 2000.
    let s = openCircuitSensitivity(20, suVPerMPerS, coilOhms = 400, shuntOhms = 1600)
    doAssert abs(s.voltsPerUnit - 16) < 1e-12 and s.form == snOpenCircuit
  let s0 = checkedSi(value, unit)
  if not coilOhms.finite or coilOhms < 0:
    raise newException(ValueError, "the coil resistance must be finite and ≥ 0 Ω, got " &
      $coilOhms)
  if shuntOhms.isNaN or shuntOhms <= 0:
    raise newException(ValueError, "the shunt resistance must be positive (Inf: none), got " &
      $shuntOhms)
  if inputOhms.isNaN or inputOhms <= 0:
    raise newException(ValueError, "the input impedance must be positive (Inf: not " &
      "counted), got " & $inputOhms)
  let
    n = float64(inSeries)
    m = float64(inParallel)
    load = 1 / (1 / shuntOhms + 1 / inputOhms) # Inf when both are
    rc = n * coilOhms / m
    divider = (if load == Inf: 1.0 else: load / (rc + load))
    v = n * s0 * divider
  if not v.finite or v <= 0:
    raise newException(ValueError, "an open-circuit sensitivity of " & $value & " " & $unit &
      " with these resistances is out of range: " & $v & " V per unit")
  Sensitivity(form: snOpenCircuit, unit: quantityOf(unit), voltsPerUnit: v,
    entered: value, enteredUnit: unit, coilOhms: coilOhms, shuntOhms: shuntOhms,
    inputOhms: inputOhms, inSeries: inSeries, inParallel: inParallel)

func isSet*(s: Sensitivity): bool {.inline, raises: [].} =
  ## The sensitivity was entered (not `Sensitivity()`).
  s.form != snNone

func form*(s: Sensitivity): SensitivityForm {.inline, raises: [].} =
  ## How it was entered.
  s.form

func unit*(s: Sensitivity): AmplitudeUnit {.inline, raises: [].} =
  ## What the sensor measures: `auVelocity`, `auAcceleration`, `auPressure`
  ## or `auForce` (`auRecorded` when not set).
  s.unit

func voltsPerUnit*(s: Sensitivity): float64 {.inline, raises: [].} =
  ## The effective sensitivity in volts per SI unit (V/(m/s), V/(m/s²),
  ## V/Pa, V/N); NaN when not set.
  if s.form == snNone: NaN else: s.voltsPerUnit

proc valueIn*(s: Sensitivity, unit: SensitivityUnit): float64 =
  ## The effective sensitivity in `unit` (to show it in the unit a spec
  ## sheet uses). Raises `ValueError` when it is not set or `unit` measures
  ## another quantity.
  runnableExamples:
    let s = sensitivity(20, suVPerBar)
    doAssert abs(s.valueIn(suMicroVPerPa) - 200) < 1e-9
    doAssert abs(s.valueIn(suDbReVPerMicroPa) + 193.979) < 1e-3
  if s.form == snNone:
    raise newException(ValueError, "the sensitivity is not set")
  if quantityOf(unit) != s.unit:
    raise newException(ValueError, "a sensitivity per " & $s.unit & " cannot be given in " &
      $unit)
  fromSi(s.voltsPerUnit, unit)

func num(x: float64): string =
  # Six significant digits, without trailing zeros.
  if x == Inf: return "∞"
  let s = formatFloat(x, ffDefault, 6)
  let e = s.find('e')
  var mant = (if e < 0: s else: s[0 ..< e])
  if '.' in mant:
    mant.trimZeros()
    if mant.endsWith('.'): mant.setLen(mant.len - 1)
  result = (if e < 0: mant else: mant & s[e .. ^1])

func siUnit(u: AmplitudeUnit): string =
  case u
  of auVelocity: $suVPerMPerS
  of auAcceleration: $suVPerMPerS2
  of auPressure: $suVPerPa
  of auForce: $suVPerN
  else: ""

func `$`*(s: Sensitivity): string =
  ## The sensitivity as entered and, when that is not the same, its
  ## effective value in SI: `"22.5 V/(m/s)"`, `"-194 dB re 1 V/µPa (0.0001995
  ## V/Pa)"`, `"16 V/(m/s) (open circuit 20 V/(m/s), coil 400 Ω, shunt 1600 Ω)"`.
  case s.form
  of snNone: result = "not set"
  of snAsWired:
    result = num(s.entered) & " " & $s.enteredUnit
    if s.enteredUnit notin {suVPerMPerS, suVPerMPerS2, suVPerPa, suVPerN}:
      result.add " (" & num(s.voltsPerUnit) & " " & siUnit(s.unit) & ")"
  of snOpenCircuit:
    result = num(s.voltsPerUnit) & " " & siUnit(s.unit) & " (open circuit " & num(s.entered) &
      " " & $s.enteredUnit
    if s.inSeries > 1 or s.inParallel > 1:
      result.add " × " & $s.inSeries & " in series"
      if s.inParallel > 1: result.add ", " & $s.inParallel & " strings in parallel"
    result.add ", coil " & num(s.coilOhms) & " Ω, shunt " & num(s.shuntOhms) & " Ω"
    if s.inputOhms != Inf: result.add ", input " & num(s.inputOhms) & " Ω"
    result.add ")"

func toVolts(value: float64, u: DescaleUnit): float64 =
  case u
  of duVPerCount: value
  of duMilliVPerCount: value * 1e-3
  of duMicroVPerCount: value * 1e-6
  of duNanoVPerCount: value * 1e-9

proc descale*(value: float64, unit: DescaleUnit): Descale =
  ## A recorder's descale: `value` in `unit` (volts at its input per
  ## recorded count). Raises `ValueError` unless `value` is finite and
  ## positive.
  runnableExamples:
    doAssert descale(0.5, duMicroVPerCount).voltsPerCount == 5e-7
  if not value.finite or value <= 0: badValue("a descale", value)
  let v = toVolts(value, unit)
  if not v.finite or v <= 0:
    raise newException(ValueError, "a descale of " & $value & " " & $unit & " is out of range")
  Descale(voltsPerCount: v, entered: value, enteredUnit: unit)

func isSet*(d: Descale): bool {.inline, raises: [].} =
  ## The descale was entered (not `Descale()`).
  d.voltsPerCount > 0

func voltsPerCount*(d: Descale): float64 {.inline, raises: [].} =
  ## Volts per recorded unit; NaN when not set.
  if d.voltsPerCount > 0: d.voltsPerCount else: NaN

proc valueIn*(d: Descale, unit: DescaleUnit): float64 =
  ## The descale in `unit`. Raises `ValueError` when it is not set.
  if not d.isSet: raise newException(ValueError, "the descale is not set")
  d.voltsPerCount / toVolts(1, unit)

func `$`*(d: Descale): string =
  ## The descale as entered: `"0.5 µV/count"`.
  result = (if d.isSet: num(d.entered) & " " & $d.enteredUnit else: "not set")

func initFileScale*(): FileScale =
  ## A trace whose headers say nothing about its scale.
  FileScale(descale: NaN, sensitivity: NaN, unit: auRecorded)

func `$`*(f: FileScale): string =
  ## What the headers give, in SI: `"1e-06 V/count, 20 V/(m/s)"`.
  if f.physical:
    return (if f.descale.isNaN: "?" else: num(f.descale)) & " " & $f.unit & "/count (physical)"
  result = (if f.descale.isNaN: "no descale" else: num(f.descale) & " V/count")
  result.add ", "
  if not f.sensitivity.isNaN: result.add num(f.sensitivity) & " " & siUnit(f.unit)
  elif f.unit != auRecorded: result.add "no sensitivity (the headers name " & $f.unit & ")"
  else: result.add "no sensitivity"

# ------------------------------------------------------------------- mass --

func toKilograms(value: float64, u: MassUnit): float64 =
  case u
  of muKilogram: value
  of muPound: value * pound
  of muTonne: value * 1e3

proc mass*(value: float64, unit: MassUnit): Mass =
  ## A mass of `value` in `unit` (a vibrator's reaction mass or baseplate).
  ## Raises `ValueError` unless `value` is finite and positive.
  runnableExamples:
    doAssert mass(9000, muPound).kilograms == 9000 * 0.45359237
    doAssert $mass(2.45, muTonne) == "2.45 t"
  if not value.finite or value <= 0: badValue("a mass", value)
  let kg = toKilograms(value, unit)
  if not kg.finite or kg <= 0:
    raise newException(ValueError, "a mass of " & $value & " " & $unit & " is out of range")
  Mass(kilograms: kg, entered: value, enteredUnit: unit)

func isSet*(m: Mass): bool {.inline, raises: [].} =
  ## The mass was entered (not `Mass()`).
  m.kilograms > 0

func kilograms*(m: Mass): float64 {.inline, raises: [].} =
  ## The mass in kilograms; NaN when not set.
  if m.kilograms > 0: m.kilograms else: NaN

proc valueIn*(m: Mass, unit: MassUnit): float64 =
  ## The mass in `unit`. Raises `ValueError` when it is not set.
  if not m.isSet: raise newException(ValueError, "the mass is not set")
  m.kilograms / toKilograms(1, unit)

func `$`*(m: Mass): string =
  ## The mass as entered: `"2450 kg"`, `"9000 lb"`.
  result = (if m.isSet: num(m.entered) & " " & $m.enteredUnit else: "not set")
