## Amplitude units (task 14.1): sensitivities and descales entered with their
## units, converted to SI exactly; the damped sensitivity from the
## open-circuit one and the resistances; `valueIn` round trips; refusals.

import std/[assertions, math, strutils]
import processSEG

func closeTo(a, b: float64, rel = 1e-12): bool = abs(a - b) <= rel * max(abs(a), abs(b))

block sensitivity_units_to_si:
  # One unit of each, against the SI value worked by hand.
  doAssert sensitivity(28.8, suVPerMPerS).voltsPerUnit == 28.8
  doAssert closeTo(sensitivity(1, suVPerInPerS).voltsPerUnit, 39.37007874015748) # 1/0.0254
  doAssert closeTo(sensitivity(0.73, suVPerInPerS).voltsPerUnit, 0.73 / 0.0254)
  doAssert sensitivity(2, suVPerMPerS2).voltsPerUnit == 2
  doAssert closeTo(sensitivity(1, suVPerG).voltsPerUnit, 0.10197162129779283) # 1/9.80665
  doAssert closeTo(sensitivity(20, suVPerBar).voltsPerUnit, 2e-4)
  doAssert sensitivity(3e-4, suVPerPa).voltsPerUnit == 3e-4
  doAssert closeTo(sensitivity(200, suMicroVPerPa).voltsPerUnit, 2e-4)
  # dB re 1 V/µPa: 20 V/bar = 2e-10 V/µPa = 20·log10(2e-10) = −193.979 dB.
  let db = 20 * log10(2e-10)
  doAssert closeTo(db, -193.97940008672037)
  doAssert closeTo(sensitivity(db, suDbReVPerMicroPa).voltsPerUnit, 2e-4)
  doAssert closeTo(sensitivity(-180, suDbReVPerMicroPa).voltsPerUnit, 1e-9 * 1e6) # 1e-3 V/Pa
  # Quantities.
  for (u, q) in [(suVPerMPerS, auVelocity), (suVPerInPerS, auVelocity),
      (suVPerMPerS2, auAcceleration), (suVPerG, auAcceleration), (suVPerBar, auPressure),
      (suVPerPa, auPressure), (suMicroVPerPa, auPressure), (suDbReVPerMicroPa, auPressure)]:
    doAssert quantityOf(u) == q
    doAssert sensitivity(1, u).unit == q
    doAssert sensitivity(1, u).form == snAsWired and sensitivity(1, u).isSet

block value_in_round_trips:
  for u in SensitivityUnit:
    let v = (if u == suDbReVPerMicroPa: -190.5 else: 12.75)
    let s = sensitivity(v, u)
    doAssert closeTo(s.valueIn(u), v, 1e-12), $u
    for w in SensitivityUnit:
      if quantityOf(w) == s.unit:
        doAssert closeTo(sensitivity(s.valueIn(w), w).voltsPerUnit, s.voltsPerUnit, 1e-12)
      else:
        doAssertRaises(ValueError): discard s.valueIn(w)
  doAssert closeTo(sensitivity(20, suVPerBar).valueIn(suMicroVPerPa), 200)
  doAssert closeTo(sensitivity(1, suVPerMPerS).valueIn(suVPerInPerS), 0.0254)

block open_circuit_divider:
  # The coil is a source of EMF S₀·v behind its resistance R_c, loaded by
  # the shunt: V = S₀·v·R_s / (R_c + R_s). Worked by hand: 20 V/(m/s), a
  # 400 Ω coil and a 1600 Ω shunt give 20 · 1600 / 2000 = 16 V/(m/s).
  let a = openCircuitSensitivity(20, suVPerMPerS, coilOhms = 400, shuntOhms = 1600)
  doAssert closeTo(a.voltsPerUnit, 16) and a.form == snOpenCircuit and a.unit == auVelocity
  # The recorder's 1600 Ω input in parallel with the shunt: R_L = 800 Ω,
  # 20 · 800 / 1200 = 13.333 V/(m/s).
  let b = openCircuitSensitivity(20, suVPerMPerS, 400, 1600, inputOhms = 1600)
  doAssert closeTo(b.voltsPerUnit, 40 / 3)
  # A string of 6 in series, 2 strings in parallel, shunted by 1800 Ω:
  # EMF 6·20 = 120 V/(m/s), resistance 6·400/2 = 1200 Ω: 120 · 1800 / 3000 = 72.
  let c = openCircuitSensitivity(20, suVPerMPerS, 400, 1800, inSeries = 6, inParallel = 2)
  doAssert closeTo(c.voltsPerUnit, 72)
  # A common spec-sheet case in inches: 0.73 V/(in/s) open circuit (28.74
  # V/(m/s)), a 375 Ω coil, 1339 Ω shunt: × 1339/1714.
  let d = openCircuitSensitivity(0.73, suVPerInPerS, 375, 1339)
  doAssert closeTo(d.voltsPerUnit, 0.73 / 0.0254 * 1339 / 1714)
  # No shunt and no input counted: the open-circuit value itself; a large
  # input impedance barely loads it.
  doAssert closeTo(openCircuitSensitivity(20, suVPerMPerS, 400, Inf).voltsPerUnit, 20)
  doAssert closeTo(openCircuitSensitivity(20, suVPerMPerS, 400, Inf, inputOhms = 20e3).voltsPerUnit,
    20 * 20e3 / 20.4e3)
  # A zero coil resistance: no division.
  doAssert closeTo(openCircuitSensitivity(20, suVPerMPerS, 0, 1000).voltsPerUnit, 20)

block printing:
  doAssert $sensitivity(22.5, suVPerMPerS) == "22.5 V/(m/s)"
  doAssert $sensitivity(20, suVPerBar) == "20 V/bar (0.0002 V/Pa)"
  let s = $openCircuitSensitivity(20, suVPerMPerS, coilOhms = 400, shuntOhms = 1600)
  doAssert s == "16 V/(m/s) (open circuit 20 V/(m/s), coil 400 Ω, shunt 1600 Ω)", s
  let t = $openCircuitSensitivity(20, suVPerMPerS, 400, 1800, inputOhms = 1e4, inSeries = 6,
    inParallel = 2)
  doAssert "× 6 in series, 2 strings in parallel" in t and "input 10000 Ω" in t, t
  doAssert $Sensitivity() == "not set"
  doAssert $descale(0.5, duMicroVPerCount) == "0.5 µV/count"
  doAssert $initFileScale() == "no descale, no sensitivity"
  doAssert $FileScale(descale: 1e-6, sensitivity: 20, unit: auVelocity) ==
    "1e-06 V/count, 20 V/(m/s)"

block descale_units:
  doAssert descale(1, duVPerCount).voltsPerCount == 1
  doAssert closeTo(descale(2, duMilliVPerCount).voltsPerCount, 2e-3)
  doAssert closeTo(descale(0.5, duMicroVPerCount).voltsPerCount, 5e-7)
  doAssert closeTo(descale(250, duNanoVPerCount).voltsPerCount, 2.5e-7)
  doAssert closeTo(descale(0.5, duMicroVPerCount).valueIn(duNanoVPerCount), 500)
  doAssert descale(1, duVPerCount).isSet and not Descale().isSet
  doAssert Descale().voltsPerCount.isNaN and Sensitivity().voltsPerUnit.isNaN
  doAssert not Sensitivity().isSet and Sensitivity().form == snNone

block refusals:
  for bad in [0.0, -1.0, NaN, Inf, -Inf]:
    doAssertRaises(ValueError): discard sensitivity(bad, suVPerMPerS)
    doAssertRaises(ValueError): discard descale(bad, duMicroVPerCount)
  for bad in [NaN, Inf, -Inf]:
    doAssertRaises(ValueError): discard sensitivity(bad, suDbReVPerMicroPa)
  # A value whose SI form is out of range.
  doAssertRaises(ValueError): discard sensitivity(1e308, suVPerInPerS)
  doAssertRaises(ValueError): discard sensitivity(1e5, suDbReVPerMicroPa)
  doAssertRaises(ValueError): discard descale(1e-320, duNanoVPerCount)
  # Resistances.
  for bad in [-1.0, NaN, Inf]:
    doAssertRaises(ValueError):
      discard openCircuitSensitivity(20, suVPerMPerS, coilOhms = bad, shuntOhms = 1000)
  for bad in [0.0, -5.0, NaN]:
    doAssertRaises(ValueError):
      discard openCircuitSensitivity(20, suVPerMPerS, coilOhms = 400, shuntOhms = bad)
    doAssertRaises(ValueError):
      discard openCircuitSensitivity(20, suVPerMPerS, 400, 1000, inputOhms = bad)
  doAssertRaises(ValueError): discard openCircuitSensitivity(0, suVPerMPerS, 400, 1000)
  doAssertRaises(ValueError): discard Sensitivity().valueIn(suVPerMPerS)
  doAssertRaises(ValueError): discard Descale().valueIn(duVPerCount)
