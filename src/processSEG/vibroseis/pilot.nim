## Pilots (`Pilot`): the reference sweep that vibroseis records are
## correlated with, generated (`generatePilot`) or taken from a recorded
## trace (`pilotFromRow`, and `pilotFromTrace` in `adapters`). Both give the
## same type, so correlation does not care where a pilot came from.
##
## ```nim
## let spec = linearSweep(8, 80, lengthMs = 8000, dtMs = 2,
##   taper = initSweepTaper(250, 250))
## let p = generatePilot(spec, float32)         # samples, dtMs, t0Ms = 0, sweep
## doAssert p.len == 4001 and p.sweep.isSome
## # Write it with parseSEG (processSEG describes, parseSEG writes: D14):
## #   w.writeTrace(p.pilotKeys, p.samples, overrides = spec.sweepOverrides)
## ```
##
## **Time.** Sample `j` of a pilot is at `t0Ms + j·dtMs` from the time
## break. A generated pilot starts at 0; a recorded one at its trace's
## start time, which can be a fraction of a sample (correlation shifts by
## it exactly, in the frequency domain).
##
## **Sample rate** (task 10.3b). Correlation refuses a pilot whose `dtMs`
## differs from the records': `resamplePilot` converts it explicitly, with a
## band-limited (Kaiser-windowed sinc) interpolation, and refuses when the
## pilot's sweep reaches above the new anti-alias limit.
##
## **Instrument response.** Data reach the recorder through its anti-alias
## filter, often minimum phase; a recorded pilot went through it too, a
## synthetic one did not, which leaves the correlated data with the filter's
## phase. `applyInstrumentFilter` passes a pilot through the same filter
## (`InstrumentFilter`: Butterworth high-cut and optional low-cut, minimum
## or zero phase, as SEG-Y trace bytes 141-152 describe the recorder's
## filters), so that it matches the data.
##
## **Headers.** `sweepOverrides` gives the SEG-Y sweep fields of a spec, for
## parseSEG's `writeTrace` (`trace.` fields) or `HeaderEditor.setFileField`
## (`binary.` fields), and `pilotKeys` a pilot trace's keys (a sweep trace,
## identification code 6). Header values are whole Hz and ms (int16).

import std/[math, options]
import parseSEG
import ../core/[types, workspace]
import ../filter/[design, apply, iir]
import taper, sweep, kaiser

export options

type
  Pilot*[T: SampleType] = object
    ## A pilot sweep: its samples at `dtMs`, sample 0 at `t0Ms` from the
    ## time break, and the sweep it is when known. Make one with
    ## `generatePilot`, `initPilot`, `pilotFromRow` or `pilotFromTrace`.
    samples*: seq[T]
    dtMs*: float64
    t0Ms*: float64
    sweep*: Option[SweepSpec]

  InstrumentPhase* = enum
    ## The phase of a recorder's filters.
    ipMinimumPhase ## causal (analogue and minimum-phase digital filters)
    ipZeroPhase    ## linear phase with its delay removed

  InstrumentFilter* = object
    ## A recorder's filters, as Butterworth responses (SEG-Y trace bytes
    ## 141-152: alias filter and low-cut frequency and slope). A NaN or 0
    ## frequency means no such filter. Slopes are of the magnitude, in
    ## dB/octave: a multiple of 6 (minimum phase) or 12 (zero phase).
    highCutHz*: float64
    highSlopeDbPerOctave*: float64
    lowCutHz*: float64
    lowSlopeDbPerOctave*: float64
    phase*: InstrumentPhase

proc badPilot(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func len*[T](p: Pilot[T]): int {.inline, raises: [].} =
  ## The number of samples.
  p.samples.len

func lengthMs*[T](p: Pilot[T]): float64 {.inline, raises: [].} =
  ## The pilot's duration, `(len − 1)·dtMs`.
  float64(max(p.samples.len - 1, 0)) * p.dtMs

proc validate*[T](p: Pilot[T]) =
  ## Checks a pilot: at least two samples, all finite, not all zero; a
  ## finite positive `dtMs`; a finite `t0Ms`; the sweep's `dtMs` equal to
  ## the pilot's. Raises `ValueError`.
  if p.samples.len < 2: badPilot("a pilot needs at least two samples, got " & $p.samples.len)
  if not (p.dtMs > 0 and p.dtMs < Inf):
    badPilot("a pilot's dtMs must be finite and positive, got " & $p.dtMs)
  if p.t0Ms.isNaN or abs(p.t0Ms) == Inf:
    badPilot("a pilot's t0Ms must be finite, got " & $p.t0Ms)
  var energy = 0.0
  for i, v in p.samples:
    let x = float64(v)
    if x.isNaN or abs(x) == Inf: badPilot("pilot sample " & $i & " is not finite")
    energy += x * x
  if energy == 0: badPilot("the pilot is all zeros")
  if p.sweep.isSome and abs(p.sweep.get.dtMs - p.dtMs) > 1e-9 * p.dtMs:
    badPilot("the pilot's sweep is sampled at " & $p.sweep.get.dtMs & " ms, the pilot at " &
      $p.dtMs & " ms")

proc initPilot*[T: SampleType](samples: sink seq[T], dtMs: float64, t0Ms = 0.0,
    sweep = none(SweepSpec)): Pilot[T] =
  ## A pilot from samples the caller has (a vendor file, a ground-force
  ## record). Raises `ValueError` as `validate`.
  result = Pilot[T](samples: samples, dtMs: dtMs, t0Ms: t0Ms, sweep: sweep)
  result.validate

proc generatePilot*(spec: SweepSpec, T: typedesc[SampleType] = float64): Pilot[T] =
  ## The pilot of the sweep `spec`, from `t0Ms = 0`, with `sweep` set.
  ## Raises `ValueError` for an invalid spec.
  result = Pilot[T](samples: spec.sweepSamples(T), dtMs: spec.dtMs, t0Ms: 0,
    sweep: some(spec))

proc pilotFromRow*[T: SampleType](meta: ChunkMeta, data: Tensor[T], row: int,
    t0Ms = NaN, sweep = none(SweepSpec)): Pilot[T] =
  ## The pilot recorded in row `row` of a block (an auxiliary pilot or
  ## ground-force trace the caller read with parseSEG): its live samples
  ## (`liveSamples`), the meta's `dtMs`, and its start time
  ## (`sampleTimeMs(row, 0)`, which includes `firstSample`) unless `t0Ms` is
  ## given. The row need not be live (pilots are auxiliary traces), but must
  ## not be a failed one. Raises `ValueError` for a row outside the block, a
  ## failed row, a shape mismatch, an unknown start time without `t0Ms`, or
  ## a pilot `validate` rejects.
  meta.checkShape(data)
  if row notin 0 ..< meta.len: badPilot("row " & $row & " is outside 0 ..< " & $meta.len)
  if meta.isFailed(row): badPilot("row " & $row & " could not be decoded")
  let n = meta.liveSamples(row, data.shape[1])
  var s = newSeq[T](n)
  for j in 0 ..< n: s[j] = data[row, j]
  let t0 = if t0Ms.isNaN: meta.sampleTimeMs(row, 0) else: t0Ms
  if t0.isNaN: badPilot("row " & $row & " has no start time in its keys: pass t0Ms")
  initPilot(s, meta.dtMs, t0, sweep)

# ------------------------------------------------------------- headers --

func pilotKeys*[T](p: Pilot[T]): TraceKeys =
  ## Keys for writing `p` as a SEG-Y trace through parseSEG's
  ## `writeTrace(w, keys, samples, overrides)`: its interval and start time,
  ## identification code 6 (sweep), status auxiliary; every other key
  ## unknown.
  result = initTraceKeys()
  result.intervalUs = p.dtMs * 1000
  result.t0Ms = p.t0Ms
  result.traceCode = 6
  result.status = tsAuxiliary

proc int16Field(v: float64, what: string): int16 =
  let r = round(v)
  if not (r >= float64(low(int16)) and r <= float64(high(int16))):
    badPilot(what & " = " & $v & " does not fit a SEG-Y int16 field")
  int16(r)

proc sweepOverrides*(spec: SweepSpec, binary = false): seq[HeaderOverride] =
  ## The SEG-Y sweep fields of `spec`: start and end frequency (Hz), length
  ## (ms), sweep type (`segySweepType`), start and end taper lengths (ms)
  ## and taper type (`segyTaperType`). As `trace.` fields (bytes 127-140)
  ## for `writeTrace`'s `overrides`, or with `binary`, as `binary.` fields
  ## (3233-3248, without the sweep channel) for
  ## `HeaderEditor.setFileField(o.name, o.value)` after writing. Values are
  ## rounded to whole Hz and ms. Raises `ValueError` for a value outside
  ## int16 (a sweep over 32.767 s).
  let pre = if binary: "binary." else: "trace."
  let t = spec.taper
  let ts = if t.startMs.isNaN: 0.0 else: t.startMs
  let te = if t.endMs.isNaN: 0.0 else: t.endMs
  @[headerOverride(pre & "sweepFreqStart", int16Field(spec.f1Hz, "f1Hz")),
    headerOverride(pre & "sweepFreqEnd", int16Field(spec.f2Hz, "f2Hz")),
    headerOverride(pre & "sweepLength", int16Field(spec.lengthMs, "lengthMs")),
    headerOverride(pre & "sweepType", spec.segySweepType),
    headerOverride(pre & "sweepTaperStart", int16Field(ts, "taper startMs")),
    headerOverride(pre & "sweepTaperEnd", int16Field(te, "taper endMs")),
    headerOverride(pre & "taperType", t.segyTaperType)]

func sweepFromHeaders*(f1Hz, f2Hz, lengthMs, sweepType, taperStartMs, taperEndMs,
    taperType, dtMs: float64): Option[SweepSpec] =
  ## The sweep that SEG-Y sweep fields describe, when they describe one
  ## fully: frequencies and a length above 0, and type 1 (linear) or 3
  ## (exponential); otherwise none (types 2 and 4 do not say the law). The
  ## taper shape is linear for type 1, cos² for 2 and 3 (other). The spec is
  ## not validated: a recorded sweep may sit closer to Nyquist than
  ## generation allows (`maxNyquistFraction` is 1).
  if not (f1Hz > 0 and f2Hz > 0 and lengthMs > 0 and dtMs > 0) or
      sweepType notin [1.0, 3.0]:
    return none(SweepSpec)
  let shape = if taperType == 1: tpLinear else: tpCos2
  some(SweepSpec(law: if sweepType == 1: slLinear else: slExponential, f1Hz: f1Hz,
    f2Hz: f2Hz, lengthMs: lengthMs, dtMs: dtMs, amplitude: 1,
    taper: SweepTaper(startMs: max(taperStartMs, 0), endMs: max(taperEndMs, 0),
    shape: shape, param: NaN), maxNyquistFraction: 1))

# ------------------------------------------------------------ resampling --

proc resamplePilot*[T: SampleType](p: Pilot[T], dtMs: float64, cutoffFraction = 0.9,
    halfWidth = 24): Pilot[T] =
  ## `p` at the interval `dtMs`, by band-limited interpolation: a sinc low-pass
  ## at `cutoffFraction` of the lower of the two Nyquist frequencies, Kaiser
  ## windowed (β = 8) over `halfWidth` samples of the coarser interval on
  ## each side. The new pilot covers the same time span from the same
  ## `t0Ms`. Its sweep, when known, takes the new `dtMs`, and
  ## `cutoffFraction` as its `maxNyquistFraction` (the band the new pilot
  ## holds). Raises `ValueError` for a bad `dtMs`, `cutoffFraction` outside
  ## (0, 1], `halfWidth < 2`, or a known sweep above the cutoff.
  p.validate
  if not (dtMs > 0 and dtMs < Inf): badPilot("dtMs must be finite and positive, got " & $dtMs)
  if not (cutoffFraction > 0 and cutoffFraction <= 1):
    badPilot("cutoffFraction must be in (0, 1], got " & $cutoffFraction)
  if halfWidth < 2: badPilot("halfWidth must be ≥ 2, got " & $halfWidth)
  let coarse = max(dtMs, p.dtMs)
  let fc = cutoffFraction * 500 / coarse # Hz
  var sweep = p.sweep
  if sweep.isSome:
    var s = sweep.get
    s.dtMs = dtMs
    s.maxNyquistFraction = cutoffFraction # what the resampled pilot holds
    let top = max(s.f1Hz, s.f2Hz)
    if top > fc:
      badPilot("the sweep reaches " & $top & " Hz, above the resampling cutoff " & $fc &
        " Hz at " & $dtMs & " ms")
    try: s.validate
    except ValueError as e: badPilot("the resampled sweep is invalid: " & e.msg)
    sweep = some(s)
  let n = int(floor(p.lengthMs / dtMs * (1 + 1e-9))) + 1
  var outS = newSeq[T](n)
  let w = float64(halfWidth) * coarse # ms
  let beta = 8.0
  let i0b = besselI0(beta)
  let scale = 2 * fc / 1000 * p.dtMs # sinc gain per input sample
  for j in 0 ..< n:
    let t = float64(j) * dtMs # from t0
    let lo = max(0, int(ceil((t - w) / p.dtMs)))
    let hi = min(p.samples.len - 1, int(floor((t + w) / p.dtMs)))
    var acc = 0.0
    for i in lo .. hi:
      let d = float64(i) * p.dtMs - t # ms
      let u = d / w
      if abs(u) >= 1: continue
      let x = 2 * fc / 1000 * d
      let sinc = if x == 0: 1.0 else: sin(PI * x) / (PI * x)
      acc += float64(p.samples[i]) * sinc * besselI0(beta * sqrt(1 - u * u)) / i0b
    outS[j] = T(acc * scale)
  initPilot(outS, dtMs, p.t0Ms, sweep)

# -------------------------------------------------------- instrument filter --

func hasFreq(f: float64): bool {.inline.} = f > 0 and f < Inf

proc toSpec(f: InstrumentFilter): FilterSpec =
  let hi = hasFreq(f.highCutHz)
  let lo = hasFreq(f.lowCutHz)
  if hi and lo:
    if f.lowSlopeDbPerOctave != f.highSlopeDbPerOctave:
      badPilot("an instrument band-pass needs equal slopes here, got " &
        $f.lowSlopeDbPerOctave & " and " & $f.highSlopeDbPerOctave & " dB/octave")
    butterBandPass(f.lowCutHz, f.highCutHz, f.highSlopeDbPerOctave)
  elif hi: butterHighCut(f.highCutHz, f.highSlopeDbPerOctave)
  elif lo: butterLowCut(f.lowCutHz, f.lowSlopeDbPerOctave)
  else: badPilot("an instrument filter needs a high-cut or a low-cut frequency")

proc applyInstrumentFilter*[T: SampleType](p: Pilot[T], f: InstrumentFilter): Pilot[T] =
  ## `p` passed through the recorder's filters `f`, from rest: zeros are
  ## placed before and after it, so a minimum-phase filter starts from a
  ## zero state and its response is causal, and the result keeps `p`'s
  ## samples and start time (the filtered tail past the pilot's end is cut,
  ## as the record is). Raises `ValueError` for a filter without a
  ## frequency, unequal band-pass slopes, or a spec the Butterworth design
  ## rejects (slopes, frequencies above Nyquist).
  p.validate
  let spec = f.toSpec
  let n = p.len
  case f.phase
  of ipMinimumPhase:
    let probe = initIirPlan(T, spec, p.dtMs, n, ipMinimum, trNone)
    let pad = probe.requiredMargin + 1
    let plan = initIirPlan(T, spec, p.dtMs, n + 2 * pad, ipMinimum, trNone)
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var buf = newSeq[T](n + 2 * pad)
    for j in 0 ..< n: buf[pad + j] = p.samples[j]
    buf.applyFilter(plan, ws)
    result = p
    for j in 0 ..< n: result.samples[j] = buf[pad + j]
  of ipZeroPhase:
    let probe = initFilterPlan(T, spec, p.dtMs, n, trNone)
    let pad = probe.requiredMargin + 1
    let plan = initFilterPlan(T, spec, p.dtMs, n + 2 * pad, trNone)
    var ws = initWorkspace()
    ws.reserve(plan.workspaceSize)
    var buf = newSeq[T](n + 2 * pad)
    for j in 0 ..< n: buf[pad + j] = p.samples[j]
    buf.applyFilter(plan, ws)
    result = p
    for j in 0 ..< n: result.samples[j] = buf[pad + j]
