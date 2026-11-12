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
## band-limited (Kaiser-windowed sinc) interpolation, and refuses a pilot
## whose samples hold energy above the interpolator's passband
## (`energyAbove`; what its headers say of the sweep is not asked).
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
## identification code 6). Header values are whole Hz and ms (int16): a
## sweep with a fractional frequency or length has no header description
## and `sweepOverrides` refuses it. Read back (`sweepFromHeaders`), fields
## count as a description only when a trace at that interval could hold
## the sweep; controllers that write other units (0.1 Hz) are common.

import std/[math, options]
import parseSEG
import ../core/[types, workspace]
import ../fft/plan
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

proc int16Field(v: float64, what, unit: string): int16 =
  let r = round(v)
  if not (r >= float64(low(int16)) and r <= float64(high(int16))):
    badPilot(what & " = " & $v & " does not fit a SEG-Y int16 field")
  if abs(v - r) > 1e-9 * max(1.0, abs(r)):
    badPilot(what & " = " & $v & " " & unit & " is not a whole number: the SEG-Y sweep " &
      "fields hold whole " & unit & ", and a rounded value would be read back as " &
      "the sweep (write the trace without them)")
  int16(r)

proc sweepOverrides*(spec: SweepSpec, binary = false): seq[HeaderOverride] =
  ## The SEG-Y sweep fields of `spec`: start and end frequency (Hz), length
  ## (ms), sweep type (`segySweepType`), start and end taper lengths (ms)
  ## and taper type (`segyTaperType`). As `trace.` fields (bytes 127-140)
  ## for `writeTrace`'s `overrides`, or with `binary`, as `binary.` fields
  ## (3233-3248, without the sweep channel) for
  ## `HeaderEditor.setFileField(o.name, o.value)` after writing. The fields
  ## are whole Hz and ms: raises `ValueError` for a frequency, length or
  ## taper that is not a whole number (rounded, it would be read back as a
  ## sweep the pilot is not), or outside int16 (a sweep over 32.767 s).
  let pre = if binary: "binary." else: "trace."
  let t = spec.taper
  let ts = if t.startMs.isNaN: 0.0 else: t.startMs
  let te = if t.endMs.isNaN: 0.0 else: t.endMs
  @[headerOverride(pre & "sweepFreqStart", int16Field(spec.f1Hz, "f1Hz", "Hz")),
    headerOverride(pre & "sweepFreqEnd", int16Field(spec.f2Hz, "f2Hz", "Hz")),
    headerOverride(pre & "sweepLength", int16Field(spec.lengthMs, "lengthMs", "ms")),
    headerOverride(pre & "sweepType", spec.segySweepType),
    headerOverride(pre & "sweepTaperStart", int16Field(ts, "taper startMs", "ms")),
    headerOverride(pre & "sweepTaperEnd", int16Field(te, "taper endMs", "ms")),
    headerOverride(pre & "taperType", t.segyTaperType)]

proc sweepFromHeaders*(f1Hz, f2Hz, lengthMs, sweepType, taperStartMs, taperEndMs,
    taperType, dtMs: float64, samples = 0): Option[SweepSpec] =
  ## The sweep that SEG-Y sweep fields describe, when they describe one
  ## that a trace at `dtMs` can hold: type 1 (linear) or 3 (exponential;
  ## types 2 and 4 do not say the law), and a spec that `validate` accepts
  ## at `dtMs` with frequencies up to Nyquist (`maxNyquistFraction` is 1: a
  ## recorded sweep may sit closer to Nyquist than generation allows); with
  ## `samples` (the trace's sample count; 0: not checked) also no longer
  ## than the trace. Otherwise none: fields that do not fit the trace say
  ## nothing of it (a controller writing 80 and 1100 for an 8–110 Hz sweep
  ## at 1 ms). The taper shape is linear for type 1, cos² for 2 and 3
  ## (other).
  if sweepType notin [1.0, 3.0] or not (dtMs > 0 and dtMs < Inf):
    return none(SweepSpec)
  if samples > 0 and not (lengthMs <= float64(samples) * dtMs):
    return none(SweepSpec)
  let spec = SweepSpec(law: if sweepType == 1: slLinear else: slExponential, f1Hz: f1Hz,
    f2Hz: f2Hz, lengthMs: lengthMs, dtMs: dtMs, amplitude: 1,
    taper: SweepTaper(startMs: max(taperStartMs, 0), endMs: max(taperEndMs, 0),
    shape: if taperType == 1: tpLinear else: tpCos2, param: NaN), maxNyquistFraction: 1)
  try:
    spec.validate
    result = some(spec)
  except ValueError:
    result = none(SweepSpec)

# -------------------------------------------------------------- spectrum --

proc energyAbove*[T: SampleType](p: Pilot[T], fHz: float64): float64 =
  ## The fraction of the pilot's energy at frequencies above `fHz`, from
  ## the power spectrum of its samples: 1 for `fHz < 0`, 0 from Nyquist up.
  ## What a pilot holds, whatever its headers say. NaN for a pilot without
  ## energy or with a sample that is not finite (`validate` rejects both).
  if p.samples.len == 0 or not (p.dtMs > 0): return NaN
  let n = nextFastLen(p.samples.len)
  let fft = initFftPlan(float64, n)
  var ws = initWorkspace()
  ws.reserve(fft.workspaceSize(batched = false))
  var x = newSeq[float64](n)
  for j, v in p.samples: x[j] = float64(v)
  var re, im = newSeq[float64](fft.bins)
  fft.forward(x, re, im, ws)
  let binHz = 1000 / (float64(n) * p.dtMs)
  var total, above = 0.0
  for k in 0 ..< fft.bins:
    # Each bin but 0 Hz and Nyquist stands for itself and its mirror image.
    let e = (re[k] * re[k] + im[k] * im[k]) * (if k == 0 or 2 * k == n: 1.0 else: 2.0)
    total += e
    if float64(k) * binHz > fHz: above += e
  above / total

# ------------------------------------------------------------ resampling --

const
  resampleLeak = 0.005
    # The share of a pilot's energy that may lie above the resampling
    # passband: the side lobes of an 8 s sweep that ends there (0.06 % with
    # 250 ms tapers, 0.4 % without) pass, a sweep that runs 1 % of its band
    # beyond does not.
  kaiserBeta = 8.0
  transitionWidth = sqrt(1 + (kaiserBeta / PI) * (kaiserBeta / PI))
    # 2.736: the half width of the Kaiser window's main lobe, in units of
    # 1 / (the window's length).

const maxPhases = 64 # the most kernels tabulated; beyond, one per output sample

func kernelTap(d, w, fc, i0b: float64): float64 {.inline.} =
  # The Kaiser-windowed sinc of cutoff `fc` Hz at `d` ms from its centre, 0
  # from the window's end `w` on.
  let u = d / w
  if abs(u) >= 1: 0.0
  else:
    let x = 2 * fc / 1000 * d
    (if x == 0: 1.0 else: sin(PI * x) / (PI * x)) *
      besselI0(kaiserBeta * sqrt(1 - u * u)) / i0b

proc dot[T](h: ptr UncheckedArray[float64], x: ptr UncheckedArray[T],
    n: int): float64 {.raises: [].} =
  # Σ h[k]·x[k] over k < n, in four partial sums (the loop vectorizes).
  {.push boundChecks: off, overflowChecks: off.}
  var s: array[4, float64]
  var k = 0
  while k + 4 <= n:
    for l in 0 ..< 4: s[l] += h[k + l] * float64(x[k + l])
    k += 4
  result = (s[0] + s[1]) + (s[2] + s[3])
  while k < n:
    result += h[k] * float64(x[k])
    inc k
  {.pop.}

proc resamplePilot*[T: SampleType](p: Pilot[T], dtMs: float64, cutoffFraction = 0.9,
    halfWidth = 32, bandLimit = false): Pilot[T] =
  ## `p` at the interval `dtMs`, by band-limited interpolation: a sinc low-pass
  ## at `cutoffFraction` of the lower of the two Nyquist frequencies, Kaiser
  ## windowed (β = 8) over `halfWidth` samples of the coarser interval on
  ## each side. The new pilot covers the same time span from the same
  ## `t0Ms`.
  ##
  ## **Band.** The window gives the low-pass a transition band of
  ## `2.736 / halfWidth` of that Nyquist on each side of the cutoff: flat
  ## (to 0.01 dB) up to the passband edge, `cutoffFraction − 2.736 /
  ## halfWidth` (0.8145 with the defaults, so a sweep generated with the
  ## default `maxNyquistFraction` of 0.8 fits), and closed (−80 dB) from
  ## `cutoffFraction + 2.736 / halfWidth` (0.9855: below Nyquist, nothing
  ## aliases). A pilot whose samples hold energy above the passband edge
  ## (`energyAbove`: more than 0.5 % of it) is refused, whatever its sweep
  ## description says: the result would not be the same sweep. With
  ## `bandLimit` it is cut there instead, which is what matches records
  ## that went through an anti-alias filter at the new interval; the result
  ## is then no longer described. Otherwise the description, when there is
  ## one, takes the new `dtMs` and the passband edge as its
  ## `maxNyquistFraction` (the band the new pilot holds), and is kept only
  ## if it is still valid (not, for instance, when its length is not a
  ## whole number of the new samples).
  ##
  ## **Ends.** Within the window's reach of the pilot's ends part of the
  ## kernel has no samples under it; the rest is rescaled to the whole
  ## kernel's sum, so a pilot that does not start at zero keeps its level
  ## there (to the extent that a one-sided kernel can interpolate).
  ##
  ## Raises `ValueError` for a bad `dtMs`, `cutoffFraction` outside (0, 1],
  ## a `halfWidth` that leaves no passband (`≤ 2.736 / cutoffFraction`), a
  ## result of over 2³¹ samples, or energy above the passband without
  ## `bandLimit`.
  p.validate
  if not (dtMs > 0 and dtMs < Inf): badPilot("dtMs must be finite and positive, got " & $dtMs)
  if not (cutoffFraction > 0 and cutoffFraction <= 1):
    badPilot("cutoffFraction must be in (0, 1], got " & $cutoffFraction)
  if halfWidth < 1 or cutoffFraction <= transitionWidth / float64(halfWidth):
    badPilot("halfWidth = " & $halfWidth & " leaves no passband below the cutoff: it " &
      "must be above " & $(transitionWidth / cutoffFraction))
  let coarse = max(dtMs, p.dtMs)
  let nyquist = 500 / coarse # Hz
  let fc = cutoffFraction * nyquist
  let edge = fc - transitionWidth / float64(halfWidth) * nyquist       # the passband's end
  let cut = p.energyAbove(edge) > resampleLeak
  if cut and not bandLimit:
    badPilot($(round(1e4 * p.energyAbove(edge)) / 100) & " % of the pilot's energy is " &
      "above " & $(round(10 * edge) / 10) & " Hz, where the resampling to " & $dtMs &
      " ms starts to cut (bandLimit = true accepts that)")
  var sweep = none(SweepSpec)
  if p.sweep.isSome and not cut:
    var s = p.sweep.get
    s.dtMs = dtMs
    s.maxNyquistFraction = min(edge * dtMs / 500, 1.0) # what the resampled pilot holds
    try:
      s.validate
      sweep = some(s)
    except ValueError:
      discard # the samples are fine; the description no longer says what they are
  let steps = p.lengthMs / dtMs
  if steps > float64(high(int32)):
    badPilot("resampling to " & $dtMs & " ms gives over 2³¹ samples")
  let n = int(floor(steps * (1 + 1e-9))) + 1
  var outS = newSeq[T](n)
  let w = float64(halfWidth) * coarse # ms
  let i0b = besselI0(kaiserBeta)
  # Half a kernel reaches `reach` input samples; a whole one has `taps`
  # under it, from −reach.
  let reach = int(floor(w / p.dtMs))
  let taps = 2 * reach + 2
  # Output j lies at input position j·ratio = whole + phase / phases, and
  # its taps depend on the phase alone. With the two intervals in a ratio
  # of small whole numbers (0.5 → 2 ms: one phase; 2 → 0.5 ms: four) the
  # kernels are tabulated once, a Bessel function and a sine per tap, and
  # every output is a dot product (30 s at 0.25 ms to 4 ms: 358 → 5.5 ms).
  let ratio = dtMs / p.dtMs
  var phases = 0
  var num = 0
  for q in 1 .. min(maxPhases, n - 1):
    let a = round(ratio * float64(q))
    if phases == 0 and a >= 1 and a < 1e15 and abs(ratio * float64(q) - a) <= 1e-9 * a:
      phases = q
      num = int(a)
  var kernel = newSeq[float64](max(phases, 1) * taps)
  var whole = newSeq[float64](max(phases, 1)) # each kernel's sum
  let xs = cast[ptr UncheckedArray[T]](addr p.samples[0])
  template fill(ph: int, frac: float64) =
    # The taps of the kernel centred `frac` input samples after a sample,
    # and their sum.
    whole[ph] = 0
    for k in 0 ..< taps:
      let h = kernelTap((float64(k - reach) - frac) * p.dtMs, w, fc, i0b)
      kernel[ph * taps + k] = h
      whole[ph] += h
  for ph in 0 ..< phases: fill(ph, float64(ph) / float64(phases))
  for j in 0 ..< n:
    var base, ph: int
    if phases > 0:
      # Exact in integers: j < 2³¹ and num < 2⁵³ / 64 would overflow only
      # for a ratio beyond any pilot's.
      base = (j * num) div phases
      ph = (j * num) mod phases
    else:
      let pos = float64(j) * ratio
      base = int(floor(pos))
      fill(0, pos - float64(base))
    # The taps with a sample under them: all, away from the pilot's ends.
    let lo = max(0, reach - base)
    let hi = min(taps, p.samples.len - base + reach)
    let h = cast[ptr UncheckedArray[float64]](addr kernel[ph * taps + lo])
    let acc = dot(h, cast[ptr UncheckedArray[T]](addr xs[base - reach + lo]), hi - lo)
    var sum = whole[ph]
    if hi - lo < taps:
      sum = 0
      for k in lo ..< hi: sum += kernel[ph * taps + k]
    # The taps' own sum: unit gain at 0 Hz, also where the pilot's end cuts
    # the kernel (half a kernel sums to half).
    outS[j] = T(acc / (if sum > 0.25 * whole[ph]: sum else: whole[ph]))
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
