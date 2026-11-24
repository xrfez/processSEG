# processSEG API reference

`import processSEG` gives the surface listed here, module by module in the order of
`src/processSEG.nim`. It does not re-export parseSEG: import both (`import parseSEG,
processSEG`) for files, `TraceKeys`, `TraceRows` and the tensor API. This page is the
reference, one line per name or overload group; for task-oriented guides with tested
examples see [`guide/`](guide/README.md), and for the contract with a viewer (who reads the
file, threads, stale buffers, zoom windows) see [`integration.md`](integration.md). The
doc comments (`nim docs`, into `htmldocs/`) hold the details: formulas, edge cases, limits.
`tests/tdocs.nim` checks that every name the umbrella exports appears on this page.

## Conventions

These hold throughout; a row below only mentions what departs from them.

- **Bad arguments raise `ValueError`; data never does.** A constructor (`init*Plan`,
  `compile`, `bandPass`, `lmoMute`, `cdpBinning`, ...) validates its parameters and raises
  `ValueError` naming the problem. A block form raises `ValueError` when the tensor or
  block does not match the `ChunkMeta` (row count, C-contiguity, a width over
  `plan.samples`, another `dtMs`), **before any sample changes**. A buffer of the wrong
  length passed to a single-row kernel is a defect (`AssertionDefect`, or `IndexDefect` with
  bound checks on), not an exception. A workspace reserved too small is one in every build
  mode, `-d:danger` included (`IndexDefect` from `scratch`, or the `AssertionDefect` of a
  plan's own check): no kernel writes past its scratch. Applying a valid plan to any data
  never raises: NaN and Inf stay in their own row.
- **`{.raises: [].}` kernels.** The single-row kernels (`openArray[T]`: `demean`, `agc`,
  `applyFilter`, `shiftSamples`, `timeShift`, `nmoInto`, `plan.pick`, `forward`, ...), the accessors of
  plans and metas, and the per-trace primitives (`lmoTime`, `binOf`, `staticMs`,
  `checkGeometry`, `acc.add`) are declared `{.raises: [].}`, so they can be called from a
  parseSEG `ChunkTransform` or another non-raising context. Three row kernels check an
  argument and raise `ValueError`: `clip` (its limits), `taperPads` (a ramp that does not
  fit) and `percentileAbs` (`p` outside 0 .. 100). The block forms raise only the argument
  `ValueError` above.
- **Argument order.** One rule for every block, single-trace and pool form:
  `verb([pool,] [plan,] src[, dst], meta, operands..., results..., ws, options = ...)`.
  - A compiled plan (`PipelinePlan`, `FilterPlan`, `IirPlan`, `AgcPlan`, `NormalizePlan`,
    `GainCurvePlan`, `FirstBreakPlan`, `CorrelationPlan`, `SlicePlan`, `SpectrumPlan`,
    `FftPlan`) comes first, so `plan.verb(...)` reads as the plan applied:
    `plan.applyFilter(b, meta, ws)`, `plan.pickInto(b, meta, picks, ws)`. A call without
    a plan starts with its data: `b.applyNmo(meta, velocity, params, ws)`.
  - The rows (`RowBlock` or tensor), then a separate output block, then the `ChunkMeta`
    (a single-trace form has no meta: `plan.agc(x, ws)`).
  - Then what the operation needs besides the data (parameter objects, velocities, statics,
    picks, a window length), then the caller's arrays it fills per row (picks, amplitudes,
    SNR, gains).
  - The `Workspace` is the last required argument; defaulted options follow it.
  - Plan constructors: `initXPlan(T, what it does..., dtMs, samples, options)`.
- **NaN means unknown.** Keys are read as parseSEG stores them, NaN where the headers
  cannot give a value (an offset without geometry, an unknown SEG-D `t0Ms`). processSEG
  never substitutes a default: anything derived from an unknown key is NaN (a tick, a
  static, a pick time), and the kernels leave such rows alone. `dtMs` itself is checked to
  be finite and positive.
- **Units.** Time in ms, frequency in Hz, gain in dB, velocity in distance units per
  second, with distance in the file's coordinate units (`summary.coordinateUnits`); no
  feet/metres conversion happens. Amplitudes are as recorded until a calibration converts
  them, per row, to V, m/s, m/s², Pa or N (`core/calibrate`).
- **Time axis.** Sample `j` of row `r` is at `t0Ms[r] + (firstSample + j) · dtMs`
  (`ChunkMeta.sampleTimeMs`), where `firstSample` is the origin of a decoded sample window
  (0 for whole traces).
- **Live rows.** A row is live when its trace's `status` is `tsLive` or `tsUnknown` and it
  is not a failed row (`ChunkMeta.isLive`). Processing touches the live rows only, each up
  to its live length (`numSamples − firstSample`, clamped to the block width:
  `liveSamples`); dead, auxiliary and failed rows and the zero padding past a trace's end
  are left as they are, and are left out of statistics, spectra and stacks.
- **Workspaces, no allocation.** Kernels that need scratch take a `Workspace` and expose
  `workspaceSize`; `ws.reserve(size)` grows it (a no-op when it fits). Once it is
  reserved, applying a plan, a pool job, statistics, spectra, moveout, statics, NMO and
  stacking allocate nothing. What allocates: the constructors (`compile`, `init*Plan`,
  `newProcessPool`, `initVelocityField`, `initStackAccumulator`, the velocity parsers),
  `reserve` when it grows, `bruteStack`, `fitShot` (a little scratch per call) and
  results returned as `seq` (`flaggedRows`, `knots`).
- **Sample types.** Every kernel is generic over parseSEG's `SampleType` (`float32` or
  `float64`). Plans that hold tables in the sample type take it as their first argument
  (`initFilterPlan(float32, ...)`, `flow.compile(float64, ...)`) and apply to that type
  only; `AgcPlan`, `NormalizePlan` and `NmoParams` serve both. Sums run in float64.
- **Forms.** Most operations come as a single-row kernel on `var openArray[T]` (the caller
  passes the live part, `b.row(r, n)`), a `RowBlock[T]` form and a `Tensor[T]` form, both
  with a `ChunkMeta`. The block and tensor forms work in place. A `PipelinePlan` also
  applies `src → dst` (the raw buffer kept, the copy fused into the first op). Procs ending
  in `Into` fill a caller's buffer of the stated length and allocate nothing.
- **Threads.** Plans, metas and blocks are immutable or raw views with no
  reference-counted parts: any number of threads may share one plan, each with its own
  `Workspace` and disjoint rows (`RowBlock.sub`, `ChunkMeta.sub`). Nothing touches a
  global except the plan `generation` counter.

## core/types: `ChunkMeta`

The metadata of a block of rows: parseSEG's `TraceRows` view plus `dtMs`, `firstSample`,
the failed rows and, optionally, a caller's per-row `FileScale` column. It copies no header
value.

| Proc | What |
|---|---|
| `ChunkMeta`; `initChunkMeta(rows: TraceRows, dtMs, firstSample = 0, failed: openArray[int] = [], scales: openArray[FileScale] = []): ChunkMeta` | Builds and validates a meta once. `ValueError` when the view has no keys (`hasKeys`), `dtMs` is not finite and positive, `firstSample < 0`, `failed` is not ascending within `0 ..< rows.len`, or `scales` is neither empty nor one per row. `failed` and `scales` are not copied: they must outlive the meta. |
| `m.hasFileScales`, `m.fileScale(row): lent FileScale` | Whether the meta points at a `FileScale` column (`fileScalesInto`), and a row's entry in place, for a calibration. |
| `m.rows`, `m.len`, `m.dtMs`, `m.firstSample`, `m.failedCount`, `m.failedRow(i)` | The fields, read-only. |
| `m.keys(row): lent TraceKeys` | Row `row`'s keys in place (parseSEG's `keysAt`). |
| `m.isFailed(row)`, `m.isLive(row)` | A failed row (O(log n)); a live row (status `tsLive`/`tsUnknown`, not failed). |
| `m.liveSamples(row, cols)` | The live length of the row in a block `cols` wide. |
| `m.t0Ms(row)`, `m.sampleTimeMs(row, col)` | The time of the trace's sample 0, and of a column of the block (NaN when `t0Ms` is). |
| `sameInterval(dtMs, planDtMs): bool` | A block's interval fits a plan's: equal within a relative 1e-9 (a header's 2000 µs / 1000 and a computed 1000 / 500 Hz can differ in the last bit). Every block form checks its meta's `dtMs` with it. |
| `samplesInLength(lengthMs, dtMs): int`, `lengthOfSamples(samples, dtMs): float64` | A time length is N × dt: 12000 ms at 1 ms is 12000 samples (the last at 11999 ms). `samplesInLength` cuts a length that is not a whole multiple of `dtMs` down to whole samples, with a relative 1e-6 tolerance for float rounding; 0 for a length or interval that is not finite and positive, or a count over 2³¹ − 1. |
| `m.sub(rows: Slice[int]): ChunkMeta` | The meta of some rows (row 0 is `rows.a`), sharing the view, failed list and file scales: split work with `RowBlock.sub`. |
| `checkShape(m, samples: Tensor[T])`, `checkShape(m, b: RowBlock[T])` | `ValueError` unless one row per row of `m` (and, for a tensor, rank 2 and C-contiguous). |
| `SampleType` | parseSEG's `float32 \| float64`, re-exported. |
| `StopCheck`; `Stopped` | `StopCheck` is `proc (): bool {.gcsafe, raises: [].}`. True: abandon the job. Nil means never stop. Passed last, default nil, to a long kernel that can be abandoned between units of work. `Stopped` (`-1`) is the result of a call its stop check abandoned; the output is then unspecified. |

## core/rows: `RowBlock`

| Proc | What |
|---|---|
| `RowBlock[T]`; `rowBlock(samples: Tensor[T])`, `rowBlock(data: openArray[T], rows, cols)` | A raw, zero-copy view of `[rows, cols]` row-major storage. `ValueError` for a tensor that is not rank 2 and C-contiguous, or a size mismatch. Never make one from a temporary. |
| `b.len`, `b.cols`, `b.rowPtr(row)` | Rows, samples per row, a row's storage. |
| `b.row(r)`, `b.row(r, n)` (templates) | Row `r`, or its first `n` samples, as an `openArray[T]` in place (pass to a `var openArray[T]` kernel). |
| `b.sub(rows: Slice[int]): RowBlock[T]` | Some rows as a block of their own, for threads. |

## core/workspace: `Workspace`

| Proc | What |
|---|---|
| `WorkspaceSize` (`f32`, `f64`, `u32`: element counts); `a + b`, `max(a, b)` | The scratch a plan needs; `+` for buffers used together, `max` for steps run one after another. |
| `Workspace`; `initWorkspace()` | Grow-only per-thread scratch; cannot be copied (move it into its thread). |
| `ws.reserve(size)`, `ws.reserve(plan)`, `ws.capacity`, `ws.fits(size)` | Grow to hold `size`, or what any plan with a `workspaceSize` needs (a no-op when it does; contents not kept; `ValueError` for a negative size); what it holds; whether `reserve` would allocate. |
| `ws.scratch(float32 \| float64 \| uint32, n, at = 0): ptr UncheckedArray[T]` | `n` elements from element `at`, for your own kernels; valid until the next `reserve`. |

## core/basic: demean, detrend, scale, clip, polarity, pad taper

Each op has a row kernel `op(x: var openArray[T], ...)` and block forms
`op(b: RowBlock[T], meta, ...)` and `op(samples: var Tensor[T], meta, ...)` on the live rows.

| Proc | What |
|---|---|
| `demean(x)` | Subtract the mean. |
| `detrend(x)` | Subtract the least-squares line (also removes the mean). |
| `scale(x, factor)` | Multiply by `factor` (block forms: `ValueError` unless finite). |
| `clip(x, lo, hi)` | Limit to `lo .. hi`; NaN stays NaN. `ValueError` for `lo > hi` or a NaN limit. |
| `reversePolarity(x)` | Negate. |
| `cosTaperInto(ramp)`, `taperPads(x, ramp)` | A rising cos² ramp (never exactly 0 or 1), and the taper of the pads at both ends of a padded trace, never its data (`ValueError` when `2·ramp.len > x.len`). Building blocks for your own padded kernels. |

## core/mute: top and bottom mutes

| Proc | What |
|---|---|
| `MuteKind` (`mkTime`, `mkTable`, `mkLmo`); `MuteCurve`, `c.kind` | The mute time per trace, by one of three shapes. |
| `timeMute(tMs)`, `tableMute(offsets, timesMs)`, `lmoMute(velocity, t0Ms = 0, padMs = 0)` | One time; linear in `\|offset\|` between knots, flat beyond; `t0Ms + \|offset\|/velocity + padMs`. Validated (`ValueError`). |
| `c.muteTimeMs(offset)` | The curve's time at an offset (sign ignored; NaN for a NaN offset where it matters). |
| `muteTop(x, startCol, taperCols)`, `muteBottom(x, endCol, taperCols)` | Row kernels in fractional columns: zero before `startCol` (after `endCol`) with a sin² taper on the muted side. |
| `muteTop(b \| samples, meta, curve, taperMs = 0)`, `muteBottom(...)` | Each live row muted at its curve time, on its own time axis. |

## core/units: amplitude units, sensitivities, descales

Values are entered with their units and held in SI. Enter a sensor's sensitivity as it holds
wired to the recorder (the damped figure for a shunted geophone); it holds in the flat band
above the sensor's natural frequency only. Guide: [calibration](guide/calibration.md).

| Proc | What |
|---|---|
| `AmplitudeUnit` (`auRecorded`, `auVolt`, `auVelocity`, `auAcceleration`, `auPressure`, `auForce`) | A row's unit: as recorded, V at the recorder's input, m/s, m/s², Pa, N (a vibrator's ground force). `$` gives the symbol. |
| `SensitivityUnit` (`suVPerMPerS`, `suVPerInPerS`, `suVPerMPerS2`, `suVPerG`, `suVPerBar`, `suVPerPa`, `suMicroVPerPa`, `suDbReVPerMicroPa`, `suVPerN`, `suVPerKN`, `suVPerLbf`); `quantityOf(unit)` | The units a sensitivity is entered in (`$`: `V/(m/s)`, `dB re 1 V/µPa`, `V/kN`, ...), and what each measures (1 lbf = 4.4482216152605 N). |
| `Sensitivity`; `sensitivity(value, unit)` | A sensor's sensitivity **as wired to the recorder** (a geophone's damped figure), used as it is. `ValueError` unless finite and positive (finite in dB). |
| `openCircuitSensitivity(value, unit, coilOhms, shuntOhms, inputOhms = Inf, inSeries = 1, inParallel = 1)` | The as-wired value from the open-circuit one: `n·S₀ · R_L / (n·R_c/m + R_L)`, `R_L = shunt ∥ input` (`Inf`: none). `ValueError` for bad resistances. |
| `s.isSet`, `s.form` (`SensitivityForm`: `snNone`, `snAsWired`, `snOpenCircuit`), `s.unit`, `s.voltsPerUnit`, `s.valueIn(unit)`, `$s` | Read back: the effective SI value (NaN when not set), in any unit of the same quantity (`ValueError` otherwise), and as text with how it was entered. |
| `DescaleUnit` (`duVPerCount`, `duMilliVPerCount`, `duMicroVPerCount`, `duNanoVPerCount`); `Descale`; `descale(value, unit)` | A recorder's volts per count. `ValueError` unless finite and positive. |
| `d.isSet`, `d.voltsPerCount`, `d.valueIn(unit)`, `$d` | Read back (NaN when not set). |
| `FileScale` (`descale`, `sensitivity`, `unit`, `physical`); `initFileScale()`; `$f` | What a file's headers say about one trace's scale, in SI (NaN where they say nothing), as `fileScalesInto` reads them. |
| `MassUnit` (`muKilogram`, `muPound`, `muTonne`); `Mass`; `mass(value, unit)` | A mass entered with its unit (a vibrator's reaction mass or baseplate; 1 lb = 0.45359237 kg). `ValueError` unless finite and positive. |
| `m.isSet`, `m.kilograms`, `m.valueIn(unit)`, `$m` | Read back (NaN when not set; `valueIn` raises `ValueError` then). |

## core/calibrate: recorded values to volts or physical units

Per row, `factor = descale / (10^(gain/20) · sensitivity)` (`ctVolts`: without the
sensitivity). Each factor comes from the most specific matching rule (station > channel >
channel set > component > sensor type > the catch-all rule) or from the file's `FileScale`
column, in the order `fileFactors` sets; disagreements are flagged, never silent. Live rows
without the factors their target needs, or whose factor is out of range (it or its inverse
beyond a float32), become NaN; dead rows are converted when they can be and kept otherwise;
auxiliary rows go to volts (`scaleAuxiliary`); failed rows are not touched. Guide:
[calibration](guide/calibration.md).

| Proc | What |
|---|---|
| `CalibrationTarget` (`ctPhysical`, `ctVolts`), `FileFactors` (`ffFallback`, `ffFirst`, `ffIgnore`) | What to convert to; how the file's factors rank against the rules (default: specific rules, then the file, then the catch-all). |
| `RuleMatch` (`sensorType`, `component`, `channelSet`, `channel`, `station`: unset by default), `$m` | Which rows a rule applies to: SEG-D sensor type, component, 0-based channel set, channel, receiver station (`some stationKey(line, point, index)`, index 0: any). |
| `CalibrationRule` (`match`, `descale`, `sensitivity`, `gainDb`), `$r` | Any of the three factors for the matched rows; `gainDb` (NaN: none) is a recorder gain the descale does not hold, divided out. |
| `Calibration` (`rules`, `target`, `fileFactors`, `scaleAuxiliary`); `initCalibration(rules = [], target = ctPhysical, fileFactors = ffFallback, scaleAuxiliary = true)`, `$c` | Plain data: the rules and how to apply them. |
| `CalibrationPlan`; `initCalibrationPlan(cal)`, `plan.target` | Validated and indexed. `ValueError` (naming the rule) for a rule without a factor, a gain that is not finite or whose linear factor is not, a criterion out of range, two rules with the same criteria and values, or two catch-alls. |
| `FactorSource` (`fsNone`, `fsRule`, `fsFile`), `CalibrationStatus` (`csConverted`, `csUnresolved`, `csKept`), `RowCalibration` (`factor`, `unit`, `status`, `descaleFrom`, `sensitivityFrom`, `descaleRule`, `sensitivityRule`, `gainRule`, `conflict`, `mismatch`) | How one row is calibrated: its factor (1 when kept, NaN when unresolved), unit, where each factor came from, and the flags. |
| `CalibrationSummary` (`units`, `converted`, `unresolved`, `kept`, `noDescale`, `noSensitivity`, `conflicts`, `mismatches`), `$s` | Counts over the rows of a call. |
| `plan.resolveInto(meta, rows: var openArray[RowCalibration]): CalibrationSummary` | Per row, without touching a sample: for a viewer's labels, warnings, or a GPU uniform. |
| `plan.calibrate(b \| samples, meta)`, `plan.calibrateInto(src, dst, meta)` (blocks or tensors) | Apply in place, or `src → dst` (raw kept; kept rows and padding copied). Return the summary (discardable). |
| `plan.uncalibrate(b \| samples, meta)`, `plan.uncalibrateInto(src, dst, meta)` | The inverse: the plan's units back to recorded values, for writing counts. |
| `scaleRows(b \| samples, meta, factors)`, `scaleRowsInto(src, dst, meta, factors)` | The plain per-row kernel: every row that is not failed times its factor, up to its live length (NaN fills; `ValueError` for an infinite factor or a wrong count). |
| `scaleInto(src, dst, factor)` | Row kernel: `dst[i] = src[i] · factor`. |
| `RatioRange` (`rows`, `minDb`, `maxDb`); `ratioDbInto(a, b, meta, dst): RatioRange` | Per row, how much larger data comes out under plan `a` than under `b`, in dB (NaN where either does not convert, or to another unit): compare two candidate tables. |

## fft/plan: `FftPlan`

| Proc | What |
|---|---|
| `fftLanes` (8) | Most traces a batch runs. `lanesFor` never returns more. Workspaces stay sized for this. |
| `lanesFor(nfft, T)`, `lanesFor(plan)` | Width of a batched transform: float32 8 below 262144, else 4; float64 8 below 65536, 4 below 262144, else 2. Where the width differs from 8, an optimised build can differ in the last place from the previous release. |
| `isFastLen(n)`, `nextFastLen(n, even = true)` | 2·3·5-smooth lengths; the next one up (even by default: about twice as fast). |
| `FftPlan[T]`; `initFftPlan(T, n)`, `p.len`, `p.bins`, `p.workspaceSize(batched = true)` | A real FFT of length `n` (`ValueError` unless smooth); `n div 2 + 1` bins. |
| `p.forward(x, re, im, ws)`, `p.inverse(re, im, x, ws)` | One trace: unscaled forward, inverse scaled by 1/n. |
| `p.forwardLanes(x, re, im, ws)`, `p.inverseLanes(re, im, x, ws)` | `lanesFor(p)` interleaved traces at once (SIMD across traces). |
| `p.forwardLanes(lanes, x, re, im, ws)`, `p.inverseLanes(lanes, re, im, x, ws)` | An explicit width 1, 2, 4 or 8 (not above `fftLanes`). |
| `gatherLanes(b, first, x, lanes = fftLanes)`, `scatterLanes(x, b, first, lanes = fftLanes)` | Rows of a block into and out of the interleaved lane layout. |

## fft/spectrum: `SpectrumPlan`

| Proc | What |
|---|---|
| `SpectrumPlan[T]`; `initSpectrumPlan(T, dtMs, windowLen, taperFraction = 0.1, nfft = 0, demean = true)` | Windowed amplitude spectra: demeaned, Tukey-tapered, zero-padded to `nfft` (0: the next power of two). |
| `sp.bins`, `sp.windowLen`, `sp.nfft`, `sp.binHz(k)`, `sp.sides(k)`, `sp.workspaceSize` | Its axes, the one-sided factor of bin `k` (1 at 0 Hz and at Nyquist, else 2), and scratch. |
| `sp.averageAmplitudeInto(b \| samples, meta, firstCol, amp: var openArray[float64], ws, rows = 0 .. high(int)): int` | The power-averaged amplitude spectrum of the window `firstCol ..< firstCol + windowLen` over the live rows in `rows`; returns the rows used (rows with NaN or Inf are left out). A sinusoid of amplitude A on a bin reads A. |
| `sp.addPowerRows(b, meta, firstCol, row0, row1, acc, ws): int` | Adds the scaled power of rows `row0 .. row1` into `acc` (`bins` values), the quantity the average sums. Returns how many were added. A row that is not live, that has no live sample past `firstCol`, or whose window holds a NaN or an infinity is left out. |
| `sp.amplitudeInto(x, amp, phase, ws)` | One window of one trace: amplitude and phase (radians). |
| `amplitudePhaseInto(re, im, amp, phase)` | Modulus and phase of a raw `forward` spectrum, unscaled. |
| `DbReference` (`dbPeak`, `dbAbsolute`); `toDbInto(amp, db, reference = dbPeak, floorDb = -120, refAmplitude = 1)` | 20·log10 of the amplitudes, 0 dB at the peak or at `refAmplitude` (`dbAbsolute`; 1 by default, so present callers are unchanged), clamped at `floorDb`. `dbPeak` ignores `refAmplitude`. |
| `unwrapPhaseInto(phaseDeg, shown, dst)` | Unwrap along shown finite bins. A gap is not crossed: the next drawn bin restarts from the last drawn value, and a step over 180° is ±360. NaN where not drawn. Same length or `AssertionDefect`. |
| `circularMeanDeg(angles): tuple[meanDeg, resultant]` | Circular mean of angles in degrees, in (−180, 180], and the mean resultant length 0..1. NaN and infinite angles are left out; none left gives `(NaN, 0)`; cancelling angles (opposite) give `(NaN, ~0)`. |
| `removeDelayInto(unwrapped, hz, shown, dst): float64` | Least squares `p = a + b·f` on drawn finite bins; `dst = p − (a + b·f)`, NaN elsewhere. Returns the delay in ms, `−b·1000/360`; NaN if fewer than 2 bins or the denominator is below `1e-12·max(1, Σf²)`. |
| `SpectrumAccumulator[T]`; `initSpectrumAccumulator(T, dtMs, windowLen, nfft = 0, taperFraction = 0.1, demean = true)` | A power sum of `averageAmplitudeInto` over several calls. Same window as `initSpectrumPlan`. |
| `acc.bins`, `acc.windowLen`, `acc.nfft`, `acc.dtMs`, `acc.binHz(k)`, `acc.used`, `acc.workspaceSize`, `acc.clear` | Its axes, the rows added, its scratch, and a reset of the sum. |
| `acc.add(b \| samples, meta, firstCol, ws, rows = 0 .. high(int)): int` | Adds `amp²·rows` of that window. The same `ValueError`s as `averageAmplitudeInto`. Returns the rows added. Nothing allocates after init. |
| `acc.amplitudesInto(dst)` | `sqrt(power/used)`, 0 if none. A sinusoid of amplitude A on a bin reads A. |

## fft/hilbert: `HilbertPlan`

| Proc | What |
|---|---|
| `HilbertPlan[T]`; `initHilbertPlan(T, samples, nfft = 0)`, `hp.samples`, `hp.nfft`, `hp.workspaceSize` | The Hilbert transform of traces of up to `samples` samples, zero padded to `nfft` (0: `nextFastLen(2·samples)`, so nothing wraps). `ValueError` for `samples < 1` or an `nfft` below `2·samples` or not smooth. |
| `hp.hilbertInto(x, h, ws)` | `H[x]`, the imaginary part of the analytic signal (`H[cos] = sin`; 0 Hz and Nyquist to zero), `x.len` values. |
| `hp.envelopeInto(x, env, ws)` | The envelope `\|x + i·H[x]\|`. |
| `hp.analyticInto(x, dtMs, env, phaseDeg, hz, ws, floorDb = -60)` | Envelope, phase in (−180, 180] and instantaneous Hz from one Hilbert. Phase is NaN where the envelope is 0. Hz is the circular mean of the two one-sample phase steps around the sample (each the argument of `z[n]·conj(z[n-1])`, in (−π, π]), so it is exact from 0 up to Nyquist; it belongs to that sample, is positive for a rising `cos` frequency, and is NaN at the first and last samples and where that sample or a neighbour is 0 or below `peak·10^(floorDb/20)` (a zero trace is all NaN). Outputs are float64, length `x.len`. `dtMs` is required (a plan has no interval). |

## fft/spectrogram: `SpectrogramPlan`

| Proc | What |
|---|---|
| `SpectrogramPlan[T]`; `initSpectrogramPlan(T, windowLen, hop, dtMs, samples, nfft = 0)` | The short-time Fourier amplitudes of traces of up to `samples` samples: Hann frames of `windowLen` samples centred on columns `0, hop, 2·hop, ...`, zero padded to `nfft` (0: the next power of two of the window). `ValueError` for `windowLen < 2`, `hop < 1`, `samples < 1`, a bad `dtMs` or `nfft`. |
| `sg.frames`, `sg.bins`, `sg.windowLen`, `sg.hop`, `sg.nfft`, `sg.samples`, `sg.dtMs`, `sg.frameCol(frame)`, `sg.binHz(k)`, `sg.workspaceSize` | Its axes: frame `i` centred on column `i·hop`, bin `k` at `k / (nfft·dt)`. |
| `sg.spectrogramInto(x, amp, ws)` | Every frame's amplitudes into `amp` (`frames·bins`, frame after frame), scaled so that a sinusoid of amplitude A on a bin reads A (a frame cut by the trace's ends by its own Σw). |
| `sg.averageWorkspace`; `sg.averageSpectrogramInto(b \| samples, meta, amp, ws, rows = 0 .. high(int), stop: StopCheck = nil): int` | Power average of `spectrogramInto` over the live rows, same scale. A float64 row with a NaN, an infinity, a subnormal, or a nonzero sample below 1e-100 or above 1e140 is left out before the transform; a frame with a NaN or infinity drops that trace. Returns the rows used, or `Stopped` when `stop` said to abandon (`amp` then unspecified). Checked once per live row; nil never stops. The plan and the workspace stay usable. `openArray[openArray]` is not a Nim parameter, so this takes a `RowBlock` and a `ChunkMeta`, as every other block form does. |
| `averagedRow(x, n): bool`; `amplitudesFinite(x, n): bool` | The two row rules of the averaged spectrogram, exported for the pool form: `averagedRow` is false for a float64 row whose `n` input samples leave the safe range (a NaN, an infinity, a subnormal, or a nonzero sample below 1e-100 or above 1e140); `amplitudesFinite` is false when the row's `n` amplitudes hold a NaN or an infinity. Both take a pointer to the first value |

## fft/fk: `FkPlan`

| Proc | What |
|---|---|
| `FkAxes` (`nFreq`, `nK`, `dfHz`, `dkPerTrace`); `FkPlan[T]`; `initFkPlan(T, dtMs, traces, samples, taperFraction = 0.1)` | f-k of `traces` live rows by `samples` columns. Tukey both axes. Time padded to `nextFastLen(samples)`, traces to `nextFastLen(2·traces)`. `ValueError` for fewer than 4 traces, fewer than 8 samples, a bad `dtMs` or taper. |
| `plan.axes`, `plan.nFreq`, `plan.nK`, `plan.dfHz`, `plan.dkPerTrace`, `plan.traces`, `plan.samples`, `plan.dtMs`, `plan.workspaceSize` | Axes (zero k at `nK div 2`, positive k is later on later traces) and scratch (piece buffers, the shared spectrum `2·traces·nFreq`, and the live-row indices; the float32 f64 region grows by that list, and the total can also be smaller than before). `dkPerTrace` is cycles per trace; cycles/km stay in the viewer. |
| `fkRows(meta, cols, firstCol, nCols, rows = 0 .. high(int)): int` | Live rows that enter the window: the `traces` to build the plan for. |
| `plan.fkAmplitudeInto(b \| samples, meta, firstCol, amp, ws, rows = 0 .. high(int), stop: StopCheck = nil): int` | Unscaled modulus, `nK·nFreq`, frequency fastest. Returns the live rows used, or `Stopped` when `stop` said to abandon (`amp` then unspecified). Checked once per trace that enters and once per frequency of the spatial pass; nil never stops. The plan and the workspace stay usable. `ValueError` when the live count is not the plan's. A wrong `amp` length is an `AssertionDefect` (every other Into kernel), not the `ValueError` the audit suggested. The pool form is `fkAmplitudeInto(pool, plan, ...)` (no workspace, no stop). |
| `fkPoolScratch(plan)`; `fkLeavesRange(b, meta, firstCol, nSamples, rows)`; `fkTimeRun(...)`; `fkFreqRun(...)` | The pieces both `fkAmplitudeInto` forms call. Not a second way to run the transform. |

## filter/design: `FilterSpec`

| Proc | What |
|---|---|
| `Corners` (`f1 < f2 ≤ f3 < f4`, Hz); `bandPass(c)`, `lowCut(f1, f2)`, `highCut(f3, f4)` | cos² trapezoid responses (`f2 = 0`: no low cut; `f3 = Inf`: no high cut). `ValueError` for corners out of order. |
| `butterBandPass(lowHz, highHz, slopeDbPerOctave = 24, highSlopeDbPerOctave = NaN)`, `butterLowCut(hz, slope = 24)`, `butterHighCut(hz, slope = 24)` | Butterworth magnitudes, −6 dB corners, effective slopes in dB/octave (for `FilterPlan` or, as biquads, `IirPlan`). |
| `Notch` (`freqHz`, `widthHz`); `spec.withNotch(freqHz, widthHz)`, `spec.withZeroDc` | Add a notch; set bin 0 to 0. |
| `FilterSpec` (`kind`, `corners` or `lowHz`/`lowSlope`/`highHz`/`highSlope`, `notches`, `zeroDc`); `ResponseKind` (`rkTrapezoid`, `rkButterworth`) | The zero-phase response, plain data. |
| `validate(spec, dtMs)`, `nyquistHz(dtMs)` | Check a (hand-built) spec against Nyquist (`ValueError`). |
| `response(spec, f, dtMs)`, `responseInto(spec, dtMs, nfft, h)` | The gain at `f` Hz; per FFT bin. |
| `impulseLength(spec, dtMs)`, `energyFraction` (0.999), `narrowestRampHz(spec)` | Samples each side of the impulse response holding 99.9 % of its energy (the zoom margin `L`); the narrowest feature. |

## filter/apply: `FilterPlan` (zero-phase FFT)

| Proc | What |
|---|---|
| `TrendRemoval` (`trNone`, `trMean`, `trLinear`) | What is removed before padding and added back after, scaled by the gain at 0 Hz. |
| `FilterPlan[T]`; `initFilterPlan(T, spec, dtMs, samples, trend = trLinear)` | A filter for traces up to `samples` long: zero pads (a response that rejects 0 Hz, with `trLinear`) or odd-reflection pads of `L`, a pad-only taper, an FFT of at least `samples + 2L` (zeros) or `samples + 4L` (reflections). Zero pads are cut to `samples` when `L` is longer than the trace (a 0.02 Hz low-cut: nfft 6144 for 2001 samples, not 50 000; the output within 0.4 % of the uncut one). |
| `plan.requiredMargin`, `plan.samples`, `plan.nfft`, `plan.spec`, `plan.dtMs`, `plan.binHz`, `plan.gainAt(k)`, `plan.workspaceSize` | Its margin `L`, size and response. |
| `applyFilter(plan, x \| b, meta \| samples, meta, ws)` | Filter one trace, or the live rows in place (`lanesFor` at a time). |

## filter/iir: `IirPlan` (Butterworth biquads)

| Proc | What |
|---|---|
| `IirPhase` (`ipZero`: forward-backward, \|H\|²; `ipMinimum`: one causal pass, \|H\|) | The phase. |
| `IirPlan[T]`; `initIirPlan(T, spec, dtMs, samples, phase = ipZero, trend = trLinear)` | A cascade for a `butter*` spec without notches; slopes in multiples of 12 dB/octave (zero phase) or 6 (minimum phase). |
| `plan.requiredMargin`, `plan.samples`, `plan.phase`, `plan.order`, `plan.spec`, `plan.magnitudeAt(f)`, `plan.workspaceSize` | Its margin (before the window for minimum phase), size and response. |
| `causalEnergyFraction` (0.9999) | The energy a minimum-phase margin holds. |
| `applyFilter(plan, x \| b, meta \| samples, meta, ws)` | As the FFT plan; the block form filters `iirLanes` live rows at a time. |
| `iirLanes` (24) | Traces per batch of the block form. |

## gain/gain: fixed gain, normalization, AGC

| Proc | What |
|---|---|
| `dbToFactor(dB)`; `scaleDb(x \| b, meta \| samples, meta, dB)` | 10^(dB/20); multiply by it (`ValueError` unless finite). |
| `NormStat` (`normMax`, `normRms`, `normPercentile`), `NormScope` (`perTrace`, `perPanel`) | What normalization brings to its target, per trace or per block. |
| `NormalizePlan`; `initNormalizePlan(stat, samples, scope = perTrace, target = 1, percentile = 99)` | A normalization; `p.stat`, `p.scope`, `p.target`, `p.percentile`, `p.samples`, `p.workspaceSize`. |
| `normalize(plan, x \| b, meta \| samples, meta, ws)` | Normalize in place; an all-zero or non-finite trace (panel) is left alone. |
| `plan.panelFactor(b, meta, ws): float64` | The one factor a `perPanel` normalization would apply (NaN: none), for a display-gain uniform. |
| `AgcMode` (`agcRms`, `agcMeanAbs`); `AgcPlan`; `initAgcPlan(windowMs, dtMs, samples, mode = agcRms, target = 1, epsilon = 1e-9)` | AGC over a centred window of `2h + 1` samples. |
| `p.requiredMargin` (h), `p.windowSamples`, `p.windowMs`, `p.mode`, `p.samples`, `p.target`, `p.epsilon`, `p.dtMs`, `p.workspaceSize` | Its window and size. |
| `agc(plan, x \| b, meta \| samples, meta[, gain \| gains], ws)` | AGC in place; the forms with `gain`/`gains` keep each sample's gain (1 where untouched). |
| `removeGain(x, gain)`, `removeGain(b \| samples, meta, gains)` | Divide by a kept gain: undo AGC. |

## gain/curves: time-variant gain

| Proc | What |
|---|---|
| `GainKnot` (`tMs`, `dB`), `GainCurveKind` (`gcKnots`, `gcPower`, `gcExponential`), `GainCurve`, `c.kind` | A gain against time. |
| `knotGain(knots)`, `powerGain(power, refMs = 1000, minMs = 1)`, `exponentialGain(dbPerS)` | Linear in dB between knots; `(t/refMs)^power` (2: spherical divergence); dB per second. Validated. |
| `c.gainDbAt(tMs)`, `c.factorAt(tMs)` | The gain at a time, in dB and as a factor. |
| `heldGain(curve, untilMs, step = 1.02): GainCurve` | The curve on `0 .. untilMs`, flat after. Before 0 it holds the gain at 0 only when the first knot is at or after 0 (an exponential always; a knot curve is flat before its first knot). Exponential: two knots. Power: knots from `minMs` every `step` in time (1.02 is within 0.001 dB of `powerGain(2)`). Knots: the knots before `untilMs`, plus the gain there. `ValueError` if `untilMs` is not finite and > 0, `step` is not finite and > 1, the curve is unbuilt, or a power curve would take more than 100_000 knots. |
| `GainCurvePlan[T]`; `initGainCurvePlan(T, curve, dtMs, samples, tMinMs = 0, tMaxMs = NaN)` | The curve tabulated over `tMinMs .. tMaxMs` (NaN: the record length); times outside the table are exact but slower. `p.samples`, `p.dtMs`, `p.curve`, `p.tableLen`, `p.tableMs`. |
| `applyGainCurve(plan, x, startMs)`, `applyGainCurve(plan, b \| samples, meta)` | Multiply each live row by the curve at its own times (`t0Ms`, `firstSample`). |
| `applyGainCurve(plan, b \| samples, meta, ws)`, `p.workspaceSize` | The same for blocks that may lie off the table (a zoom window, a recording delay): the factors off it are computed once per run of rows with one time axis, not per sample (the pipeline uses this form). Results off the table differ from the other form by a rounding. |

## gain/stats: statistics for display gain

| Proc | What |
|---|---|
| `TraceStats` (`rms`, `maxAbs`); `traceStats(x)` | RMS and max \|x\| of one trace's samples. |
| `traceStatsInto(b \| samples, meta, stats)` | One per row; NaN for dead, auxiliary and failed rows. |
| `PanelStats` (`rms`, `maxAbs`, `clip`, `rows`, `samples`); `panelStats(b \| samples, meta, ws, clipPercentile = 99, rows = 0 .. high(int))` | Over the live rows in `rows`, leaving out rows with NaN or Inf; `clip` is a percentile of \|x\| from a histogram (0.4 % relative resolution; NaN `clipPercentile` skips it). |
| `panelStatsWorkspace` | The scratch `panelStats` needs (four 256 KB histograms). |
| `percentileAbs(x, p, ws)`, `percentileWorkspace(samples)` | The exact percentile of \|x\| of one trace, and its scratch. |

## core/shift: fractional shifts

| Proc | What |
|---|---|
| `ShiftInterp` (`siLinear`, `siCubic`: 4-point Catmull-Rom) | Interpolation of fractional shifts (statics, LMO, NMO). |
| `shiftSamples(x, shift, interp = siLinear)` | `x[j]` becomes the old `x[j + shift]`, zero outside; a NaN shift leaves `x` alone. |
| `placedWhole(stored, wholeSamples)` | `stored` placed `wholeSamples` later: that many zeros before it; a negative number drops that many first samples. A new trace, longer or shorter (empty when the drop covers `stored`). A positive amount is later, not earlier as in `shiftSamples`. A positive amount past the range of a sample count raises `ValueError`. |
| `placedSinc(stored, shiftMs, dtMs)` | `stored` moved later by `shiftMs`, a fraction of a sample included: Kaiser-windowed sinc, 24 taps each side, beta 8, zeros outside `stored` (nothing wraps). Longer by the shift, shorter when negative. A non-positive `dtMs`, an empty `stored`, a NaN or infinite `shiftMs`, or a shift whose quotient by `dtMs` is beyond the range of a sample count returns a copy of `stored`. A fraction below 1e-9 above a whole sample is `placedWhole` (so it raises `ValueError` only for a whole number past the range of a count). 100.3 ms at 1 ms is not snapped to 100. |

## core/timeshift: whole-sample shift

| Proc | What |
|---|---|
| `timeShiftInto(src, dst, shift)`, `timeShift(x, shift)` | `dst[j]` is the old sample at `j + shift` (positive: earlier, as `shiftSamples`). `shift` is a whole number of samples. Length kept. What leaves one end is clipped; the vacated end is zeros. NaN kept. `|shift| >= len` is all zeros. Empty and one-sample traces are fine. Nothing allocates. |
| `wholeSamples(ms, dtMs)` | Whole samples in `ms` at interval `dtMs`: `ms / dtMs` rounded to nearest, ties away from zero (`1.5` is `2`, `-1.5` is `-2`). A non-finite `ms`, or a `dtMs` that is not a normal positive number (zero, negative, subnormal, NaN or infinite), is 0. Callers snap a millisecond shift with this. |

## velocity/model: velocity functions and fields

| Proc | What |
|---|---|
| `VelocityKind` (`vkConstant`, `vkGradient`, `vkKnots`), `VelocityFunction`, `f.kind`, `a == b` | An RMS velocity against two-way time. |
| `constantVelocity(v)`, `gradientVelocity(v0, k)`, `rmsVelocity(knots)`, `intervalVelocity(tMs, vint)` | A constant; an interval velocity `v0 + k·t`; RMS knots interpolated linearly in V²·t; layers (built as RMS knots). Validated. |
| `VelocityKnot` (`tMs`, `vrms`); `f.knots` | A knot; a copy of a knots function's knots. |
| `f.vrmsAt(tMs)`, `f.vintAt(tMs)`, `f.vrmsInto(startMs, dtMs, vrms)` | RMS and interval velocity at a time; a column of RMS velocities. |
| `dixIntervalInto(tMs, vrms, vint)`, `rmsFromIntervalInto(tMs, vint, vrms)` | Dix conversion and its inverse (NaN where Vint² < 0). |
| `FieldKind` (`fk2D`, `fk3D`), `VelocityField`; `initVelocityField(locations, functions, dtMs, tMaxMs)`, `initVelocityField(inlines, crosslines, functions, dtMs, tMaxMs)`, `initVelocityField(picks, dtMs, tMaxMs)` | Functions at nodes (CDP numbers or distances in 2D, an inline × crossline grid in 3D) resampled onto one time grid; from picks, 2D or 3D is detected. Allocates nodes × samples × 8 bytes (50 000 nodes of 1501 times: 600 MB, filled in 0.2 s). |
| `field.kind`, `field.len`, `field.dtMs`, `field.samples`, `field.tMaxMs`, `field.locations`, `field.crosslines` | Its nodes and grid (`locations`: 2D nodes, or the 3D inline axis; `tMaxMs`: the last grid time, after which the velocities stay constant). |
| `field.vrmsAt(location, tMs)`, `field.vrmsAt(inline, crossline, tMs)`, `field.vrmsInto(location \| inline, crossline, startMs, dtMs, vrms)` | Interpolated RMS velocity (NaN for the other dimensionality or a NaN location). |
| `VelocityPick` (`cdp`, `inline`, `crossline`, `velocity`) | A function at one place, as a velocity file holds it. |

## velocity/formats: VELF, HANDVEL, CSV

| Proc | What |
|---|---|
| `parseVelf(text)`, `parseHandvel(text)`, `parseVelocityCsv(text)` | Picks from Western VELF cards, ProMAX HANDVEL blocks, or `cdp,tMs,vrms` / `inline,crossline,tMs,vrms` CSV. `ValueError` naming the line: a bad number, times that do not increase, a NaN or infinite location, a velocity below `MinPickVelocity`. |
| `MinPickVelocity` (100) | The least velocity a pick file may hold, in distance units per second: a smaller one is a misread file (a decimal comma in a CSV, a time and a velocity swapped, km/s). |
| `parseVelocities(text)`, `readVelocities(path)` | The format detected (HANDVEL, VELF, else CSV); from a file (`IOError` when unreadable). |

## velocity/moveout: overlays and LMO

| Proc | What |
|---|---|
| `lmoTime(offset, velocity, t0Ms)`, `nmoTime(offset, velocity, t0Ms)` | `t0 + 1000·\|x\|/v`; `sqrt(t0² + (1000·x/v)²)`, in ms. |
| `MoveoutKind` (`mvLinear`, `mvHyperbolic`), `Moveout`; `linearMoveout(velocity, t0Ms = 0)`, `hyperbolicMoveout(velocity, t0Ms = 0)`, `hyperbolicMoveout(f: VelocityFunction, t0Ms)` | A curve at a constant velocity (or a function's RMS velocity at `t0Ms`). |
| `m.kind`, `m.velocity`, `m.t0Ms`, `m.timeMs(offset)` | Its parameters and its time at an offset. |
| `moveoutTimesInto(rows \| meta, m, times)` | One time per row, ms, from each row's offset (every row, whatever its status; NaN for a NaN offset). |
| `moveoutTimesInto(rows \| meta, field, t0Ms, times)`, `moveoutTimesInto(rows, field, locations, t0Ms, times)` | Hyperbolas with a field's velocity at each row's location (`ensemble`, or `inline`/`crossline`), or at caller locations. |
| `moveoutTimesInto(rows \| meta, field, binning, t0Ms, times)`, `moveoutColumnsInto(meta, field, binning, t0Ms, cols)`, `checkSuits(field, binning)` | Hyperbolas with the field's velocity at the location of each row's bin (`binning.locationOf`): what a stack with `binning` corrects, also for bins from midpoints (`lineBinning`, `gridBinning`), whose numbers are not the header numbers. NaN for a row in no bin. `checkSuits` raises `ValueError` unless the field has nodes and is 2D for a 2D binning, 3D for a 3D one. |
| `moveoutColumnsInto(meta, m \| field, [t0Ms,] cols)` | The same as fractional block columns, ready to draw: `(t − t0Ms)/dtMs − firstSample`. |
| `applyLmo(b \| samples, meta, velocity, biasMs = 0, interp = siLinear)`, `removeLmo(...)` | Flatten first breaks in place: every live row earlier by `1000·\|x\|/v − biasMs`; and back. A display transform, not a pipeline op. |

## statics/elevation: datum statics

| Proc | What |
|---|---|
| `DatumStatics`; `datumStatics(datumZ, vReplacement)`, `d.datumZ`, `d.vReplacement` | A flat datum and a replacement velocity. |
| `d.staticMs(k: TraceKeys)`, `d.staticMs(k, st: TraceStatics)` | One trace's datum static from its elevations (and, with parseSEG's `TraceStatics` header columns, its datum elevations and depths); NaN when an elevation is. |
| `headerStaticMs(st)`, `headerStaticsInto(statics, staticsMs)` | The source + group static the headers record. |
| `staticsInto(meta, d, staticsMs)`, `staticsInto(meta, d, statics, staticsMs)` | One static per row, ms. |
| `applyStatics(b \| samples, meta, staticsMs, interp = siLinear)`, `applyStatics(b \| samples, meta, d, interp)` | Shift each live row by its static (the sample at t moves to t + static); the datum form computes them row by row. NaN statics leave a row alone. |

## stack/nmo: NMO

| Proc | What |
|---|---|
| `NmoParams`; `initNmoParams(interp = siLinear, maxStretch = 0.3, taperMs = 0)`, `p.interp`, `p.maxStretch`, `p.taperMs` | Interpolation and stretch mute (`maxStretch = Inf`: none). |
| `nmoStretch(offset, vrms, t0Ms)` | `t/t₀ − 1` at a constant velocity. |
| `nmoInto(src, srcStartMs, dst, dstStartMs, dtMs, offset, vrms, p): int` | The kernel: one row onto a zero-offset time axis (which may differ from the input's); returns the first unmuted column. A float32 row is interpolated in float32 (within 1e-6 of the float64 evaluation, relative to the samples' size). |
| `vrmsToSloth(v)`, `nmoSlothInto(src, srcStartMs, dst, dstStartMs, dtMs, offset, sloth, p): int` | The same kernel fed the sloth `1/Vrms²` of each output sample: convert a velocity column once, in place, and correct many rows with it without a division per sample (the same samples as `nmoInto`). |
| `nmoWorkspace(T, cols)` | The scratch of the block forms. |
| `applyNmo(b \| samples, meta, f: VelocityFunction \| field: VelocityField, p, ws)` | NMO in place, each row from its own `t0Ms`; a field is read at each row's location. Rows with a NaN offset, `t0Ms` or location are left alone. |
| `applyNmo(b \| samples, meta, field, binning, p, ws)` | NMO with the field read at the location of each row's bin, as `bruteStack` with that binning reads it: the preview of what will be stacked. Rows in no bin are left alone; `ValueError` when the field does not suit the binning. |

## stack/binning: CMP bins

| Proc | What |
|---|---|
| `BinningKind` (`bnCdp`, `bnLine`, `bnGrid`, `bnInlineCrossline`), `Binning` | A bin definition. |
| `cdpBinning(firstCdp, count, step = 1)`, `lineBinning(x0, y0, azimuthDeg, binSize, count, firstCdp = 1, maxOffLine = Inf, cdpStep = 1)` | 2D: by header CDP (`ensemble`); by midpoints projected onto a line, bin `i` numbered CDP `firstCdp + i·cdpStep`. |
| `gridBinning(x0, y0, azimuthDeg, crosslineSpacing, inlineSpacing, inlines, crosslines, firstInline = 1, firstCrossline = 1, clockwise = true, inlineStep = 1, crosslineStep = 1)`, `inlineCrosslineBinning(firstInline, inlines, firstCrossline, crosslines, inlineStep = 1, crosslineStep = 1)` | 3D: a rotated grid of midpoints, numbered in steps of `inlineStep` and `crosslineStep`; header inline and crossline numbers. `gridBinning` takes its spacings crossline first and its counts and first numbers inline first: pass them by name. |
| `b.kind`, `b.len`, `b.inlines`, `b.crosslines`, `b.is3D` | Its size. |
| `b.binOf(k: TraceKeys): int`, `binsInto(rows \| meta, b, bins)` | A trace's bin, −1 for none (NaN key, outside, off the line). |
| `b.binAt(x, y): int` | The bin of a midpoint position (line and grid binnings; −1 for none, always −1 for header binnings); the maths `binOf` uses. |
| `BinLocation` (`cdp`, `inline`, `crossline`, `x`, `y`); `b.location(bin)`, `b.locationOf(k: TraceKeys)` | What a bin stands for (NaN where undefined); the location of a trace's bin (all NaN when it is in none): where a stack reads a `VelocityField` for the trace. |

## stack/stack: `StackAccumulator`

| Proc | What |
|---|---|
| `StackAccumulator[T]`; `initStackAccumulator[T](bins, samples, dtMs, startMs = 0)`, `stackMemory(T, bins, samples)` | Sums and per-sample folds, and per bin a trace count and the midpoint sums: `bins × (samples × (sizeof(T) + 4) + 24)` bytes (`ValueError` for a negative size or one beyond `int`). A copy (`=copy`, `=dup`) is a full copy of the sums. |
| `acc.bins`, `acc.samples`, `acc.dtMs`, `acc.startMs`, `acc.foldAt(bin, j)`, `acc.sumAt(bin, j)` | Its shape and contents. |
| `acc.add(bin, x, first = 0, last = -1): bool`, `acc.addLive(bin, x): bool`, `liveSpan(x)` | Add a trace in any order (false: bin < 0 or a non-finite sample, nothing added); `addLive` counts the fold only between the first and last non-zero samples, so mutes do not count. |
| `acc.addMidpoint(bin, cmpX, cmpY)`; `acc.tracesAt(bin)`, `acc.cmpXAt(bin)`, `acc.cmpYAt(bin)` | Per bin: the traces added (a fold map), and the mean of the midpoints given for them (NaN when none): where a bin of header numbers lies on the map. |
| `acc.merge(other)`, `acc.clear` | Add another thread's share (`ValueError` unless the bins, samples and time axis agree: the interval by `sameInterval`, the start within 1e-9 of a sample); zero it. |
| `FoldNorm` (`fnNone`, `fnFold`, `fnSqrtFold`); `acc.stackInto(dst, norm = fnFold)`, `acc.foldInto(dst: var Tensor[int32])` | The normalized stack and the fold copied into caller tensors `[bins, samples]`; the accumulator stays open. |
| `acc.finishInto(data, fold, norm = fnFold)` | Ends the accumulation without a copy: the sums are normalized in place and the accumulator's storage is moved into `data` and `fold` (a large stack is held once); `acc` is left with no bins. |

## batch: brute stack of a file

| Proc | What |
|---|---|
| `BruteStackOptions` (`flow`, `statics`, `staticsInterp`, `nmo`, `norm`, `samples`, `startMs`, `workers`, `chunkTraces`, `maxRingBytes`, `velocityCacheBytes`); `initBruteStackOptions(flow = initPipeline(), nmo = initNmoParams(), norm = fnFold, workers = 1)` | The processing and resources of a job. `statics` is an `Option[DatumStatics]`: `some(datumStatics(datumZ, vReplacement))` applies them first, `none` (the default) applies none; the module re-exports std/`options`. `velocityCacheBytes` (8 MiB): each worker's cache of `VelocityField` columns, one per bin it met last, so shot-sorted input reads the field once per bin; it never changes the result. |
| `bruteStack(f: SeismicFile, T, [traces: Slice[int] \| openArray[int],] binning, velocity: VelocityField \| VelocityFunction, opts = initBruteStackOptions(), progress: Progress = nil): StackResult[T]` | Statics, the flow and NMO inside parseSEG's prefetcher (`uniform = true`) on `opts.workers` threads; the calling thread stacks the corrected traces in order, so the result is the same bit for bit for any number of workers. `traces`: the whole file, a range, or a list of trace indices (a map selection, one channel set; stacked in list order, a trace listed twice counts twice). Blocks; `progress(done, total)` counts traces, and false raises `SegCancelledError`. `ValueError` for a file without keys, a trace outside the file, mixed intervals, a field that does not suit the binning or ends before the stack's last sample (`field.tMaxMs`), or bad options. |
| `StackResult[T]` (`data`, `fold`, `binning`, `dtMs`, `intervalUs`, `startMs`, `binTraces`, `cmpX`, `cmpY`, `stacked`, `skipped`, `failed`) | The stack `[bins, samples]`, its fold and counts. Per bin: `binTraces`, the traces stacked into it (a fold map), and `cmpX`, `cmpY`, their mean midpoint (NaN for an empty bin). `intervalUs` is the input's interval in µs, which `stackKeys` writes (`dtMs · 1000` is not always whole microseconds; 0 in a result built by hand: `dtMs · 1000`). |
| `bruteStackMemory(T, bins, samples, inSamples, workers, chunkTraces = 256, maxRingBytes = 4 shl 20, velocityCacheBytes = 8 shl 20)` | An estimate of the job's memory in bytes: one stack (it becomes the result), the prefetch ring and each worker's scratch and velocity columns. `ValueError` for bad arguments or a total beyond `int`. |
| `writeStack(path, res, opts = SegyOutputOptions(traceSorting: 4)): DiagLog` | The stack as SEG-Y, one trace per bin, headers from `TraceKeys` (CDP or bin index, inline/crossline, centre or mean midpoint, fold in `numHorzStacked`); empty bins are dead traces. Each trace is `stackKeys` + `stackOverrides`. |
| `stackKeys(res, bin): TraceKeys` | Bin `bin`'s stacked-trace keys (D14): CDP or bin index, inline/crossline, as midpoint the bin centre (line and grid binnings) or the mean midpoint of its traces (header CDP and inline/crossline binnings), offset 0, `startMs`, `intervalUs`, `tsDead` with no fold. For parseSEG's `writeTrace(w, keys, samples, overrides)`. |
| `stackOverrides(res, bin): array[1, HeaderOverride]` | The fold as `trace.numHorzStacked` (largest per-sample fold, clamped to int16). |
| `maxFold(res, bin): int32` | The largest per-sample fold of a bin. |

## qc/geometry: geometry flags from the keys

| Proc | What |
|---|---|
| `GeometryFlag` (`gfNoOffset`, `gfZeroOffset`, `gfNegativeOffset`, `gfNoCoordinates`, `gfOffsetMismatch`, `gfMidpointMismatch`), `GeometryFlags` | Findings on one trace. |
| `GeometryQc` (`tolerance = 1`, `relTolerance = 0.01`) | Tolerances, absolute and relative to the distance. |
| `GeometryCheck` (`flags`, `distance`, `offsetError`, `midpointError`); `checkGeometry(k, qc = GeometryQc())` | One trace's findings. |
| `checkGeometryInto(meta \| rows, dst, qc = GeometryQc())` | One per row. Rows that are not live (dead, auxiliary; failed rows of a meta) get no flags and NaN values, in both forms. |
| `badGeometry`; `flaggedRows(checks, flags = badGeometry): seq[int]` | The default flags (no, zero or mismatched offset); the rows with any of them. |

## qc/firstbreak: first-break picking

| Proc | What |
|---|---|
| `PickMethod` (`pmStaLta`, `pmEnergyRatio`, `pmCoppens`) | The picker. |
| `FirstBreakParams` (`pickMethod`, `windowMs`, `ltaMs`, `stabilization`, `lowCutHz`, `highCutHz`, `slopeDbPerOctave`, `searchStartMs`, `searchEndMs`, `guideVelocity`, `guideT0Ms`, `beforeMs`, `afterMs`, `snrWindowMs`, `lowDb`, `highDb`) | Defaults set in the type: energy ratio, 20 ms window, no filter, no guide. |
| `FirstBreakPlan[T]`; `initFirstBreakPlan(T, params, dtMs, samples)`, `plan.params`, `plan.dtMs`, `plan.samples`, `plan.workspaceSize` | A compiled picker (optional minimum-phase band-pass, fixed or LMO-guided window). |
| `plan.requiredMargin` | The samples a zoom window must hold on each side of a row's search window to pick as the whole trace does, bit for bit: what the attribute and the confidence read before the window, plus the band-pass's margin. `plan.samples` for a plan that reads whole traces (no window, or `pmCoppens`). |
| `FirstBreakPick` (`timeMs`, `col`, `snrDb`, `confidence`); `noPick()` | One pick: its time, its column in the block (an int; −1 without a pick, as `timeMs` and `snrDb` are NaN), confidence 0 .. 1 from the SNR (`lowDb` .. `highDb`). `FirstBreakPick()` is `noPick()`. |
| `plan.pick(x, ws, offset = NaN, t0Ms = 0)`, `plan.pickInto(b \| samples, meta, dst, ws)` | Pick one trace, or every live row (others get `noPick`). The samples are not changed. With a search window, STA/LTA and the energy ratio read and band-pass only the window and its margins; zeros before a row's first and after its last non-zero sample are not data. |
| `checkPick(plan, b, meta, picks)`; `pickInto(pool, plan, b \| samples, meta, dst)` | The checks of `pickInto` (`ValueError`); the picks with the rows split across a `ProcessPool`'s workers (identical results). |

## qc/snr: signal-to-noise ratios

| Proc | What |
|---|---|
| `SnrEstimator` (`seRatio`, `seExcess`) | `Ps / Pn` (signal + noise over noise: noise alone ≈ 0 dB) or `(Ps − Pn) / Pn` (signal over noise, unbiased). |
| `WindowSnrParams` (`signalStartMs`, `signalEndMs`, `noiseStartMs`, `noiseEndMs`, `estimator`, `floorDb`) | The two windows in ms, both ends included; defaults: 0 .. 40 after a pick over −60 .. −20, `seRatio`. |
| `DefaultSnrFloorDb` | −40 dB: returned for no signal, and every SNR clamped up to it (`floorDb = -Inf`: no clamp). |
| `meanPower(x, cols)`, `powerSnrDb(signal, noise, estimator = seRatio, floorDb, stabilization = 0)`, `windowSnrDb(x, signal, noise: Slice[int], estimator, floorDb, stabilization)` | The kernels: mean power over columns (NaN when empty), dB from two mean powers, both on one trace. NaN for an empty window or a zero noise power. |
| `windowSnrInto(b \| samples, meta, params, [refMs \| guide: Moveout,] dst)` | The windowed SNR of every row, in dB: windows at fixed trace times, relative to a time per row (picks), or to a moveout guide at each offset (LMO). NaN for non-live rows, NaN references, empty windows. |
| `coherenceSnrWorkspace(cols)`; `coherenceSnrInto(b \| samples, meta, windowMs, dst, ws, floorDb = DefaultSnrFloorDb)` | The SNR of a flattened (NMO/LMO-corrected) gather per column, over a centred window: signal power over noise power per trace from the mean trace and the residuals (an unbiased estimator, the semblance relation). Rows count over their `liveSpan`. The time does not depend on the window; the scratch is `6·cols + 2` float64 (`ValueError` when `ws` holds less). |

## qc/shotfit: per-shot refraction fit

| Proc | What |
|---|---|
| `RefractionModel` (`rmTwoSegment`, `rmOffsetBinned`) | Two lines in \|offset\| (L1 fit), or a median per offset bin. |
| `ShotFitParams` (`model`, `minConfidence`, `binWidth`, `localTraces`, `maxResidualMs`, `madFactor`, `correlate`, `windowMs`, `maxLagMs`, `lagToleranceMs`, `minCoherence`) | What is checked; defaults set in the type. |
| `ShotFit` (`model`, `interceptMs`, `v1`, `v2`, `crossover`, `used`, `asymmetryMs`, `threshold`, `position`, `modelMs`, `residualMs`, `localMs`, `lagMs`, `coherence`, `suspect`) | One shot's fit; the per-row seqs are resized and reused. |
| `fitShot(meta, picks, params, fit)`, `fitShot(b \| samples, meta, picks, params, fit)` | Fit one shot's rows; flag traces whose residual jumps against their neighbours, and (with samples) the odd ones out by cross-correlation. `asymmetryMs` hints at a misplaced source. |

## qc/breakfit: robust start and trimmed model

Which first-break picks a model is fitted to. `medianLineStart` ranks the confident picks by distance from the straight line of least median of squares (the best of `StartLines` lines through pairs of picks, a fixed generator). `trimmedModel` starts from the closer half of those and drops picks farther than `max(TrimFloorMs, TrimSigmas · sigma)`, refitting up to `TrimRounds` times. Not `trusted` with fewer than `MinTrimmedPicks` left. A start on a second segment is tried again from the kept picks and the nearest dropped ones (`MinSecondPicks`, `RestartSigmaShare`, `RestartSigmaFloorMs`). Times in ms. An empty input is NaN or an empty result, never an error. Parallel arrays of different lengths raise `ValueError`.

| Proc | What |
|---|---|
| `ConfidentPick`, `MadToSigma`, `TrimSigmas`, `TrimFloorMs`, `TrimRounds`, `MinTrimmedPicks`, `StartLines`, `MinSecondPicks`, `SecondGain`, `SecondGainMs`, `RestartSigmaShare`, `RestartSigmaFloorMs` | The constants of the trim and of a second segment. `ConfidentPick` is `ShotFitParams.minConfidence`'s default. |
| `ModelFn` | `proc (use: openArray[bool]): seq[float64] {.closure.}`: the caller's model at every trace, fitted to the picks `use` marks; empty when it cannot be fitted. |
| `Trimmed` (`trusted`, `model`, `kept`, `used`, `sigmaMs`, `rounds`) | What `trimmedModel` found. `model` is empty unless `trusted`; `sigmaMs` is NaN for none. When the next set falls under `MinTrimmedPicks`, `used` is the rejected count while `kept` is still the set just fitted (12 confident picks, the fit matches 9: trusted=false, used=9, 10 trues in `kept`, model empty). |
| `median(xs)`, `robustSigma(xs)` | The median (NaN when empty; the middle value, or the mean of the two middle); `MadToSigma` times the median absolute deviation about it. |
| `lineFit(x, t, use)` | Least absolute deviations, `t = interceptMs + msPerUnit · x` (40 reweighted rounds). `ok` is false for fewer than two finite points `use` marks, all at one `x`, or a line that does not rise. `ValueError` when `x`, `t` and `use` differ in length. |
| `medianIfCloser(dev, best)` | The median of `dev` when it can lie below `best`, else +Inf. Rearranges `dev` when that median is selected. A NaN is taken by a full sort. |
| `medianLineStart(offs, pickMs, confident)` | Confident picks with a known offset, and their distance from the robust line, nearest first. Empty for fewer than two. `ValueError` when `offs`, `pickMs` and `confident` differ in length. |
| `trimmedModel(offsetAbs, pickMs, confidence, fit)` | The trimmed model. The last call of `fit` is the model kept: a rejected restart refits the first set, so the caller's view of the last fit is that one. `ValueError` when `offsetAbs`, `pickMs` and `confidence` differ in length. |
| `keepSecond(beyond, sigmaOne, sigmaTwo)` | A second segment stays when `beyond` is at least `MinSecondPicks` and `sigmaTwo` is below `SecondGain` of `sigmaOne`, by at least `SecondGainMs`. NaN sigmas do not. |

## qc/fold: fold maps of a survey

Data in: a parseSEG `SurveyGeometry` (R and S records for the grid; X or O relations for the pre-plot fold); traces as `ChunkMeta` or `TraceRows` for the actual fold.

| Proc | What |
|---|---|
| `surveyGrid(g, binInline = NaN, binCrossline = NaN, azimuthDeg = NaN, centre = true, trimStrays = true, maxBins = 1 shl 26): Binning` | A `gridBinning` covering the midpoint of every source–receiver pair: inlines along `azimuthDeg` (default: the receiver-line azimuth, median bearing between consecutive stations), bins `binInline` along them (`gridBinning`'s `crosslineSpacing`; default half the median receiver interval) by `binCrossline` across (default half the median source interval: the natural bin of an orthogonal survey). Extent: half the sum of the source and receiver extents on each axis; numbers from 1. `centre`: bin centres at the midpoints' dominant phase per axis (trimmed circular mean of source and receiver phases, Rayleigh test; none significant: the minimum corner), within half a bin of the corner; `false`: the corner. `trimStrays`: stations outside quartile fences (±3·max(IQR, 10 bins) per axis) are left out of the extent, their pairs counted `outside`. `maxBins`: the most bins the grid may have (a mistyped bin size asks for millions of times more, and out of memory cannot be caught). `ValueError` for a bad given value, no stations with coordinates, a default that cannot be measured, or more than `maxBins` bins (the message names the bin sizes and the extent). |
| `FoldOptions` (`offsetBounds`, `azimuthSectors`, `sectorStartDeg`, `offsetRange`) | What is counted besides the fold: offset classes (ascending edges, n edges → n − 1 classes `[e_k, e_k+1)`), source → receiver bearing sectors (sector k from `sectorStartDeg + k·360°/n`), each bin's offset range. |
| `FoldMap`; `initFoldMap(binning, opts = FoldOptions(), maxBins = 1 shl 26)` | Pairs per bin of a `Binning`, empty. `ValueError` for bad options, or a binning of more than `maxBins` bins (nothing is allocated). |
| `clone(m): FoldMap` | An independent copy with its own counts. A `FoldMap` is moved, not copied: `=copy` and `=dup` are compile errors, because copied tensors would share their counts. |
| `m.binning`, `m.options`, `m.fold` (`[inlines, crosslines]` int32), `m.offsetFold` (`[classes, inlines, crosslines]`), `m.azimuthFold` (`[sectors, inlines, crosslines]`), `m.offsetMin`, `m.offsetMax` (float32, NaN: empty bin), `m.binned`, `m.outside`, `m.noCoordinates`, `m.notLive`; `clear(m)` | The counts (tensors not asked for are empty), lent: a tensor assigned from one shares the map's storage, so `clone` it to keep a snapshot. `clear` zeroes them, keeping the storage. |
| `preplotFold(g, binning, opts = FoldOptions(), domain = rdSource, maxBins = 1 shl 26): FoldMap` | The fold of every X relation (O with `rdReceiver`): midpoint bin by `binAt`, station coordinates from per-line sorted arrays with a run cursor (no hash lookup per pair). `ValueError` for a header binning, or more than `maxBins` bins. |
| `addFold(m, meta \| rows)` | Adds the live traces (D12: live or unknown status, not a failed row), in any order and chunking: midpoint `cmpX/cmpY`, else half-way between `srcX/srcY` and `rcvX/rcvY` (parseSEG `applySurvey` fills them); offset from the coordinates, else `\|offset\|`; bearing from the coordinates only. Header binnings use `binOf`. No allocation; the rows overload raises `ValueError` without keys. |
| `foldDifference(preplot, actual): Tensor[int32]` | Pre-plot − actual per bin (`[inlines, crosslines]`): positive where traces are missing, negative where traces fall that the pre-plot has fewer of (geometry errors). `ValueError` for maps over different bins. |

## qc/record: field records against the XPS

Data in: a parseSEG `SurveyGeometry` (X records) and one field record's traces (`ChunkMeta` or `TraceRows`).

| Proc | What |
|---|---|
| `RecordMatch` (`rmAuto`, `rmSourceStation`, `rmFieldRecord`), `TraceMatch` (`tmAuto`, `tmStation`, `tmChannel`) | How the X record is found (header source station via `recordsAt`, the FFID choosing among repeats; or the FFID) and how traces match it (receiver station; or channel). Auto: by station when the headers have one; for the record also by FFID when the source station has no X record. A station whose line and point are both 0 (a header left empty) is none, as NaN. |
| `StationRange` (`line`, `firstPoint`, `lastPoint`, `count`), `LineCompleteness` (`line`, `expected`, `live`, `missing`) | Consecutive missing X stations on a line; one receiver line's counts. |
| `RecordCompleteness` (`fieldRecord`, `source`, `hasSource`, `recordMatch`, `traceMatch`, `xpsRecord`, `candidates`, `xpsSource`, `recordMismatch`, `sourceMismatch`, `traces`, `live`, `dead`, `aux`, `failed`, `expected`, `present`, `presentLive`, `missing`, `unlisted`, `duplicates`, `channelMismatch`, `noStation`, `missingStations`, `missingRanges`, `lines`), `$` | One record: the X record (−1: none or ambiguous) with the other key checked (`recordMismatch`: X number ≠ FFID; `sourceMismatch`: X source ≠ header source), trace counts, station counts (multi-component channels count once; `missing` = `expected − present`, a count as in `LineCompleteness` and the summary), the missing stations one by one (`missingStations: seq[StationKey]`, as the X record names them, for a map) and as ranges (`missingRanges: seq[StationRange]`), per-line counts; differing channel numbers are informational. |
| `recordCompleteness(g, meta \| rows, recordMatch = rmAuto, traceMatch = tmAuto)` | Hashes the data's stations once, walks the X record's stations once. Failed rows (meta) are present, not live; auxiliary rows take no part, failed or not. The rows overload raises `ValueError` without keys. |
| `CompletenessSummary` (`records`, `matched`, `unmatched`, `ambiguous`, `recordMismatches`, `sourceMismatches`, `xpsWithoutData`, `expected`, `present`, `presentLive`, `missing`, `unlisted`, `duplicates`), `summarize(g, records)` | Totals over many records, and the X records none matched. |

## qc/amplitude: amplitude QC of field records

| Proc | What |
|---|---|
| `AmplitudeFlag` (`afNonFinite`, `afDead`, `afConstant`, `afClipped`, `afSpiky`, `afDcOffset`, `afWeak`, `afNoisy`) | Why a trace stands out: NaN/Inf samples; every finite sample 0, or one non-zero value; two runs at the maximum (or one long one); samples far above their neighbours; a large mean; RMS far below or above the record's. |
| `AmplitudeParams` (`startMs`, `endMs`, `clipRun = 3`, `clipTolerance = 0`, `spikeFactor = 10`, `spikeHalfWindowMs = 20`, `dcRatio = 0.2`, `outlierZ = 3.5`), `validate(params)` | The window on the trace axis (default: the whole trace) and the thresholds: a clip is a run of `clipRun` samples at `≥ (1 − clipTolerance)·max` or `≤ (1 − clipTolerance)·min` (each polarity at its own level when both reach the maximum magnitude within 1 %), flagged from two runs or one of `2·clipRun`; a spike deviates from the mean by more than `spikeFactor` × its neighbours' RMS deviation within ±`spikeHalfWindowMs` (itself left out); DC: `\|mean\| > dcRatio·σ`; outliers: `\|robustZ\| > outlierZ`. Inf turns `spikeFactor`, `dcRatio`, `outlierZ` off. `validate` raises `ValueError`. |
| `TraceAmplitude` (`samples`, `rmsDb`, `maxDb`, `mean`, `nonFinite`, `clipped`, `spikes`, `robustZ`, `flags`) | One trace over its finite samples; `samples = 0` and NaN for a non-live row or an empty window (a finite window needs `t0Ms`). |
| `traceAmplitude(x, dtMs, params = AmplitudeParams())` | One trace's samples (the window is not applied); allocates its scratch. |
| `traceAmplitudeWorkspace(cols)`; `traceAmplitudeInto(b \| samples, meta, params, dst, ws)`; `traceAmplitudeInto(pool, b \| samples, meta, params, dst)` | Every row into `dst[row]`, the live samples in the window, with `2·cols + 1` float64 of scratch in `ws` (`ValueError` when it holds less); with a `ProcessPool`, rows split across its workers, each with its own workspace (identical results). No allocation. |
| `RecordAmplitude` (`traces`, `medianRmsDb`, `madDb`, `counts`, `flagged`), `recordAmplitudeWorkspace(traces)`, `recordAmplitude(traces, params, ws)` | One record: median and MAD of `rmsDb` over the traces with samples, a finite RMS and neither dead nor constant; each one's `robustZ = (rmsDb − median) / (1.4826·MAD)` and `afWeak` / `afNoisy` (set or cleared), counts per flag. No allocation. |

## qc/exclusion: noisy traces outside the exclusion area

Data in: a gather's `ChunkMeta`, its `TraceAmplitude`s (`traceAmplitudeInto`), optionally its first-break picks (`qc/firstbreak`) and `ShotFit` (`qc/shotfit`).

| Proc | What |
|---|---|
| `GatherDomain` (`gdCommonSource`, `gdCommonReceiver`); `gatherDomainOf(meta)` | A shot gather (the far end is the receiver) or a receiver gather (the far end is the source); the second when every seismic row has one receiver station and there is more than one source station. |
| `SensorClass` (`scSingle`, `scPressure`, `scMotion`), `TraceRole` (`sensor`, `rotated`, `group`, `node`, `line`, `station`, `guessed`) | A trace's place in the statistics: pressure and motion are compared separately, an unrotated node's motion traces are one unit, rotated components are one group each; the far end's line and position along it. |
| `traceRolesInto(meta, domain, rotated, dst)` | Roles from the keys: sensor class from SEG-D `sensorType` or the SEG-Y trace code (11 pressure, 12-14 motion, 15-17 rotated motion), nodes from the far-end station, lines from header line numbers, else coordinates, else station numbers, else one line. `rotated`: one flag per row, or empty. A non-live row gets group −1. `ValueError` for another length. |
| `AreaChoice` (`acAuto`, `acFixedTime`, `acFixedRadius`, `acSteps`, `acOff`, `acTrend`), `AreaRule` (`arModel`, `arTrend`, `arRadius`, `arSteps`, `arFixedTime`, `arFixedRadius`, `arNone`), `AreaReason` (`rnNoPicks`, `rnFewPicks`, `rnNoOffsets`, `rnNoPosition`, `rnFewTraces`, `rnNoLoudness`, `rnOverrideNeedsModel`, `rnOverrideNeedsOffsets`, `rnFewArrivals`, `rnAllInside`, `rnUnknownSensors`, `rnNoStationIds`, `rnNoHeaderOffsets`, `rnBadModel`, `rnOneStation`), `OffsetOrigin` (`ooUnknown`, `ooHeaders`, `ooSurvey`), `LineSource` (`lsHeaders`, `lsCoordinates`, `lsStations`, `lsUnknown`) | What the user chose, which rule set the area, why a rule before it did not apply, where the offsets came from (for wording), and how the lines were found. |
| `ExclusionParams` (`choice`, `minConfidence`, `trendTraces`, `levelDb`, `edgeMs`, `radius`, `steps`, `stepsOnAllLines`, `minArrivals`, `minFraction`, `neighbours`, `minNeighbours`, `neighbourZ`, `offsetOrigin`), `validate(params)` | The choice (automatic: model, radius, steps, none; the trend only as `acTrend`, experimental), the confidence of a usable pick, the stations without a pick a line's trend bridges (7; a piece of a line needs more consistent picks than that), the level drop that places the edge (6 dB), the fixed time, radius and step count (8) with `stepsOnAllLines` (default: the lines the shot reaches), the coherent arrivals needed (8 and 5 % of the rows), the neighbour window (7 each side, at least 4) and the neighbour z-score (3.5; Inf: none). `validate` raises `ValueError`. |
| `NoiseState` (`nsJudged`, `nsInArea`, `nsUnusable`, `nsIncompleteNode`, `nsUngrouped`), `NeighbourState` (`nbJudged`, `nbInArea`, `nbUnusable`, `nbFewNeighbours`, `nbUngrouped`) | Per trace: whether it was judged, and if not why. |
| `GroupNoise` (`sensor`, `component`, `units`, `judged`, `inArea`, `noisy`, `weak`, `louder`, `medianDb`, `madDb`, `edge`) | One group's counts in units, its level (`medianDb`, `madDb` of the judged units: what a folder compares across receiver gathers; NaN when the area covered the group) and its edge. |
| `NoiseResult` (`domain`, `rule`, `reasons`, `offsetOrigin`, `edgeMs`, `radius`, `steps`, `lines`, `linesReached`, `linesFromHeaders`, `lineSource`, `agreement`, `excluded`, `judged`, `notJudged`, `arrival`, `inside`, `state`, `partial`, `neighbourDb`, `neighbourZ`, `louder`, `neighbourState`, `earlyBreak`, `groups`) | What was found; the per-row seqs are resized and reused. `agreement`: under `acTrend` with a model, the share of traces on which the two areas agree; NaN else. `earlyBreak`: a trace outside the area whose own confident pick is earlier than the edge, or inside it earlier than its arrival by more than pick noise (a burst before the break), a note, never a flag. The model is used only when it is physical (positive velocities, times rising with offset; else `rnBadModel`); a record whose far ends are all at one station has no area (`rnOneStation`). The trend (`acTrend` → `arTrend`) is a valley per line: earliest at the apex, not decreasing away from it, fitted to the picks consistent with it (module documentation). Experimental, with known failures: a late confident noise pick at a flank's outer end counts as consistent, so the trend ramps up to it and that line's near traces fall outside; a run of 6 or more burst picks just past the last visible break can win the flank and have the true apex cut; the vote between lines trusts station numbers that need not run along the line. |
| `noiseWorkspace(rows)`; `noiseQc(meta, amps, picks, roles, amp, params, domain, res, ws)`; `noiseQc(meta, amps, picks, fit, roles, amp, params, domain, res, ws)` | The statistics outside the area: sets `robustZ`, `afNoisy` and `afWeak` of `amps` (as `recordAmplitude` does for the record) and fills `res`, with the neighbour finding. The second form takes the caller's `ShotFit`, so a record is neither picked nor fitted twice; `picks` may be empty. The first fits the shot itself (it allocates in `fitShot`). `ValueError` for mismatched lengths, invalid parameters or a workspace smaller than `noiseWorkspace(meta.len)`. After the first gather of a size the second form allocates nothing. |

## qc/signalend: where a trace's signal ends

The last time a trace's signal is still up, in ms. `blockLevels` is the largest `|sample|` of each block of about 100 ms (at least 2 samples); a NaN sample is left out, and a block of only NaN is 0. `lastLoudMs` sorts the finite levels and takes the one at index `int(n · 0.9)` (the largest when there are 10 or fewer). The end is the time of the last level at least a tenth of that, plus the block length. With `peakFloor`, the level must also reach that fraction of the loudest, and the blocks after it stay in while they stay at least half that level. `signalEndMs` runs that on the trace with `peakFloor` = `RippleFraction`, and not later than `len · dtMs`. Shorter than 4 samples is NaN. A longer trace whose blocks measure no end reads as ending at `len · dtMs` (`min` drops a NaN). `levels` and `timesMs` are parallel: different lengths raise `ValueError`. An empty pair is NaN, not an error.

| Proc | What |
|---|---|
| `RippleFraction` | 1e-3: a block quieter than this fraction of the loudest is not signal (−60 dB). `signalEndMs` passes it as `peakFloor`. |
| `blockLevels(x, dtMs)` | The largest `|sample|` of each block of about 100 ms, and the block's length in ms. NaN samples are left out. |
| `lastLoudMs(levels, timesMs, blockMs = 0, peakFloor = 0)` | The time (ms) the signal ends. NaN when nothing was measured or the levels are silence. NaN levels are not measured. `ValueError` when `levels` and `timesMs` differ in length. |
| `signalEndMs(x, dtMs)` | Where the signal of `x` ends, from its blocks. NaN when `x` is shorter than 4 samples; otherwise not later than `len · dtMs`. |

## pipeline: `Op`, `Pipeline`, `PipelinePlan`

| Proc | What |
|---|---|
| `OpKind` (`opDemean`, `opDetrend`, `opScale`, `opGainDb`, `opClip`, `opPolarity`, `opMuteTop`, `opMuteBottom`, `opFilter`, `opIir`, `opAgc`, `opNormalize`, `opGainCurve`, `opCalibrate`, `opUncalibrate`), `Op` | One operation as plain data: `op.kind`, and its parameters by kind through `op.factor`, `op.dB`, `op.clip`, `op.mute`, `op.filter`, `op.iir`, `op.agc`, `op.norm`, `op.gainCurve`, `op.calibration`, read or edited in place (`flow[2].agc.windowMs = 300`). Asking for another kind's parameters raises `ValueError`, in every build mode. |
| `ClipParams`, `MuteParams`, `FilterParams`, `IirParams`, `AgcParams`, `NormalizeParams`, `GainCurveParams` | The parameter groups, edited in place (`flow[2].agc.windowMs = 300`). |
| `demeanOp()`, `detrendOp()`, `scaleOp(factor)`, `gainDbOp(dB)`, `clipOp(lo, hi)`, `polarityOp()`, `muteTopOp(curve, taperMs = 0)`, `muteBottomOp(curve, taperMs = 0)`, `filterOp(spec, trend = trLinear)`, `bandPassOp(corners, trend)`, `iirOp(spec, phase = ipZero, trend)`, `agcOp(windowMs, mode, target, epsilon)`, `normalizeOp(stat, scope, target, percentile)`, `gainCurveOp(curve, tMinMs = 0, tMaxMs = NaN)` | The builders; nothing is checked until `compile` (`bandPassOp` checks its corners). |
| `calibrateOp(cal: Calibration)`, `uncalibrateOp(cal)` | Recorded values to volts or physical units per row (`core/calibrate`), with the file's factors from the meta's `scales`; and back. `compile` refuses either after an op that changes amplitudes (scale, gain, clip, AGC, normalization, gain curve) and more than one of each; they convert dead rows, and auxiliary rows to volts, too. |
| `Pipeline`; `initPipeline(ops = [])`, `p.len`, `p[i]` (`[]`, also `var`), `p[i] = op` (`[]=`), `p.add(op)`, `p.insert(op, i)`, `p.delete(i)`, `p.clear`, `items`, `pairs` | An ordered, editable list of ops. |
| `PipelinePlan[T]`; `p.compile(T, dtMs, samples, recordSamples = 0)` | Validate and precompute every op for blocks up to `samples` wide; `ValueError` names the op (`op 1 (opFilter): ...`). `recordSamples`: the trace length when `samples` is a zoom window's width, so that gain-curve tables cover the whole record (`plan.recordSamples`). |
| `plan.generation`, `plan.dtMs`, `plan.samples`, `plan.recordSamples`, `plan.len`, `plan.requiredMargin`, `plan.workspaceSize`, `ws.reserve(plan)` | A plan's id (never 0, increasing), size, zoom margin (sum of the ops'), scratch. Any block up to `samples` wide works. |
| `PlanAmplitude` (`paRecorded`, `paCalibrated`, `paRelative`); `plan.amplitudeUnits` | What the output amplitudes are for recorded input: as recorded, in the calibration's per-row units (`resolveInto`), or relative (a gain op follows the calibration). |
| `p.requiredMargin(dtMs)`, `decodeWindow(shown, margin, traceLen)` | The margin without compiling; the samples to decode for a zoom (shown ± margin, clamped). |
| `plan.apply(b \| samples, meta, ws)`, `plan.apply(src, dst, meta, ws)` (blocks or tensors) | Run in place, or `src → dst` with the copy fused into the first op; tiles of about 256 KB. Never grows `ws`; `dst` is not resized. |
| `appliedResponseInto(flow, T, dtMs, samples, hz, gain, ws)` | The response as applied: a unit impulse at `samples div 2` through `compile` and `apply`, FFT and IIR ops forced to `trNone` (the caller's flow is not edited). `gain[i] = \|Σ x[j] exp(−2πi·f·j·dt)\|` at arbitrary Hz, not only bin centres, by a complex rotation per sample snapped onto that angle (Θ(`samples` · `hz.len`) arithmetic; one sine and cosine per frequency). Not `response()` (design, no edges). The impulse row is live, `t0Ms` 0, offset 0, no file scales. `ValueError` if the flow will not compile. Compiles every call, so it allocates the plan and the impulse (the rotation state stays in registers); `ws` is grown if it does not hold the plan. |
| `checkApply(plan, src, dst, meta)`, `plan.partCount`, `plan.partNeedsPanel(part)`, `plan.panelFactorOf(b, meta, part, ws)`, `plan.runPart(src, dst, meta, part, factor, ws)` | The pieces of `apply`, for a caller's own threads: a per-panel normalization splits the plan into parts. |

## core/pool: `ProcessPool`

| Proc | What |
|---|---|
| `ProcessPool`; `newProcessPool(workers = 4)`, `p.workers` | Persistent worker threads, each with its own `Workspace`. `ValueError` for `workers < 1`, `ResourceExhaustedError` when the OS refuses a thread. `=copy` and `=dup` are compile errors (move it); `=destroy` stops and joins the workers; `=wasMoved` empties it. |
| `pool.apply(plan, b \| samples, meta)`, `pool.apply(plan, src, dst, meta)` (blocks or tensors) | `plan.apply` with the rows split across the workers; blocks until done, identical results. One job at a time; a second caller waits. |
| `pool.ping` | An empty job: the per-job overhead. |
| `pool.averageSpectrogramInto(sg, b \| samples, meta, amp, rows = 0 .. high(int)): int` | `averageSpectrogramInto` with rows in fixed batches of 16, summed in order on the calling thread. Same bits for any worker count; the grouped sum can differ from the serial running sum in the last bits. No workspace and no `stop`. |
| `pool.averageAmplitudeInto(sp, b \| samples, meta, firstCol, amp, rows = 0 .. high(int)): int` | `averageAmplitudeInto` with rows in fixed batches of 16, summed in order on the calling thread. Same bits for any worker count; the grouped sum can differ from the serial running sum in the last bits (within about 1e-12 of the peak). Same rows left out. No workspace and no `stop`. `ValueError` before `amp` changes. |
| `correlateInto(pool, plan: CorrelationPlan[T], src, dst: RowBlock[T] \| Tensor[T], meta)` | Vibroseis correlation with the rows split across the workers; checks as `checkCorrelate` on the caller's thread. |
| `traceAmplitudeInto(pool, b \| samples, meta, params, dst)` | Amplitude QC (`qc/amplitude`) with the rows split across the workers, each reserving `traceAmplitudeWorkspace`. |
| `pickInto(pool, plan: FirstBreakPlan[T], b \| samples, meta, dst)` | First-break picking (`qc/firstbreak`) with the rows split across the workers; checks as `checkPick` on the caller's thread. |
| `fkAmplitudeInto(pool, plan: FkPlan[T], b \| samples, meta, firstCol, amp, rows = 0 .. high(int)): int` | f-k modulus (`fft/fk`) with the time pass split in runs of `fftLanes` traces and the spatial pass split by frequency. Same bits as `plan.fkAmplitudeInto` for any worker count, including a float64 window measured with `hypot` (decided once, before the jobs). `ValueError` as the serial form, on the caller, before any job. No `stop`. Returns the live rows used. |

## adapters: metas from parseSEG reads

| Proc | What |
|---|---|
| `intervalMs(rows: TraceRows)` | The rows' common interval from their keys' `intervalUs` (`ValueError` when unknown, 0 or mixed). |
| `chunkMeta(c: PrefetchChunk[T], dtMs = NaN, scales = [])` | The meta of a prefetch chunk: its rows, interval, `firstSample` and `failedRows`, in place. `dtMs` overrides headers without an interval; `scales`, a `FileScale` per row (`fileScalesInto`), is pointed at, not copied. |
| `chunkMeta(f, traces: Slice[int] \| openArray[int], [samples: Slice[int],] failed = [], dtMs = NaN, scales = [])` | The meta of a `readTracesInto` of a range or list (and sample window), with its `failedRows` list and file scales. |
| `pilotFromTrace(f, trace, T = float64, t0Ms = NaN, dtMs = NaN): Pilot[T]` | A recorded pilot (aux sweep or ground-force trace) from a file: samples, interval and start time from its keys, `sweep` from the SEG-Y sweep fields (trace header, else binary) when they describe a linear or exponential sweep that the trace can be: valid at its interval, no longer than it, and reaching the upper half of its band in the samples (`energyAbove`; some controllers write 0.1 Hz). |

## filescales: what a file says about its scale

| Proc | What |
|---|---|
| `fileScalesInto(f, traces: openArray[int] \| Slice[int], dst)`, `fileScalesInto(f, rows: TraceRows, dst)` | One `FileScale` per trace from its headers (SEG-D MP factor, Rev 3 descale multiplier and Sensor Info STY with its physical unit, STY read as mV per unit: an assumption, see the module; SEG-Y bytes 169-170, 203-212; SEG-2 DESCALING_FACTOR / STACK), for a meta's `scales`. SEG-Y headers that are all zero are unknown, not volts. Allocates a little scratch; `ValueError` for a wrong `dst` length or a trace outside the file. |
| `ScaleGroup` (`first`, `count`, `channelSet`, `fields`, `scale`); `scaleReport(f[, traces: Slice[int]])`, `$g` | The raw scale headers of the traces, grouped by distinct values (with the gain fields: SEG-Y 119-124, SEG-2 FIXED_GAIN, the SEG-D format code), and the `FileScale` they give: to check a file before calibrating. |
| `segyUnitOverrides(unit): seq[HeaderOverride]` | The SEG-Y trace header values stating samples in `unit` (203-204 set: Pa 1, V 2, m/s 6, m/s² 7, N 8; 169-170 and 205-212 zero), for parseSEG's `writeTrace`; `fileScalesInto` reads them back as physical. |

## vibroseis/taper: sweep-end tapers

| Proc | What |
|---|---|
| `TaperShape` (`tpLinear`, `tpCosine`, `tpCos2`, `tpHamming`, `tpBlackman`, `tpBlackmanHarris`, `tpKaiser`, `tpGaussian`, `tpPower`) | Ramp shapes, each the rising half of its window. |
| `SweepTaper` (`startMs`, `endMs`, `shape`, `param`); `initSweepTaper(startMs, endMs, shape = tpCos2, param = NaN)`, `noTaper()` | Start and end ramps in ms (0 or NaN: none); `param` is β, σ or the exponent, NaN by default also in an object built by hand: the shape's 6, 0.4 or 2. |
| `taperWeight(shape, x, param = NaN)` | The ramp `w(x)`, `x` in 0 .. 1. |
| `applyTaper(x: var openArray[T], dtMs, taper)` | Applies the ramps in place to a sweep of `x.len` samples (`x.len·dtMs` ms, a length of N × dt). |
| `validate(taper)`, `checkFits(taper, sweepMs)` | `ValueError` for bad lengths or parameters, or ramps that overlap (`startMs + endMs > sweepMs`; meeting is allowed). |
| `segyTaperType(shape \| taper)` | SEG-Y taper type: 1 linear, 2 cos², 3 other. |

## vibroseis/sweep: sweep generation

| Proc | What |
|---|---|
| `SweepLaw` (`slLinear`, `slDbPerOctave`, `slDbPerHz`, `slExponential`, `slTable`), `SweepPoint` (`tMs`, `fHz`) | How the frequency runs; a table point. |
| `SweepSpec` (`law`, `f1Hz`, `f2Hz`, `lengthMs`, `dtMs`, `boostDb`, `table`, `startPhaseDeg`, `amplitude`, `taper`, `maxNyquistFraction`) | A sweep. |
| `linearSweep(f1Hz, f2Hz, lengthMs, dtMs, taper = noTaper(), startPhaseDeg = 0, amplitude = 1, maxNyquistFraction = 0.8)`; `dbPerOctaveSweep(..., dtMs, boostDb, ...)`, `dbPerHzSweep(..., dtMs, boostDb, ...)`, `exponentialSweep(...)`, `tableSweep(points, dtMs, ...)` | Validated specs. A dB boost is of the amplitude spectrum, per octave or per Hz, and may add up to 300 dB over the band per octave, 150 dB per Hz. |
| `validate(spec)` | `ValueError`: frequencies ≤ 0, above `maxNyquistFraction`·Nyquist or Nyquist, a length under one sample, a bad table, boost or taper. |
| `sweepLen(spec)` | Samples: `lengthMs / dtMs` (a length is N x dt: the last sample is at `lengthMs - dtMs`); 0 for a spec whose interval or length `validate` rejects. |
| `sweepSamples(spec, T = float64): seq[T]`, `sweepInto(spec, x: var openArray[T])` | The sweep, `amplitude·taper·sin(φ₀ + 2π∫f)`. |
| `SweepHarmonic` (`order`, `amp`, `phaseDeg`) | One term of a distorted sweep: `amp·sin(order·φ + rad(phaseDeg))`, `φ` the fundamental's phase in radians (`startPhaseDeg` included), `rad` converting degrees to radians. Order 1, `amp` 1, phase 0 is the fundamental. |
| `sweepHarmonics(spec, T = float64, harmonics = [], samples = 0): seq[T]`, `sweepHarmonicsInto(spec, x, harmonics = [])` | The sweep plus `harmonics` (empty: `sweepSamples`). `samples` 0 is `sweepLen`; a longer buffer is the sweep and then zeros. A term whose instantaneous frequency `order·f(t)` is at or above Nyquist is left out of that sample (cut, not folded, and not a reason to refuse the call; order 1 is never cut). `ValueError` for an invalid spec, an empty buffer, or a harmonic whose `order` is not finite and > 0 or whose `amp` or `phaseDeg` is not finite. |
| `frequencyAt(spec, tMs)`, `phaseAt(spec, tMs)` | The instantaneous frequency (`f1Hz` and `f2Hz` exactly at the two ends, never outside them for the laws that run one way); the phase in radians (closed-form laws; NaN for the dB laws). |
| `segySweepType(spec)` | SEG-Y sweep type: 1 linear, 3 exponential, 4 other. |

## vibroseis/pilot: generated and recorded pilots

| Proc | What |
|---|---|
| `Pilot[T]` (`samples`, `dtMs`, `t0Ms`, `sweep: Option[SweepSpec]`); `len`, `lengthMs` | A pilot: sample `j` at `t0Ms + j·dtMs` from the time break. (`std/options` is re-exported.) |
| `generatePilot(spec, T = float64)` | The pilot of a sweep, from 0 ms, `sweep` set. |
| `initPilot(samples, dtMs, t0Ms = 0, sweep = none(SweepSpec))`, `validate(pilot)` | A pilot from the caller's samples; `ValueError` for fewer than 2 samples, non-finite or all-zero samples, a bad `dtMs` or `t0Ms`, a sweep at another interval. |
| `pilotFromRow(meta, data, row, t0Ms = NaN, sweep = none(SweepSpec))` | A recorded pilot from a block row: its live samples, `meta.dtMs`, `sampleTimeMs(row, 0)` as `t0Ms`. |
| `pilotKeys(pilot): TraceKeys` | Keys for parseSEG's `writeTrace`: interval, start time, trace code 6 (sweep), auxiliary. |
| `sweepOverrides(spec, binary = false): seq[HeaderOverride]` | The SEG-Y sweep fields (start/end Hz, length, type, taper lengths and type) as `trace.` overrides, or `binary.` fields for `HeaderEditor.setFileField`. The fields are whole Hz and ms: `ValueError` for a fractional value (it is not rounded) or one outside int16. |
| `sweepFromHeaders(f1Hz, f2Hz, lengthMs, sweepType, taperStartMs, taperEndMs, taperType, dtMs, samples = 0): Option[SweepSpec]` | The sweep SEG-Y fields describe (linear or exponential only), or none unless it is valid at `dtMs` up to Nyquist and, with `samples`, no longer than the trace. |
| `energyAbove(pilot, fHz): float64` | The fraction of the pilot's energy above `fHz`, from its samples' power spectrum. |
| `resamplePilot(pilot, dtMs, cutoffFraction = 0.9, halfWidth = 32, bandLimit = false)` | Band-limited resampling (Kaiser-windowed sinc at `cutoffFraction` of the lower Nyquist; flat up to the passband edge `cutoffFraction − 2.736 / halfWidth`, 0.8145 by default, closed from `cutoffFraction + 2.736 / halfWidth`). `ValueError` when over 0.5 % of the samples' energy lies above the passband edge, whatever the sweep description says; `bandLimit` cuts it instead (and drops the description). The description is kept, with the passband edge as its `maxNyquistFraction`, only if still valid. Taps beyond the pilot's ends are made up for by rescaling. |
| `InstrumentFilter` (`highCutHz`, `highSlopeDbPerOctave`, `lowCutHz`, `lowSlopeDbPerOctave`, `phase: IirPhase = ipMinimum`) | A recorder's filters as Butterworth responses (SEG-Y bytes 141-152). |
| `applyInstrumentFilter(pilot, filter)` | The pilot through the recorder's filters from rest (causal for minimum phase), same length. |

## vibroseis/recfilter: a reference through a recorder's filter

| Item | Description |
|---|---|
| `RecordingFilter` (`kind: RecFilterKind` = `rfkNone` / `rfkLinearPhase` / `rfkMinimumPhase`; `taps`; `highCutHz`, `highSlopeDbPerOctave`, `lowCutHz`, `lowSlopeDbPerOctave` (0 Hz = off); `coefficients`; `family: RecFilterFamily` = `rffButterworth` / `rffChebyshev`; `highOrder`, `lowOrder`; `rippleDb`; `shiftSamples`) | A recorder's filter and its bulk delay, all typed by the caller. Linear phase: a symmetric FIR of `taps` coefficients designed from the cutoffs and roll-offs. Minimum phase: typed FIR `coefficients`, or a Butterworth/Chebyshev cascade of the given orders. `shiftSamples`: whole samples moved earlier after filtering. |
| `applyRecordingFilter(pilot, filter)` | The pilot through the filter from rest, then shifted: the pilot's length, `dtMs` and `t0Ms`, the ring-out brought in by a positive shift kept, zeros shifted in by a negative one. `rfkNone` returns the pilot unchanged. `ValueError` as `checkFilter`, or when the output is not finite. |
| `checkFilter(filter, dtMs, samples = 0)` | `ValueError` saying what is wrong (cutoff not below Nyquist, low-cut not below high-cut, slope, order, ripple, taps, coefficients, a shift of `samples` or more). |
| `linearPhaseTaps(filter, dtMs)`, `linearDelaySamples(taps)`, `linearShiftSamples(taps)`, `coefficientsSymmetric(coefficients)` | The designed coefficients of a linear-phase filter; its delay `(taps − 1)/2` and the whole shift that backs it out, `floor((taps − 1)/2)` (an even count leaves half a sample); whether typed coefficients are symmetric. Typed `coefficients` replace the design under either kind. A low-cut below about `2·fs/taps` Hz cannot be designed with that many taps. |
| `responseAt(filter, dtMs, hz): Complex64` | The complex gain of the whole filter at `hz` (shift not included). |
| `bulkDelaySamples(filter, dtMs, fromHz, toHz)`, `suggestedShift(...)` | The least-squares delay over a band (exact for linear phase; the high-cut alone for a family) and it rounded: a suggestion for a minimum-phase filter, whose delay depends on frequency. |
| `maxDesignTaps` (2047), `maxCoefficients` (16384), `maxOrder` (32), `minRippleDb` (0.001), `maxRippleDb` (12), `maxSlopeDbPerOctave` (1000) | The limits `checkFilter` enforces. |
| `shiftResidualDeg(filter, dtMs, fromHz, toHz, shiftSamples)` | Mean and peak `\|phase\|` in degrees the filter leaves in the band after the shift. |

## vibroseis/correlate: correlation and sweep deconvolution

| Proc | What |
|---|---|
| `CorrelationMode` (`cmCorrelate`, `cmDeconvolve`) | `X·conj(P)`, or `X·conj(P)/(\|P\|² + ε)`. |
| `initCorrelationPlan(T, pilot, dtMs, recordSamples, outSamples = 0, mode = cmCorrelate, epsPercent = 1, normalize = true)` | A plan for records of up to `recordSamples`; `outSamples` 0 is the listen time `recordSamples − pilot.len`. `ValueError` for a pilot at another `dtMs` (use `resamplePilot`), no listen time, a bad `outSamples` or `epsPercent`. |
| `recordSamples`, `outSamples`, `pilotSamples`, `dtMs`, `mode`, `nfft`, `scale`, `workspaceSize` | Plan accessors; `scale` is the normalization (1/wavelet peak). |
| `correlateInto(plan, x: openArray[T], y: var openArray[T], ws)` | One record. |
| `correlateInto(plan, src, dst: RowBlock[T] \| Tensor[T], meta, ws)`, `checkCorrelate(plan, src, dst, meta)` | All rows but dead and failed ones, each to its live length; column `m` at the record's `t0Ms + m·dtMs` (the pilot's `t0Ms` removed exactly). `ValueError` for shapes, `dtMs`, `firstSample > 0`. |
| `correlate(plan, src, meta): Tensor[T]` | Allocating form. |

## vibroseis/vstack: `RecordStack`

| Proc | What |
|---|---|
| `RecordStackMode` (`rsVertical`, `rsDiversity`), `DiversityWeight` (`dwInversePower`, `dwInversePowerSq`) | The plain sum, or weights `1/p` or `1/p²` from each trace's power `p` in a centred window. |
| `RecordStackParams` (`mode = rsVertical`, `windowMs = 128`, `weight = dwInversePower`, `powerFloor = 0.01`, `clipLevel = 0`) | The window, weight, floor (× the trace's median window power) and clip (× its square root; 0 or Inf: none) apply to `rsDiversity`. |
| `RecordStack[T]`; `initRecordStack[T](rows, samples, dtMs, params = RecordStackParams())`, `recordStackMemory(rows, samples, mode = rsVertical)` | Float64 sums and int32 per-sample counts of the output rows, 12 bytes a sample (20 for diversity). `ValueError` for a bad shape, interval or diversity option. |
| `acc.rows`, `acc.samples`, `acc.dtMs`, `acc.params`, `acc.windowSamples`, `acc.workspaceSize`, `acc.countAt(row, j)` | Its shape, window (`2h + 1` samples), scratch (`3·samples + 1` float64, diversity only) and counts. |
| `acc.add(outRow, x, ws): bool` | Add one trace to an output row, in any order (false: `outRow < 0` or a non-finite sample, nothing added). No allocation once `ws` holds `workspaceSize`. |
| `acc.addRecord(b \| samples, [meta,] ws, rowMap = []): int` | Add a record's rows, row `r` to output row `rowMap[r]` (`-1`: skipped; empty: row `r`); with a meta only live rows, over their `liveSamples`. Returns the rows added; `ValueError` (nothing added) for a shape or map that does not fit, and with a meta for another interval than the stack's or a window (`firstSample > 0`). |
| `acc.merge(other)`, `acc.clear` | Add another thread's share; zero it. |
| `acc.stackInto(dst, norm = fnFold)`, `acc.foldInto(dst: var Tensor[int32])` | The stack `[rows, samples]`: the mean (diversity: `Σ w·x / Σ w`), × count (`fnNone`) or × √count (`fnSqrtFold`); the per-sample count. |

## vibroseis/sweeptrack: vibrator signals against sweep time

A signal (ground force, mass or baseplate acceleration) against its reference (the pilot),
at points every `stepMs` along the sweep: harmonic k's complex amplitude over a whole number
of periods of the reference (`cycles`, at least `minWindowMs`), with a cos⁴ weight in phase
(orthogonal to every other harmonic for any sweep law). The reference's phase comes from its
own analytic signal, demeaned and without its content below the band. Guide: [vibrator
QC](guide/vibrator-qc.md), with the plot each call fills.

| Proc | What |
|---|---|
| `TrackParams` (`harmonics = 5`, `cycles = 4`, `minWindowMs = 100`, `stepMs = 10`, `minReferenceDb = -40`, `maxHarmonicFraction = 0.8`, `maxFundamentalFraction = 0.8`, `harmonicBandHz = NaN`, `harmonicBandAuto = false`, `thd = thdFundamental`, `thdHarmonics = thFixed`), `validate(params)`, `trackedHarmonics(params)` | Highest harmonic order tracked (1 .. 16; THD sums 2 .. it), the window in periods (≥ 3) and its minimum length (where `cycles` periods are shorter, the smallest even number of periods that reaches it; 0 .. 1e6, 0: off), the points' spacing, where the sweep is (the reference's envelope within this of its peak), where harmonic k is measured (`k·f` below this fraction of Nyquist), where the fundamental is (its own fraction), a band edge in Hz (NaN: off) that makes THD sum the harmonics within it, up to 16 (`trackedHarmonics`: 16 then), what THD is relative to, which harmonics it sums. `ValueError` out of range. |
| `ThdReference` (`thdFundamental`, `thdTotal`) | THD as `√(Σ_{k≥2} A_k²) / A_1`, or over `√(Σ_{k≥1} A_k²)`. |
| `ThdHarmonics` (`thFixed`, `thAllThatFit`) | THD over every harmonic 2 .. `harmonics` (NaN where one is past the limit), or over those measured at the point (the count falls along the sweep). |
| `SweepTrackPlan[T]`; `initSweepTrackPlan(T, params, dtMs, samples)`, `tp.params`, `tp.dtMs`, `tp.samples`, `tp.points`, `tp.stepSamples`, `tp.pointMs(point)`, `tp.workspaceSize` | Tracking of signals of up to `samples` samples; points every `stepSamples` from sample 0. |
| `SweepReference`; `initSweepReference(tp)`, `tp.referenceInto(x, dst, ws)`, `r.span`, `r.flatSpan`, `r.extent`, `r.topFrequencyHz` (an estimate), `r.largestChordHz` (a lower bound), `r.isMeasured(point)`, `r.cyclesAt(point)` | A reference prepared once (its phase, frequency, extent and each point's window): the sweep's samples (empty when it has none: zeros, a NaN), its flat part between the tapers, both in ms, an estimate of the sweep's highest frequency and a lower bound of it, whether a point's window lies inside the sweep and its length in periods (0: not measured). |
| `SweepExtent` (`startMs`, `endMs`, `flatStartMs`, `flatEndMs`; NaN: no sweep) | Where the sweep and its flat part lie, ms from the rows' first sample: the flat part from the reference's envelope averaged over a period (where it stops rising and starts falling; the sweep's end on a side where it never does). |
| `SweepTrack` (`timeMs`, `referenceHz`, `frequencyHz`, `phaseDeg`, `amplitude`, `thdPct`, `thdCount`, `peak`, `extent`, `lastFundamentalMs`, `bandHz`, `fundamentalCut`); `initSweepTrack(tp)`, `t.len`, `t.harmonics` | One value per point, NaN where not measured: the reference's and the signal fundamental's instantaneous frequency, the phase to the reference (degrees, (−180, 180], positive when the signal leads), each harmonic's peak amplitude (`amplitude[order − 1]`: `[0]` is the fundamental, a ground force's *fundamental force*), THD % and the harmonics it sums (0 where NaN), the largest \|s\| per period of the band-limited signal (*peak force*); the reference's extent; the last point whose fundamental is measured, and whether the fundamental's frequency limit cut the track. |
| `tp.trackInto(x, reference, track, ws)` | Every measure of `x` against `reference`. No allocation; never raises. |
| `track.distortionInto(order, dst)` | `100·A_order / A_1` % per point. `ValueError` for an order outside `2 .. harmonics`. |
| `compareTracksInto(a, b, phaseDeg, ratioDb)` | Signal `a` against `b` per point: the fundamentals' phase difference (wrapped) and ratio in dB. |
| `wrapDeg(deg)` | An angle wrapped to (−180, 180]. |

## vibroseis/sweepedge: the top frequency of a reference sweep

The top frequency of a reference, read from its samples, and the uncertainty of that
reading. The frequency law near the top is taken to where the sweep ends. The end ramp's
shape is not assumed: the fall is fitted against powers of the time left and the cosine
squared, and the data weigh them. `hz` is NaN when there is no reading.

| Proc | What |
|---|---|
| `UncertainFraction` (0.006) | An uncertainty (three standard errors) above this fraction of the top frequency marks the reading uncertain. |
| `TopReason` (`tpNone`, `tpNoSweep`, `tpTaperShort`, `tpNoTop`, `tpNotSure`) | Why a reading is uncertain, or why there is none. `tpNone`: no sentence, the reading is sure enough. `tpNoSweep`: the reference has no sweep to read. `tpTaperShort`: the end taper of the sweep is too short or too noisy to be fitted. `tpNoTop`: the sweep's frequency law gives no top. `tpNotSure`: the end of the sweep is not sure enough. |
| `TopReading` (`hz`, `uncertaintyHz`, `uncertain`, `endMs`, `lastMs`, `rateHzS`, `why`) | The reading. `hz` NaN: none. `endMs` is where the sweep is taken to end (its top-frequency end). `uncertaintyHz` is three standard errors (Inf when there is no reading). `lastMs` is the last point of the fall that was fitted. `rateHzS` is the sweep's rate at its end, Hz per second. `why` is a `TopReason`. |
| `notBefore(tEnd, tLast, dir)` | `tEnd` kept at or past `tLast`, the last point of the fall that was fitted, in the sweep's direction (`dir` 1: later in time, -1: earlier). |
| `Weighed` (`tEnd`, `seSq`, `spread`, `best`) | The shapes' roots weighed by their fits: the weighted root (kept at or past the last point fitted), the weighted squared standard error of the roots that count, the standard deviation of the roots between the shapes, and the shape that fits best. |
| `weighShapes(roots, sses, floors, rootSe, n, kCorr, tLast, dir)` | The shapes weighted by how well each fits. The end is never short of the last point fitted. `ValueError` when the arrays differ in length. |
| `readTop(x, dtMs, track, extent)` | The reference's top frequency: the end of the sweep where its frequency is highest. `x` is the reference's samples (the first at 0 ms), `track` its frequency track and `extent` its measured sweep and flat part. Allocates per call. `ValueError` when `track.timeMs` and `track.referenceHz` differ in length. |

## vibroseis/crosscorrelate: a signal against its reference

| Proc | What |
|---|---|
| `CorrelationNorm` (`cnCoefficient`, `cnReference`, `cnNone`) | Divided by `√(Σs²·Σr²)` (1 for the reference's shape), by `Σr²` (the reference scaled by a peaks at a), or not. |
| `CrossCorrelationPlan[T]`; `initCrossCorrelationPlan(T, maxLagMs, dtMs, samples, taperFraction = 0.1, nfft = 0)` | Correlation of rows of up to `samples` samples over the lags `−L .. L` (`L = maxLagMs / dtMs`), by FFT without wrap; the lag window's spectrum on `nfft` points (0: the next power of two of `2L + 1`), Tukey-tapered over `taperFraction` at each end. `ValueError` for a bad `dtMs`, `samples < 2`, a lag under one sample or past the samples, a bad taper or `nfft`. |
| `cp.samples`, `cp.maxLag`, `cp.lags`, `cp.lagMs(i)`, `cp.dtMs`, `cp.nfft`, `cp.bins`, `cp.binHz(k)`, `cp.workspaceSize` | Its axes: `lags[i]` is lag `i − L` (`lagMs`), a delay τ peaks at +τ. |
| `cp.crossCorrelateInto(x, reference, lags, [envelope,] ws, norm = cnCoefficient)` | `Σ x[j + m]·reference[j]` at the lags, and the envelope (the modulus of the whole correlation's analytic signal). NaN for a row without energy. |
| `cp.lagSpectrumInto(lags, amp, phaseDeg, ws)` | The tapered lag window's amplitude and phase (degrees, from lag zero): the signal's phase relative to the reference against frequency (a delay τ: −360·f·τ). |
| `cp.compareInto(a, b, ratioDb, phaseDeg, ws)` | `a` against `b` per bin: `10·log10(Ŝ_aa/Ŝ_bb)` dB and `arg Ŝ_ab` degrees, from the lag-windowed spectra. |


## vibroseis/matchpeak: the correlation envelope's peak

The match is the envelope's peak (a parabola through it and its two neighbours),
the time shift is that peak's place, and the rotation is the correlation's phase
there. A peak under `NoClearMatch` that does not stand `ClearPeakToSidelobe`
times above the envelope elsewhere is no clear match. A peak at an end of the
lags is no peak.

| Proc | What |
|---|---|
| `NoClearMatch` (0.3), `ClearPeakToSidelobe` (4.5), `SidelobeGapMs` (20), `MinSidelobeSpanMs` (40), `PreciseShiftPeakToSidelobe` (20), `PreciseShiftMatch` (0.7) | A peak at or above `NoClearMatch`, or `ClearPeakToSidelobe` times the RMS beyond `SidelobeGapMs` (when at least `MinSidelobeSpanMs` of lags lie there), is clear. A shift is to 0.1 ms above `PreciseShiftPeakToSidelobe` or at `PreciseShiftMatch`; otherwise to the millisecond. |
| `MatchVerdict` (`mvNothing`, `mvNoPeak`, `mvWeak`, `mvClear`) | No finite sample; the peak is at an end of the lags; under the clear thresholds; a clear interior peak. |
| `Match` (`verdict`, `match`, `shiftMs`, `rotationDeg`, `at`, `offset`, `sidelobe`, `about`) | The reading: `match` at most 1 (NaN for `mvNothing`); `shiftMs` NaN unless `mvClear`; `rotationDeg` NaN until `rotationOf`; `at` is -1 for none; `offset` within half a sample; `about` when the shift is only good to a millisecond. |
| `peakAt(y, i)` | The parabola through `y[i-1]`, `y[i]`, `y[i+1]`: offset from `i` (within half a sample) and height. A flat top, or `i` at an end, is the sample itself. `ValueError` when `i` is outside `0 ..< y.len` (an empty `y` too). |
| `matchOf(envelope, lagsMs)` | The match and the shift. NaN samples are skipped. The clear test comes before the end test, so a weak peak on the last lag is `mvWeak`, not `mvNoPeak`. `ValueError` when the arrays differ in length. |
| `rotationOf(c, dtMs, at, offset)` | The phase turn in degrees of the float32 correlation `c` (lag 0 in the middle) at sample `at`, taken `offset` towards the next sample. NaN when it cannot be read, including a `dtMs` that is not finite or not greater than 0. Allocates a Hilbert plan, a workspace and the analytic buffers. |

## vibroseis/vibqc: vibrator QC of a record

Guide: [vibrator QC](guide/vibrator-qc.md); the viewer's plot → call table:
[integration.md](integration.md#vibrator-signature-analysis).

| Proc | What |
|---|---|
| `VibSignal` (`vsReference`, `vsGroundForce`, `vsMass`, `vsBaseplate`, `vsWeightedSum`), `VibRole` (`vsReference .. vsBaseplate`), `defaultOrder` | A unit's signals: the four roles a row can have, and the ground force generated from M and B; the usual channel order (Ref, GF, M, B). |
| `Polarity` (`poAsRecorded`, `poReversed`, `poAutomatic`) | How a role's row is taken: as it is, times −1, or reversed when its mean phase over the judged part is beyond ±90° (the ground force alone; M and B together from the ground force they make; not the reference). |
| `VibUnit` (`rows`, `units`, `sensitivity`, `polarity`, `reactionMass`, `baseplateMass`); `initVibUnit(reference, groundForce = -1, mass = -1, baseplate = -1, reactionMass = Mass(), baseplateMass = Mass())`, `vibUnitsInOrder(count, firstRow = 0, order = defaultOrder, reactionMass, baseplateMass)` | One unit: each role's row (−1: none; the reference is required), the unit each row is in as passed, an optional sensitivity per role (a row in volts to m/s² or N), a polarity per role (as recorded by default), the masses. `vibUnitsInOrder` lays out `count` units whose channels follow each other in `order` (`ValueError` for a bad count or order). The library never assigns a role by position, nor from a row's status. |
| `groundForceInto(mass, baseplate, reactionMass, baseplateMass, dst)` | `Mr·a_M + Mb·a_B`, the masses in kg: N for accelerations in m/s² (NaN for a mass not set). |
| `LimitKind` (`lkPhase`, `lkThd`, `lkHarmonic`, `lkForceLow`, `lkForceHigh`, `lkPeakForce`, `lkMeanPhase`, `lkMeanThd`, `lkMeanForce`), `VibLimits` (`phaseDeg`, `thdPct`, `harmonicPct`, `forceLow`, `forceHigh`, `peakForce`, `meanPhaseDeg`, `meanThdPct`, `meanForceLow`; NaN: none; the judged part: `skipStartMs`, `skipEndMs`, NaN: the tapers; `fromMs`, `toMs`, NaN: no bound), `validate(limits)` | Optional limits per point and on the means over the sweep, checked over the judged part: the sweep without `skipStartMs` at its start and `skipEndMs` at its end (NaN: its tapers, from the reference's envelope), within `fromMs .. toMs`. `validate`: `ValueError` for a skip that is not NaN or a finite length ≥ 0. |
| `judgedPart(extent, limits = VibLimits())`, `judgedInto(track, limits, dst)` | The judged part's first and last point time (`NaN .. NaN` without a sweep); per point, whether it lies in it (a viewer dims the others). |
| `limitFlagsInto(track, limits, flags)`, `TrackSummary` (`points`, `phasePoints`, `thdPoints`, `meanPhaseDeg`, `meanAbsPhaseDeg`, `maxAbsPhaseDeg`, `meanThdPct`, `maxThdPct`, `meanForce`, `minForce`, `maxForce`, `maxPeak`, `exceeded`, `failed`, `reversed`); `summarize(track, limits = VibLimits())` | Each point's set of limits exceeded; a signal's statistics over the judged part (the points with a fundamental, a phase and a THD: a limit judges those; the mean phase circular, weighted by the fundamental), the points beyond each limit, the limits failed, and whether it was reversed (set by `analyzeVibrators`). |
| `reversePolarity(track)`, `automaticPolarity(track, limits = VibLimits()): bool` | A track's measures for the signal times −1 (phases turned by 180°; no second pass); reverses a track whose mean phase over the judged part is beyond ±90°, and says whether it did. |
| `WindowStats`, `WindowSummary`; `track.windowStats(limits, windowMs = 500, minFill = 0.5, originMs = NaN)` | The controller-style statistics of a track: per window aligned to the sweep's start, the mean \|phase\|, mean fundamental, largest peak and mean THD; over the sweep their average (mean of the windows) and worst window with its start time. Judged, measured points only; a window cut short is kept when its points span `minFill` of it. `ValueError` for a bad window length or fill. |
| `UnitSignature[T]` (`signals`, `reference`, `tracks`, `units`, `summaries`, `flags`, `judged`, `reversed`, `weightedSum`); `vibratorWorkspace(tp)`; `tp.analyzeVibrators(b \| samples, meta, units, limits, results, ws)` | Every measure of every unit of a record: each signal, in its polarity, tracked against the unit's reference, the generated force, the summaries over the judged part (limits on the two ground forces), per-point flags, which points are judged, and the signals reversed. Sized on the first call, refilled without allocating after. `ValueError` (before any result changes) for a block that does not fit, another interval, a window, rows wider than the plan, limits `validate` refuses, a unit without a reference, a row outside the block, a sensitivity on the reference, on a row not in volts or of another quantity, one mass without the other, an automatic polarity on the reference, on M or B without the generated force or on one of them only, or M and B in different units. |

## continuous/slice: continuous recordings into records

| Proc | What |
|---|---|
| `SliceShot` (`time`, `id`, `srcLine`, `srcPoint`, `srcX`, `srcY`, `srcZ`, `srcIndex`); `initSliceShot(time, id, srcLine = NaN, ..., srcIndex = 0)` | A source event: UTC POSIX seconds and record number, and its station (point index 0: unknown) and position when known. |
| `SliceWindow` (`startMs`, `lengthMs`) | The window at each shot: start relative to the shot (may be negative), length a whole number of samples. |
| `SliceOrder` (`soReceiverGathers`, `soShotRecords`), `SliceAlign` (`saNearest`, `saLinear`, `saCubic`) | Output order; sub-sample window starts kept (nearest sample) or shifted out with `core/shift`. |
| `SliceStreams`; `initSliceStreams(meta, [times, [receivers]])` | Once per dataset: the receivers of the segment rows and each segment's place on its receiver's sample grid (two sorts: 130 to 260 ms for a million rows). Times default to `TraceKeys.time`, receivers to the station (`stationKey(rcvLine, rcvPoint, rcvIndex)`: line and point to 1/100, with the point index), then `channel` and `component`; segments are never stitched across receivers. A value with no pointer into the meta: keep it, and plan from it on any thread. `ValueError` for mismatched array lengths, `dtMs` under 1e-4 ms, or a row of more than 2⁴⁸ samples. |
| `streams.dtMs`, `streams.rowCount`, `streams.receiverCount`, `streams.receiverKeys(receiver)`, `streams.overlaps`, `streams.timingIssues`, `streams.untimed`, `streams.otherInterval` | The dataset's receivers and data problems, as a plan lists them. |
| `SlicePlan`; `planSlices(streams, shots, window, order = soReceiverGathers, align = saNearest)` | Lays out every (receiver, shot) trace, at the cost of those traces alone (0.06 ms for a shot on 1000 receivers, whatever the number of segments). `ValueError` for a non-whole window or a non-finite shot time. |
| `planSlices(meta, [times, [receivers,]] shots, window, order = soReceiverGathers, align = saNearest)` | `initSliceStreams` and `planSlices` in one call, for a plan made once (the streams are built again on every call). `ValueError` as both. |
| `plan.len`, `plan.samples`, `plan.dtMs`, `plan.margin`, `plan.order`, `plan.align`, `plan.window`, `plan.shotCount`, `plan.receiverCount`, `plan.shotOf(trace)`, `plan.receiverOf(trace)`, `plan.receiverKeys(receiver)` | Its shape and trace layout. |
| `SlicePiece` (`row`, `first`, `dst`, `count`); `pieces(plan, trace)`, `plan.rowsNeeded`, `plan.tracesNeeded`, `plan.maxColumn` | The rows and columns each trace reads (`row = -1`: a gap). `rowsNeeded`: the rows of the planning view any trace reads; `tracesNeeded`: their trace numbers, the only traces to read from the file (`chunkMeta(f, plan.tracesNeeded)`). |
| `plan.fraction(trace)`, `plan.gapSamples(trace)`, `plan.untimed`, `plan.otherInterval` | Sub-sample remainder of the window start (−0.5 to 0.5); uncovered samples; live rows left out for having no finite time, or another sample interval than the meta's (data properties: reported, never raised). |
| `SliceOverlap` (`earlier`, `later`, `first`, `samples`), `plan.overlaps`; `SliceTimingIssue` (`earlier`, `later`, `residual`), `plan.timingIssues` | Overlaps (the later segment wins, always); segments off the previous one's sample grid (tolerance max(0.01 sample, 1 µs)). |
| `sliceKeys(plan, trace)` | The output trace's `TraceKeys` (D14): receiver keys; record, source line, point, index and coordinates from the shot alone; `time` = shot time, `t0Ms` = `startMs`; `intervalUs` exactly as the receiver's keys hold it (`dtMs · 1000` when they hold none). |
| `sliceWorkspace(T, plan)`, `SliceReport`, `initSliceReport(plan)` | Scratch (only with interpolation) and a report sized for the plan. |
| `extractInto(plan, data, dst, meta, report, ws)` (blocks or tensors) | Copies and stitches into `dst` (`plan.len × plan.samples`) from all rows of the planning view or only `rowsNeeded` (read as `tracesNeeded`); gaps zero. The same bits as the piecewise cut. Allocates nothing. |
| `pieceSource(plan, pc, rowSamples)` | How many source samples piece `pc` needs and where they start: `pc.first ..< pc.first + pc.count` clipped to the row. The alignment margin is already in `first` (clipped at the segment's ends); do not subtract `margin` again. A gap piece needs nothing. |
| `prepareSliceReport(plan, report)`, `beginSliceTrace(plan, trace, report)`, `addPiece(plan, trace, pc, samples, dst, report, ws, failed = false)`, `finishSliceTrace(plan, trace, dst, report, ws)` | Cut one trace from the columns `pieceSource` names, without holding the row from sample 0. `addPiece` zero-fills a gap piece and a failed row (a clipped margin is that gap piece; a failed row is recorded apart from coverage gaps). A data piece must be exactly the length `pieceSource` returns. `addPiece` and `finishSliceTrace` must share `dst`. `finishSliceTrace` applies the same shift as `extractInto`. Pieces of one trace in iterator order; traces in any order. Allocates nothing after `reserve`. `ValueError` for a bad trace, report, piece, length or `dst`. |
| `checkSliceOverlap(plan, i, earlier, later, report)` | Writes `osMatch` or `osMismatch` for overlap `i` when both runs are given (a NaN equals a NaN). The earlier row's trimmed tail is in no piece, so the check stays `osNotRead` until then. |
| `report.len`, `report.gapSamples(trace)`, `report.gapCount(trace)`, `gaps(report, trace)` (`SliceGap`: `first`, `count`) | Zero-filled coverage gaps per trace. A failed read is not one of these. |
| `report.failedSamples(trace)`, `report.failedCount(trace)`, `failedPieces(report, trace)` (`SliceGap`: `first`, `count`) | Zero-filled runs whose row the caller marked failed, apart from coverage gaps. |
| `OverlapStatus` (`osNotRead`, `osMatch`, `osMismatch`), `SliceOverlapCheck` (`status`, `differing`, `maxAbsDiff`), `report.overlapCheck(i)`, `report.mismatches` | Overlapping samples compared when both rows are read; a mismatch is a timing error. |
| `driftCorrectTimes(times, t0, offset0, t1, offset1)`, `driftCorrectedTimes(...)` | Linear clock-drift correction of one receiver's times, in place or a copy, before `planSlices`. |

## survey/patch: receiver patches of an X survey

Data in: a parseSEG `SurveyGeometry` (R, S and X records). Distances are along the receiver
lines (inline, the layout azimuth) and across them (crossline, 90° clockwise), relative to the
record's source, in grid units.

| Proc | What |
|---|---|
| `LineStation` (`point`, `index`, `inline`, `crossline`), `ReceiverLine` (`line`, `stations`, `pointStep`, `crossline`, `inlineMin`, `inlineMax`, `short`), `ReceiverLayout` (`azimuthDeg`, `interval`, `lineSpacing`, `minLineStations`, `lines`) | The receiver lines, sorted by crossline position; their stations by point. A `short` line (infill, relocated) is not a template line and not in the line spacing. |
| `receiverLayout(g, azimuthDeg = NaN, interval = NaN, lineSpacing = NaN, minLineStations = 0)` | Azimuth: median bearing between consecutive stations of a line (point order); interval: median station distance; line spacing: median crossline gap between lines that are not short; each measured unless given. Short: fewer than `minLineStations` stations (0: half the median line's). `ValueError` for a bad given value, or one that cannot be measured. |
| `layout.inlineOf(x, y)`, `layout.crosslineOf(x, y)` | Positions in the layout frame. |
| `PatchSide` (`psInlineBack`, `psInlineForward`, `psCrosslineLeft`, `psCrosslineRight`), `PatchLine` (`line`, `stations`, `firstPoint`, `lastPoint`) | Sides of a patch; one receiver line of a record. |
| `RecordPatch` (`record`, `source`, `sourceX`, `sourceY`, `channels`, `lines`, `noCoordinates`, `inlineMin`, `inlineMax`, `crosslineMin`, `crosslineMax`, `maxOffset`, `clipped`) | One X record: its lines and distinct stations per line, receiver extent relative to the source (NaN without coordinates), the sides where it stops short of the full patch. |
| `Spread` (`min`, `q1`, `median`, `q3`, `max`), `spreadOf(values)`, `$` | Five-number summary (nearest rank, NaN skipped). |
| `PatchDescription` (`layout`, `records`, `lines`, `channels`, `inlineHalfWidth`, `crosslineHalfWidth`, `maxOffset`, `boxInline`, `boxCrossline`, `clippedRecords`, `noSource`, `regular`, `templateLines`, `templateChannels`), `$` | The survey: spreads over records, the full patch (median half-widths: a record's farther side each way), and the lines × stations template of the unclipped records (exact when `regular`, else nominal). |
| `describePatches(g, [layout \| azimuthDeg = NaN], clipTolerance = 0.75)` | Describes every X record. A side is clipped when the record's receivers stop short of the full patch by more than `clipTolerance` × interval (inline) or × line spacing (crossline). `ValueError` for a negative tolerance. |
| `PatchEdge` (`peRoll`, `peShift`, `peStatic`), `PatchPlacement` (`ppCoordinates`, `ppNumbers`), `NumberMap` (`scale = 1`, `offset = 0`), `PatchBox` (`inlineMin`, `inlineMax`, `crosslineMin`, `crosslineMax`; infinite by default) | Edge modes (centred and clipped; shifted inside; one fixed spread), how the source is placed (by position, or receiver line/point = `scale` × source's + `offset`), a box in the layout frame. |
| `PatchTemplate` (`lines`, `channels`, `lineOffset`, `stationOffset`, `edge = peRoll`, `staticBox`, `placement = ppCoordinates`, `lineMap`, `pointMap`; distance field: `inlineHalfWidth`, `crosslineHalfWidth`, `boxAzimuthDeg = NaN`, `inlineDistance`, `crosslineDistance`, `radius`, `maxOffset`, Inf: none), `initPatchTemplate(lines, channels, edge = peRoll)`, `initDistancePatch(inlineHalfWidth, crosslineHalfWidth, edge = peRoll, radius = Inf, maxOffset = Inf)`, `hasTemplate`, `hasDistance` | A lines × station-slots template centred on the source plus the offsets (in lines and stations), distance limits (a box along the receiver-line or a given azimuth, centred on the source plus distances; a radius about the box centre; a maximum offset), or both (their intersection). |
| `PatchShot` (`record`, `source`), `patchShots(f: SpsFile \| g: SurveyGeometry, firstRecord = 1)` | The records to generate: S records in file order, or the geometry's sources in line/point order. |
| `generateRelations(g, [layout,] shots, template): seq[SpsRelation]` | X records: per shot the template lines nearest the source (not short), on each the station slots around it (one slot per `pointStep`; a slot missing from R is a hole), plus short-line stations inside the patch; channels 1..n by ascending line and point, one record per run of consecutive stations. Distance limits alone take every R station inside (short lines too); with a template, its receivers inside. A shot whose source has no coordinates gets none (`ppCoordinates`, or distance limits), nor does one whose source line or point is not a finite number (any mode). `peShift` moves a distance box inside the receivers' extent (bounding box in the box frame); it takes a template or distance limits, not both. `ValueError` for neither (roll, shift), an empty template, shift with both, a non-finite offset or map, a negative or NaN limit. Write with parseSEG `writeSps`; O records with `receiverRelations`. |
| `RecordAgreement` (`record`, `listed`, `generated`, `common`, `missing`, `extra`), `jaccard(a)`, `listedFraction(a)` | One record in two relation sets, by distinct receiver stations; `missing`: listed, not generated; `extra`: generated, not listed (receivers dropped in the field). |
| `compareRelations(listed, generated: SurveyGeometry): seq[RecordAgreement]` | Per X record of either, ascending (index 0 and 1 alike). |

## synth: synthetic traces and gathers

In-memory samples only: this module never opens a file. A direct wave, a refraction
and ground roll are linear events at different velocities and frequencies; a reflection
is a hyperbola. Guide: [synthetic records](guide/synthetic.md).

| Proc | What |
|---|---|
| `ricker(tauMs, freqHz)` | The Ricker wavelet, 1 at its centre, peak frequency `freqHz` (the amplitude spectrum peaks there). |
| `addRicker(x, dtMs, centreMs, freqHz, amp = 1, t0Ms = 0, firstSample = 0)`, `addSinusoid(x, dtMs, freqHz, amp = 1, phaseRad = 0, t0Ms = 0)` | Add a wavelet or `amp·sin(2π·f·t + phaseRad)` into a trace. `addRicker` takes `firstSample`: its column `j` is at `t0Ms + (firstSample + j)·dtMs`. `addSinusoid` has no `firstSample`: column `j` is at `t0Ms + j·dtMs`. The Ricker is not added more than 3 periods from its centre. `ValueError` for a bad `dtMs` or `freqHz`, for a non-finite `amp`, `t0Ms` or `centreMs` (`addRicker`) or `amp`, `phaseRad` or `t0Ms` (`addSinusoid`), and for `firstSample < 0` (`addRicker`). |
| `addNoise(x, sigma, seed = DefaultNoiseSeed)`, `addNoise(samples, sigma, seed = DefaultNoiseSeed)`; `DefaultNoiseSeed` (`1`) | White Gaussian noise of standard deviation `sigma` (`std/random`: `initRand(seed)`, then `gauss`). The same seed, length and `sigma` give the same samples on one machine and one Nim version. Not guaranteed across Nim versions (`std/random` can change) or platforms (`gauss` calls `ln`, which C libraries may round differently); a noise file that must be identical everywhere is generated once and stored. A left-out seed is `DefaultNoiseSeed`, so it repeats too. Two traces with the same seed, or none, get the same noise: give each trace its own seed. The tensor form seeds row `r` with `seed + r` (wrapping in int64). Seed 0 draws the same noise as 2147483647 (`initRand` replaces 0): avoid 0. `sigma` 0 leaves the samples unchanged. `ValueError` when `sigma` is not finite and ≥ 0. |
| `linearMs(t0Ms, velocity, offset)`, `hyperbolicMs(t0Ms, velocity, offset)` | Travel time in ms: `t0 + 1000·\|offset\| / velocity`, and `sqrt(t0² + (1000·\|offset\| / velocity)²)`. The same values as `lmoTime` and `nmoTime`. |
| `addLinear(samples, offsets, dtMs, t0Ms, velocity, freqHz, amp = 1, shiftsMs = [], traceT0Ms = [], firstSample = 0, liveCols = [])`, `addHyperbola(...)` | A Ricker at that travel time on every row, plus `shiftsMs[row]` when given. Column `j` is at `traceT0Ms[row] + (firstSample + j)·dtMs` (empty `traceT0Ms`: 0). Only the first `liveCols[row]` columns are written (empty: the whole row). A NaN offset or time adds nothing on that row. `ValueError` for a bad interval, frequency, velocity or amplitude, a non-finite `t0Ms` or `firstSample < 0`, a gather that is not rank 2 and C-contiguous, an array whose length is neither empty nor one per row, or a `liveCols` entry outside `0 ..` the row width (checked before any row is written). |
