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
  `applyFilter`, `shiftSamples`, `nmoInto`, `plan.pick`, `forward`, ...), the accessors of
  plans and metas, and the per-trace primitives (`lmoTime`, `binOf`, `staticMs`,
  `checkGeometry`, `acc.add`) are declared `{.raises: [].}`, so they can be called from a
  parseSEG `ChunkTransform` or another non-raising context. Three row kernels check an
  argument and raise `ValueError`: `clip` (its limits), `taperPads` (a ramp that does not
  fit) and `percentileAbs` (`p` outside 0 .. 100). The block forms raise only the argument
  `ValueError` above.
- **NaN means unknown.** Keys are read as parseSEG stores them, NaN where the headers
  cannot give a value (an offset without geometry, an unknown SEG-D `t0Ms`). processSEG
  never substitutes a default: anything derived from an unknown key is NaN (a tick, a
  static, a pick time), and the kernels leave such rows alone. `dtMs` itself is checked to
  be finite and positive.
- **Units.** Time in ms, frequency in Hz, gain in dB, velocity in distance units per
  second, with distance in the file's coordinate units (`summary.coordinateUnits`); no
  feet/metres conversion happens.
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

The metadata of a block of rows: parseSEG's `TraceRows` view plus `dtMs`, `firstSample`
and the failed rows. It copies no header value.

| Proc | What |
|---|---|
| `ChunkMeta`; `initChunkMeta(rows: TraceRows, dtMs, firstSample = 0, failed: openArray[int] = []): ChunkMeta` | Builds and validates a meta once. `ValueError` when the view has no keys (`hasKeys`), `dtMs` is not finite and positive, `firstSample < 0`, or `failed` is not ascending within `0 ..< rows.len`. `failed` is not copied: it must outlive the meta. |
| `m.rows`, `m.len`, `m.dtMs`, `m.firstSample`, `m.failedCount`, `m.failedRow(i)` | The fields, read-only. |
| `m.keys(row): lent TraceKeys` | Row `row`'s keys in place (parseSEG's `keysAt`). |
| `m.isFailed(row)`, `m.isLive(row)` | A failed row (O(log n)); a live row (status `tsLive`/`tsUnknown`, not failed). |
| `m.liveSamples(row, cols)` | The live length of the row in a block `cols` wide. |
| `m.t0Ms(row)`, `m.sampleTimeMs(row, col)` | The time of the trace's sample 0, and of a column of the block (NaN when `t0Ms` is). |
| `m.sub(rows: Slice[int]): ChunkMeta` | The meta of some rows (row 0 is `rows.a`), sharing the view and failed list: split work with `RowBlock.sub`. |
| `checkShape(m, samples: Tensor[T])`, `checkShape(m, b: RowBlock[T])` | `ValueError` unless one row per row of `m` (and, for a tensor, rank 2 and C-contiguous). |
| `SampleType` | parseSEG's `float32 \| float64`, re-exported. |

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

## fft/plan: `FftPlan`

| Proc | What |
|---|---|
| `fftLanes` (8) | Traces per batch of the batched transforms. |
| `isFastLen(n)`, `nextFastLen(n, even = true)` | 2·3·5-smooth lengths; the next one up (even by default: about twice as fast). |
| `FftPlan[T]`; `initFftPlan(T, n)`, `p.len`, `p.bins`, `p.workspaceSize(batched = true)` | A real FFT of length `n` (`ValueError` unless smooth); `n div 2 + 1` bins. |
| `p.forward(x, re, im, ws)`, `p.inverse(re, im, x, ws)` | One trace: unscaled forward, inverse scaled by 1/n. |
| `p.forwardLanes(x, re, im, ws)`, `p.inverseLanes(re, im, x, ws)` | `fftLanes` interleaved traces at once (SIMD across traces). |
| `gatherLanes(b, first, x)`, `scatterLanes(x, b, first)` | Rows of a block into and out of the interleaved lane layout. |

## fft/spectrum: `SpectrumPlan`

| Proc | What |
|---|---|
| `SpectrumPlan[T]`; `initSpectrumPlan(T, windowLen, dtMs, taperFraction = 0.1, nfft = 0, demean = true)` | Windowed amplitude spectra: demeaned, Tukey-tapered, zero-padded to `nfft` (0: the next power of two). |
| `sp.bins`, `sp.windowLen`, `sp.nfft`, `sp.binHz(k)`, `sp.workspaceSize` | Its axes and scratch. |
| `sp.averageAmplitudeInto(b \| samples, meta, firstCol, amp: var openArray[float64], ws, rows = 0 .. high(int)): int` | The power-averaged amplitude spectrum of the window `firstCol ..< firstCol + windowLen` over the live rows in `rows`; returns the rows used (rows with NaN or Inf are left out). A sinusoid of amplitude A on a bin reads A. |
| `sp.amplitudeInto(x, amp, phase, ws)` | One window of one trace: amplitude and phase (radians). |
| `amplitudePhaseInto(re, im, amp, phase)` | Modulus and phase of a raw `forward` spectrum, unscaled. |
| `DbReference` (`dbPeak`, `dbAbsolute`); `toDbInto(amp, db, reference = dbPeak, floorDb = -120)` | 20·log10 of the amplitudes, 0 dB at the peak or at 1, clamped at `floorDb`. |

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
| `FilterPlan[T]`; `initFilterPlan(T, spec, dtMs, samples, trend = trLinear)` | A filter for traces up to `samples` long: odd-reflection pads of `L`, a pad-only taper, an FFT of at least `samples + 4L`. |
| `plan.requiredMargin`, `plan.samples`, `plan.nfft`, `plan.spec`, `plan.dtMs`, `plan.binHz`, `plan.gainAt(k)`, `plan.workspaceSize` | Its margin `L`, size and response. |
| `applyFilter(x \| b, meta \| samples, meta, plan, ws)` | Filter one trace, or the live rows in place (`fftLanes` at a time). |

## filter/iir: `IirPlan` (Butterworth biquads)

| Proc | What |
|---|---|
| `IirPhase` (`ipZero`: forward-backward, \|H\|²; `ipMinimum`: one causal pass, \|H\|) | The phase. |
| `IirPlan[T]`; `initIirPlan(T, spec, dtMs, samples, phase = ipZero, trend = trLinear)` | A cascade for a `butter*` spec without notches; slopes in multiples of 12 dB/octave (zero phase) or 6 (minimum phase). |
| `plan.requiredMargin`, `plan.samples`, `plan.phase`, `plan.order`, `plan.spec`, `plan.magnitudeAt(f)`, `plan.workspaceSize` | Its margin (before the window for minimum phase), size and response. |
| `causalEnergyFraction` (0.9999) | The energy a minimum-phase margin holds. |
| `applyFilter(x \| b, meta \| samples, meta, plan, ws)` | As the FFT plan. |

## gain/gain: fixed gain, normalization, AGC

| Proc | What |
|---|---|
| `dbToFactor(dB)`; `scaleDb(x \| b, meta \| samples, meta, dB)` | 10^(dB/20); multiply by it (`ValueError` unless finite). |
| `NormStat` (`normMax`, `normRms`, `normPercentile`), `NormScope` (`perTrace`, `perPanel`) | What normalization brings to its target, per trace or per block. |
| `NormalizePlan`; `initNormalizePlan(stat, samples, scope = perTrace, target = 1, percentile = 99)` | A normalization; `p.stat`, `p.scope`, `p.target`, `p.percentile`, `p.samples`, `p.workspaceSize`. |
| `normalize(x \| b, meta \| samples, meta, plan, ws)` | Normalize in place; an all-zero or non-finite trace (panel) is left alone. |
| `panelFactor(b, meta, plan, ws): float64` | The one factor a `perPanel` normalization would apply (NaN: none), for a display-gain uniform. |
| `AgcMode` (`agcRms`, `agcMeanAbs`); `AgcPlan`; `initAgcPlan(windowMs, dtMs, samples, mode = agcRms, target = 1, epsilon = 1e-9)` | AGC over a centred window of `2h + 1` samples. |
| `p.requiredMargin` (h), `p.windowSamples`, `p.windowMs`, `p.mode`, `p.samples`, `p.target`, `p.epsilon`, `p.dtMs`, `p.workspaceSize` | Its window and size. |
| `agc(x \| b, meta \| samples, meta, plan, ws[, gain \| gains])` | AGC in place; the forms with `gain`/`gains` keep each sample's gain (1 where untouched). |
| `removeGain(x, gain)`, `removeGain(b \| samples, meta, gains)` | Divide by a kept gain: undo AGC. |

## gain/curves: time-variant gain

| Proc | What |
|---|---|
| `GainKnot` (`tMs`, `dB`), `GainCurveKind` (`gcKnots`, `gcPower`, `gcExponential`), `GainCurve`, `c.kind` | A gain against time. |
| `knotGain(knots)`, `powerGain(power, refMs = 1000, minMs = 1)`, `exponentialGain(dbPerS)` | Linear in dB between knots; `(t/refMs)^power` (2: spherical divergence); dB per second. Validated. |
| `c.gainDbAt(tMs)`, `c.factorAt(tMs)` | The gain at a time, in dB and as a factor. |
| `GainCurvePlan[T]`; `initGainCurvePlan(T, curve, dtMs, samples, tMinMs = 0, tMaxMs = NaN)` | The curve tabulated over `tMinMs .. tMaxMs` (NaN: the record length); times outside the table are exact but slower. `p.samples`, `p.dtMs`, `p.curve`, `p.tableLen`, `p.tableMs`. |
| `applyGainCurve(x, plan, startMs)`, `applyGainCurve(b \| samples, meta, plan)` | Multiply each live row by the curve at its own times (`t0Ms`, `firstSample`). |
| `applyGainCurve(b \| samples, meta, plan, ws)`, `p.workspaceSize` | The same for blocks that may lie off the table (a zoom window, a recording delay): the factors off it are computed once per run of rows with one time axis, not per sample (the pipeline uses this form). Results off the table differ from the other form by a rounding. |

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
| `acc.merge(other)`, `acc.clear` | Add another thread's share; zero it. |
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
| `checkGeometryInto(meta \| rows, dst, qc = GeometryQc())` | One per row (the meta form: live rows only). |
| `badGeometry`; `flaggedRows(checks, flags = badGeometry): seq[int]` | The default flags (no, zero or mismatched offset); the rows with any of them. |

## qc/firstbreak: first-break picking

| Proc | What |
|---|---|
| `PickMethod` (`pmStaLta`, `pmEnergyRatio`, `pmCoppens`) | The picker. |
| `FirstBreakParams` (`pickMethod`, `windowMs`, `ltaMs`, `stabilization`, `lowCutHz`, `highCutHz`, `slopeDbPerOctave`, `searchStartMs`, `searchEndMs`, `guideVelocity`, `guideT0Ms`, `beforeMs`, `afterMs`, `snrWindowMs`, `lowDb`, `highDb`) | Defaults set in the type: energy ratio, 20 ms window, no filter, no guide. |
| `FirstBreakPlan[T]`; `initFirstBreakPlan(T, params, dtMs, samples)`, `plan.params`, `plan.dtMs`, `plan.samples`, `plan.workspaceSize` | A compiled picker (optional minimum-phase band-pass, fixed or LMO-guided window). |
| `plan.requiredMargin` | The samples a zoom window must hold on each side of a row's search window to pick as the whole trace does, bit for bit: what the attribute and the confidence read before the window, plus the band-pass's margin. `plan.samples` for a plan that reads whole traces (no window, or `pmCoppens`). |
| `FirstBreakPick` (`timeMs`, `col`, `snrDb`, `confidence`); `noPick()` | One pick: its time, its column in the block (an int; −1 without a pick, as `timeMs` and `snrDb` are NaN), confidence 0 .. 1 from the SNR (`lowDb` .. `highDb`). `FirstBreakPick()` is `noPick()`. |
| `plan.pick(x, ws, offset = NaN, t0Ms = 0)`, `plan.pickInto(b \| samples, meta, ws, dst)` | Pick one trace, or every live row (others get `noPick`). The samples are not changed. With a search window, STA/LTA and the energy ratio read and band-pass only the window and its margins; zeros before a row's first and after its last non-zero sample are not data. |
| `checkPick(plan, b, meta, picks)`; `pickInto(pool, plan, b \| samples, meta, dst)` | The checks of `pickInto` (`ValueError`); the picks with the rows split across a `ProcessPool`'s workers (identical results). |

## qc/snr: signal-to-noise ratios

| Proc | What |
|---|---|
| `SnrEstimator` (`seRatio`, `seExcess`) | `Ps / Pn` (signal + noise over noise: noise alone ≈ 0 dB) or `(Ps − Pn) / Pn` (signal over noise, unbiased). |
| `WindowSnrParams` (`signalStartMs`, `signalEndMs`, `noiseStartMs`, `noiseEndMs`, `estimator`, `floorDb`) | The two windows in ms, both ends included; defaults: 0 .. 40 after a pick over −60 .. −20, `seRatio`. |
| `DefaultSnrFloorDb` | −40 dB: returned for no signal, and every SNR clamped up to it (`floorDb = -Inf`: no clamp). |
| `meanPower(x, cols)`, `powerSnrDb(signal, noise, estimator = seRatio, floorDb, stabilization = 0)`, `windowSnrDb(x, signal, noise: Slice[int], estimator, floorDb, stabilization)` | The kernels: mean power over columns (NaN when empty), dB from two mean powers, both on one trace. NaN for an empty window or a zero noise power. |
| `windowSnrInto(b \| samples, meta, params, [refMs \| guide: Moveout,] dst)` | The windowed SNR of every row, in dB: windows at fixed trace times, relative to a time per row (picks), or to a moveout guide at each offset (LMO). NaN for non-live rows, NaN references, empty windows. |
| `coherenceSnrWorkspace(cols)`; `coherenceSnrInto(b \| samples, meta, windowMs, ws, dst, floorDb = DefaultSnrFloorDb)` | The SNR of a flattened (NMO/LMO-corrected) gather per column, over a centred window: signal power over noise power per trace from the mean trace and the residuals (an unbiased estimator, the semblance relation). Rows count over their `liveSpan`. The time does not depend on the window; the scratch is `6·cols + 2` float64 (`ValueError` when `ws` holds less). |

## qc/shotfit: per-shot refraction fit

| Proc | What |
|---|---|
| `RefractionModel` (`rmTwoSegment`, `rmOffsetBinned`) | Two lines in \|offset\| (L1 fit), or a median per offset bin. |
| `ShotFitParams` (`model`, `minConfidence`, `binWidth`, `localTraces`, `maxResidualMs`, `madFactor`, `correlate`, `windowMs`, `maxLagMs`, `lagToleranceMs`, `minCoherence`) | What is checked; defaults set in the type. |
| `ShotFit` (`model`, `interceptMs`, `v1`, `v2`, `crossover`, `used`, `asymmetryMs`, `threshold`, `position`, `modelMs`, `residualMs`, `localMs`, `lagMs`, `coherence`, `suspect`) | One shot's fit; the per-row seqs are resized and reused. |
| `fitShot(meta, picks, params, fit)`, `fitShot(b \| samples, meta, picks, params, fit)` | Fit one shot's rows; flag traces whose residual jumps against their neighbours, and (with samples) the odd ones out by cross-correlation. `asymmetryMs` hints at a misplaced source. |

## qc/fold: fold maps of a survey

Data in: a parseSEG `SurveyGeometry` (R and S records for the grid; X or O relations for the pre-plot fold); traces as `ChunkMeta` or `TraceRows` for the actual fold.

| Proc | What |
|---|---|
| `surveyGrid(g, binInline = NaN, binCrossline = NaN, azimuthDeg = NaN, centre = true, trimStrays = true): Binning` | A `gridBinning` covering the midpoint of every source–receiver pair: inlines along `azimuthDeg` (default: the receiver-line azimuth, median bearing between consecutive stations), bins `binInline` along them (`gridBinning`'s `crosslineSpacing`; default half the median receiver interval) by `binCrossline` across (default half the median source interval: the natural bin of an orthogonal survey). Extent: half the sum of the source and receiver extents on each axis; numbers from 1. `centre`: bin centres at the midpoints' dominant phase per axis (trimmed circular mean of source and receiver phases, Rayleigh test; none significant: the minimum corner), within half a bin of the corner; `false`: the corner. `trimStrays`: stations outside quartile fences (±3·max(IQR, 10 bins) per axis) are left out of the extent, their pairs counted `outside`. `ValueError` for a bad given value, no stations with coordinates, or a default that cannot be measured. |
| `FoldOptions` (`offsetBounds`, `azimuthSectors`, `sectorStartDeg`, `offsetRange`) | What is counted besides the fold: offset classes (ascending edges, n edges → n − 1 classes `[e_k, e_k+1)`), source → receiver bearing sectors (sector k from `sectorStartDeg + k·360°/n`), each bin's offset range. |
| `FoldMap`; `initFoldMap(binning, opts = FoldOptions())` | Pairs per bin of a `Binning`, empty. `ValueError` for bad options. |
| `m.binning`, `m.options`, `m.fold` (`[inlines, crosslines]` int32), `m.offsetFold` (`[classes, inlines, crosslines]`), `m.azimuthFold` (`[sectors, inlines, crosslines]`), `m.offsetMin`, `m.offsetMax` (float32, NaN: empty bin), `m.binned`, `m.outside`, `m.noCoordinates`, `m.notLive`; `clear(m)` | The counts (tensors not asked for are empty); `clear` zeroes them, keeping the storage. |
| `preplotFold(g, binning, opts = FoldOptions(), domain = rdSource): FoldMap` | The fold of every X relation (O with `rdReceiver`): midpoint bin by `binAt`, station coordinates from per-line sorted arrays with a run cursor (no hash lookup per pair). `ValueError` for a header binning. |
| `addFold(m, meta \| rows)` | Adds the live traces (D12: live or unknown status, not a failed row), in any order and chunking: midpoint `cmpX/cmpY`, else half-way between `srcX/srcY` and `rcvX/rcvY` (parseSEG `applySurvey` fills them); offset from the coordinates, else `\|offset\|`; bearing from the coordinates only. Header binnings use `binOf`. No allocation; the rows overload raises `ValueError` without keys. |
| `foldDifference(preplot, actual): Tensor[int32]` | Pre-plot − actual per bin (`[inlines, crosslines]`): positive where traces are missing, negative where traces fall that the pre-plot has fewer of (geometry errors). `ValueError` for maps over different bins. |

## qc/record: field records against the XPS

Data in: a parseSEG `SurveyGeometry` (X records) and one field record's traces (`ChunkMeta` or `TraceRows`).

| Proc | What |
|---|---|
| `RecordMatch` (`rmAuto`, `rmSourceStation`, `rmFieldRecord`), `TraceMatch` (`tmAuto`, `tmStation`, `tmChannel`) | How the X record is found (header source station via `recordsAt`, the FFID choosing among repeats; or the FFID) and how traces match it (receiver station; or channel). Auto: by station when the headers have one. |
| `StationRange` (`line`, `firstPoint`, `lastPoint`, `count`), `LineCompleteness` (`line`, `expected`, `live`, `missing`) | Consecutive missing X stations on a line; one receiver line's counts. |
| `RecordCompleteness` (`fieldRecord`, `source`, `hasSource`, `recordMatch`, `traceMatch`, `xpsRecord`, `candidates`, `xpsSource`, `recordMismatch`, `sourceMismatch`, `traces`, `live`, `dead`, `aux`, `failed`, `expected`, `present`, `presentLive`, `unlisted`, `duplicates`, `channelMismatch`, `noStation`, `missing`, `lines`), `missingStations(c)`, `$` | One record: the X record (−1: none or ambiguous) with the other key checked (`recordMismatch`: X number ≠ FFID; `sourceMismatch`: X source ≠ header source), trace counts, station counts (multi-component channels count once), missing ranges, per-line counts; differing channel numbers are informational. |
| `recordCompleteness(g, meta \| rows, recordMatch = rmAuto, traceMatch = tmAuto)` | Hashes the data's stations once, walks the X record's stations once. Failed rows (meta) are present, not live. The rows overload raises `ValueError` without keys. |
| `CompletenessSummary` (`records`, `matched`, `unmatched`, `ambiguous`, `recordMismatches`, `sourceMismatches`, `xpsWithoutData`, `expected`, `present`, `presentLive`, `missing`, `unlisted`, `duplicates`), `summarize(g, records)` | Totals over many records, and the X records none matched. |
| `$`(StationKey) | `line/point`, `.index` when above 1. |

## qc/amplitude: amplitude QC of field records

| Proc | What |
|---|---|
| `AmplitudeFlag` (`afNonFinite`, `afDead`, `afConstant`, `afClipped`, `afSpiky`, `afDcOffset`, `afWeak`, `afNoisy`) | Why a trace stands out: NaN/Inf samples; every finite sample 0, or one non-zero value; two runs at the maximum (or one long one); samples far above their neighbours; a large mean; RMS far below or above the record's. |
| `AmplitudeParams` (`startMs`, `endMs`, `clipRun = 3`, `clipTolerance = 0`, `spikeFactor = 10`, `spikeHalfWindowMs = 20`, `dcRatio = 0.2`, `outlierZ = 3.5`), `validate(params)` | The window on the trace axis (default: the whole trace) and the thresholds: a clip is a run of `clipRun` samples at `≥ (1 − clipTolerance)·max` or `≤ (1 − clipTolerance)·min` (each polarity at its own level when both reach the maximum magnitude within 1 %), flagged from two runs or one of `2·clipRun`; a spike deviates from the mean by more than `spikeFactor` × its neighbours' RMS deviation within ±`spikeHalfWindowMs` (itself left out); DC: `\|mean\| > dcRatio·σ`; outliers: `\|robustZ\| > outlierZ`. Inf turns `spikeFactor`, `dcRatio`, `outlierZ` off. `validate` raises `ValueError`. |
| `TraceAmplitude` (`samples`, `rmsDb`, `maxDb`, `mean`, `nonFinite`, `clipped`, `spikes`, `robustZ`, `flags`) | One trace over its finite samples; `samples = 0` and NaN for a non-live row or an empty window (a finite window needs `t0Ms`). |
| `traceAmplitude(x, dtMs, params = AmplitudeParams())` | One trace's samples (the window is not applied); allocates its scratch. |
| `traceAmplitudeWorkspace(cols)`; `traceAmplitudeInto(b \| samples, meta, params, dst, ws)`; `traceAmplitudeInto(pool, b \| samples, meta, params, dst)` | Every row into `dst[row]`, the live samples in the window, with `2·cols + 1` float64 of scratch in `ws` (`ValueError` when it holds less); with a `ProcessPool`, rows split across its workers, each with its own workspace (identical results). No allocation. |
| `RecordAmplitude` (`traces`, `medianRmsDb`, `madDb`, `counts`, `flagged`), `recordAmplitudeWorkspace(traces)`, `recordAmplitude(traces, params, ws)` | One record: median and MAD of `rmsDb` over the traces with samples, a finite RMS and neither dead nor constant; each one's `robustZ = (rmsDb − median) / (1.4826·MAD)` and `afWeak` / `afNoisy` (set or cleared), counts per flag. No allocation. |

## pipeline: `Op`, `Pipeline`, `PipelinePlan`

| Proc | What |
|---|---|
| `OpKind` (`opDemean`, `opDetrend`, `opScale`, `opGainDb`, `opClip`, `opPolarity`, `opMuteTop`, `opMuteBottom`, `opFilter`, `opIir`, `opAgc`, `opNormalize`, `opGainCurve`), `Op` | One operation as plain data: `op.kind`, and its parameters by kind through `op.factor`, `op.dB`, `op.clip`, `op.mute`, `op.filter`, `op.iir`, `op.agc`, `op.norm`, `op.gainCurve`, read or edited in place (`flow[2].agc.windowMs = 300`). Asking for another kind's parameters raises `ValueError`, in every build mode. |
| `ClipParams`, `MuteParams`, `FilterParams`, `IirParams`, `AgcParams`, `NormalizeParams`, `GainCurveParams` | The parameter groups, edited in place (`flow[2].agc.windowMs = 300`). |
| `demeanOp()`, `detrendOp()`, `scaleOp(factor)`, `gainDbOp(dB)`, `clipOp(lo, hi)`, `polarityOp()`, `muteTopOp(curve, taperMs = 0)`, `muteBottomOp(curve, taperMs = 0)`, `filterOp(spec, trend = trLinear)`, `bandPassOp(corners, trend)`, `iirOp(spec, phase = ipZero, trend)`, `agcOp(windowMs, mode, target, epsilon)`, `normalizeOp(stat, scope, target, percentile)`, `gainCurveOp(curve, tMinMs = 0, tMaxMs = NaN)` | The builders; nothing is checked until `compile` (`bandPassOp` checks its corners). |
| `Pipeline`; `initPipeline(ops = [])`, `p.len`, `p[i]` (`[]`, also `var`), `p[i] = op` (`[]=`), `p.add(op)`, `p.insert(op, i)`, `p.delete(i)`, `p.clear`, `items`, `pairs` | An ordered, editable list of ops. |
| `PipelinePlan[T]`; `p.compile(T, dtMs, samples, recordSamples = 0)` | Validate and precompute every op for blocks up to `samples` wide; `ValueError` names the op (`op 1 (opFilter): ...`). `recordSamples`: the trace length when `samples` is a zoom window's width, so that gain-curve tables cover the whole record (`plan.recordSamples`). |
| `plan.generation`, `plan.dtMs`, `plan.samples`, `plan.recordSamples`, `plan.len`, `plan.requiredMargin`, `plan.workspaceSize`, `ws.reserve(plan)` | A plan's id (never 0, increasing), size, zoom margin (sum of the ops'), scratch. Any block up to `samples` wide works. |
| `p.requiredMargin(dtMs)`, `decodeWindow(shown, margin, traceLen)` | The margin without compiling; the samples to decode for a zoom (shown ± margin, clamped). |
| `plan.apply(b \| samples, meta, ws)`, `plan.apply(src, dst, meta, ws)` (blocks or tensors) | Run in place, or `src → dst` with the copy fused into the first op; tiles of about 256 KB. Never grows `ws`; `dst` is not resized. |
| `checkApply(plan, src, dst, meta)`, `plan.partCount`, `plan.partNeedsPanel(part)`, `plan.panelFactorOf(part, b, meta, ws)`, `plan.runPart(part, src, dst, meta, factor, ws)` | The pieces of `apply`, for a caller's own threads: a per-panel normalization splits the plan into parts. |

## core/pool: `ProcessPool`

| Proc | What |
|---|---|
| `ProcessPool`; `newProcessPool(workers = 4)`, `p.workers` | Persistent worker threads, each with its own `Workspace`. `ValueError` for `workers < 1`, `ResourceExhaustedError` when the OS refuses a thread. `=copy` and `=dup` are compile errors (move it); `=destroy` stops and joins the workers; `=wasMoved` empties it. |
| `pool.apply(plan, b \| samples, meta)`, `pool.apply(plan, src, dst, meta)` (blocks or tensors) | `plan.apply` with the rows split across the workers; blocks until done, identical results. One job at a time; a second caller waits. |
| `pool.ping` | An empty job: the per-job overhead. |
| `correlateInto(pool, plan: CorrelationPlan[T], src, dst: RowBlock[T] \| Tensor[T], meta)` | Vibroseis correlation with the rows split across the workers; checks as `checkCorrelate` on the caller's thread. |
| `traceAmplitudeInto(pool, b \| samples, meta, params, dst)` | Amplitude QC (`qc/amplitude`) with the rows split across the workers, each reserving `traceAmplitudeWorkspace`. |
| `pickInto(pool, plan: FirstBreakPlan[T], b \| samples, meta, dst)` | First-break picking (`qc/firstbreak`) with the rows split across the workers; checks as `checkPick` on the caller's thread. |

## adapters: metas from parseSEG reads

| Proc | What |
|---|---|
| `intervalMs(rows: TraceRows)` | The rows' common interval from their keys' `intervalUs` (`ValueError` when unknown, 0 or mixed). |
| `chunkMeta(c: PrefetchChunk[T], dtMs = NaN)` | The meta of a prefetch chunk: its rows, interval, `firstSample` and `failedRows`, in place. `dtMs` overrides headers without an interval. |
| `chunkMeta(f, traces: Slice[int] \| openArray[int], [samples: Slice[int],] failed = [], dtMs = NaN)` | The meta of a `readTracesInto` of a range or list (and sample window), with its `failedRows` list. |
| `pilotFromTrace(f, trace, T = float64, t0Ms = NaN, dtMs = NaN): Pilot[T]` | A recorded pilot (aux sweep or ground-force trace) from a file: samples, interval and start time from its keys, `sweep` from the SEG-Y sweep fields (trace header, else binary) when they describe a linear or exponential sweep that the trace can be: valid at its interval, no longer than it, and reaching the upper half of its band in the samples (`energyAbove`; some controllers write 0.1 Hz). |

## vibroseis/taper: sweep-end tapers

| Proc | What |
|---|---|
| `TaperShape` (`tpLinear`, `tpCosine`, `tpCos2`, `tpHamming`, `tpBlackman`, `tpBlackmanHarris`, `tpKaiser`, `tpGaussian`, `tpPower`) | Ramp shapes, each the rising half of its window. |
| `SweepTaper` (`startMs`, `endMs`, `shape`, `param`); `initSweepTaper(startMs, endMs, shape = tpCos2, param = NaN)`, `noTaper()` | Start and end ramps in ms (0 or NaN: none); `param` is β, σ or the exponent, NaN by default also in an object built by hand: the shape's 6, 0.4 or 2. |
| `taperWeight(shape, x, param = NaN)` | The ramp `w(x)`, `x` in 0 .. 1. |
| `applyTaper(x: var openArray[T], dtMs, taper)` | Applies the ramps in place to a sweep of `x.len` samples (`(x.len − 1)·dtMs` ms). |
| `validate(taper)`, `checkFits(taper, sweepMs)` | `ValueError` for bad lengths or parameters, or ramps that overlap (`startMs + endMs > sweepMs`; meeting is allowed). |
| `segyTaperType(shape \| taper)` | SEG-Y taper type: 1 linear, 2 cos², 3 other. |

## vibroseis/sweep: sweep generation

| Proc | What |
|---|---|
| `SweepLaw` (`slLinear`, `slDbPerOctave`, `slDbPerHz`, `slExponential`, `slTable`), `SweepPoint` (`tMs`, `fHz`) | How the frequency runs; a table point. |
| `SweepSpec` (`law`, `f1Hz`, `f2Hz`, `lengthMs`, `dtMs`, `boostDb`, `table`, `startPhaseDeg`, `amplitude`, `taper`, `maxNyquistFraction`) | A sweep. |
| `linearSweep(f1Hz, f2Hz, lengthMs, dtMs, taper = noTaper(), startPhaseDeg = 0, amplitude = 1, maxNyquistFraction = 0.8)`; `dbPerOctaveSweep(..., dtMs, boostDb, ...)`, `dbPerHzSweep(..., dtMs, boostDb, ...)`, `exponentialSweep(...)`, `tableSweep(points, dtMs, ...)` | Validated specs. A dB boost is of the amplitude spectrum, per octave or per Hz, and may add up to 300 dB over the band per octave, 150 dB per Hz. |
| `validate(spec)` | `ValueError`: frequencies ≤ 0, above `maxNyquistFraction`·Nyquist or Nyquist, a length under one sample, a bad table, boost or taper. |
| `sweepLen(spec)` | Samples: `lengthMs / dtMs + 1` (both ends); 0 for a spec whose interval or length `validate` rejects. |
| `sweepSamples(spec, T = float64): seq[T]`, `sweepInto(spec, x: var openArray[T])` | The sweep, `amplitude·taper·sin(φ₀ + 2π∫f)`. |
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

## vibroseis/correlate: correlation and sweep deconvolution

| Proc | What |
|---|---|
| `CorrelationMode` (`cmCorrelate`, `cmDeconvolve`) | `X·conj(P)`, or `X·conj(P)/(\|P\|² + ε)`. |
| `initCorrelationPlan(T, pilot, recordSamples, dtMs, outSamples = 0, mode = cmCorrelate, epsPercent = 1, normalize = true)` | A plan for records of up to `recordSamples`; `outSamples` 0 is the listen time `recordSamples − pilot.len + 1`. `ValueError` for a pilot at another `dtMs` (use `resamplePilot`), no listen time, a bad `outSamples` or `epsPercent`. |
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

## continuous/slice: continuous recordings into records

| Proc | What |
|---|---|
| `SliceShot` (`time`, `id`, `srcLine`, `srcPoint`, `srcX`, `srcY`, `srcZ`); `initSliceShot(time, id, srcLine = NaN, ...)` | A source event: UTC POSIX seconds and record number. |
| `SliceWindow` (`startMs`, `lengthMs`) | The window at each shot: start relative to the shot (may be negative), length a whole number of samples. |
| `SliceOrder` (`soReceiverGathers`, `soShotRecords`), `SliceAlign` (`saNearest`, `saLinear`, `saCubic`) | Output order; sub-sample window starts kept (nearest sample) or shifted out with `core/shift`. |
| `SlicePlan`; `planSlices(meta, [times, [receivers,]] shots, window, order = soReceiverGathers, align = saNearest)` | Lays out every (receiver, shot) trace from segment rows. Times default to `TraceKeys.time`, receivers to (`rcvLine`, `rcvPoint`, `channel`, `component`); segments are never stitched across receivers. `ValueError` for a non-whole window, a non-finite shot time, mismatched array lengths, or a row of another interval. |
| `plan.len`, `plan.samples`, `plan.dtMs`, `plan.margin`, `plan.order`, `plan.align`, `plan.window`, `plan.shotCount`, `plan.receiverCount`, `plan.shotOf(trace)`, `plan.receiverOf(trace)`, `plan.receiverKeys(receiver)` | Its shape and trace layout. |
| `SlicePiece` (`row`, `first`, `dst`, `count`); `pieces(plan, trace)`, `plan.rowsNeeded`, `plan.maxColumn` | The rows and columns each trace reads (`row = -1`: a gap); the only rows to read. |
| `plan.fraction(trace)`, `plan.gapSamples(trace)`, `plan.untimed` | Sub-sample remainder of the window start (−0.5 to 0.5); uncovered samples; live rows without a finite time. |
| `SliceOverlap` (`earlier`, `later`, `first`, `samples`), `plan.overlaps`; `SliceTimingIssue` (`earlier`, `later`, `residual`), `plan.timingIssues` | Overlaps (the later segment wins, always); segments off the previous one's sample grid (tolerance max(0.01 sample, 1 µs)). |
| `sliceKeys(plan, trace)` | The output trace's `TraceKeys` (D14): receiver keys, shot record and source, `time` = shot time, `t0Ms` = `startMs`. |
| `sliceWorkspace(T, plan)`, `SliceReport`, `initSliceReport(plan)` | Scratch (only with interpolation) and a report sized for the plan. |
| `extractInto(plan, data, meta, dst, report, ws)` (blocks or tensors) | Copies and stitches into `dst` (`plan.len × plan.samples`) from all rows or only `rowsNeeded`; gaps zero. Allocates nothing. |
| `report.len`, `report.gapSamples(trace)`, `report.gapCount(trace)`, `gaps(report, trace)` (`SliceGap`: `first`, `count`) | Zero-filled column runs per trace, failed rows included. |
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
| `generateRelations(g, [layout,] shots, template): seq[SpsRelation]` | X records: per shot the template lines nearest the source (not short), on each the station slots around it (one slot per `pointStep`; a slot missing from R is a hole), plus short-line stations inside the patch; channels 1..n by ascending line and point, one record per run of consecutive stations. Distance limits alone take every R station inside (short lines too); with a template, its receivers inside. A shot whose source has no coordinates gets none (`ppCoordinates`, or distance limits). `peShift` moves a distance box inside the receivers' extent (bounding box in the box frame); it takes a template or distance limits, not both. `ValueError` for neither (roll, shift), an empty template, shift with both, a non-finite offset or map, a negative or NaN limit. Write with parseSEG `writeSps`; O records with `receiverRelations`. |
| `RecordAgreement` (`record`, `listed`, `generated`, `common`, `missing`, `extra`), `jaccard(a)`, `listedFraction(a)` | One record in two relation sets, by distinct receiver stations; `missing`: listed, not generated; `extra`: generated, not listed (receivers dropped in the field). |
| `compareRelations(listed, generated: SurveyGeometry): seq[RecordAgreement]` | Per X record of either, ascending (index 0 and 1 alike). |
