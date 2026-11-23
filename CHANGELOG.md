# Changelog

## 0.11.0

- `vibroseis/matchpeak`: the correlation envelope's peak, its time shift and the phase turn there. `MatchVerdict`, `Match`, `peakAt`, `matchOf`, `rotationOf`, `rotationOfInto`; constants `NoClearMatch`, `ClearPeakToSidelobe`, `SidelobeGapMs`, `MinSidelobeSpanMs`, `PreciseShiftPeakToSidelobe`, `PreciseShiftMatch`. Parallel arrays of different length raise `ValueError`.
- `qc/breakfit`: the robust start and trimmed model of first-break picks. `median`, `robustSigma`, `lineFit`, `medianIfCloser`, `medianLineStart`, `trimmedModel`, `keepSecond`, `ModelFn`, `Trimmed`; constants `ConfidentPick`, `MadToSigma`, `TrimSigmas`, `TrimFloorMs`, `TrimRounds`, `MinTrimmedPicks`, `StartLines`, `MinSecondPicks`, `SecondGain`, `SecondGainMs`, `RestartSigmaShare`, `RestartSigmaFloorMs`.
- `qc/exclusion`: noisy and weak traces of a gather outside an exclusion area around its common point (the source of a shot, the receiver of a receiver gather), where the loud near-source traces are not judged noisy. `noiseQc` (with and without the caller's `ShotFit`) sets `robustZ`, `afNoisy` and `afWeak` and fills a `NoiseResult` that always states which rule set the area (`arModel`: the refraction model; `arRadius`: offsets from a survey or header coordinates; `arSteps`: a station count either side of the source on the lines the shot reaches; `arNone`; the automatic chain is model, radius, steps, none), why the earlier ones did not apply (`reasons`), the edge (the level rule: the arrival where the level falls within `levelDb` of the far traces'), and per trace `state`, `inside`, `arrival`. Overrides: `acFixedTime` (on the model), `acFixedRadius`, `acSteps` (default 8, `stepsOnAllLines`), `acOff`, and `acTrend` (`arTrend`, experimental, never chosen automatically). The model is refused when it is not physical (`rnBadModel`: velocities not positive or not finite, times falling with offset); a record whose far ends are all at one station has no area (`rnOneStation`), and a line of one station no steps; `rnOverrideNeedsModel` replaces `rnOverrideNeedsPicks`. Weak is judged everywhere. A second finding, louder than the neighbours on the same line (`louder`, `neighbourDb`, `neighbourState`), and a note `earlyBreak`. The hydrophone is its own group; unrotated node geophones are judged per node (`partial` for 2 of 3 components, `nsIncompleteNode` below), rotated ones per component; each group's level (`groups[].medianDb`) is returned for comparing the gathers of a folder. `traceRolesInto` (sensor class, group, node, line and station from the keys: header lines, else coordinates, else station numbers), `gatherDomainOf`, `validate`, `noiseWorkspace`. The experimental trend is a valley fit per line (earliest at the station nearest the source, not decreasing away from it on either side; `trendTraces` is now the stations without a pick it bridges): the most picks consistent with one apex within pick noise, a burst run at the apex cut off as a notch, the apex of a line in doubt by a few picks taken nearest the other lines' vote, and each flank's least-absolute-deviation monotone fit. It follows the picks within their noise at any moveout per station, and the area is one run of stations around each line's apex, so burst runs far from the source are judged; a piece with consistent picks for under half of its picks (random picks) has no trend. Experimental, with known failures: a late confident noise pick at a flank's outer end counts as consistent, so the trend ramps up to it and that line's near traces fall outside; a run of 6 or more burst picks just past the last visible break can win the flank and have the true apex cut; the vote between lines trusts station numbers that need not run along the line. The edge's level maximum is searched in the earlier half of the arrivals. Units tied with the edge are inside unless the whole group is tied. A header offset of 0 counts as unset unless every row carrying it is at one identified far-end station (a real 0); a multi-component record of unknown sensor classes is judged per station on the summed energy of its traces, never as one group (`TraceRole.guessed`); `AreaReason` also says when a group had too few arrivals for an edge, when the area covered a whole group (weak is then judged against all of it), when the sensors were unknown and when stations could not be identified (unrotated motion rows with no station identity are not judged: `nsUngrouped`, `nbUngrouped`), and that the model had no header offsets (`rnNoHeaderOffsets`, apart from `rnNoOffsets` for the radius). `groups[].medianDb` and `madDb` are NaN when the area covered a group. `earlyBreak` is also set inside the area, where a trace's own pick is earlier than its arrival by more than pick noise. `qc/amplitude` is unchanged.

## 0.10.0

- `vibroseis/recfilter`: a reference through a recorder's filter with its bulk delay backed out. `RecordingFilter` (`rfkNone`; `rfkLinearPhase` with `taps`, cutoffs and roll-offs; `rfkMinimumPhase` with typed `coefficients` or a Butterworth/Chebyshev `family`, orders, cutoffs and `rippleDb`; `shiftSamples`, whole samples), `applyRecordingFilter` (`rfkNone` returns the pilot bit for bit; the length and `t0Ms` are kept), `checkFilter`, `linearPhaseTaps`, `linearDelaySamples`, `responseAt`, `bulkDelaySamples`, `suggestedShift`, `shiftResidualDeg`. Nothing is read from headers: every parameter is the caller's. Typed `coefficients` replace the design under either kind; `linearShiftSamples`, `coefficientsSymmetric`, `minRippleDb`. A non-finite output and a ripple below 0.001 dB are refused.

## 0.9.0

- Behaviour change, one rule for lengths: a length in time is N × dt (12000 ms at 1 ms is 12000 samples, the first at 0 ms, the last at 11999 ms). `samplesInLength(lengthMs, dtMs)` and `lengthOfSamples(samples, dtMs)` are the shared helpers (a length that is not a whole multiple of dt is cut down to whole samples).
- `sweepLen` is `lengthMs / dtMs` (was `+ 1`): a sweep of 8000 ms at 2 ms has 4000 samples, not 4001; the end taper's ramp ends at `lengthMs`, one interval after the last sample. `Pilot.lengthMs` is `len · dtMs` (was `(len − 1) · dtMs`); `resamplePilot` gives `lengthMs / dtMs` samples. `applyTaper` takes the sweep to last `x.len · dtMs`.
- `initCorrelationPlan`: the default `outSamples` is the listen time, `recordSamples − pilot.len` (was `+ 1`).


## 0.8.0

- `TrackParams.harmonicBandHz`: THD over the harmonics within a band (`k·f ≤` the edge), as a vibrator controller's distortion; `trackedHarmonics`, `topFrequencyHz` (an estimate of the reference's top frequency).
- `TrackParams.maxFundamentalFraction` (default 0.8, as before); `SweepTrack.fundamentalCut` and `lastFundamentalMs` say where the fundamental stops being measured.
- `TrackParams.harmonicBandAuto` sets the band at the reference's top frequency inside `trackInto` (one pass); `SweepTrack.bandHz` is the edge used. In-band THD is NaN where harmonics are in band but none was measured (0 only where none is in band), and `lkHarmonic` judges the in-band harmonics only. `topFrequencyHz` is an estimate (a chord fit: about 0.4 % on a clean reference with tapers to 0.5 s; wrong with noise after the sweep, tapers over 1 s, short steep exponential sweeps or a cut reference): give `harmonicBandHz` from the sweep's design for QC. `largestChordHz` is a lower bound of the top.
- Behaviour change: `maxHarmonicFraction` no longer limits the fundamental (it follows `maxFundamentalFraction`, 0.8 by default): a caller that set a non-default `maxHarmonicFraction` gets the fundamental measured to 0.8 of Nyquist regardless.
- `windowStats` takes `originMs` (default the extent's start).
- `windowStats`, `WindowStats`, `WindowSummary`: per-window (500 ms) mean phase, force, peak and THD and their average and worst window over the sweep.
- `sweeptrack`: the peak interpolator has 16 taps each side (was 8): a pure fundamental reads within 3e-4 up to 0.8 of Nyquist (it read up to 4 % low from 0.7).

## 0.7.1

- `analyticInto`: the instantaneous frequency was wrong above a quarter of the sampling rate (it read the true value minus fs/2); the phase step is now taken per sample (circular mean of the two steps around the sample), so it is right up to Nyquist for a clean tone.
- `circularMeanDeg`: the circular mean of angles in degrees and the resultant length.

## 0.7.0

Exports added since 0.6.0:

- `timeShiftInto`, `timeShift`: a trace shifted by a whole number of samples, length kept. `wholeSamples` snaps milliseconds, round to nearest, ties away from zero.
- `FkAxes`, `FkPlan`, `initFkPlan`, `axes`, `nFreq`, `nK`, `dfHz`, `dkPerTrace`, `fkRows`, `fkAmplitudeInto`: f-k amplitude of a gather.
- `analyticInto`: envelope, phase and instantaneous Hz from one Hilbert.
- `averageWorkspace`, `averageSpectrogramInto`: a spectrogram averaged over traces.
- `unwrapPhaseInto`, `removeDelayInto`: phase unwrap and linear-delay removal. `SpectrumAccumulator`, `initSpectrumAccumulator`, `used`, `clear`, `amplitudesInto`: a power sum of spectra.
- `heldGain`: a gain curve held flat after a time.
- `appliedResponseInto`: the response of a flow as applied.
- `traces`: the trace count of an f-k plan. `samples`, `dtMs`, `workspaceSize`: overloads for `FkPlan`, and `dtMs` and `workspaceSize` also for `SpectrumAccumulator`.
- `bins`, `windowLen`, `nfft`, `binHz`: overloads for `SpectrumAccumulator`, as for `SpectrumPlan`.
- `add`: overloads for `SpectrumAccumulator`, adding a row block or a tensor.
- `toDbInto` gained a defaulted `refAmplitude = 1.0` parameter: with `dbAbsolute`, 0 dB is that amplitude. Calls that do not pass it compile unchanged.
