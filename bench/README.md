# Benchmarks

`nim bench` builds and runs every `bench/b*.nim` with
`-d:danger --opt:speed --debuginfo:on --linedir:on` (plus `-march=x86-64-v3` on x86-64), and
each run prints the machine, compiler and flags above its numbers. `benchutil.nim` is the
harness: warm-up, then 15 samples of at least 20 ms each (`-d:benchSamples`,
`-d:benchWarmup`, `-d:benchMinSampleMs`), reporting the median and minimum per iteration
and GB/s.

After each run of a body the harness puts a compiler memory barrier (`clobberMemory`).
Without it, gcc merged the repeated runs of an inlined in-place loop and reported twice the
real bandwidth (scale in place: 140 GB/s instead of 75). A body that keeps all its results
in locals also needs to write them somewhere the compiler must assume is read.

Wall-clock numbers move by a few percent between runs: compare medians from several runs,
and use cachegrind (`Ir` counts) to judge small changes.

## Recorded results

Each entry gives the date, the machine line and the numbers, as the benchmark prints them.
The last full `nim bench` run is from 2026-11-13 (after processSEG 0.3.0, on parseSEG
0.10.0, on a machine with nothing else running): every table of current numbers below
comes from it. Sections that record how a kernel got faster keep their earlier numbers,
with the 2026-11-13 figure added.

### bmemory: memory baseline, 2000 × 2001 float32 (16 MB per buffer)

2026-11-13, AMD Ryzen 9 3950X (32 threads), linux/amd64, Nim 2.4.0, gcc 12.2.0,
`-d:danger --opt:speed --debuginfo:on --linedir:on -march=x86-64-v3`:

| kernel | median | GB/s (median) |
|---|---|---|
| copy (copyMem) | 590 µs | 54.2 |
| scale in place | 325 µs | 98.4 |

The working set (16–32 MB) sits largely in the 3950X's L3, so these are cache rather than
DRAM bandwidths: the ceiling that a single-pass panel kernel on one thread can reach.

### bfft: real FFT, 2000 traces, forward + inverse (task 2.1)

Target: forward + inverse over a 2000-trace float32 panel at n = 2048 in ≤ 20 ms on one
thread. **Met: 4.61 ms** (11.4 ms when first met, 2026-11-10).

2026-11-13, AMD Ryzen 9 3950X (32 threads), linux/amd64, Nim 2.4.0, gcc 12.2.0,
`-d:danger --opt:speed --debuginfo:on --linedir:on -march=x86-64-v3`:

| case | panel (median) | per trace and transform | GFlop/s (2.5·n·log₂n) |
|---|---|---|---|
| f32 lanes n = 2048 | 4.61 ms | 1.15 µs | 48.9 |
| f32 lanes n = 2025 (odd) | 10.8 ms | 2.69 µs | 20.7 |
| f32 lanes n = 4096 | 10.3 ms | 2.59 µs | 47.5 |
| f32 lanes n = 1000 | 2.33 ms | 0.58 µs | 42.8 |
| f64 lanes n = 2048 | 9.16 ms | 2.29 µs | 24.6 |
| f32 one trace n = 2048 | — | 4.51 µs | — |

How it got there (each step measured):
- first version 43.2 ms: every butterfly loop was scalar. The radix-4/5 passes stored to
  four or five offsets of one output pointer, and GCC needs more runtime alias checks
  than its limit (10) to vectorize that. Each butterfly is now an inline kernel with every
  stream as its own `noalias` pointer: 16.0 ms;
- cachegrind then put 64 % of instructions in the real-FFT pack/split loops (scalar: a
  runtime lane count and a seq field reloaded per iteration). Static lanes and `noalias`
  kernels there: 11.4 ms;
- Phase 12 (12.1): the work arrays a fixed phase apart in the L1/L2 ways (`workStride`),
  fixed-width lane loops and the workspace scratch on a cache line: 4.61 ms.

An odd length runs a full-length complex FFT, so `nextFastLen` returns even lengths by
default (2001 → 2048, not 2025). Per point, 1000, 2048 and 4096 now run at about the
same rate (43–49 GFlop/s): with the work arrays staggered, buffers that leave L1 no longer
cost a power-of-two length its speed.

### Long records (60 s at 1 ms)

Same machine and flags, 200 traces, forward + inverse, 2026-11-13:

| case | per trace and transform | ns per n·log₂n |
|---|---|---|
| f32 lanes n = 65536 (60001 samples padded to a power of two) | 56.5 µs | 0.054 |
| f32 lanes n = 60000 (2⁵·3·5⁴) | 54.2 µs | 0.057 |
| f32 one trace n = 65536 | 165 µs | 0.157 |

A power of two now costs what the 2·3·5 length near it costs (it was 3× slower: 233 against
74 µs, 2026-11-10), and the lanes are 3× faster than one trace at this size. At 131072
points a transform costs 123 µs per trace (12.1; 2219 µs before).

Before 12.1 (2026-11-10), forward only, ns per n·log₂n and trace, from a sweep (5 samples
each):

| n | lanes | one trace |
|---|---|---|
| 32768 | 0.17 | 0.24 |
| 61440 (2¹²·15) | 0.38 | 0.21 |
| 60000 | 0.08 | 0.17 |
| 65536 | 0.22 | 0.23 |
| 122880 | 0.59 | 0.23 |
| 131072 | 1.73 | 0.80 |

- Then, from m ≈ 30k complex points the 8-lane work buffers (4·m·8·4 B ≥ 4 MB) left the
  cache and batching stopped paying, and lengths with many factors of 2 aliased in cache
  (Stockham strides of 2^k): 61440 and 131072 were the worst. The staggered work arrays
  of 12.1 removed the aliasing, so a cache-blocked (four-step) FFT is not needed up to 2^17.
- A 60 s record at 1 ms (65536) costs about 56 µs per trace forward in the lanes, so a
  1000-trace spectrum average takes about 60 ms on one thread.

## Filters (task 3.4)

`nim bench` (bench/bfilter.nim), same machine and flags as above. A panel is 2000 traces ×
2001 samples of float32 at 2 ms, filtered in place (live rows, edge handling included).
"4 workers" is 4 threads started once, 500 rows each, shared read-only plan, one
`Workspace` each: the wall time per round. The pool's hand-off (5.2) is not included.

2026-11-13:

| filter | L (pad) | nfft | 1 thread | 4 workers | speedup |
|---|---|---|---|---|---|
| FFT band-pass cos² 4-8-60-80 Hz | 110 | 2250 | 11.9 ms (5.95 µs/trace) | 3.15 ms | 3.78× |
| FFT low-cut cos² 0.5–1 Hz | 881 | 3840 | 17.7 ms (8.86 µs/trace) | 4.62 ms | 3.84× |
| IIR zero-phase Butterworth 4–60 Hz, 24 dB/oct (4 poles) | 99 | — | 10.8 ms (5.38 µs/trace) | 2.79 ms | 3.85× |

Earlier (task 3.4, after the 3.5 review's per-edge L): 23.6 / 42.2 / 25.0 ms on one
thread, 6.2 / 11.5 / 6.5 ms on 4 workers, with nfft 2500 and 5760. Since then: the FFT
itself (12.1), lane-wise pack/unpack and 8-accumulator trend sums (12.2), the FFT size
`samples + 2L` when the response rejects 0 Hz (zero pads, 12.3) and the IIR on 24 lanes.

Where the FFT band-pass went before 12.2 (cachegrind, instructions): the FFT passes about
45 %, the strided pack/unpack (with its float64 conversions) about 30 %, the trend sums
about 15 % (scalar float64: a float sum keeps its IEEE order), the gain multiply about 6 %.

**Viewer budget** (set from these numbers): one filter over a 2000 × 2001 panel costs
about **3 ms on 4 workers** (4.6 ms for a narrow 0.5 Hz low-cut ramp), so a filter plus AGC
fits a 60 Hz frame (16.7 ms) with room for the draw; on one thread it takes 11–18 ms, about
a frame, so a viewer without the pool should filter off the draw thread. Cost scales with traces ×
nfft·log(nfft): halve the panel, halve the time.

## Gain (Phase 4)

`nim bench` (bench/bgain.nim), same machine and flags as above, one thread, a panel of
2000 traces × 2001 samples of float32 (16 MB), 2026-11-13.

| operation | panel (median) | GB/s |
|---|---|---|
| `traceStatsInto` (RMS and max per trace) | 0.83 ms | 19.2 |
| `panelStats`, no clip | 0.83 ms | 19.3 |
| `panelStats`, p99 clip (histogram) | 4.3 ms | 3.7 |

- The statistics are two vectorized passes per row (Σx² in eight float64 accumulators,
  then max |x| as an integer max of the bit patterns). One fused loop with a float max
  stayed scalar in `-d:danger` (1.7 ms); tvectorized builds in `-d:release`, where it was
  packed, so check danger builds by hand too.
- The percentile clip is a scatter into a 65536-bin histogram, about 3 cycles a sample;
  four interleaved sub-histograms took it from 5.1 to 4.2 ms, and the 4.4 review's single
  row filter for both passes (Σx² again) to 4.5 ms. It is a value to recompute
  when the data or the selection changes, not per frame.

AGC and normalization (same run, 1 thread):

| operation | panel (median) |
|---|---|
| `agc` RMS, 100 ms window | 4.5 ms |
| `agc` RMS, 500 ms window | 5.1 ms |
| `agc` RMS, 2000 ms window | 6.8 ms |
| `normalize` per panel, p99 | 6.0 ms |
| `applyGainCurve` (knots, t0 on the grid) | 0.48 ms (67 GB/s) |

- AGC is O(n). Each window sum is a block suffix plus a block prefix (no subtraction, for
  precision); the block sums are built two blocks at a time (four independent float64 add
  chains, 12.3), a serial loop that is most of the time. The window matters only through
  the windows cut by the trace ends: a 2000 ms window is half of a 4 s trace.
- How it got there: 11.9 / 17.2 / 28.2 ms at first (the cut windows at the trace ends ran
  a scalar path with integer `div`/`mod`); closed forms per range, 9.3 / 11.1 / 16.3;
  int32 counts (int64 → float64 has no packed form on AVX2), 9.2 / 10.5 / 13.3; the gain
  step in float32 on level-scaled means, 6.5 / 7.8 / 10.6; ascending affine indices in
  every range (no gathers), 6.3 / 6.7 / 6.7; two blocks per iteration of the block sums
  and the 2026-11-13 run, 4.5 / 5.1 / 6.8.

## Process pool (task 5.2)

`nim bench` (bench/bpool.nim), same machine and flags. The pool is created once, outside
the timing. "ping" is an empty job on every worker (`pool.ping`); "empty plan" is
`pool.apply` of an empty plan on one row, in place (argument checks included). 2026-11-13:

| workers | ping | empty plan, 1 row |
|---|---|---|
| 1 | 6.5 µs | 6.8 µs |
| 2 | 6.8 µs | 6.3 µs |
| 4 | 6.9 µs | 7.1 µs |
| 8 | 12.5 µs | 13.4 µs |

So a job costs a wake-up and a join, microseconds, against milliseconds of work.

A viewer pipeline, demean + band-pass 4-8-60-80 + AGC 500 ms, on a 2000 × 2001 float32
panel, src → dst (the copy fused into the first op):

| run | time per panel | speedup |
|---|---|---|
| `plan.apply`, 1 thread (no pool) | 18.7 ms | 1 |
| pool, 1 worker | 18.7 ms | 1.00× |
| pool, 2 workers | 9.6 ms | 1.94× |
| pool, 4 workers | 5.1 ms | 3.66× |
| pool, 8 workers | 2.9 ms | 6.46× |

(Task 5.2: 32.2 ms on one thread, 8.6 ms on 4 workers, 4.3 ms on 8.) The tiled `apply`
(every op on ~256 KB of rows while it is in cache) costs about what the ops cost one after
another over the whole panel (11.9 ms band-pass + 5.1 ms AGC + demean). With the faster
kernels the 8-worker run scales less (6.5×, was 7.4×): its 2.9 ms is closer to the
per-job and memory costs.

## Pre-stack, QC and stacking (task 9.2)

`nim bench` (bench/bprestack.nim), same machine and flags as above, one thread, a split-spread
shot of 2000 traces × 2001 samples of float32 at 2 ms (16 MB): a refraction and a reflection
over noise.

2026-11-11, AMD Ryzen 9 3950X, Nim 2.4.0, gcc 12.2.0,
`-d:danger --opt:speed --debuginfo:on --linedir:on -march=x86-64-v3`:

| operation | before 9.2 | after | 2026-11-13 |
|---|---|---|---|
| `applyStatics` linear / cubic | 6.6 / 14.9 ms | 6.1 / 15.5 ms | 1.90 / 2.74 ms |
| `applyLmo` + `removeLmo` | 10.8 ms | 10.3 ms | 0.81 ms |
| `applyNmo` linear | 36.3 ms | 17.5 ms | 7.64 ms |
| `applyNmo` cubic | 54.5 ms | 24.7 ms | 9.76 ms |
| `addLive`, 2000 traces into 200 bins | 3.5 ms (13.8 GB/s) | 3.4 ms | 1.03 ms (46.6 GB/s) |
| `pickInto` STA/LTA, MER, Coppens | 20.0 / 6.0 / 10.5 ms | 18.9 / 5.3 / 9.8 ms | guided: next section |
| the same with a 5–60 Hz band-pass | 100.6 / 86.5 / 91.0 ms | 44.1 / 30.7 / 35.3 ms | guided: next section |
| `fitShot` on picks (two-segment, binned) | 38.1 / 38.2 ms | 20.7 / 20.8 ms | 19.9 / 20.1 ms |
| `fitShot` with cross-correlation | 45.1 ms | 27.6 ms | 26.7 ms |

Phase 12 (review xF) reworked these kernels: `shiftSamples`, which statics and LMO use,
interpolates the interior in vectorized blocks; the NMO kernel; the stack's finite check as
a bit test.

What changed (each measured, cachegrind first):
- **First-break band-pass:** `pickInto` filtered each row with the one-trace IIR path
  (`sosSingle`, scalar: 13 % of the instructions of a 400-trace profile). It now filters
  `fftLanes` rows at a time through the block filter (the lanes kernel, `sosLanes`) into
  workspace copies, then picks each copy; a plan without a band-pass picks from the rows in
  place (no copy). The filter costs about 25 ms of the panel now, as the IIR block filter does.
  The workspace grows to `fftLanes` copies of a trace for a filtered plan.
- **NMO:** the RMS-velocity column is computed once for rows that share a start time (and,
  with a field, a location), not per row (11 % of the profile); the input columns
  (`sqrt`, two divisions) are computed in blocks of 256 by a straight-line loop,
  `moveoutCols`, which vectorizes: it lives in the nmo module's own C code, built with
  `-fno-math-errno` (`localPassC`; a generic kernel compiles in its caller's module, so the
  flag could not reach it), the index converted as an int32 (int64 → float64 has no packed
  AVX2 form), and the "before time 0" test moved to the interpolation loop. Linear 36 →
  25 ms from splitting the loops, then 17.5 ms vectorized.
- **Shot fit:** `l1Fit` (the IRLS of the two-segment fit) was 55 % of a 5-fit profile. Its
  normal equations are now six symmetric sums and three right-hand sides in locals (the loop
  vectorizes), and the golden-section refinement keeps one interior point per step (one fit
  per step, not two). Instructions 458 M → 330 M for the profile, 38 → 21 ms on 2000 traces.
- Not changed: STA/LTA (19 ms) is a serial loop of divisions per sample; statics and LMO
  are per-row fractional shifts, bandwidth-bound in the cubic case. Picking and the shot fit
  are batch QC, run once per shot.

`tvectorized` now also checks `moveoutCols` and `l1Fit` for packed instructions in release
and danger builds. `nim testRelease` passed (34 files, release and danger) after these
changes.

### First-break picking in a search window

`bench/bprestack.nim`, the same shot, every pick guided by the refraction
(`guideVelocity` 2500, 50 ms before and 100 ms after it: 76 of the 2001 columns), one
thread.

2026-11-12, AMD Ryzen 9 3950X, Nim 2.4.0, gcc 12.2.0,
`-d:danger --opt:speed --debuginfo:on --linedir:on -march=x86-64-v3`, medians:

| `pickInto`, guided | whole traces read | the window's stretch read | 2026-11-13 |
|---|---|---|---|
| STA/LTA, MER, Coppens | 20.1 / 5.7 / 10.3 ms | 1.36 / 1.04 / 5.55 ms | 1.26 / 0.97 / 5.16 ms |
| the same with a 5–60 Hz band-pass | 42.9 / 28.4 / 33.1 ms | 5.84 / 5.21 / 28.1 ms | 3.29 / 2.89 / 16.0 ms |

The last column: the band-pass now runs in batches of the IIR's 24 lanes (`iirLanes`),
where it ran 8 rows per call and left two thirds of every batch empty; the pool hands
`pickInto` runs of whole batches. The picks are the same bit for bit.

- With a search window, STA/LTA and the energy ratio depend on the window's columns and a
  bounded stretch around them (the STA and LTA or the ratio's windows, the confidence's
  windows, the causal band-pass's `requiredMargin` before them). `pickInto` and `pick` now
  copy, band-pass and scan that stretch alone, where they read every trace to its end; the
  band-pass runs in the same lanes as before, on the stretches. On 2000 traces of 3000
  samples with a 150 ms window (the reviewer's case, minima of 9): energy ratio 8.2 → 1.0 ms,
  with an 8–60 Hz band-pass 40.7 → 4.0 ms; STA/LTA 30.1 → 1.3 ms and 62.7 → 4.3 ms.
- The picks: without a band-pass, the same column on every trace of that case and of
  21 600 others (three shots, three noise levels, with and without ground roll, guided and
  fixed windows); the SNR within 0.002 dB (its stabilization is now relative to the stretch).
  With a band-pass the filter starts at the stretch, not at the trace, and is not preceded
  by a detrend: 5 of 12 000 picks of the reviewer's case move (four by one sample, one by
  four, on flat maxima), the mean pick error stays the same to 0.01 ms.
- Coppens reads the whole trace (its ratio is cumulative from the trace's start and its
  stabilization relative to the whole trace): same picks, bit for bit. It gains only from
  its ratio being computed about the window instead of everywhere.

## QC and vibroseis

`nim bench` (bench/bqc.nim), one thread: amplitude QC of a 2000 × 3000 float32 panel, the
coherence SNR of a 24 × 6000 gather, the correlation of 200 records of 20 s at 2 ms with a
12 s sweep (nfft 16200) and of one such record, the shot fit of 2000 picks, the resampling
of a 30 s pilot at 0.25 ms.

2026-11-13, AMD Ryzen 9 3950X, Nim 2.4.0, gcc 12.2.0,
`-d:danger --opt:speed --debuginfo:on --linedir:on -march=x86-64-v3`, medians:

| operation | before | after | 2026-11-13 run |
|---|---|---|---|
| `traceAmplitudeInto`, 2000 × 3000 | 38.6 ms (6.4 ns a sample) | 14.2 ms (2.4 ns) | 13.5 ms (2.3 ns) |
| `coherenceSnrInto`, window 0.1 / 1 / 12 s | 0.375 / 2.42 / 21.0 ms | 0.153 / 0.162 / 0.185 ms | 0.148 / 0.156 / 0.177 ms |
| `correlateInto`, 200 records | 7.90 ms (39.5 µs a record) | 6.18 ms (30.9 µs) | 6.08 ms (30.4 µs) |
| `correlateInto`, one record | 254 µs | 86 µs | 84 µs |
| `fitShot` on 2000 picks, two-segment / binned | 22.6 / 22.8 ms, 1793 / 4111 allocations | 22.3 / 22.4 ms, 2 / 3 allocations | 21.6 / 21.9 ms |
| `resamplePilot` to 4 ms / to 0.5 ms | 265 / 259 ms | 5.4 / 5.9 ms | 5.2 / 5.5 ms |

- **Amplitude QC.** One loop per trace did the sums, the extremes and a finite test
  sample by sample, a second the clip runs, a third a sliding sum of squares that had to
  be rebuilt whenever a large term left it. Now: `Σx` and `Σx²` in eight partial sums; the
  extremes, the finite test and the constant test as three integer reductions on the
  samples' bits (float `min`/`max` reductions do not vectorize: they would have to be
  reordered); a vector count of the samples at the maximum, so that only the traces with
  `clipRun` of them (few) get the run by run pass; and for the spikes the window sums from
  per-block suffix and prefix sums (no subtraction, so no rebuilding), with the loop over
  the windows not cut by the trace's ends straight-line, in vectors. A trace with a NaN
  or an Inf takes the former sample by sample path. Counts and flags are the same on all
  8800 traces of a comparison (spikes, clips, DC, quantized, short, constant and dead
  ones); the mean differs in the 13th digit on three (the order of the sums). The spike
  pass's sums need `2·cols + 1` float64, hence the `Workspace` the call now takes. What
  is left is mostly two serial sums per sample for the block sums, about 1 ns.
- **Coherence SNR.** Each output column summed its window: cols × window operations. The
  four window sums now come from block sums (the residual and the coherent energy, as
  suffix plus prefix of blocks one window long: no subtraction, so a quiet window after a
  loud one keeps its precision and a NaN stays in its own windows) and from running sums
  (the two counts, whole numbers, exact as differences): the time no longer depends on the
  window. Equal to a loop over every window within 1e-9 dB (tested with a 10⁴ : 1 loud and
  quiet stretch); the scratch grows from 3 to 6 arrays of `cols`.
- **Correlation, a block.** The two FFTs are 5.3 of the 6.2 ms now; what was around them
  cost half as much again. The spectrum's multiplication by the pilot's was vectorized
  across bins, shuffling the eight lanes in and out of each register (17.3 ms for 250
  batches of 8101 bins on its own; 4.1 ms as one vector operation per bin, which a kernel
  that is not inlined gets). The lane buffer is filled and read sample by sample, one pass
  with a stream per record, and zeroed only past the longest record, where it was zeroed
  whole and then written lane by lane (9.8 → 3.3 ms and 6.9 → 3.1 ms for 2000 records).
  2000 records: 77.8 → 64.4 ms.
- **Correlation, one record.** It ran the eight-lane path with seven empty lanes; it now
  takes the one-trace FFT.
- The outputs are equal to the last bit or one from it: 3727 of 105 064 values of four
  cases differ, by at most 1.9e-8 at a peak of 0.2 (a float32's rounding; the products are
  fused in another order), and the one-record form agrees with its row of the block form
  to the same.
- **Shot fit.** A call allocated once or twice per row: the median of each trace's
  neighbours sorted a seq (the sort's buffer), and the binned model copied its bins for
  every row. The medians now select in place (the same values), and every list is scratch
  kept in the `ShotFit`; what is left is the buffers of the three sorts of a call. The
  time is the L1 fit's and hardly moves; the outputs are the same to the last bit (five
  shots of 7 to 2000 rows, both models, with and without samples).
- **Resampling.** A Bessel series and a sine per tap of every output sample became one
  kernel per phase of the two intervals' ratio (one for 0.25 → 4 ms, two for 0.25 → 0.5 ms),
  tabulated once, and a dot product per output. (Before: 24 samples of the coarser
  interval a side; after: 32, the new default.)

## Brute stack

`nim bench` (bench/bprestack.nim), 12 000 traces × 1001 samples of float32 at 2 ms read
from a SEG-Y file, shot-sorted, stacked by `bruteStack` (statics, the flow and NMO in
parseSEG's prefetcher; the calling thread stacks), 2026-11-13, medians:

| case | 1 worker | 4 workers |
|---|---|---|
| a velocity function | 28.7 ms (1.68 GB/s) | 9.5 ms (5.07 GB/s) |
| a 2D velocity field | 33.0 ms | 12.7 ms |
| statics + band-pass + cubic NMO | 78.2 ms | 22.2 ms |

The result is the same bit for bit for any number of workers (the stacking stays on the
calling thread, in trace order).
