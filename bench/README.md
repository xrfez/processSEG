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

### bmemory: memory baseline, 2000 × 2001 float32 (16 MB per buffer)

2026-11-10, AMD Ryzen 9 3950X (32 threads), linux/amd64, Nim 2.4.0, gcc 12.2.0,
`-d:danger --opt:speed --debuginfo:on --linedir:on -march=x86-64-v3`:

| kernel | median | GB/s (median) |
|---|---|---|
| copy (copyMem) | 629 µs | 50.9 |
| scale in place | 427 µs | 75.0 |

The working set (16–32 MB) sits largely in the 3950X's L3, so these are cache rather than
DRAM bandwidths: the ceiling that a single-pass panel kernel on one thread can reach.

### bfft: real FFT, 2000 traces, forward + inverse (task 2.1)

Target: forward + inverse over a 2000-trace float32 panel at n = 2048 in ≤ 20 ms on one
thread. **Met: 11.4 ms.**

2026-11-10, AMD Ryzen 9 3950X (32 threads), linux/amd64, Nim 2.4.0, gcc 12.2.0,
`-d:danger --opt:speed --debuginfo:on --linedir:on -march=x86-64-v3`:

| case | panel (median) | per trace and transform | GFlop/s (2.5·n·log₂n) |
|---|---|---|---|
| f32 lanes n = 2048 | 11.4 ms | 2.84 µs | 19.8 |
| f32 lanes n = 2025 (odd) | 14.4 ms | 3.60 µs | 15.4 |
| f32 lanes n = 4096 | 24.5 ms | 6.13 µs | 20.0 |
| f32 lanes n = 1000 | 3.26 ms | 0.82 µs | 30.5 |
| f64 lanes n = 2048 | 19.6 ms | 4.90 µs | 11.5 |
| f32 one trace n = 2048 | — | 4.61 µs | — |

How it got there (each step measured):
- first version 43.2 ms: every butterfly loop was scalar. The radix-4/5 passes stored to
  four or five offsets of one output pointer, and GCC needs more runtime alias checks
  than its limit (10) to vectorize that. Each butterfly is now an inline kernel with every
  stream as its own `noalias` pointer: 16.0 ms;
- cachegrind then put 64 % of instructions in the real-FFT pack/split loops (scalar: a
  runtime lane count and a seq field reloaded per iteration). Static lanes and `noalias`
  kernels there: 11.4 ms.

An odd length runs a full-length complex FFT, so `nextFastLen` returns even lengths by
default (2001 → 2048, not 2025). n = 1000 is the fastest per point because its work
buffers (4 × 500 × 8 lanes × 4 B = 64 KB) nearly fit L1; at 2048 they are 256 KB (L2).

### Long records (60 s at 1 ms)

Same machine and flags, 200 traces, forward + inverse:

| case | per trace and transform | ns per n·log₂n |
|---|---|---|
| f32 lanes n = 65536 (60001 samples padded to a power of two) | 233 µs | 0.22 |
| f32 lanes n = 60000 (2⁵·3·5⁴) | 74 µs | 0.07 |
| f32 one trace n = 65536 | 241 µs | 0.23 |

A power-of-two length is about 3× slower than the 2·3·5 length near it at this size.
Forward only, ns per n·log₂n and trace, from a sweep (5 samples each):

| n | lanes | one trace |
|---|---|---|
| 32768 | 0.17 | 0.24 |
| 61440 (2¹²·15) | 0.38 | 0.21 |
| 60000 | 0.08 | 0.17 |
| 65536 | 0.22 | 0.23 |
| 122880 | 0.59 | 0.23 |
| 131072 | 1.73 | 0.80 |

- From m ≈ 30k complex points the 8-lane work buffers (4·m·8·4 B ≥ 4 MB) no longer fit
  in cache, and batching stops paying.
- Lengths with many factors of 2 alias in cache (Stockham strides of 2^k): 61440 and
  131072 are the worst.
- A 60 s record at 1 ms (65536) is still fine: about 0.12 ms per trace forward, so a
  1000-trace spectrum average takes about 0.12 s on one thread.
- Longer power-of-two records (131072 and up) would need a cache-blocked (four-step)
  FFT; it is in PLAN.md "Later".

## Filters (task 3.4)

`nim bench` (bench/bfilter.nim), same machine and flags as above. A panel is 2000 traces ×
2001 samples of float32 at 2 ms, filtered in place (live rows, edge handling included).
"4 workers" is 4 threads started once, 500 rows each, shared read-only plan, one
`Workspace` each: the wall time per round. The pool's hand-off (5.2) is not included.

| filter | L (pad) | nfft | 1 thread | 4 workers | speedup |
|---|---|---|---|---|---|
| FFT band-pass cos² 4-8-60-80 Hz | 110 | 2500 | 23.6 ms (11.8 µs/trace) | 6.2 ms | 3.82× |
| FFT low-cut cos² 0.5–1 Hz | 881 | 5760 | 42.2 ms (21.1 µs/trace) | 11.5 ms | 3.67× |
| IIR zero-phase Butterworth 4–60 Hz, 24 dB/oct (4 poles) | 99 | — | 25.0 ms (12.5 µs/trace) | 6.5 ms | 3.86× |

(Re-measured after the 3.5 review: L is now taken per edge, so the band-passes got longer
pads: 67 → 110 and 45 → 99.)

Where the FFT band-pass goes (cachegrind, instructions): the FFT passes about 45 %, the
strided pack/unpack (with its float64 conversions) about 30 %, the trend sums about 15 %
(scalar float64: a float sum keeps its IEEE order), the gain multiply about 6 %. The
pack and the trend are the next targets (an 8-accumulator sum and packing 8 rows at once),
probably worth 1.2–1.3×; not done, as the panel already fits a frame (PLAN.md "Later").

**Viewer budget** (set from these numbers): one filter over a 2000 × 2001 panel costs
about **6 ms on 4 workers** (12 ms for a narrow 0.5 Hz low-cut ramp), so a filter plus AGC
fits a 60 Hz frame (16.7 ms) with room for the draw; on one thread it takes 1.5–2.5 frames,
so a viewer without the pool should filter off the draw thread. Cost scales with traces ×
nfft·log(nfft): halve the panel, halve the time.

## Gain (Phase 4)

`nim bench` (bench/bgain.nim), same machine and flags as above, one thread, a panel of
2000 traces × 2001 samples of float32 (16 MB).

| operation | panel (median) | GB/s |
|---|---|---|
| `traceStatsInto` (RMS and max per trace) | 0.86 ms | 18.7 |
| `panelStats`, no clip | 0.96 ms | 16.7 |
| `panelStats`, p99 clip (histogram) | 4.5 ms | 3.5 |

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
| `agc` RMS, 100 ms window | 6.3 ms |
| `agc` RMS, 500 ms window | 6.7 ms |
| `agc` RMS, 2000 ms window | 6.7 ms |
| `normalize` per panel, p99 | 6.2 ms |
| `applyGainCurve` (knots, t0 on the grid) | 0.57 ms (56 GB/s) |

- AGC is O(n): the cost does not depend on the window. Each window sum is a block suffix
  plus a block prefix (no subtraction, for precision), built in one loop with two
  independent float64 add chains; that loop is serial and is most of the time.
- How it got there: 11.9 / 17.2 / 28.2 ms at first (the cut windows at the trace ends ran
  a scalar path with integer `div`/`mod`); closed forms per range, 9.3 / 11.1 / 16.3;
  int32 counts (int64 → float64 has no packed form on AVX2), 9.2 / 10.5 / 13.3; the gain
  step in float32 on level-scaled means, 6.5 / 7.8 / 10.6; ascending affine indices in
  every range (no gathers), 6.3 / 6.7 / 6.7.

## Process pool (task 5.2)

`nim bench` (bench/bpool.nim), same machine and flags. The pool is created once, outside
the timing. "ping" is an empty job on every worker (`pool.ping`); "empty plan" is
`pool.apply` of an empty plan on one row, in place (argument checks included).

| workers | ping | empty plan, 1 row |
|---|---|---|
| 1 | 6.4 µs | 6.8 µs |
| 2 | 7.4 µs | 7.8 µs |
| 4 | 8.3 µs | 8.0 µs |
| 8 | 12.3 µs | 12.4 µs |

So a job costs a wake-up and a join, microseconds, against milliseconds of work.

A viewer pipeline, demean + band-pass 4-8-60-80 + AGC 500 ms, on a 2000 × 2001 float32
panel, src → dst (the copy fused into the first op):

| run | time per panel | speedup |
|---|---|---|
| `plan.apply`, 1 thread (no pool) | 32.2 ms | 1 |
| pool, 1 worker | 32.5 ms | 0.99× |
| pool, 2 workers | 16.5 ms | 1.95× |
| pool, 4 workers | 8.6 ms | 3.75× |
| pool, 8 workers | 4.3 ms | 7.44× |

The tiled `apply` (every op on ~256 KB of rows while it is in cache) costs the same as the
ops run one after another over the whole panel (23.6 ms band-pass + 6.5 ms AGC + demean).

## Pre-stack, QC and stacking (task 9.2)

`nim bench` (bench/bprestack.nim), same machine and flags as above, one thread, a split-spread
shot of 2000 traces × 2001 samples of float32 at 2 ms (16 MB): a refraction and a reflection
over noise.

2026-11-11, AMD Ryzen 9 3950X, Nim 2.4.0, gcc 12.2.0,
`-d:danger --opt:speed --debuginfo:on --linedir:on -march=x86-64-v3`:

| operation | before 9.2 | after |
|---|---|---|
| `applyStatics` linear / cubic | 6.6 / 14.9 ms | 6.1 / 15.5 ms |
| `applyLmo` + `removeLmo` | 10.8 ms | 10.3 ms |
| `applyNmo` linear | 36.3 ms | 17.5 ms |
| `applyNmo` cubic | 54.5 ms | 24.7 ms |
| `addLive`, 2000 traces into 200 bins | 3.5 ms (13.8 GB/s) | 3.4 ms |
| `pickInto` STA/LTA, MER, Coppens | 20.0 / 6.0 / 10.5 ms | 18.9 / 5.3 / 9.8 ms |
| the same with a 5–60 Hz band-pass | 100.6 / 86.5 / 91.0 ms | 44.1 / 30.7 / 35.3 ms |
| `fitShot` on picks (two-segment, binned) | 38.1 / 38.2 ms | 20.7 / 20.8 ms |
| `fitShot` with cross-correlation | 45.1 ms | 27.6 ms |

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
