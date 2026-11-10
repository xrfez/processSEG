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
