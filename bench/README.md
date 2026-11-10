# Benchmarks

`nim bench` builds and runs every `bench/b*.nim` with
`-d:danger --opt:speed --debuginfo:on --linedir:on` (plus `-march=x86-64-v3` on x86-64), and
each run prints the machine, compiler and flags above its numbers. `benchutil.nim` is the
harness: warm-up, then 15 samples of at least 20 ms each (`-d:benchSamples`,
`-d:benchWarmup`, `-d:benchMinSampleMs`), reporting the median and minimum per iteration
and GB/s.

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
