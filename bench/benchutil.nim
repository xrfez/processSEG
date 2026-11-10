## A small benchmark harness: a timing loop that records the build and machine.
##
## Every result line carries the build flags and the machine, so a number is
## never quoted without them (`nim-performance`). A benchmark runs warm-up
## iterations, then `samples` timed samples; each sample repeats the body until
## it has run for at least `minSampleMs`, and the per-iteration time of the
## sample is kept. The median and the minimum over samples are reported, with
## throughput from the bytes one iteration touches.
##
##   let r = bench("copy f32 2000x2001", bytesPerIter = 2 * n * 4):
##     copyMem(dst[0].addr, src[0].addr, n * 4)
##   r.report
##
## Wall-clock numbers on a shared machine move by a few percent: compare medians
## over several runs, and use cachegrind for small differences.

import std/[algorithm, monotimes, os, osproc, strformat, strutils, times]

const
  benchFlags* {.strdefine.} = "(unknown flags)"
    ## The build flags, passed by the `nim bench` task as `-d:benchFlags=...`.
  ccName = when defined(gcc): "gcc" elif defined(clang): "clang"
    elif defined(vcc): "vcc" else: "cc"
  ccVersion = staticExec(ccName & " --version").splitLines()[0]

type
  BenchResult* = object
    name*: string
    medianNs*, minNs*: float64 ## per iteration
    bytesPerIter*: int
    iterations*: int           ## timed iterations in total

proc cpuModel*(): string =
  ## The CPU's model name, or the host CPU architecture when it is unknown.
  result = hostCPU
  when defined(linux):
    try:
      for line in lines("/proc/cpuinfo"):
        if line.startsWith("model name"):
          return line.split(':', 1)[1].strip()
    except IOError, OSError:
      discard
  elif defined(macosx):
    try:
      let (output, code) = execCmdEx("sysctl -n machdep.cpu.brand_string")
      if code == 0 and output.strip().len > 0:
        result = output.strip()
    except OSError, IOError:
      discard

proc machineLine*(): string =
  ## One line describing the machine, OS, compiler and flags.
  &"{cpuModel()} ({countProcessors()} threads), {hostOS}/{hostCPU}, " &
    &"Nim {NimVersion}, {ccVersion}, flags: {benchFlags}"

proc printHeader*(title: string) =
  ## Prints the benchmark file's title and the machine line.
  echo "## ", title
  echo "# ", machineLine()
  echo "# ", now().format("yyyy-MM-dd HH:mm")

func gbPerS*(r: BenchResult, ns: float64): float64 =
  ## Throughput in GB/s (10⁹ bytes) at `ns` per iteration.
  if ns > 0: r.bytesPerIter.float64 / ns else: 0.0

proc report*(r: BenchResult) =
  ## Prints one result line: median and minimum time, and GB/s at each.
  var line = &"{r.name:<40} median {r.medianNs / 1e3:10.2f} µs  min {r.minNs / 1e3:10.2f} µs"
  if r.bytesPerIter > 0:
    line.add &"  {r.gbPerS(r.medianNs):7.2f} GB/s (best {r.gbPerS(r.minNs):.2f})"
  echo line

proc benchResult*(name: string, perIter: var seq[float64], bytes,
    iterations: int):
    BenchResult =
  ## The result from per-iteration sample times (sorts `perIter`). Used by
  ## `bench`, where a template parameter cannot share a field's name.
  perIter.sort()
  BenchResult(name: name, medianNs: perIter[perIter.len div 2], minNs: perIter[0],
    bytesPerIter: bytes, iterations: iterations)

template bench*(benchName: string, bytesPerIter: int,
    body: untyped): BenchResult =
  ## Times `body` (see the module doc) and returns the result; `report` prints it.
  ## `samples`, `warmup` and `minSampleMs` come from `-d:benchSamples=...` etc.
  block:
    const
      samples {.intdefine: "benchSamples".} = 15
      warmup {.intdefine: "benchWarmup".} = 3
      minSampleMs {.intdefine: "benchMinSampleMs".} = 20
    for _ in 0..<warmup:
      body
    var perIter: seq[float64]
    var total = 0
    for _ in 0..<samples:
      var n = 0
      let start = getMonoTime()
      var elapsed: int64
      while true:
        body
        inc n
        elapsed = inNanoseconds(getMonoTime() - start)
        if elapsed >= minSampleMs * 1_000_000:
          break
      perIter.add elapsed.float64 / n.float64
      total += n
    benchResult(benchName, perIter, bytesPerIter, total)
