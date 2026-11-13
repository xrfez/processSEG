## Filter throughput (task 3.4): a band-pass of a viewer panel, 2000 traces ×
## 2001 samples of float32 at 2 ms, on one thread and on 4 worker threads.
##
## Cases: the FFT band-pass (cos² 4-8-60-80 Hz, the usual display filter),
## a 0.5–1 Hz low-cut (the longest pads: L = 881), and the zero-phase IIR
## Butterworth band-pass 4–60 Hz at 24 dB/octave.
##
## The 4-worker number uses 4 threads started once, each filtering its own
## 500 rows with its own `Workspace` and the shared, read-only plan; the time
## per panel is the wall time of a round of all 4 divided by the rounds. It
## leaves out the hand-off of a persistent pool (task 5.2, measured there),
## which is microseconds against milliseconds here.

import std/[monotimes, strutils, times, algorithm]
import benchutil
import processSEG
import ../tests/utils/synth

const
  traces = 2000
  ns = 2001
  dt = 2.0
  workers = 4
  rounds = 20

proc panel(rows: int): SynthGather[float32] =
  result = cmpGather[float32](cmpX = 0, offsets = newSeq[float64](rows), ns, dt)
  for r in 0..<rows:
    var x = newSeq[float32](ns)
    x.addNoise(1, seed = r)
    x.addRicker(dt, 1000 + float64(r), 25, amp = 4)
    x.addDc(3)
    for j in 0..<ns: result.data[r, j] = x[j]

type
  Job[P] = object
    plan: ptr P
    data: ptr Tensor[float32]
    meta: ptr ChunkMeta
    rounds: int

proc worker[P](job: Job[P]) {.thread.} =
  var ws = initWorkspace()
  ws.reserve(job.plan[].workspaceSize)
  for _ in 0..<job.rounds:
    job.plan[].applyFilter(job.data[], job.meta[], ws)

proc threaded[P](name: string, plan: P) =
  var parts: array[workers, SynthGather[float32]]
  var metas: array[workers, ChunkMeta]
  for w in 0..<workers:
    parts[w] = panel(traces div workers)
    metas[w] = initChunkMeta(parts[w].rows, dtMs = dt)
  var samples: seq[float64]
  for _ in 0..<5:
    var th: array[workers, Thread[Job[P]]]
    let start = getMonoTime()
    for w in 0..<workers:
      createThread(th[w], worker[P], Job[P](plan: addr plan,
        data: addr parts[w].data, meta: addr metas[w], rounds: rounds))
    joinThreads(th)
    samples.add float64(inNanoseconds(getMonoTime() - start)) / rounds
  samples.sort()
  echo name, " ", workers, " workers  median ",
    formatFloat(samples[2] / 1e6, ffDecimal, 2), " ms per panel  min ",
    formatFloat(samples[0] / 1e6, ffDecimal, 2), " ms (", rounds,
    " rounds per sample, thread start included once)"

proc single[P](name: string, plan: P) =
  var g = panel(traces)
  let meta = initChunkMeta(g.rows, dtMs = dt)
  var ws = initWorkspace()
  ws.reserve(plan.workspaceSize)
  let r = bench(name & " 1 thread", bytesPerIter = 2 * traces * ns * 4):
    plan.applyFilter(g.data, meta, ws)
  r.report
  echo "    ", formatFloat(r.medianNs / 1e6, ffDecimal, 2), " ms per panel, ",
    formatFloat(r.medianNs / traces / 1000, ffDecimal, 2), " µs per trace"

proc main() =
  printHeader "filters, " & $traces & " traces × " & $ns & " samples f32 at 2 ms"
  let bp = initFilterPlan(float32, bandPass(Corners(f1: 4, f2: 8, f3: 60, f4: 80)), dt, ns)
  echo "# FFT band-pass 4-8-60-80: L = ", bp.requiredMargin, ", nfft = ", bp.nfft
  single("fft band-pass", bp)
  threaded("fft band-pass", bp)
  let lc = initFilterPlan(float32, lowCut(0.5, 1), dt, ns)
  echo "# FFT low-cut 0.5-1: L = ", lc.requiredMargin, ", nfft = ", lc.nfft
  single("fft low-cut  ", lc)
  threaded("fft low-cut  ", lc)
  let iir = initIirPlan(float32, butterBandPass(4, 60, 24), dt, ns)
  echo "# IIR zero-phase band-pass 4-60 24 dB/oct: order ", iir.order, ", pad ",
    iir.requiredMargin
  single("iir band-pass", iir)
  threaded("iir band-pass", iir)

main()
