## Amplitude QC of field records: per-trace statistics (RMS and maximum in
## dB, mean, non-finite, clipped and spiky samples) with flags for traces
## that are dead, constant, clipped, spiky or DC-offset by their samples,
## whatever their header says; then per record, the median and MAD of the
## traces' RMS in dB and flags for the weak and noisy outliers.
##
## ```nim
## var amps = newSeq[TraceAmplitude](meta.len)
## traceAmplitudeInto(samples, meta, AmplitudeParams(), amps)  # or pool.traceAmplitudeInto
## var ws = initWorkspace()
## ws.reserve(recordAmplitudeWorkspace(meta.len))
## let rec = recordAmplitude(amps, AmplitudeParams(), ws)    # adds afWeak / afNoisy
## echo rec.medianRmsDb, " dB; noisy: ", rec.counts[afNoisy]
## for r, a in amps:
##   if a.flags != {}: echo r, ": ", a.flags
## ```
##
## **Which samples.** The live samples of the live rows (`ChunkMeta.isLive`,
## `liveSamples`), cut to the time window `[startMs, endMs]` on the trace
## axis (`ChunkMeta.sampleTimeMs`, both ends included; the default is the
## whole trace, whatever `t0Ms`). A non-live row, or one with no sample in
## the window, has `samples = 0`, NaN statistics and no flags. Statistics are
## over the finite samples; `nonFinite` counts the others (NaN, ±Inf).
##
## **Per trace** (one pass for the sums, one for clips and spikes):
## - `rmsDb`, `maxDb`: `20·log₁₀` of the RMS and of `max |x|` (in data
##   units; −Inf for zeros); `mean`, the DC level;
## - `afNonFinite`: a NaN or Inf sample;
## - `afDead`: every finite sample is 0 ("dead by amplitude"); `afConstant`:
##   every finite sample equals the same non-zero value. Neither gets the
##   flags below;
## - `afClipped`: `clipped` counts the samples in runs of at least `clipRun`
##   consecutive samples at the trace's maximum magnitude (`|x| ≥ (1 −
##   clipTolerance)·max |x|`), positive or negative: a recorder or a
##   conversion that saturated. Asymmetric clipping is seen at the larger
##   level only;
## - `afSpiky`: `spikes` counts the samples whose deviation from the mean is
##   more than `spikeFactor` times the RMS deviation of their neighbours
##   within `±spikeWindowMs` (the sample itself left out, so a spike does not
##   hide itself; two spikes closer than the window can). A non-zero sample
##   among zeros is a spike;
## - `afDcOffset`: `|mean| > dcRatio × σ`, σ the standard deviation.
##
## **Per record.** `recordAmplitude` takes a record's traces: those with
## samples, a finite `rmsDb` and neither `afDead` nor `afConstant` (the
## traces `RecordAmplitude.traces` counts) give the median and MAD of
## `rmsDb`, and each gets its robust z-score `robustZ = (rmsDb − median) /
## (1.4826·MAD)`; `afWeak` below `−outlierZ`, `afNoisy` above `outlierZ`
## (with MAD 0, any difference from the median is an outlier). Give it one
## record at a time: a whole line of shots mixes near and far records.
##
## **SNR.** For an SNR per trace use `qc/snr.windowSnrInto` on the same
## block and windows (a signal window after the first break, a noise window
## before it); it is not repeated here.
##
## Nothing allocates (the record statistics use a `Workspace`); nothing
## raises on data. The pool runs `traceAmplitudeInto` across its workers
## (`core/pool`).

import std/math
import parseSEG
import ../core/[rows, types, workspace]
import ../gain/stats

type
  AmplitudeFlag* = enum
    ## Why a trace stands out (module documentation).
    afNonFinite = "non-finite" ## A NaN or Inf sample.
    afDead = "dead"            ## Every finite sample is 0.
    afConstant = "constant"    ## Every finite sample is the same non-zero value.
    afClipped = "clipped"      ## Runs of samples at the maximum magnitude.
    afSpiky = "spiky"          ## Samples far above their neighbours.
    afDcOffset = "dc offset"   ## A mean large against the deviation.
    afWeak = "weak"            ## RMS far below the record's (`recordAmplitude`).
    afNoisy = "noisy"          ## RMS far above the record's (`recordAmplitude`).

  AmplitudeParams* = object
    ## What the amplitude QC measures and its thresholds; the defaults
    ## suit field records.
    startMs*: float64 = -Inf   ## Window start on the trace axis, ms.
    endMs*: float64 = Inf      ## Window end, ms (both ends included).
    clipRun*: int = 3          ## Shortest run at the maximum that is a clip (≥ 1).
    clipTolerance*: float64 = 0 ## Relative: `|x| ≥ (1 − tol)·max` is at the maximum.
    spikeFactor*: float64 = 10 ## A spike: deviation > factor × the neighbours' RMS.
    spikeWindowMs*: float64 = 20 ## Half-width of the neighbourhood, ms.
    dcRatio*: float64 = 0.2    ## DC offset: `|mean| > dcRatio × σ` (Inf: off).
    outlierZ*: float64 = 3.5   ## Weak / noisy: `|robustZ| > outlierZ` (Inf: off).

  TraceAmplitude* = object
    ## The amplitude statistics of one trace (module documentation).
    samples*: int       ## Samples examined: live, in the window.
    rmsDb*: float64     ## `20·log₁₀ RMS` of the finite samples; NaN without any.
    maxDb*: float64     ## `20·log₁₀ max |x|`; NaN without finite samples.
    mean*: float64      ## The mean (DC level); NaN without finite samples.
    nonFinite*: int     ## NaN and Inf samples.
    clipped*: int       ## Samples in clipped runs.
    spikes*: int        ## Spike samples.
    robustZ*: float64 = NaN ## Of `rmsDb` in its record (`recordAmplitude`).
    flags*: set[AmplitudeFlag]

  RecordAmplitude* = object
    ## The amplitude statistics of one record.
    traces*: int           ## Traces in the statistics (module documentation).
    medianRmsDb*: float64  ## Their median `rmsDb`; NaN without traces.
    madDb*: float64        ## The median absolute deviation of `rmsDb`, dB.
    counts*: array[AmplitudeFlag, int] ## Traces with each flag.
    flagged*: int          ## Traces with any flag.

proc badAmplitude(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc validate*(p: AmplitudeParams) =
  ## Raises `ValueError` when `p` is not valid: a NaN or reversed window,
  ## `clipRun < 1`, `clipTolerance` outside `0 ..< 1`, a `spikeFactor`,
  ## `dcRatio` or `outlierZ` not above 0, or a `spikeWindowMs` not positive
  ## and finite. Every proc taking `AmplitudeParams` checks it.
  if p.startMs.isNaN or p.endMs.isNaN or not (p.startMs <= p.endMs):
    badAmplitude("the window must have startMs ≤ endMs, got " & $p.startMs &
      " .. " & $p.endMs)
  if p.clipRun < 1: badAmplitude("clipRun must be ≥ 1, got " & $p.clipRun)
  if not (p.clipTolerance >= 0 and p.clipTolerance < 1):
    badAmplitude("clipTolerance must be in 0 ..< 1, got " & $p.clipTolerance)
  if not (p.spikeFactor > 0): # Inf: no spikes
    badAmplitude("spikeFactor must be > 0, got " & $p.spikeFactor)
  if not (p.spikeWindowMs > 0 and p.spikeWindowMs < Inf):
    badAmplitude("spikeWindowMs must be > 0 and finite, got " & $p.spikeWindowMs)
  if not (p.dcRatio > 0): badAmplitude("dcRatio must be > 0, got " & $p.dcRatio)
  if not (p.outlierZ > 0): badAmplitude("outlierZ must be > 0, got " & $p.outlierZ)

func db(x: float64): float64 {.inline.} = 20 * log10(x)

# ------------------------------------------------------------------ kernel --

func devSq(v, mean: float64): float64 {.inline.} =
  ## The squared deviation of a finite `v` from `mean`; 0 for NaN or Inf.
  if v - v == 0: (v - mean) * (v - mean) else: 0.0

{.push boundChecks: off.} # lengths come from the openArray
func traceAmplitudeImpl[T](x: openArray[T], half: int,
    p: AmplitudeParams): TraceAmplitude {.raises: [].} =
  result = TraceAmplitude(samples: x.len, rmsDb: NaN, maxDb: NaN, mean: NaN)
  var sum, sumSq, mx = 0.0
  var lo = Inf
  var hi = -Inf
  var finiteN = 0
  for i in 0 ..< x.len:
    let v = float64(x[i])
    if v - v == 0: # finite
      sum += v
      sumSq += v * v
      mx = max(mx, abs(v))
      lo = min(lo, v)
      hi = max(hi, v)
      inc finiteN
  result.nonFinite = x.len - finiteN
  if result.nonFinite > 0: result.flags.incl afNonFinite
  if finiteN == 0:
    return
  let n = float64(finiteN)
  let mean = sum / n
  result.mean = mean
  result.rmsDb = db(sqrt(sumSq / n))
  result.maxDb = db(mx)
  if mx == 0:
    result.flags.incl afDead
    return
  if lo == hi:
    result.flags.incl afConstant
    return
  let sigma = sqrt(max(sumSq / n - mean * mean, 0.0))
  if abs(mean) > p.dcRatio * sigma: result.flags.incl afDcOffset

  # Clipped runs at the maximum magnitude.
  let level = (1 - p.clipTolerance) * mx
  var run = 0
  for i in 0 .. x.len:
    let at = i < x.len and abs(float64(x[i])) >= level
    if at: inc run
    else:
      if run >= p.clipRun: result.clipped += run
      run = 0

  # Spikes: the deviation against its neighbours' RMS deviation in
  # [i − half, i + half], itself left out (a sliding sum of squares).
  if p.spikeFactor < Inf:
    template dev(j: int): float64 = devSq(float64(x[j]), mean)
    let k2 = p.spikeFactor * p.spikeFactor
    var win = 0.0
    for j in 0 .. min(half, x.len - 1): win += dev(j)
    for i in 0 ..< x.len:
      if i > 0:
        if i + half < x.len: win += dev(i + half)
        if i - half - 1 >= 0: win -= dev(i - half - 1)
      let d = dev(i)
      let count = min(i + half, x.len - 1) - max(i - half, 0) # neighbours
      let others = max(win - d, 0.0)
      if count > 0 and d > 0 and d * float64(count) > k2 * others:
        inc result.spikes
  if result.clipped > 0: result.flags.incl afClipped
  if result.spikes > 0: result.flags.incl afSpiky
{.pop.}

func spikeHalf(p: AmplitudeParams, dtMs: float64): int {.inline.} =
  max(int(floor(p.spikeWindowMs / dtMs + 1e-9)), 1)

proc traceAmplitude*[T: SampleType](x: openArray[T], dtMs: float64,
    params = AmplitudeParams()): TraceAmplitude =
  ## The amplitude statistics of one trace's samples `x` (all of them: the
  ## window is not applied), `dtMs` apart. Raises `ValueError` for invalid
  ## `params` or a `dtMs` that is not positive and finite.
  validate(params)
  if not (dtMs > 0 and dtMs < Inf): badAmplitude("dtMs must be > 0, got " & $dtMs)
  traceAmplitudeImpl(x, spikeHalf(params, dtMs), params)

func windowOf(meta: ChunkMeta, r, n: int, p: AmplitudeParams): Slice[int] =
  # The columns 0 ..< n of row r whose time is in the window; all for an
  # infinite window; none for a NaN t0 with a finite one.
  if p.startMs == -Inf and p.endMs == Inf: return 0 ..< n
  const tol = 1e-9
  let origin = meta.sampleTimeMs(r, 0)
  if origin.isNaN: return 1 .. 0
  let lo = max(ceil((p.startMs - origin) / meta.dtMs - tol), 0.0)
  let hi = min(floor((p.endMs - origin) / meta.dtMs + tol), float64(n - 1))
  if not (lo <= hi): 1 .. 0 else: int(lo) .. int(hi)

proc traceAmplitudeInto*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    params: AmplitudeParams, dst: var openArray[TraceAmplitude]) =
  ## The amplitude statistics of every row of `b` into `dst[row]` (module
  ## documentation); `robustZ` stays NaN and `afWeak`/`afNoisy` unset until
  ## `recordAmplitude`. Raises `ValueError` for invalid `params`, when `b`
  ## does not match `meta`, or `dst.len != b.len`. Allocates nothing.
  meta.checkShape(b)
  validate(params)
  if dst.len != b.len:
    badAmplitude("dst holds " & $dst.len & " values, the block has " & $b.len & " rows")
  let half = spikeHalf(params, meta.dtMs)
  for r in 0 ..< b.len:
    var a = TraceAmplitude(rmsDb: NaN, maxDb: NaN, mean: NaN)
    if meta.isLive(r):
      let w = windowOf(meta, r, meta.liveSamples(r, b.cols), params)
      if w.len > 0:
        a = traceAmplitudeImpl(b.rowPtr(r).toOpenArray(w.a, w.b), half, params)
    dst[r] = a

proc traceAmplitudeInto*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    params: AmplitudeParams, dst: var openArray[TraceAmplitude]) =
  ## `traceAmplitudeInto` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  traceAmplitudeInto(rowBlock(samples), meta, params, dst)

# ------------------------------------------------------------------ record --

func recordAmplitudeWorkspace*(traces: int): WorkspaceSize {.raises: [].} =
  ## The scratch `recordAmplitude` needs for a record of up to `traces` traces.
  WorkspaceSize(f64: 2 * traces)

func usable(a: TraceAmplitude): bool {.inline.} =
  a.samples > 0 and a.rmsDb - a.rmsDb == 0 and
    a.flags * {afDead, afConstant} == {}

proc recordAmplitude*(traces: var openArray[TraceAmplitude], params = AmplitudeParams(),
    ws: var Workspace): RecordAmplitude =
  ## The statistics of one record's `traces` (from `traceAmplitudeInto`):
  ## median and MAD of `rmsDb`, each trace's `robustZ`, `afWeak` and
  ## `afNoisy` (set or cleared), and the flag counts. `ws` must hold
  ## `recordAmplitudeWorkspace(traces.len)`. Raises `ValueError` for invalid
  ## `params`. Allocates nothing.
  validate(params)
  result = RecordAmplitude(medianRmsDb: NaN, madDb: NaN)
  let n = traces.len
  let v = ws.scratch(float64, n, at = n) # percentileAbs uses 0 ..< n
  var lowest = Inf
  var m = 0
  for a in traces.mitems:
    a.flags.excl {afWeak, afNoisy}
    a.robustZ = NaN
    if usable(a):
      lowest = min(lowest, a.rmsDb)
      inc m
  if m > 0:
    # Shifted to ≥ 0, so percentileAbs (of |x|) gives the median.
    var k = 0
    for a in traces:
      if usable(a):
        v[k] = a.rmsDb - lowest
        inc k
    let median = percentileAbs(v.toOpenArray(0, m - 1), 50, ws) + lowest
    k = 0
    for a in traces:
      if usable(a):
        v[k] = abs(a.rmsDb - median)
        inc k
    let mad = percentileAbs(v.toOpenArray(0, m - 1), 50, ws)
    result.traces = m
    result.medianRmsDb = median
    result.madDb = mad
    let scale = 1.4826 * mad
    for a in traces.mitems:
      if not usable(a): continue
      let d = a.rmsDb - median
      a.robustZ = (if scale > 0: d / scale elif d == 0: 0.0 elif d > 0: Inf else: -Inf)
      if a.robustZ > params.outlierZ: a.flags.incl afNoisy
      elif a.robustZ < -params.outlierZ: a.flags.incl afWeak
  for a in traces:
    for f in a.flags: inc result.counts[f]
    if a.flags != {}: inc result.flagged
