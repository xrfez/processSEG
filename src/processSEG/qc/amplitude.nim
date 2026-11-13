## Amplitude QC of field records: per-trace statistics (RMS and maximum in
## dB, mean, non-finite, clipped and spiky samples) with flags for traces
## that are dead, constant, clipped, spiky or DC-offset by their samples,
## whatever their header says; then per record, the median and MAD of the
## traces' RMS in dB and flags for the weak and noisy outliers.
##
## ```nim
## var amps = newSeq[TraceAmplitude](meta.len)
## var ws = initWorkspace()
## ws.reserve(max(traceAmplitudeWorkspace(samples.shape[1]),
##   recordAmplitudeWorkspace(meta.len)))
## traceAmplitudeInto(samples, meta, AmplitudeParams(), amps, ws) # or pool.traceAmplitudeInto
## let rec = recordAmplitude(amps, AmplitudeParams(), ws)        # adds afWeak / afNoisy
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
## **Per trace** (vector passes for the sums, the extremes and the count
## of samples at the maximum; block sums for the spikes' windows; a trace
## with a NaN or an Inf goes sample by sample instead):
## - `rmsDb`, `maxDb`: `20·log₁₀` of the RMS and of `max |x|` (in data
##   units; −Inf for zeros); `mean`, the DC level;
## - `afNonFinite`: a NaN or Inf sample;
## - `afDead`: every finite sample is 0 ("dead by amplitude"); `afConstant`:
##   every finite sample equals the same non-zero value. Neither gets the
##   flags below;
## - `afClipped`: `clipped` counts the samples in runs of at least `clipRun`
##   consecutive samples at the trace's maximum, `x ≥ (1 − clipTolerance)·
##   max x` or `x ≤ (1 − clipTolerance)·min x`: a recorder or a conversion
##   that saturated. Each polarity has its own level when both reach the
##   maximum magnitude (within 1 %, or `clipTolerance` if larger), as the
##   two rails of an integer recorder do (+2ⁿ − 1 and −2ⁿ); a side that
##   stays below it is not examined, so asymmetric clipping is seen at the
##   larger level only. Inf samples are not clips. The flag needs two such
##   runs, or one of `2·clipRun` samples: weak data recorded as integers
##   sits at its largest count for a few samples somewhere, which `clipped`
##   counts and the flag ignores (data of very few counts, with many such
##   runs, still reads as clipped);
## - `afSpiky`: `spikes` counts the samples whose deviation from the mean is
##   more than `spikeFactor` times the RMS deviation of their neighbours
##   within `±spikeHalfWindowMs` (the sample itself left out, so a spike does not
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
## Nothing allocates (both steps take their scratch from a `Workspace`:
## `traceAmplitudeWorkspace`, `recordAmplitudeWorkspace`; the one-trace
## `traceAmplitude` makes its own); nothing raises on data. The pool runs
## `traceAmplitudeInto` across its workers, each with its workspace
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
    afClipped = "clipped"      ## Two runs of samples at the maximum, or a long one.
    afSpiky = "spiky"          ## Samples far above their neighbours.
    afDcOffset = "dc offset"   ## A mean large against the deviation.
    afWeak = "weak"            ## RMS far below the record's (`recordAmplitude`).
    afNoisy = "noisy"          ## RMS far above the record's (`recordAmplitude`).

  AmplitudeParams* = object
    ## What the amplitude QC measures and its thresholds; the defaults
    ## suit field records.
    startMs*: float64 = -Inf ## Window start on the trace axis, ms.
    endMs*: float64 = Inf ## Window end, ms (both ends included).
    clipRun*: int = 3 ## Shortest run at the maximum that is a clip (≥ 2).
    clipTolerance*: float64 = 0 ## Relative: `|x| ≥ (1 − tol)·max` is at the maximum.
    spikeFactor*: float64 = 10 ## A spike: deviation > factor × the neighbours' RMS.
    spikeHalfWindowMs*: float64 = 20 ## The neighbours: those within this many ms, each side.
    dcRatio*: float64 = 0.2 ## DC offset: `|mean| > dcRatio × σ` (Inf: off).
    outlierZ*: float64 = 3.5 ## Weak / noisy: `|robustZ| > outlierZ` (Inf: off).

  TraceAmplitude* = object
    ## The amplitude statistics of one trace (module documentation).
    samples*: int           ## Samples examined: live, in the window.
    rmsDb*: float64         ## `20·log₁₀ RMS` of the finite samples; NaN without any.
    maxDb*: float64         ## `20·log₁₀ max |x|`; NaN without finite samples.
    mean*: float64          ## The mean (DC level); NaN without finite samples.
    nonFinite*: int         ## NaN and Inf samples.
    clipped*: int           ## Samples in runs of `clipRun` and more at the maximum.
    spikes*: int            ## Spike samples.
    robustZ*: float64 = NaN ## Of `rmsDb` in its record (`recordAmplitude`).
    flags*: set[AmplitudeFlag]

  RecordAmplitude* = object
    ## The amplitude statistics of one record.
    traces*: int                       ## Traces in the statistics (module documentation).
    medianRmsDb*: float64              ## Their median `rmsDb`; NaN without traces.
    madDb*: float64                    ## The median absolute deviation of `rmsDb`, dB.
    counts*: array[AmplitudeFlag, int] ## Traces with each flag.
    flagged*: int                      ## Traces with any flag.

proc badAmplitude(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc validate*(p: AmplitudeParams) =
  ## Raises `ValueError` when `p` is not valid: a NaN or reversed window,
  ## `clipRun < 1`, `clipTolerance` outside `0 ..< 1`, a `spikeFactor`,
  ## `dcRatio` or `outlierZ` not above 0, or a `spikeHalfWindowMs` not positive
  ## and finite. Every proc taking `AmplitudeParams` checks it.
  if p.startMs.isNaN or p.endMs.isNaN or not (p.startMs <= p.endMs):
    badAmplitude("the window must have startMs ≤ endMs, got " & $p.startMs &
      " .. " & $p.endMs)
  if p.clipRun < 2: badAmplitude("clipRun must be ≥ 2, got " & $p.clipRun)
  if not (p.clipTolerance >= 0 and p.clipTolerance < 1):
    badAmplitude("clipTolerance must be in 0 ..< 1, got " & $p.clipTolerance)
  if not (p.spikeFactor > 0): # Inf: no spikes
    badAmplitude("spikeFactor must be > 0, got " & $p.spikeFactor)
  if not (p.spikeHalfWindowMs > 0 and p.spikeHalfWindowMs < Inf):
    badAmplitude("spikeHalfWindowMs must be > 0 and finite, got " & $p.spikeHalfWindowMs)
  if not (p.dcRatio > 0): badAmplitude("dcRatio must be > 0, got " & $p.dcRatio)
  if not (p.outlierZ > 0): badAmplitude("outlierZ must be > 0, got " & $p.outlierZ)

func db(x: float64): float64 {.inline.} = 20 * log10(x)

# ------------------------------------------------------------------ kernel --

func devSq(v, mean: float64): float64 {.inline.} =
  ## The squared deviation of a finite `v` from `mean`; 0 for NaN or Inf.
  if v - v == 0: (v - mean) * (v - mean) else: 0.0

{.push boundChecks: off.} # lengths come from the openArray
func scalarAmplitude[T](x: openArray[T], half: int,
    p: AmplitudeParams): TraceAmplitude {.raises: [].} =
  ## The statistics of a trace with samples that are not finite: over its
  ## finite ones, sample by sample.
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

  # Clipped runs at the maximum. Each polarity has its own level when both
  # reach the maximum magnitude (within 1 %, or the tolerance): a
  # two's-complement recorder clips at +2ⁿ − 1 and −2ⁿ, and with one level
  # for both only the negative rail counted. A side that stays below that
  # is not clipped, whatever runs it has at its own extreme.
  let near = (1 - max(p.clipTolerance, 0.01)) * mx
  let up = (if hi >= near: (1 - p.clipTolerance) * hi else: Inf)
  let down = (if -lo >= near: (1 - p.clipTolerance) * lo else: -Inf)
  var run, runs, longest = 0
  for i in 0 .. x.len:
    let v = (if i < x.len: float64(x[i]) else: 0.0)
    if i < x.len and (v >= up or v <= down) and v - v == 0: # Inf is not a clip
      inc run
    else:
      if run >= p.clipRun:
        result.clipped += run
        inc runs
        longest = max(longest, run)
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
        if i - half - 1 >= 0:
          let leaving = dev(i - half - 1)
          if leaving > 0.25 * win:
            # A dominant term leaves: subtracting it would cancel the rest
            # (1e-16 of a spike's square can exceed its neighbours' energy).
            win = 0
            for j in max(i - half, 0) .. min(i + half, x.len - 1): win += dev(j)
          else:
            win -= leaving
      let d = dev(i)
      let count = min(i + half, x.len - 1) - max(i - half, 0)                     # neighbours
      let others = max(win - d, 0.0)
      if count > 0 and d > 0 and d * float64(count) > k2 * others:
        inc result.spikes
  # One short run at the maximum is what weak data recorded as integers
  # gives somewhere (4 traces of the 3954 of a field record, 83 dB and more
  # below its level): a clip repeats, or lasts.
  if runs >= 2 or longest >= 2 * p.clipRun: result.flags.incl afClipped
  if result.spikes > 0: result.flags.incl afSpiky
{.pop.}

type
  F64 = ptr UncheckedArray[float64]


{.push boundChecks: off, overflowChecks: off.}

proc sumAndSquares[T](x: ptr UncheckedArray[T], n: int, sum, sumSq: var float64) {.
    raises: [].} =
  ## `Σx` and `Σx²` of `x[0 ..< n]` in float64, each in eight partial sums
  ## (the loop vectorizes).
  var s, q: array[8, float64]
  let whole = n div 8 * 8
  var i = 0
  while i < whole:
    for j in 0 ..< 8:
      let v = float64(x[i + j])
      s[j] += v
      q[j] += v * v
    i += 8
  for k in whole ..< n:
    let v = float64(x[k])
    s[0] += v
    q[0] += v * v
  sum = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]))
  sumSq = ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]))

proc extremes[T](x: ptr UncheckedArray[T], n: int, most, least: var T,
    constant: var bool): bool {.raises: [].} =
  ## `most`: the largest of the positive samples of `x[0 ..< n]` (0 without
  ## any); `least`: the largest magnitude of the negative ones (0 without
  ## any); `constant`: every sample has the first one's bits. Returns
  ## whether all are finite. On the samples' bits, as integers: magnitudes
  ## order as their bit patterns do and a NaN or an Inf is at or above
  ## Inf's, so the loop is three integer reductions, which vectorize
  ## (float `min`/`max` reductions do not: they would have to be reordered).
  when T is float32:
    type B = int32
    const (signShift, infBits) = (31, 0x7F80_0000'i32)
  else:
    type B = int64
    const (signShift, infBits) = (63, 0x7FF0_0000_0000_0000'i64)
  let first = cast[B](x[0])
  var pos, neg, differing: B = 0
  for i in 0 ..< n:
    let b = cast[B](x[i])
    let negative = ashr(b, signShift) # all ones for a set sign bit
    let magnitude = b and high(B)
    pos = max(pos, magnitude and not negative)
    neg = max(neg, magnitude and negative)
    differing = differing + B(b != first)
  most = cast[T](pos)
  least = cast[T](neg)
  constant = differing == 0
  max(pos, neg) < infBits

proc countBeyond[T](x: ptr UncheckedArray[T], n: int, up, down: float64): int {.raises: [].} =
  ## The samples at or above `up` or at or below `down` (a vector loop).
  var c: array[8, int32]
  let whole = n div 8 * 8
  var i = 0
  while i < whole:
    for j in 0 ..< 8:
      let v = float64(x[i + j])
      c[j] += int32(v >= up) or int32(v <= down)
    i += 8
  for k in whole ..< n:
    let v = float64(x[k])
    c[0] += int32(v >= up) or int32(v <= down)
  for j in 0 ..< 8: result += int(c[j])

proc clipRuns[T](x: ptr UncheckedArray[T], n: int, up, down: float64, clipRun: int,
    clipped, runs, longest: var int) {.raises: [].} =
  ## The runs of `clipRun` and more samples beyond the levels: their
  ## samples, their number and the longest.
  var run = 0
  for i in 0 .. n:
    if i < n and (float64(x[i]) >= up or float64(x[i]) <= down):
      inc run
    else:
      if run >= clipRun:
        clipped += run
        inc runs
        longest = max(longest, run)
      run = 0

proc spikeCount[T](x: ptr UncheckedArray[T], n, half: int, mean, k2: float64,
    s, p: F64): int {.raises: [].} =
  ## The samples whose squared deviation from `mean`, times the neighbours
  ## in `[i − half, i + half]`, exceeds `k2` times those neighbours' summed
  ## squared deviations. The window sums come without subtraction, so a
  ## huge sample does not cancel its neighbours: per block of `w = 2·half +
  ## 1` samples, `s[k]` is the sum from `k` through the block's end and
  ## `p[k]` the sum from the block's start to before `k` (`p[n]` closes the
  ## last block). A whole window `[a, a + w)` is `s[a] + p[a + w]`, which is
  ## all the loop over the interior needs: it vectorizes.
  let w = 2 * half + 1
  var bs = 0
  var fwd = 0.0
  while bs < n:
    let be = min(bs + w, n)
    var bwd = 0.0
    fwd = 0
    for k in 0 ..< be - bs:
      let f = float64(x[bs + k]) - mean
      let b = float64(x[be - 1 - k]) - mean
      p[bs + k] = fwd
      fwd += f * f
      bwd += b * b
      s[be - 1 - k] = bwd
    bs = be
  p[n] = (if n mod w == 0: 0.0 else: fwd)
  template spike(i: int, others: float64, neighbours: int): int32 =
    let dv = float64(x[i]) - mean
    let d = dv * dv
    int32(neighbours > 0) and int32(d > 0) and
      int32(d * float64(neighbours) > k2 * max(others - d, 0.0))
  template atAnEnd(i: int): int32 =
    # A window cut by the trace: from the start (inside the first block),
    # or to the end (the rest of its block and, if another follows, the
    # last block, which is whole in `s` at its start).
    let a = max(0, i - half)
    let b = min(n - 1, i + half)
    let last = (n - 1) div w * w
    let sum = if a == 0: p[b + 1]
              elif a < last: s[a] + s[last]
              else: s[a]
    spike(i, sum, b - a)
  var c: int32 = 0
  for i in 0 ..< min(half, n): c += atAnEnd(i)
  for i in half ..< n - half: c += spike(i, s[i - half] + p[i + half + 1], 2 * half)
  for i in max(half, n - half) ..< n: c += atAnEnd(i)
  int(c)

{.pop.}

proc traceAmplitudeImpl[T](x: openArray[T], half: int, p: AmplitudeParams,
    scratch: F64): TraceAmplitude {.raises: [].} =
  ## The statistics of the trace `x`, with `2·x.len + 1` float64 of
  ## scratch.
  let n = x.len
  if n == 0: return TraceAmplitude(rmsDb: NaN, maxDb: NaN, mean: NaN)
  let xp = cast[ptr UncheckedArray[T]](addr x[0])
  var most, least: T
  var constant: bool
  if not extremes(xp, n, most, least, constant): return scalarAmplitude(x, half, p)
  let hi = float64(most) # the largest positive sample, or 0
  let lo = -float64(least) # the most negative one, or 0
  var sum, sumSq: float64
  sumAndSquares(xp, n, sum, sumSq)
  let mean = sum / float64(n)
  let mx = max(-lo, hi)
  result = TraceAmplitude(samples: n, mean: mean, rmsDb: db(sqrt(sumSq / float64(n))),
    maxDb: db(mx))
  if mx == 0:
    result.flags.incl afDead
    return
  if constant:
    result.flags.incl afConstant
    return
  let sigma = sqrt(max(sumSq / float64(n) - mean * mean, 0.0))
  if abs(mean) > p.dcRatio * sigma: result.flags.incl afDcOffset
  # Clipped runs at the maximum. Each polarity has its own level when both
  # reach the maximum magnitude (within 1 %, or the tolerance): a
  # two's-complement recorder clips at +2ⁿ − 1 and −2ⁿ, and with one level
  # for both only the negative rail counted. A side that stays below that
  # is not clipped, whatever runs it has at its own extreme.
  let near = (1 - max(p.clipTolerance, 0.01)) * mx
  let up = (if hi >= near: (1 - p.clipTolerance) * hi else: Inf)
  let down = (if -lo >= near: (1 - p.clipTolerance) * lo else: -Inf)
  var runs, longest = 0
  # Few traces have even clipRun samples at their maximum: a count first,
  # in a vector loop, spares the others the run by run pass.
  if countBeyond(xp, n, up, down) >= p.clipRun:
    clipRuns(xp, n, up, down, p.clipRun, result.clipped, runs, longest)
  # One short run at the maximum is what weak data recorded as integers
  # gives somewhere (4 traces of the 3954 of a field record, 83 dB and more
  # below its level): a clip repeats, or lasts.
  if runs >= 2 or longest >= 2 * p.clipRun: result.flags.incl afClipped
  if p.spikeFactor < Inf:
    result.spikes = spikeCount(xp, n, half, mean, p.spikeFactor * p.spikeFactor, scratch,
      cast[F64](addr scratch[n]))
  if result.spikes > 0: result.flags.incl afSpiky

func spikeHalf(p: AmplitudeParams, dtMs: float64): int {.inline.} =
  max(int(floor(p.spikeHalfWindowMs / dtMs + 1e-9)), 1)

proc traceAmplitude*[T: SampleType](x: openArray[T], dtMs: float64,
    params = AmplitudeParams()): TraceAmplitude =
  ## The amplitude statistics of one trace's samples `x` (all of them: the
  ## window is not applied), `dtMs` apart, with scratch of its own (it
  ## allocates; `traceAmplitudeInto` does not). Raises `ValueError` for
  ## invalid `params` or a `dtMs` that is not positive and finite.
  validate(params)
  if not (dtMs > 0 and dtMs < Inf): badAmplitude("dtMs must be > 0, got " & $dtMs)
  var scratch = newSeq[float64](2 * x.len + 1)
  traceAmplitudeImpl(x, spikeHalf(params, dtMs), params, cast[F64](addr scratch[0]))

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

func traceAmplitudeWorkspace*(cols: int): WorkspaceSize {.raises: [].} =
  ## The scratch `traceAmplitudeInto` needs for a block `cols` samples wide:
  ## `2·cols + 1` float64 (the spike pass's window sums).
  WorkspaceSize(f64: 2 * max(cols, 0) + 1)

proc traceAmplitudeInto*[T: SampleType](b: RowBlock[T], meta: ChunkMeta,
    params: AmplitudeParams, dst: var openArray[TraceAmplitude], ws: var Workspace) =
  ## The amplitude statistics of every row of `b` into `dst[row]` (module
  ## documentation); `robustZ` stays NaN and `afWeak`/`afNoisy` unset until
  ## `recordAmplitude`. `ws` must hold `traceAmplitudeWorkspace(b.cols)`.
  ## Raises `ValueError` for invalid `params`, when `b` does not match
  ## `meta`, `dst.len != b.len`, or a workspace too small. Allocates
  ## nothing.
  meta.checkShape(b)
  validate(params)
  if dst.len != b.len:
    badAmplitude("dst holds " & $dst.len & " values, the block has " & $b.len & " rows")
  if not ws.fits(traceAmplitudeWorkspace(b.cols)):
    badAmplitude("the workspace holds " & $ws.capacity & "; reserve " &
      "traceAmplitudeWorkspace(" & $b.cols & ")")
  let half = spikeHalf(params, meta.dtMs)
  let scratch = ws.scratch(float64, 2 * b.cols + 1)
  for r in 0 ..< b.len:
    var a = TraceAmplitude(rmsDb: NaN, maxDb: NaN, mean: NaN)
    if meta.isLive(r):
      let w = windowOf(meta, r, meta.liveSamples(r, b.cols), params)
      if w.len > 0:
        a = traceAmplitudeImpl(b.rowPtr(r).toOpenArray(w.a, w.b), half, params, scratch)
    dst[r] = a

proc traceAmplitudeInto*[T: SampleType](samples: Tensor[T], meta: ChunkMeta,
    params: AmplitudeParams, dst: var openArray[TraceAmplitude], ws: var Workspace) =
  ## `traceAmplitudeInto` on a `[rows, cols]` tensor (C-contiguous; raises
  ## `ValueError` otherwise).
  meta.checkShape(samples)
  traceAmplitudeInto(rowBlock(samples), meta, params, dst, ws)

# ------------------------------------------------------------------ record --

func recordAmplitudeWorkspace*(traces: int): WorkspaceSize {.raises: [].} =
  ## The scratch `recordAmplitude` needs for a record of up to `traces` traces.
  WorkspaceSize(f64: 2 * traces)

func usable(a: TraceAmplitude): bool {.inline.} =
  a.samples > 0 and a.rmsDb - a.rmsDb == 0 and
    a.flags * {afDead, afConstant} == {}

proc recordAmplitude*(traces: var openArray[TraceAmplitude], params: AmplitudeParams,
    ws: var Workspace): RecordAmplitude =
  ## The statistics of one record's `traces` (from `traceAmplitudeInto`):
  ## median and MAD of `rmsDb`, each trace's `robustZ`, `afWeak` and
  ## `afNoisy` (set or cleared), and the flag counts. `ws` must hold
  ## `recordAmplitudeWorkspace(traces.len)`. Raises `ValueError` for invalid
  ## `params` or a workspace too small. Allocates nothing.
  validate(params)
  if not ws.fits(recordAmplitudeWorkspace(traces.len)):
    badAmplitude("the workspace holds " & $ws.capacity & "; reserve " &
      "recordAmplitudeWorkspace(" & $traces.len & ")")
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
