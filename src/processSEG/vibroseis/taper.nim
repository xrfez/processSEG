## Sweep-end tapers (`SweepTaper`): the ramps that start and end a vibroseis
## sweep, so that it starts and stops without a jump.
##
## ```nim
## let taper = initSweepTaper(startMs = 300, endMs = 300, shape = tpCos2)
## var sweep = newSeq[float64](5001)            # 10 s at 2 ms, t = 0 .. 10000 ms
## for j in 0 ..< sweep.len: sweep[j] = 1.0
## sweep.applyTaper(dtMs = 2, taper)            # in place
## doAssert sweep[0] == 0 and sweep[2500] == 1
## ```
##
## **Time axis.** Sample `j` of the sweep is at `t = j·dtMs`, so a sweep of
## `n` samples lasts `(n − 1)·dtMs` (its first and last samples are its two
## ends). The start ramp weights `t < startMs` by `w(t / startMs)`, the end
## ramp weights `t > T − endMs` by `w((T − t) / endMs)`: tapers are set in
## ms, need not be a whole number of samples, and do not depend on `dtMs`
## beyond where the samples fall. Each ramp is 1 where it ends.
##
## **Shapes** (`TaperShape`): each is the rising half of its window, `w(x)`
## for `x` from 0 (the sweep's end) to 1 (full amplitude), with `w(1) = 1`
## and zero slope there except for `tpLinear`, `tpCosine` and `tpPower`:
##
## | Shape | `w(x)` | `w(0)` |
## |---|---|---|
## | `tpLinear` | `x` | 0 |
## | `tpCosine` | `sin(πx/2)` (a quarter cosine) | 0 |
## | `tpCos2` | `sin²(πx/2) = (1 − cos πx)/2`, the Hann window | 0 |
## | `tpHamming` | `0.54 − 0.46·cos πx` | 0.08 |
## | `tpBlackman` | `0.42 − 0.5·cos πx + 0.08·cos 2πx` | 0 |
## | `tpBlackmanHarris` | 4-term: `0.35875 − 0.48829·cos πx + 0.14128·cos 2πx − 0.01168·cos 3πx` | 6·10⁻⁵ |
## | `tpKaiser` | `I₀(β·√(1 − (1 − x)²)) / I₀(β)`, `param` = β ≥ 0 | `1/I₀(β)` |
## | `tpGaussian` | `exp(−½·((1 − x)/σ)²)`, `param` = σ > 0, in ramp lengths | `exp(−1/(2σ²))` |
## | `tpPower` | `xᵖ`, `param` = p > 0 (1 is linear) | 0 |
##
## **SEG-Y.** The taper type of the trace header (bytes 139-140) and of the
## binary header (3247-3248) knows linear (1), cos² (2) and other (3):
## `segyTaperType`.
##
## **Lengths.** A zero or NaN length means no ramp at that end. The only
## refusal is overlap: `startMs + endMs` longer than the sweep (the ramps may
## meet, not cross).

import std/math
import parseSEG
import kaiser

type
  TaperShape* = enum
    ## The shape of a sweep taper ramp (module documentation).
    tpLinear         ## `x`
    tpCosine         ## `sin(πx/2)`
    tpCos2           ## `sin²(πx/2)`, Hann
    tpHamming        ## Hamming, `w(0) = 0.08`
    tpBlackman       ## Blackman
    tpBlackmanHarris ## 4-term Blackman–Harris
    tpKaiser         ## Kaiser, `param` = β
    tpGaussian       ## Gaussian, `param` = σ in ramp lengths
    tpPower          ## `xᵖ`, `param` = p

  SweepTaper* = object
    ## The ramps at the two ends of a sweep: lengths in ms (0 or NaN: none)
    ## and one shape for both. Make with `initSweepTaper`, or as an object:
    ## a `param` left out is the shape's default.
    startMs*: float64
    endMs*: float64
    shape*: TaperShape
    param*: float64 = NaN ## β (Kaiser), σ (Gaussian) or p (power); NaN: the shape's default.

func defaultParam(shape: TaperShape): float64 =
  case shape
  of tpKaiser: 6.0
  of tpGaussian: 0.4
  of tpPower: 2.0
  else: NaN

proc badTaper(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

proc validate*(t: SweepTaper) =
  ## Checks the lengths (≥ 0 or NaN) and the shape's parameter (NaN: its
  ## default, which is valid). Raises `ValueError`.
  for (name, v) in [("startMs", t.startMs), ("endMs", t.endMs)]:
    if not v.isNaN and not (v >= 0 and v < Inf):
      badTaper("a taper " & name & " must be ≥ 0 and finite (or NaN for none), got " & $v)
  let param = if t.param.isNaN: defaultParam(t.shape) else: t.param
  case t.shape
  of tpKaiser:
    if not (param >= 0 and param <= 50):
      badTaper("a Kaiser taper needs 0 ≤ β ≤ 50, got " & $param)
  of tpGaussian:
    if not (param > 0 and param < Inf):
      badTaper("a Gaussian taper needs σ > 0, got " & $param)
  of tpPower:
    if not (param > 0 and param < Inf):
      badTaper("a power taper needs an exponent > 0, got " & $param)
  else: discard

proc initSweepTaper*(startMs, endMs: float64, shape = tpCos2,
    param = NaN): SweepTaper =
  ## Ramps of `startMs` and `endMs` (0 or NaN: none) of shape `shape`. A NaN
  ## `param` takes the shape's default: β = 6 (Kaiser), σ = 0.4 (Gaussian),
  ## p = 2 (power). Raises `ValueError` as `validate`.
  runnableExamples:
    let t = initSweepTaper(500, 250, tpKaiser)
    doAssert t.param == 6 and t.segyTaperType == 3
  result = SweepTaper(startMs: startMs, endMs: endMs, shape: shape,
    param: if param.isNaN: defaultParam(shape) else: param)
  result.validate

func noTaper*(): SweepTaper {.inline.} =
  ## No ramps.
  SweepTaper(startMs: 0, endMs: 0, shape: tpLinear, param: NaN)

func segyTaperType*(shape: TaperShape): int16 {.raises: [].} =
  ## The SEG-Y taper type code: 1 linear, 2 cos², 3 other.
  case shape
  of tpLinear: 1
  of tpCos2: 2
  else: 3

func segyTaperType*(t: SweepTaper): int16 {.inline, raises: [].} =
  ## The SEG-Y taper type code of `t`'s shape.
  segyTaperType(t.shape)

func taperWeight*(shape: TaperShape, x: float64, param = NaN): float64 {.raises: [].} =
  ## The ramp `w(x)` of `shape` at `x`, clamped to `[0, 1]` (1 beyond the
  ## ramp, `w(0)` before it); see the module documentation. A NaN `param`
  ## takes the shape's default.
  let p = if param.isNaN: defaultParam(shape) else: param
  let x = clamp(x, 0.0, 1.0)
  case shape
  of tpLinear: x
  of tpCosine: sin(PI / 2 * x)
  of tpCos2: 0.5 - 0.5 * cos(PI * x)
  of tpHamming: 0.54 - 0.46 * cos(PI * x)
  of tpBlackman: 0.42 - 0.5 * cos(PI * x) + 0.08 * cos(2 * PI * x)
  of tpBlackmanHarris:
    0.35875 - 0.48829 * cos(PI * x) + 0.14128 * cos(2 * PI * x) - 0.01168 * cos(3 * PI * x)
  of tpKaiser:
    let u = 1 - x
    besselI0(p * sqrt(max(0.0, 1 - u * u))) / besselI0(p)
  of tpGaussian:
    let u = (1 - x) / p
    exp(-0.5 * u * u)
  of tpPower: pow(x, p)

func length(v: float64): float64 {.inline.} =
  if v.isNaN: 0.0 else: v

proc checkFits*(t: SweepTaper, sweepMs: float64) =
  ## Raises `ValueError` when the ramps overlap in a sweep of `sweepMs`:
  ## `startMs + endMs > sweepMs` (beyond a relative 1e-9, so ramps that
  ## meet are allowed), or when `t` fails `validate`.
  t.validate
  let total = length(t.startMs) + length(t.endMs)
  if total > sweepMs * (1 + 1e-9):
    badTaper("the sweep tapers overlap: " & $length(t.startMs) & " + " &
      $length(t.endMs) & " ms is longer than the " & $sweepMs & " ms sweep")

proc applyTaper*[T: SampleType](x: var openArray[T], dtMs: float64, t: SweepTaper) =
  ## Applies the ramps of `t` in place to the sweep `x`, whose sample `j` is
  ## at `j·dtMs` (the sweep lasts `(x.len − 1)·dtMs`). Raises `ValueError`
  ## for a `dtMs` that is not finite and positive, or as `checkFits`.
  if not (dtMs > 0 and dtMs < Inf):
    badTaper("dtMs must be finite and positive, got " & $dtMs)
  let sweepMs = float64(max(x.len - 1, 0)) * dtMs
  t.checkFits(sweepMs)
  let s = length(t.startMs)
  let e = length(t.endMs)
  if s > 0:
    var j = 0
    while j < x.len and float64(j) * dtMs < s:
      x[j] = T(float64(x[j]) * taperWeight(t.shape, float64(j) * dtMs / s, t.param))
      inc j
  if e > 0:
    var j = x.len - 1
    while j >= 0 and sweepMs - float64(j) * dtMs < e:
      x[j] = T(float64(x[j]) * taperWeight(t.shape, (sweepMs - float64(j) * dtMs) / e, t.param))
      dec j
