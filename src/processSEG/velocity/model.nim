## Velocity models: RMS velocity functions of time (`VelocityFunction`),
## Dix conversion between RMS and interval velocities, and fields of
## functions over CDPs or an inline × crossline grid (`VelocityField`).
## Reading VELF, HANDVEL and CSV picks is in `velocity/formats`.
##
## ```nim
## let f = rmsVelocity([VelocityKnot(tMs: 0, vrms: 1500),
##                      VelocityKnot(tMs: 1000, vrms: 1800),
##                      VelocityKnot(tMs: 2500, vrms: 2400)])
## doAssert f.vrmsAt(1000) == 1800
## let field = initVelocityField([100.0, 200.0], [f, f2], dtMs = 4, tMaxMs = 6000)
## var v = newSeq[float64](2001)
## field.vrmsInto(150, startMs = 0, dtMs = 2, v)        # at CDP 150, per sample
## ```
##
## **Units.** Times are two-way times in ms from time zero, velocities in
## distance units per second (the file's coordinate units), as in
## `velocity/moveout`.
##
## **Interpolation in time.** A function given by knots `(tᵢ, Vᵢ)` is
## interpolated linearly in `W = V²·t`, not in `V`: between two knots the
## interval velocity is then constant, `Vint² = (W₂ − W₁)/(t₂ − t₁)`, which
## is exactly the Dix interval velocity of the knots, so a model never has an
## interval velocity the knots do not imply. Before the first knot the RMS
## velocity is `V₁` (one layer from time zero), after the last it stays
## `Vₙ`. **A knot at t = 0** is the exception: `W` from `(0, 0)` would
## ignore its velocity (Vrms would jump to the next layer's at once), so
## from 0 to the second knot the RMS velocity is linear in time, `V₀` to
## `V₁`, as picked; its interval velocity there is the instantaneous
## `sqrt(dW/dt) = sqrt(V² + 2t·V·dV/dt)`, and the layer's Dix velocity over
## the whole segment is still `V₁`, as `dixIntervalInto` gives.
##
## **Unphysical knots.** Knots whose `W` decreases (an RMS velocity that
## drops too fast) imply a negative `Vint²`. They are accepted, because
## picked velocities often have them, and the RMS velocity between them is
## still positive; `vintAt` and `dixIntervalInto` give NaN there.
##
## **Fields.** A `VelocityField` resamples every function onto one time grid
## `0, dt, …, tMax` and stores `V²` per node and grid time; a query
## interpolates `V²` (equivalently `W`) linearly in space between the
## neighbouring nodes, then linearly in `W` between grid times (in `V²`
## between the first two, where `W` is 0 at 0 whatever the velocity). Interpolating
## `W` keeps the result physical: the interval `Vint²` of a weighted mix of
## functions is the same mix of their `Vint²`. 2D fields are along one
## coordinate (a CDP number or a distance along the line); 3D fields are on a
## full inline × crossline grid of nodes (bilinear). Outside the nodes the
## nearest node's values are used, and past `tMax` the RMS velocity stays
## constant (`tMaxMs`; `bruteStack` refuses a field that ends before its
## stack does). Memory: nodes × grid samples × 8 bytes (2500 nodes at 6 s and
## 4 ms: 36 MB); a grid coarser than the data's interval is fine, the time
## interpolation is exact for knots on the grid (between knots past a first
## segment from zero), and otherwise the RMS velocity is within ≲ 1e-3
## (relative: worst at a knot off the grid, or in the first few grid times
## after a knot at zero).
##
## Builders and constructors raise `ValueError` for bad arguments. Point
## queries (`vrmsAt`, `vintAt`) never raise: NaN in, NaN out, and NaN for a
## query of the wrong field kind. The `vrmsInto` forms check their arguments
## once, so they raise for a bad `dtMs` or the wrong field kind.

import std/[algorithm, math, sequtils]

type
  VelocityKnot* = object
    ## A point of a `rmsVelocity` function.
    tMs*: float64  ## Two-way time, ms (≥ 0).
    vrms*: float64 ## RMS velocity, distance units per second.

  VelocityKind* = enum
    ## The shape of a `VelocityFunction`.
    vkConstant ## one velocity at every time
    vkGradient ## interval velocity `v0 + k·t`
    vkKnots    ## RMS knots, interpolated in `V²·t`

  VelocityFunction* = object
    ## An RMS velocity against two-way time; build it with
    ## `constantVelocity`, `gradientVelocity`, `rmsVelocity` or
    ## `intervalVelocity`, which validate it.
    case kind: VelocityKind
    of vkConstant:
      v: float64
    of vkGradient:
      v0, k: float64
    of vkKnots:
      knots: seq[VelocityKnot]

  FieldKind* = enum
    ## The layout of a `VelocityField`'s nodes.
    fk2D ## along one coordinate (CDP number or distance)
    fk3D ## an inline × crossline grid

  VelocityField* = object
    ## Velocity functions at nodes, resampled onto one time grid; see the
    ## module documentation. Build it with `initVelocityField`.
    kind: FieldKind
    dtMs: float64
    samples: int
    xs: seq[float64] ## 2D locations, or the 3D inline axis
    ys: seq[float64] ## the 3D crossline axis (empty in 2D)
    v2: seq[float64] ## node-major: V² at node n, grid time j: v2[n·samples + j]

proc badArg(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func isFinite(x: float64): bool {.inline.} =
  x.classify notin {fcNan, fcInf, fcNegInf}

proc checkVelocity(v: float64, what: string) =
  if not v.isFinite or v <= 0:
    badArg(what & " must be finite and positive, got " & $v)

# ---------------------------------------------------------------- functions --

proc constantVelocity*(v: float64): VelocityFunction =
  ## The RMS (and interval) velocity `v` at every time. Raises `ValueError`
  ## unless `v` is finite and positive.
  checkVelocity(v, "a velocity")
  VelocityFunction(kind: vkConstant, v: v)

proc gradientVelocity*(v0, k: float64): VelocityFunction =
  ## An interval velocity growing linearly with two-way time,
  ## `Vint(t) = v0 + k·t/1000` (`k` in distance units per second, per second
  ## of two-way time), so `Vrms² = v0² + v0·k·s + k²s²/3` with `s = t/1000`.
  ## A negative `k` is allowed; `vintAt` and `vrmsAt` are NaN once
  ## `Vint ≤ 0` (past `t = −1000·v0/k`). Raises
  ## `ValueError` unless `v0` is finite and positive and `k` finite.
  checkVelocity(v0, "a gradient's v0")
  if not k.isFinite:
    badArg("a gradient's k must be finite, got " & $k)
  VelocityFunction(kind: vkGradient, v0: v0, k: k)

proc checkTimes(tMs: openArray[float64], what: string) =
  for i, t in tMs:
    if not t.isFinite or t < 0:
      badArg(what & " " & $i & ": tMs must be finite and ≥ 0, got " & $t)
    if i > 0 and not (t > tMs[i - 1]):
      badArg(what & " " & $i & ": tMs " & $t & " must be greater than " & what &
        " " & $(i - 1) & "'s " & $tMs[i - 1] & " (times must increase)")

proc rmsVelocity*(knots: openArray[VelocityKnot]): VelocityFunction =
  ## RMS velocity knots `(tMs, vrms)`, interpolated linearly in `V²·t` (see
  ## the module documentation). Raises `ValueError` without knots, or when a
  ## time is negative, not finite or not greater than the one before
  ## (the message names the knot), or a velocity is not finite and positive.
  if knots.len == 0:
    badArg("a velocity function needs at least one knot")
  var times = newSeq[float64](knots.len)
  for i, k in knots:
    times[i] = k.tMs
    checkVelocity(k.vrms, "knot " & $i & ": vrms")
  checkTimes(times, "knot")
  VelocityFunction(kind: vkKnots, knots: @knots)

func kind*(f: VelocityFunction): VelocityKind {.inline, raises: [].} =
  ## Constant, gradient or knots.
  f.kind

func `==`*(a, b: VelocityFunction): bool {.raises: [].} =
  ## The same kind and parameters (knots compared exactly).
  if a.kind != b.kind: return false
  case a.kind
  of vkConstant: a.v == b.v
  of vkGradient: a.v0 == b.v0 and a.k == b.k
  of vkKnots: a.knots == b.knots

func knots*(f: VelocityFunction): seq[VelocityKnot] {.raises: [].} =
  ## A copy of the knots of a `vkKnots` function (empty for the other kinds).
  if f.kind == vkKnots: f.knots else: @[]

func w(k: VelocityKnot): float64 {.inline.} = k.vrms * k.vrms * k.tMs

func bracket(k: seq[VelocityKnot], t: float64): int {.inline.} =
  ## The `i` with `k[i].tMs ≤ t < k[i+1].tMs`, for `k[0].tMs ≤ t < k[^1].tMs`.
  var lo = 0
  var hi = k.len - 1
  while hi - lo > 1:
    let mid = (lo + hi) div 2
    if k[mid].tMs <= t: lo = mid else: hi = mid
  lo

func gradientV2(v0, k, tMs: float64): float64 {.inline.} =
  ## Vrms² of a gradient function at `tMs`; NaN past the time of Vint = 0.
  let s = max(tMs, 0) / 1000
  if k < 0 and s > -v0 / k: NaN
  else: v0 * v0 + v0 * k * s + k * k * s * s / 3

func knotsV2(k: seq[VelocityKnot], i: int, tMs: float64): float64 {.inline.} =
  ## Vrms² of a knots function at `tMs`, in its segment `i`
  ## (`k[i].tMs ≤ tMs < k[i+1].tMs`, after the first knot).
  if i == 0 and k[0].tMs == 0:
    # A knot at zero: the RMS velocity is linear from V₀ to V₁ (W = V²t
    # from (0, 0) would ignore V₀).
    let v = k[0].vrms + (k[1].vrms - k[0].vrms) * tMs / k[1].tMs
    v * v
  else:
    let w0 = k[i].w
    (w0 + (k[i + 1].w - w0) * (tMs - k[i].tMs) / (k[i + 1].tMs - k[i].tMs)) / tMs

func vrmsAt*(f: VelocityFunction, tMs: float64): float64 {.raises: [].} =
  ## The RMS velocity at two-way time `tMs` (NaN for a NaN time; times
  ## before 0 take the velocity at 0). A default function gives NaN.
  if tMs.isNaN: return NaN
  case f.kind
  of vkConstant:
    if f.v > 0: f.v else: NaN
  of vkGradient:
    if not (f.v0 > 0): return NaN
    sqrt(gradientV2(f.v0, f.k, tMs))
  of vkKnots:
    template k: untyped = f.knots
    if k.len == 0: return NaN
    if not (tMs > k[0].tMs): return k[0].vrms
    if tMs >= k[^1].tMs: return k[^1].vrms
    sqrt(knotsV2(k, bracket(k, tMs), tMs))

func vintAt*(f: VelocityFunction, tMs: float64): float64 {.raises: [].} =
  ## The interval velocity at two-way time `tMs`: for knots, that of the
  ## layer `tᵢ ≤ t < tᵢ₊₁` (`V₁` before the first knot, `Vₙ` from the last);
  ## NaN where it is not real (`Vint² < 0`, or `Vint ≤ 0` for a gradient) or
  ## for a NaN time.
  if tMs.isNaN: return NaN
  case f.kind
  of vkConstant:
    if f.v > 0: f.v else: NaN
  of vkGradient:
    let v = f.v0 + f.k * max(tMs, 0) / 1000
    if v > 0: v else: NaN
  of vkKnots:
    let k = f.knots
    if k.len == 0: return NaN
    if tMs < k[0].tMs: return k[0].vrms
    if tMs >= k[^1].tMs: return k[^1].vrms
    let i = bracket(k, tMs)
    if i == 0 and k[0].tMs == 0:
      # Vrms linear from V₀: the instantaneous Vint² = dW/dt = V² + 2tV·V'.
      let dv = (k[1].vrms - k[0].vrms) / k[1].tMs
      let v = k[0].vrms + dv * tMs
      let v2 = v * v + 2 * tMs * v * dv
      return if v2 >= 0: sqrt(v2) else: NaN
    let v2 = (k[i + 1].w - k[i].w) / (k[i + 1].tMs - k[i].tMs)
    if v2 >= 0: sqrt(v2) else: NaN

proc checkGrid(dtMs: float64) =
  if not dtMs.isFinite or dtMs <= 0:
    badArg("dtMs must be finite and positive, got " & $dtMs)

proc vrmsInto*(f: VelocityFunction, startMs, dtMs: float64,
    vrms: var openArray[float64]) =
  ## Fills `vrms[j]` with the RMS velocity at `startMs + j·dtMs`. Raises
  ## `ValueError` unless `dtMs` is finite and positive. Allocates nothing.
  checkGrid(dtMs)
  for j in 0..<vrms.len:
    vrms[j] = f.vrmsAt(startMs + float64(j) * dtMs)

# ---------------------------------------------------------------------- Dix --

proc checkSameLen(a, b, c: int) =
  if a != b or a != c:
    badArg("tMs, the input and the output must have the same length, got " &
      $a & ", " & $b & " and " & $c)

proc dixIntervalInto*(tMs, vrms: openArray[float64],
    vint: var openArray[float64]) =
  ## Dix conversion: `vint[i]` is the interval velocity of layer `i`, from
  ## `tMs[i−1]` (time zero for `i = 0`) to `tMs[i]`,
  ## `Vint² = (Vᵢ²tᵢ − Vᵢ₋₁²tᵢ₋₁)/(tᵢ − tᵢ₋₁)`, and `vint[0] = vrms[0]`.
  ## NaN where `Vint²` is negative. Raises `ValueError` for different
  ## lengths, times that are negative, not finite or not increasing, or
  ## velocities that are not finite and positive. `rmsFromIntervalInto`
  ## inverts it.
  checkSameLen(tMs.len, vrms.len, vint.len)
  checkTimes(tMs, "time")
  for i, v in vrms: checkVelocity(v, "vrms " & $i)
  var wPrev, tPrev = 0.0
  for i in 0..<tMs.len:
    let wi = vrms[i] * vrms[i] * tMs[i]
    if i == 0:
      vint[0] = vrms[0]
    else:
      let v2 = (wi - wPrev) / (tMs[i] - tPrev)
      vint[i] = if v2 >= 0: sqrt(v2) else: NaN
    wPrev = wi
    tPrev = tMs[i]

proc rmsFromIntervalInto*(tMs, vint: openArray[float64],
    vrms: var openArray[float64]) =
  ## The inverse of `dixIntervalInto`: `vint[i]` is the interval velocity
  ## from `tMs[i−1]` (zero for `i = 0`) to `tMs[i]`, and `vrms[i]` the RMS
  ## velocity at `tMs[i]` (`vrms[0] = vint[0]`). Raises `ValueError` as
  ## `dixIntervalInto`.
  checkSameLen(tMs.len, vint.len, vrms.len)
  checkTimes(tMs, "time")
  for i, v in vint: checkVelocity(v, "vint " & $i)
  var wSum, tPrev = 0.0
  for i in 0..<tMs.len:
    wSum += vint[i] * vint[i] * (tMs[i] - tPrev)
    vrms[i] = if tMs[i] > 0: sqrt(wSum / tMs[i]) else: vint[i]
    tPrev = tMs[i]

proc intervalVelocity*(tMs, vint: openArray[float64]): VelocityFunction =
  ## A function from layers: `vint[i]` from `tMs[i−1]` (zero for `i = 0`) to
  ## `tMs[i]`, kept past the last time. Built as RMS knots
  ## (`rmsFromIntervalInto`), so `vintAt` gives the layers back; to keep the
  ## last layer's velocity below `tMs[^1]` it gets one more knot, far below
  ## (1000 times deeper). A first layer at `tMs[0] = 0` has no thickness:
  ## with more layers it is dropped (its knot at 0 would make the RMS
  ## velocity linear from it, see `vrmsAt`, and change the next layer).
  ## Raises `ValueError` as `dixIntervalInto`, or without layers.
  if tMs.len == 0:
    badArg("a velocity function needs at least one layer")
  var vrms = newSeq[float64](tMs.len)
  rmsFromIntervalInto(tMs, vint, vrms)
  let first = if tMs.len > 1 and tMs[0] == 0: 1 else: 0
  var knots = newSeq[VelocityKnot](tMs.len - first)
  for i in first..<tMs.len:
    knots[i - first] = VelocityKnot(tMs: tMs[i], vrms: vrms[i])
  if knots[^1].tMs > 0 and vint[^1] != vrms[^1]:
    # Keep the last layer's interval velocity past tMs[^1]: one more knot far
    # below, where W keeps growing at vint².
    let t = knots[^1].tMs
    let tEnd = t + max(t, 1000) * 1000
    knots.add VelocityKnot(tMs: tEnd, vrms: sqrt((vrms[^1] * vrms[^1] * t +
      vint[^1] * vint[^1] * (tEnd - t)) / tEnd))
  VelocityFunction(kind: vkKnots, knots: knots)

# -------------------------------------------------------------------- field --

proc checkAxis(xs: openArray[float64], what: string) =
  if xs.len == 0:
    badArg("a velocity field needs at least one " & what)
  for i, x in xs:
    if not x.isFinite:
      badArg(what & " " & $i & " must be finite, got " & $x)
    if i > 0 and not (x > xs[i - 1]):
      badArg(what & " " & $i & " (" & $x & ") must be greater than " & what &
        " " & $(i - 1) & " (" & $xs[i - 1] & ")")

proc fillNode(v2: var openArray[float64], f: VelocityFunction, dtMs: float64) =
  ## `v2[j]` = Vrms² of `f` at `j·dtMs`. The grid times increase, so a knots
  ## function is swept once, its segment moving forward (no search per
  ## time), and V² is stored as it comes, `W/t`, not as the square of a
  ## square root: 0.25 against 0.76 s for 50 000 nodes of 1501 times.
  case f.kind
  of vkConstant:
    for x in v2.mitems: x = f.v * f.v
  of vkGradient:
    for j in 0..<v2.len: v2[j] = gradientV2(f.v0, f.k, float64(j) * dtMs)
  of vkKnots:
    template k: untyped = f.knots
    var i = 0
    for j in 0..<v2.len:
      let t = float64(j) * dtMs
      if not (t > k[0].tMs): v2[j] = k[0].vrms * k[0].vrms
      elif t >= k[^1].tMs: v2[j] = k[^1].vrms * k[^1].vrms
      else:
        while k[i + 1].tMs <= t: inc i # ends: t is before the last knot
        v2[j] = knotsV2(k, i, t)

proc fill(field: var VelocityField, functions: openArray[VelocityFunction]) =
  field.v2 = newSeq[float64](functions.len * field.samples)
  for n, f in functions:
    if f.kind == vkKnots and f.knots.len == 0 or f.kind == vkConstant and
        not (f.v > 0) or f.kind == vkGradient and not (f.v0 > 0):
      badArg("function " & $n & " is not initialized (build it with " &
        "constantVelocity, gradientVelocity, rmsVelocity or intervalVelocity)")
    fillNode(field.v2.toOpenArray(n * field.samples, (n + 1) * field.samples - 1),
      f, field.dtMs)

proc initGrid(dtMs, tMaxMs: float64): int =
  checkGrid(dtMs)
  if not tMaxMs.isFinite or tMaxMs < 0:
    badArg("tMaxMs must be finite and ≥ 0, got " & $tMaxMs)
  let r = tMaxMs / dtMs
  if not (r < float64(1 shl 24)):
    badArg("the field's time grid would have " & $r & " samples (at most " &
      $(1 shl 24) & "); use a larger dtMs")
  floor(r + 1e-9).int + 1

proc initVelocityField*(locations: openArray[float64],
    functions: openArray[VelocityFunction], dtMs,
        tMaxMs: float64): VelocityField =
  ## A 2D field: `functions[i]` at `locations[i]` (CDP numbers or distances
  ## along the line, strictly increasing), resampled every `dtMs` from 0 to
  ## `tMaxMs`. Raises `ValueError` for different lengths, no nodes,
  ## locations that are not finite or not increasing, a bad grid or a
  ## default function.
  if locations.len != functions.len:
    badArg("locations has " & $locations.len & " elements, functions " &
      $functions.len)
  checkAxis(locations, "location")
  result = VelocityField(kind: fk2D, dtMs: dtMs, samples: initGrid(dtMs, tMaxMs),
    xs: @locations)
  result.fill(functions)

proc initVelocityField*(inlines, crosslines: openArray[float64],
    functions: openArray[VelocityFunction], dtMs,
        tMaxMs: float64): VelocityField =
  ## A 3D field on the grid `inlines × crosslines` (each strictly
  ## increasing): `functions[i·crosslines.len + j]` is at
  ## `(inlines[i], crosslines[j])`. Raises `ValueError` as the 2D form, and
  ## when `functions.len != inlines.len · crosslines.len`.
  checkAxis(inlines, "inline")
  checkAxis(crosslines, "crossline")
  if functions.len != inlines.len * crosslines.len:
    badArg("a " & $inlines.len & " × " & $crosslines.len & " grid needs " &
      $(inlines.len * crosslines.len) & " functions, got " & $functions.len)
  result = VelocityField(kind: fk3D, dtMs: dtMs, samples: initGrid(dtMs, tMaxMs),
    xs: @inlines, ys: @crosslines)
  result.fill(functions)

func kind*(field: VelocityField): FieldKind {.inline, raises: [].} =
  ## 2D or 3D.
  field.kind

func len*(field: VelocityField): int {.inline, raises: [].} =
  ## The number of nodes.
  if field.samples == 0: 0 else: field.v2.len div field.samples

func dtMs*(field: VelocityField): float64 {.inline, raises: [].} =
  ## The time grid's interval, ms.
  field.dtMs

func samples*(field: VelocityField): int {.inline, raises: [].} =
  ## The number of grid times, `0 .. (samples − 1)·dtMs`.
  field.samples

func tMaxMs*(field: VelocityField): float64 {.inline, raises: [].} =
  ## The last grid time, `(samples − 1)·dtMs`: the field's functions are
  ## known up to it, and its velocities stay constant after it.
  float64(max(field.samples - 1, 0)) * field.dtMs

func locations*(field: VelocityField): lent seq[float64] {.inline, raises: [].} =
  ## The 2D node locations, or the 3D inline axis.
  field.xs

func crosslines*(field: VelocityField): lent seq[float64] {.inline, raises: [].} =
  ## The 3D crossline axis (empty for a 2D field).
  field.ys

type Weights = object
  ## Up to four nodes and their weights.
  n: int
  node: array[4, int]
  w: array[4, float64]

func axisPos(xs: seq[float64], x: float64): (int, float64) {.inline.} =
  ## The lower node and the weight of the upper one, clamped at the ends.
  if not (x > xs[0]): return (0, 0.0)
  if x >= xs[^1]: return (xs.len - 1, 0.0)
  var lo = 0
  var hi = xs.len - 1
  while hi - lo > 1:
    let mid = (lo + hi) div 2
    if xs[mid] <= x: lo = mid else: hi = mid
  (lo, (x - xs[lo]) / (xs[hi] - xs[lo]))

func weights2D(field: VelocityField, x: float64): Weights {.inline.} =
  let (i, f) = axisPos(field.xs, x)
  result.n = 1
  result.node[0] = i
  result.w[0] = 1 - f
  if f > 0:
    result.n = 2
    result.node[1] = i + 1
    result.w[1] = f

func weights3D(field: VelocityField, il, xl: float64): Weights {.inline.} =
  let (i, fi) = axisPos(field.xs, il)
  let (j, fj) = axisPos(field.ys, xl)
  let ny = field.ys.len
  for (di, wi) in [(0, 1 - fi), (1, fi)]:
    for (dj, wj) in [(0, 1 - fj), (1, fj)]:
      if wi > 0 and wj > 0:
        result.node[result.n] = (i + di) * ny + j + dj
        result.w[result.n] = wi * wj
        inc result.n

func v2At(field: VelocityField, ws: Weights, j: int): float64 {.inline.} =
  for m in 0..<ws.n:
    result += ws.w[m] * field.v2[ws.node[m] * field.samples + j]

func vrmsWith(field: VelocityField, ws: Weights,
    tMs: float64): float64 {.inline.} =
  if tMs.isNaN or field.samples == 0: return NaN
  # The grid position, by the reciprocal as `fillWith` takes it for a whole
  # column: a point query and a column agree.
  let u = tMs * (1 / field.dtMs)
  if not (u > 0): return sqrt(field.v2At(ws, 0))
  if u >= float64(field.samples - 1):
    return sqrt(field.v2At(ws, field.samples - 1))
  let i = int(u)
  let f = u - float64(i)
  if i == 0:
    # From time zero W = V²t starts at 0 whatever V is: interpolate V².
    let v0 = field.v2At(ws, 0)
    return sqrt(v0 + (field.v2At(ws, 1) - v0) * f)
  # Linear in W = V²·t between grid times i and i + 1.
  let w0 = field.v2At(ws, i) * float64(i)
  let w1 = field.v2At(ws, i + 1) * float64(i + 1)
  sqrt((w0 + (w1 - w0) * f) / u)

func vrmsAt*(field: VelocityField, location, tMs: float64): float64 {.raises: [].} =
  ## The RMS velocity of a 2D field at `location` and two-way time `tMs`
  ## (NaN when either is NaN, or for a 3D field).
  if field.kind != fk2D or location.isNaN or field.xs.len == 0: return NaN
  field.vrmsWith(field.weights2D(location), tMs)

func vrmsAt*(field: VelocityField, inline, crossline,
    tMs: float64): float64 {.raises: [].} =
  ## The RMS velocity of a 3D field at `(inline, crossline)` and `tMs` (NaN
  ## when any is NaN, or for a 2D field).
  if field.kind != fk3D or inline.isNaN or crossline.isNaN or field.xs.len == 0:
    return NaN
  field.vrmsWith(field.weights3D(inline, crossline), tMs)

proc fillWith(field: VelocityField, ws: Weights, startMs, dtMs: float64,
    vrms: var openArray[float64]) =
  # `vrmsWith` at startMs + j·dtMs, for every j. The times increase, so the
  # grid cell they fall in only moves forward: the nodes are blended once
  # per grid time (`lo`, `hi`: the V² at the cell's two grid times), not
  # for each output sample, and the grid position takes a multiplication.
  let last = field.samples - 1
  let perGrid = 1 / field.dtMs
  var cell = -1
  var lo, hi = 0.0
  template enter(i: int) =
    if i != cell:
      lo = if cell >= 0 and i == cell + 1: hi else: field.v2At(ws, i)
      hi = field.v2At(ws, min(i + 1, last))
      cell = i
  for j in 0..<vrms.len:
    let u = (startMs + float64(j) * dtMs) * perGrid
    if u.isNaN:
      vrms[j] = NaN
    elif not (u > 0):
      enter(0)
      vrms[j] = sqrt(lo)
    elif u >= float64(last):
      enter(last)
      vrms[j] = sqrt(lo)
    else:
      let i = int(u)
      let f = u - float64(i)
      enter(i)
      if i == 0:
        # From time zero W = V²t starts at 0 whatever V is: interpolate V².
        vrms[j] = sqrt(lo + (hi - lo) * f)
      else:
        # Linear in W = V²·t between grid times i and i + 1.
        let w0 = lo * float64(i)
        let w1 = hi * float64(i + 1)
        vrms[j] = sqrt((w0 + (w1 - w0) * f) / u)

proc vrmsInto*(field: VelocityField, location, startMs, dtMs: float64,
    vrms: var openArray[float64]) =
  ## Fills `vrms[j]` with the 2D field's RMS velocity at `location` and
  ## `startMs + j·dtMs` (all NaN for a NaN location). Raises `ValueError`
  ## for a 3D field or a `dtMs` that is not finite and positive. Allocates
  ## nothing.
  checkGrid(dtMs)
  if field.kind != fk2D: badArg("vrmsInto(location): the field is 3D")
  if location.isNaN or field.xs.len == 0:
    for v in vrms.mitems: v = NaN
  else:
    field.fillWith(field.weights2D(location), startMs, dtMs, vrms)

proc vrmsInto*(field: VelocityField, inline, crossline, startMs, dtMs: float64,
    vrms: var openArray[float64]) =
  ## The 3D form of `vrmsInto`, at `(inline, crossline)`. Raises
  ## `ValueError` for a 2D field or a bad `dtMs`.
  checkGrid(dtMs)
  if field.kind != fk3D: badArg("vrmsInto(inline, crossline): the field is 2D")
  if inline.isNaN or crossline.isNaN or field.xs.len == 0:
    for v in vrms.mitems: v = NaN
  else:
    field.fillWith(field.weights3D(inline, crossline), startMs, dtMs, vrms)

# -------------------------------------------------------------------- picks --

type
  VelocityPick* = object
    ## A velocity function at one place, as read from a velocity file
    ## (`velocity/formats`): a 2D pick has a `cdp` and NaN `inline` and
    ## `crossline`; a 3D pick the reverse.
    cdp*: float64
    inline*, crossline*: float64
    velocity*: VelocityFunction

func is3D(p: VelocityPick): bool {.inline.} =
  not p.inline.isNaN and not p.crossline.isNaN

proc initVelocityField*(picks: openArray[VelocityPick],
    dtMs, tMaxMs: float64): VelocityField =
  ## A field from picks, in any order: 2D by `cdp` when no pick has an
  ## inline and crossline, 3D when every pick has both; the 3D picks must
  ## cover a full inline × crossline grid. Raises `ValueError` for mixed or
  ## duplicate picks, a missing grid node, NaN locations, or as the other
  ## constructors.
  if picks.len == 0:
    badArg("a velocity field needs at least one pick")
  let threeD = picks[0].is3D
  for i, p in picks:
    if p.is3D != threeD:
      badArg("pick " & $i & " is " & (if threeD: "2D" else: "3D") &
        ", pick 0 is not: a field is 2D (cdp) or 3D (inline, crossline)")
  if not threeD:
    var order = newSeq[(float64, int)](picks.len)
    for i in 0..<picks.len: order[i] = (picks[i].cdp, i)
    order.sort
    var locs = newSeq[float64](picks.len)
    var fs = newSeq[VelocityFunction](picks.len)
    for k, (_, i) in order:
      if picks[i].cdp.isNaN:
        badArg("pick " & $i & " has no cdp" & (if picks[i].inline.isNaN and
          picks[i].crossline.isNaN: ", inline or crossline" else:
          ", and a 3D pick needs both an inline and a crossline"))
      locs[k] = picks[i].cdp
      fs[k] = picks[i].velocity
    return initVelocityField(locs, fs, dtMs, tMaxMs)
  var ils, xls: seq[float64]
  for p in picks:
    ils.add p.inline
    xls.add p.crossline
  ils.sort
  xls.sort
  ils = ils.deduplicate(isSorted = true)
  xls = xls.deduplicate(isSorted = true)
  if picks.len < ils.len * xls.len:
    badArg("3D picks must cover a full inline × crossline grid: " &
      $picks.len &
      " picks for " & $ils.len & " inlines × " & $xls.len & " crosslines")
  var fs = newSeq[VelocityFunction](ils.len * xls.len)
  var seen = newSeq[bool](fs.len)
  for i, p in picks:
    let n = ils.binarySearch(p.inline) * xls.len + xls.binarySearch(p.crossline)
    if seen[n]:
      badArg("pick " & $i & " repeats inline " & $p.inline & ", crossline " &
        $p.crossline)
    seen[n] = true
    fs[n] = p.velocity
  for n, s in seen:
    if not s:
      badArg("3D picks must cover a full inline × crossline grid: inline " &
        $ils[n div xls.len] & ", crossline " & $xls[n mod xls.len] & " is missing")
  initVelocityField(ils, xls, fs, dtMs, tMaxMs)
