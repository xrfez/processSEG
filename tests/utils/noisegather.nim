## Synthetic gathers for the exclusion-area tests: a shot over three
## receiver lines with a loud zone near the source, planted noise, the same
## gather re-sorted into a receiver gather, and four-component nodes.
##
## Geometry: `lines` receiver lines `lineGap` apart (line j at y = j·lineGap),
## `stations` receivers `spacing` apart on each (x = i·spacing), the source
## `srcOffLine` off line 0 at station `srcStation`. Every trace is ambient
## noise (0.02, ±2 dB of scatter per station), a first break at
## `30 + d/8` ms (a Ricker, 25 Hz) and ground roll (10 Hz, at `d/2.5` ms),
## the ground roll scaled by the natural level `9·exp(−(d/1200)²)`: the
## stations near the source are about 20 dB above the far ones, and the
## loud zone ends near the 9th station from the source.

import std/[math, random]
import processSEG
import synth
export synth

type
  NoiseSpec* = object
    lines*: int = 3
    stations*: int = 48
    spacing*: float64 = 220
    lineGap*: float64 = 1400
    srcStation*: float64 = 24
    srcOffLine*: float64 = -330
    dtMs*: float64 = 2
    cols*: int = 750
    seed*: int64 = 7

  NoiseGather* = object
    g*: SynthGather[float32]
    picks*: seq[FirstBreakPick] ## what a picker returns: confident where the break shows
    dist*: seq[float64]         ## each trace's true distance to the common point
    line*, station*: seq[int]   ## 0-based, of the far end
    spec*: NoiseSpec

func breakMs*(d: float64): float64 = 30 + d / 8

func natural*(d: float64): float64 = 1 + 9 * exp(-((d / 1500) ^ 2))

func row*(ng: NoiseGather, line, station: int): int = line * ng.spec.stations + station

proc rmsOf(x: openArray[float32]): float64 =
  var s = 0.0
  for v in x: s += float64(v) * float64(v)
  sqrt(s / float64(max(x.len, 1)))

proc setRow(ng: var NoiseGather, r: int, x: openArray[float32]) =
  for j in 0 ..< x.len: ng.g.data[r, j] = x[j]

proc getRow(ng: NoiseGather, r: int): seq[float32] =
  result = newSeq[float32](ng.g.cols)
  for j in 0 ..< result.len: result[j] = ng.g.data[r, j]

proc fillTrace(x: var seq[float32], dtMs, d, ambient: float64, seed: int64, level = 1.0) =
  ## Ambient noise, the break and the ground roll of a trace `d` from the common point.
  x.addNoise(ambient, seed)
  let loud = (natural(d) - 1) * level
  x.addRicker(dtMs, breakMs(d), 25, 0.1 * level)
  let tg = d / 2.5
  for j in 0 ..< x.len:
    let t = float64(j) * dtMs
    x[j] += float32(0.11 * loud * exp(-(((t - tg) / 100) ^ 2)) * sin(2 * PI * 10 * t / 1000))

proc scatter(seed: int64, i: int): float64 =
  ## ±2 dB, as a factor.
  var rng = initRand(seed * 1000 + i)
  pow(10.0, (rng.rand(1.0) - 0.5) * 4 / 20)

proc noiseGather*(spec = NoiseSpec()): NoiseGather =
  ## The shot gather described in the module documentation, with picks.
  let n = spec.lines * spec.stations
  var keys = newSeq[TraceKeys](n)
  result.spec = spec
  result.dist = newSeq[float64](n)
  result.line = newSeq[int](n)
  result.station = newSeq[int](n)
  let sx = spec.srcStation * spec.spacing
  let sy = spec.srcOffLine
  for j in 0 ..< spec.lines:
    for i in 0 ..< spec.stations:
      let r = j * spec.stations + i
      var k = initTraceKeys()
      k.record = 1
      k.srcLine = 2001
      k.srcPoint = 1
      k.srcX = sx
      k.srcY = sy
      k.rcvLine = float64(1001 + j)
      k.rcvPoint = float64(101 + i)
      k.rcvX = float64(i) * spec.spacing
      k.rcvY = float64(j) * spec.lineGap
      k.offset = hypot(k.rcvX - sx, k.rcvY - sy)
      k.channel = float64(r + 1)
      k.t0Ms = 0
      k.intervalUs = spec.dtMs * 1000
      k.traceCode = 1
      k.status = tsLive
      keys[r] = k
      result.dist[r] = k.offset
      result.line[r] = j
      result.station[r] = i
  result.g = initSynthGather[float32](keys, spec.cols, spec.dtMs)
  result.picks = newSeq[FirstBreakPick](n)
  var rng = initRand(spec.seed)
  for r in 0 ..< n:
    var x = newSeq[float32](spec.cols)
    x.fillTrace(spec.dtMs, result.dist[r], 0.02 * scatter(spec.seed, r), spec.seed + r)
    result.setRow(r, x)
    let d = result.dist[r]
    if d < 4000:
      result.picks[r] = FirstBreakPick(timeMs: breakMs(d) + rng.rand(4.0) - 2, col: 0,
        snrDb: 20, confidence: 1)
    else: # the picker's noise picks: random times, no confidence
      result.picks[r] = FirstBreakPick(timeMs: 100 + rng.rand(800.0), col: 0, snrDb: 2,
        confidence: 0.1)

proc stripOffsets*(ng: var NoiseGather) =
  for k in ng.g.keys.mitems: k.offset = NaN

proc stripSourceXY*(ng: var NoiseGather) =
  for k in ng.g.keys.mitems:
    k.srcX = NaN
    k.srcY = NaN

proc stripReceiverXY*(ng: var NoiseGather) =
  for k in ng.g.keys.mitems:
    k.rcvX = NaN
    k.rcvY = NaN

proc stripLines*(ng: var NoiseGather) =
  for k in ng.g.keys.mitems: k.rcvLine = NaN

proc stripPoints*(ng: var NoiseGather) =
  for k in ng.g.keys.mitems: k.rcvPoint = NaN

proc stripPicks*(ng: var NoiseGather) =
  ng.picks.setLen 0

proc plantHum*(ng: var NoiseGather, r: int, db = 20.0) =
  ## 60 Hz hum so that the trace's power rises by `db`.
  var x = ng.getRow(r)
  let own = rmsOf(x)
  x.addSinusoid(ng.spec.dtMs, 60, own * sqrt(2 * (pow(10.0, db / 10) - 1)))
  ng.setRow(r, x)

proc plantBroadband*(ng: var NoiseGather, r: int, db = 20.0) =
  var x = ng.getRow(r)
  let own = rmsOf(x)
  x.addNoise(own * sqrt(pow(10.0, db / 10) - 1), 900 + r)
  ng.setRow(r, x)

proc plantBurst*(ng: var NoiseGather, r: int, fromMs = 100.0, db = 20.0) =
  ## Noise from `fromMs` on, `db` above the trace's own power; the picker
  ## then finds a confident early break at `fromMs`.
  var x = ng.getRow(r)
  let own = rmsOf(x)
  var b = newSeq[float32](x.len)
  b.addNoise(own * sqrt(pow(10.0, db / 10) - 1) * sqrt(float64(x.len) / float64(
    max(1, x.len - int(fromMs / ng.spec.dtMs)))), 700 + r)
  for j in int(fromMs / ng.spec.dtMs) ..< x.len: x[j] += b[j]
  ng.setRow(r, x)
  if ng.picks.len > 0:
    ng.picks[r] = FirstBreakPick(timeMs: fromMs, col: 0, snrDb: 25, confidence: 1)

proc scaleTrace*(ng: var NoiseGather, r: int, db: float64) =
  ## The whole trace `db` louder (negative: weaker).
  var x = ng.getRow(r)
  for v in x.mitems: v = float32(float64(v) * pow(10.0, db / 20))
  ng.setRow(r, x)

proc receiverGather*(ng: NoiseGather): NoiseGather =
  ## The same traces as one receiver's gather (reciprocity): the source and
  ## receiver stations trade places, so the far end is the old receiver,
  ## which becomes the source. Each source line has its own strength (±3 dB)
  ## and each shot its own (±1.5 dB).
  result = ng
  var rng = initRand(ng.spec.seed + 99)
  var lineDb = newSeq[float64](ng.spec.lines)
  for j in 0 ..< ng.spec.lines: lineDb[j] = (rng.rand(1.0) - 0.5) * 6
  for r in 0 ..< ng.g.len:
    var k = result.g.keys[r]
    swap(k.srcX, k.rcvX)
    swap(k.srcY, k.rcvY)
    swap(k.srcLine, k.rcvLine)
    swap(k.srcPoint, k.rcvPoint)
    result.g.keys[r] = k
    result.scaleTrace(r, lineDb[ng.line[r]] + (rng.rand(1.0) - 0.5) * 3)

# ------------------------------------------------------------------ nodes --

type NodeGather* = object
  g*: SynthGather[float32]
  nodes*: int
  rotated*: seq[bool]

func nodeRow*(node, comp: int): int = 4 * node + comp ## comp 0 hydrophone, 1-3 geophones

proc nodeGather*(nodes = 24, rotatedToVTR = false, seed = 5'i64, cols = 500,
    dtMs = 2.0): NodeGather =
  ## `nodes` four-component nodes, each a hydrophone (code 11) and three
  ## geophones (12-14, or 15-17 when `rotatedToVTR`). The three world
  ## components have different levels (the vertical 12 dB above the
  ## horizontals), and each node's geophones are the world vector turned by
  ## a random 3-D rotation unless rotated to V/T/R; the hydrophone is 20 dB
  ## above the geophones' units.
  let n = 4 * nodes
  var keys = newSeq[TraceKeys](n)
  for i in 0 ..< nodes:
    for c in 0 ..< 4:
      var k = initTraceKeys()
      k.record = 1
      k.srcX = 0
      k.srcY = 0
      k.rcvLine = 1
      k.rcvPoint = float64(101 + i)
      k.rcvX = float64(i) * 50
      k.rcvY = 0
      k.offset = float64(i) * 50
      k.t0Ms = 0
      k.intervalUs = dtMs * 1000
      k.traceCode = int16(if c == 0: 11 elif rotatedToVTR: 14 + c else: 11 + c)
      k.component = uint8(k.traceCode)
      k.status = tsLive
      keys[nodeRow(i, c)] = k
  result.nodes = nodes
  result.g = initSynthGather[float32](keys, cols, dtMs)
  result.rotated = newSeq[bool](n)
  var rng = initRand(seed)
  for i in 0 ..< nodes:
    var world: array[3, seq[float32]]
    let level = [0.04, 0.012, 0.012]
    for c in 0 ..< 3:
      world[c] = newSeq[float32](cols)
      world[c].addNoise(level[c], seed * 100 + i * 4 + c)
      world[c].addRicker(dtMs, 100 + 3.0 * float64(i), 25, 4 * level[c])
    # A random rotation: Gram-Schmidt of random vectors.
    var m: array[3, array[3, float64]]
    for a in 0 ..< 3:
      for b in 0 ..< 3: m[a][b] = rng.rand(2.0) - 1
    if not rotatedToVTR:
      var l = sqrt(m[0][0] ^ 2 + m[0][1] ^ 2 + m[0][2] ^ 2)
      for b in 0 ..< 3: m[0][b] /= l
      var dot = m[1][0] * m[0][0] + m[1][1] * m[0][1] + m[1][2] * m[0][2]
      for b in 0 ..< 3: m[1][b] -= dot * m[0][b]
      l = sqrt(m[1][0] ^ 2 + m[1][1] ^ 2 + m[1][2] ^ 2)
      for b in 0 ..< 3: m[1][b] /= l
      m[2] = [m[0][1] * m[1][2] - m[0][2] * m[1][1], m[0][2] * m[1][0] - m[0][0] * m[1][2],
        m[0][0] * m[1][1] - m[0][1] * m[1][0]]
    else:
      m = [[0.0, 0, 1], [0.0, 1, 0], [1.0, 0, 0]] # V, T, R: the world axes, each its own level
    for c in 0 ..< 3:
      for j in 0 ..< cols:
        var v = 0.0
        for w in 0 ..< 3: v += m[c][w] * float64(world[w][j])
        result.g.data[nodeRow(i, c + 1), j] = float32(v)
    var p = newSeq[float32](cols)
    p.addNoise(0.4, seed * 100 + i * 4 + 3)
    p.addRicker(dtMs, 100 + 3.0 * float64(i), 25, 1.6)
    for j in 0 ..< cols: result.g.data[nodeRow(i, 0), j] = p[j]
    for c in 1 ..< 4: result.rotated[nodeRow(i, c)] = rotatedToVTR

# -------------------------------------------------------- uncorrelated record --

proc uncorrelatedGather*(n = 48, cols = 4000, dtMs = 2.0, seed = 3'i64): SynthGather[float32] =
  ## `n` traces that each hold an 8 s sweep from t = 0 (5 to 80 Hz) with
  ## ±2 dB of scatter: no first break, no geometry.
  var keys = newSeq[TraceKeys](n)
  for r in 0 ..< n:
    var k = initTraceKeys()
    k.record = 1
    k.channel = float64(r + 1)
    k.t0Ms = 0
    k.intervalUs = dtMs * 1000
    k.traceCode = 1
    k.status = tsLive
    keys[r] = k
  result = initSynthGather[float32](keys, cols, dtMs)
  for r in 0 ..< n:
    let a = scatter(seed, r)
    for j in 0 ..< cols:
      let t = float64(j) * dtMs / 1000
      result.data[r, j] = float32(a * sin(2 * PI * (5 * t + 75 * t * t / (2 * 8))))
    var x = newSeq[float32](cols)
    x.addNoise(0.05, seed * 10 + r)
    for j in 0 ..< cols: result.data[r, j] = result.data[r, j] + x[j]
