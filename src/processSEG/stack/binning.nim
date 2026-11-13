## CMP binning: which stack bin a trace falls in, from its keys.
##
## ```nim
## let b2 = cdpBinning(firstCdp = 101, count = 400)          # header CDP
## let bl = lineBinning(x0, y0, azimuthDeg = 30, binSize = 12.5, count = 800)
## let b3 = gridBinning(x0, y0, azimuthDeg = 30, crosslineSpacing = 25,
##                      inlineSpacing = 25, inlines = 200, crosslines = 300)
## let bk = inlineCrosslineBinning(1001, 200, 1, 300)        # header il/xl
## var bins = newSeq[int](meta.len)                          # once; reused
## binsInto(meta.rows, b3, bins)                             # −1: no bin
## let loc = b3.location(bins[0])                            # inline, crossline, x, y
## ```
##
## **2D.** `cdpBinning` takes the keys' `ensemble` (the CDP number) as it
## is: bin `i` holds CDP `firstCdp + i·step`. `lineBinning` projects the
## midpoint (`cmpX`, `cmpY`) onto a straight line from `(x0, y0)` along
## `azimuthDeg`: bin `i` is centred `i·binSize` along it and numbered CDP
## `firstCdp + i·cdpStep`. Midpoints further than `maxOffLine` from the
## line are in no bin.
##
## **3D.** Bins are numbered `inline · crosslines + crossline` (row-major,
## the inline index first), `len = inlines · crosslines`. `gridBinning`
## defines the grid by its origin, the centre of the first bin (inline
## `firstInline`, crossline `firstCrossline`); crossline numbers increase
## along `azimuthDeg`, every `crosslineSpacing`; inline numbers increase
## 90° clockwise of it (`clockwise = false`: anticlockwise), every
## `inlineSpacing`; the numbers go in steps of `inlineStep` and
## `crosslineStep`. `inlineCrosslineBinning` takes the keys' `inline` and
## `crossline` (parseSEG's layout fields).
##
## `gridBinning` takes its spacings crossline first and its counts and
## first numbers inline first: pass them by name, as above.
##
## **Conventions.** Azimuths are in degrees clockwise from north (+y), so
## 90° is +x. Distances are in the file's coordinate units. A trace falls
## in the nearest bin: bin `i` holds the positions from half a bin before
## its centre (included) to half a bin after it (excluded), so a midpoint
## exactly half-way goes to the higher one, and the first bin's lower edge
## is in, the last bin's upper edge out. A NaN key, or a position outside
## the bins, gives −1; no key gives a bin outside `0 ..< len`.

import std/math
import parseSEG
import ../core/types

type
  BinningKind* = enum
    ## How traces are assigned to bins.
    bnCdp             ## 2D, by the header CDP (`ensemble`)
    bnLine            ## 2D, by projecting the midpoint onto a line
    bnGrid            ## 3D, by a rotated grid of midpoints
    bnInlineCrossline ## 3D, by the header inline and crossline numbers

  Binning* = object
    ## A bin definition; build it with `cdpBinning`, `lineBinning`,
    ## `gridBinning` or `inlineCrosslineBinning`, which validate it.
    kind: BinningKind
    inlines, crosslines: int # 2D: 1 × count
    first0, step0: float64 # cdp, or inline numbering (3D)
    first1, step1: float64 # crossline numbering (3D)
    x0, y0: float64 # line or grid origin
    ux, uy: float64 # unit vector of the line / the crossline axis
    vx, vy: float64 # unit vector of the inline axis (3D)
    size0, size1: float64 # bin sizes: along the line / crossline, inline
    maxOff: float64

  BinLocation* = object
    ## What a bin stands for; NaN where the binning does not define it.
    cdp*: float64       ## 2D: the CDP number.
    inline*: float64    ## 3D: the inline number.
    crossline*: float64 ## 3D: the crossline number.
    x*, y*: float64     ## The bin centre (line and grid binnings).

proc badBinning(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, msg)

func finite(x: float64): bool {.inline.} = x.classify notin {fcNan, fcInf, fcNegInf}

proc checkFinite(x: float64, what: string) =
  if not finite(x): badBinning(what & " must be finite, got " & $x)

proc checkPositive(x: float64, what: string) =
  if not finite(x) or x <= 0: badBinning(what &
      " must be finite and positive, got " & $x)

proc checkCount(n: int, what: string) =
  if n <= 0: badBinning(what & " must be positive, got " & $n)

proc cdpBinning*(firstCdp: float64, count: int, step = 1.0): Binning =
  ## 2D bins by the header CDP: bin `i` holds CDP `firstCdp + i·step`,
  ## for `i in 0 ..< count`. Raises `ValueError` unless `firstCdp` is
  ## finite, `count` positive and `step` finite and positive.
  checkFinite(firstCdp, "firstCdp")
  checkCount(count, "count")
  checkPositive(step, "step")
  Binning(kind: bnCdp, inlines: 1, crosslines: count, first0: firstCdp,
    step0: step)

func unitAt(azimuthDeg: float64): (float64, float64) =
  let a = degToRad(azimuthDeg)
  (sin(a), cos(a))

proc lineBinning*(x0, y0, azimuthDeg, binSize: float64, count: int,
    firstCdp = 1.0, maxOffLine = Inf, cdpStep = 1.0): Binning =
  ## 2D bins along a straight line from `(x0, y0)` (the centre of bin 0)
  ## along `azimuthDeg`, `binSize` apart; bin `i` is numbered CDP
  ## `firstCdp + i·cdpStep`. A midpoint further than `maxOffLine` from the
  ## line is in no bin. Raises `ValueError` unless the coordinates, azimuth
  ## and `firstCdp` are finite, `binSize` and `cdpStep` finite and positive,
  ## `count` positive and `maxOffLine` positive.
  checkFinite(x0, "x0")
  checkFinite(y0, "y0")
  checkFinite(azimuthDeg, "azimuthDeg")
  checkFinite(firstCdp, "firstCdp")
  checkPositive(binSize, "binSize")
  checkPositive(cdpStep, "cdpStep")
  checkCount(count, "count")
  if maxOffLine.isNaN or maxOffLine <= 0:
    badBinning("maxOffLine must be positive, got " & $maxOffLine)
  let (ux, uy) = unitAt(azimuthDeg)
  Binning(kind: bnLine, inlines: 1, crosslines: count, first0: firstCdp,
    step0: cdpStep, x0: x0, y0: y0, ux: ux, uy: uy, size0: binSize,
    maxOff: maxOffLine)

proc gridBinning*(x0, y0, azimuthDeg, crosslineSpacing, inlineSpacing: float64,
    inlines, crosslines: int, firstInline = 1.0, firstCrossline = 1.0,
    clockwise = true, inlineStep = 1.0, crosslineStep = 1.0): Binning =
  ## 3D bins on a rotated grid: `(x0, y0)` is the centre of the bin of
  ## inline `firstInline`, crossline `firstCrossline`; crossline numbers
  ## increase along `azimuthDeg`, by `crosslineStep` every
  ## `crosslineSpacing`, inline numbers 90° clockwise of it (anticlockwise
  ## with `clockwise = false`), by `inlineStep` every `inlineSpacing`.
  ## Mind the order: the spacings come crossline first, the counts and the
  ## first numbers inline first; pass them by name. Raises `ValueError`
  ## unless the numbers are finite, the spacings and steps finite and
  ## positive, and the counts positive.
  checkFinite(x0, "x0")
  checkFinite(y0, "y0")
  checkFinite(azimuthDeg, "azimuthDeg")
  checkFinite(firstInline, "firstInline")
  checkFinite(firstCrossline, "firstCrossline")
  checkPositive(crosslineSpacing, "crosslineSpacing")
  checkPositive(inlineSpacing, "inlineSpacing")
  checkPositive(inlineStep, "inlineStep")
  checkPositive(crosslineStep, "crosslineStep")
  checkCount(inlines, "inlines")
  checkCount(crosslines, "crosslines")
  if inlines > high(int) div crosslines:
    badBinning("inlines × crosslines overflows")
  let (ux, uy) = unitAt(azimuthDeg)
  let (vx, vy) = unitAt(azimuthDeg + (if clockwise: 90.0 else: -90.0))
  Binning(kind: bnGrid, inlines: inlines, crosslines: crosslines,
    first0: firstInline, step0: inlineStep, first1: firstCrossline,
    step1: crosslineStep, x0: x0, y0: y0, ux: ux, uy: uy, vx: vx, vy: vy,
    size0: crosslineSpacing, size1: inlineSpacing)

proc inlineCrosslineBinning*(firstInline: float64, inlines: int,
    firstCrossline: float64, crosslines: int, inlineStep = 1.0,
    crosslineStep = 1.0): Binning =
  ## 3D bins by the header inline and crossline numbers: inline
  ## `firstInline + i·inlineStep`, crossline `firstCrossline +
  ## j·crosslineStep`. Raises `ValueError` unless the numbers are finite,
  ## the steps finite and positive, and the counts positive.
  checkFinite(firstInline, "firstInline")
  checkFinite(firstCrossline, "firstCrossline")
  checkPositive(inlineStep, "inlineStep")
  checkPositive(crosslineStep, "crosslineStep")
  checkCount(inlines, "inlines")
  checkCount(crosslines, "crosslines")
  if inlines > high(int) div crosslines:
    badBinning("inlines × crosslines overflows")
  Binning(kind: bnInlineCrossline, inlines: inlines, crosslines: crosslines,
    first0: firstInline, step0: inlineStep, first1: firstCrossline,
    step1: crosslineStep)

func kind*(b: Binning): BinningKind {.inline, raises: [].} =
  ## How traces are assigned.
  b.kind

func len*(b: Binning): int {.inline, raises: [].} =
  ## The number of bins (`inlines · crosslines`; 2D: `crosslines`).
  b.inlines * b.crosslines

func inlines*(b: Binning): int {.inline, raises: [].} =
  ## Inlines in a 3D binning; 1 in 2D.
  b.inlines

func crosslines*(b: Binning): int {.inline, raises: [].} =
  ## Crosslines in a 3D binning; the bin count in 2D.
  b.crosslines

func is3D*(b: Binning): bool {.inline, raises: [].} =
  ## The binning is a 3D grid.
  b.kind in {bnGrid, bnInlineCrossline}

func nearest(u: float64, n: int): int {.inline.} =
  # The index of the nearest bin to position `u` (in bins): bin i holds
  # [i − 0.5, i + 0.5), and −1 is outside [−0.5, n − 0.5) or NaN.
  # floor(u + 0.5) is not it: one ulp below an upper edge the sum rounds up
  # to the next integer (pred(0.5) + 0.5 == 1.0), which gave the next bin
  # and, on the last one, bin n. u − floor(u) is exact, so the half-way test
  # is; the `min` only guards an n beyond float64's integers.
  if not (u >= -0.5 and u < float64(n) - 0.5): return -1
  let whole = floor(u)
  min(int(whole) + int(u - whole >= 0.5), n - 1)

func gridBin(b: Binning, u, v: float64): int {.inline.} =
  let i = nearest(v, b.inlines)
  let j = nearest(u, b.crosslines)
  if i < 0 or j < 0: -1 else: i * b.crosslines + j

func binAt*(b: Binning, x, y: float64): int {.raises: [].} =
  ## The bin of a midpoint at `(x, y)` in a line or grid binning, or −1
  ## when it falls in none (NaN, outside the bins, too far from the line);
  ## always −1 for the header binnings (`bnCdp`, `bnInlineCrossline`).
  ## `binOf` uses the same maths.
  let dx = x - b.x0
  let dy = y - b.y0
  case b.kind
  of bnLine:
    if not (abs(dx * b.uy - dy * b.ux) <= b.maxOff): return -1
    nearest((dx * b.ux + dy * b.uy) / b.size0, b.crosslines)
  of bnGrid:
    b.gridBin((dx * b.ux + dy * b.uy) / b.size0, (dx * b.vx + dy * b.vy) / b.size1)
  of bnCdp, bnInlineCrossline: -1

func binOf*(b: Binning, k: TraceKeys): int {.raises: [].} =
  ## The bin of a trace with keys `k`, or −1 when it falls in none (a NaN
  ## key, outside the bins, too far from the line).
  case b.kind
  of bnCdp:
    nearest((k.ensemble - b.first0) / b.step0, b.crosslines)
  of bnLine, bnGrid:
    b.binAt(k.cmpX, k.cmpY)
  of bnInlineCrossline:
    b.gridBin((k.crossline - b.first1) / b.step1, (k.inline - b.first0) / b.step0)

proc binsInto*(rows: TraceRows, b: Binning, bins: var openArray[int]) =
  ## Fills `bins[i]` with row `i`'s bin (−1 for none), for every row
  ## whatever its status. Raises `ValueError` when the view has no keys or
  ## `bins.len != rows.len`. Allocates nothing.
  if not rows.hasKeys:
    raise newException(ValueError, "the rows have no keys: open the file with " &
      "keys = true (or call ensureKeys) before making the view")
  if bins.len != rows.len:
    raise newException(ValueError, "bins has " & $bins.len &
      " elements, the rows " & $rows.len)
  for i in 0 ..< rows.len:
    bins[i] = b.binOf(rows.keysAt(i))

proc binsInto*(meta: ChunkMeta, b: Binning, bins: var openArray[int]) =
  ## `binsInto(meta.rows, b, bins)`.
  binsInto(meta.rows, b, bins)

proc location*(b: Binning, bin: int): BinLocation =
  ## What bin `bin` stands for: its CDP number (2D), its inline and
  ## crossline numbers (3D) and its centre (line and grid binnings); NaN
  ## for what the binning does not define. Raises `ValueError` for a bin
  ## outside `0 ..< len`.
  if bin < 0 or bin >= b.len:
    badBinning("bin " & $bin & " is outside 0 ..< " & $b.len)
  result = BinLocation(cdp: NaN, inline: NaN, crossline: NaN, x: NaN, y: NaN)
  case b.kind
  of bnCdp:
    result.cdp = b.first0 + float64(bin) * b.step0
  of bnLine:
    result.cdp = b.first0 + float64(bin) * b.step0
    result.x = b.x0 + float64(bin) * b.size0 * b.ux
    result.y = b.y0 + float64(bin) * b.size0 * b.uy
  of bnGrid, bnInlineCrossline:
    let i = bin div b.crosslines
    let j = bin mod b.crosslines
    result.inline = b.first0 + float64(i) * b.step0
    result.crossline = b.first1 + float64(j) * b.step1
    if b.kind == bnGrid:
      let a = float64(j) * b.size0
      let c = float64(i) * b.size1
      result.x = b.x0 + a * b.ux + c * b.vx
      result.y = b.y0 + a * b.uy + c * b.vy
