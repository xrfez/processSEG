## Fold maps of a survey: the bin grid of its pre-plot (SPS) geometry.
##
## ```nim
## var g = initSurveyGeometry()                   # parseSEG
## for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
## let grid = surveyGrid(g)                       # natural bins at the receiver azimuth
## let fine = surveyGrid(g, binInline = 25, binCrossline = 25)
## ```
##
## **Grid.** `surveyGrid` builds a `gridBinning` (`stack/binning`) whose
## inlines run along the receiver lines: crossline numbers increase along
## the azimuth, inline numbers 90° clockwise of it, both from 1. Its bins
## cover the midpoint of every source–receiver pair of the geometry (so
## every pre-plot midpoint, whatever the X records say): along each grid
## axis a midpoint `(s + r) / 2` lies between half the sum of the source and
## receiver minima and half the sum of their maxima, which is the grid's
## extent. Only stations with coordinates count. The grid's origin is that
## minimum corner, a midpoint itself on a regular survey, so bins are centred
## on the midpoints there.
##
## **Defaults.** The azimuth is the receiver-line azimuth: the median
## bearing between consecutive stations of a receiver line, in point order
## (as `survey/patch.receiverLayout`). The *natural bin* is half the median
## receiver interval along the receiver lines (`binInline`) by half the
## median source interval across them (`binCrossline`), each interval the
## median distance between consecutive points of a line. That is the
## natural bin of an orthogonal survey (source lines across the receiver
## lines); for other layouts give `binCrossline`. Any argument given
## overrides its default, and a caller's own `Binning` works everywhere a
## `surveyGrid` does.

import std/[algorithm, math, tables]
import parseSEG
import ../stack/binning

func finite(x: float64): bool {.inline.} = x.classify notin {fcNan, fcInf, fcNegInf}

func median(v: var seq[float64]): float64 =
  if v.len == 0: return NaN
  v.sort
  v[(v.len - 1) div 2]

proc lineSteps(t: Table[StationKey, Coordinates], bearings, steps: var seq[float64]) =
  ## Bearings and distances between consecutive stations of each line, in
  ## point (then index) order; stations without coordinates are skipped.
  var byLine: Table[float64, seq[(float64, int16, float64, float64)]]
  for k, c in t:
    if finite(c.x) and finite(c.y):
      byLine.mgetOrPut(k.line, @[]).add (k.point, k.index, c.x, c.y)
  for st in byLine.mvalues:
    st.sort(proc (a, b: (float64, int16, float64, float64)): int =
      result = cmp(a[0], b[0])
      if result == 0: result = cmp(a[1], b[1]))
    for i in 1 ..< st.len:
      let (dx, dy) = (st[i][2] - st[i - 1][2], st[i][3] - st[i - 1][3])
      let d = hypot(dx, dy)
      if d > 0:
        steps.add d
        bearings.add floorMod(radToDeg(arctan2(dx, dy)), 360.0)

func medianBearing(bearings: var seq[float64]): float64 =
  # Unwrapped around one bearing, so 359° and 1° agree.
  let reference = bearings[0]
  for b in bearings.mitems: b = reference + floorMod(b - reference + 180.0, 360.0) - 180.0
  floorMod(median(bearings), 360.0)

proc extent(t: Table[StationKey, Coordinates], azimuthDeg: float64,
    lo, hi: var array[2, float64]) =
  ## Min and max of the stations along (0) and across (1) the azimuth.
  let (s, c) = (sin(degToRad(azimuthDeg)), cos(degToRad(azimuthDeg)))
  lo = [Inf, Inf]
  hi = [-Inf, -Inf]
  for p in t.values:
    if finite(p.x) and finite(p.y):
      let a = p.x * s + p.y * c
      let x = p.x * c - p.y * s
      lo = [min(lo[0], a), min(lo[1], x)]
      hi = [max(hi[0], a), max(hi[1], x)]

proc badGrid(msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, "surveyGrid: " & msg)

proc surveyGrid*(g: SurveyGeometry, binInline = NaN, binCrossline = NaN,
    azimuthDeg = NaN): Binning =
  ## The bin grid covering every source–receiver midpoint of `g` (see the
  ## module documentation): inlines along `azimuthDeg`, bins `binInline`
  ## along them (`gridBinning`'s `crosslineSpacing`) by `binCrossline`
  ## across (its `inlineSpacing`). NaN arguments take their defaults: the
  ## receiver-line azimuth, half the receiver interval, half the source
  ## interval. Raises `ValueError` for a given value that is not finite (or
  ## not positive, for the bins), when there are no receivers or no sources
  ## with coordinates, or when a default cannot be measured (no line with
  ## two stations at distinct positions).
  for (v, name) in [(binInline, "binInline"), (binCrossline, "binCrossline")]:
    if not v.isNaN and not (v > 0 and v < Inf):
      badGrid(name & " must be > 0 and finite, got " & $v)
  if not azimuthDeg.isNaN and not finite(azimuthDeg):
    badGrid("azimuthDeg must be finite, got " & $azimuthDeg)
  var azimuth = azimuthDeg
  var binI = binInline
  var binC = binCrossline
  if azimuth.isNaN or binI.isNaN:
    var bearings, steps: seq[float64]
    lineSteps(g.table.receivers, bearings, steps)
    if steps.len == 0:
      badGrid("no receiver line has two stations with coordinates; give " &
        (if azimuth.isNaN: "azimuthDeg" else: "binInline"))
    if azimuth.isNaN: azimuth = medianBearing(bearings)
    if binI.isNaN: binI = median(steps) / 2
  if binC.isNaN:
    var bearings, steps: seq[float64]
    lineSteps(g.table.sources, bearings, steps)
    if steps.len == 0:
      badGrid("no source line has two points with coordinates; give binCrossline")
    binC = median(steps) / 2
  var rLo, rHi, sLo, sHi: array[2, float64]
  extent(g.table.receivers, azimuth, rLo, rHi)
  extent(g.table.sources, azimuth, sLo, sHi)
  if rLo[0] == Inf: badGrid("no receiver has coordinates")
  if sLo[0] == Inf: badGrid("no source has coordinates")
  let lo = [(rLo[0] + sLo[0]) / 2, (rLo[1] + sLo[1]) / 2]
  let hi = [(rHi[0] + sHi[0]) / 2, (rHi[1] + sHi[1]) / 2]
  # Nearest-bin assignment: position u (in bins) needs u < n − 0.5.
  let crosslines = int(floor((hi[0] - lo[0]) / binI + 0.5)) + 1
  let inlines = int(floor((hi[1] - lo[1]) / binC + 0.5)) + 1
  let (s, c) = (sin(degToRad(azimuth)), cos(degToRad(azimuth)))
  let x0 = lo[0] * s + lo[1] * c
  let y0 = lo[0] * c - lo[1] * s
  gridBinning(x0, y0, azimuth, crosslineSpacing = binI, inlineSpacing = binC,
    inlines = inlines, crosslines = crosslines)
