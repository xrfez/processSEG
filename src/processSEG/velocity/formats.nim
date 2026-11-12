## Reading velocity picks: Western VELF cards, HANDVEL blocks and CSV, into
## `VelocityPick`s (`velocity/model`), ready for `initVelocityField`.
##
## ```nim
## let picks = readVelocities("line12.velf")     # detects the format
## let field = initVelocityField(picks, dtMs = 4, tMaxMs = 6000)
## ```
##
## Every pick's pairs are RMS knots `(tMs, vrms)` (see `rmsVelocity`); a
## time that does not increase is reported with its line and location.
## Errors raise `ValueError` naming the line (1-based). Lines starting with
## `*` or `#`, and blank lines, are comments everywhere.
##
## **Refused values.** A location (CDP, inline, crossline) that is NaN or
## infinite is refused in every format. So is a velocity below
## `MinPickVelocity` (100 distance units per second, under any medium's in
## m/s or ft/s): it means a misread file, not a slow rock. The usual cause
## is a CSV written with decimal commas (`100,0,1500,5` is CDP 100, 0 ms,
## 1500.5 m/s, and reads as a 3D pick of 5 m/s at 1500 ms); others are a
## time and a velocity swapped, or velocities in km/s (convert them: the
## library works in the file's coordinate units per second). Velocity
## functions built in code (`rmsVelocity`) have no such bound.
##
## **HANDVEL** (ProMAX-style export): a `HANDVEL` card with the location,
## one number (a CDP) or two (inline, crossline), followed by lines of
## `t v` pairs, until the next `HANDVEL` or any other card:
##
##     HANDVEL     1001
##             0     1500      500     1800
##          1000     2100
##
## **VELF** (Western cards): `VELF` cards of whitespace-separated numbers.
## A card with an odd count starts a new function, its first number the
## CDP; a card with an even count continues the previous one. CDPs must
## increase through the file. That catches a continuation card that lost a
## number (its first time reads as a CDP) when a function with a smaller
## CDP follows it. It cannot catch one that no function follows (the
## file's last card) or whose first time is below the next CDP: the card
## then reads as one more function, at a "CDP" that is a time. The count of
## picks and their CDPs show it; compare them with the line's. Other cards
## (`SPNT`, `LINE`, …) are skipped. Fixed-column cards whose fields touch
## (a 5-digit time next to a 5-digit velocity) and card sequence numbers in
## columns 73–80 are not supported:
##
##     VELF       101       0   1500    500   1800   1000   2100
##     VELF                2000   2500
##
## **CSV**: comma-separated `cdp,tMs,vrms` (2D) or
## `inline,crossline,tMs,vrms` (3D), one knot per line, with an optional
## header line (a first line with a field that is a word: `cdp`, not `nan`
## or an empty field). Rows of one location may
## be anywhere in the file; their times must increase in file order.

import std/[math, strutils, tables]
import model

const MinPickVelocity* = 100.0
  ## The least velocity a pick file may hold, in distance units per second:
  ## the parsers refuse a smaller one as a misread file (see the module
  ## documentation).

proc parseError(line: int, msg: string) {.noinline, noreturn.} =
  raise newException(ValueError, "line " & $line & ": " & msg)

func isComment(s: string): bool =
  s.len == 0 or s[0] in {'*', '#'}

func isNumber(t: string): bool =
  try:
    discard parseFloat(t)
    true
  except ValueError:
    false

func isCard(t: string): bool =
  ## A card name (`LINE`, `END`): a word that is not a number, so `nan` and
  ## `inf` are not cards.
  t.len > 0 and t[0] in Letters and not t.isNumber

proc numbers(tokens: openArray[string], line: int): seq[float64] =
  for t in tokens:
    try:
      result.add parseFloat(t)
    except ValueError:
      parseError(line, "'" & t & "' is not a number")

func num(x: float64): string =
  ## `x` as written in a velocity file: 1001, not 1001.0.
  if x == floor(x) and abs(x) < 1e15: $int64(x) else: $x

func where(cdp, il, xl: float64): string =
  if il.isNaN: "cdp " & num(cdp) else: "inline " & num(il) & ", crossline " & num(xl)

type Builder = object
  ## The pick being read: its location, its knots and its first line.
  cdp, il, xl: float64
  line: int
  knots: seq[VelocityKnot]
  open: bool
  csv: bool ## From a CSV: a too small velocity is most likely a decimal comma.

proc addKnot(b: var Builder, tMs, vrms: float64, line: int) =
  ## Checks the knot here, so that an error names its own line.
  let at = where(b.cdp, b.il, b.xl)
  if tMs.classify in {fcNan, fcInf, fcNegInf} or tMs < 0:
    parseError(line, at & ": a time must be finite and ≥ 0, got " & $tMs)
  if b.knots.len > 0 and not (tMs > b.knots[^1].tMs):
    parseError(line, at & ": time " & num(tMs) & " after " & num(b.knots[
        ^1].tMs) &
      " (times must increase)")
  if vrms.classify in {fcNan, fcInf, fcNegInf} or vrms <= 0:
    parseError(line, at & ": a velocity must be finite and positive, got " & $vrms)
  if vrms < MinPickVelocity:
    parseError(line, at & ": a velocity of " & num(vrms) & " is below " &
      num(MinPickVelocity) & ": " & (if b.csv: "a decimal comma (fields are " &
      "separated by commas, decimals written with a point), " else: "") &
      "a time and a velocity swapped, or velocities not in distance units " &
      "per second?")
  b.knots.add VelocityKnot(tMs: tMs, vrms: vrms)

proc addPairs(b: var Builder, xs: openArray[float64], line: int) =
  if xs.len mod 2 != 0:
    parseError(line, "an odd count of numbers (" & $xs.len &
      "): times and velocities come in pairs")
  for i in countup(0, xs.len - 2, 2):
    b.addKnot(xs[i], xs[i + 1], line)

proc finish(b: var Builder, picks: var seq[VelocityPick]) =
  if not b.open: return
  b.open = false
  let at = where(b.cdp, b.il, b.xl)
  if b.knots.len == 0:
    parseError(b.line, at & " has no time-velocity pairs")
  try:
    picks.add VelocityPick(cdp: b.cdp, inline: b.il, crossline: b.xl,
      velocity: rmsVelocity(b.knots))
  except ValueError as e:
    parseError(b.line, at & ": " & e.msg)
  b.knots.setLen 0

proc start(b: var Builder, loc: openArray[float64], line: int) =
  # A NaN or infinite location is not a place: `nan` and `1e400` parse as
  # numbers, and would only fail later, in the field, without a line.
  for x in loc:
    if x.isNaN: parseError(line, "a NaN location")
    if x.classify in {fcInf, fcNegInf}: parseError(line, "an infinite location")
  b.open = true
  b.line = line
  b.cdp = NaN
  b.il = NaN
  b.xl = NaN
  case loc.len
  of 1: b.cdp = loc[0]
  of 2: (b.il = loc[0]; b.xl = loc[1])
  else: parseError(line, "a location is one number (cdp) or two (inline, " &
      "crossline), got " & $loc.len)

func withoutBom(text: string): string =
  ## `text` without a leading UTF-8 byte-order mark (editors on Windows write
  ## one).
  if text.startsWith("\xEF\xBB\xBF"): text[3 .. ^1] else: text

proc parseHandvel*(text: string): seq[VelocityPick] =
  ## HANDVEL blocks (see the module documentation). Raises `ValueError` for
  ## a bad number, odd pairs, a block without pairs, numbers before the first
  ## `HANDVEL` card, or an invalid function.
  let text = text.withoutBom
  var b: Builder
  var lineNo = 0
  for raw in text.splitLines:
    inc lineNo
    let s = raw.strip
    if s.isComment: continue
    let tokens = s.splitWhitespace
    if tokens[0].cmpIgnoreCase("HANDVEL") == 0:
      b.finish(result)
      b.start(numbers(tokens.toOpenArray(1, tokens.high), lineNo), lineNo)
    elif tokens[0].isCard:
      b.finish(result)
    elif not b.open:
      parseError(lineNo, "numbers outside a HANDVEL block")
    else:
      b.addPairs(numbers(tokens, lineNo), lineNo)
  b.finish(result)

proc parseVelf*(text: string): seq[VelocityPick] =
  ## VELF cards (see the module documentation). Raises `ValueError` for a
  ## bad number, a continuation card before the first function, a function
  ## without pairs, or an invalid function.
  let text = text.withoutBom
  var b: Builder
  var lineNo = 0
  for raw in text.splitLines:
    inc lineNo
    let s = raw.strip
    if s.isComment: continue
    let tokens = s.splitWhitespace
    if tokens[0].cmpIgnoreCase("VELF") != 0: continue
    let xs = numbers(tokens.toOpenArray(1, tokens.high), lineNo)
    if xs.len == 0: continue # a bare card
    if xs.len mod 2 == 1:
      if b.open and xs[0] <= b.cdp: # a NaN cdp: reported by `start`
        parseError(lineNo, "cdp " & num(xs[0]) & " after cdp " & num(b.cdp) &
          ": VELF cdps must increase (or is this a continuation card with an " &
          "odd count of numbers?)")
      b.finish(result)
      b.start(xs.toOpenArray(0, 0), lineNo)
      b.addPairs(xs.toOpenArray(1, xs.high), lineNo)
    elif not b.open:
      parseError(lineNo, "a VELF continuation card (an even count of " &
        "numbers) before the first function")
    else:
      b.addPairs(xs, lineNo)
  b.finish(result)

proc parseVelocityCsv*(text: string): seq[VelocityPick] =
  ## CSV knots (see the module documentation). Raises `ValueError` for a
  ## line with another column count, a bad number, or an invalid function
  ## (times not increasing within one location).
  let text = text.withoutBom
  var builders: seq[Builder]
  var index = initTable[(float64, float64), int]()
  var cols = 0
  var lineNo = 0
  for raw in text.splitLines:
    inc lineNo
    let s = raw.strip
    if s.isComment: continue
    var fields = s.split(',')
    for f in fields.mitems: f = f.strip
    if cols == 0:
      if fields.len notin [3, 4]:
        parseError(lineNo, "a velocity CSV has 3 columns (cdp,tMs,vrms) or 4 " &
          "(inline,crossline,tMs,vrms), got " & $fields.len)
      cols = fields.len
      var header = false
      for f in fields:
        if f.isCard: header = true # a name, not a number or an empty field
      if header: continue
    if fields.len != cols:
      parseError(lineNo, $fields.len & " columns, the file has " & $cols)
    let xs = numbers(fields, lineNo)
    let key = if cols == 3: (xs[0], 0.0) else: (xs[0], xs[1])
    var i = -1
    if not (key[0].isNaN or key[1].isNaN): # NaN ≠ NaN as a key: `start` reports it
      i = index.getOrDefault(key, -1)
    if i < 0:
      i = builders.len
      builders.add Builder(csv: true)
      builders[i].start(xs.toOpenArray(0, cols - 3), lineNo)
      index[key] = i
    builders[i].addKnot(xs[cols - 2], xs[cols - 1], lineNo)
  for b in builders.mitems:
    b.finish(result)

proc parseVelocities*(text: string): seq[VelocityPick] =
  ## Picks in any of the three formats: HANDVEL when a `HANDVEL` card comes
  ## first, VELF when a `VELF` card does, CSV otherwise. Raises `ValueError`
  ## as the format's parser.
  let text = text.withoutBom
  for raw in text.splitLines:
    let s = raw.strip
    if s.isComment: continue
    let first = s.splitWhitespace[0]
    if first.cmpIgnoreCase("HANDVEL") == 0: return parseHandvel(text)
    if first.cmpIgnoreCase("VELF") == 0: return parseVelf(text)
    if first.isCard and ',' notin s: continue # another card (LINE, SPNT)
    break
  parseVelocityCsv(text)

proc readVelocities*(path: string): seq[VelocityPick] =
  ## `parseVelocities` on the file at `path`. Raises `IOError` when it
  ## cannot be read, and `ValueError` as the parsers.
  parseVelocities(readFile(path))
