## Fuzz target: the velocity file parsers (VELF, HANDVEL, CSV, auto-detect)
## on arbitrary text, then a field from the picks and its lookups. Bad text
## must raise `ValueError`; anything else is a finding.

import std/math
import processSEG
import fuzzutil

fuzzTarget:
  var text = newString(len)
  if len > 0: copyMem(addr text[0], data, len)
  for parse in [parseVelf, parseHandvel, parseVelocityCsv, parseVelocities]:
    try:
      let picks = parse(text)
      for p in picks:
        # A pick has a finite place, 2D or 3D, and plausible velocities.
        let at = if p.cdp.isNaN: p.inline + p.crossline else: p.cdp
        doAssert at.classify notin {fcNan, fcInf, fcNegInf}
        for k in p.velocity.knots:
          doAssert not k.tMs.isNaN and k.vrms >= MinPickVelocity
        discard p.velocity.vrmsAt(1000)
        discard p.velocity.vintAt(1000)
      let field = initVelocityField(picks, dtMs = 4, tMaxMs = 2000)
      if field.kind == fk2D:
        var v = newSeq[float64](50)
        field.vrmsInto(field.locations[0] + 0.5, 0, 4, v)
    except ValueError:
      discard
