## The station-first survey join on a real survey (parseSEG `survey/join`):
## the Morgantown 3D SPS set (S, R, X) and one of its SEG-D shots, whose FFID
## (151) is another shot's X record and whose trace numbers are not the X
## file's channels. Skipped when the example files are absent (they are not
## in the repository). The X → O inversion runs in release builds only.

import std/[assertions, math, options, os, tables]
when defined(release) or defined(danger): import std/times
import parseSEG

const
  dir = currentSourcePath().parentDir.parentDir / "exampleFiles"
  base = dir / "sps" / "PGI_NNE_West_Morgantown_3D_Post261019_NAD27_SPCS_WV_N_USFeet"
  shot = dir / "00000151_12900_11300.sgd"

proc present(): bool =
  for p in [base & ".sps", base & ".rps", base & ".xps", shot]:
    if not fileExists(p): return false
  true

if not present():
  echo "  skipped: Morgantown example files not present"
else:
  var g = initSurveyGeometry()
  for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
  doAssert not g.log.hasErrors
  doAssert g.sourceCount == 5519 and g.receiverCount == 6385
  let src = stationKey(129, 113, 0)

  block lookups:
    doAssert g.recordsAt(src) == @[77'i64], "shot 129/113 is X record 77, not its FFID 151"
    doAssert g.expectedChannels(77) == 1127
    doAssert g.relation(151, 1).get.source == stationKey(511, 117, 1)
    var records = 0
    for r in g.records:
      inc records
      if r mod 50 == 0: # iteration equals `relation`, every channel of a sample of records
        var n = 0
        for (c, rel) in g.relations(r):
          doAssert g.relation(r, c).get == rel
          doAssert g.channelOf(r, rel.receiver) == some(c)
          inc n
        doAssert n == g.expectedChannels(r)
    doAssert records == 5519

  block join_by_stations:
    # The SEG-D Rev 3 headers already hold coordinates; with `overwrite` the
    # SPS ones replace them. Either way the stations are the header's.
    for overwrite in [false, true]:
      var f = openSeismic(shot, keys = true)
      var stats: JoinStats
      let changed = f.applySurvey(g, stats, overwrite)
      doAssert stats.matches[jmStations] == 3954 and stats.matches[jmAuxiliary] == 3, $stats
      doAssert stats.noSourceXy == 0 and stats.noReceiverXy == 0, $stats
      var worst = 0.0
      for i in 0 ..< f.len:
        let k = f.index.keys[i]
        if k.status != tsAuxiliary:
          doAssert k.srcLine == 129 and k.srcPoint == 113, "never record 151's shot"
          let c = g.table.receivers[stationKey(k.rcvLine, k.rcvPoint, 1)]
          worst = max(worst, hypot(k.rcvX - c.x, k.rcvY - c.y))
          if overwrite:
            doAssert abs(k.srcX - 1834704.4) < 0.01 and abs(k.srcY - 445679.5) < 0.01
            doAssert k.rcvX == c.x and k.rcvY == c.y
      if overwrite: doAssert worst == 0
      echo "  overwrite = ", overwrite, ": ", stats, "; receiver X/Y vs RPS within ",
        round(worst, 2), " ft"
      if not overwrite: doAssert changed == stats.changed
      f.close()

  block header_without_receiver_station:
    # Filled from the source station's X record (77), not from the FFID's (151).
    var f = openSeismic(shot, keys = true)
    var tested = 0
    for i in 0 ..< f.len:
      var k = f.index.keys[i]
      if k.status == tsAuxiliary: continue
      let ch = int64(k.channel)
      let (a, b) = (g.relation(77, ch), g.relation(151, ch))
      if a.isSome and b.isSome and a.get.receiver != b.get.receiver:
        doAssert k.record == 151
        (k.rcvLine, k.rcvPoint) = (NaN, NaN)
        doAssert k.joinStations(g) == jmSource
        doAssert stationKey(k.rcvLine, k.rcvPoint, k.rcvIndex) == a.get.receiver
        inc tested
    doAssert tested > 100, $tested
    f.close()

  when defined(release) or defined(danger):
    block receiver_domain:
      let t0 = epochTime()
      let rels = g.receiverRelations()
      let built = epochTime() - t0
      var o = initSurveyGeometry()
      o.addSps(SpsFile(revision: sps21, relations: rels)) # O records
      # X → O → pairs equals X's pairs: the same count, and every O pair is
      # listed by an X record of its source.
      var xPairs, oPairs = 0
      for r in g.records: xPairs += g.expectedChannels(r)
      var gathers = 0
      for r in o.records(rdReceiver):
        inc gathers
        for (_, rel) in o.relations(r, rdReceiver):
          inc oPairs
          if r mod 97 == 0:
            var listed = false
            for x in g.recordsAt(rel.source):
              if g.channelOf(x, rel.receiver).isSome: listed = true
            doAssert listed
      doAssert oPairs == xPairs, $oPairs & " vs " & $xPairs
      doAssert gathers <= g.receiverCount
      let text = spsText(SpsFile(revision: sps21, relations: rels))
      doAssert not text.log.hasErrors
      echo "  Morgantown: ", xPairs, " pairs, ", gathers, " receiver gathers, ", rels.len,
        " O records (", text.text.len div 1_000_000, " MB) built in ", round(built, 2), " s"
