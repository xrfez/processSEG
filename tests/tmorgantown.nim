## The survey QC example on real data (task 11.6): the Morgantown 3D SPS
## files and one of its shots, `00000151_12900_11300.sgd`, from the survey
## files to a QC report: the station join (its FFID points at another
## shot), the record's completeness against the XPS, its fold against the
## XPS record's, and amplitude QC of its traces. Skipped when the files
## (gitignored) are absent.

import std/[assertions, os, strutils, times]
import parseSEG
import processSEG

const
  dir = currentSourcePath().parentDir.parentDir / "exampleFiles"
  base = dir / "sps" / "PGI_NNE_West_Morgantown_3D_Post261019_NAD27_SPCS_WV_N_USFeet"
  shotFile = dir / "00000151_12900_11300.sgd"

proc ms(t0: float): string = formatFloat((cpuTime() - t0) * 1000, ffDecimal, 2) & " ms"

block morgantown_example:
  if not (fileExists(base & ".sps") and fileExists(base & ".rps") and
      fileExists(base & ".xps") and fileExists(shotFile)):
    echo "  skipped: Morgantown files not present"
  else:
    var g = initSurveyGeometry()
    for ext in [".sps", ".rps", ".xps"]: g.addSps(base & ext)
    doAssert not g.log.hasErrors
    doAssert g.gridUnit.measurementSystem == 2 # H20 feet, H201 0.3048006096 (US survey)
    var f = openSeismic(shotFile, keys = true)
    defer: f.close()

    # 1. Join by station: every live trace carries both stations.
    var stats: JoinStats
    discard f.applySurvey(g, stats)
    echo "  join: ", stats.matches[jmStations], " by stations, ",
      stats.matches[jmRecord], " by record, ", stats.matches[jmAuxiliary], " auxiliary"
    doAssert stats.matches[jmStations] == 3954 and stats.matches[jmRecord] == 0
    doAssert stats.noSourceXy == 0 and stats.noReceiverXy == 0

    # 2. Completeness: XPS 77 by the source station, not record 151.
    let rows = f.rowsOf(0 ..< f.len)
    var t0 = cpuTime()
    let c = g.recordCompleteness(rows)
    echo "  completeness (", ms(t0), "): ", c
    doAssert c.xpsRecord == 77 and c.recordMismatch and not c.sourceMismatch
    doAssert c.expected == 1127 and c.present == 1096 and c.missing == 31
    doAssert c.unlisted == 2858 and c.aux == 3

    # 3. Fold of the file against the pre-plot fold of XPS 77 alone.
    let grid = surveyGrid(g)
    var xps77 = initSurveyGeometry()
    for ext in [".sps", ".rps"]: xps77.addSps(base & ext)
    var x = readSps(base & ".xps")
    var kept: seq[SpsRelation]
    for r in x.relations:
      if r.fieldRecord == 77: kept.add r
    xps77.addSps(SpsFile(relations: kept))
    let pre = xps77.preplotFold(grid)
    var actual = initFoldMap(grid)
    actual.addFold(rows)
    let diff = foldDifference(pre, actual)
    var missing, extra = 0
    for v in diff:
      if v > 0: missing += v else: extra -= v
    echo "  fold: pre-plot ", pre.binned, ", actual ", actual.binned, " (", actual.notLive,
      " not live); pre − actual: ", missing, " missing, ", extra, " extra"
    doAssert pre.binned == 1127 and actual.binned == 3954
    doAssert missing == 31 and extra == 2858 # the stations of step 2, bin by bin

    # 4. Amplitude QC of the record.
    var cols = 0
    for i in 0 ..< f.len: cols = max(cols, int(f.traceEntry(i).numSamples))
    var samples = zeros[float32](f.len, cols)
    var failed: seq[int]
    discard f.readTracesInto(0 ..< f.len, samples, failed)
    let meta = chunkMeta(f, 0 ..< f.len, failed = failed)
    var amps = newSeq[TraceAmplitude](f.len)
    var pool = newProcessPool(4)
    t0 = epochTime()
    pool.traceAmplitudeInto(samples, meta, AmplitudeParams(), amps)
    let poolMs = (epochTime() - t0) * 1000
    var ws = initWorkspace()
    ws.reserve(recordAmplitudeWorkspace(f.len))
    let rec = recordAmplitude(amps, AmplitudeParams(), ws)
    var counts: seq[string]
    for fl in AmplitudeFlag:
      if rec.counts[fl] > 0: counts.add $fl & " " & $rec.counts[fl]
    echo "  amplitude (", formatFloat(poolMs, ffDecimal, 1), " ms, pool of 4, ", f.len,
      " × ", cols, "): median RMS ", formatFloat(rec.medianRmsDb, ffDecimal, 1), " dB, MAD ",
      formatFloat(rec.madDb, ffDecimal, 1), " dB; ", rec.flagged, " flagged: ", counts.join(", ")
    doAssert rec.traces > 3900 and rec.traces <= 3954
    doAssert rec.flagged < rec.traces div 10 # QC flags the exceptions, not the record
