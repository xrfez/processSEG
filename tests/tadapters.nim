## Adapters (processSEG/adapters): `chunkMeta` from prefetch chunks and from
## `readTracesInto`, on parseSEG's demo files: SEG-Y, traces past the end of a file,
## sample windows, SPS-joined SEG-D Rev 2 and a mixed-interval SEG-D file.

import std/[assertions, math, os, strutils]
import parseSEG
import processSEG
import utils/[allocs, demo]

let tmp = getTempDir() / "processseg_tadapters"
removeDir(tmp)
createDir(tmp)

block prefetch_chunks_of_segy:
  var f = openSeismic(demoDir() / "survey.sgy", keys = true)
  defer: f.close()
  var p = newPrefetcher[float32](f, 0 ..< f.len, chunkTraces = 7,
      uniform = true)
  defer: p.close()
  var c: PrefetchChunk[float32]
  var seen = 0
  while p.next(c):
    let meta = chunkMeta(c)
    doAssert meta.len == c.count and meta.dtMs == 2 and meta.firstSample == 0
    doAssert meta.failedCount == 0
    meta.checkShape(c.samples)
    for row in 0..<meta.len:
      doAssert meta.keys(row).offset == f.traceKeys(c.first + row).offset
      doAssert meta.liveSamples(row, c.samples.shape[1]) ==
        int(f.traceEntry(c.first + row).numSamples)
    seen += c.count
  doAssert seen == f.len

block a_chunk_meta_does_not_allocate:
  var f = openSeismic(demoDir() / "survey.sgy", keys = true)
  defer: f.close()
  var p = newPrefetcher[float32](f, 0 ..< f.len, chunkTraces = 16)
  defer: p.close()
  var c: PrefetchChunk[float32]
  doAssert p.next(c)
  var total = 0
  let n = allocsIn:
    for i in 0..<100:
      total += chunkMeta(c).len
  doAssert n == 0, "chunkMeta allocated " & $n & " times"
  doAssert total == 100 * c.count

block failed_rows_of_a_truncated_file:
  # Opening a file cut short drops its incomplete last trace from the index,
  # so the truncation is simulated as parseSEG's tests do it: three traces
  # whose samples lie past the end of the file decode as zero rows.
  var f = openSeismic(demoDir() / "survey.sgy", keys = true)
  defer: f.close()
  let cols = int(f.traceEntry(0).numSamples) # before the entries change
  for t in [3, 4, 11]: f.index.entries[t].numSamples = 1_000_000_000
  let n = 12
  var p = newPrefetcher[float32](f, 0 ..< n, chunkTraces = n, cols = cols)
  defer: p.close()
  var c: PrefetchChunk[float32]
  doAssert p.next(c)
  doAssert c.failedRows == @[3, 4, 11]
  let fromChunk = chunkMeta(c)
  doAssert fromChunk.failedCount == c.failed
  # readTracesInto reports the same rows (parseSEG patch F2).
  var buf = zeros[float32](n, cols)
  var failed: seq[int]
  doAssert f.readTracesInto(0 ..< n, buf, failed) == c.failed
  doAssert failed == c.failedRows
  let fromRead = chunkMeta(f, 0 ..< n, failed = failed)
  for row in 0..<n:
    doAssert fromRead.isFailed(row) == fromChunk.isFailed(row)
    doAssert fromRead.isLive(row) == fromChunk.isLive(row)
  doAssert not fromRead.isLive(failed[0])
  # The list is reused: a second read into it does not allocate.
  let again = allocsIn:
    discard f.readTracesInto(0 ..< n, buf, failed)
  doAssert again == 0, "a reused failed-row list allocated " & $again & " times"
  # A list of traces, in list order: positions in the list.
  let last = n - 1
  let selection = @[last, 0, 1]
  var small = zeros[float32](3, cols)
  doAssert f.readTracesInto(selection, small, failed) == 1 and failed == @[0]
  let listMeta = chunkMeta(f, selection, failed = failed)
  doAssert listMeta.isFailed(0) and listMeta.isLive(1)
  doAssert listMeta.keys(0).offset == f.traceKeys(last).offset

block sample_windows:
  var f = openSeismic(demoDir() / "survey.sgy", keys = true)
  defer: f.close()
  let ns = int(f.traceEntry(0).numSamples)
  var p = newPrefetcher[float32](f, 0 ..< 4, chunkTraces = 4, cols = 64,
    firstSample = ns - 20)
  defer: p.close()
  var c: PrefetchChunk[float32]
  doAssert p.next(c)
  let meta = chunkMeta(c)
  doAssert meta.firstSample == ns - 20 and meta.liveSamples(0, 64) == 20
  doAssert meta.sampleTimeMs(0, 0) == meta.t0Ms(0) + float64(ns - 20) * 2
  # The same window through readTracesInto.
  var buf = zeros[float32](4, 64)
  var failed: seq[int]
  doAssert f.readTracesInto(0 ..< 4, (ns - 20) ..< (ns + 44), buf, failed) == 0
  let m2 = chunkMeta(f, 0 ..< 4, firstSample = ns - 20, failed = failed)
  doAssert m2.liveSamples(3, 64) == 20 and buf[0, 5] == c.samples[0, 5]

block sps_joined_segd_rev2:
  # record.sgd has stations (receiver 10/104, source 20/99.5) but no
  # coordinates: no offset until the survey is joined.
  let path = tmp / "record.sgd"
  copyFile(demoDir() / "record.sgd", path)
  var f = openSeismic(path, keys = true)
  defer: f.close()
  let before = chunkMeta(f, 0 ..< f.len)
  doAssert before.dtMs == 2 and before.keys(0).offset.isNaN
  var g: SurveyGeometry
  g.addReceiver(10, 104, 1, 1000, 2000, 50)
  g.addSource(20, 99.5, 1, 1300, 2400, 40)
  doAssert f.applySurvey(g) == 1
  let meta = chunkMeta(f, 0 ..< f.len) # views made after the join
  doAssert abs(meta.keys(0).offset - 500) < 1e-9, $meta.keys(0).offset
  doAssert meta.keys(0).rcvZ == 50
  var p = newPrefetcher[float32](f, 0 ..< f.len, chunkTraces = 4,
      uniform = true)
  defer: p.close()
  var c: PrefetchChunk[float32]
  doAssert p.next(c)
  doAssert chunkMeta(c).keys(0).offset == meta.keys(0).offset

proc twoIntervals(path: string) =
  ## shot.sgd with its channel sets 2 and 3 at 2 ms (the rest at 1 ms), as
  ## parseSEG's own test makes it. Trace order: CS1, CS3, CS2, CS4, CS4.
  var d = readSegd(demoDir() / "shot.sgd")
  for c in d.records[0].record.scanTypes[0].descriptorsRev3.mitems:
    if c.channelSetNumber in [2'u16, 3]:
      c.samplingInterval = 2000
      c.endTime = int32(2000 * (int(c.numSamples) - 1))
  let sink = openDiskSink(path)
  try: doAssert not writeSegd(sink, d).isFatal
  finally: sink.close()

block mixed_interval_segd:
  let path = tmp / "two.sgd"
  twoIntervals(path)
  var f = openSeismic(path, keys = true)
  defer: f.close()
  doAssertRaises ValueError:
    discard chunkMeta(f, 0 ..< f.len)
  var mixed = newPrefetcher[float32](f, 0 ..< f.len, chunkTraces = 8, cols = 4)
  var c: PrefetchChunk[float32]
  doAssert mixed.next(c) and c.count == 5
  try:
    discard chunkMeta(c)
    doAssert false, "a mixed chunk must raise"
  except ValueError as e:
    doAssert "uniform = true" in e.msg, e.msg
  mixed.close()
  var u = newPrefetcher[float32](f, 0 ..< f.len, chunkTraces = 8, cols = 4,
    uniform = true)
  defer: u.close()
  var dts: seq[float64]
  while u.next(c):
    let meta = chunkMeta(c)
    dts.add meta.dtMs
    for row in 0..<meta.len:
      doAssert meta.keys(row).intervalUs == meta.dtMs * 1000
  doAssert dts == @[1.0, 2.0, 2.0, 1.0], $dts
  # uniformRuns gives the same split for a reader.
  var runs: seq[float64]
  for run in f.uniformRuns(0 ..< f.len):
    runs.add chunkMeta(f, run).dtMs
  doAssert runs == dts
  # Auxiliary channel sets are not live.
  let first = chunkMeta(f, 0 .. 0)
  doAssert not first.isLive(0)

block errors:
  var f = openSeismic(demoDir() / "survey.sgy") # no stored keys
  defer: f.close()
  doAssertRaises ValueError:
    discard chunkMeta(f, 0 ..< 3)
  doAssertRaises ValueError:
    discard intervalMs(f.rowsOf(0 ..< 3))
  f.ensureKeys()
  doAssertRaises ValueError:
    discard chunkMeta(f, 0 ..< 0) # no rows to take the interval from
  doAssertRaises ValueError:
    discard chunkMeta(f, 0 ..< f.len + 1)
  var keys = @[initTraceKeys(), initTraceKeys()]
  keys[0].intervalUs = 2000
  let counts = @[10, 10]
  doAssertRaises ValueError:
    discard intervalMs(rowsOf(keys, counts)) # row 1's interval is NaN
  keys[1].intervalUs = 4000
  doAssertRaises ValueError:
    discard intervalMs(rowsOf(keys, counts))
  keys[1].intervalUs = 2000
  doAssert intervalMs(rowsOf(keys, counts)) == 2
