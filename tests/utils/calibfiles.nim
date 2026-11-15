## Small synthetic files whose headers carry scale information, written with
## parseSEG, for the calibration tests. Each writer
## documents what its headers say; the samples are a constant per trace
## (`rowValue`), so a calibrated value is checked against the factor alone.

import std/[math, options, os]
import parseSEG

func rowValue*(trace: int): float64 =
  ## The value of every sample of trace `trace` in these files.
  float64(1000 * (trace + 1))

proc rev2Path*(): string = getTempDir() / "processseg_cal_rev2.sgd"
proc rev3Path*(): string = getTempDir() / "processseg_cal_rev3.sgd"
proc segyPath*(): string = getTempDir() / "processseg_cal.sgy"
proc seg2Path*(): string = getTempDir() / "processseg_cal.sg2"

const ns = 100

proc writeRev2*(path = rev2Path()) =
  ## SEG-D Rev 2.0, format 8058, record 7, 100 samples at 2 ms:
  ## - channel set 0: auxiliary (channel type 9), MP 0 (1 mV/count), 1 trace;
  ## - channel set 1: seismic, MP −10 (2⁻¹⁰ mV/count), 3 traces, sensor type 2
  ##   (vertical geophone), channels 1-3, receivers line 10, points 101-103;
  ## - channel set 2: seismic, MP 3.5 (2^3.5 mV/count), 2 traces, sensor type 1
  ##   (hydrophone), channels 4-5, points 104-105.
  const fileNo = 7
  let fmt = sd8058
  var r = SegdRecord(layout: slRev2, format: some(fmt),
    revision: SegdRevision(major: 2, minor: 0))
  r.gh1 = SegdGeneralHeader1(fileNumber: fileNo, formatCode: ord(fmt), year: 26,
    additionalGhBlocks: 2, julianDay: 319, hour: 8, minute: 0, second: 0,
    manufacturerCode: 0, baseScanInterval16: 32, recordType: 8, recordLength: BcdAllF,
    scanTypesPerRecord: 1, channelSetsPerScanType: 3, skewBlocks: 0,
    extendedHeaderBlocks: 0, externalHeaderBlocks: 0)
  r.gh2 = some SegdGeneralHeader2(extFileNumber: fileNo, revisionMajor: 2,
    extRecordLength: uint32(2 * ns), blockNumber: 2)
  r.sourceHeaders = @[SegdGeneralHeaderN(sourceLineNumber: 20, sourcePointNumber: 1,
    sourcePointIndex: 1, blockNumber: 3, sourceSetNumber: 1)]
  var descs: seq[SegdChannelSetDescriptor]
  for (n, ctype, count, mp) in [(1, 9'u8, 1, 0.0), (2, 1'u8, 3, -10.0), (3, 1'u8, 2, 3.5)]:
    var cs = SegdChannelSetDescriptor(scanTypeNumber: 1, channelSetNumber: n, startTime: 0,
      endTime: uint16(2 * ns), channelCount: count, channelType: ctype, gainControl: 3,
      traceHeaderExtensions: 1, verticalStack: 1, arrayForming: 1)
    cs.mpFactor = mp
    descs.add cs
  r.scanTypes = @[SegdScanType(layout: slRev2, descriptors: descs)]
  var traces: seq[SegdTrace]
  var k = 0
  for (cs, count, sensor) in [(1, 1, 0'u8), (2, 3, 2'u8), (3, 2, 1'u8)]:
    for tn in 1 .. count:
      var t = SegdTrace(index: k, numSamples: ns)
      let channel = (if cs == 1: tn else: k) # the seismic channels 1-5
      t.header = SegdTraceHeader(fileNumber: fileNo, scanTypeNumber: 1, channelSetNumber: cs,
        traceNumber: channel, traceHeaderExtensions: 1)
      t.ext1 = some SegdTraceHeaderExt1(receiverLineNumber: (if cs == 1: 0 else: 10),
        receiverPointNumber: (if cs == 1: 0 else: int32(100 + k)), receiverPointIndex: 1,
        samplesPerTrace: ns, sensorType: sensor)
      var v = newSeq[float64](ns)
      for x in v.mitems: x = rowValue(k)
      t.samples = newSeq[byte](dataBytes(fmt, ns))
      encodeSamples(fmt, v, t.samples)
      traces.add t
      inc k
  r.traceCount = traces.len
  let sink = openDiskSink(path)
  try:
    let log = writeSegd(sink, SegdData(records: @[SegdRecordData(record: r, traces: traces)]))
    doAssert log.len == 0, $log[0]
  finally:
    sink.close()

proc writeRev3*(path = rev3Path(), geophones = 2, lastUnit = 0x12'u8) =
  ## SEG-D Rev 3.0, format 8058, record 9, 100 samples at 1 ms:
  ## - channel set 0: auxiliary (0x20), descale multiplier 0 (unknown), 1 trace;
  ## - channel set 1: seismic, DSM 1.5e-4 mV/count, `geophones` traces, sensor type 2,
  ##   each with a Sensor Info block: STY 20 mV per mm/s (THE#1 physical unit
  ##   0x03), i.e. 20 V/(m/s); receivers line 10, points 101-102;
  ## - channel set 2: seismic, DSM 2e-4 mV/count, physical unit Pa (0x12, or
  ##   `lastUnit`) in the channel set, 2 traces, sensor type 1: the first with
  ##   STY 0.2 mV/Pa (2e-4 V/Pa), the second with STY 0 (unknown); points 103-104.
  const fileNo = 9
  let fmt = sd8058
  var r = SegdRecord(layout: slRev3, format: some(fmt),
    revision: SegdRevision(major: 3, minor: 0))
  r.gh1 = SegdGeneralHeader1(fileNumber: fileNo, formatCode: ord(fmt), year: 26,
    additionalGhBlocks: 2, julianDay: 319, hour: 9, minute: 0, second: 0,
    manufacturerCode: 0, baseScanInterval16: 16, recordType: 8, recordLength: BcdAllF,
    scanTypesPerRecord: 1, channelSetsPerScanType: 3, skewBlocks: 0,
    extendedHeaderBlocks: 0, externalHeaderBlocks: 0)
  r.gh2Rev3 = SegdGeneralHeader2Rev3(extFileNumber: fileNo, revisionMajor: 3,
    extRecordLength: uint32(1000 * (ns - 1)), blockType: 2)
  r.gh3 = SegdGeneralHeader3(timeZero: 1_478_000_000_000_000, blockType: 3)
  var descs: seq[SegdChannelSetDescriptorRev3]
  for (n, ctype, count, exts, dsm, unit) in [(1, 0x20'u8, 1, 1'u8, 0.0'f32, 0'u8),
      (2, 0x10'u8, geophones, 2'u8, 1.5e-4'f32, 0x03'u8),
      (3, 0x10'u8, 2, 2'u8, 2e-4'f32, lastUnit)]:
    descs.add SegdChannelSetDescriptorRev3(scanTypeNumber: 1, channelSetNumber: uint16(n),
      channelType: ctype, endTime: int32(1000 * (ns - 1)), numSamples: ns,
      descaleMultiplier: dsm, channelsInSet: uint32(count), samplingInterval: 1000,
      traceHeaderExtensions: exts, verticalStack: 1, physicalUnit: unit, blockType1: 0x30,
      blockType2: 0x31, blockType3: 0x32)
  r.scanTypes = @[SegdScanType(layout: slRev3, descriptorsRev3: descs)]
  var traces: seq[SegdTrace]
  var k = 0
  for (cs, count, sensor, unit) in [(1, 1, 0'u8, 0'u8), (2, geophones, 2'u8, 0x03'u8),
      (3, 2, 1'u8, 0'u8)]:
    for tn in 1 .. count:
      let seismic = cs != 1
      var t = SegdTrace(layout: slRev3, index: k, numSamples: ns)
      t.header = SegdTraceHeader(fileNumber: fileNo, scanTypeNumber: 1, channelSetNumber: cs,
        traceNumber: tn, traceHeaderExtensions: (if seismic: 2'u8 else: 1'u8))
      t.ext1Rev3 = some SegdTraceHeaderExt1Rev3(receiverLineNumber: (if seismic: 10 else: 0),
        receiverPointNumber: (if seismic: int32(100 + k) else: 0), receiverPointIndex: 1,
        numSamples: ns, sensorType: sensor, physicalUnit: unit, blockType: 0x40)
      if seismic:
        let sty = (if cs == 2: 20'f32 elif tn == 1: 0.2'f32 else: 0'f32)
        t.extensions = @(SegdSensorInfo(sensorSensitivity: sty, blockType: 0x41).encoded)
      var v = newSeq[float64](ns)
      for x in v.mitems: x = rowValue(k)
      t.samples = newSeq[byte](dataBytes(fmt, ns))
      encodeSamples(fmt, v, t.samples)
      traces.add t
      inc k
  computeRev3Sizes(r, traces)
  r.traceCount = traces.len
  let sink = openDiskSink(path)
  try:
    let log = writeSegd(sink, SegdData(records: @[SegdRecordData(record: r, traces: traces)]))
    doAssert log.len == 0, $log[0]
  finally:
    sink.close()

type SegyCase* = object
  ## The scale fields of one trace of `writeSegy`.
  n*, valueUnit*, mantissa*, exponent*, units*: int

const segyCases* = [
  SegyCase(),                    # 0: nothing set: unknown
  SegyCase(n: 20),               # 1: N only: 2⁻²⁰ V (Rev 1's definition)
  SegyCase(n: 10, valueUnit: 3), # 2: 2⁻¹⁰ mV
  SegyCase(n: 10, valueUnit: 2, mantissa: 5, exponent: 1, units: 6),
    # 3: 2⁻¹⁰ V, transduction 50 (m/s per V): 0.02 V/(m/s)
  SegyCase(n: 3, valueUnit: 1),  # 4: already physical: 2⁻³ Pa
  SegyCase(valueUnit: 2, mantissa: 2, exponent: -3, units: 1),
    # 5: 1 V, transduction 0.002 Pa per V: 500 V/Pa
  SegyCase(mantissa: 1024),
    # 6: a transduction constant without units, no value unit, N = 0: unknown
  SegyCase(mantissa: 5, exponent: -1, units: 6)]
    # 7: no value unit, N = 0, transduction 0.5 m/s per count: already physical

proc writeSegy*(path = segyPath()) =
  ## SEG-Y Rev 2.1, IEEE float, one trace per `segyCases` entry (100 samples
  ## at 2 ms), its scale fields set by header overrides.
  var w = openSeismicWriter(path, SegyOutputOptions())
  for i, c in segyCases:
    var k = initTraceKeys()
    k.intervalUs = 2000
    k.t0Ms = 0
    k.channel = float64(i + 1)
    var v = newSeq[float32](ns)
    for x in v.mitems: x = float32(rowValue(i))
    w.writeTrace(k, v, [headerOverride("trace.traceWeightingFactor", c.n),
      headerOverride("trace.traceValueUnit", c.valueUnit),
      headerOverride("trace.transductionMantissa", c.mantissa),
      headerOverride("trace.transductionExponent", c.exponent),
      headerOverride("trace.transductionUnits", c.units)])
  let log = w.close()
  doAssert log.len == 0, $log[0]

proc writeSeg2*(path = seg2Path()) =
  ## SEG-2, float32, three traces of 100 samples at 1 ms: DESCALING_FACTOR
  ## 0.5 mV with STACK 2 (0.25 mV/count), none (unknown), and 0.001 mV with
  ## no STACK (1e-6 V/count).
  var fdb = initSeg2File(boLittle)
  fdb.traceCount = 3
  var w = openSeg2Writer(path, fdb)
  var x = newSeq[float32](ns)
  for i in 0 ..< 3:
    var t = Seg2Trace(formatCode: 4)
    t["CHANNEL_NUMBER"] = $(i + 1)
    if i == 0: t["DESCALING_FACTOR"] = "0.5"
    elif i == 2: t["DESCALING_FACTOR"] = "0.001"
    t["SAMPLE_INTERVAL"] = "0.001"
    if i == 0: t["STACK"] = "2"
    for k in 0 ..< ns: x[k] = float32(rowValue(i))
    w.writeTrace(t, x)
  w.close()
  doAssert w.log.len == 0, $w.log[0]

proc marinePath*(): string = getTempDir() / "processseg_cal_marine.sgd"

const
  marineDsm* = [1.49012e-4, 2.98023e-4, 2.98023e-4, 2.98023e-4]
    ## `writeMarineNodes`' descale multipliers by component (mV per count).

proc writeMarineNodes*(path = marinePath(), units = 3) =
  ## A record of `units` marine nodes as a recorder that writes counts gives
  ## it: SEG-D Rev 3.0, 24-bit integers (8036), 100 samples at 1 ms, one
  ## channel set per component, each node one hydrophone (component 1,
  ## sensor type 1) and three geophones (components 2-4: vertical, inline,
  ## crossline; sensor types 2-4), the receivers on line 10 at points 101 ...
  ## Channel set `c` holds component `c + 1` of every node, with the
  ## descale multiplier `marineDsm[c]` and no Sensor Info: the sensitivities
  ## are applied later, from a table. Trace `t` is channel set `t div units`,
  ## node `t mod units`, and holds `rowValue(t)` counts.
  const fileNo = 11
  let fmt = sd8036
  var r = SegdRecord(layout: slRev3, format: some(fmt),
    revision: SegdRevision(major: 3, minor: 0))
  r.gh1 = SegdGeneralHeader1(fileNumber: fileNo, formatCode: ord(fmt), year: 26,
    additionalGhBlocks: 2, julianDay: 319, hour: 10, minute: 0, second: 0,
    manufacturerCode: 0, baseScanInterval16: 16, recordType: 8, recordLength: BcdAllF,
    scanTypesPerRecord: 1, channelSetsPerScanType: 4, skewBlocks: 0,
    extendedHeaderBlocks: 0, externalHeaderBlocks: 0)
  r.gh2Rev3 = SegdGeneralHeader2Rev3(extFileNumber: fileNo, revisionMajor: 3,
    extRecordLength: uint32(1000 * (ns - 1)), blockType: 2)
  r.gh3 = SegdGeneralHeader3(timeZero: 1_478_000_000_000_000, blockType: 3)
  var descs: seq[SegdChannelSetDescriptorRev3]
  for c in 0 ..< 4:
    descs.add SegdChannelSetDescriptorRev3(scanTypeNumber: 1, channelSetNumber: uint16(c + 1),
      channelType: 0x10, endTime: int32(1000 * (ns - 1)), numSamples: ns,
      descaleMultiplier: float32(marineDsm[c]), channelsInSet: uint32(units),
      samplingInterval: 1000, traceHeaderExtensions: 1, verticalStack: 1, blockType1: 0x30,
      blockType2: 0x31, blockType3: 0x32)
  r.scanTypes = @[SegdScanType(layout: slRev3, descriptorsRev3: descs)]
  var traces: seq[SegdTrace]
  for c in 0 ..< 4:
    for u in 0 ..< units:
      let k = c * units + u
      var t = SegdTrace(layout: slRev3, index: k, numSamples: ns)
      t.header = SegdTraceHeader(fileNumber: fileNo, scanTypeNumber: 1, channelSetNumber: c + 1,
        traceNumber: u + 1, traceHeaderExtensions: 1)
      t.ext1Rev3 = some SegdTraceHeaderExt1Rev3(receiverLineNumber: 10,
        receiverPointNumber: int32(101 + u), receiverPointIndex: 1, groupIndex: uint8(c + 1),
        numSamples: ns, sensorType: uint8(c + 1), blockType: 0x40)
      var v = newSeq[float64](ns)
      for x in v.mitems: x = rowValue(k)
      t.samples = newSeq[byte](dataBytes(fmt, ns))
      encodeSamples(fmt, v, t.samples)
      traces.add t
  computeRev3Sizes(r, traces)
  r.traceCount = traces.len
  let sink = openDiskSink(path)
  try:
    let log = writeSegd(sink, SegdData(records: @[SegdRecordData(record: r, traces: traces)]))
    doAssert log.len == 0, $log[0]
  finally:
    sink.close()
