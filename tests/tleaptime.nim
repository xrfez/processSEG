## The leap-second table as a library input, and GPS time, through
## `import processSEG` alone.
##
## Broken by not installing a clean document in `useLeapTableNif`: a good
## text left the shipped table in place.

import std/assertions
import processSEG

proc main*() =
  defer: useLeapTable(defaultLeapTable())

  block reexports_and_default_table:
    let shipped = defaultLeapTable()
    doAssert currentLeapTable() == shipped
    doAssert shipped.entries.len > 0
    doAssert leapKnownAt(1_527_811_200'i64 * 1_000_000)
    discard LeapSecond(year: 1980, month: 1, day: 1, gpsMinusUtc: 0)
    discard LeapDate(year: 2027, month: 1, day: 1)

  block bad_nif_leaves_the_table:
    let before = currentLeapTable()
    let problems = useLeapTableNif("this is not a leap table")
    doAssert problems.len > 0
    doAssert currentLeapTable() == before
    let bad = """(leapSeconds
  (version 1)
  (checkedTo 2030 1 1)
  (leap (date 1990 1 1) (gpsMinusUtc 5))
)"""
    let again = useLeapTableNif(bad)
    doAssert again.len > 0
    doAssert currentLeapTable() == before

  block good_nif_installs:
    let before = utcUsOfGps(1_200_000_000_000_000'i64)
    const text = """(leapSeconds
  (version 1)
  (checkedTo 2030 1 1)
  (leap (date 1980 1 1) (gpsMinusUtc 0))
  (leap (date 2017 1 1) (gpsMinusUtc 1))
)"""
    let parsed = parseLeapTable(text)
    doAssert parsed.problems.len == 0
    let problems = useLeapTableNif(text)
    doAssert problems.len == 0
    doAssert currentLeapTable().entries[^1].gpsMinusUtc == 1
    # Offset 1 instead of the shipped 18: the same GPS instant is 17 s later.
    doAssert utcUsOfGps(1_200_000_000_000_000'i64) - before == 17_000_000
    useLeapTable(defaultLeapTable())

  block shot_follows_the_table:
    const
      GpsEpoch = 315964800'i64
      Off = 18'i64
      BeforeSec = 1527811200'i64 # 2018-06-01
      AfterSec = 1811808000'i64  # 2027-06-01
    let beforeGps = (BeforeSec - GpsEpoch + Off) * 1_000_000
    let afterGps = (AfterSec - GpsEpoch + Off) * 1_000_000
    useLeapTable(defaultLeapTable())
    let before0 = initSliceShotGps(beforeGps, 1).timeUs
    let after0 = initSliceShotGps(afterGps, 2).timeUs
    doAssert before0 == BeforeSec * 1_000_000
    doAssert after0 == AfterSec * 1_000_000
    doAssert initSliceShotGps(beforeGps, 1).timeUsKnown
    doAssert gpsUsToUtcSeconds(beforeGps) == float64(before0) / 1e6
    let extended = withLeap(defaultLeapTable(), LeapDate(year: 2027, month: 1, day: 1))
    useLeapTable(extended)
    doAssert initSliceShotGps(beforeGps, 1).timeUs == before0
    doAssert initSliceShotGps(afterGps, 2).timeUs == after0 - 1_000_000
    doAssert utcUsOfGps(afterGps) == after0 - 1_000_000

when isMainModule: main()
