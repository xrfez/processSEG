## GPS time and the leap-second table.
##
## The table is one process-wide table shared with parseSEG: `useLeapTable`
## here is the same proc, and `gpsUsToUtcUs` reads the same pointer. This
## module does not read a file; the caller passes the `.nif` text to
## `useLeapTableNif`. Copies installed by `useLeapTable` are not reclaimed:
## do not swap in a loop.

from parseSEG import gpsUsToUtcUs, useLeapTable, currentLeapTable,
  defaultLeapTable, parseLeapTable, withLeap, leapKnownAt, LeapTable,
  LeapSecond, LeapDate

export useLeapTable, currentLeapTable, defaultLeapTable, parseLeapTable
export withLeap, leapKnownAt, LeapTable, LeapSecond, LeapDate

proc useLeapTableNif*(nifText: string): seq[string] =
  ## Parses a leap-second document and, when it has no problems, installs it
  ## (`useLeapTable`). Returns the problems; empty means that table is now in
  ## use. Does not read a file: the caller does. A document with problems
  ## leaves the table unchanged.
  let parsed = parseLeapTable(nifText)
  if parsed.problems.len == 0:
    useLeapTable(parsed.table)
  result = parsed.problems

proc utcUsOfGps*(gpsUs: int64): int64 {.gcsafe, raises: [].} =
  ## UTC microseconds since 1970 of `gpsUs` (microseconds since the GPS
  ## epoch, 1980-01-06). Integer arithmetic through `gpsUsToUtcUs` and the
  ## current leap table, so the microsecond is exact. An input within one day
  ## of either `int64` end saturates, as `gpsUsToUtcUs` does.
  gpsUsToUtcUs(gpsUs)

proc gpsUsToUtcSeconds*(gpsUs: int64): float64 {.gcsafe, raises: [].} =
  ## UTC POSIX seconds of `gpsUs`. A float64 second near today is about a
  ## quarter microsecond coarse (the unit in the last place of 1.7e9 is
  ## 2⁻²² seconds); use `utcUsOfGps` when the microsecond matters.
  float64(utcUsOfGps(gpsUs)) / 1e6
