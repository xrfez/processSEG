## Allocation counting for "no allocation in a steady state" checks. The tests
## are built with `-d:nimAllocStats` (tests/config.nims); the count covers
## every thread.

import std/[parseutils, strutils]

proc allocCount*(s: AllocStats): int =
  ## The allocations counted in `s`. The stdlib keeps the field private, so it
  ## is read from `$s` ("(allocCount: N, deallocCount: M)").
  let text = $s
  let at = text.find("allocCount: ")
  doAssert at >= 0, "unexpected AllocStats format: " & text
  doAssert parseInt(text, result, at + "allocCount: ".len) > 0

template allocsIn*(body: untyped): int =
  ## The number of heap allocations `body` makes.
  let before = getAllocStats()
  body
  allocCount(getAllocStats() - before)
