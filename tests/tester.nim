## Test runner: compiles and runs every `tests/t*.nim` file in its own process.
##
## The runner's command-line arguments are passed to the compiler for every test,
## so one runner build serves each mode:
##
##   nim c -r tests/tester.nim                    # debug
##   nim c -r tests/tester.nim -d:release         # release (also -d:danger)
##   nim c -r tests/tester.nim -d:addressSanitizer
##
## Helper modules must not start with `t` (they live in `tests/utils/`).

import std/[algorithm, os, osproc, strutils]

proc main() =
  let
    testDir = currentSourcePath().parentDir
    flags = commandLineParams().quoteShellCommand
    nim = getCurrentCompilerExe().quoteShell
  var tests: seq[string]
  for f in walkFiles(testDir / "t*.nim"):
    if f.extractFilename != "tester.nim":
      tests.add f
  tests.sort()
  if tests.len == 0:
    quit "FAILURE no tests found in " & testDir
  var failed: seq[string]
  for f in tests:
    let cmd = nim & " c -r --hints:off " & flags & " " & f.quoteShell
    echo "Running: ", f.extractFilename, " ", flags
    if execShellCmd(cmd) != 0:
      failed.add f.extractFilename
  if failed.len > 0:
    quit "FAILURE " & $failed.len & " of " & $tests.len & " test files: " &
      failed.join(", ")
  echo "All ", tests.len, " test files passed."

main()
