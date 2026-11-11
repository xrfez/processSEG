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
##
## **Reject tests**, `tests/reject/r*.nim`, must *fail* to compile (to C, with
## the same flags): each holds a line `## Expect: <text>`, and the compiler's
## output must contain that text. They check contracts that `compiles()` cannot
## see, such as a `=copy` that is an error.

import std/[algorithm, os, osproc, streams, strutils]

proc run(exe: string, args: openArray[string]): int =
  ## Runs `exe` with `args` directly (no shell, so no quoting rules), with
  ## this process's stdout and stderr; returns the exit code.
  var p = startProcess(exe, args = args, options = {poParentStreams})
  result = p.waitForExit()
  p.close()

proc expected(f: string): string =
  ## The text after `## Expect: ` in reject test `f`.
  for line in lines(f):
    if line.startsWith("## Expect: "):
      return line["## Expect: ".len .. ^1].strip
  ""

proc rejected(nim, f: string, flags: openArray[string], testDir,
    cache: string): bool =
  ## Reject test `f` fails to compile with its expected message.
  let want = expected(f)
  if want.len == 0:
    echo "  no `## Expect:` line in ", f
    return false
  let args = @["c", "--compileOnly", "--hints:off", "--path:" & testDir / ".." / "src",
    "--nimcache:" & cache] & @flags & @[f]
  var p = startProcess(nim, args = args, options = {poStdErrToStdOut})
  let output = p.outputStream.readAll()
  let code = p.waitForExit()
  p.close()
  if code == 0:
    echo "  compiled, but must not"
  elif want notin output:
    echo "  failed without `", want, "`:\n", output
  result = code != 0 and want in output

proc main() =
  let
    testDir = currentSourcePath().parentDir
    flags = commandLineParams()
    nim = getCurrentCompilerExe()
  var tests: seq[string]
  for f in walkFiles(testDir / "t*.nim"):
    if f.extractFilename != "tester.nim":
      tests.add f
  tests.sort()
  if tests.len == 0:
    quit "FAILURE no tests found in " & testDir
  var failed: seq[string]
  for f in tests:
    echo "Running: ", f.extractFilename, " ", flags.join(" ")
    if run(nim, @["c", "-r", "--hints:off"] & flags & @[f]) != 0:
      failed.add f.extractFilename
  var rejects: seq[string]
  for f in walkFiles(testDir / "reject" / "r*.nim"):
    rejects.add f
  rejects.sort()
  let cache = getTempDir() / ("processseg_reject_" & $getCurrentProcessId())
  for f in rejects:
    echo "Rejecting: ", f.extractFilename, " ", flags.join(" ")
    if not rejected(nim, f, flags, testDir, cache):
      failed.add "reject/" & f.extractFilename
  removeDir(cache)
  let total = tests.len + rejects.len
  if failed.len > 0:
    quit "FAILURE " & $failed.len & " of " & $total & " test files: " &
      failed.join(", ")
  echo "All ", total, " test files passed."

main()
