## Test runner: one compile of the suite, each test in its own process.
##
## Every `tests/t*.nim` exports `proc main*()` and ends with
## `when isMainModule: main()`, so `nim c -r tests/tfoo.nim` still runs that
## test alone, and importing the module runs nothing. The runner writes one
## program into the cache (not tracked) that imports every joined test and
## dispatches on its first argument: `alltests tfoo` calls `tfoo.main()`,
## `alltests --list` prints the names and returns before those imports run
## (a Windows pipe does not show a quit that happens while they load). It
## compiles that program once, then runs `alltests <name>` per test.
##
## A test that cannot join the aggregate carries one comment line
##
##   # nimtest: separate: <reason>
##
## and is compiled alone, as every test used to be. `--separate` compiles
## every test alone, for when the aggregate itself does not compile.
## `tests/reject/r*.nim` stay separate too: each must fail to compile, and
## holds a line `## Expect: <text>` that the compiler's output must contain.
##
## Arguments that do not start with `-` select tests (`tfoo` or `tfoo.nim`);
## the rest are compiler flags, passed through as before:
##
##   nim c -r tests/tester.nim
##   nim c -r tests/tester.nim -d:release
##   nim c -r tests/tester.nim -d:addressSanitizer
##   nim c -r tests/tester.nim tbasic
##   nim c -r tests/tester.nim --separate

import std/[algorithm, os, osproc, streams, strutils]
import utils/listlines

const separateMark = "# nimtest: separate:"

proc run(exe: string, args: openArray[string], workingDir = ""): int =
  ## Runs `exe` with `args` directly (no shell), with this process's stdout
  ## and stderr; returns the exit code. `workingDir` empty keeps the current
  ## directory (a relative `.exe` on Windows is otherwise searched from wherever
  ## the parent was started).
  var p = startProcess(exe, workingDir = workingDir, args = args,
    options = {poParentStreams})
  result = p.waitForExit()
  p.close()

proc capture(exe: string, args: openArray[string], workingDir = ""): tuple[
    output: string, code: int] =
  var p = startProcess(exe, workingDir = workingDir, args = args,
    options = {poStdErrToStdOut})
  result.output = p.outputStream.readAll()
  result.code = p.waitForExit()
  p.close()

proc expected(f: string): string =
  ## The text after `## Expect: ` in reject test `f`.
  for line in lines(f):
    if line.startsWith("## Expect: "):
      return line["## Expect: ".len .. ^1].strip
  ""

proc separateReason(f: string): string =
  ## The reason after `# nimtest: separate:`, or "" when `f` joins the aggregate.
  for line in lines(f):
    let s = line.strip
    if s.startsWith(separateMark) or s.startsWith("#" & separateMark):
      let rest = s[s.find(separateMark) + separateMark.len .. ^1].strip
      if rest.len == 0:
        quit "FAILURE " & f & " has an empty `# nimtest: separate:` reason"
      return rest
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

proc nimQuote(s: string): string =
  ## A Nim string literal of `s`, for the generated config.
  result = "\""
  for c in s:
    if c == '\\': result.add "\\\\"
    elif c == '"': result.add "\\\""
    else: result.add c
  result.add "\""

proc writeSuiteConfig(dir, root: string) =
  ## The switches `tests/config.nims` would have applied had the aggregate
  ## lived in `tests/`. The project's own config still loads (this directory
  ## is under it): outdir, CPU target.
  var s = "switch(\"path\", " & nimQuote(root / "src") & ")\n"
  s.add "switch(\"path\", " & nimQuote(root / "tests") & ")\n"
  s.add "switch(\"define\", \"nimAllocStats\")\n"
  s.add """
when defined(addressSanitizer):
  switch("debugger", "native")
  switch("define", "noSignalHandler")
  switch("define", "useMalloc")
  when defined(vcc):
    switch("passC", "/fsanitize=address")
  elif defined(windows):
    {.error: "nim asan needs gcc/clang on Linux or macOS, or MSVC (--cc:vcc)".}
  else:
    switch("passC", "-fsanitize=address -fno-omit-frame-pointer")
    switch("passL", "-fsanitize=address -fno-omit-frame-pointer")
"""
  writeFile(dir / "config.nims", s)

proc aggregateSource(names: seq[string]): string =
  ## One program: `alltests <name>` runs that test, `alltests --list` prints them.
  ##
  ## `--list` is a C `main` (the aggregate is compiled `--noMain`) that writes
  ## the names, flushes, and returns before `NimMain`. Import-time code and a
  ## `{.dynlib.}` load run inside `NimMain`; on Windows either can `rawQuit`
  ## without flushing a pipe, so the list would be empty and the exit code
  ## unexplained. `paramStr` is not used: on Windows it re-parses
  ## `GetCommandLineW` and can disagree with the argv `startProcess` passed.
  result = "## Generated by tests/tester.nim. Not tracked.\n"
  for n in names:
    result.add "import " & n & " as m_" & n & "\n"
  result.add "\n{.emit: \"\"\"\n#include <stdio.h>\n#include <string.h>\n"
  result.add "static const char *const suiteNames[] = {\n"
  for n in names:
    result.add "  " & nimQuote(n) & ",\n"
  result.add """  0
};
static char suiteArgBuf[512];
const char *suiteArg;
static int suiteIsList(const char *a) {
  char buf[64];
  size_t n, i;
  if (a == 0) return 0;
  while (*a == ' ' || *a == '\t' || *a == '"') a++;
  n = strlen(a);
  while (n > 0 && (a[n - 1] == '\r' || a[n - 1] == ' ' || a[n - 1] == '\t' ||
      a[n - 1] == '"')) n--;
  if (n >= sizeof(buf)) return 0;
  for (i = 0; i < n; i++) buf[i] = a[i];
  buf[n] = 0;
  return strcmp(buf, "--list") == 0;
}
static int suiteWantList(int argc, char **argv) {
  int i;
  if (argc <= 1) return 1;
  for (i = 1; i < argc; i++) if (suiteIsList(argv[i])) return 1;
  return 0;
}
static int suitePrintList(void) {
  int i;
  for (i = 0; suiteNames[i] != 0; i++) {
    if (fprintf(stdout, "%s\n", suiteNames[i]) < 0) return 1;
  }
  if (fflush(stdout) != 0) return 1;
  return 0;
}
void NimMain(void);
int main(int argc, char **argv) {
  size_t n;
  int i;
  const char *a;
  if (suiteWantList(argc, argv)) return suitePrintList();
  suiteArgBuf[0] = 0;
  if (argc >= 2 && argv[1] != 0) {
    a = argv[1];
    while (*a == '"') a++;
    n = strlen(a);
    while (n > 0 && (a[n - 1] == '\r' || a[n - 1] == '"')) n--;
    if (n >= sizeof(suiteArgBuf)) n = sizeof(suiteArgBuf) - 1;
    for (i = 0; i < (int)n; i++) suiteArgBuf[i] = a[i];
    suiteArgBuf[n] = 0;
  }
  suiteArg = suiteArgBuf;
  NimMain();
  return 0;
}
"""
  result.add "\"\"\".}\n\n"
  result.add "var suiteArg {.importc: \"suiteArg\", nodecl.}: cstring\n\n"
  result.add "proc run(name: string) =\n  case name\n"
  for n in names:
    result.add "  of " & nimQuote(n) & ", " & nimQuote(n & ".nim") & ", " &
      nimQuote(n & ".exe") & ":\n"
    result.add "    m_" & n & ".main()\n"
  result.add """  else:
    stdout.write("unknown test: " & name & "\n")
    stdout.flushFile()
    quit(1)

when isMainModule:
  run(if suiteArg.isNil: "" else: $suiteArg)
"""

proc moduleName(f: string): string =
  f.extractFilename.changeFileExt("")

proc selected(name, file: string, picked: seq[string]): bool =
  if picked.len == 0: return true
  for p in picked:
    if p == name or p == file.extractFilename or p == "reject" / file.extractFilename:
      return true
  false

proc main() =
  let
    testDir = currentSourcePath().parentDir
    root = testDir.parentDir
    nim = getCurrentCompilerExe()
  var
    flags: seq[string] = @[]
    picked: seq[string] = @[]
    separateAll = false
  for a in commandLineParams():
    if a == "--separate":
      separateAll = true
    elif a.len > 0 and a[0] != '-':
      picked.add a
    else:
      flags.add a
  var tests: seq[string] = @[]
  for f in walkFiles(testDir / "t*.nim"):
    if f.extractFilename != "tester.nim":
      tests.add f
  tests.sort()
  if tests.len == 0:
    quit "FAILURE no tests found in " & testDir
  for p in picked:
    var known = false
    for f in tests:
      if selected(moduleName(f), f, @[p]):
        known = true
    if not known:
      for f in walkFiles(testDir / "reject" / "r*.nim"):
        if p == f.extractFilename or p == "reject" / f.extractFilename:
          known = true
      if not known:
        quit "FAILURE unknown test: " & p
  var joined: seq[string] = @[]
  var alone = 0
  for f in tests:
    if selected(moduleName(f), f, picked):
      if separateAll or separateReason(f).len > 0:
        inc alone
      else:
        joined.add f
  # Absolute, so Windows CreateProcess does not resolve a relative `.exe`
  # against a different working directory than the one the tests were written for.
  let
    work = absolutePath(if root.len == 0: getCurrentDir() else: root)
    exe = work / "bin" / "alltests".addFileExt(ExeExt)
  if joined.len > 0:
    createDir(work / ".nimcache")
    createDir(work / "bin")
    let genDir = work / ".nimcache" / "suite"
    createDir(genDir)
    var names: seq[string] = @[]
    for f in joined:
      names.add moduleName(f)
    writeFile(genDir / "alltests.nim", aggregateSource(names))
    writeSuiteConfig(genDir, work)
    echo "Compiling: alltests ", flags.join(" ")
    # Forward slashes: Nim's `-o:` value is taken whole, but a Windows C
    # compile line treats `\` as an escape.
    let outOpt = "-o:" & exe.replace('\\', '/')
    if run(nim, @["c", "--hints:off", "--noMain", outOpt] & flags & @[
        genDir / "alltests.nim"]) != 0:
      quit "FAILURE the aggregate did not compile. Re-run with --separate."
    let listed = capture(exe, @["--list"], work)
    if listed.code != 0 or listedNames(listed.output) != names:
      echo "FAILURE alltests --list does not match the joined tests: ",
        listFailure(listed.code, listed.output, names)
      quit(1)
  var failed: seq[string] = @[]
  for f in tests:
    let name = moduleName(f)
    if not selected(name, f, picked):
      discard
    elif separateAll or separateReason(f).len > 0:
      echo "Running: ", f.extractFilename, " ", flags.join(" ")
      if run(nim, @["c", "-r", "--hints:off"] & flags & @[f]) != 0:
        failed.add f.extractFilename
    else:
      echo "Running: ", f.extractFilename, " ", flags.join(" ")
      if run(exe, @[name], work) != 0:
        failed.add f.extractFilename
  var rejects: seq[string] = @[]
  for f in walkFiles(testDir / "reject" / "r*.nim"):
    if selected("", f, picked):
      rejects.add f
  rejects.sort()
  let cache = getTempDir() / ("processseg_reject_" & $getCurrentProcessId())
  for f in rejects:
    echo "Rejecting: ", f.extractFilename, " ", flags.join(" ")
    if not rejected(nim, f, flags, testDir, cache):
      failed.add "reject/" & f.extractFilename
  if rejects.len > 0:
    removeDir(cache)
  let total = joined.len + alone + rejects.len
  if failed.len > 0:
    quit "FAILURE " & $failed.len & " of " & $total & " test files: " &
      failed.join(", ")
  echo "All ", total, " test files passed."

main()
