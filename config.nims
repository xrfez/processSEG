# Strict build configuration and task orchestrator for processseg
import std/[os, strutils]

switch("outdir", "bin")

# parseSEG is compiled into this program, so these flags decide its decode
# speed (parseSEG README, "Build flags"). x86-64-v3 (AVX2: Intel 2013+ / AMD
# Zen 2017+) runs its sample decoders up to 2x faster than baseline x86-64; a
# v3 binary crashes with an illegal instruction on CPUs without AVX2, so the
# portable tasks pass -d:baselineCpu. Other CPUs (arm64) keep the default.
# zig cc rejects gcc's CPU names, so the zig tasks pass -d:zigcc for its own.
# -d:release is passed per task: a global define would make `nim test` and
# `nim nimDebug` release builds.
when defined(amd64) and not defined(baselineCpu):
  when defined(zigcc):
    switch("passC", "-mcpu=x86_64_v3")
  else:
    switch("passC", "-march=x86-64-v3")

const baseTarget = when defined(amd64): " --passC:-march=x86-64 " else: " "

let zigccExe =
  if findExe("zigcc").len > 0:
    "zigcc"
  elif defined(windows):
    "tools/zigcc.cmd"
  else:
    "./tools/zigcc"

task nimRelease, "Build native host-optimized binary (aggressive speed, march=native)":
  ## Compiles high-performance binary optimized strictly for the current host CPU.
  selfExec "c -d:release -d:danger --opt:speed --passC:\"-march=native -mtune=native\" src/processseg.nim"

task nimPortable, "Build portable binary for generic CPUs (stripped, baseline instructions)":
  ## Compiles stripped, portable binary targeting generic CPU baseline compatibility.
  selfExec "c -d:release -d:baselineCpu --opt:size --passC:\"-mtune=generic\" --passL:\"-s\"" & baseTarget & "src/processseg.nim"

task nimDebug, "Build debug binary with stack traces and assertions":
  ## Compiles debug binary with full assertions, line tracing, and symbols.
  selfExec "c --debuginfo:on --assertions:on --lineTrace:on --stackTrace:on src/processseg.nim"

task zigRelease, "Build native host-optimized binary using Zig CC (O3, march=native)":
  ## Compiles native binary using Zig toolchain with -O3 optimization for host CPU.
  selfExec "c -d:zigcc --cc:clang --clang.exe:\"" & zigccExe & "\" --clang.linkerexe:\"" & zigccExe & "\" -d:release -d:danger --opt:speed --passC:\"-O3 -march=native\" src/processseg.nim"

task zigPortable, "Build portable binary for generic CPUs using Zig CC (stripped, baseline instructions)":
  ## Compiles stripped binary using Zig toolchain targeting generic baseline CPUs and size reduction.
  selfExec "c -d:zigcc --cc:clang --clang.exe:\"" & zigccExe & "\" --clang.linkerexe:\"" & zigccExe & "\" -d:release -d:baselineCpu --opt:size --passC:\"-Oz\" --passC:\"-mcpu=baseline\" --passL:\"-s\" src/processseg.nim"

task zigDebug, "Build debug binary using Zig CC with debug symbols":
  ## Compiles debug binary using Zig toolchain preserving debug symbols.
  selfExec "c -d:zigcc --cc:clang --clang.exe:\"" & zigccExe & "\" --clang.linkerexe:\"" & zigccExe & "\" --debuginfo:on src/processseg.nim"

task test, "Run test suite":
  ## Executes unittest suite.
  selfExec "c -r --verbosity:0 tests/test_all.nim"

task testRelease, "Run the test suite in -d:release and -d:danger builds":
  ## Executes unittest suite with the optimization levels the release builds use.
  selfExec "c -r --verbosity:0 -d:release tests/test_all.nim"
  selfExec "c -r --verbosity:0 -d:danger tests/test_all.nim"

task fmt, "Format codebase via nimpretty":
  ## Formats source and test files in-place.
  exec "nimpretty --indent:2 src/processseg.nim"

task docs, "Generate HTML documentation":
  ## Emits HTML documentation into htmldocs directory.
  selfExec "doc --project --outdir:htmldocs src/processseg.nim"

# ------------------------------------------------- parseSEG patch branch --
# processSEG patches parseSEG in place, in the Atlas checkout deps/parseSEG,
# on the branch `processseg-integration` (processSEG.nimble requires that
# branch). The branch is later opened as a pull request against parseSEG.
const
  parseSEGDir = "deps/parseSEG"
  parseSEGBranch = "processseg-integration"

task parseSEGTest, "Run parseSEG's own test suite in deps/parseSEG (the patch branch)":
  ## parseSEG's nim.cfg expects its dependencies in its own deps/: link
  ## processSEG's Atlas checkouts there (deps/* is gitignored in parseSEG),
  ## then run its `test` task without this project's config (--skipParentCfg).
  for dir in listDirs("deps"):
    let name = dir.lastPathPart
    if name != "parseSEG" and not name.startsWith("."):
      exec "ln -sfn ../../" & name & " " & parseSEGDir & "/deps/" & name
  withDir parseSEGDir:
    exec quoteShell(getCurrentCompilerExe()) & " --skipParentCfg test"

task parseSEGBranch, "Put deps/parseSEG back on the patch branch after an atlas command":
  ## `atlas install` checks out the branch head detached; new commits must go
  ## on the branch, so switch to it (refuses if HEAD has moved off its head).
  withDir parseSEGDir:
    exec "test \"$(git rev-parse HEAD)\" = \"$(git rev-parse " & parseSEGBranch &
      ")\" || { echo 'deps/parseSEG HEAD is not the head of " & parseSEGBranch &
      "; check before switching'; exit 1; }"
    exec "git switch " & parseSEGBranch
