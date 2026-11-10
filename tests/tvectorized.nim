## The hot kernel loops compile to packed SIMD: builds `utils/veckernels.nim`
## in `-d:release` (bound and overflow checks on outside the kernels, the CPU
## target from config.nims) and checks every instance of each kernel in the
## disassembly. Runs on x86-64 Linux with gcc and binutils, in the debug suite
## only (the sub-build has its own flags); skipped elsewhere.

import std/[os, osproc, strutils]

const
  # Kernel instance symbols (`<name>_u<n>__veckernels`) that must hold packed
  # instructions. A zero fill may be a memset call instead.
  kernels = ["sum64", "sumIndexed64", "demean", "detrend", "scale", "clip",
    "reversePolarity", "taperPads", "muteTop"]
  memsetOk = ["muteTop"]

proc isPacked(insn: string): bool =
  ## Packed arithmetic, compares, logic or conversions on vector registers;
  ## not moves, shuffles or broadcasts.
  if "%xmm" notin insn and "%ymm" notin insn and "%zmm" notin insn:
    return false
  var m = insn.splitWhitespace()[0]
  if m.startsWith("v"): m = m[1 .. ^1]
  if not (m.startsWith("p") or m.endsWith("ps") or m.endsWith("pd")):
    return false
  for skip in ["mov", "shuf", "unpck", "broadcast", "perm", "insr", "extr",
      "blend"]:
    if skip in m: return false
  # xor of a register with itself only zeroes it.
  if "xor" in m:
    let ops = insn.splitWhitespace()[^1].split(',')
    return ops.len == 3 and not (ops[0] == ops[1] and ops[1] == ops[2])
  true

proc main() =
  when not (defined(amd64) and defined(linux)) or defined(release) or
      defined(addressSanitizer):
    echo "  skipped (needs a debug run on x86-64 Linux)"
    return
  if findExe("objdump").len == 0 or findExe("nm").len == 0:
    echo "  skipped (objdump or nm missing)"
    return
  let
    here = currentSourcePath().parentDir
    dir = getTempDir() / "processSEG_vectorized"
    bin = dir / "veckernels.bin"
  createDir(dir)
  let build = execCmdEx(quoteShell(getCurrentCompilerExe()) &
    " c -f --hints:off -d:release --path:" & quoteShell(here / ".." / "src") &
    " --nimcache:" & quoteShell(dir / "nc") & " -o:" & quoteShell(bin) & " " &
    quoteShell(here / "utils" / "veckernels.nim"))
  doAssert build.exitCode == 0, build.output
  let syms = execCmdEx("nm " & quoteShell(bin))
  doAssert syms.exitCode == 0, syms.output
  for k in kernels:
    var found = 0
    for line in syms.output.splitLines:
      let sym = line.splitWhitespace()
      if sym.len < 3: continue
      let name = sym[^1]
      if not (name.startsWith(k & "_u") and name.endsWith("__veckernels")):
        continue
      inc found
      let dis = execCmdEx("objdump -d --no-show-raw-insn --disassemble=" &
        name & " " & quoteShell(bin))
      doAssert dis.exitCode == 0, dis.output
      var packed = 0
      var memset = false
      for l in dis.output.splitLines:
        let cols = l.split('\t')
        if cols.len >= 2:
          let insn = cols[1 .. ^1].join(" ").strip()
          if insn.isPacked: inc packed
          if "memset" in insn: memset = true
      doAssert packed > 0 or (memset and k in memsetOk),
        name & " has no packed SIMD instructions"
    doAssert found > 0, "no instance of " & k & " in the kernel build"
    echo "  ", k, ": ", found, " instance(s) vectorized"

main()
