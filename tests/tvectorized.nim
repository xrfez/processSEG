## The hot kernel loops compile to packed SIMD: builds `utils/veckernels.nim`
## in `-d:release` (bound and overflow checks on outside the kernels, the CPU
## target from config.nims) and checks every instance of each kernel in the
## disassembly. Runs on x86-64 Linux with gcc and binutils, in the debug suite
## only (the sub-build has its own flags); skipped elsewhere. To check the
## check, `PROCESSSEG_VEC_FLAGS="--passC:-fno-tree-vectorize
## --passC:-fno-tree-slp-vectorize"` builds the kernels scalar: every kernel
## must then fail.

import std/[os, osproc, strutils]

const
  # Kernel instance symbols (`<name>_u<n>__veckernels`) that must hold packed
  # instructions. The mutes are not here: their zero fill is a memset call
  # whether or not GCC vectorizes, and their short taper calls `sin`.
  kernels = ["sum64", "sumIndexed64", "demean", "detrend", "scale", "clip",
    "reversePolarity", "taperPads",
    # FFT (instances of the lanes path only: the one-trace path, L = 1, is
      # scalar by design). The butterflies are inlined into the passes, the
      # split kernels into forwardImpl/inverseImpl.
    "pass2", "pass3", "pass4", "pass5", "forwardImpl", "inverseImpl",
    # Filter: the gain multiply of the lanes path (packing is strided, scalar).
    "gainLanes",
    # Gain statistics: Σx² (eight accumulators), max |x| (integer max).
    "sumSq64", "maxAbs"]
  # Non-generic kernels: their symbols carry their own module (and GCC clones
  # such as `.constprop.0` count too).
  moduleKernels = [("sosLanes", "ZfilterZiir")]

proc isPacked(insn: string): bool =
  ## Packed arithmetic, compares or conversions on vector registers; not
  ## moves, shuffles or broadcasts. Logic ops (`xor`, `and`, `or`) count only
  ## on 256-bit registers: a scalar float negation or abs is an `xorps`/`andps`
  ## with a constant on an `xmm` register.
  if "%xmm" notin insn and "%ymm" notin insn and "%zmm" notin insn:
    return false
  var m = insn.splitWhitespace()[0]
  if m.startsWith("v"): m = m[1 .. ^1]
  if not (m.startsWith("p") or m.endsWith("ps") or m.endsWith("pd")):
    return false
  for skip in ["mov", "shuf", "unpck", "broadcast", "perm", "insr", "extr",
      "blend"]:
    if skip in m: return false
  for logic in ["xor", "and", "or"]:
    if logic in m:
      return "%ymm" in insn or "%zmm" in insn
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
    " c -f --hints:off -d:release " & getEnv("PROCESSSEG_VEC_FLAGS") &
        " --path:" & quoteShell(here / ".." / "src") &
    " --nimcache:" & quoteShell(dir / "nc") & " -o:" & quoteShell(bin) & " " &
    quoteShell(here / "utils" / "veckernels.nim"))
  doAssert build.exitCode == 0, build.output
  let syms = execCmdEx("nm " & quoteShell(bin))
  doAssert syms.exitCode == 0, syms.output
  var all: seq[(string, string)]
  for k in kernels: all.add (k, "__veckernels")
  for (k, m) in moduleKernels: all.add (k, m)
  for (k, suffix) in all:
    var found = 0
    for line in syms.output.splitLines:
      let sym = line.splitWhitespace()
      if sym.len < 3: continue
      let name = sym[^1]
      if not (name.startsWith(k & "_u") and name.split('.')[0].endsWith(suffix)):
        continue
      inc found
      let dis = execCmdEx("objdump -d --no-show-raw-insn --disassemble=" &
        name & " " & quoteShell(bin))
      doAssert dis.exitCode == 0, dis.output
      var packed = 0
      for l in dis.output.splitLines:
        let cols = l.split('\t')
        if cols.len >= 2 and cols[1 .. ^1].join(" ").strip().isPacked:
          inc packed
      doAssert packed > 0, name & " has no packed SIMD instructions"
    doAssert found > 0, "no instance of " & k & " in the kernel build"
    echo "  ", k, ": ", found, " instance(s) vectorized"

main()
