## The suite's shape: every `tests/t*.nim` exports `main` and runs it only
## when that file is the program, so the aggregate can import it without
## running it. Helper procs, consts and types may stay at top level; a
## top-level `let`, `var`, `block` or other statement would run on import.
##
## Break: delete `proc main*` from any `tests/t*.nim`, or drop
## `when isMainModule: main()`, or move a `block` back to column 0. This
## file then fails.

import std/[assertions, os, strutils]

proc fail(msg: string) {.noreturn.} =
  quit "tonesuite: " & msg

proc col0Kind(line: string): string =
  ## A classification of a column-0 line, or "" when it is not one.
  if line.len == 0 or line[0] in {' ', '\t'}: return ""
  let s = line.strip
  if s.len == 0 or s.startsWith("#"): return ""
  if s.startsWith("import ") or s.startsWith("from ") or s.startsWith("export ") or
      s.startsWith("include "):
    return "import"
  if s.startsWith("when ") and "import " in s:
    return "import"
  if s.startsWith("{.") and s.endsWith(".}"):
    return "pragma"
  if s.startsWith("proc main*"):
    return "main"
  if s.startsWith("when isMainModule:"):
    return "entry"
  for k in ["proc ", "func ", "template ", "macro ", "iterator ", "converter ",
      "method ", "type", "const"]:
    if s.startsWith(k): return "decl"
  "stmt"

const
  tq = "\"" & "\"" & "\""
  sq = "'" & "'" & "'"

proc check(path: string) =
  let text = readFile(path)
  let name = path.extractFilename
  if "proc main*" notin text:
    fail name & " has no `proc main*`"
  if not text.strip(leading = false).endsWith("when isMainModule: main()"):
    fail name & " does not end with `when isMainModule: main()`"
  var mains, entries = 0
  var inTriple = ""
  for line in text.splitLines:
    if inTriple.len > 0:
      # String contents stay at column 0; they are not statements.
      if line.strip.endsWith(inTriple):
        inTriple = ""
      continue
    if tq in line or sq in line:
      let q = if tq in line: tq else: sq
      if line.count(q) mod 2 == 1:
        inTriple = q
    let k = col0Kind(line)
    if k == "main": inc mains
    elif k == "entry": inc entries
    elif k == "stmt":
      fail name & " runs on import: " & line.strip
  if mains != 1:
    fail name & " has " & $mains & " `proc main*`"
  if entries != 1:
    fail name & " has " & $entries & " `when isMainModule` lines"

proc main*() =
  let testDir = currentSourcePath().parentDir
  var n = 0
  for f in walkFiles(testDir / "t*.nim"):
    let name = f.extractFilename
    if name == "tester.nim":
      discard
    else:
      check(f)
      inc n
  doAssert n > 0, "no tests found"
  echo "tonesuite: ", n, " test files export main"

when isMainModule: main()
