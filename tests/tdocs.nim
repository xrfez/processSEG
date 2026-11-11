## The documentation's checks (task 9.3): the usage guides' examples run, and
## `docs/api.md` names every symbol the umbrella exports.
##
## **Examples.** Every fenced code block of `docs/guide/*.md` and of
## `README.md` whose info string is `nim` is a complete program and is
## compiled and run here, so the guides cannot go stale. Conventions for the pages:
##
## - ```` ```nim ```` blocks run; ```` ```nim skip ```` blocks (any info
##   string with `skip` after `nim`) are shown but not compiled; other
##   languages (`sh`, `text`) are ignored;
## - a block imports what it uses (`import parseSEG, processSEG`) and must
##   not import the test utilities: it builds its own data (synthetic
##   `TraceKeys` and `rowsOf(keys, numSamples)`, or a small SEG-Y written
##   with parseSEG's `openSeismicWriter`);
## - blocks check their results with `doAssert`, so a wrong number fails.
##
## Each block becomes one module, `doc_<page>_<line>.nim` (`line`: the
## block's opening fence), imported in page order by one generated main
## program, so all the examples cost a single compile (about as long as one
## test file), and each block keeps its own module scope. Their top-level
## code runs at start-up, in that order, in one fresh scratch directory:
## blocks that write files use their own names. A compile error or a failed
## `doAssert` names the generated module, `doc_<page>_<fence line>.nim`, at
## the line of the page (the module is padded with blank lines to match). The program is built with the project's search paths (`src`,
## `tests`, and the Atlas paths of `nim.cfg`, made absolute) and with this
## test's own `-d:release`, `-d:danger` or `-d:addressSanitizer`, so
## `nim test`, `nim testRelease` and `nim asan` run the examples in each
## mode.
##
## **API reference.** `nim jsondoc --project` (with `--doccmd:skip`: no
## runnable examples are run) lists every exported symbol of every module,
## with the compiler's own view of what is exported, so nothing here parses
## Nim declarations. The modules checked are those `src/processSEG.nim`
## re-exports (its `import` paths, matched by name to its `export` list),
## plus the single symbols those modules re-export themselves
## (`export edges.TrendRemoval`, `export SampleType`), and every enum value
## of their enum types. Each name must occur in `docs/api.md`: identifiers
## as whole words, operators (`` `+` ``, `` `=destroy` ``) as text.

import std/[algorithm, assertions, json, os, osproc, sets, strutils]

const
  root = currentSourcePath().parentDir.parentDir
  guideDir = root / "docs" / "guide"

type
  Snippet = object
    page: string ## Page name without `.md`.
    line: int    ## 1-based line of the opening fence.
    code: string

proc fail(msg: string) {.noreturn.} =
  ## Stops with `msg`, keeping the scratch directory (the generated modules)
  ## for a look.
  quit "tdocs: " & msg

proc run(cmd, what: string, dir = ""): string =
  ## Runs `cmd` in `dir`; fails with its output when it does not exit with 0.
  let (o, code) = execCmdEx(cmd, workingDir = dir)
  if code != 0:
    echo o
    fail what & " failed (exit code " & $code & "): " & cmd
  o

proc snippets(path: string): seq[Snippet] =
  ## The runnable `nim` blocks of the Markdown file `path`, without the
  ## indentation of their fence (blocks inside list items).
  let page = path.splitFile.name
  var inBlock, keep = false
  var indent = 0
  var cur: Snippet
  let lines = readFile(path).splitLines
  for i, line in lines:
    let s = line.strip
    if not inBlock and s.startsWith("```"):
      inBlock = true
      indent = line.len - line.strip(trailing = false).len
      let info = s[3 .. ^1].splitWhitespace
      keep = info.len > 0 and info[0] == "nim" and "skip" notin info
      cur = Snippet(page: page, line: i + 1)
    elif inBlock and s == "```":
      inBlock = false
      if keep: result.add cur
    elif inBlock and keep:
      let cut = min(indent, line.len - line.strip(trailing = false).len)
      cur.code.add line[cut .. ^1] & "\n"
  if inBlock: fail path & ": a code block is not closed"

proc ident(s: string): string =
  for c in s: result.add(if c in IdentChars: c else: '_')

proc nimPaths(): string =
  ## The project's search paths, absolute, for a program outside the tree.
  result = " --path:" & quoteShell(root / "src") & " --path:" & quoteShell(root / "tests")
  for line in readFile(root / "nim.cfg").splitLines:
    let s = line.strip
    if s == "--noNimblePath":
      result.add " --noNimblePath"
    elif s.startsWith("--path:"):
      result.add " --path:" & quoteShell(root / s["--path:".len .. ^1].strip(chars = {'"'}))

proc modeFlags(): string =
  ## This test's build mode, passed on to the examples.
  when defined(danger): result.add " -d:danger"
  elif defined(release): result.add " -d:release"
  when defined(addressSanitizer):
    result.add " -d:addressSanitizer --debugger:native -d:noSignalHandler -d:useMalloc" &
      " --passC:-fsanitize=address --passC:-fno-omit-frame-pointer" &
      " --passL:-fsanitize=address --passL:-fno-omit-frame-pointer"

let tmp = getTempDir() / ("processseg_tdocs_" & $getCurrentProcessId())
removeDir(tmp)
createDir(tmp)

# ------------------------------------------------------------- examples --

block guide_examples:
  var pages: seq[string]
  for f in walkFiles(guideDir / "*.md"): pages.add f
  pages.sort()
  pages.add root / "README.md" # its quick start, as `doc_README_<line>`
  var all: seq[Snippet]
  for f in pages: all.add snippets(f)
  doAssert all.len > 0, "no examples found in " & guideDir
  let gen = tmp / "nim"
  createDir(gen)
  var main = "## Generated by tests/tdocs.nim: runs every Nim example of docs/guide.\n"
  for s in all:
    let name = "doc_" & ident(s.page) & "_" & $s.line
    # Padded so that a line of the module is the same line of the page.
    writeFile(gen / name & ".nim", "# " & s.page & ".md, the block at line " &
      $s.line & repeat('\n', s.line) & s.code)
    main.add "import " & name & "\n"
  writeFile(gen / "docs_main.nim", main)
  let exe = gen / "docs_main".addFileExt(ExeExt)
  discard run(getCurrentCompilerExe().quoteShell & " c --hints:off --warnings:off" &
    nimPaths() & modeFlags() & " -d:nimAllocStats --nimcache:" & quoteShell(tmp / "nimcache") &
    " -o:" & quoteShell(exe) & " " & quoteShell(gen / "docs_main.nim"),
    "compiling the guide examples (module doc_<page>_<fence line>)")
  let runDir = tmp / "run"
  createDir(runDir)
  discard run(quoteShell(exe), "running the guide examples (module doc_<page>_<fence line>)",
    runDir)
  echo "tdocs: ", all.len, " examples of ", pages.len, " guide pages ok"

# -------------------------------------------------------- API reference --

proc hasWord(text, w: string): bool =
  ## `w` occurs in `text` with no identifier character on either side.
  var i = text.find(w)
  while i >= 0:
    let before = i == 0 or text[i - 1] notin IdentChars
    let after = i + w.len >= text.len or text[i + w.len] notin IdentChars
    if before and after: return true
    i = text.find(w, i + 1)
  false

proc enumValues(code: string): seq[string] =
  ## The values of an enum declaration as `nim jsondoc` prints it
  ## (`Name = enum a, ## doc` ...).
  let body = code[code.find("enum") + 4 .. ^1]
  for line in body.splitLines:
    let decl = line.split("##")[0]
    for part in decl.split(','):
      let v = part.split('=')[0].strip
      if v.len > 0 and v.validIdentifier: result.add v

proc umbrellaModules(umbrella: string): seq[string] =
  ## The module paths (relative to `src/processSEG/`) that the umbrella
  ## re-exports: its `import` paths whose last part its `export` list names.
  var imported: seq[string]
  var exported: HashSet[string]
  var inExport = false
  for raw in umbrella.splitLines:
    let line = raw.split('#')[0]
    if line.startsWith("import processSEG/"):
      let spec = line["import processSEG/".len .. ^1].strip
      let b = spec.find('[')
      if b < 0:
        imported.add spec
      else:
        for m in spec[b + 1 ..< spec.find(']')].split(','):
          imported.add spec[0 ..< b] & m.strip
    if line.startsWith("export "):
      inExport = true
    elif not line.startsWith(" "):
      inExport = false
    if inExport:
      for w in line.replace("export ", "").split(','):
        if w.strip.len > 0: exported.incl w.strip
  for m in imported:
    if m.splitPath.tail in exported: result.add m
  doAssert result.len == exported.len, "every exported module is imported by path"

block api_md_names_every_export:
  let docOut = tmp / "jsondoc"
  discard run(getCurrentCompilerExe().quoteShell & " jsondoc --project --doccmd:skip" &
    " --hints:off --warnings:off" & nimPaths() & " --nimcache:" &
    quoteShell(tmp / "nimcache_doc") & " --outdir:" & quoteShell(docOut) & " " &
    quoteShell(root / "src" / "processSEG.nim"), "nim jsondoc")
  var names: seq[(string, string)] # (module, name)
  proc addEntries(module: string, entries: JsonNode, only = "") =
    for e in entries:
      let name = e["name"].getStr
      if only.len > 0 and name != only: continue
      names.add (module, name)
      if e["type"].getStr == "skType" and "= enum" in e["code"].getStr:
        for v in enumValues(e["code"].getStr): names.add (module, v)
  let modules = umbrellaModules(readFile(root / "src" / "processSEG.nim"))
  doAssert modules.len > 20
  for m in modules:
    addEntries(m, parseFile(docOut / "processSEG" / m & ".json")["entries"])
    # Single symbols the module re-exports: `export a.B` from a sibling
    # module, `export B` from parseSEG (no JSON of its own here).
    for raw in readFile(root / "src" / "processSEG" / m & ".nim").splitLines:
      if raw.startsWith("export "):
        for item in raw["export ".len .. ^1].split(','):
          let it = item.strip
          let dot = it.find('.')
          if dot < 0:
            names.add (m, it)
          else:
            let src = m.parentDir / it[0 ..< dot]
            addEntries(m, parseFile(docOut / "processSEG" / src & ".json")["entries"],
              only = it[dot + 1 .. ^1])
  let api = readFile(root / "docs" / "api.md")
  var missing: seq[string]
  var seen: HashSet[string]
  for (m, n) in names:
    if n in seen: continue
    seen.incl n
    let bare = n.strip(chars = {'`'})
    let found =
      if bare.validIdentifier: api.hasWord(bare)
      else: bare in api
    if not found: missing.add m & ": " & bare
  if missing.len > 0:
    fail "docs/api.md does not name " & $missing.len & " exported symbols:\n  " &
      missing.join("\n  ")
  echo "tdocs: docs/api.md names all ", seen.len, " exported names"

removeDir(tmp)
