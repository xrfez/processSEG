## parseSEG's generated demo files (`survey.sgy`, `shot.sgd`, `record.sgd`,
## `line.sg2`), written by parseSEG's own generator (`tools/demofiles`), as
## parseSEG's tests make them.

import std/os
import "../../deps/parseSEG/tools/demofiles"

var demoDirPath: string

proc demoDir*(): string =
  ## A directory holding the demo files, emptied and written on the first
  ## call of the process (the test runner runs one test at a time).
  if demoDirPath.len == 0:
    demoDirPath = getTempDir() / "processseg_demofiles"
    removeDir(demoDirPath)
    writeDemoFiles(demoDirPath)
  demoDirPath
