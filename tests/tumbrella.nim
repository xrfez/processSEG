## Smoke test: the umbrella module compiles and imports cleanly.

import std/assertions
{.warning[UnusedImport]: off.}
import processSEG

block umbrella_imports:
  doAssert declared(processSEG), "the umbrella module should be importable"
