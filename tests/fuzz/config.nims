# libFuzzer builds of the fuzz targets in tests/fuzz (the `nim fuzz` task).
# `-d:fuzzStandalone` builds a replay binary instead (no libFuzzer): it runs
# each file named on its command line through the target.
switch("path", "$projectdir/../../src")
switch("path", "$projectdir/..")
switch("cc", "clang")
switch("panics", "on")
switch("define", "noSignalHandler")
switch("define", "useMalloc")
switch("debugger", "native")
when not defined(fuzzStandalone):
  switch("noMain", "on")
  switch("passC", "-fsanitize=fuzzer,address,undefined")
  switch("passL", "-fsanitize=fuzzer,address,undefined")
else:
  switch("passC", "-fsanitize=address,undefined")
  switch("passL", "-fsanitize=address,undefined")
