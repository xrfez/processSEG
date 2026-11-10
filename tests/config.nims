# Shared switches for every test in tests/ (the project's config.nims still
# applies: outdir, CPU target).
switch("path", "$projectdir/../src")
switch("path", "$projectdir")

# `-d:addressSanitizer` (the `nim asan` task): ASan with C malloc, so every
# allocation is tracked, and no Nim signal handler, so ASan reports directly.
when defined(addressSanitizer):
  switch("debugger", "native")
  switch("define", "noSignalHandler")
  switch("define", "useMalloc")
  when defined(windows):
    switch("passC", "/fsanitize=address")
  else:
    switch("passC", "-fsanitize=address -fno-omit-frame-pointer")
    switch("passL", "-fsanitize=address -fno-omit-frame-pointer")
