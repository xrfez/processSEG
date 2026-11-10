# Shared switches for every test in tests/ (the project's config.nims still
# applies: outdir, CPU target).
switch("path", "$projectdir/../src")
switch("path", "$projectdir")

# `-d:addressSanitizer` (the `nim asan` task): ASan with C malloc, so every
# allocation is tracked, and no Nim signal handler, so ASan reports directly.
# gcc and clang (Linux, macOS) or MSVC (`--cc:vcc`); MinGW has no ASan.
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
