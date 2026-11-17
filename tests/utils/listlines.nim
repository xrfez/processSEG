## Lines of `alltests --list`, for the runner's check. Windows text mode turns
## each `\n` into `\r\n` on a pipe; a comparison that keeps the `\r` never
## matches, and a log that honours `\r` shows none of the names.

import std/strutils

proc withoutCr*(s: string): string =
  ## `s` with every `\r` removed.
  result = newStringOfCap(s.len)
  for c in s:
    if c != '\r':
      result.add c

proc listedNames*(output: string): seq[string] =
  ## The names in `output`. Split on `\n` only, then strip `\r` from each
  ## line, so a Windows pipe's `\r\n` matches the name. A final newline does
  ## not add an empty name. Empty output is no names.
  if output.len == 0:
    return
  var line = ""
  for c in output:
    if c == '\n':
      result.add withoutCr(line)
      line.setLen(0)
    else:
      line.add c
  if line.len > 0:
    result.add withoutCr(line)
  elif result.len > 0 and result[^1].len == 0:
    result.setLen(result.len - 1)

proc exitHex*(code: int): string =
  ## `code` as an 8-digit hex Windows status (low 32 bits).
  "0x" & toHex(uint32(int32(code)))

proc firstDiff*(got, expected: seq[string]): string =
  ## The first line where `got` and `expected` differ.
  let n = min(got.len, expected.len)
  for i in 0 ..< n:
    if got[i] != expected[i]:
      return "line " & $i & ": got " & got[i].repr & " expected " & expected[i].repr
  if got.len > expected.len:
    return "line " & $expected.len & ": got " & got[expected.len].repr & " expected end"
  if got.len < expected.len:
    return "line " & $got.len & ": got end expected " & expected[got.len].repr
  ""

proc listFailure*(code: int, output: string, expected: seq[string]): string =
  ## What `alltests --list` did, when it does not match `expected`.
  let got = listedNames(output)
  "exit " & $code & " (" & exitHex(code) & "), output " & $output.len &
    " bytes, expected " & $expected.len & " names, " & firstDiff(got, expected)
