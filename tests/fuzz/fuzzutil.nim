## Shared pieces of the fuzz targets: the libFuzzer entry glue and a byte
## reader that turns the fuzzer's input into parameters.

import std/math

type ByteReader* = object
  ## Reads values off the fuzzer's input; zeros once it runs out.
  data*: ptr UncheckedArray[byte]
  len*, pos*: int

func left*(r: ByteReader): int {.inline.} = r.len - r.pos

proc u8*(r: var ByteReader): int =
  if r.pos < r.len:
    result = int(r.data[r.pos])
    inc r.pos

proc u16*(r: var ByteReader): int = r.u8 or (r.u8 shl 8)

proc pick*(r: var ByteReader, n: int): int =
  ## 0 ..< n.
  r.u8 mod n

proc unit*(r: var ByteReader): float64 =
  ## 0 .. 1.
  float64(r.u16) / 65535

proc range*(r: var ByteReader, lo, hi: float64): float64 =
  ## Mostly `lo .. hi`; one in 16 a raw float64 (NaN, Inf, huge, subnormal).
  if r.pick(16) == 0:
    var bits: uint64
    for k in 0..7: bits = bits or (uint64(r.u8) shl (8 * k))
    cast[float64](bits)
  else:
    lo + r.unit * (hi - lo)

proc rawF32*(r: var ByteReader): float32 =
  var bits: uint32
  for k in 0..3: bits = bits or (uint32(r.u8) shl (8 * k))
  cast[float32](bits)

template fuzzTarget*(body: untyped) =
  ## Defines the libFuzzer entry points around `body`, which sees `data`
  ## (`ptr UncheckedArray[byte]`) and `len`. With `-d:fuzzStandalone`, a
  ## main that replays the files named on the command line.
  proc initialize(): cint {.cdecl, exportc: "LLVMFuzzerInitialize".} =
    when not defined(fuzzStandalone):
      {.emit: "N_CDECL(void, NimMain)(void); NimMain();".}

  proc testOneInput(data {.inject.}: ptr UncheckedArray[byte],
      len {.inject.}: int): cint {.cdecl, exportc: "LLVMFuzzerTestOneInput",
      raises: [].} =
    result = 0
    try:
      body
    except CatchableError as e:
      # Expected rejections are caught inside `body`; anything that gets
      # here is a finding.
      doAssert false, "unexpected " & $e.name & ": " & e.msg

  when defined(fuzzStandalone):
    import std/[cmdline, syncio]
    for i in 1..paramCount():
      var buf = readFile(paramStr(i))
      discard testOneInput(cast[ptr UncheckedArray[byte]](cstring(buf)), buf.len)
    stderr.write "replayed " & $paramCount() & " inputs\n"

func sameSample*(a, b: float64): bool {.inline.} =
  ## Equal, or both NaN.
  a == b or (a.isNaN and b.isNaN)
