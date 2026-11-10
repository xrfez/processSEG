## Test suite for processseg.
## Exercises module functionality using the standard unittest framework.

import std/unittest
import ../src/processSEG

suite "processseg test suite":
  test "basic initialization":
    let ctx = initContext("test_env")
    check ctx.name == "test_env"
    check ctx.isActive == true
    check execute(ctx) == true
