## Must not compile: a ProcessPool cannot be copied (core/pool).
## Expect: is not available for type <ProcessPool>
import processSEG

var a = newProcessPool(2)
var b = a
discard a.workers + b.workers
