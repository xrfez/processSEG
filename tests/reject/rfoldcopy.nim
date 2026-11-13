## Must not compile: a FoldMap cannot be copied (qc/fold); its counts are
## tensors, which a copy would share. `clone` makes an independent one.
## Expect: '=copy' is not available for type <FoldMap>
import processSEG

var a = initFoldMap(gridBinning(0, 0, 0, 10, 10, 4, 4))
var b = a
a.clear
b.clear
