## Must not compile: a Workspace cannot be copied (core/workspace).
## Expect: '=copy' is not available for type <Workspace>
import processSEG

var a = initWorkspace()
var b = a
a.reserve(WorkspaceSize(f32: 1))
b.reserve(WorkspaceSize(f32: 1))
