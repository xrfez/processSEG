## Core library module for processSEG.
## Implements exported types and library operations.

type
  Context* = object
    ## Operational state container for processSEG.
    name*: string
    isActive*: bool

proc initContext*(name: string): Context =
  ## Initializes and returns a new Context instance.
  result = Context(name: name, isActive: true)

proc execute*(ctx: Context): bool =
  ## Executes the configured operation within the context.
  result = ctx.isActive
