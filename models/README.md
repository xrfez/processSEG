# Models

Small TLA-style models checked with `tlanif` (the `nim-formal-verification` skill).
They are not run in CI (it has no tlanif); rerun them after changing the code they cite.

| Model | Code | Checks |
|---|---|---|
| `pool_handoff.nif` | `src/processSEG/core/pool.nim` (`workerLoop`, `run`, `=destroy`) | every job runs exactly once on every worker before the caller returns; no lost wake-up (the caller on `done`, `joinThread` on a sleeping worker), including a destroy right after creation |

```sh
S=.claude/skills/nim-formal-verification/scripts/tlacheck.py
python3 $S models/pool_handoff.nif --expect:pass                   # 235 states
python3 $S models/pool_handoff.nif --expect:fail 'Witness=(true)'  # 2 jobs, then a clean shutdown
for m in 1 2 3 4 5; do python3 $S models/pool_handoff.nif --expect:fail Mut=$m; done
```

Mutants (each must fail): 1 a worker runs after a wake-up without rechecking; 2 the
worker's wait ignores `stop`; 3 the caller waits once instead of in a loop; 4 the last
worker does not signal `done`; 5 `signal` instead of `broadcast` on `wake`.
