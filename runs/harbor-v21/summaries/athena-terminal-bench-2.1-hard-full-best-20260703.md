# Athena Terminal-Bench 2.1 Hard Full Best Summary

- Model: `trustedrouter/athena`
- Base URL: `https://api.trustedrouter.com/v1`
- Dataset: Terminal-Bench 2.1
- Scope: all 30 hard tasks
- Selection: best verifier-backed Athena result per task across Fast-Hard, Slow-Hard, and Slowest-Hard runs
- Strict score: 19/30 = 63.33%
- Partial Macro: 22.166667/30 = 73.89%
- Partial Micro: 84/99 verifier tests = 84.85%

## Components

| Subset | Strict | Partial Macro | Partial Micro | Calls | Advisor calls | Cost | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| Fast-Hard | 5/8 | 5.833333/8 | 16/19 | 62 | 5 | $5.195268 | `athena-terminal-bench-2.1-fast-hard-20260703T163316Z.json` |
| Slow-Hard | 7/10 | 8.166667/10 | 41/45 | 322 | n/a | $48.426741 | `athena-terminal-bench-2.1-slow-hard-final-20260702.json` |
| Slowest-Hard | 7/12 | 8.166667/12 | 27/35 | 368 | 17 | $69.624297 | `athena-slowest-hard-remaining5-after-credit-20260703T042649Z.json` |

The Fast-Hard score includes a replay-backed regrade for `feal-differential-cryptanalysis`; the raw run exception was a local verifier bootstrap patch bug, and the saved Athena responses pass against the fixed verifier.

The Slowest-Hard score uses the updated rollup from `athena-slowest-hard-remaining5-after-credit-20260703T042649Z.json`, which supersedes the earlier 5/12 strict summary.
