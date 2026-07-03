# Athena Terminal-Bench 2.1 Fast-Hard Summary

- Model: `trustedrouter/athena`
- Base URL: `https://api.trustedrouter.com/v1`
- Subset: Terminal-Bench 2.1 Fast-Hard, 8 tasks
- Main job: `athena-fast-hard-8-20260703T163316Z`
- FEAL differential replay regrade: `athena-fast-hard-feal-diff-regrade-replay-20260703T172331Z`
- Strict score: 5/8 = 62.50%
- Partial Macro: 5.833333/8 = 72.92%
- Partial Micro: 16/19 verifier tests = 84.21%
- Calls: 62
- Advisor calls: 5
- Cost: $5.195268
- Main job runtime: 49m05s

The raw Harbor result was 4 pass, 3 fail, and 1 verifier exception. The exception was `feal-differential-cryptanalysis`, caused by a local verifier bootstrap patch bug that emitted invalid shell syntax. Replaying the saved Athena responses against the fixed verifier passed, so the final score uses that replay-backed grade.

Partial Macro is computed as the average of each task's verifier partial score (`passed_tests / total_tests`) using the best verifier-backed run for that task.

## Task Scores

| Task | Strict | Partial | Tests | Calls | Advisor calls | Cost | Trial time | Model time | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `feal-differential-cryptanalysis` | PASS | 1.000000 | 1/1 | 5 | 0 | $0.269247 | 149.2s | 123.6s | Replayed regrade fixed local verifier patch bug. |
| `model-extraction-relu-logits` | FAIL | 0.000000 | 0/1 | 6 | 3 | $1.251053 | 864.2s | 662.4s | Hard-coded the visible 20-hidden-unit instance; verifier rewrote to a hidden 30-unit instance. |
| `password-recovery` | PASS | 1.000000 | 2/2 | 12 | 0 | $0.618451 | 215.5s | 174.7s | Passed verifier. |
| `feal-linear-cryptanalysis` | PASS | 1.000000 | 1/1 | 8 | 0 | $1.000253 | 449.2s | 403.0s | Passed verifier. |
| `configure-git-webserver` | FAIL | 0.000000 | 0/1 | 10 | 0 | $0.333656 | 182.3s | 94.0s | Failed with HTTP 000; web service was not reachable at verifier time. |
| `polyglot-rust-c` | PASS | 1.000000 | 1/1 | 8 | 2 | $1.272850 | 874.5s | 780.9s | Passed verifier. |
| `fix-code-vulnerability` | PASS | 1.000000 | 6/6 | 9 | 0 | $0.359862 | 117.0s | 77.2s | Passed verifier. |
| `cancel-async-tasks` | FAIL | 0.833333 | 5/6 | 4 | 0 | $0.089896 | 89.4s | 43.4s | Failed queued-task cancellation cleanup edge. |

## Remaining Strict Failures

- `model-extraction-relu-logits`: needs a solver that infers the hidden verifier architecture instead of fitting only the visible `forward.py`.
- `configure-git-webserver`: needs stronger final service verification before stopping.
- `cancel-async-tasks`: only the queued-task cancellation cleanup edge failed.

## Artifacts

- Main run: `runs/harbor-v21/athena-fast-hard-8-20260703T163316Z`
- Main driver log: `runs/harbor-v21/athena-fast-hard-8-20260703T163316Z.driver.log`
- Replay regrade: `runs/harbor-v21/athena-fast-hard-feal-diff-regrade-replay-20260703T172331Z`
- Replay regrade driver log: `runs/harbor-v21/athena-fast-hard-feal-diff-regrade-replay-20260703T172331Z.driver.log`
