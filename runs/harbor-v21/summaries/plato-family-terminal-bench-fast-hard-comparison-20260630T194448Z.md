# Plato Family Terminal-Bench v2.1 Fast-Hard Comparison

| Model | Score | Wall Min | Cost | Route Calls | Advisor Calls | Retries | Exceptions | Job |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `plato-1.1` | 4/8 = 0.500 | 34.66 | $1.1763 | 54 | 11 | 0 | 3 | `plato-1-1-fast-hard-8-20260630T145006Z` |
| `plato-1.0` | 4/8 = 0.500 | 154.36 | $2.2900 | 278 | 28 | 0 | 0 | `plato-1-0-fast-hard-8-20260630T155703Z` |
| `plato-pro-1.0` | 4/8 = 0.500 | 67.29 | $6.1946 | 185 | 7 | 1 | 0 | `plato-pro-1-0-fast-hard-8-20260630T183438Z` |

## Task Outcomes

| Task | Plato 1.1 | Plato 1.0 | Plato Pro 1.0 |
|---|---:|---:|---:|
| `feal-differential-cryptanalysis` | FAIL (0.0, 8 calls) | FAIL (0.0, 63 calls) | PASS (1.0, 22 calls) |
| `model-extraction-relu-logits` | VERIFIER_TIMEOUT | FAIL (0.0, 11 calls) | FAIL (0.0, 14 calls) |
| `password-recovery` | PASS (1.0, 6 calls) | PASS (1.0, 69 calls) | PASS (1.0, 13 calls) |
| `feal-linear-cryptanalysis` | PASS (1.0, 10 calls) | PASS (1.0, 37 calls) | FAIL (0.0, 61 calls) |
| `configure-git-webserver` | PASS (1.0, 16 calls) | PASS (1.0, 24 calls) | PASS (1.0, 9 calls) |
| `polyglot-rust-c` | SETUP_EXCEPTION | FAIL (0.0, 31 calls) | FAIL (0.0, 49 calls) |
| `fix-code-vulnerability` | SETUP_EXCEPTION | PASS (1.0, 29 calls) | PASS (1.0, 11 calls) |
| `cancel-async-tasks` | PASS (1.0, 4 calls) | FAIL (0.0, 14 calls) | FAIL (0.0, 6 calls) |

## Notes
- Plato 1.0 and Plato Pro 1.0 were rerun after Docker apt retry/prebuild setup and 8x verifier timeout multipliers.
- Both hardened reruns completed with zero final Harbor setup/verifier exceptions.
- Plato Pro 1.0 recorded one Harbor retry after a TrustedRouter/InternalError read-timeout path, then completed successfully.

## Caveat Snapshot
- `plato-1.1`: model-extraction-relu-logits reached agent completion but Harbor verifier timed out after 180 seconds; official score is exception, not pass.; polyglot-rust-c and fix-code-vulnerability failed before the agent ran due Docker build/package installation errors; these are harness/environment setup exceptions in this run.; password-recovery answer value is intentionally omitted from this summary; full replay artifacts retain the trace.
- `plato-1.0`: No Harbor setup/verifier exceptions in this run.; model-extraction-relu-logits reached verifier and scored 0.0; this validates the verifier-timeout multiplier path, not model correctness.; polyglot-rust-c and fix-code-vulnerability reached scored results; prior Docker setup exceptions did not recur.
- `plato-pro-1.0`: No final Harbor setup/verifier exceptions in this run.; Harbor recorded 1 retry during the run after a TrustedRouter/InternalError read-timeout path; final job still completed with zero errored trials.; model-extraction-relu-logits reached verifier and scored 0.0; no verifier timeout.; polyglot-rust-c and fix-code-vulnerability reached scored results; prior Docker setup exceptions did not recur.
