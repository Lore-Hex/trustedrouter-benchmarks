# Athena Terminal-Bench 2.1 Hard Partial Summary

- Generated: `2026-07-02T14:29:27Z`
- Model: `trustedrouter/athena`
- Metric: Partial Macro, computed as the mean of each task's verifier partial score (`passed_tests / total_tests`).
- Scope with Athena data: 22 tasks (`Slow-Hard` + `Slowest-Hard`).
- Missing Athena slice: the 8 `Fast-Hard` tasks were not run with Athena in the saved artifacts.

## Scores

| Scope | Strict | Partial Macro | Partial Micro | Notes |
|---|---:|---:|---:|---|
| Athena-run TB2.1 hard tasks | 12/22 = 54.55% | 15.000000/22 = 68.18% | 61/75 = 81.33% | Uses saved Athena Slow-Hard and Slowest-Hard artifacts. |
| Full 30-task TB2.1 hard suite, unrun Fast-Hard counted as 0 | 12/30 = 40.00% | 15.000000/30 = 50.00% | n/a | Conservative placeholder until Athena Fast-Hard is run. |

## By Slice

| Slice | Strict | Partial Macro | Partial Micro | Source |
|---|---:|---:|---:|---|
| Slow-Hard | 7/10 = 70.00% | 8.166667/10 = 81.67% | 41/45 = 91.11% | `athena-terminal-bench-2.1-slow-hard-final-20260702` |
| Slowest-Hard | 5/12 = 41.67% | 6.833333/12 = 56.94% | 20/30 = 66.67% | `athena-terminal-bench-2.1-slowest-hard-20260702T035257Z` |
| Fast-Hard | 0/8 if counted missing | 0.000000/8 if counted missing | n/a | No saved Athena run found. |

## Slowest-Hard Partial Components

`llm-inference-batching-scheduler` is conservatively counted as `0/1` because its verifier failed while downloading pytest dependencies before CTRF was produced, and the task container was already deleted before this summary was generated.

| Task | Strict | Partial | Tests |
|---|---:|---:|---:|
| `dna-assembly` | PASS | 1.000000 | 1/1 |
| `make-doom-for-mips` | FAIL | 0.000000 | 0/3 |
| `protein-assembly` | FAIL | 0.000000 | 0/1 |
| `regex-chess` | PASS | 1.000000 | 4/4 |
| `gpt2-codegolf` | FAIL | 0.000000 | 0/1 |
| `install-windows-3.11` | FAIL | 0.500000 | 2/4 |
| `fix-ocaml-gc` | PASS | 1.000000 | 1/1 |
| `video-processing` | PASS | 1.000000 | 5/5 |
| `make-mips-interpreter` | FAIL | 0.666667 | 2/3 |
| `sparql-university` | PASS | 1.000000 | 3/3 |
| `torch-pipeline-parallelism` | FAIL | 0.666667 | 2/3 |
| `llm-inference-batching-scheduler` | FAIL | 0.000000 | 0/1 |
