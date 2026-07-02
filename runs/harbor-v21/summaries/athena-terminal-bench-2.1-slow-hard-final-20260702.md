# Athena Terminal-Bench 2.1 Slow-Hard Final Summary

- Model: `trustedrouter/athena`
- Base URL: `https://api.trustedrouter.com/v1`
- Subset: Terminal-Bench 2.1 Slow-Hard, 10 tasks
- Final repair job for the last two unknown rows: `athena-slow-hard-repair5-2-20260702T015803Z`
- Strict score: 7/10 = 70.00%
- Partial Macro: 8.166667/10 = 81.67%
- Partial Micro: 41/45 verifier tests = 91.11%
- Selected-run calls: 322
- Selected-run cost: $48.426741

Partial Macro is computed as the average of each task's verifier partial score (`passed_tests / total_tests`) using the best completed verifier-backed run for that task. `path-tracing-reverse` uses the completed repair1 verifier score of 2/3; the later repair3 one-pixel local result is diagnostic only because it did not reach a final verifier score.

## Last Two Reruns

The two previously unknown rows were rerun after pruning exhausted Docker networks and using the fixed TrustedRouter SDK timeout/retry adapter.

| Task | Result | Calls | Cost | Verifier tests | Notes |
|---|---:|---:|---:|---:|---|
| `write-compressor` | PASS | 4 | $0.526488 | 3/3 | Produced `/app/data.comp` at 2423 bytes, under the 2500-byte limit, with exact decompression. |
| `circuit-fibsqrt` | PASS | 20 | $6.410338 | 3/3 | Produced `/app/gates.txt` with 4908 lines, under the 32000-line limit, and passed examples, edge cases, and random checks. |

Repair5 summary: 2 completed, 0 errored, Harbor mean 1.0, recorded cost $6.936826.

Repair4 was an infrastructure miss before any model calls: Docker reported `all predefined address pools have been fully subnetted`. The unused Docker networks were pruned, and repair5 completed successfully.

## Final Task Scores

| Task | Strict | Partial | Tests | Calls | Cost | Selected source | Notes |
|---|---:|---:|---:|---:|---:|---|---|
| `train-fasttext` | FAIL | 0.500000 | 1/2 | 60 | $4.543670 | `athena-slow-hard-10-20260701T150915Z` | Accuracy failed at 0.51 vs required 0.62; size test passed. |
| `path-tracing-reverse` | FAIL | 0.666667 | 2/3 | 60 | $12.916060 | `athena-slow-hard-repair-6-20260701T184156Z` | Completed verifier had image existence/compile pass and similarity fail. |
| `extract-moves-from-video` | FAIL | 0.000000 | 0/2 | 60 | $4.668530 | `athena-slow-hard-10-20260701T150915Z` | No valid `/app/solution.txt` after the OCR attempt. |
| `sam-cell-seg` | PASS | 1.000000 | 9/9 | 30 | $2.150656 | `athena-slow-hard-10-20260701T150915Z` | Passed verifier. |
| `torch-tensor-parallelism` | PASS | 1.000000 | 3/3 | 5 | $0.356967 | `athena-slow-hard-10-20260701T150915Z` | Passed verifier. |
| `path-tracing` | PASS | 1.000000 | 5/5 | 48 | $14.656649 | `athena-slow-hard-repair3-6-20260701T230457Z` | Repaired from transport timeout to strict pass. |
| `mcmc-sampling-stan` | PASS | 1.000000 | 6/6 | 25 | $1.878120 | `athena-slow-hard-10-20260701T150915Z` | Passed verifier. |
| `circuit-fibsqrt` | PASS | 1.000000 | 3/3 | 20 | $6.410338 | `athena-slow-hard-repair5-2-20260702T015803Z` | Last-two rerun pass. |
| `bn-fit-modify` | PASS | 1.000000 | 9/9 | 10 | $0.319263 | `athena-slow-hard-10-20260701T150915Z` | Passed verifier. |
| `write-compressor` | PASS | 1.000000 | 3/3 | 4 | $0.526488 | `athena-slow-hard-repair5-2-20260702T015803Z` | Last-two rerun pass. |

## Remaining Strict Failures

- `train-fasttext`: verifier partial 1/2; needs accuracy improvement.
- `path-tracing-reverse`: verifier partial 2/3; later local attempt got close but did not produce a final verifier pass.
- `extract-moves-from-video`: verifier partial 0/2; needs a valid solution file.
