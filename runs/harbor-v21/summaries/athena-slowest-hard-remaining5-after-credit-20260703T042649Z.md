# Athena Slowest-Hard Remaining 5 Rerun After Credit Increase

- Job: `athena-slowest-hard-remaining5-after-credit-20260703T042649Z`
- Model: `trustedrouter/athena`
- Base URL: `https://api.trustedrouter.com/v1`
- Dataset: Terminal-Bench 2.1
- Task set: five remaining unresolved slowest-hard tasks rerun after credit increase
- Strict score: 2/5 (0.400)
- Partial macro: 0.400
- Partial micro: 9/14 (0.643)
- Top-level Harbor cost: $38.061682
- Sum of per-call metadata cost: $38.061682
- Total runtime: 3h 19m 38s
- Exceptions: 0

| Task | Reward | Partial | Calls | Episodes | Advisors | Advisor attempts | Cost $ | Wall min | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `make-doom-for-mips` | 0.0 | 0/3 | 60 | 60 | 4 | 4 | 9.388 | 28.8 | clean run |
| `protein-assembly` | 0.0 | 0/1 | 14 | 14 | 0 | 0 | 2.121 | 13.3 | 1 parser warnings |
| `gpt2-codegolf` | 0.0 | 0/1 | 61 | 60 | 5 | 5 | 21.317 | 110.5 | 1 empty/error retry; 5 parser warnings; verifier apt-get update prelude was manually unstuck; official pytest failed |
| `torch-pipeline-parallelism` | 1.0 | 3/3 | 10 | 10 | 7 | 7 | 4.463 | 42.2 | 1 parser warnings; flipped to pass vs prior slowest-hard run |
| `llm-inference-batching-scheduler` | 1.0 | 6/6 | 8 | 8 | 0 | 0 | 0.773 | 4.9 | flipped to pass vs prior slowest-hard run |

## Updated 12-Task Slowest-Hard Rollup

Using this rerun for the five rerun tasks and the previous Athena slowest-hard run for the other seven tasks:

- Strict score: 7/12 (0.583)
- Partial macro: 0.681
- Partial micro: 27/35 (0.771)

| Task | Reward | Partial | Source |
|---|---:|---:|---|
| `dna-assembly` | 1.0 | 1/1 | `athena-slowest-hard-12-20260702T035257Z` |
| `make-doom-for-mips` | 0.0 | 0/3 | `athena-slowest-hard-remaining5-after-credit-20260703T042649Z` |
| `protein-assembly` | 0.0 | 0/1 | `athena-slowest-hard-remaining5-after-credit-20260703T042649Z` |
| `regex-chess` | 1.0 | 4/4 | `athena-slowest-hard-12-20260702T035257Z` |
| `gpt2-codegolf` | 0.0 | 0/1 | `athena-slowest-hard-remaining5-after-credit-20260703T042649Z` |
| `install-windows-3.11` | 0.0 | 2/4 | `athena-slowest-hard-12-20260702T035257Z` |
| `fix-ocaml-gc` | 1.0 | 1/1 | `athena-slowest-hard-12-20260702T035257Z` |
| `video-processing` | 1.0 | 5/5 | `athena-slowest-hard-12-20260702T035257Z` |
| `make-mips-interpreter` | 0.0 | 2/3 | `athena-slowest-hard-12-20260702T035257Z` |
| `sparql-university` | 1.0 | 3/3 | `athena-slowest-hard-12-20260702T035257Z` |
| `torch-pipeline-parallelism` | 1.0 | 3/3 | `athena-slowest-hard-remaining5-after-credit-20260703T042649Z` |
| `llm-inference-batching-scheduler` | 1.0 | 6/6 | `athena-slowest-hard-remaining5-after-credit-20260703T042649Z` |

## Notes

- This proper rerun excludes the earlier streaming smoke attempt athena-smoke-llm-scheduler-after-credit-20260703T000000Z, which was cancelled after a stalled stream call.
- gpt2-codegolf verifier setup stalled in apt-get update; curl was already installed, so only the stuck apt-get update process was terminated and the official pytest verifier then ran to reward 0.
- torch-pipeline-parallelism flipped from the prior slowest-hard run failure to reward 1 in this rerun.
- llm-inference-batching-scheduler flipped from the prior slowest-hard run failure to reward 1 in this rerun.

## Artifacts

- Run directory: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-slowest-hard-remaining5-after-credit-20260703T042649Z`
- Replay manifest: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-slowest-hard-remaining5-after-credit-20260703T042649Z/replay_manifest.json`
- JSON summary: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/summaries/athena-slowest-hard-remaining5-after-credit-20260703T042649Z.json`
- Excluded streaming smoke run: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-smoke-llm-scheduler-after-credit-20260703T000000Z`
