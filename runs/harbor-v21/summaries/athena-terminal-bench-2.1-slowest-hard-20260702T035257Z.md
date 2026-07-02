# Athena Terminal-Bench 2.1 Slowest-Hard Run

- Job: `athena-slowest-hard-12-20260702T035257Z`
- Model: `trustedrouter/athena`
- Base URL: `https://api.trustedrouter.com/v1`
- Dataset: Terminal-Bench 2.1
- Tasks: 12 slowest-hard tasks requested
- Strict score on completed graded tasks: 4/11
- Strict score counting verifier-stuck incomplete task as unresolved: 4/12
- Top-level recorded cost before interrupt: $62.127097
- Sum of per-call metadata cost: $67.670184
- Run stopped manually because `fix-ocaml-gc` verifier hung in `apt-get update` for >15 minutes before tests.

| Task | Reward | Status | Calls | Steps | Advisors | Cost $ | Wall min | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `dna-assembly` | 1.0 | pass | 13 | 14 | 0 | 4.702 | 42.2 | passed |
| `make-doom-for-mips` | 0.0 | fail | 60 | 61 | 1 | 9.377 | 25.7 | built/runs MIPS ELF, capped at runtime Doom init error |
| `protein-assembly` | 0.0 | fail | 17 | 18 | 0 | 2.424 | 12.5 | failed verifier despite internal validation |
| `regex-chess` | 1.0 | pass | 24 | 25 | 0 | 2.837 | 32.8 | passed |
| `gpt2-codegolf` | 0.0 | fail | 60 | 61 | 5 | 20.560 | 102.4 | fixed major layout issue; failed multi-token generation before cap |
| `install-windows-3.11` | 0.0 | fail | 40 | 41 | 0 | 4.720 | 21.8 | partial local progress, verifier failed |
| `fix-ocaml-gc` | - | no_result | 60 | 61 | 0 | 5.543 | 26.6 | patch found/applied; verifier stuck in apt before tests; no grade |
| `video-processing` | 1.0 | pass | 18 | 19 | 0 | 1.576 | 5.3 | passed |
| `make-mips-interpreter` | 0.0 | fail | 52 | 53 | 1 | 11.615 | 39.9 | local BMP looked OK; verifier failed |
| `sparql-university` | 1.0 | pass | 8 | 9 | 0 | 0.569 | 4.5 | passed |
| `torch-pipeline-parallelism` | 0.0 | fail | 11 | 12 | 0 | 1.957 | 22.7 | syntax-complete implementation; verifier failed |
| `llm-inference-batching-scheduler` | 0.0 | fail | 9 | 10 | 1 | 1.791 | 15.6 | local metrics claimed pass; verifier failed |

## Important Caveat

`fix-ocaml-gc` should not be counted as a model-graded failure from this run: Athena identified and applied a plausible one-line GC fix (`pool_sweep` should advance by `wh`, not `Whsize_hd(hd)`), and the OCaml build progressed into late `make opt.opt`; then the official verifier hung in `apt-get update` before tests ran. Harbor was manually stopped to avoid leaving the thread blocked indefinitely.

## Artifacts

- Run directory: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-slowest-hard-12-20260702T035257Z`
- Replay manifest: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-slowest-hard-12-20260702T035257Z/replay_manifest.json`
- JSON summary: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/summaries/athena-terminal-bench-2.1-slowest-hard-20260702T035257Z.json`
