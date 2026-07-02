# Athena Terminal-Bench 2.1 Slowest-Hard Run

- Job: `athena-slowest-hard-12-20260702T035257Z`
- Model: `trustedrouter/athena`
- Base URL: `https://api.trustedrouter.com/v1`
- Dataset: Terminal-Bench 2.1
- Tasks: 12 slowest-hard tasks requested
- Strict score after `fix-ocaml-gc` regrade: 5/12
- Original Harbor score before regrade: 4/11 completed graded tasks; `fix-ocaml-gc` was verifier-stuck.
- Top-level recorded cost before interrupt: $62.127097
- Sum of per-call metadata cost: $67.670184
- Run stopped manually because `fix-ocaml-gc` verifier hung in `apt-get update` for >15 minutes before tests; it was later regraded in-place and passed.

| Task | Reward | Status | Calls | Steps | Advisors | Cost $ | Wall min | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `dna-assembly` | 1.0 | pass | 13 | 14 | 0 | 4.702 | 42.2 | passed |
| `make-doom-for-mips` | 0.0 | fail | 60 | 61 | 1 | 9.377 | 25.7 | built/runs MIPS ELF, capped at runtime Doom init error |
| `protein-assembly` | 0.0 | fail | 17 | 18 | 0 | 2.424 | 12.5 | failed verifier despite internal validation |
| `regex-chess` | 1.0 | pass | 24 | 25 | 0 | 2.837 | 32.8 | passed |
| `gpt2-codegolf` | 0.0 | fail | 60 | 61 | 5 | 20.560 | 102.4 | fixed major layout issue; failed multi-token generation before cap |
| `install-windows-3.11` | 0.0 | fail | 40 | 41 | 0 | 4.720 | 21.8 | partial local progress, verifier failed |
| `fix-ocaml-gc` | 1.0 | pass | 60 | 61 | 0 | 5.543 | 26.6 | regrade passed: 40 basic tests passed, pytest passed |
| `video-processing` | 1.0 | pass | 18 | 19 | 0 | 1.576 | 5.3 | passed |
| `make-mips-interpreter` | 0.0 | fail | 52 | 53 | 1 | 11.615 | 39.9 | local BMP looked OK; verifier failed |
| `sparql-university` | 1.0 | pass | 8 | 9 | 0 | 0.569 | 4.5 | passed |
| `torch-pipeline-parallelism` | 0.0 | fail | 11 | 12 | 0 | 1.957 | 22.7 | syntax-complete implementation; verifier failed |
| `llm-inference-batching-scheduler` | 0.0 | fail | 9 | 10 | 1 | 1.791 | 15.6 | local metrics claimed pass; verifier failed |

## Regrade Note

`fix-ocaml-gc` was regraded in-place from the saved Harbor container after bypassing only the flaky dependency-install prelude. The verifier body ran the clean testsuite restore, `make clean && ./configure && make -j4`, `make -C testsuite one DIR=tests/basic`, and the official pytest assertion. The regrade produced `40 tests passed`, pytest passed, and `reward-regrade.txt = 1`.

## Artifacts

- Run directory: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-slowest-hard-12-20260702T035257Z`
- Replay manifest: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-slowest-hard-12-20260702T035257Z/replay_manifest.json`
- Fix OCaml GC regrade log: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-slowest-hard-12-20260702T035257Z/fix-ocaml-gc__j7GxqWX/verifier/test-stdout-regrade.txt`
- Fix OCaml GC regrade reward: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/athena-slowest-hard-12-20260702T035257Z/fix-ocaml-gc__j7GxqWX/verifier/reward-regrade.txt`
- JSON summary: `/Users/jperla/claude/trustedrouter-benchmarks/runs/harbor-v21/summaries/athena-terminal-bench-2.1-slowest-hard-20260702T035257Z.json`
