# Socrates Family Terminal-Bench 2.1 Hard Summary

Generated: `2026-06-29T20:00:00Z`

Partial Macro is the published benchmark metric on arena.ai for these runs.

## Aggregate

- Included models: `trustedrouter/socrates-1.0`, `trustedrouter/socrates-pro-1.0`, `trustedrouter/socrates-pro-plus-1.0`
- Distinct tasks: 30/30
- Full pass: 20/30 = 66.67%
- Partial Macro: 24.016667/30 = 80.06%
- Partial Micro: 84/99 = 84.85%
- Total best-run cost: $196.489358
- Total best-run episodes: 1463

## Task Results

| Task | Model | Reward | Partial | Tests | Episodes | Cost | Time | Trial |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `feal-differential-cryptanalysis` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 1/1 | 20 | $1.055 | 13m51s | `runs/harbor-v21/socrates-pro-plus-failed3-20260628T234131Z/feal-differential-cryptanalysis__4YgwXWY` |
| `model-extraction-relu-logits` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.000 | 0/1 | 7 | $0.148 | 4m23s | `runs/harbor-v21/socrates-pro-plus-failed3-20260628T234131Z/model-extraction-relu-logits__WhJqC8o` |
| `password-recovery` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 2/2 | 21 | $0.710 | 9m05s | `runs/harbor-v21/socrates-pro-plus-failed3-20260628T234131Z/password-recovery__F8f6Xnr` |
| `feal-linear-cryptanalysis` | `trustedrouter/socrates-1.0` | 1.000 | 1.000 | 1/1 | 51 | $1.026 | 41m45s | `runs/harbor-v21/socrates-stream-fast-hard-8-20260628T201945Z/feal-linear-cryptanalysis__8Nq4dkG` |
| `configure-git-webserver` | `trustedrouter/socrates-1.0` | 1.000 | 1.000 | 1/1 | 11 | $0.138 | 2m08s | `runs/harbor-v21/socrates-fast-hard-remaining6-20260628T215045Z/configure-git-webserver__mWSGxhQ` |
| `polyglot-rust-c` | `trustedrouter/socrates-1.0` | 1.000 | 1.000 | 1/1 | 17 | $0.744 | 9m28s | `runs/harbor-v21/socrates-fast-hard-remaining6-20260628T215045Z/polyglot-rust-c__7ZgJRuF` |
| `fix-code-vulnerability` | `trustedrouter/socrates-1.0` | 1.000 | 1.000 | 6/6 | 30 | $1.493 | 14m33s | `runs/harbor-v21/socrates-fast-hard-remaining6-20260628T215045Z/fix-code-vulnerability__f2pxfeX` |
| `cancel-async-tasks` | `trustedrouter/socrates-1.0` | 1.000 | 1.000 | 6/6 | 4 | $0.003 | 1m30s | `runs/harbor-v21/socrates-stream-fast-hard-8-20260628T201945Z/cancel-async-tasks__bWiDK6Y` |
| `train-fasttext` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.500 | 1/2 | 200 | $17.661 | 3h01m49s | `runs/harbor-v21/socrates-pro-plus-slower-hard-rerun4-20260629T044810Z/train-fasttext__rEKHeTW` |
| `path-tracing-reverse` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 3/3 | 45 | $6.182 | 26m29s | `runs/harbor-v21/socrates-pro-plus-slower-hard-rerun4-20260629T044810Z/path-tracing-reverse__iAhaqPS` |
| `extract-moves-from-video` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.500 | 1/2 | 92 | $5.495 | 1h09m01s | `runs/harbor-v21/socrates-pro-plus-slower-hard-rerun4-20260629T044810Z/extract-moves-from-video__VTWVBtv` |
| `sam-cell-seg` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 9/9 | 18 | $0.572 | 10m52s | `runs/harbor-v21/socrates-pro-plus-slower-hard-10-20260629T005938Z/sam-cell-seg__zkRcbmg` |
| `torch-tensor-parallelism` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 3/3 | 5 | $0.058 | 7m47s | `runs/harbor-v21/socrates-pro-plus-slower-hard-10-20260629T005938Z/torch-tensor-parallelism__XpKQWGX` |
| `path-tracing` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 5/5 | 117 | $34.776 | 1h47m36s | `runs/harbor-v21/socrates-pro-plus-slower-hard-rerun5-20260629T030017Z/path-tracing__uszkj8J` |
| `mcmc-sampling-stan` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 6/6 | 55 | $3.859 | 44m07s | `runs/harbor-v21/socrates-pro-plus-slower-hard-10-20260629T005938Z/mcmc-sampling-stan__G8XhFoc` |
| `circuit-fibsqrt` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 3/3 | 12 | $0.708 | 7m33s | `runs/harbor-v21/socrates-pro-plus-slower-hard-10-20260629T005938Z/circuit-fibsqrt__Jeqswyp` |
| `bn-fit-modify` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 9/9 | 14 | $0.272 | 14m30s | `runs/harbor-v21/socrates-pro-plus-slower-hard-rerun4-20260629T044810Z/bn-fit-modify__jKK8WqU` |
| `write-compressor` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 3/3 | 9 | $0.601 | 12m11s | `runs/harbor-v21/socrates-pro-plus-caveat-reruns-20260629T185743Z/write-compressor__DoRZuVN` |
| `dna-assembly` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 1/1 | 41 | $3.789 | 40m20s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/dna-assembly__HVrbbcz` |
| `make-doom-for-mips` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.667 | 2/3 | 189 | $24.723 | 1h17m57s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/make-doom-for-mips__wi2GJW2` |
| `protein-assembly` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 1/1 | 21 | $1.716 | 18m01s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/protein-assembly__SztfDvE` |
| `regex-chess` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.250 | 1/4 | 90 | $12.481 | 1h26m26s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/regex-chess__XTxRGkw` |
| `gpt2-codegolf` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.000 | 0/1 | 109 | $36.237 | 1h39m08s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/gpt2-codegolf__z4XvfRX` |
| `install-windows-3.11` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.500 | 2/4 | 63 | $4.588 | 37m05s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/install-windows-3.11__NaDLhpK` |
| `fix-ocaml-gc` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 1/1 | 47 | $1.428 | 37m57s | `runs/harbor-v21/socrates-pro-plus-caveat-reruns-20260629T185743Z/fix-ocaml-gc__LLLPU7s` |
| `video-processing` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.600 | 3/5 | 10 | $0.207 | 3m00s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/video-processing__ZWPz8rj` |
| `make-mips-interpreter` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.333 | 1/3 | 136 | $34.433 | 1h19m29s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/make-mips-interpreter__R2U47iz` |
| `sparql-university` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 3/3 | 5 | $0.120 | 2m13s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/sparql-university__TvWXz5V` |
| `torch-pipeline-parallelism` | `trustedrouter/socrates-pro-plus-1.0` | 0.000 | 0.667 | 2/3 | 8 | $0.125 | 7m26s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/torch-pipeline-parallelism__VJAMnwX` |
| `llm-inference-batching-scheduler` | `trustedrouter/socrates-pro-plus-1.0` | 1.000 | 1.000 | 6/6 | 16 | $1.141 | 5m36s | `runs/harbor-v21/socrates-pro-plus-slowest-hard-12-20260629T094328Z/llm-inference-batching-scheduler__zgUeWBE` |

## Notes

- password-recovery verifier result was normalized after replaying the verifier logic inside the live task container; password value is intentionally not recorded in this summary.
- path-tracing verifier result was normalized from saved reward.txt=1 and CTRF 5/5 after the Harbor verifier collection was cancelled.
- write-compressor was rerun and manually normalized from the task verifier assertions after official verifier setup stalled in apt; output decompressed byte-identically and satisfied size checks.
- fix-ocaml-gc was rerun and the official verifier passed.
