# Plato Pro 1.0 Terminal-Bench v2.1 Fast-Hard Run

- Job: `plato-pro-1-0-fast-hard-8-20260630T183438Z`
- Model route: `trustedrouter/plato-pro-1.0` via `https://api.trustedrouter.com/v1`.
- Official score: **4/8 = 0.500**. Harbor mean: `0.5`.
- Wall time: **67.29 min**. Cost: **$6.1946**.
- Route calls: **185**; worker starts: **188**; advisor calls: **7** (`6` advice, `1` final).
- Harbor retries: **1**.
- Replay manifest: `runs/harbor-v21/plato-pro-1-0-fast-hard-8-20260630T183438Z/replay_manifest.json` (997 files, 32092351 bytes).

| Task | Status | Reward | Min | Steps | Calls | Advisor | Worker fails | Cost | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `feal-differential-cryptanalysis` | PASS | 1.0 | 9.52 | 21 | 22 | 2 | 1 | $0.5073 | Passed after Harbor retry from earlier InternalError. |
| `model-extraction-relu-logits` | FAIL | 0.0 | 3.27 | 15 | 14 | 0 | 0 | $0.2375 | Verifier ran and returned reward 0; no verifier timeout. |
| `password-recovery` | PASS | 1.0 | 3.37 | 14 | 13 | 0 | 0 | $0.3814 | Answer value omitted from summary; replay contains trace. |
| `feal-linear-cryptanalysis` | FAIL | 0.0 | 14.82 | 61 | 61 | 2 | 0 | $3.1451 |  |
| `configure-git-webserver` | PASS | 1.0 | 3.3 | 10 | 9 | 0 | 0 | $0.0823 |  |
| `polyglot-rust-c` | FAIL | 0.0 | 14.99 | 49 | 49 | 3 | 0 | $1.6302 | Agent ran to scored result; no Docker setup exception. |
| `fix-code-vulnerability` | PASS | 1.0 | 1.62 | 12 | 11 | 0 | 0 | $0.1528 | Agent ran to scored result; no Docker setup exception. |
| `cancel-async-tasks` | FAIL | 0.0 | 1.58 | 7 | 6 | 0 | 0 | $0.0579 |  |

## Caveats
- No final Harbor setup/verifier exceptions in this run.
- Harbor recorded 1 retry during the run after a TrustedRouter/InternalError read-timeout path; final job still completed with zero errored trials.
- model-extraction-relu-logits reached verifier and scored 0.0; no verifier timeout.
- polyglot-rust-c and fix-code-vulnerability reached scored results; prior Docker setup exceptions did not recur.

## Internal Usage
- `trustedrouter/prometheus-1.0`: 3 done events, $0.1267
- `z-ai/glm-5.2`: 187 done events, $6.0967
