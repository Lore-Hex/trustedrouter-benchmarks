# Plato 1.0 Terminal-Bench v2.1 Fast-Hard Run

- Job: `plato-1-0-fast-hard-8-20260630T155703Z`
- Model route: `trustedrouter/plato-1.0` via `https://api.trustedrouter.com/v1`.
- Official score: **4/8 = 0.500**. Harbor mean: `0.5`.
- Wall time: **154.36 min**. Cost: **$2.2900**.
- Route calls: **278**; worker starts: **314**; advisor calls: **28** (`25` advice, `3` final).
- Replay manifest: `runs/harbor-v21/plato-1-0-fast-hard-8-20260630T155703Z/replay_manifest.json` (1466 files, 356858161 bytes).

| Task | Status | Reward | Min | Steps | Calls | Advisor | Worker fails | Cost | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `feal-differential-cryptanalysis` | FAIL | 0.0 | 52.49 | 61 | 63 | 2 | 1 | $0.9822 |  |
| `model-extraction-relu-logits` | FAIL | 0.0 | 5.07 | 12 | 11 | 0 | 0 | $0.0158 | Verifier ran and returned reward 0; no verifier timeout. |
| `password-recovery` | PASS | 1.0 | 17.63 | 61 | 69 | 11 | 0 | $0.6582 | Answer value omitted from summary; replay contains trace. |
| `feal-linear-cryptanalysis` | PASS | 1.0 | 21.55 | 34 | 37 | 4 | 1 | $0.3585 |  |
| `configure-git-webserver` | PASS | 1.0 | 7.33 | 25 | 24 | 1 | 0 | $0.0231 |  |
| `polyglot-rust-c` | FAIL | 0.0 | 38.41 | 29 | 31 | 6 | 1 | $0.1280 | Agent ran to scored result; no Docker setup exception. |
| `fix-code-vulnerability` | PASS | 1.0 | 4.68 | 29 | 29 | 2 | 0 | $0.0814 | Agent ran to scored result; no Docker setup exception. |
| `cancel-async-tasks` | FAIL | 0.0 | 7.18 | 15 | 14 | 2 | 0 | $0.0428 |  |

## Caveats
- No Harbor setup/verifier exceptions in this run.
- model-extraction-relu-logits reached verifier and scored 0.0; this validates the verifier-timeout multiplier path, not model correctness.
- polyglot-rust-c and fix-code-vulnerability reached scored results; prior Docker setup exceptions did not recur.

## Internal Usage
- `deepseek/deepseek-v4-flash`: 312 done events, $1.7703
- `trustedrouter/plato-pro-1.0`: 28 done events, $1.0653
