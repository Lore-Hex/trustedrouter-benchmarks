# Plato 1.2 Terminal-Bench v2.1 Fast-Hard Aborted Run

- Model route: `trustedrouter/advisor` via `https://api.trustedrouter.com/v1`.
- Plato 1.2 definition tested: worker `z-ai/glm-5.2-fast`; advisors `trustedrouter/prometheus-1.0-1m`, `moonshotai/kimi-k2.7-code`; max one advice call.
- Initial high-cap attempt: `plato-1-2-hardened-fast-hard-8-20260701T011823Z`, stopped after one runaway `model-extraction` attempt; observed cost before stop about `$0.7102`.
- Bounded attempt: `plato-1-2-bounded-fast-hard-8-20260701T012650Z`, `max_tokens=16384`, stopped at Feal-linear after 19 calls / `$1.6303` on that task.
- Proper observed score before repairs: **0/3 official graded tasks passed**, with one aborted task and four tasks not attempted.
- Replay manifests: `runs/harbor-v21/plato-1-2-hardened-fast-hard-8-20260701T011823Z/replay_manifest.json`, `runs/harbor-v21/plato-1-2-bounded-fast-hard-8-20260701T012650Z/replay_manifest.json`.

| Task | Status | Reward | Min | Episodes | Calls | Worker length | Worker failed | Advisor starts | Advisor-final | Cost | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cancel-async-tasks` | FAIL | 0.0 | 1.88 | 5 | 5 | 0 | 0 | 0 | 0 | $0.1794 |  |
| `feal-differential-cryptanalysis` | FAIL | 0.0 | 4.27 | 6 | 6 | 1 | 1 | 0 | 1 | $0.4640 | Verifier-infra contaminated: `gcc` missing during test C extension build; bootstrap patched afterward. |
| `feal-linear-cryptanalysis` | ABORT | null | 12.33 | 19 | 19 | 4 | 4 | 0 | 4 | $1.6303 | Stopped during heredoc/session recovery loop; Harbor cleanup recorded `Event loop is closed`. |
| `model-extraction-relu-logits` | FAIL | 0.0 | 7.03 | 6 | 6 | 2 | 2 | 2 | 2 | $0.5805 | Model self-verified success, official verifier returned 0.0. |

## Not Attempted
- `password-recovery`
- `configure-git-webserver`
- `polyglot-rust-c`
- `fix-code-vulnerability`

## Findings
- GLM 5.2 Fast repeatedly spent large bounded completions on internal derivation before acting.
- Feal differential official 0 was verifier-infra contaminated: test build failed because gcc was unavailable; bootstrap patch now installs gcc/libc6-dev.
- Feal linear got stuck in an incomplete heredoc/session recovery loop and was stopped at 19 calls / $1.630274.
- model-extraction self-verified success but official hidden verifier returned 0.0.
- cancel-async-tasks returned official 0.0 after 5 episodes.

## Follow-Up
- The verifier bootstrap now installs `gcc libc6-dev` in patched Dockerfiles and rewrites legacy Feal test-side `apt install -y gcc` into an idempotent guarded install.
- A fair Feal differential repair rerun should be run after this patch if we want to know whether the model solution itself passes.
