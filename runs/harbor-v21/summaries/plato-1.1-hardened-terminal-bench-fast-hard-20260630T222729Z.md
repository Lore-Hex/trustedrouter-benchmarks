# Plato 1.1 Hardened Terminal-Bench v2.1 Fast-Hard Run

- Job: `plato-1-1-hardened-fast-hard-8-20260630T200940Z`
- Model route: `trustedrouter/advisor` via `https://api.trustedrouter.com/v1`.
- Plato 1.1 definition: worker `xiaomi/mimo-v2.5-pro-ultraspeed`, advisor `trustedrouter/prometheus`, max one advice call.
- Official score: **4/8 = 0.500**. Harbor mean: `0.5`.
- Wall time: **135.33 min**. Cost: **$2.9274**.
- Route calls: **124**; worker starts: **140**; advisor calls: **27** (`13` advice, `14` final).
- Harbor retries: **0**.
- Replay manifest: `runs/harbor-v21/plato-1-1-hardened-fast-hard-8-20260630T200940Z/replay_manifest.json` (688 files, 12345247 bytes).

| Task | Status | Reward | Min | Steps | Calls | Advisor | Worker fails | Cost | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `feal-differential-cryptanalysis` | PASS | 1.0 | 11.39 | 20 | 21 | 5 | 0 | $0.6760 |  |
| `model-extraction-relu-logits` | ERROR | err | 51.7 | 25 | 26 | 9 | 7 | $0.8511 | VerifierTimeoutError after 1440s; agent trace produced a locally checked solution before verifier hung. |
| `password-recovery` | PASS | 1.0 | 1.87 | 12 | 11 | 0 | 0 | $0.1179 | Official pass; recovered password is intentionally omitted from summary. |
| `feal-linear-cryptanalysis` | ERROR | err | 49.13 | 13 | 18 | 6 | 4 | $0.7179 | VerifierTimeoutError after 1440s; agent recovered keys and locally verified decryptions, but verifier hung in apt-get update. |
| `configure-git-webserver` | PASS | 1.0 | 6.57 | 15 | 14 | 3 | 2 | $0.1171 | End-to-end SSH clone, push, post-receive deploy, and curl test passed in episode. |
| `polyglot-rust-c` | FAIL | 0.0 | 2.63 | 6 | 7 | 0 | 0 | $0.0594 | Local rustc/g++ smoke passed in the episode, but official verifier returned reward 0.0. |
| `fix-code-vulnerability` | PASS | 1.0 | 8.84 | 22 | 22 | 3 | 1 | $0.3618 | Official pass after report.jsonl creation and pytest verification. |
| `cancel-async-tasks` | FAIL | 0.0 | 3.2 | 5 | 5 | 1 | 0 | $0.0262 | Verifier returned reward 0.0 after short local implementation attempt. |

## Caveats
- This is a full 8-task hardened rerun, but two tasks are official VerifierTimeoutError exceptions, so the true official score is 4/8, not 6/8 local-success-looking.
- model-extraction-relu-logits timed out in the official verifier after 1440s even though the agent trace produced a locally checked solution.
- feal-linear-cryptanalysis timed out in the official verifier after 1440s; the verifier stdout was stuck in apt-get update, while the agent trace had locally verified recovered keys/decryptions.
- polyglot-rust-c compiled and ran locally in the episode, but the official verifier scored it 0.0.
- No Harbor retries were recorded for this run.

## Internal Usage
- `xiaomi/mimo-v2.5-pro-ultraspeed`: 132 top-level attempt events, $2.7191
- `z-ai/glm-5.2`: 21 top-level attempt events, $0.4303
