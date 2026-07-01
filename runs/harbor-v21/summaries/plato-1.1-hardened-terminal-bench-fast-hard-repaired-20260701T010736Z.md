# Plato 1.1 Hardened Terminal-Bench v2.1 Fast-Hard Repaired Grade

- Base job: `plato-1-1-hardened-fast-hard-8-20260630T200940Z`
- Repair job: `plato-1-1-hardened-fast-hard-timeout-repair-2-20260701T003402Z`
- Model route: `trustedrouter/advisor` via `https://api.trustedrouter.com/v1`.
- Plato 1.1 definition: worker `xiaomi/mimo-v2.5-pro-ultraspeed`, advisor `trustedrouter/prometheus`, max one advice call.
- Raw Harbor score before verifier repair: **4/8 = 0.500** with 2 `VerifierTimeoutError` exceptions.
- Proper score after verifier repair: **5/8 = 0.625** with 0 exceptions.
- Repair job score on the two timeout tasks: **1/2 = 0.500**.
- Base job cost: **$2.9274**. Repair job cost: **$1.5930**. Substitution cost for the repaired 8-task grade: **$2.9513**.
- Repair replay manifest: `runs/harbor-v21/plato-1-1-hardened-fast-hard-timeout-repair-2-20260701T003402Z/replay_manifest.json`.
- Manual original-state regrade evidence: `runs/harbor-v21/manual-regrades/plato-1.1-hardened-fast-hard-20260701T003402Z/feal-linear-cryptanalysis/verifier-regrade.log`.

| Task | Status | Reward | Source | Min | Steps | Calls | Advisor | Worker fails | Cost | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `feal-differential-cryptanalysis` | PASS | 1.0 | original_job | 11.39 | 20 | 21 | 5 | 0 | $0.6760 |  |
| `model-extraction-relu-logits` | FAIL | 0.0 | repair_job | 20.58 | 19 | 25 | 8 | 2 | $0.7842 | Repaired timeout: normal verifier FAIL on hidden rows. |
| `password-recovery` | PASS | 1.0 | original_job | 1.87 | 12 | 11 | 0 | 0 | $0.1179 | Official pass; recovered password is intentionally omitted from summary. |
| `feal-linear-cryptanalysis` | PASS | 1.0 | repair_job | 12.96 | 15 | 15 | 6 | 4 | $0.8088 | Repaired timeout: fresh verifier PASS; original transcript also regraded PASS. |
| `configure-git-webserver` | PASS | 1.0 | original_job | 6.57 | 15 | 14 | 3 | 2 | $0.1171 | End-to-end SSH clone, push, post-receive deploy, and curl test passed in episode. |
| `polyglot-rust-c` | FAIL | 0.0 | original_job | 2.63 | 6 | 7 | 0 | 0 | $0.0594 | Local rustc/g++ smoke passed in the episode, but official verifier returned reward 0.0. |
| `fix-code-vulnerability` | PASS | 1.0 | original_job | 8.84 | 22 | 22 | 3 | 1 | $0.3618 | Official pass after report.jsonl creation and pytest verification. |
| `cancel-async-tasks` | FAIL | 0.0 | original_job | 3.2 | 5 | 5 | 1 | 0 | $0.0262 | Verifier returned reward 0.0 after short local implementation attempt. |

## What Changed
- `model-extraction-relu-logits`: original verifier timeout is now a clean verifier result, reward `0.0`. The verifier reached pytest and failed on hidden rows instead of hanging.
- `feal-linear-cryptanalysis`: original verifier timeout is now a clean verifier result, reward `1.0`. The original transcript recovered the same seeds and the fresh repair run also passed the official verifier.
- The verifier bootstrap patch preinstalls/skips `curl` and `uv` setup where possible and bounds the untrusted `steal.py` subprocess in `model-extraction` so verifier hangs become normal grades.
