# Plato 1.1 Terminal-Bench v2.1 Fast-Hard Run

- Job: `plato-1-1-fast-hard-8-20260630T145006Z`
- Local label: `plato-1.1` = `trustedrouter/advisor` with worker `xiaomi/mimo-v2.5-pro-ultraspeed` and advisor `trustedrouter/prometheus`.
- Official score: **4/8 = 0.500**. Harbor mean: `0.5`.
- Wall time: **34.66 min**. Cost: **$1.1763**.
- Route calls: **54**; worker model calls: **61**; Prometheus advisor calls: **11** (`4` advice, `7` final).
- Replay manifest: `runs/harbor-v21/plato-1-1-fast-hard-8-20260630T145006Z/replay_manifest.json` (338 files, 4193944 bytes).

| Task | Status | Reward | Min | Calls | Advisor | Worker fails | Cost | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `feal-differential-cryptanalysis` | FAIL | 0.0 | 4.79 | 8 | 2 | 1 | $0.1708 | Agent ran; verifier reward 0 |
| `model-extraction-relu-logits` | VERIFIER_TIMEOUT |  | 10.39 | 10 | 3 | 2 | $0.2010 | VerifierTimeoutError |
| `password-recovery` | PASS | 1.0 | 3.49 | 6 | 2 | 2 | $0.0552 |  |
| `feal-linear-cryptanalysis` | PASS | 1.0 | 6.17 | 10 | 1 | 0 | $0.5096 |  |
| `configure-git-webserver` | PASS | 1.0 | 7.08 | 16 | 2 | 1 | $0.2166 |  |
| `polyglot-rust-c` | SETUP_EXCEPTION |  | 0.54 | 0 | 0 | 0 |  | Docker/setup exception |
| `fix-code-vulnerability` | SETUP_EXCEPTION |  | 0.15 | 0 | 0 | 0 |  | Docker/setup exception |
| `cancel-async-tasks` | PASS | 1.0 | 2.03 | 4 | 1 | 1 | $0.0231 |  |

## Caveats
- model-extraction-relu-logits reached agent completion but Harbor verifier timed out after 180 seconds; official score is exception, not pass.
- polyglot-rust-c and fix-code-vulnerability failed before the agent ran due Docker build/package installation errors; these are harness/environment setup exceptions in this run.
- password-recovery answer value is intentionally omitted from this summary; full replay artifacts retain the trace.

## Internal Usage
- `xiaomi/mimo-v2.5-pro-ultraspeed@xiaomi/prepaid`: 61 calls, $1.0516
- `z-ai/glm-5.2@baseten/prepaid`: 11 calls, $0.1413

