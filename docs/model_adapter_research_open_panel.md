# Open-panel model adapter research

Date: 2026-06-25

Purpose: identify model-specific request settings before rerunning the 5 MMLU-Pro + 5 ProtocolQA open-panel slice. We want full model quality, not low-reasoning shortcuts.

## Current policy

- Do not disable or reduce reasoning just to save tokens.
- Use adequate generation caps up front instead of retry ladders.
- Treat no visible output after a large cap as an integration/provider failure, not an ordinary wrong answer.
- Prefer provider pins only where route probes showed material differences.
- Avoid setting parameters that a model family documents as fixed or unsupported.

## Adapter table

| Model | Adapter recommendation | Why |
|---|---|---|
| `moonshotai/kimi-k2.6` | Use large output cap, start at `32768`; omit `temperature`/`top_p` if the router allows; keep thinking enabled. | Kimi K2.6 docs say thinking mode is native, 256K context, and show a 32K-token example. They also document fixed temperature/top-p values for K2.6/K2.5 and say other values may error. Our runs with smaller caps produced empty visible outputs. |
| `z-ai/glm-5.2` | Use large output cap, start at `32768`; keep thinking enabled/default; do not lower `reasoning_effort`. | Z.ai docs list GLM-5.2 default output cap `65536`, max `131072`, and default `reasoning_effort=max`. Our 4096 rerun still left no-answer rows. |
| `deepseek/deepseek-v4-pro` | Use large output cap, start at `32768`; keep reasoning on/default; capture visible `content` only for scoring, but log usage and empty-output failures separately. | DeepSeek reasoning-model docs say `max_tokens` includes CoT and final answer, default `32K`, max `64K`. Our low caps were too small for visible final output on some rows. |
| `minimax/minimax-m3` | Use `8192` or higher; keep `thinking=adaptive` if accepted; avoid forcing disabled thinking. | MiniMax M3 docs/model card describe thinking modes: `enabled`, `adaptive`, `disabled`. It already answered cleanly in our mixed run, so it does not need a 32K default unless no-answer appears. |
| `xiaomi/mimo-v2.5-pro` | Use large output cap, start at `32768` until we find Xiaomi-specific reasoning controls; keep default reasoning behavior. | Xiaomi docs describe MiMo V2.5 Pro as a 1T/42B-active, 1M-context agentic model. Our 4096 rerun improved some rows but still needs enough budget for reasoning-heavy behavior. |
| `nvidia/nvidia-nemotron-3-ultra-550b-a55b` | `8192` is likely sufficient for MCQ; keep provider default `baseten`. | Nemotron answered cleanly in the mixed run and scored well. NVIDIA docs describe it as a reasoning-capable 550B/55B-active text model; no special request controls found yet. |
| `google/gemma-4-31b-it` | Pin provider to `deepinfra`; `8192` is sufficient for MCQ; no special reasoning adapter needed yet. | Provider probe showed DeepInfra was the fastest reliable Gemma route. Gemma answered cleanly and scored strongly in the mixed run. |

## Provider notes from TrustedRouter catalog/probes

- `google/gemma-4-31b-it`: use `deepinfra` for now. DeepInfra, Lightning, Novita, and Tinfoil all scored 5/10 on LitQA2; DeepInfra was fastest. Avoid Parasail unless retested.
- `moonshotai/kimi-k2.6`: many provider endpoints exist. We have not yet selected the fastest reliable Kimi provider for this mixed benchmark.
- `z-ai/glm-5.2`: many provider endpoints exist. We have not yet selected the fastest reliable GLM provider for this mixed benchmark.
- `deepseek/deepseek-v4-pro`: many provider endpoints exist. We have not yet selected the fastest reliable DeepSeek provider for this mixed benchmark.
- `minimax/minimax-m3`: provider options are SiliconFlow, MiniMax, and Wafer.
- `xiaomi/mimo-v2.5-pro`: only Xiaomi provider shown in the catalog.
- `nvidia/nvidia-nemotron-3-ultra-550b-a55b`: only Baseten provider shown in the catalog.

## Harness support

Implemented in this repo:

- `trbench.client.chat()` accepts `temperature=None` and omits the `temperature` request field.
- `trbench/adapters.py` loads per-model adapter settings from JSON.
- `trbench/evals/mmlu_pro/run.py` accepts `--adapter-file`.
- `trbench/evals/lab_bench_litqa2/run.py` accepts `--adapter-file`.
- Result JSONs record the adapter file path and per-model adapter settings used.

For Kimi K2.6, official docs say K2.6/K2.5 use fixed temperature/top-p and other values may error, so the adapter omits `temperature`.

Recommended adapter fields:

```json
{
  "max_tokens": 32768,
  "timeout": 300,
  "omit_temperature": true,
  "extra_body": {},
  "provider": "deepinfra"
}
```

## Next run recommendation

Rerun the mixed 5 MMLU-Pro + 5 ProtocolQA slice only after adding adapter support.

Suggested first adapter config:

- Kimi: `max_tokens=32768`, omit temperature.
- GLM: `max_tokens=32768`.
- DeepSeek: `max_tokens=32768`.
- MiMo: `max_tokens=32768`.
- Minimax: `max_tokens=8192`.
- Nemotron: `max_tokens=8192`.
- Gemma: `max_tokens=8192`, `TRBENCH_PROVIDER=deepinfra`.
