# LAB-bench LitQA2 synthesis experiment log

Date: 2026-06-25

## Scope

We used the first 10 LitQA2 LAB-bench multiple-choice questions as a cheap microscope for model complementarity and synthesis behavior. This is a smoke/pilot slice only; it is not enough for model-ranking confidence or SOTA claims.

## Saved artifacts

- Original 6-model run: `results/lab_bench_litqa2_smoke10_6models_routed.json`
- Original 6-model summary: `results/lab_bench_litqa2_smoke10_6models_routed_summary.json`
- Added Gemma/Opus/GPT-5.5 provider summary: `results/lab_bench_litqa2_added3_provider_summary.json`
- Added Gemini/Fusion summary: `results/lab_bench_litqa2_added_gemini_fusion_summary.json`
- Custom Synth raw run: `results/lab_bench_litqa2_custom_synth_deepseek_opus_gemma_gemini_kimi_10.json`
- Custom Synth summary: `results/lab_bench_litqa2_custom_synth_summary.json`
- Open-only minority judge/synth process: `results/lab_bench_litqa2_open_minority_judge_kimi_synth_glm_10.json`
- Provider probes: `results/gemma4_provider_probe.json`, `results/opus48_provider_probe.json`, `results/gpt55_provider_probe.json`, `results/gemini35flash_provider_probe.json`

## Model results on the 10-question slice

| Model / method | Score | Notes |
|---|---:|---|
| DeepSeek V4 Pro | 6/10 | Best single open model in initial run |
| Opus 4.8 | 6/10 | Only selected model, with Gemini Flash, to solve Q3 |
| TrustedRouter Fusion/Synth default | 6/10 | One error row; solved Q6 but timed out on Q3 |
| Custom Synth: DeepSeek+Opus+Gemma+Gemini+Kimi, Opus judge/final | 6/10 | Fixed Q3, lost Q6 and Q7 |
| Open-only minority judge/synth: Kimi judge, GLM final | 6/10 | Same as majority/best single; still missed Q6 |
| Kimi K2.6 | 5/10 | Same pattern as GLM on this slice |
| GLM 5.2 | 5/10 | Blog-favored synthesizer, but not tested as final inside explicit Synth yet except open-only process |
| Gemma 4 31B | 5/10 | Crucial unique Q6 rescue signal |
| Gemini 3.5 Flash | 5/10 | Correct on Q3 and Q7 |
| GPT-5.5 | 5/10 | 4k rerun; one disconnect and one hidden-reasoning no-answer |
| Minimax M3 | 4/10 | No unique wins in this slice |
| Nemotron 3 Ultra | 4/10 | No unique wins in this slice |
| MiMo 2.5 Pro | 3/10 | No unique wins in this slice |

## Complementarity observations

- Full selected panel oracle reached 8/10: Q5 and Q9 were unsolved by every tested model.
- Open-only panel oracle reached 7/10: only Gemma solved Q6; no open model solved Q3, Q5, or Q9.
- Majority vote was not enough. It scored 6/10 on the open-only panel and 6/10 when replacing Gemma with GPT-5.5.
- Gemma is valuable despite only scoring 5/10 because it contributed the lone correct Q6 signal.
- Opus and Gemini Flash were important for Q3 in the broader panel.
- GPT-5.5 did not add enough marginal value on this slice to justify its route/runtime instability for the current cost-aware open-panel experiments.

## Synthesis lessons

- The panel often contains enough signal, but synthesis can fail to preserve minority correct answers.
- A prompt that merely says "do not choose by majority vote" was not sufficient. Kimi judge + GLM final still ignored Gemma's lone correct Q6 answer.
- Custom Synth with Opus as both judge and final synthesizer matched the best single model but did not beat it. This supports the TrustedRouter blog's "fusion is two jobs" warning: judging and final synthesis should be separated.
- `trustedrouter/synth` and `trustedrouter/fusion` are catalog aliases for the same route. Prefer `trustedrouter/synth` going forward.
- The public catalog does not disclose default Synth panel members; reproducible experiments should pass explicit `extra_body.plugins` configuration.

## Code change

`trbench/evals/lab_bench_litqa2/run.py` now supports `--extra-body`, matching the BFCL runner. This lets us run explicit Synth/Fusion plugin configs reproducibly.

## Recommended next experiment

Do not spend more on elaborate synthesis with only 10 questions. Next, expand the open-only panel to 20 LitQA2 questions, then run offline analytics before any more judge/synth calls.

Update: after the mixed 5-question MMLU-Pro + 5-question ProtocolQA run, use larger output budgets for all future MCQ smoke runs. `max_tokens=256` produced many empty visible outputs because reasoning-heavy models spent the whole budget internally. Even `4096` left some Kimi/GLM/DeepSeek no-answer rows on the mixed run, so future runs should start at `8192` for reasoning-heavy open models unless model-specific reasoning controls are added.

### Step 1: Run/open-panel generation on 20 questions

Panel:

- `moonshotai/kimi-k2.6`
- `z-ai/glm-5.2`
- `minimax/minimax-m3`
- `deepseek/deepseek-v4-pro`
- `xiaomi/mimo-v2.5-pro`
- `nvidia/nvidia-nemotron-3-ultra-550b-a55b`
- `google/gemma-4-31b-it` pinned to the fastest reliable provider from probes, `deepinfra`

Suggested output:

- `results/lab_bench_litqa2_open7_20.json`
- `results/lab_bench_litqa2_open7_20_summary.json`

### Step 2: Compute offline analytics

Before judge/synth calls, compute:

- pairwise answer agreement
- pairwise correctness correlation
- unique-correct counts
- rescue matrix: when model A is wrong, how often model B is right
- oracle gain from adding each model
- majority vote, best-model tie-break vote, and correlation-penalized weighted vote

### Step 3: Only adjudicate disagreement cases

Use Kimi judge + GLM final as the first open-only process because it matches the TrustedRouter blog prior, but add a stronger minority-defense/debate step for cases where:

- a low-correlation model is the lone dissenter
- a historically useful rescue model dissents
- the majority margin is weak

Target: beat 6/10-like best-single behavior and recover oracle-style minority cases without adding new losses.

## Next mixed benchmark packet

Use a 10-task mixed packet to test whether the synthesis process generalizes
beyond LitQA2-style multiple choice:

- 3 GPQA Diamond Biology MCQ tasks
- 3 HLE Bio/Chem Gold Biology MCQ tasks
- 2 IFBench precise instruction-following tasks
- 2 AA-LCR long-context reasoning tasks

Implementation:

- Runner: `trbench.evals.mixed_bio_packet.run`
- Manifest: `results/mixed_bio_reasoning_packet_manifest.json`
- Default panel: open seven model panel from the adapter file
- Adapter file: `results/open_panel_adapter_recommendations.json`

Exact command:

```bash
uv run python -m trbench.evals.mixed_bio_packet.run \
  --adapter-file results/open_panel_adapter_recommendations.json \
  --out results/mixed_bio_reasoning_packet_open7.json
```

Scoring note: GPQA, HLE, and the selected IFBench rows have deterministic
scoring. AA-LCR rows include the extracted document text in the prompt, but
should be graded by a semantic judge pass rather than exact string matching.

### Mixed packet run status

AA-LCR was left out for the first pass because even the cheapest selected rows
were about 72k input tokens each and require semantic judging. The current
deterministic packet is therefore 8 tasks: 3 GPQA Bio, 3 HLE Bio, and 2 IFBench.

Saved files:

- Manifest: `results/mixed_bio_reasoning_packet_no_lcr_manifest.json`
- Short-packet adapter: `results/open_panel_adapter_short_packet.json`
- Six-model run excluding Kimi: `results/mixed_bio_reasoning_packet_no_lcr_open6_no_kimi.json`

Six-model results on the 8 scored tasks:

| Model | Score |
|---|---:|
| Minimax M3 | 5/8 |
| GLM 5.2 | 3/8 |
| Nemotron 3 Ultra | 3/8 |
| Gemma 4 31B | 2/8 |
| MiMo 2.5 Pro | 2/8 |
| DeepSeek V4 Pro | 1/8 |

Kimi note: Kimi stalled on this packet even after excluding AA-LCR, lowering the
short-packet cap to 8192, using `timeout=180`, omitting temperature, and setting
`--retries 0`. Treat Kimi as a separate provider/adapter probe before adding it
back to packet runs. Next Kimi probe should run one task at a time with
`concurrency=1`, `max_tokens=4096`, `timeout=120`, `--retries 0`, and provider
pinning if viable routes are available.

### Wrapper/provider troubleshooting update

Validated short-packet adapter:

- `results/open_panel_adapter_validated_short_packet.json`

Provider/request-path findings:

- Kimi default route stalled. `baseten` + direct POST completed all 12 tasks with
  no no-answer rows.
- GLM default route exhausted the visible output budget in hidden reasoning
  (`finish=length`, empty text). `wafer` + direct POST completed all 12 tasks
  with no no-answer rows.
- DeepSeek default route produced no-answer rows. On the failed rows, `tinfoil`
  + streaming returned visible answers for all probes and solved 2/3.
- MiMo has only the Xiaomi route. Raising max output to 16384 on streaming
  recovered several no-answer rows, including a unique `hle_bio_0` rescue. Direct
  POST is inconsistent and slow for MiMo.

Probe-adjusted open7 result:

- `results/mixed_bio_reasoning_packet_12_open7_validated_probe_adjusted.json`

Probe-adjusted scores:

| Model | Score | Notes |
|---|---:|---|
| MiMo V2.5 Pro | 6/12 | Slow, but recovered unique `hle_bio_0` and `ifbench_2`; keep for now |
| Minimax M3 | 5/12 | Reliable, no no-answer rows |
| Nemotron 3 Ultra | 5/12 | Reliable, unique `hle_bio_6` |
| DeepSeek V4 Pro | 4/12 | Better with Tinfoil route; unique `ifbench_0` |
| Kimi K2.6 | 4/12 | Baseten route reliable |
| Gemma 4 31B | 3/12 | Reliable but no unique wins on this packet |
| GLM 5.2 | 3/12 | Wafer route reliable; lower score than default no-answer-contaminated run |

MiMo should not be dropped yet: after wrapper fixes, the open-panel oracle is
9/12 with MiMo and 7/12 without MiMo. It is expensive/slow, so use it
selectively or cache its outputs, but it is currently additive.
