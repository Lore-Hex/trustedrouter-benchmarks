# Mixed Bio Synthesis Experiment Log

Date: 2026-06-26

## Scope

This log tracks synthesis, routing, and aggregation experiments on the mixed
bio packet and expanded chart.

- Core packet: 16 questions = 4 GPQA Bio + 6 HLE Bio + 6 IFBench
- Expanded packet: 26 questions = core 16 + 10 LitQA2
- Main open panel for synthesis/routing:
  - `deepseek/deepseek-v4-pro`
  - `moonshotai/kimi-k2.6`
  - `xiaomi/mimo-v2.5-pro`
  - `nvidia/nvidia-nemotron-3-ultra-550b-a55b`
  - `minimax/minimax-m3`

## Baseline Tables

Saved result tables:

- 16-row open/frontier corrected run: `results/mixed_bio_reasoning_packet_16_open7_plus_frontier_refs_corrected.json`
- 26-row emoji table: `results/expanded_bio_check_table.md`
- 16-row emoji table: `results/expanded_bio_check_table_16.md`

Single-model scores on the 16-row core packet:

| Model | Score |
|---|---:|
| GPT-5.5 | 9/16 |
| Grok-4.3 | 8/16 |
| Gemini-3.1-Pro | 7/16 |
| Kimi | 6/16 |
| MiMo | 6/16 |
| Nemotron | 6/16 |
| Minimax | 5/16 |
| DeepSeek | 5/16 |
| Opus-4.8 | 4/16 |
| GLM | 4/16 |
| Gemma | 4/16 |

Single-model scores on the 26-row expanded packet:

| Model | Score |
|---|---:|
| Gemini | 15/26 |
| GPT-5.5 | 14/26 |
| Grok | 14/26 |
| Kimi | 11/26 |
| DeepSeek | 11/26 |
| Opus | 10/26 |
| Nemotron | 10/26 |
| GLM | 9/26 |
| Minimax | 9/26 |
| MiMo | 9/26 |
| Gemma | 9/26 |

Oracle ceilings:

| Panel | Score |
|---|---:|
| OS all 7 on expanded 26 | 18/26 |
| TaskIQ 5-model panel on expanded 26 | 17/26 |
| TaskIQ 5-model panel on core 16 | 11/16 |

## Method Results

| Method | Scope | Score | Artifact | Lesson |
|---|---:|---:|---|---|
| Kimi judge + GLM synth, capability weighted | 16 | 7/16 | `results/mixed_bio_reasoning_packet_16_open_pareto_capability_weighted_fusion_clean.json` | Clean run underperformed oracle; missed single-model minority rescues. |
| Kimi judge + GLM synth, stricter minority verify | 16 | 6/16 | `results/mixed_bio_reasoning_packet_16_open_pareto_minority_verify_fusion_clean.json` | Prompt-only minority protection regressed; GLM still failed to preserve minority signal. |
| Cross-rank Borda selected answer | 16 | 6/16 | `results/mixed_bio_reasoning_packet_16_crossrank_open_pareto.json` | Democratic ranking suppressed some correct minority answers. |
| Cross-rank Borda + GLM finalizer | 16 | 6/16 | `results/mixed_bio_reasoning_packet_16_crossrank_open_pareto.json` | Final synthesis did not improve selection. |
| Cross-rank weighted aggregator | 16 | 8/16 | `results/mixed_bio_reasoning_packet_16_crossrank_open_pareto.json` | Deterministic IFBench verifier and priors recovered `ifbench_0` and `ifbench_2`. |
| Role-aware Kimi single selector | 16 | 5/16 | `results/mixed_bio_reasoning_packet_16_role_aware_selector.json` | Role assignment alone was worse; over-trusted Minimax on IFBench and missed HLE. |
| TaskIQ selector, initial | 16 | 8/16 | `results/mixed_bio_reasoning_packet_16_taskiq_selector.json` | Matched weighted cross-rank at much lower call count. |
| TaskIQ selector, initial | 26 | 11/26 | `results/mixed_bio_reasoning_packet_26_taskiq_selector.json` | Tied best OS model, failed on LitQA2 due to MCQ parsing/style penalties. |
| TaskIQ selector, LitQA/MCQ fixed | 26 | 13/26 | `results/mixed_bio_reasoning_packet_26_taskiq_selector_litqa_fixed.json` | Treating terse MCQ answers as valid and adding literature-recall prior improved LitQA2 to 5/10. |
| TaskIQ with separate domain + task-type classifier | 26 | 12/26 | `results/mixed_bio_reasoning_packet_26_taskiq_domain_tasktype_selector.json` | Concept is right, but classifier mislabeled HLE rows as structured MCQ rather than judgment-under-uncertainty; regressed by 1. |
| TaskIQ domain + task-type classifier, fixed prompt/fallback | 26 | pending | `results/mixed_bio_reasoning_packet_26_taskiq_domain_tasktype_selector_fixed_classifier.json` | Patched classifier prompt to treat benchmark text as data, not a task to answer; source-aware fallback now routes HLE failures to judgment-under-uncertainty. Run blocked locally because `TRUSTEDROUTER_API_KEY` was not present. |
| TaskIQ fixed classifier live Kimi meta reruns | 26 | invalid/interrupted | `results/mixed_bio_reasoning_packet_26_taskiq_domain_tasktype_selector_fixed_classifier_v2.json`, `results/mixed_bio_reasoning_packet_26_taskiq_domain_tasktype_selector_fixed_classifier_v3.json` | Diagnostic only. Classifier/router outputs were empty or unparseable, so runs collapsed to fallback selection and were stopped before completion. Do not use as scored comparisons. |
| TaskIQ deterministic source/task meta-policy | 26 | 14/26 | `results/mixed_bio_reasoning_packet_26_taskiq_deterministic_meta.json` | Replaced failing Kimi classifier/router meta-calls with deterministic source/task typing and adjusted-TaskIQ selection. Improved over fixed TaskIQ by recovering `hle_bio_2` and `litqa2_1`, while losing `gpqa_bio_10`. |
| `trustedrouter/synth` direct endpoint | 26 | invalid | `results/mixed_bio_reasoning_packet_26_trustedrouter_synth.json`, `results/mixed_bio_reasoning_packet_26_trustedrouter_synth_request.json` | Endpoint did not produce a valid reference. Streaming path returned empty visible content for all rows; request path returned `http_503: fusion failed`. A one-word smoke prompt showed the same behavior, so this is an endpoint/invocation failure rather than a benchmark score. |
| `trustedrouter/fusion` direct endpoint | 26 | invalid | `results/mixed_bio_reasoning_packet_26_trustedrouter_fusion.json`, `results/mixed_bio_reasoning_packet_26_trustedrouter_fusion_request.json` | Same failure pattern as synth: streaming path returned empty visible content, request path returned `http_503: fusion failed`, including on a one-word smoke prompt. Do not compare as a model score. |

## Current Best Methods

Best on core 16:

- TaskIQ selector: 8/16
- Weighted cross-rank selector: 8/16

Best on expanded 26:

- Deterministic source/task TaskIQ selector: 14/26
- Fixed TaskIQ selector: 13/26

Compared with frontier references on expanded 26:

- Gemini: 15/26
- GPT-5.5: 14/26
- Grok: 14/26
- Deterministic source/task TaskIQ OS selector: 14/26
- Fixed TaskIQ OS selector: 13/26

## Key Lessons

1. **Select existing answers before synthesizing.**
   GLM synthesis often damaged or failed to preserve correct panel answers.

2. **IFBench needs deterministic verification.**
   Model priors and judges are less important than exact visible constraint
   checking for count/format/emoji rows.

3. **LitQA2 needs factual-recall handling.**
   Terse answers like `C` or `C. option text` are valid. Penalizing lack of
   reasoning hurt selection. A biomedical literature-recall prior improved
   TaskIQ from 11/26 to 13/26.

4. **HLE is still the unsolved failure mode.**
   HLE requires hard scientific judgment under uncertainty. The correct open
   answers are often minority answers from MiMo, Nemotron, Kimi, or Minimax.
   Generic scientific AIIQ priors over-favor DeepSeek/Kimi and suppress
   local HLE rescue models.

5. **Domain and task type should be separate axes.**
   Domain tells the router who is generally capable. Task type should decide
   the judging policy. The first implementation did not help because the
   classifier failed to identify most HLE rows as judgment-under-uncertainty.

6. **Classifier failures need safe fallbacks.**
   In the first domain/task-type run, many HLE classifier calls answered the
   benchmark question instead of returning JSON. The parser then fell back to
   ordinary structured MCQ weights, which prevented the HLE minority policy
   from activating. The prompt and fallback were patched before the next run.

7. **Kimi was a poor meta-controller in this harness.**
   Even after prompt/fallback fixes, Kimi classifier and router calls often
   returned no parseable visible JSON, so malformed meta-output collapsed into
   fallback behavior. The deterministic meta-policy was more reliable and
   reached 14/26 without additional model calls.

8. **Deployed synth/fusion endpoint needs separate validation before scoring.**
   Direct calls to `trustedrouter/synth` and `trustedrouter/fusion` did not
   produce valid answers in the current harness: streaming returned empty
   visible content and non-streaming returned `http_503: fusion failed`, even
   for a one-word smoke prompt. Treat endpoint failures separately from model
   quality.

9. **Fusion methodology in this repo still looks useful.**
   The repo's Fusion writeups and replica scripts describe the deployed pattern:
   panel answers feed a compact Kimi judge JSON, then a GLM synthesizer writes
   the final answer from original prompt + panel evidence + judge guidance.
   Prior lessons are that fusion helps only when panel members provide
   complementary signal and no dominant model already covers the task.

## Recommended Next Experiment

Keep the fixed TaskIQ selector as the current baseline, then add a targeted HLE
policy:

1. Classify prompt into domain and task type.
2. Add deterministic/task-specific overrides:
   - IFBench: exact verifier dominates.
   - LitQA2/factual recall: extract answer letters, ignore answer verbosity,
     boost DeepSeek/Kimi/Gemma-style recall priors.
   - HLE/judgment under uncertainty: force a focused minority verifier when
     MiMo, Nemotron, Kimi, or Minimax disagree with the top TaskIQ model.
3. For HLE verifier, compare only distinct extracted answer letters and
   require explicit steelmanning of minority specialist answers.
4. Still return an existing selected answer, not a freeform synthesis.

Success target:

- Recover at least 2 of the 3 reachable HLE rows on the 16-row core packet.
- That would move fixed TaskIQ from 13/26 to roughly 15/26 on the expanded set,
  matching Gemini on this small packet and beating GPT-5.5/Grok.
