# Synthesis model roadmap

Date: 2026-06-25

This roadmap splits future work into two synthesis products with different constraints.

## Goal 1: Frontier Synthesis

Objective: build a high-capability synthesis model using frontier closed models, aiming long term to exceed Fable 5 / Mythos 5 across broad benchmarks.

Initial target panel:

- `anthropic/claude-opus-4.8`
- `openai/gpt-5.5`
- `google/gemini-3.1-pro-preview`
- optional open-model support members when they add useful independent signal, especially `deepseek/deepseek-v4-pro`, `google/gemma-4-31b-it`, and `z-ai/glm-5.2`

Key constraints:

- Cost and latency are secondary to quality, but still need adaptive routing.
- GPT-5.5 needs careful handling because the LitQA2 pilot showed hidden-reasoning no-answer and disconnect behavior.
- Claims against Fable 5 / Mythos 5 require broad benchmark coverage, not a single benchmark slice.

Architecture direction:

1. Generate independent proposals from frontier panel members.
2. Normalize outputs into comparable claims/actions/solutions.
3. Cluster semantically equivalent candidates.
4. Run disagreement-only adjudication.
5. Protect minority evidence through debate/cross-examination when a strong model dissents.
6. Use a separate final synthesizer, not necessarily the same model as the judge.
7. Run task-native verification when available.

Initial 10-question LitQA2 translation:

- Treat the first 10 LitQA2 rows as a debugging microscope only.
- Use saved rows to identify failure modes:
  - Q3: Opus/Gemini Flash minority signal mattered.
  - Q6: Gemma/Fusion minority signal mattered.
  - Q5/Q9: no tested model solved them, so synthesis without retrieval/first-principles reasoning had no panel evidence.
- For Frontier Synthesis, rerun with `gemini-3.1-pro-preview` added, because the current pilot used Gemini 3.5 Flash rather than Gemini 3.1 Pro.
- Compare:
  - best single frontier model
  - simple majority
  - weighted vote
  - default `trustedrouter/synth`
  - disagreement-only judge/synth
  - debate-then-synth on disagreement cases

Recommended near-term benchmark ladder:

1. LitQA2 20-question pilot with frontier panel.
2. LitQA2 50-question pilot if the 20-question result shows lift.
3. Broader LAB-bench subsets: LitQA2 + DbQA + SuppQA + ProtocolQA.
4. Add non-MCQ benchmarks to test general synthesis:
   - SimpleQA Verified for factuality
   - BFCL for tool calls
   - GSM8K/AIME/math evals for reasoning
   - SciCode or LiveCodeBench for code
   - IFEval for instruction following

Success criteria:

- On small pilots: exceed best single model and default Synth without added regressions on easy consensus rows.
- On 50+ paired items: positive paired bootstrap / McNemar signal versus best single model.
- On broad suite: improve average rank or win rate versus Fable 5 / Mythos 5 baselines, once those baselines are available in the same harness.

## Goal 2: OS Pareto Synthesis

Objective: build an open-source/open-weight-only synthesis model that sits on the intelligence/cost Pareto frontier. It should beat GLM 5.2 and Kimi K2.6 while costing far less than GPT-5.5.

Initial target panel:

- `moonshotai/kimi-k2.6`
- `z-ai/glm-5.2`
- `minimax/minimax-m3`
- `deepseek/deepseek-v4-pro`
- `xiaomi/mimo-v2.5-pro`
- `nvidia/nvidia-nemotron-3-ultra-550b-a55b`
- `google/gemma-4-31b-it`

Provider notes:

- For Gemma 4, the pilot found `deepinfra`, `lightning`, `novita`, and `tinfoil` all scored 5/10; `deepinfra` was fastest.
- Avoid unstable/error-heavy routes such as Gemma/Parasail from the pilot unless retested.

Architecture direction:

1. Cheap answer-only panel first.
2. Early exit on unanimous or high-margin agreement.
3. Offline weighted/correlation-aware aggregation for low-cost tasks.
4. Invoke Kimi judge + GLM final only for disagreement cases.
5. Add minority-defense/debate only when a low-correlation or historically useful rescue model dissents.
6. Keep strict budget caps and cache all panel outputs.

Initial 10-question LitQA2 translation:

- Open-only best single: DeepSeek scored 6/10.
- Open-only simple majority: 6/10.
- Open-only oracle: 7/10.
- The key missed opportunity was Q6, where Gemma alone had the correct answer.
- Kimi judge + GLM final with a minority-preserving prompt still scored 6/10 and missed Q6.
- Therefore, the next open-source process should not merely warn "do not follow majority"; it should explicitly force a defense and critique of lone dissenters.

Recommended next experiment:

Run the open-only panel on 20 LitQA2 questions before spending more on synthesis calls.

Outputs:

- `results/lab_bench_litqa2_open7_20.json`
- `results/lab_bench_litqa2_open7_20_summary.json`
- `results/lab_bench_litqa2_open7_20_analytics.json`

Offline analytics before judge/synth:

- pairwise answer agreement
- pairwise correctness correlation
- unique-correct counts
- rescue matrix
- oracle gain from adding each model
- majority vote
- best-model tie-break vote
- correlation-penalized weighted vote

Synthesis processes to test after analytics:

1. Disagreement-only Kimi judge + GLM final.
2. Lone-dissenter defense: if a low-correlation model is alone, run a short advocate prompt for that answer before judging.
3. Top-2 debate: majority cluster versus strongest minority cluster, followed by GLM final.
4. Correlation-penalized weighted vote as a non-LLM baseline.

Success criteria:

- Beat both GLM 5.2 and Kimi K2.6 on paired benchmark rows.
- Beat simple majority and best single open model on 20-50 item pilots.
- Demonstrate better accuracy/cost than GPT-5.5 on at least one benchmark slice.
- Preserve easy consensus rows through early exit.
- Show positive marginal value from each included open model; remove redundant models that add cost but no rescue signal.

## Shared principles

- Do not overfit to the first 10 LitQA2 questions. Use them to debug process failures only.
- Always save raw panel outputs so aggregation methods can be swept offline without rerunning models.
- Evaluate synthesis methods on paired rows, not independent samples.
- Track latency, token cost, provider errors, no-answer rate, and accuracy together.
- Prefer adaptive escalation over always-on expensive debate.
- Separate judge and final synthesis roles unless a specific experiment is testing same-model collapse.
- Measure oracle upper bound before judging process quality. If the panel oracle cannot beat the best single model, synthesis has little headroom.
