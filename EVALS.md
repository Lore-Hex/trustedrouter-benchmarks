# Eval roadmap

Selected from a deep-research sweep (2025–26 snapshots) ranked by **signal-per-dollar
× how cleanly it separates the Chinese open-weight models × ease of faithful public
replication**. Numbers are leaderboard snapshots and move fast — re-pull before
publishing. No per-run dollar cost was independently measured; "cheap" rests on small
item counts, deterministic or short judges, and no exotic infra.

## Shipped

### IFEval — instruction-following ✅
- **What:** 541 prompts, 25 verifiable instruction types; strict + loose × prompt + instruction accuracy.
- **Why first:** the cheapest useful eval here — zero-shot, no system prompt, **no judge model, no sandbox**. Scoring is pure-Python and deterministic.
- **Harness:** Google's official verifiers, vendored verbatim under `trbench/evals/ifeval/vendor/` (Apache-2.0). Runner + scorer in this repo.
- **Distinguishing:** instruction-following separates models well; the published panels are Western, so a clean **Chinese-panel** table is new.
- **Cost:** ~541 short completions/model. Cheap.

## Planned (wrap the canonical harness — do not reinvent the scorer)

### Aider polyglot — repo-edit coding
- **What:** 225 fixed Exercism exercises across C++, Go, Java, JS, Python, Rust; scored by **real unit tests** (no judge).
- **Distinguishing:** ~34-pt spread across DeepSeek/Qwen/Kimi families (Nov-2025). Newest gen (GLM-5, Kimi K2-Thinking, DeepSeek V4, Qwen3-Max) not yet on the public board → running them is the novel part.
- **Harness:** aider's own benchmark harness pointed at the TR base_url. Needs the six language toolchains (Docker).
- **Caveat:** part of the published spread reflects model *size*; unproven for the very latest gen.

### Terminal-Bench 2.0 — agentic terminal/coding
- **What:** Terminus 2 harness; the widest discriminating range in the field (SOTA 84.7% → floor 3.1%).
- **Distinguishing:** **best per-model Chinese ranking** on a fixed harness — GLM-5 52.4 > Kimi K2.5 43.2 > DeepSeek-V3.2 39.6 > Kimi-K2-Thinking 35.7 > GLM-4.7 33.4 > … > Qwen3-Coder 23.9. (GLM-4.6 ≈ Qwen3-Coder are a tie.)
- **Cost:** agentic + Docker sandbox + multi-turn → **expensive on a full run; use a small-N task subset** to stay cheap.

### SimpleQA Verified — closed-book factuality
- **What:** 1,000 prompts (Google DeepMind), **run with no tools** (search → near-100%). Unsaturated: Gemini 2.5 Pro F1 55.6 → 11.1.
- **Distinguishing:** the published panel is almost entirely Western (only DeepSeek R1) → **re-running the full Chinese panel is genuinely new data.**
- **Harness:** dataset on HuggingFace (`google/simpleqa-verified`) + short LLM judge.

### Chinese SimpleQA — Chinese-language factuality
- **What:** 3,000 short Chinese Q&A, 6 topics / 99 subtopics; tool-free, LLM-judged.
- **⚠️ License:** the "MIT / forked from simple-evals" claim was **refuted in research** — verify the actual license and scorer before republishing; may need a clean-room scorer. Run a subset to stay cheap.

### tau2-bench — agentic tool-use
- **What:** dual-control conversational agents across 5 domains; **MIT**, canonical CLI (`tau2 run`).
- **Cost:** keep cheap with `--num-tasks` (e.g. 5–20/domain). Agentic multi-turn token overhead.

### LiveCodeBench — contamination-proof coding
- **What:** competition problems with release dates; `--start_date`/`--end_date` filtering to score only post-cutoff problems. Good hygiene complement to the static coding evals.

## Deliberately skipped

- **GPQA Diamond** — cheap (198 Q) but **near-saturated/compressed** at the frontier (Chinese models cluster 2–4 pts under Western leaders) → almost no model-distinguishing signal.

## Open questions (carry into each run)

1. Actual measured per-run **$ cost** on small-N subsets, especially the agentic evals.
2. **Chinese SimpleQA's real license** + canonical scorer.
3. Do the Western-built factuality/IF evals (SimpleQA Verified, IFEval) **retain discriminating power on the Chinese panel** once re-run?
4. A cheap, openly-licensed **agentic Chinese-language** eval to pair with Chinese SimpleQA (gap — none verified).
