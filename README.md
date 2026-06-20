# TrustedRouter Benchmarks

Cheap, hard, *model-distinguishing* capability evals, run on the leading Chinese
open-weight models through the [TrustedRouter](https://trustedrouter.com) gateway
and published with an open, reproducible harness.

> [!WARNING]
> **Running these evals can get your account banned.** They route real prompts to
> upstream labs (Anthropic, OpenAI, Google, Z.ai, Moonshot, DeepSeek, etc.). High
> request volume — and, for some evals, edge-case content — can trip provider
> usage limits and get an API key **rate-limited, suspended, or banned**. Use a
> disposable key you are willing to lose, never your production or personal one.

## Why this repo

Public leaderboards under-cover the newest Chinese flagships and almost never run
the Western-built factuality / instruction-following evals on them. The goal here
is the data nobody else publishes:

1. the **latest Chinese open-weight models** (GLM-5, Kimi K2.7/K2-Thinking,
   DeepSeek V3.2/V4, Qwen3, MiniMax M3, Hunyuan, MiMo) on a fixed harness,
2. each eval run **solo and through TrustedRouter Fusion** — does fusing the
   open models beat the best single one, and beat frontier?
3. only evals that are **cheap** (small item counts, deterministic or short
   judges, no exotic infra) yet still **unsaturated** and **discriminating**.

Companion to [PrometheusBench](https://github.com/Lore-Hex/PrometheusBench)
(refusal/permissiveness) — that one measures whether a model *will* answer; these
measure whether it *can*.

## The panel

TrustedRouter model ids, in `trbench/panel.py`. Chinese open-weight models plus a
small Western frontier reference line. Prune to what your account routes.

## Evals

Picked from a deep-research sweep for signal-per-dollar × Chinese-model
separation × ease of faithful public replication. Full rationale and the
saturated benchmarks we deliberately skip are in [EVALS.md](EVALS.md).

| Eval | Measures | Scorer | Status |
|---|---|---|---|
| **IFEval** | instruction-following | deterministic Python verifiers (no judge) | ✅ runnable |
| **GSM8K** | grade-school math reasoning | deterministic numeric match (no judge) | ✅ runnable (saturated) |
| **AIME 2025** | competition math | deterministic integer match (no judge) | ✅ runnable |
| **MATH-500** | contest math | vendored Hendrycks answer-equivalence (no judge) | ✅ runnable |
| **SimpleQA Verified** | closed-book factuality | GPT-4.1 autorater (no tools) | ✅ runnable |
| **Chinese SimpleQA** | Chinese-language factuality | LLM judge (no tools) | ✅ runnable |
| **MMLU-Pro** | broad knowledge / reasoning | deterministic letter-match (no judge) | ✅ runnable |
| **Aider polyglot** | repo-edit coding | real unit tests (no judge) | ✅ runnable (Python subset) |
| **LiveCodeBench** | contamination-resistant coding | date-windowed, containerized execution | ✅ runnable (stdin subset) |
| **tau2-bench** | agentic tool-use | upstream CLI, pass^1 (no judge) | ✅ runnable |
| **Terminal-Bench 2.0** | agentic terminal/coding | Docker harness (small-N subset) | planned |

GPQA Diamond is intentionally excluded — near-saturated at the frontier, so it
barely separates top models.

## Run IFEval

```bash
uv venv && uv pip install -e .
export TRUSTEDROUTER_API_KEY="sk-..."   # a throwaway key

# cheap smoke (first 20 prompts, a couple of models)
python -m trbench.evals.ifeval.run --models z-ai/glm-5.1,deepseek/deepseek-v4-pro \
  --prompt-limit 20 --out results/ifeval_smoke.json

# full panel (541 prompts)
python -m trbench.evals.ifeval.run --out results/ifeval.json

# score + render chart + splice into a README
python -m trbench.evals.ifeval.score results/ifeval.json \
  --svg assets/ifeval.svg --readme README.md
```

IFEval is the cheapest eval here: zero-shot, no system prompt, no judge model, no
sandbox — just 541 prompts scored by Google's deterministic verifiers.

<!-- IFEVAL_RESULTS_START -->

IFEval snapshot: `2026-06-17T19:39:21.984871+00:00` via `api.trustedrouter.com`. 100 prompts, 11 models. Deterministic Python verifiers (no judge).

![IFEval chart](assets/ifeval.svg)

| Rank | Model | IFEval | Prompt-strict | Prompt-loose | Inst-strict | Inst-loose | Errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `google/gemini-3.1-pro-preview` | 98.4 | 98.0 | 98.0 | 98.8 | 98.8 | 0 |
| 2 | `openai/gpt-5.5` | 95.2 | 94.0 | 94.0 | 96.3 | 96.3 | 0 |
| 3 | `z-ai/glm-5.2` | 93.9 | 94.0 | 94.0 | 93.9 | 93.9 | 4 |
| 4 | `anthropic/claude-opus-4.8` | 91.5 | 87.0 | 92.0 | 92.0 | 95.1 | 0 |
| 5 | `tencent/hy3-preview` | 91.2 | 87.0 | 92.0 | 91.4 | 94.5 | 0 |
| 6 | `deepseek/deepseek-v4-flash` | 50.2 | 35.0 | 50.0 | 51.5 | 64.4 | 0 |
| 7 | `xiaomi/mimo-v2.5` | 37.6 | 27.0 | 30.0 | 44.8 | 48.5 | 0 |
| 8 | `moonshotai/kimi-k2.7-code` | 36.5 | 23.0 | 34.0 | 41.1 | 47.9 | 4 |
| 9 | `deepseek/deepseek-v4-pro` | 36.2 | 24.0 | 30.0 | 42.3 | 48.5 | 2 |
| 10 | `z-ai/glm-5` | 32.1 | 22.0 | 25.0 | 39.9 | 41.7 | 3 |
| 11 | `z-ai/glm-5.1` | 32.1 | 23.0 | 23.0 | 41.1 | 41.1 | 4 |

<!-- IFEVAL_RESULTS_END -->

> **`z-ai/glm-5.2` is scored via the `tinfoil` confidential provider.** Its default
> TrustedRouter route returns empty responses (a provider bug) — at face value it
> scored near-zero and was omitted. Routed through tinfoil it answers cleanly (0
> empty) and lands at its real level (#3 here). Reproduce: `TRBENCH_PROVIDER=tinfoil`.
> The same applies to its MMLU-Pro row below.

## Run GSM8K

```bash
export TRUSTEDROUTER_API_KEY="sk-..."   # a throwaway key

# cheap smoke (first 10 problems, a couple of models)
python -m trbench.evals.gsm8k.run --models z-ai/glm-5.1,deepseek/deepseek-v4-flash \
  --prompt-limit 10 --out results/gsm8k_smoke.json

# full panel (1319 problems)
python -m trbench.evals.gsm8k.run --out results/gsm8k.json

# score + render chart + splice into a README
python -m trbench.evals.gsm8k.score results/gsm8k.json \
  --svg assets/gsm8k.svg --readme README.md
```

GSM8K is grade-school math word problems. Like IFEval it needs no judge and no
sandbox: the model is asked to end with a `#### <answer>` marker and a
deterministic numeric matcher checks exact match (falling back to the last
number in the output). The canonical 1319-problem test split is fetched from the
official OpenAI repo at runtime and cached under `.data/`.

<!-- GSM8K_RESULTS_START -->

GSM8K snapshot: `2026-06-17T23:14:26.377661+00:00` via `api.trustedrouter.com`. 30-problem subset, 12 models. Deterministic numeric match (no judge). Three models omitted (`z-ai/glm-5.2`, `moonshotai/kimi-k2.6`, `deepseek/deepseek-v3.2`) for >10% TrustedRouter route errors during the run. Note GSM8K is near-saturated at this level — the whole panel clusters at 93–100%, the open models at 100% — so it separates these models far less than the harder evals here; it's kept as a cheap deterministic sanity check, not a discriminator.

![GSM8K chart](assets/gsm8k.svg)

| Rank | Model | Accuracy | Correct | Total | Errors |
|---:|---|---:|---:|---:|---:|
| 1 | `deepseek/deepseek-v4-flash` | 100.0 | 30 | 30 | 0 |
| 2 | `deepseek/deepseek-v4-pro` | 100.0 | 30 | 30 | 0 |
| 3 | `moonshotai/kimi-k2.7-code` | 100.0 | 30 | 30 | 0 |
| 4 | `tencent/hy3-preview` | 100.0 | 30 | 30 | 0 |
| 5 | `xiaomi/mimo-v2.5` | 100.0 | 30 | 30 | 0 |
| 6 | `xiaomi/mimo-v2.5-pro` | 100.0 | 30 | 30 | 0 |
| 7 | `anthropic/claude-opus-4.8` | 96.7 | 29 | 30 | 0 |
| 8 | `google/gemini-3.1-pro-preview` | 96.7 | 29 | 30 | 0 |
| 9 | `minimax/minimax-m3` | 96.7 | 29 | 30 | 0 |
| 10 | `openai/gpt-5.5` | 96.7 | 29 | 30 | 0 |
| 11 | `z-ai/glm-5.1` | 96.7 | 29 | 30 | 1 |
| 12 | `z-ai/glm-5` | 93.3 | 28 | 30 | 2 |

<!-- GSM8K_RESULTS_END -->

GSM8K is near-saturated, so **AIME 2025** and **MATH-500** are the real math
discriminators (and they have published numbers to calibrate against).

## Run AIME / MATH-500

```bash
export TRUSTEDROUTER_API_KEY="sk-..."   # a throwaway key

# AIME 2025 — 30 competition problems, integer answers
python -m trbench.evals.aime.run --out results/aime.json
python -m trbench.evals.aime.score results/aime.json --svg assets/aime.svg --readme README.md

# MATH-500 — 500 contest problems, LaTeX answers
python -m trbench.evals.math500.run --out results/math500.json
python -m trbench.evals.math500.score results/math500.json --svg assets/math500.svg --readme README.md
```

Both are deterministic (no judge): the model reasons and puts its final answer
in `\boxed{}`. AIME checks exact integer match; MATH-500 uses the vendored
Hendrycks answer-equivalence checker (LaTeX normalization), so grading matches
published results. `--max-tokens` defaults to 16384 — these need long solutions,
and a reasoning model that truncates will look artificially weak.

<!-- AIME_RESULTS_START -->
<!-- AIME_RESULTS_END -->

<!-- MATH500_RESULTS_START -->
<!-- MATH500_RESULTS_END -->

## Methodology

- **Calibrated against published numbers.** Before trusting any new result, we
  run reference models with authoritative published scores and confirm the
  harness reproduces them (e.g. SimpleQA Verified `gemini-2.5-pro` should land on
  the published F1 of 55.6). See [CALIBRATION.md](CALIBRATION.md).
- **Faithful, not reinvented.** Each eval uses the canonical dataset and scorer.
  IFEval vendors Google's official verifiers verbatim (only the imports are made
  package-relative); see [NOTICE](NOTICE).
- **No judge where possible.** IFEval and Aider score deterministically; the
  factuality evals use a short LLM judge run with no tools.
- **Untrusted code is sandboxed.** Aider runs model-generated Python, so by
  default each test executes in a throwaway Docker container (`--network none`,
  read-only FS, caps dropped, non-root, memory/CPU/PID limits). `--sandbox host`
  falls back to the host (throwaway VM only); `--sandbox docker` requires it.
- **Reproducible: the raw run replays are published.** Every run's per-item
  responses live in `results/*.json` and are committed, so any number here can be
  re-scored and audited end to end. The one exception is Chinese SimpleQA, whose
  replay embeds a dataset that ships no license; only its aggregate scores are
  published. Raw dataset caches (`.data/`) stay out of git.

## License

Apache-2.0. Vendored IFEval verifiers are Apache-2.0 from
[google-research](https://github.com/google-research/google-research/tree/master/instruction_following_eval).

<!-- AIDER_POLYGLOT_RESULTS_START -->

Aider polyglot (Python subset) snapshot: `2026-06-17T13:21:24.049770+00:00`. 34 Exercism exercises, pass@1, real unit tests (no judge).

![Aider polyglot chart](assets/aider_polyglot.svg)

| Rank | Model | Pass% | Passed | Total | Errors |
|---:|---|---:|---:|---:|---:|
| 1 | `anthropic/claude-opus-4.8` | 88.2 | 30 | 34 | 0 |
| 2 | `tencent/hy3-preview` | 41.2 | 14 | 34 | 0 |
| 3 | `deepseek/deepseek-v3.2` | 38.2 | 13 | 34 | 0 |
| 4 | `moonshotai/kimi-k2.7-code` | 32.4 | 11 | 34 | 0 |
| 5 | `xiaomi/mimo-v2.5-pro` | 26.5 | 9 | 34 | 0 |
| 6 | `deepseek/deepseek-v4-flash` | 23.5 | 8 | 34 | 0 |
| 7 | `deepseek/deepseek-v4-pro` | 20.6 | 7 | 34 | 1 |
| 8 | `z-ai/glm-5` | 20.6 | 7 | 34 | 1 |
| 9 | `minimax/minimax-m3` | 14.7 | 5 | 34 | 0 |
| 10 | `moonshotai/kimi-k2.6` | 14.7 | 5 | 34 | 3 |
| 11 | `z-ai/glm-5.1` | 11.8 | 4 | 34 | 7 |
| 12 | `z-ai/glm-5.2` | 2.9 | 1 | 34 | 18 |
| 13 | `xiaomi/mimo-v2.5` | 0.0 | 0 | 34 | 0 |

<!-- AIDER_POLYGLOT_RESULTS_END -->

<!-- SIMPLEQA_VERIFIED_RESULTS_START -->

SimpleQA Verified snapshot: `2026-06-17T13:23:05.762381+00:00`. 250 closed-book questions, no tools. Judge: `openai/gpt-4.1`. F-score = harmonic mean of accuracy and accuracy-given-attempted.

![SimpleQA Verified chart](assets/simpleqa_verified.svg)

| Rank | Model | F-score | Correct% | Attempted% | Acc|attempted | Empty | Errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `deepseek/deepseek-v4-pro` | 55.1 | 52.7 | 91.3 | 57.7 | 0 | 9 |
| 2 | `anthropic/claude-opus-4.8` | 53.1 | 40.8 | 53.6 | 76.1 | 0 | 0 |
| 3 | `z-ai/glm-5.1` | 51.6 | 45.2 | 75.2 | 60.1 | 0 | 0 |
| 4 | `moonshotai/kimi-k2.6` | 49.7 | 43.2 | 74.0 | 58.4 | 0 | 0 |
| 5 | `z-ai/glm-5.2` | 48.3 | 40.0 | 65.6 | 61.0 | 8 | 0 |
| 6 | `z-ai/glm-5` | 46.5 | 44.8 | 92.8 | 48.3 | 0 | 0 |
| 7 | `moonshotai/kimi-k2.7-code` | 41.1 | 28.6 | 39.4 | 72.6 | 0 | 9 |
| 8 | `deepseek/deepseek-v4-flash` | 37.2 | 32.9 | 76.8 | 42.9 | 0 | 4 |
| 9 | `xiaomi/mimo-v2.5-pro` | 33.6 | 29.2 | 73.6 | 39.7 | 0 | 0 |
| 10 | `tencent/hy3-preview` | 28.0 | 27.6 | 96.8 | 28.5 | 0 | 0 |
| 11 | `deepseek/deepseek-v3.2` | 25.9 | 25.6 | 97.6 | 26.2 | 0 | 4 |
| 12 | `minimax/minimax-m3` | 24.1 | 17.6 | 46.0 | 38.3 | 0 | 0 |
| 13 | `xiaomi/mimo-v2.5` | 20.9 | 19.6 | 87.6 | 22.4 | 0 | 0 |

<!-- SIMPLEQA_VERIFIED_RESULTS_END -->

> **Grading & budget.** Graded with Google's exact autorater — `openai/gpt-4.1`
> plus the published *modified* SimpleQA Verified grader prompt (direct-answer +
> numeric acceptable-range rules). Generation budget is 32768 tokens: the verbose
> reasoning models (the GLM family, Kimi K2.6) spend >8192 tokens thinking on hard
> questions and would otherwise truncate before the answer — re-running them at the
> higher budget moved glm-5.1 29.8→49.7 and kimi-k2.6 31.0→49.2. The **Empty**
> column counts answers that came back blank (truncated/no committed answer): they
> grade NOT_ATTEMPTED, so a high count means a budget/generation problem, not
> knowledge. `z-ai/glm-5.2` is pathological here — it returns empty on 231/250 even
> at 32768 (runaway reasoning that never commits), so its 6.1 reflects that, not its
> factual knowledge. See [CALIBRATION.md](CALIBRATION.md).

<!-- CHINESE_SIMPLEQA_RESULTS_START -->

Chinese SimpleQA snapshot: `2026-06-17T13:26:37.168268+00:00`. 250 closed-book Chinese questions, no tools. Judge: `google/gemini-2.5-flash`.

![Chinese SimpleQA chart](assets/chinese_simpleqa.svg)

| Rank | Model | F-score | Correct% | Attempted% | Acc|attempted | Errors |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `deepseek/deepseek-v4-pro` | 75.9 | 73.9 | 94.8 | 78.0 | 1 |
| 2 | `deepseek/deepseek-v3.2` | 72.6 | 71.8 | 98.0 | 73.3 | 5 |
| 3 | `deepseek/deepseek-v4-flash` | 72.4 | 68.8 | 90.0 | 76.4 | 0 |
| 4 | `moonshotai/kimi-k2.6` | 71.6 | 62.0 | 73.2 | 84.7 | 0 |
| 5 | `anthropic/claude-opus-4.8` | 71.3 | 61.8 | 73.5 | 84.2 | 1 |
| 6 | `moonshotai/kimi-k2.7-code` | 71.3 | 57.7 | 61.9 | 93.2 | 11 |
| 7 | `z-ai/glm-5.2` | 70.4 | 59.1 | 67.8 | 87.1 | 101 |
| 8 | `xiaomi/mimo-v2.5-pro` | 68.3 | 65.2 | 90.8 | 71.8 | 0 |
| 9 | `z-ai/glm-5` | 67.1 | 58.5 | 74.2 | 78.8 | 2 |
| 10 | `tencent/hy3-preview` | 66.5 | 65.2 | 96.0 | 67.9 | 0 |
| 11 | `xiaomi/mimo-v2.5` | 62.3 | 59.2 | 90.0 | 65.8 | 0 |
| 12 | `minimax/minimax-m3` | 59.7 | 54.0 | 80.8 | 66.8 | 0 |
| 13 | `z-ai/glm-5.1` | 59.7 | 46.0 | 54.0 | 85.2 | 0 |

<!-- CHINESE_SIMPLEQA_RESULTS_END -->

<!-- TAU2_RESULTS_START -->

tau2-bench snapshot: `2026-06-19T02:23:54.885426+00:00`. Domain `retail`, 20 tasks x 1 trial(s), agent vs fixed user `openai/gpt-4.1`. Metric: pass^1 (task reward == 1). Small-subset numbers run high; the ranking is the signal.

![tau2-bench chart](assets/tau2.svg)

| Rank | Model | pass^1 | avg_reward | Tasks | Errors |
|---:|---|---:|---:|---:|---:|
| 1 | `deepseek/deepseek-v4-flash` | 100.0 | 100.0 | 20 | 0 |
| 2 | `openai/gpt-5.5` | 100.0 | 100.0 | 20 | 0 |
| 3 | `deepseek/deepseek-v4-pro` | 95.0 | 95.0 | 20 | 0 |
| 4 | `minimax/minimax-m3` | 95.0 | 95.0 | 20 | 0 |
| 5 | `anthropic/claude-opus-4.8` | 94.1 | 94.1 | 17 | 3 |
| 6 | `google/gemini-3.1-pro-preview` | 90.0 | 90.0 | 20 | 0 |
| 7 | `moonshotai/kimi-k2.6` | 90.0 | 90.0 | 20 | 0 |
| 8 | `tencent/hy3-preview` | 90.0 | 90.0 | 20 | 0 |
| 9 | `z-ai/glm-5.2` | 90.0 | 90.0 | 20 | 0 |
| 10 | `z-ai/glm-5` | 85.0 | 85.0 | 20 | 0 |
| 11 | `deepseek/deepseek-v3.2` | 82.4 | 82.4 | 17 | 3 |
| 12 | `moonshotai/kimi-k2.7-code` | 80.0 | 80.0 | 20 | 0 |
| 13 | `xiaomi/mimo-v2.5-pro` | 80.0 | 80.0 | 20 | 0 |
| 14 | `z-ai/glm-5.1` | 80.0 | 80.0 | 20 | 0 |
| 15 | `xiaomi/mimo-v2.5` | 70.0 | 70.0 | 20 | 0 |

<!-- TAU2_RESULTS_END -->

<!-- MMLU_PRO_RESULTS_START -->

MMLU-Pro snapshot: `2026-06-19T04:56:04.874288+00:00` via `api.trustedrouter.com`. 150 questions, 15 models. Zero-shot CoT, deterministic letter-match (no judge).

![MMLU-Pro chart](assets/mmlu_pro.svg)

| Rank | Model | Accuracy | Correct | Total | No answer | Errors |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `anthropic/claude-opus-4.8` | 88.0 | 132 | 150 | 0 | 0 |
| 2 | `moonshotai/kimi-k2.6` | 87.3 | 131 | 150 | 0 | 0 |
| 3 | `moonshotai/kimi-k2.7-code` | 86.0 | 129 | 150 | 0 | 0 |
| 4 | `openai/gpt-5.5` | 85.3 | 128 | 150 | 0 | 1 |
| 5 | `deepseek/deepseek-v3.2` | 84.0 | 126 | 150 | 1 | 0 |
| 6 | `deepseek/deepseek-v4-flash` | 84.0 | 126 | 150 | 0 | 0 |
| 7 | `minimax/minimax-m3` | 84.0 | 126 | 150 | 0 | 0 |
| 8 | `deepseek/deepseek-v4-pro` | 83.3 | 125 | 150 | 2 | 0 |
| 9 | `xiaomi/mimo-v2.5` | 83.3 | 125 | 150 | 0 | 0 |
| 10 | `google/gemini-3.1-pro-preview` | 83.3 | 125 | 150 | 0 | 4 |
| 11 | `xiaomi/mimo-v2.5-pro` | 82.0 | 123 | 150 | 1 | 0 |
| 12 | `z-ai/glm-5` | 82.0 | 123 | 150 | 2 | 0 |
| 13 | `z-ai/glm-5.2` | 82.0 | 123 | 150 | 0 | 14 |
| 14 | `z-ai/glm-5.1` | 80.7 | 121 | 150 | 5 | 0 |
| 15 | `tencent/hy3-preview` | 76.0 | 114 | 150 | 5 | 0 |

<!-- MMLU_PRO_RESULTS_END -->

<!-- BEAM_128K_RESULTS_START -->

*BEAM 128K results coming soon.*

<!-- BEAM_128K_RESULTS_END -->

<!-- TERMINAL_BENCH_RESULTS_START -->

*Terminal-Bench results coming soon.*

<!-- TERMINAL_BENCH_RESULTS_END -->
