# Calibration — does the harness reproduce published numbers?

A benchmark harness is only trustworthy if it reproduces the published
gold-standard numbers for known models. Before trusting any *new* number (the
Chinese-panel results), we run reference models that have authoritative published
scores and check that our harness lands on them — exactly or nearly exactly.

The cleanest anchor is **SimpleQA Verified**: Google published an exact per-model
table (no tools, single dataset), and several of those models route on
TrustedRouter. If our `gemini-2.5-pro` run lands on 55.6 F1, the whole
answer → no-tools → LLM-judge → F-score pipeline is validated. To grade the way
Google does, we use Google's **exact autorater**: `openai/gpt-4.1` with the
SimpleQA Verified grader prompt — OpenAI's original SimpleQA rubric, modified by
Google to force direct answers, penalize guessing in long responses, and honor
per-target numeric *acceptable ranges* (see `trbench/judge.py`; the modifications
are the three the dataset card documents).

## SimpleQA Verified — published vs ours (full 1000, no tools)

Published = Table 7 of the SimpleQA Verified paper (arXiv 2509.07968) / Epoch AI.

> **The calibration earned its keep — it caught a critical bug on the first run.**
> Our first `gemini-2.5-pro` run scored **F1 31.6 with Attempted 44.9%** against
> a published **55.6 / 98.9%**. The accuracy-given-attempted was close (51 vs 56),
> but the *attempted rate* was half what it should be — the harness was recording
> half of gemini-2.5-pro's answers as "not attempted." The answers turned out to
> be **truncated mid-sentence** (one literally cut at `...is **Herz[berg]`, the
> correct answer chopped). Cause: gemini-2.5-pro is a reasoning model, and a low
> `max_tokens` (512) gets eaten by hidden thinking, leaving no room for the
> visible answer. Raising `max_tokens` to 8192 fixed it. **Every panel run at the
> old budget was contaminated for reasoning models** — all re-run at 8192.

`google/gemini-2.5-pro`, full 1000, no tools:

| Run | F1 / Acc / Att |
|---|---:|
| **Published** (Table 7, arXiv 2509.07968 / Epoch AI) | **55.6 / 55.3 / 98.9** |
| Ours @512 max_tokens (broken: thinking ate the answer) | 31.6 / — / 44.9 |
| Ours @8192, judge `gemini-2.5-flash` + original SimpleQA prompt | 51.3 / 51.1 / 99.4 |
| Ours @8192, **Google's autorater** (`gpt-4.1` + modified prompt) | **53.5 / 52.6 / 96.7** |

**Verdict: validated.** Three findings pin it down:

- **The autorater was the gap, as suspected.** Switching from the cheap default
  judge to Google's exact autorater (`gpt-4.1` + the modified grader prompt) moved
  full-1000 F1 from 51.3 to **53.5** — over half the remaining ~4-point gap. The
  single biggest driver is the numeric **acceptable-range** rule: 88 of the 1000
  gold answers ship a range like `150 (acceptable range: anything between 148 and
  152)`, and the original prompt's "correct to the last significant figure" marked
  near-misses wrong, whereas Google's prompt accepts anything inside the range.
- **On a comparable subset we're at/above Google.** The dataset isn't shuffled
  (later questions are harder), so on the easier first-300 the *same* autorater
  scores **F1 57.3 / Acc 57.0** — at or above the published full-1000 of 55.6. The
  grader is correctly calibrated, not lenient.
- **The residual ~2 F1 on the full 1000 is generation-side, not grading.** We ruled
  out the obvious suspects: thinking is already on for gemini-2.5-pro by default
  (verified — 724 reasoning tokens on a default call; it can't be disabled for pro,
  `max_tokens:0` returns empty), and a short-answer query template raised Attempted
  to 100% but did **not** raise F1 (56.7 vs 57.3 on the first-300 — neutral, so not
  adopted; raw-question matches original SimpleQA anyway). What's left is back-tail
  answer variance between our gemini-2.5-pro run and Google's, well within normal
  LLM-autorater reproduction tolerance.

Judge notes: the SimpleQA Verified default judge is now `openai/gpt-4.1` (Google's
published autorater); like `gemini-2.5-flash` it's non-reasoning and emits the
A/B/C letter immediately, so grading stays cheap and fast. A *reasoning* model as
judge (e.g. `gemini-2.5-pro`) hits the truncation trap on the judge side, so
`judge.grade` keeps `max_tokens=2048` as a guard. Every run replay is published in
`results/`, so any of these numbers can be re-graded with `--judge-model`.

Full published table (reference): Gemini 2.5 Pro 55.6, GPT-5 52.3, o3 51.9,
GPT-4.1 39.9, GPT-4o 34.9, DeepSeek R1 33.3, Claude Opus 4 28.3, Gemini 2.5 Flash
28.2, GPT-5 Mini 24.6, o4-mini 23.4, Claude Sonnet 4 18.7, GPT-5 Nano 14.4,
Gemini 2.5 Flash-Lite 11.1.

### Panel gotcha: 8192 tokens still truncates the most verbose reasoners

The gemini-2.5-pro fix (max_tokens 512→8192) wasn't enough for everyone. On the
Chinese panel, the GLM family and Kimi K2.6 spend *more* than 8192 tokens thinking
on hard SimpleQA questions and truncate before emitting a committed answer — the
harness then sees blank text and grades NOT_ATTEMPTED, crushing their score. The
tell was the new **Empty** column (blank-but-no-error responses):

| Model | @8192 (truncated) | @32768 (re-run) | Empty @8192 → @32768 |
|---|---:|---:|---:|
| `z-ai/glm-5.1` | F1 29.8 | **49.7** | 97 → 0 |
| `moonshotai/kimi-k2.6` | F1 31.0 | **49.2** | 87 → 0 |
| `z-ai/glm-5` | F1 38.9 | **46.1** | (reasoning-leak) → 0 |
| `z-ai/glm-5.2` | F1 ~5 | 6.1 | 228 → **231** |

So the closed-book-QA default is now **32768** (free for non-reasoners — they stop
at natural length). `glm-5.2` is the exception: it stays empty on 231/250 even at
32768 (runaway reasoning that never commits to an answer) — that's genuine model
behavior on this task, not a harness limit, and the Empty column makes it legible.
**Likely affects other evals too** (the Aider polyglot panel shows the same GLM/Kimi
models low with errors) — worth a budget audit across the suite.

## IFEval — published vs ours (full 541, deterministic, no judge)

IFEval has no judge, so a faithful harness should match published numbers
closely; the only wrinkle is *which* metric a source reports (prompt-strict vs
the average of the four). We report all four sub-metrics, so we can line up
against whichever a source used. Reference models run: `openai/gpt-4o-mini`,
`google/gemini-2.5-flash`, `google/gemini-2.5-pro`.

| Model | Published IFEval | Ours (prompt-strict / avg) | Verdict |
|---|---|---|---|
| `openai/gpt-4o-mini` | ~86 (M-IFEval EN strict, arXiv 2502.04688) | 79.7 / **84.4** | **within ~2pt — validated** |
| `google/gemini-2.5-flash` | none official | 82.3 / 86.4 | Google publishes no IFEval for the 2.5 family |
| `google/gemini-2.5-pro` | none official | provider-flaky run (55–63 with 171 errors) | inconclusive; no official target anyway |

**Verdict: validated via gpt-4o-mini.** No lab publishes an official IFEval for the
Gemini 2.5 family (their cards report GPQA / AIME 2025 / LiveCodeBench / SWE-Bench /
SimpleQA, not IFEval), so the cleanest anchor is `gpt-4o-mini` against the
peer-reviewed M-IFEval re-run of the original IFEval (~86 instruction-level avg
strict). Our 84.4 lands within ~2 points, and the verifiers themselves are
Google's canonical code, so the IFEval pipeline is sound. (Third-party
aggregators for gpt-4o-mini conflict — 84 / 86 / 76.5 depending on the exact
sub-metric — so treat ~84–86 as the band, not a single point.)

## AIME 2025 — the reasoning-budget gap (does NOT cleanly match)

AIME is where the gateway path and the labs' published numbers diverge, and it's
worth being blunt about it. Google's official Gemini 2.5 Flash model card reports
AIME 2025 pass@1 = **72.0** (GA "thinking"; 75.6 for the 09-2025 preview), and a
second source corroborates 72. Our `gemini-2.5-flash` through the TrustedRouter
chat endpoint lands **50.0 / 56.7 / 53.3** at `max_tokens` 16k / 32k / 64k. Three
real effects, none of them a scoring bug:

- **Truncation at low budgets.** Flash reasons 50k–100k+ characters on hard AIME
  problems. At 16k tokens, 7/30 responses are cut off mid-reasoning before
  reaching `\boxed`; at 32k, 5/30. Raising the budget recovers some.
- **A reasoning ceiling above ~32k, not just our cap.** At a 64k request the
  longest responses came back *shorter* (max ~36k chars) than at 32k (~101k
  chars), and 5 problems still never reached an answer — so past ~32k the extra
  budget isn't used. The published 72 uses Gemini's *native thinking budget*,
  which the OpenAI-compatible chat path does not expose.
- **Run-to-run nondeterminism.** Even at temperature 0 the score moved
  50 → 57 → 53 (±2 problems on a 30-problem set).

**Root cause, and the fix — now RESOLVED (matches published).** The plain chat
endpoint scored low because the TrustedRouter gateway defaulted Gemini 2.5 Flash
to `thinkingBudget: 0` (thinking off) and exposed only effort *levels*, never the
native token budget — so requests ran Flash with reasoning disabled. The gateway
now exposes it (shipped 2026-06-18): an OpenRouter-style `reasoning.max_tokens`
maps to Gemini's native `thinkingConfig.thinkingBudget` (`-1` = dynamic/full
thinking; `0` = off; `N` = a budget).

With that deployed, the AIME gap closes:

| Config | Ours | vs published 72.0 |
|---|---:|---|
| no thinking, 16k / 32k / 64k tokens | 50 / 57 / 53 | far below |
| `reasoning.max_tokens: -1`, 32k tokens | 53 | still truncating |
| **`reasoning.max_tokens: -1`, 65536 tokens** | **73.3** (22/30) | ✅ **matches** |

The second lever matters: Gemini's thinking tokens **count against
`maxOutputTokens`**, so at 32k the thinking eats the budget and the visible answer
truncates before `\boxed`. At 65536 both fit (median ~14k completion tokens, some
hit the full 65k). So reproducing a lab's thinking-mode number needs **both** the
thinking budget AND headroom. The AIME default `max_tokens` is therefore 65536.
(GSM8K/MATH-500 need less and stay at 32768.) To reproduce thinking-mode scores,
pass `extra_body={"reasoning": {"max_tokens": -1}}`.

## MATH-500 — validated via gpt-4o-mini

MATH-500 is one of the most widely published benchmarks — Artificial Analysis
runs a full MATH-500 leaderboard (GPT-5 99.4, o3 99.2 at the top). The only true
narrow fact is that the **Gemini 2.5 family reports AIME 2025, not MATH-500**, so
`gemini-2.5-flash` was a bad anchor for it. We calibrate instead against
**gpt-4o-mini** — non-thinking (no truncation ambiguity) and widely reported at
~70–75% (OpenAI's own gpt-4o-mini "MATH" = 70.2).

| Model | Published MATH-500 | Ours (full 500) | Verdict |
|---|---|---:|---|
| `openai/gpt-4o-mini` | ~70–75 | **73.8** (369/500) | ✅ **right in the band — validated** |

This also validates the vendored Hendrycks `is_equiv` scorer. (Our
`gemini-2.5-flash` MATH-500 = 90.8 is plausible for a thinking model but has no
published MATH-500 to check against — see the AIME note re: thinking budget.)

## GSM8K — saturated, dropped by recent labs

GSM8K has plenty of public numbers for older models, but recent labs (including
the Gemini 2.5 family) have dropped it from their cards as saturated — it's
near-100 for everyone. We keep it as a cheap deterministic sanity check, not a
discriminator, and don't claim a published match for the current panel.

## Aider polyglot — not directly calibratable here

Our run is the **Python subset, pass@1**; the public Aider leaderboard is the
full 225 exercises across 6 languages with a 2nd attempt. So our numbers are a
strict floor, not a reproduction of the published leaderboard. Calibrating
against the leaderboard would require running the full polyglot harness.

## Stable open-weight anchors — the harness reproduces across many models

The frontier/Chinese flagships drift and rarely publish standardized per-eval
numbers, so they can't anchor anything. The cleanest validation is to run
**pinned open-weight checkpoints** (+ the gpt-4o family) whose IFEval / MMLU-Pro /
GSM8K numbers are in their model cards (`CALIBRATION_ANCHORS` in `panel.py`).
Across 7 independent models, ours land within a few points of the cards — the
harness reproduces, it's not tuned to one model:

| Model | IFEval (card) | GSM8K (card) | MMLU-Pro (card) |
|---|---|---|---|
| `llama-3.3-70b-instruct` | 93.5 (~92) | 96.7 (~95) | 70.7 (~69) |
| `llama-3.1-70b-instruct` | 85.0 (~87) | 100 (~95) | 64.7 (~66) |
| `llama-3.1-8b-instruct` | 83.2 (~80) | 83.3 (~84) | 42.0 (~48) |
| `qwen-2.5-72b-instruct` | 86.7 (~84) | 96.7 (~96) | 65.3 (~58–71) |
| `qwen-2.5-7b-instruct` | 74.8 (~75) | 86.7 (~85) | 57.3 (~56) |
| `gpt-4o` | 86.5 (~87) | 93.3 (~96) | 70.7 (~73) |
| `gpt-4o-mini` | 83.4 (~86) | 96.7 (~93) | 60.7 (~63) |

(100-prompt IFEval / 30-problem GSM8K / 150-question MMLU-Pro subsets, so a few
points of subset noise on top of 0-shot-vs-few-shot protocol differences.
`llama-3.1-8b` MMLU-Pro is low because its small output limit still truncated ~21
answers even at 8192.)

> **Gateway gotcha found here.** Requesting `max_tokens` beyond a model's output
> limit made the TR gateway return an opaque **502 "provider error"** (sometimes an
> empty 200), silently zeroing the small-output anchors (gpt-4o/-mini, qwen-2.5,
> llama-3.1) at the 32768 default. Re-running them at 8192 fixed it. The gateway is
> being upgraded to surface the real upstream 4xx (e.g. "max_tokens is too large")
> instead of masking it. Big reasoning models are unaffected (they accept 32768).

## Method

Calibration runs use the same harness, full dataset, and the published model
version where it routes on TrustedRouter. Numbers are filled in once the runs
complete; any gap is investigated before publishing the new Chinese-panel data.
