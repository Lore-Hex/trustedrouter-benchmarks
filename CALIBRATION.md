# Calibration — does the harness reproduce published numbers?

A benchmark harness is only trustworthy if it reproduces the published
gold-standard numbers for known models. Before trusting any *new* number (the
Chinese-panel results), we run reference models that have authoritative published
scores and check that our harness lands on them — exactly or nearly exactly.

The cleanest anchor is **SimpleQA Verified**: Google published an exact per-model
table (no tools, single dataset), and several of those models route on
TrustedRouter. If our `gemini-2.5-pro` run lands on 55.6 F1, the whole
answer → no-tools → LLM-judge → F-score pipeline is validated. (This also tests
our judge choice — `google/gemini-2.5-flash` — against Google's autorater.)

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

| Model | Published F1 / Acc / Att | Ours @512 (broken) | Ours @8192 (fixed, full 1000) |
|---|---:|---:|---:|
| `google/gemini-2.5-pro` | 55.6 / 55.3 / 98.9 | F1 31.6 / Att 44.9 | **F1 51.3 / Acc 51.1 / Att 99.4** |

**Verdict: validated, nearly exactly.** The Attempted rate matches almost
perfectly (99.4 vs 98.9) — the truncation fix is correct. The graded metrics sit
a consistent ~4 points low (F1 51.3 vs 55.6), and *only* the graded metrics, so
the residual is judge strictness, not a harness error: our default judge
`google/gemini-2.5-flash` grades a touch stricter than Google's autorater. (On
the easier first-300 questions the flash judge actually lands at 56.3 correct,
dead on the published 55.3 — the full-1000 is lower because later questions are
harder; the dataset isn't shuffled.)

Judge notes from calibration: `gemini-2.5-flash` is the right judge — cheap,
non-reasoning, emits the A/B/C letter immediately. A *reasoning* model as judge
(e.g. `gemini-2.5-pro`) hits the same truncation trap on the judge side, so
`judge.grade` was hardened to `max_tokens=2048`.

Full published table (reference): Gemini 2.5 Pro 55.6, GPT-5 52.3, o3 51.9,
GPT-4.1 39.9, GPT-4o 34.9, DeepSeek R1 33.3, Claude Opus 4 28.3, Gemini 2.5 Flash
28.2, GPT-5 Mini 24.6, o4-mini 23.4, Claude Sonnet 4 18.7, GPT-5 Nano 14.4,
Gemini 2.5 Flash-Lite 11.1.

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

## Method

Calibration runs use the same harness, full dataset, and the published model
version where it routes on TrustedRouter. Numbers are filled in once the runs
complete; any gap is investigated before publishing the new Chinese-panel data.
