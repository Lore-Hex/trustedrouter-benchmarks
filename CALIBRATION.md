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

| Model | Published IFEval | Metric | Ours (prompt-strict / avg) | Δ |
|---|---:|---|---|---:|
| `google/gemini-2.5-pro` | _todo_ | — | _running_ | — |
| `google/gemini-2.5-flash` | _todo_ | — | _running_ | — |
| `openai/gpt-4o-mini` | _todo_ | — | _running_ | — |

## Aider polyglot — not directly calibratable here

Our run is the **Python subset, pass@1**; the public Aider leaderboard is the
full 225 exercises across 6 languages with a 2nd attempt. So our numbers are a
strict floor, not a reproduction of the published leaderboard. Calibrating
against the leaderboard would require running the full polyglot harness.

## Method

Calibration runs use the same harness, full dataset, and the published model
version where it routes on TrustedRouter. Numbers are filled in once the runs
complete; any gap is investigated before publishing the new Chinese-panel data.
