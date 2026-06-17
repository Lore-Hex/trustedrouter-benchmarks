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

| Model | Published F1 | Published Acc | Published Att | Ours F1 | Δ |
|---|---:|---:|---:|---:|---:|
| `google/gemini-2.5-pro` | 55.6 | 55.3 | 98.9 | _running_ | — |
| `google/gemini-2.5-flash` | 28.2 | 27.8 | 96.9 | _todo_ | — |
| GPT-4o | 34.9 | 34.4 | 97.0 | n/a on TR | — |
| DeepSeek R1 | 33.3 | 32.7 | 96.4 | n/a on TR | — |

Full published table (for reference): Gemini 2.5 Pro 55.6, GPT-5 52.3, o3 51.9,
GPT-4.1 39.9, GPT-4o 34.9, DeepSeek R1 33.3, Claude Opus 4 28.3, Gemini 2.5 Flash
28.2, GPT-5 Mini 24.6, o4-mini 23.4, Claude Sonnet 4 18.7, GPT-5 Nano 14.4,
Gemini 2.5 Flash-Lite 11.1.

A match within a couple of points = the judge + scoring are faithful. A large gap
means our judge is mis-calibrated vs Google's autorater (the thing to fix first).

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
