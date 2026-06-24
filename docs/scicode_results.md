# SciCode results (Haiku-4.5 baseline + Sonnet solo-vs-fusion)

SciCode = 80 main problems / 338 dependent subproblems of research-level scientific coding
(test split = 65 problems / 288 scored steps). Generated via Haiku/Sonnet **subagents**
(`scicode_gen.py` / `scicode_fusion_gen.py`, faithful to upstream `gencode.py`), scored with
SciCode's **own** test runner against the 1.05 GB numeric ground truth. Two settings:
**without-background** (model must recall the science — the "realistic" leaderboard setting)
and **with-background** (gold science is given — easier).

## Haiku 4.5 — test split (65 problems / 288 steps)
| setting | subproblem | main |
|---|---|---|
| without-background (realistic) | **68/288 = 23.6%** | 1/65 = 1.5% |
| with-background (science given) | **101/288 = 35.1%** | 2/65 = 3.1% |

**Calibration.** with-background lifts Haiku **+11.5 pts** (23.6 → 35.1), confirming the
**published ~43.3 is the with-background setting** (Haiku does much better when handed the
science). The residual **35.1 vs 43.3** (~8 pts) is the **generation backend**: this harness
drives generation with Claude Code *subagents* (full agentic system prompt, non-zero temp),
not a clean temperature-0 API call. For *agentic* tau2 that matched the leaderboard (Sonnet
70–75 ≈ 74); for SciCode *single-shot code-gen* the agentic wrapper + temp deflates absolute
scores ~8 pts (also 5/288 with-bg timeouts at 120 s). **So absolute SciCode numbers here run
~8 pts low vs a clean-API leaderboard run, but RELATIVE comparisons are valid** (with-vs-
without-bg, fusion-vs-solo share the backend). Harness otherwise sound: validation = 25/50
(50%), only 2/288 without-bg timeouts; gold is withheld from the HF dataset (0/291) so no
direct gold-check.

## Sonnet — solo vs fusion (per-step 5-panel → judge → evidence_decide synth)
Two runs, without-background:

| run | solo subproblem | fusion subproblem | solo main | fusion main |
|---|---|---|---|---|
| small (3 problems / 15 steps) | 9/15 (60%) | **11/15 (73%)** | 0/3 | 1/3 |
| **big (23 problems / 71 steps)** | **20/71 (28.2%)** | 18/71 (25.4%) | 0/23 | 0/23 |

**The small run's fusion "+2" was NOISE** (it landed on problem 53, 2/4→4/4). At scale fusion
is **slightly WORSE than solo** (18 vs 20; won 2 problems 24/32, lost 3: 35/41/45) — the synth
sometimes steers to a worse consensus than solo's own answer (same failure mode as tau2 5/6).

**Unified rule confirmed on a SECOND benchmark.** Same-model (single-model) fusion ≈ solo —
holds for agentic tool-calling (tau2) AND research code-gen (SciCode). A genuinely diverse,
no-dominant-member multi-MODEL panel is required to beat solo; same-model stance diversity is
not enough. Lesson: validate fusion deltas on ≥20 items — small subsets give false positives.

## Run
```bash
# Haiku baseline (without-background): gen -> Workflow -> score
python scripts/scicode_gen.py test _TEST haiku 12         # -> results/_scicode_TEST.js
# (run the .js via Workflow, save output to results/_scicode_TEST_out.json)
SCICODE_TIMEOUT=120 python scripts/scicode_score.py results/_scicode_TEST_out.json haiku test
# with-background: add `bg` to both
python scripts/scicode_gen.py test _TESTBG haiku 12 bg
SCICODE_TIMEOUT=120 python scripts/scicode_score.py results/_scicode_TESTBG_out.json haiku test bg
# Sonnet solo vs fusion on a subset:
python scripts/scicode_fusion_gen.py "53,59,65" _SF sonnet 3
```
See `scicode/VENDOR_SETUP.md` for the one-time `test_data.h5` fetch.
