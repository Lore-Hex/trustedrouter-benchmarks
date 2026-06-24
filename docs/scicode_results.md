# SciCode results (Haiku-4.5 baseline + Sonnet solo-vs-fusion)

SciCode = 80 main problems / 338 dependent subproblems of research-level scientific coding
(test split = 65 problems / 288 scored steps). Generated via Haiku/Sonnet **subagents**
(`scicode_gen.py` / `scicode_fusion_gen.py`, faithful to upstream `gencode.py`), scored with
SciCode's **own** test runner against the 1.05 GB numeric ground truth. Two settings:
**without-background** (model must recall the science — the "realistic" leaderboard setting)
and **with-background** (gold science is given — easier).

## Haiku 4.5 — test split, WITHOUT background
| metric | score |
|---|---|
| subproblem | **68/288 = 23.6%** |
| main problem | **1/65 = 1.5%** |

**Calibration.** This is the *hard* setting. On the same setting the old leaderboard has
Claude-3.5-Sonnet at 26% subproblem, so Haiku-4.5 at 23.6% is in-band (a small model can lag
a larger older one on deep science recall). The **published Haiku-4.5 ≈ 43.3 is almost
certainly the *with-background* setting** — a `with_background` run is queued to confirm
(`scicode_gen.py test _TESTBG haiku 12 bg`). Harness is sound: validation split = 25/50 sub
(50%) on the same venv, only 2/288 timeouts (not a timeout-deflation artifact), 218 genuine
fails. Gold solutions are withheld from the public HF dataset (0/291), so no direct gold-check.

## Sonnet — solo vs fusion (per-step 5-panel → judge → evidence_decide synth)
3 problems (53, 59, 65), without-background, 15 steps:
| | subproblem | main | per-problem |
|---|---|---|---|
| solo | 9/15 (60%) | 0/3 | 53:2/4 · 59:2/5 · 65:5/6 |
| **fusion** | **11/15 (73%)** | **1/3** | 53:**4/4** · 59:2/5 · 65:5/6 |

**Fusion edged solo (+2 sub, +1 main)** — driven entirely by problem 53 (2/4 → fully solved).
Unlike tau2 tool-calling (where 5 Sonnet stances converge → fusion ≈ solo), **code-gen has
real diversity surface**: stances write genuinely different implementations and the synth can
re-derive the correct one. n=3 → directional, not proof (the win is one problem); a larger
subset is needed to confirm.

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
