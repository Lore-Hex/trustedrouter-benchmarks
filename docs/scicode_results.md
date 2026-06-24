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

**Calibration.** with-background lifts Haiku **+11.5 pts** (23.6 → 35.1). The published ~43.3
is a **different test, not a backend gap** — see *Replication diagnosis* below: it's Artificial
Analysis's **independent-subproblem (gold-context)** methodology vs the **official cascading**
harness we reproduce. (An earlier guess blaming the CC-subagent backend was DISPROVEN — a
clean temp-0 TR API call also gives 34.7%.) RELATIVE comparisons here are valid (with-vs-
without-bg, fusion-vs-solo share the backend). Harness sound: validation = 25/50 (50%), only
2/288 without-bg timeouts; gold withheld from the HF dataset (0/291) so no direct gold-check.

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

## Replication diagnosis (why our Haiku ≈ 35%, published ≈ 43.3)
With-background Haiku 4.5 = **34.7% (100/288)**, and it is ROCK-SOLID — reproduces across
every controllable variable; the gap is NOT in our harness:
| variable | tested | result |
|---|---|---|
| backend / temperature | CC subagent vs TR clean **temp-0** | 35.1% vs 34.7% (same) |
| truncation | max_tokens 4096 | 0 cut-offs |
| class-based problems | pass rate | pass *more* (44%) not less |
| numpy / Python | 2.5/3.14 vs **1.26/3.12** | 100/288 both (identical) |
| prompt | vs official `process_problem_steps` | line-for-line match |
| harness | gencode vs inspect_ai | identical generation+scoring logic |

The TR call uses the exact official params (temp 0, max_tokens 4096, no system prompt). Most
fails are genuine `AssertionError`s (wrong values = real model errors).

**RESOLVED — it's the methodology, not a bug or serving.** The published 43.3 is from
[Artificial Analysis](https://artificialanalysis.ai/evaluations/scicode), whose documented
SciCode method is **independent subproblems**: the model is given the **gold/correct previous-
step code** (errors don't cascade), with background, temp 0, pass@1. The **official public
harness we reproduced uses the model's OWN previous code (cascading)** → strictly harder. That
is the whole gap (AA's top model = 60.2% vs the official subproblem ceiling ~34%). Evidence
from our own run: per-step capability *once predecessors are correct* = **65/127 = 51.2%**, and
**161/288 steps were made unreachable by an earlier broken step** — AA gives those gold
predecessors so they score independently. **43.3 sits inside our [34.7%, 51.2%] bracket.** We
can't match it exactly because the gold solutions are withheld from the public HF dataset
(0/291); a true AA-style run needs gold predecessors (obtain gold, or generate verified
pseudo-gold with a strong model, then score each subproblem independently).
Tools: `scripts/scicode_tr_gen.py` (clean temp-0 generation via TR), `scripts/scicode_score.py`
(now uses `sys.executable` env; numpy-1.26 env = `scicode/.venv312`).

## Tier-diverse fusion (Sonnet + 2 Haiku + 1 Opus → Sonnet synth) — confirms the rule
Problems 21–28 (24 steps, without-background, subagents):
| | subproblem | main |
|---|---|---|
| Sonnet solo | 4/24 (16.7%) | 0/8 |
| **tier-fusion** | **6/24 (25.0%)** | 0/8 |
| Opus solo | 9/24 (37.5%) | 2/8 |

**Tier diversity DID lift fusion over Sonnet solo (+8.3)** — different Claude TIERS make
different errors, so the Sonnet synth had real complementary signal (unlike 5 same-Sonnet
stances, which tied solo). **BUT fusion < Opus solo** — the panel had a dominant member
(Opus), so fusion landed BETWEEN the synth's tier and the best member, not above all.

### Drop Opus → Sonnet+Haiku fusion: PILOT looked like a win, but the powered run says TIE
Panel = **Sonnet + Sonnet + Haiku + Haiku** (no Opus) → Sonnet synth.

**Pilot (8 problems / 24 steps):** Haiku 8.3% · Sonnet 16.7% · fusion **29.2%** — looked like a
clean +12.5 win. **But it was small-sample noise** (those 8 problems are a hard subset; Sonnet
drew an anomalously low 16.7%).

**Powered run (23 problems / 71 steps, cluster bootstrap over problems, B=10,000):**
| | rate | 95% CI |
|---|---|---|
| Haiku solo | 12.7% | [5.6, 20.5] |
| Sonnet solo (best member) | 28.2% | [18.3, 38.4] |
| **Sonnet+Haiku fusion** | **29.6%** | [19.2, 40.8] |

**FUSION − Sonnet-solo = +1.4 pts, 95% CI [−11.0, +14.1], P(fusion>solo)=54%** (McNemar 6 vs 5).
**No significant difference — fusion ≈ Sonnet-solo.** The rule is *reinforced*, not broken:
Sonnet+Haiku still has a **dominant member** (Sonnet 28% ≫ Haiku 13%), so fusion tracks the best
member. "No dominant member" requires *comparable-strength* models — which Sonnet/Haiku aren't.
⭐ LESSON (again): validate fusion deltas on ≥20 items; the 8-problem pilot false-positived
exactly as warned. The genuinely-untested winning config = a comparable, different-family
**open-weight** panel (qwen3/deepseek/glm/kimi/minimax) via TR — gated on $. Harness
`scripts/scicode_fusion_mix.py` presets: `sh`/`tier` (+solo baselines), `shf`/`tierf`
(fusion-only — reuse saved solo replays).
