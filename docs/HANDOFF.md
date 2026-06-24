# HANDOFF — Agentic Fusion + SciCode (master)

Entry point for collaborators. Two research threads on the central question
**"does TrustedRouter Fusion (panel → judge → synth) beat the best single model?"** —
now tested in **agentic** (tau2) and **code-generation** (SciCode) settings, not just
single-shot tool-calling.

## ⭐ TL;DR — the unified finding (confirmed 3×)
**Fusion beats solo ONLY with a no-dominant-member, diverse *multi-model* panel.**
- A **same-model** panel — even 5 diverse "stances" of one model — **≈ solo** (the stances
  share the model's errors, so the synth has no complementary signal).
- A **tier-mixed Claude** panel (Sonnet+Haiku+Opus) beats the *synth's* tier but **loses to
  the dominant member** (Opus) — you'd just run Opus.
- **Sonnet+Haiku panel — pilot looked like a win, the FULL test split is a dead TIE:** an
  8-problem pilot showed fusion 29.2% vs Sonnet-solo 16.7%, but powering up to the **full 65-
  problem / 288-step test split** gives **fusion 35.7% = Sonnet-solo 35.7%** (104 = 104 passing
  steps; diff **+0.0**, 95% CI **[−4.6, +4.7]**, P=46.8%; per-problem fusion wins 7 / solo wins 7
  / 51 tie). The difference CI tightened ±12.5 → ±4.7 (~2.6×) from 23 → 65 problems. Sonnet+Haiku
  *still* has a dominant member (Sonnet ≫ Haiku), so the rule holds; the pilot was small-sample
  noise (⭐ validate fusion deltas on the FULL set — the 8- and 23-problem runs both over-stated).
- **The lever is panel composition, not the synth prompt.** Multiple synth-prompt variants
  (`evidence_decide`, backward-chaining "derive→ground→commit") all tie at same-model tier.
  **No fusion>solo win has yet survived a powered run** — the untested config that should is a
  *comparable, different-family* open-weight panel (no dominant member), gated on TR $.

Evidence:
| setting | result |
|---|---|
| tau2 retail, Sonnet solo | 70–75% (real evaluator); **2-run oracle 90%** |
| tau2, panel→synth (5 Sonnet stances), full-20 | **75% = ties best-solo** (won task 1 neither solo got; lost task 5 both solo got) |
| SciCode, Sonnet solo vs same-model fusion (23 problems) | solo 28.2% vs fusion 25.4% — **fusion slightly worse** (small-sample +2 was noise) |
| SciCode, tier-mix fusion (24 steps) | Sonnet-solo 16.7% < **fusion 25.0%** < Opus-solo 37.5% (dominant member) |
| SciCode, **Sonnet+Haiku** fusion (**FULL 65 problems / 288 steps**, error bars) | **fusion 35.7% = Sonnet-solo 35.7%** (104=104 steps; diff **+0.0**, 95% CI [−4.6,+4.7], P=47%) — dead TIE; CI tightened ~2.6× vs 23-problem run; pilot's +12.5 was noise |

The one configuration the rule says **should** win and we have **not** run: a panel of
**comparable, different-family open-weight models with no dominant member** (qwen3 /
deepseek-v3.2 / glm-4.6 / kimi-k2.6 / minimax-m2.5) → `evidence_decide` synth, vs best solo
member. Needs TrustedRouter (real $) — see *Cost / gating*.

## ⭐ SciCode "43.3 vs our 35%" — RESOLVED
The published Haiku-4.5 ≈ 43.3 is from [Artificial Analysis](https://artificialanalysis.ai/evaluations/scicode),
whose SciCode is a **different, easier methodology: independent subproblems** (model gets the
**gold** previous-step code, errors don't cascade) + with-background + temp 0 + pass@1. We
faithfully reproduce the **official public harness** (model's **own** previous code → cascades)
= **34.7% with-bg, rock-solid** (reproduces across CC-subagent vs clean temp-0 TR API, numpy
2.5 vs 1.26, verified-identical prompt, gencode==inspect_ai). Not a bug, not serving — a
different test. Can't match AA exactly: gold solutions are **withheld** from the public dataset
(0/291). Full diagnosis in `scicode_results.md`.

## File map (`scripts/`)
**tau2 agentic fusion** (uses `../tau2-bench`, real evaluator):
- `tau2_grade.py` — ⭐ score a trajectory with tau2's OWN evaluator (gold actions → 1.0). The keystone.
- `gen_retail_native.py` — solo / per-step-fusion harness (schema tool-calls, verbatim tau2 prompts).
- `gen_retail_explore.py` — read/write-asymmetric "explore" harness (single proposer).
- `gen_retail_panel.py` — panel→judge→synth (5-stance) harness. Current synth prompt = the
  backward-chaining `evidence_decide` ("derive the action → ground its ids → commit").

**SciCode** (vendored at `scicode/`; uses SciCode's own test runner):
- `scicode_gen.py` — generation chain via CC subagents. `bg` 5th arg = with-background.
- `scicode_score.py` — reconstruct cumulative files + score with SciCode's runner. `bg` 4th arg;
  `SCICODE_TIMEOUT` (default 120s); uses `sys.executable` for the test subprocess.
- `scicode_fusion_gen.py` — Sonnet solo vs same-model (5-stance) fusion.
- `scicode_fusion_mix.py` — **tier-diverse** panel (per-member model: Sonnet+2Haiku+1Opus) vs Sonnet/Opus solo.
- `scicode_tr_gen.py` — **clean temp-0 generation via the TrustedRouter SDK. COST-GUARDED** —
  refuses to run unless `SCICODE_TR_CONFIRM=1` (TR calls cost real money).
- `scicode_split_out.py` — split a `scicode_fusion_mix` Workflow output into a per-method
  `{problem_id, code}` list (`fusion` / `solo_sonnet`) for `scicode_score.py`.
- `scicode_merge65.py` — merge the per-batch scored correct_dicts into a single all-65 dict.
- `scicode_bootstrap.py` — ⭐ cluster-bootstrap error bars (resamples PROBLEMS) on the
  fusion-vs-best-solo difference. Reuses already-scored `eval_results/<model>_<bg>.json`.

**single-shot fusion synth-prompt research** (BFCL/GSM8K): `fusion_*.py` (see `project_fusion_synth_research` memory).

## How to run
### tau2 agentic fusion
```bash
# 1) generate a workflow (writes results/_retail_*.js)
python scripts/gen_retail_panel.py "0,1,2,...,19" _RUN sonnet 2
# 2) run that .js with the Workflow tool; save its output JSON
# 3) convert to {task_id: trajectory} and grade with tau2's REAL evaluator:
cd ../tau2-bench && TAU2_DB_ONLY=1 .venv/bin/python <repo>/scripts/tau2_grade.py grade <traj.json>
#    validate the grader first: tau2_grade.py goldcheck 0   # gold actions -> reward 1.0
```
### SciCode
```bash
# one-time: scicode/ venv + gdown test_data.h5 — see scicode/VENDOR_SETUP.md
python scripts/scicode_gen.py test _TEST haiku 12 [bg]     # -> results/_scicode_TEST.js
# run the .js via Workflow, save output to results/_scicode_TEST_out.json, then:
SCICODE_TIMEOUT=120 python scripts/scicode_score.py results/_scicode_TEST_out.json haiku test [bg]
# fusion variants:
python scripts/scicode_fusion_gen.py "53,59,65" _SF sonnet 3        # same-model
python scripts/scicode_fusion_mix.py "21,...,28" _MIX 2             # tier-diverse
# clean temp-0 via TR (COSTS MONEY — opt in):
SCICODE_TR_CONFIRM=1 python scripts/scicode_tr_gen.py test anthropic/claude-haiku-4.5 bg _TRBG
```

## Replays
`docs/handoff_data/` — all run outputs + graded trajectories (see its `README.md` for the
grading command per file). Raw scratch lives in gitignored `results/_*`.

## Detailed docs
- `docs/HANDOFF_fusion_agentic.md` — full tau2 agentic-fusion design + the panel→synth-decides spec.
- `docs/scicode_results.md` — full SciCode numbers + the replication diagnosis.
- `docs/fusion_agentic_findings.md`, `docs/agentic_fusion_explore_plan.md` — supporting writeups.

## Cost / gating (READ before running)
- **Subagent runs** (gen_retail_*, scicode_gen/fusion_*) use the **Claude Code quota** — there
  is a nightly session cap AND a **weekly** cap; high-volume Sonnet/Opus runs hit them. Run low
  concurrency; one workflow at a time.
- **TrustedRouter** (`scicode_tr_gen.py`, any TR fusion) = **real $** on the operator's account.
  `scicode_tr_gen.py` is cost-guarded (`SCICODE_TR_CONFIRM=1`). TR has `anthropic/claude-*` +
  230 models incl. the open-weight set; clean temp-0 calls work via `api.trustedrouter.com`.
- **SciCode gold** is withheld → can't reproduce AA's independent-subproblem 43.3 without
  generating verified pseudo-gold (strong model solves each step, keep only if it passes the
  real test targets) and scoring each subproblem independently.

## Open questions / next steps
1. ⭐ **No-dominant-member open-weight panel** (the config the rule says should beat solo):
   qwen3 / deepseek-v3.2 / glm-4.6 / kimi-k2.6 / minimax-m2.5 → evidence_decide synth, vs best
   solo member, on a SciCode (or tau2) subset. **Gated on TR $.**
2. **AA-style SciCode** (independent/gold) to land Haiku ~43.3: generate verified pseudo-gold,
   score each subproblem with gold predecessors.
3. **tau2 full-20 with a diverse open-weight panel** — the genuine fusion>solo demo (Sonnet
   self-panel only tied; oracle headroom is 90%).
