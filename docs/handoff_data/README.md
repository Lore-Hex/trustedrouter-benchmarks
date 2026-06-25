# Handoff replays

Raw run outputs + graded trajectories. See `docs/HANDOFF.md` for the full story.

## tau2 retail (grade with tau2's REAL evaluator)
`cd ../../../tau2-bench && TAU2_DB_ONLY=1 .venv/bin/python <repo>/scripts/tau2_grade.py grade <file>`
- `sonnet_solo_run1.json` / `run2.json`     — near-identical Sonnet solo (70% / 75%); 2-run oracle 90%
- `sonnet_solo_loosejson_45pct.json`        — earlier loose-JSON harness Sonnet solo (45%) — shows harness gap
- `sonnet_perstep_fusion_8tasks.json`       — per-step fusion, Sonnet, 8 tasks (~50%) ≈ solo
- `explore_v3_smoke.json`                    — read/write-asymmetric explore, tasks 0,1,2 (67% = oracle, 2× baseline)
- `panel_synth_smoke_3of3.json`             — panel→synth (3-stance) smoke, tasks 0,1,2 (3/3 — unrepresentative)
- `panel5_fullA_tasks0-1_2of2.json`         — 5-stance panel→synth, salvaged tasks 0,1 (2/2)
- `panel5_full20_traj_75pct.json`           — ⭐ 5-stance panel→synth FULL 20 = 75% (ties best-solo)
- `tau2_panel_evid_decide_retest_3of7.json` — backward-chaining evidence_decide synth retest (3/7 = old prompt; prompt isn't the lever)

## SciCode (raw workflow outputs `[{problem_id, code|solo|fusion...}]`; score with scicode_score.py)
- `scicode_haiku_test_gen.json`             — Haiku 4.5 test-split gen, without-bg (scored 23.6% sub)
- `scicode_haiku_test_withbg_gen.json`      — Haiku 4.5 test-split gen, with-bg (35.1% sub)
- `scicode_haiku_tr_cleanapi_withbg.json`   — Haiku 4.5 via clean temp-0 TR API, with-bg (34.7% — = subagent, rules out backend)
- `scicode_sonnet_solo_vs_fusion.json`      — Sonnet solo vs same-model fusion, 3 problems (fusion 11 vs 9 — NOISE)
- `scicode_sonnet_solo_vs_fusion_big.json`  — ⭐ same, 23 problems (solo 28.2% > fusion 25.4%)
- `scicode_tier_diverse_fusion.json`        — ⭐ Sonnet+2Haiku+1Opus fusion vs Sonnet/Opus solo (16.7% < 25.0% < 37.5%)
- `scicode_sonnet_haiku_fusion_nodominant.json`   — Sonnet+Haiku panel→synth, 8-problem pilot (fusion 29.2% — FALSE positive, small-sample noise)
- `scicode_sonnet_haiku_fusion_powered_23prob.json` — Sonnet+Haiku fusion code, 23 problems (powered run → TIE)
## Terminal-Bench (official harness, free `claude -p` Haiku agent)
- `terminalbench_haiku_smoke10.json` — Haiku 4.5 via free `claude -p` on a curated 10-task subset of terminal-bench-core 0.1.1 = **20% (2/10)**; harness validated (oracle + Haiku both 100% on hello-world). NOT AA's 27.3% (different scaffold + secret 47-task subset). See `docs/terminalbench_results.md`.
- `terminalbench_sonnet_smoke10.json` — Sonnet 4.6 on the SAME 10 tasks = **50% (5/10)** (strict superset of Haiku; 3 of 5 fails are `agent_timeout`).
- `terminalbench_claude_models_smoke10.json` — ⭐ consolidated 4-model board (Haiku/Sonnet/Opus-4.8/Opus-4.7) incl. the **generous-budget reruns**: default Haiku 20% · Sonnet 50% · Opus-4.8 40% · Opus-4.7 70%; with `--global-agent-timeout-sec 1800` Opus-4.8→**70%**, Opus-4.7→**80%**. Shows the score is latency-bound for slow models on the free `claude -p` path.

## SciCode (cont.)
- `scicode_sonnet_haiku_fusion_full65.json` — ⭐⭐ **FULL 65-problem / 288-step verdict** (self-contained: `verdict` + `correct_dicts` {fusion,sonnet,haiku} + per-problem `generated_code`). **fusion 35.7% = Sonnet-solo 35.7%** (104=104 steps; diff +0.0, 95% CI [−4.6,+4.7]) — dead TIE. To re-bootstrap: write `correct_dicts.{fusion,sonnet,haiku}` to `scicode/eval_results/{all65_fusion,all65_sonnet,haiku}_without_background.json`, then `python scripts/scicode_bootstrap.py all65_fusion all65_sonnet,haiku <pids> test nobg`. The 42 batch problems are fully re-scorable from `generated_code` via `scicode_score.py`.
