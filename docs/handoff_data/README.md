# Handoff result trajectories (tau2 retail, graded by tau2's REAL evaluator)
Grade with: cd ../../../tau2-bench && TAU2_DB_ONLY=1 .venv/bin/python <repo>/scripts/tau2_grade.py grade <file>

- sonnet_solo_run1.json / run2.json      — near-identical Sonnet solo (70% / 75%); 2-run oracle 90%
- sonnet_solo_loosejson_45pct.json       — earlier loose-JSON harness Sonnet solo (45%) — shows harness gap
- sonnet_perstep_fusion_8tasks.json      — per-step fusion, Sonnet, 8 tasks (50%) ~= solo
- explore_v3_smoke.json                  — v3 read/write-asymmetric explore, tasks 0,1,2 (67% = oracle, 2x baseline)
