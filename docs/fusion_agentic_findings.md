# Fusion synthesizer-prompt research + agentic fusion (tau2)

All runs here are **all-Haiku / all-Sonnet via Claude Code subagents** (TR credits were
out), orchestrated with the Workflow tool. They isolate two questions:

1. What makes `trustedrouter/fusion` beat the best single model?
2. Does running a `fusion` step at **every agent turn** ("agentic fusion") help on a
   real agentic benchmark (tau2-bench retail)?

## TL;DR

- **Single-shot fusion (tool-calling).** Fusion beats best-solo **only when no panel
  member dominates** — i.e. weak/diverse, complementary models. When one model dominates
  (real TR panel: kimi-k2.6 ≈ oracle), fusion ties it at best. This explains why deployed
  `fusion-code` ≈ glm-solo on tools.
- **Synthesizer wording matters, on the items that decide it.** On the *decidable* subset
  (panel disagrees, ≥1 right), the shipped baseline synth instruction is the **worst**;
  **`evidence_decide`** ("use the panel as evidence but INDEPENDENTLY re-derive the correct
  function + argument values from the schema; the right answer may be one only a single
  member proposed") is the **best / most robust** (+9 pts on decidable, +4 overall). The
  whole-set average hides this because ~half the items are trivial (panel agrees).
- **Method that surfaced it — decidable-focused.** Run panel-only first, grade, keep only
  the disagreement items, then sweep synth prompts on those. At n=8/20/44 full-set the
  synths looked "tied" (low power); at 77 decidable items the gap is clear.
- **Agentic fusion (tau2 retail, Haiku).** Per-step `evidence_decide` fusion **beats solo
  by ~+10 pts** on the 20-task mixed set (writes-only 25 vs 15; strict 15 vs 5). Fusion's
  win mechanism: it keeps the agent from stalling/drifting mid-sequence.
- **Reward is the REAL tau2 reward.** `scripts/tau2_grade.py` builds a `SimulationRun` from a
  trajectory and calls tau2's own `evaluate_simulation` (env-replay + DB-state compare +
  action + communicate). Validated: feeding the **gold actions → reward 1.0**. DB component
  is deterministic (no LLM); NL-judge needs an OpenAI cred (can substitute Sonnet).

## Numbers (tau2 retail, 20 mixed tasks `0–19`, REAL evaluator, DB reward)

| setup | reward |
|---|---|
| Haiku solo (my loop)        | ~15% (writes-only) |
| Haiku fusion (per-step)     | ~25% (writes-only) — **+10 over solo** |
| Sonnet solo (loose-JSON loop)        | 45% (DB, real evaluator) |
| Sonnet solo (near-identical loop)    | **70% & 75%** (two runs, avg ~72.5%, DB, real evaluator) |
| — published full retail —            | gpt-4.1-mini 66 · **Sonnet 4.6 (ours) 70–75** · o4-mini 71.5 · gpt-4.1 74.1 · claude-3.7-sonnet 78.7 |

**The 45 → ~72 jump (same model, same scorer) proves the gap was harness-looseness, not
the model.** `scripts/gen_retail_native.py` (tau2's verbatim agent + user-sim prompts +
schema-forced structured tool-calls) lands Sonnet 4.6 squarely in the published band —
≈ o4-mini / gpt-4.1 tier (validating gpt-4.1 ≈ Sonnet 4.6). Run-to-run variance is real
Sonnet non-determinism (e.g. recovering a dropped `#` in an order id). DB-only reward; NL
factor (usually ≈ 1) omitted since its judge needs an OpenAI cred (substitute Sonnet).

## Repro

```bash
# single-shot synth-prompt research (BFCL):
PYTHONPATH=. .venv/bin/python scripts/fusion_confirm.py          # 5-model panel, prompt sweep
PYTHONPATH=. .venv/bin/python scripts/fusion_qa.py build1 gsm8k 180   # generic QA harness

# agentic fusion (tau2 retail), generate + run via Workflow, then grade with REAL evaluator:
python scripts/gen_retail_native.py "0,1,...,19" "_SONNAT" sonnet solo   # writes results/_retail_native_SONNAT.js
#   -> run that .js with the Workflow tool (low concurrency; Sonnet throttles)
cd ../tau2-bench && .venv/bin/python <repo>/scripts/tau2_grade.py goldcheck 0 1 11   # gold -> 1.0
TAU2_DB_ONLY=1 .venv/bin/python <repo>/scripts/tau2_grade.py grade <traj.json>       # DB reward
```

## Caveats / gotchas

- **Throughput ceiling.** Both Workflow `agent()` and `claude -p` draw on the same Claude
  Code quota; high-volume Sonnet fusion runs hit `429 rate-limited` / `529 Overloaded`.
  Use low concurrency (≤4) and run sequentially.
- **ast_checker name normalization** (BFCL): function names are dot→underscore; emitted
  names must match or everything grades wrong.
- **Workflow subagents can't read the repo** and **can't be called from a Python loop** —
  so the real tau2 harness can't be driven by subagents; we reproduce its loop + score with
  its real evaluator instead.
- Recommended fusion-code change (validate on TR when credits return): swap the synth
  instruction at `quill-cloud-proxy enclave-go/cmd/enclave/fusion.go:1023` to the
  `evidence_decide` wording, paired with a **no-dominant-member** panel.
