# HANDOFF — Fusion synthesizer research + agentic fusion (v3)

Self-contained handoff. All runs were **all-Haiku / all-Sonnet via Claude Code subagents**
(TR credits were out), orchestrated with the Workflow tool, and graded with tau2's own
evaluator. Companion docs: `fusion_agentic_findings.md`, `agentic_fusion_explore_plan.md`.

## TL;DR results
- **Single-shot fusion (BFCL/GSM8K):** `evidence_decide` is the best synthesizer wording
  (+9 on decidable items vs the shipped baseline). Fusion beats best-solo **only with a
  no-dominant-member, diverse panel** (different models). Method: decidable-focused (panel
  → grade → keep disagreement items → sweep synth prompts on those).
- **Agentic fusion (tau2 retail, REAL evaluator, DB reward):**
  - **Sonnet solo = 70–75%** (near-identical harness) — leaderboard band (gpt-4.1 74.1,
    o4-mini 71.5, claude-3.7-sonnet 78.7). The 45%→72% jump was harness-looseness (loose
    JSON tool-calls vs schema/native + verbatim prompts), NOT the model.
  - **Per-step fusion ≈ solo** at Sonnet tier (50 vs 56 on the hard 8) — the 5 "stances"
    were one model → they agree → no diversity. Haiku stances diverged → +10.
  - **The headroom is real & at the trajectory level:** two independent Sonnet *solo* runs
    disagree on 7/20 tasks; **2-run oracle = 90%** vs solo 75%. Even one model's
    stochasticity has ~15 pts of latent headroom.
- **v3 = read/write-asymmetric "explore" (THE direction):** explore broadly on reversible
  READS (fan out in parallel), spend fusion only at irreversible WRITES. **Smoke (tasks
  0,1,2, the hardest): 67% (2/3) — hit the 2-run oracle, 2× solo/per-step-fusion (33%).**

## v3 design (validated on smoke; see gen_retail_explore.py)
Per step the agent emits `{reads:[...], write:?, message:?}`:
- `reads` → run ALL in parallel (cheap, reversible — gather complete info, reach more state)
- `write` → fusion gate: N candidate writes + self-consistency/verify, commit the agreed one
- `message` → reply to user
Failures were *premature/under-informed commits* (wrong item_ids, `#`-order-id spiral, wrong
payment), not "couldn't choose between two good paths" — so gather-then-verified-commit wins.

## Putting v3 into `trustedrouter/fusion` (the product design — CORRECTED)
Goal: **drop-in for any harness (opencode unmodified) — a standard OpenAI chat-completions
call, no new params, no client changes.** The synthesizer self-classifies read vs write.

**The panel runs every step. The asymmetry is in how proposals are AGGREGATED, forced by
reversibility — NOT in whether the panel runs.** Per call:
1. Panel (diverse proposers) propose the next action(s) for the current state.
2. **READ proposals → UNION.** Return the union of all distinct read tool_calls the panel
   proposed; the client runs them in PARALLEL. Diversity = breadth of exploration (different
   members look up different things) → more complete info. This is where the oracle headroom
   lives (the 2 solo runs hit 90% by exploring *differently* — union recovers that). Reads
   are reversible/additive, so taking many never hurts.
3. **WRITE → the synth selects ONE and returns it.** You can't union irreversible writes;
   the synth picks the single action, informed by the now-complete reads. **No N-redundancy**
   — once exploration is broad the write is basically determined. The synth decided → write it.

So: **same panel both times; UNION the reads, SELECT the write.** Each fusion call returns
either a *batch of read tool_calls* (union) or a *single write tool_call* (synth's pick); the
client just executes the returned tool_calls (parallel for reads).

Key points the design must honor (from review):
- **No client changes / no new params** — standard OpenAI call; the synth infers read/write
  from tool semantics (`get_/list_/search_/read_/grep` = read; `create_/update_/delete_/
  cancel_/edit_/commit_/send_` = write). NOT a `mutating_tools` param.
- **The panel's value on reads is BREADTH (union), not redundancy.** Panel calls on read
  steps are NOT wasted — generating diverse read directions IS the value.
- **The write is a single synth decision — NOT N candidates.** (Earlier "N candidate writes
  + verify" was over-engineering; the headroom comes from broad reads, not write-redundancy.)
- NOTE: the v3 smoke used a *single proposer* that emitted read-batches + an N-verify write.
  The refined design above (panel-union-reads / synth-select-write) is what to implement & run
  next — expect it to capture *more* of the oracle (wider read union) at *lower* write cost.

Where in code: `quill-cloud-proxy enclave-go/cmd/enclave/fusion.go`. Today panel → judge →
synth (`fusionFinalRequest`, tool branch at ~line 1023). Change: synth instruction →
`evidence_decide` + "emit all needed reads at once; only emit a mutating call when the data
supports it"; add a `runFusionWriteVerify` stage that fires only when the chosen tool is
mutating (N re-derivations + majority). Gate deploy on Joseph's go (NOT deployed).

## Open questions / next steps
1. **Full-20 explore run** (vs solo 75% and oracle 90%) — quota-limited; `python
   scripts/gen_retail_explore.py "0..19" "_EXP20" sonnet` → run the `.js` via Workflow →
   grade.
2. **Diverse open-weight panel** (the genuine fusion>solo demo) — needs TR credits back;
   Claude models are capability-ordered so a Claude panel always has a dominant member.
3. **opencode validation** — port the read/write loop with test-execution as the write
   verifier; measure lift (expect bigger lift on mid models than frontier).
4. **Reliable read/write self-classification** — the name heuristic is good-enough; confirm
   the synth can self-classify robustly with no annotation.

## Repro / scripts (in trustedrouter-benchmarks)
- `scripts/tau2_grade.py` — score a trajectory with tau2's REAL evaluator (run in tau2 venv:
  `cd ../tau2-bench && .venv/bin/python <repo>/scripts/tau2_grade.py goldcheck 0`; then
  `TAU2_DB_ONLY=1 ... grade <traj.json>`). Validated: gold actions → reward 1.0.
- `scripts/gen_retail_native.py` — near-identical solo/fusion harness (verbatim prompts +
  schema tool-calls).
- `scripts/gen_retail_explore.py` — v3 read/write-asymmetric explore harness.
- `scripts/gen_retail_fusion.py`, `fusion_*.py`, `phase*_*.py`, `grade_*.py` — single-shot
  synth-prompt research + agentic-fusion grading.
- Result trajectories live (gitignored) under `results/_*`; key ones copied to
  `docs/handoff_data/` for the record.
- Constraints: Workflow `agent()` and `claude -p` share the Claude-Code quota → 429/529/
  session-limit on high-volume runs; run low-concurrency. Subagents can't read the repo or
  be called from a Python loop (so the real tau2 orchestrator can't be subagent-driven — we
  reproduce its loop + score with its real evaluator).
