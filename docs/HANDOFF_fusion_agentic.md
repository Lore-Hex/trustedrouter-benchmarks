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

## v3 smoke harness (gen_retail_explore.py) — what was actually run
A SINGLE proposer emits `{reads:[...], write:?, message:?}` each step: reads fan out in
parallel; the write went through a small verify (the shippable design DROPS that — see below).
Smoke 2/3 = oracle. Failures were *premature/under-informed commits* (wrong item_ids,
`#`-order-id spiral, wrong payment), not "couldn't choose between two good paths" — so
gather-broadly-then-commit wins. The shippable form replaces the single proposer with the
panel→synth-decides structure (next section); no write-verify, no code classifier.

## Putting v3 into `trustedrouter/fusion` (the product design — CORRECTED)
Goal: **drop-in for any harness (opencode unmodified) — a standard OpenAI chat-completions
call, no new params, no client changes.** The synthesizer self-classifies read vs write.

**The synthesizer decides everything, every step. No read/write logic in code; no
"union-on-any-read" rule.** Structure = panel proposes diverse candidate next-moves →
**synth decides the single next move**:
1. Panel (diverse proposers) propose candidate next actions (ideas, some wrong).
2. **Synth decides** (one judgment, informed by the panel + gathered data):
   - *still investigating* → return the read/lookup tool call(s) it wants now (it may run
     several in parallel — the panel's diversity widens what it thinks to look up); OR
   - *enough info* → return the single state-changing call, exactly correct (ids/values
     from gathered data); OR
   - return a message.
3. The synth — an LLM — classifies read/write implicitly (by what it chooses) and judges
   *when to stop exploring and commit*. The panel only feeds breadth; the synth makes the call.

Each fusion call returns a normal OpenAI `tool_calls` response (a batch of reads, or one
write, or a message). Client executes whatever comes back (parallel for reads).

Design rules (from review — DO NOT regress):
- **NO code heuristic for read vs write** (no `isMutating()` string match — brittle, breaks on
  odd tool names). The synth (LLM) decides, far more flexibly. No client annotation/param.
- **The synth decides read-vs-write AND when to commit** — not a code rule like "union if any
  panel member proposed a read" (that loops forever). It's the synth's per-step judgment.
- **The panel's value is breadth of ideas** (it widens what the synth investigates / which
  write it considers); the synth converges to one move.
- **No N-candidate write redundancy** — the synth decides the write once; the headroom comes
  from broad *exploration*, not write-resampling.
- Standard OpenAI chat-completions call → drop-in for opencode unmodified.
- NOTE: the v3 smoke used a SINGLE proposer (no panel) emitting {reads,write,message}, and it
  worked (2/3 = oracle). The shippable form adds the panel feeding the synth's decision. Run
  that on the full 20 + the diverse open-weight panel before touching the enclave.

Where in code: `quill-cloud-proxy enclave-go/cmd/enclave/fusion.go`. Today panel → judge →
synth (`runFusionFinal`/`fusionFinalRequest`, tool branch at ~line 1023). The agentic version
barely touches the Go: **the synth's `tool_calls` response is returned to the client AS-IS —
no classifier, no read/write partition, no extra stage.** The ONLY real change is the SYNTH
PROMPT (the decision wording) + allowing the synth to emit MULTIPLE read tool_calls in one
response (parallel reads):

    <policy + tools with descriptions>
    Conversation + data gathered so far: <transcript>
    Other assistants proposed these next actions (ideas, some may be wrong): <panel proposals>
    Decide the SINGLE next move:
      - need info -> return the lookup/read tool call(s) to run now (several at once is fine);
      - enough    -> return the one state-changing call, exactly correct (ids from the data);
      - otherwise -> return a message.
    Don't keep investigating once you can correctly act.

The judge stays as-is. NO `isMutating()` heuristic, NO write-verify stage — the synth decides.
Gate deploy on Joseph's go (NOT deployed).

## Open questions / next steps
1. **Full-20 explore run** (vs solo 75% and oracle 90%) — quota-limited; `python
   scripts/gen_retail_explore.py "0..19" "_EXP20" sonnet` → run the `.js` via Workflow →
   grade.
2. **Diverse open-weight panel** (the genuine fusion>solo demo) — needs TR credits back;
   Claude models are capability-ordered so a Claude panel always has a dominant member.
3. **opencode validation** — port the read/write loop with test-execution as the write
   verifier; measure lift (expect bigger lift on mid models than frontier).
4. **Synth self-classification** — confirm the synth reliably decides read-vs-write AND when
   to stop exploring and commit, with NO code heuristic and NO client annotation (the brittle
   `isMutating()` string match is dropped — the LLM judges it).

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
