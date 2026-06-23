# Agentic fusion v3 — read/write-asymmetric exploration (PLAN)

## Why the first two approaches were wrong
- **Per-step fusion** (panel→judge→synth at every turn): blends 5 proposals into ONE
  committed path. An early wrong pick (e.g. the `#`-order-id spiral, wrong `item_ids`)
  dooms the whole episode — errors compound. Result: Sonnet fusion ≈ solo (50 vs 56).
- **Best-of-N full rollouts**: would capture the trajectory-level oracle, BUT agents run
  for *hours* — N×hours is impractical.

## The key data
Two independent Sonnet **solo** runs on tau2 retail (real evaluator, DB reward):
solo 70% & 75%, but they **disagree on 7/20 tasks** and the **2-run oracle = 90%**.
So there's ~15 pts of latent headroom even from one model's stochasticity. And the lost
episodes are **under-informed / premature commits** (wrong ids, wrong payment method,
giving up after a recoverable read error), not "couldn't choose between two good paths."

## The design: explore broadly on reads, deliberate (fuse) only on writes
The two action classes have opposite cost/risk, so treat them asymmetrically:
- **READ / investigative** — non-mutating, reversible, parallel-safe. Fan OUT: let the
  agent request many reads at once (all the user's orders + every candidate variant),
  gather everything cheaply, reach more state automatically. (tau2's env-replay even
  *skips* reads, so they're free for scoring.)
- **WRITE / mutating** — irreversible, usually terminal. This is the ONLY place to spend
  the expensive fusion: when the agent proposes a write, run a small `evidence_decide`
  panel that independently re-derives the correct write from the gathered data + policy,
  take the self-consistent majority, and commit only that (else gather more).

So per step the agent emits a SET: `{reads:[...], write:?, message:?}`. Reads run in
parallel (explore); a write triggers the verify-panel (commit); message → user turn.

tau2 already types tools `ToolType.READ` / `ToolType.WRITE` (`_is_mutating_tool`), so the
gate is exact. Fusion is concentrated at the few irreversible decisions; everything else
is cheap automatic exploration. No N-rollouts, no diverse-panel/TR dependency.

## Eval
Grade with tau2's REAL evaluator (`scripts/tau2_grade.py`, `TAU2_DB_ONLY=1`) vs:
solo 75% and the 2-run oracle 90%. Target: convert the oracle headroom into a single
deliverable trajectory by committing well-informed, verified writes.

## Build / run
- `scripts/gen_retail_explore.py` — generates the explore workflow (read fan-out +
  write-fusion), `python scripts/gen_retail_explore.py "<ids>" "_EXP" sonnet`.
- Needs model calls → run after the Claude-Code session quota reset (was 11:20pm PT).
- Start with a 2–3 task smoke to confirm the loop works, then the full 20.

## Smoke result (tasks 0,1,2 — the hardest, all-prior-approaches-failed)
| approach | reward |
|---|---|
| solo run1 / run2 | 33% / 33% (1/3) |
| per-step fusion | 33% (1/3) |
| 2-run oracle (ceiling) | 67% (2/3) |
| **EXPLORE v3** | **67% (2/3) — hits the oracle, 2× the baselines** |

Mechanism confirmed: task 1 & 2 recovered by fanning out all orders in parallel →
complete info → verified commit. Task 0 still lost the `#`-spiral (fan-out helps but
doesn't guarantee recovery; stochastic). n=3, but a clean 2× on the exact failures.
Next: full 20 vs solo 75% and oracle 90%.
