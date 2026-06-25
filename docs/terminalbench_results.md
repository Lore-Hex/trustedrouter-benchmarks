# Terminal-Bench — Haiku 4.5 via the free Claude-Code subscription

Terminal-Bench (laude-institute, ICLR 2026) evaluates an agent on real command-line tasks: each
task boots a Docker container with an initial state + a natural-language goal; the agent issues
shell commands over tmux, observes output, and a hidden test script decides pass/fail. We vendor
the **official** harness and grader and swap only the model backend.

## ⭐ TL;DR
- **Harness validated end-to-end** on the real Docker + official grader: `oracle` agent = **100%**
  on `hello-world` (uses the reference solution), our Haiku agent = **100%** on `hello-world`.
- On the same curated 10-task non-qemu subset of terminal-bench-core 0.1.1 (free `claude -p`):
  **Haiku 4.5 = 20% (2/10)**, **Sonnet 4.6 = 50% (5/10)**. Sonnet's passes are a strict superset
  of Haiku's. **3 of Sonnet's 5 failures are `agent_timeout`** (slower per-turn `claude -p`
  latency burns the task budget), so its true capability on this subset is likely > 50%.
- This is **"our harness," not the published 27.3%.** Artificial Analysis's *Terminal-Bench Hard*
  (Haiku 4.5 = **27.3%**) is a **different test**: a **secret 47-task subset** run with the real
  **Terminus-2 agent over a paid API**. We can't reproduce that figure exactly — same
  methodology-opacity lesson as SciCode's "43.3" (AA's subset + scaffold are undisclosed).

## What we built
- **Vendored** official Terminal-Bench **v0.2.18** (commit `1a6ffa9`), dataset
  **terminal-bench-core==0.1.1** (the 80-task launch set). See `TERMINALBENCH_SETUP.md` for the
  pinned clone (the `terminal-bench/` tree is gitignored — it's 785 MB of Docker task contexts).
- **`scripts/tb_haiku_agent.py`** — `HaikuCliTerminus`: the **real Terminus-2 scaffold** (prompt
  template, JSON parser, tmux execution, context management, official grader) with the LLM call
  swapped to the **free Claude-Code subscription**: each turn shells to
  `claude -p --model claude-haiku-4-5 --max-turns 1 --output-format text` with a `--system-prompt`
  override (strips Claude Code's default prompt), `--allowed-tools ""` (no tools), and a neutral
  cwd (no CLAUDE.md / auto-memory leakage). No API key, no $ — uses the CC quota.

## Smoke result (terminal-bench-core 0.1.1, 10 tasks, n_concurrent=3, max_episodes=40)
| task | Haiku 4.5 | Sonnet 4.6 | note |
|---|---|---|---|
| fix-permissions | ✅ | ✅ | |
| heterogeneous-dates | ✅ | ✅ | |
| simple-web-scraper | ❌ | ✅ | Sonnet only |
| openssl-selfsigned-cert | ❌ | ✅ | Sonnet only |
| csv-to-parquet | ❌ timeout | ✅ | Sonnet passed despite hitting the agent budget (work already done) |
| fix-git | ❌ | ❌ | tests failed |
| nginx-request-logging | ❌ | ❌ | tests failed |
| count-dataset-tokens | ❌ | ❌ timeout | Sonnet `agent_timeout` |
| sqlite-db-truncate | ❌ | ❌ timeout | Sonnet `agent_timeout` |
| password-recovery | ❌ timeout | ❌ timeout | both `agent_timeout` |
| **TOTAL** | **2/10 = 20.0%** | **5/10 = 50.0%** | Sonnet ⊃ Haiku passes |

Latency note: 3 of Sonnet's 5 failures are `agent_timeout` — the free `claude -p` per-turn overhead
(~3-5 s Haiku, more for Sonnet) eats the task's wall-clock budget. A faster (paid-API) backend
would let the same agent finish more of these, so both rates understate capability, Sonnet more so.

## Why 20% ≠ 27.3% (and why it's not apples-to-apples)
1. **Different scaffold.** AA uses Terminus-2 driving the model over a raw API. Our free route runs
   the model inside the `claude -p` Claude-Code runtime; even with the system prompt overridden and
   tools disabled, it is not a clean Terminus-2-over-API call. This is the dominant difference and
   is *by design* (the free path the run targeted).
2. **Latency → timeouts.** `claude -p` has ~3-5 s of CLI/startup overhead **per turn**; on
   multi-step tasks that burns the task's own wall-clock budget (`agent_timeout` on 2/10 here). A
   paid API call is much faster, so the same agent completes more tasks within budget.
3. **Different / curated task set.** AA's 47 "Hard" tasks are secret; we ran a **curated 10-task
   subset that excludes the heaviest qemu/kernel-build tasks**. Excluding the hardest tasks would
   *raise* our rate vs a fair full-set run — so on an even basis the gap to 27.3% is **larger**,
   not smaller, than 20% vs 27.3% suggests.
4. **Empty-turn episodes.** `claude -p` occasionally returns an empty turn (single-turn cap +
   no-tools); Terminus's retry absorbs most, but it costs episodes against the budget.

Primary-source context (reproducible targets, not AA's display metric): the Terminal-Bench 2.0
paper reports Haiku 4.5 ≈ **28.3% (Terminus-2)** / **29.8% (Mini-SWE)** — a different task-set
version, but a faithful API-driven scaffold.

## How to run
```bash
# one-time: see TERMINALBENCH_SETUP.md (pins v0.2.18 + dataset; needs Docker + uv)
cd terminal-bench
# validate the harness (must be 100%):
uv run tb run -d terminal-bench-core==0.1.1 -a oracle -t hello-world --no-livestream
# Haiku 4.5 via the free claude -p:
PYTHONPATH=../scripts uv run tb run -d terminal-bench-core==0.1.1 \
  --agent-import-path tb_haiku_agent:HaikuCliTerminus -m claude-haiku-4-5 \
  -t fix-permissions -t heterogeneous-dates ... \
  -k max_episodes=40 --n-concurrent 3 --no-livestream --output-path runs/_haiku
```
Env knobs for the agent: `TB_CLAUDE_BIN`, `TB_CLAUDE_CWD`, `TB_CLAUDE_TIMEOUT`.

## Open / next steps
- **Full run** (all ~80 core tasks) to get a representative rate — heavier (qemu/kernel images,
  longer wall-clock); decide after this smoke.
- **Faithful 27.3% attempt** would need the real Terminus-2 over an API (TrustedRouter — gated $ —
  or an Anthropic key) and ideally AA's exact 47-task subset (undisclosed).
- **Raise per-task budget** to isolate the latency-timeout effect from genuine capability.
