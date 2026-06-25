# Terminal-Bench — Haiku 4.5 via the free Claude-Code subscription

Terminal-Bench (laude-institute, ICLR 2026) evaluates an agent on real command-line tasks: each
task boots a Docker container with an initial state + a natural-language goal; the agent issues
shell commands over tmux, observes output, and a hidden test script decides pass/fail. We vendor
the **official** harness and grader and swap only the model backend.

## ⭐ TL;DR
- **Harness validated end-to-end** on the real Docker + official grader: `oracle` agent = **100%**
  on `hello-world` (uses the reference solution), our Haiku agent = **100%** on `hello-world`.
- **Haiku 4.5 (free `claude -p`) smoke = 20% (2/10)** on a curated non-qemu subset of
  terminal-bench-core 0.1.1. PASS: `fix-permissions`, `heterogeneous-dates`.
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
| task | result | note |
|---|---|---|
| fix-permissions | ✅ PASS | |
| heterogeneous-dates | ✅ PASS | |
| fix-git | ❌ fail | tests failed |
| nginx-request-logging | ❌ fail | tests failed |
| count-dataset-tokens | ❌ fail | tests failed |
| simple-web-scraper | ❌ fail | tests failed |
| openssl-selfsigned-cert | ❌ fail | tests failed |
| sqlite-db-truncate | ❌ fail | tests failed |
| csv-to-parquet | ❌ fail | **agent_timeout** (hit the task's 360 s budget) |
| password-recovery | ❌ fail | **agent_timeout** |
| **TOTAL** | **2/10 = 20.0%** | |

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
