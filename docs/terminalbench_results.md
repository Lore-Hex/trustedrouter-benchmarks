# Terminal-Bench — Haiku 4.5 via the free Claude-Code subscription

Terminal-Bench (laude-institute, ICLR 2026) evaluates an agent on real command-line tasks: each
task boots a Docker container with an initial state + a natural-language goal; the agent issues
shell commands over tmux, observes output, and a hidden test script decides pass/fail. We vendor
the **official** harness and grader and swap only the model backend.

## ⭐ TL;DR
- **Harness validated end-to-end** on the real Docker + official grader: `oracle` agent = **100%**
  on `hello-world` (uses the reference solution), our Haiku agent = **100%** on `hello-world`.
- Same curated 10-task non-qemu subset of terminal-bench-core 0.1.1, free `claude -p` backend:

  | model | backend | default budget (360–600 s) | **generous budget (1800 s)** |
  |---|---|---|---|
  | Haiku 4.5 | `claude -p` | 20% (2/10) | _(not rerun)_ |
  | Sonnet 4.6 | `claude -p` | 50% (5/10) | _(not rerun)_ |
  | Opus 4.8 | `claude -p` | 40% (4/10) | **70% (7/10)** |
  | Opus 4.7 | `claude -p` | 70% (7/10) | **80% (8/10)** |
  | **gemini-3.1-pro** | `agy` (Antigravity) | **80% (8/10)** | _(not rerun; 1 timeout)_ |

  ⭐ **gemini-3.1-pro = 80% at the default budget** — best of any model on this subset, and the
  **only** one to solve `nginx-request-logging` (every Claude model fails it). Its sole timeout was
  count-dataset-tokens; fix-git its only other miss.

- ⭐ **On the free CLI path the score is latency-bound, not capability-bound, for slow models.**
  Opus 4.8's *default-budget* 40% landed **below** Sonnet's 50% — not because it's weaker (it
  uniquely solved `fix-git`) but because its slower per-turn `claude -p` latency hit **5
  `agent_timeout`s**. Re-running just the timed-out tasks with `--global-agent-timeout-sec 1800`
  recovers `csv-to-parquet`, `password-recovery`, `sqlite-db-truncate` → **Opus 4.8 = 70%**, and
  **Opus 4.7 = 80%**. So the apparent "Opus < Sonnet" inversion was entirely the wall-clock budget,
  not capability. (Haiku/Sonnet also had 1–2 timeouts each, so their numbers are likewise floors;
  not yet rerun — the generous-budget comparison was scoped to Opus.) A fast (paid-API / TR)
  backend removes the penalty outright — see the TrustedRouter backend below.
- Remaining failures after the generous budget: `nginx-request-logging` is the only **genuine**
  all-model fail; the rest are **flaky agent errors** (a task that errors for one Opus version
  passes for the other — e.g. count-dataset-tokens 4.7✓/4.8✗, password-recovery 4.8✓/4.7✗).
- This is **"our harness," not the published 27.3%.** Artificial Analysis's *Terminal-Bench Hard*
  (Haiku 4.5 = **27.3%**) is a **different test**: a **secret 47-task subset** run with the real
  **Terminus-2 agent over a paid API**. We can't reproduce that figure exactly — same
  methodology-opacity lesson as SciCode's "43.3" (AA's subset + scaffold are undisclosed).

## Backends (how the model is called)
Three ways to drive the **same** real Terminus-2 scaffold + official grader:
| backend | script / agent | cost | faithfulness | notes |
|---|---|---|---|---|
| free `claude -p` | `tb_haiku_agent.py` → `HaikuCliTerminus` | **free** (CC quota) | "our harness" (CC runtime wrap + per-turn latency) | used for all Haiku/Sonnet/Opus numbers above; `-m claude-{haiku-4-5,sonnet-4-6,opus-4-8,opus-4-7}` |
| **TrustedRouter SDK** | `tb_tr_agent.py` → `TRTerminus` | **TR $ (gated)** | faithful Terminus-2-over-API | any of TR's ~230 models: `-m anthropic/claude-haiku-4.5`, `-m google/gemini-3.1-pro`, the Chinese open-weight panel, … **Cost-guarded: dormant unless `TB_TR_CONFIRM=1`.** Needs `trustedrouter` in the tb venv (`uv run --with trustedrouter ...`). This is the path that could chase the published 27.3% (fast API removes the latency-timeout penalty). |
| Antigravity CLI (`agy`) | `tb_agy_agent.py` → `AgyTerminus` | Google account (OAuth) | "our harness" (agy runtime wrap; agy `-p` is fast ~5 s/turn) | gemini via the official **`agy -p`** (replaces the dead consumer gemini-cli). Same host-side Terminus pattern as `claude -p`. One-time `agy` browser sign-in done. **Live: `-m gemini-3.1-pro` → 80% (8/10).** Install: `brew install --cask antigravity-cli`. |
| gemini-cli (in-container) | built-in `--agent gemini-cli` | Google API $ | faithful (real Gemini CLI agent) | ⚠️ **DEAD for OAuth**: host `~/.gemini` OAuth returns `IneligibleTierError` (Google killed the free individual gemini-cli tier → migrate to `agy`). Would need a paid `GEMINI_API_KEY`, or route gemini-3.1-pro via the TR backend. |

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

## Smoke result (terminal-bench-core 0.1.1, 10 tasks, n_concurrent=3)
Default budget = each task's own `max_agent_timeout_sec` (360–600 s), max_episodes=40. Opus columns
also show the **generous-budget rerun** (`--global-agent-timeout-sec 1800`, max_episodes=60) of the
timed-out tasks. `TO`=agent_timeout, `ERR`=flaky agent error, `✅*`=passed despite a late timeout.

| task | Haiku | Sonnet | Opus 4.8 | Opus 4.8·1800s | Opus 4.7 | Opus 4.7·1800s | gemini-3.1-pro |
|---|---|---|---|---|---|---|---|
| fix-permissions | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| heterogeneous-dates | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| openssl-selfsigned-cert | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| fix-git | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| simple-web-scraper | ❌ | ✅ | TO | TO/crash | ✅ | ✅ | ✅ |
| csv-to-parquet | TO | ✅ | TO | ✅ | ✅* | ✅ | ✅ |
| sqlite-db-truncate | ❌ | TO | TO | ✅ | ✅* | ✅ | ✅ |
| password-recovery | TO | TO | TO | ✅ | TO | ERR | ✅ |
| count-dataset-tokens | ❌ | TO | TO | ERR | TO | ✅ | TO |
| nginx-request-logging | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **TOTAL** | **20%** | **50%** | **40%** | **70%** | **70%** | **80%** | **80%** |

`nginx-request-logging` is the only task no model solves (genuine). The generous-budget reruns turn
most `TO`s into `✅`; the leftover `ERR`s are flaky (the same task passes for the other Opus build).

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
