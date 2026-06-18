# tau2-bench (agentic tool-use)

Dual-control conversational agents across 5 domains (sierra-research/tau2-bench,
MIT). The faithful move is to drive the upstream harness pointed at TrustedRouter,
not reimplement the user-simulator + tools — so `run.py` subprocesses the real
`tau2` CLI once per panel model and reads its canonical per-task `reward`.

Status: **validated + calibrated.** gpt-4.1 on retail (20 tasks, 1 trial) →
pass^1 **80.0**, 0 infrastructure errors (published full-retail is ~75.1). Full
panel pending.

## Setup (one-time)

tau2-bench needs Python 3.12 and heavy deps (litellm + framework), so install it
in its OWN venv outside this repo; `run.py` shells out to its CLI.

```bash
git clone --depth 1 https://github.com/sierra-research/tau2-bench ../tau2-bench
cd ../tau2-bench && python3.12 -m venv .venv && .venv/bin/pip install -e .
```

Point `--tau2-home` (or `$TAU2_HOME`) at it; default is `../tau2-bench`.

## Run

```bash
export TRUSTEDROUTER_API_KEY=...      # throwaway key; capability prompts route upstream
python -m trbench.evals.tau2.run --models openai/gpt-4.1 --domain retail \
  --num-tasks 20 --num-trials 1
```

`run.py` handles the gateway plumbing:

- **LiteLLM → TR routing.** A TR model id like `z-ai/glm-5` becomes the LiteLLM
  name `openai/z-ai/glm-5` (the `openai/` prefix selects the OpenAI-compatible
  provider; the rest is sent verbatim), with `OPENAI_API_BASE`/`OPENAI_API_KEY`
  pointed at TR. The simulated user (`--user-llm`) and the NL-assertion judge
  (`--eval-llm`) are fixed strong models so only the agent under test varies.
- **The litellm-aliasing shim** (`_shim/sitecustomize.py`) is forced onto the
  subprocess `PYTHONPATH`. tau2 hardcodes its NL-assertion judge to OpenAI's
  dated `gpt-4.1-2025-04-14`; LiteLLM rewrites that on the way out and the gateway
  rejects it, aborting any task with an NL assertion as an `infrastructure_error`.
  The shim's `model_alias_map` reroutes the dated name back through TR. (The TR
  gateway also now accepts bare/dated OpenAI ids directly, so this is belt-and-
  suspenders for a stricter gateway / offline reproduction.)
- **Clean runs.** tau2 auto-resumes from `data/simulations/<slug>/`; `run.py`
  clears it first (use `--resume` to keep it) so a re-run actually re-runs.

Metric: per-task `reward` (0–1) → `avg_reward` + `pass^k` (with `--num-trials k`).
Infrastructure-error sims are excluded from the score, matching tau2's own metric,
and surfaced in the `Errors` column. Each task is a full multi-turn conversation
(agent + user, both LLMs, plus tool calls), so a full run is expensive — hold to a
small `--num-tasks` per domain.
