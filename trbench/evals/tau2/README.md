# tau2-bench (agentic tool-use)

Dual-control conversational agents across 5 domains (MIT). The faithful move is
to drive the upstream harness pointed at TrustedRouter, not reimplement the
user-simulator + tools.

Status: **harness documented, run pending infra** — agentic multi-turn; keep it
cheap with `--num-tasks`. Upstream: https://github.com/sierra-research/tau2-bench

```bash
pip install tau2-bench
export OPENAI_API_BASE="https://api.trustedrouter.com/v1"
export OPENAI_API_KEY="$TRUSTEDROUTER_API_KEY"     # throwaway key
tau2 run --domain airline \
  --agent-llm openai/z-ai/glm-5.1 \
  --user-llm openai/google/gemini-2.5-flash \
  --num-trials 1 --num-tasks 5
```

Each task is a full multi-turn conversation (agent + simulated user, both LLMs,
plus tool calls) — a full run is expensive, so hold to a small `--num-tasks`
subset per domain and report per-domain pass rate.
