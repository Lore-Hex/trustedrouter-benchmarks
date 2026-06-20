# Terminal-Bench 2.0 (agentic terminal/coding)

The widest discriminating range in the field (SOTA ~84.7% to ~3% floor) and the
cleanest per-model Chinese ranking on a fixed harness (GLM-5 > Kimi K2.5 >
DeepSeek-V3.2 > ... > Qwen3-Coder). Runs each task in a Docker sandbox with the
Terminus 2 agent.

Status: **harness documented, run pending infra** — needs Docker; agentic and
multi-turn, so a full run is expensive. Hold to a small task subset to stay
cheap. Upstream: https://github.com/laude-institute/terminal-bench

```bash
uv tool install terminal-bench          # provides the `tb` CLI; needs Docker running
export OPENAI_API_BASE="https://api.trustedrouter.com/v1"
export OPENAI_API_KEY="$TRUSTEDROUTER_API_KEY"     # throwaway key

# Via trbench (curated 10-task subset, terminus-2 agent, per-model accuracy):
python -m trbench.evals.terminal_bench.run --models z-ai/glm-5.2 --readme README.md
```

**Use the `terminus-2` agent, not `terminus-1`.** terminus-1 demands strict JSON
(`CommandBatchResponse`) output and fatally bails (`fatal_llm_parse_error`) on any
model that emits prose — deepseek-v4-flash scored 1/10 under terminus-1 purely from
parse failures, vs. real task work under terminus-2. `run.py` defaults to terminus-2.

A full V8/Terminal-Bench run on Apple Silicon is the usual blocker (some task
images are amd64-only); run from an amd64 Linux host or a runner where the
images are cached. Report accuracy on the fixed task subset.
