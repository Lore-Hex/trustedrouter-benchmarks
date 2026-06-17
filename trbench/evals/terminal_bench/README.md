# Terminal-Bench 2.0 (agentic terminal/coding)

The widest discriminating range in the field (SOTA ~84.7% to ~3% floor) and the
cleanest per-model Chinese ranking on a fixed harness (GLM-5 > Kimi K2.5 >
DeepSeek-V3.2 > ... > Qwen3-Coder). Runs each task in a Docker sandbox with the
Terminus 2 agent.

Status: **harness documented, run pending infra** — needs Docker; agentic and
multi-turn, so a full run is expensive. Hold to a small task subset to stay
cheap. Upstream: https://github.com/laude-institute/terminal-bench

```bash
pip install terminal-bench
export OPENAI_API_BASE="https://api.trustedrouter.com/v1"
export OPENAI_API_KEY="$TRUSTEDROUTER_API_KEY"     # throwaway key
tb run --agent terminus --model openai/z-ai/glm-5.1 \
  --dataset terminal-bench-core==2.0 --n-tasks 10
```

A full V8/Terminal-Bench run on Apple Silicon is the usual blocker (some task
images are amd64-only); run from an amd64 Linux host or a runner where the
images are cached. Report accuracy on the fixed task subset.
