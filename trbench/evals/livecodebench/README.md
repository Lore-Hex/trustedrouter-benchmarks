# LiveCodeBench (contamination-proof coding)

Competition problems tagged with release dates; `--start_date`/`--end_date`
filtering scores models only on problems released AFTER their training cutoff —
the one eval here with a built-in contamination guard.

Status: **harness documented, run pending infra** — needs code execution
(stdin/stdout test cases). Upstream: https://github.com/LiveCodeBench/LiveCodeBench

```bash
pip install livecodebench   # or clone the repo
# generation pointed at TrustedRouter (OpenAI-compatible), then evaluate.
# Use a recent post-cutoff window to keep it both cheap and contamination-free:
python -m lcb_runner.runner.main \
  --model z-ai/glm-5.1 \
  --base_url https://api.trustedrouter.com/v1 \
  --scenario codegeneration \
  --start_date 2025-06-01 --end_date 2025-12-31
```

Report pass@1 on the date-filtered subset. Because problems are post-cutoff,
this stays discriminating even when static coding evals saturate.
