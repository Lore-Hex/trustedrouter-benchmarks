# SciCode (vendored)

Vendored from [scicode-bench/SciCode](https://github.com/scicode-bench/SciCode) (Apache 2.0)
to run the Haiku-4.5 / agentic SciCode baselines self-contained in this repo. Source is
checked in; the env, the 1.05 GB numeric ground truth, and generated artifacts are gitignored.

## One-time setup (from a fresh clone)
```bash
cd scicode
python -m venv .venv
.venv/bin/pip install -e . gdown
# numeric test ground truth (1.05 GB) -> eval/data/test_data.h5 (gitignored)
.venv/bin/gdown --folder "https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR" -O eval/data
```
The problem set itself loads from HuggingFace (`SciCode1/SciCode`, splits `validation`=15 /
`test`=65) at run time — no separate download.

## Run a Haiku-4.5 baseline (without-background = the realistic leaderboard setting)
From the repo root:
```bash
# 1) generate the subagent generation-chain workflow (faithful to gencode.py)
python scripts/scicode_gen.py test _TEST haiku 12        # -> results/_scicode_TEST.js
# 2) run that .js with the Workflow tool; save the result JSON to results/_scicode_TEST_out.json
# 3) reconstruct files + score with SciCode's OWN test runner
SCICODE_TIMEOUT=120 python scripts/scicode_score.py results/_scicode_TEST_out.json haiku test
```
- `scripts/scicode_gen.py` / `scripts/scicode_score.py` find this dir via `SCICODE_HOME`
  (default: `<repo>/scicode`).
- `SCICODE_TIMEOUT` is the per-step subprocess limit (default 120 s; official upstream is
  1800 s — lowered so buggy/infinite-loop generations fail fast instead of stalling the
  sequential runner; correct steps finish in seconds).

## Metrics
Test split denominators: **65 problems / 288 steps** (3 steps — 13.6, 62.1, 76.3 — are
skipped and seeded from gold, per upstream). Reported as subproblem rate and main-problem
(all-subs-pass) rate.
