# Terminal-Bench vendor setup

The `terminal-bench/` tree is **gitignored** (785 MB of Docker task contexts + its own `.git`).
Reproduce it with the pinned upstream version below. Requires **Docker** and **uv**.

```bash
cd <repo root>
git clone https://github.com/laude-institute/terminal-bench.git terminal-bench
cd terminal-bench
git checkout 1a6ffa9          # v0.2.18 (pinned)
uv run tb --help              # resolves deps into terminal-bench/.venv (gitignored)
```

Dataset used: **terminal-bench-core==0.1.1** (the 80-task launch set; `tb run -d` pulls/pins it).

Our additions live OUTSIDE this tree (so vendoring stays clean):
- `scripts/tb_haiku_agent.py` — `HaikuCliTerminus` (real Terminus-2 + Haiku via free `claude -p`).
- `docs/terminalbench_results.md` — results + the 27.3% faithfulness caveat.
- `docs/handoff_data/terminalbench_haiku_smoke10.json` — the smoke replay.

Validate the harness after cloning (must print 100%):
```bash
cd terminal-bench && uv run tb run -d terminal-bench-core==0.1.1 -a oracle -t hello-world --no-livestream
```
Then run Haiku per `docs/terminalbench_results.md`.
