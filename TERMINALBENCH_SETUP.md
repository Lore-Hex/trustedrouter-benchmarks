# Terminal-Bench setup

The `terminal-bench/` tree is a local pinned checkout and is **gitignored**. It is too large to vendor directly:
the current working copy is about 866 MB with `.venv` and runs, and the clean task/source payload is still about
164 MB because several benchmark fixtures are large.

Reproduce it with the pinned upstream version below. Requires **Docker** and **uv**.

```bash
cd <repo root>
git clone https://github.com/laude-institute/terminal-bench.git terminal-bench
cd terminal-bench
git checkout 1a6ffa9674b571da0ed040c470cb40c4d85f9b9b
uv run tb --help              # resolves deps into terminal-bench/.venv (gitignored)
```

Dataset used: **terminal-bench-core==0.1.1** (the 80-task launch set; `tb run -d` pulls/pins it).

Our additions live outside this tree:
- `scripts/tb_haiku_agent.py` — `HaikuCliTerminus` (real Terminus-2 + Haiku via free `claude -p`).
- `scripts/tb_tr_agent.py` — `TRTerminus` for TrustedRouter-backed Terminal-Bench runs.
- `docs/terminalbench_results.md` — results + the 27.3% faithfulness caveat.
- `docs/handoff_data/terminalbench_haiku_smoke10.json` — the smoke replay.

Validate the harness after cloning (must print 100%):
```bash
cd terminal-bench && uv run tb run -d terminal-bench-core==0.1.1 -a oracle -t hello-world --no-livestream
```
Then run Haiku per `docs/terminalbench_results.md`.

For TrustedRouter runs, write output outside the vendored tree, for example `--output-path ../runs`.
