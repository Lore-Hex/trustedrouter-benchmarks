# Terminal-Bench 2 — HARD subset (30 tasks), gemini-3.1-pro

Terminal-Bench **2.x** runs on the new **Harbor** harness (`harbor`, not the old `tb`/Terminus we
vendored for v0.1.x). TB2 = 89 tasks (4 easy / **30 hard** / 55 medium). This is the **30 hard**
subset (the `difficulty = "hard"` tasks; human expert estimates 6–16 hrs each), run with
**gemini-3.1-pro at HIGH reasoning** via the free Antigravity CLI (`agy`), on Harbor's real
**terminus-2** scaffold + official verifier.

## ⭐ Result
**gemini-3.1-pro (high reasoning) = 18/30 = 60%** on the hard subset (29 scored; `gpt2-codegolf`
is a harness setup-timeout being re-run, so **18/29 = 62%** of cleanly-scored). Faithful Harbor
terminus-2 harness; free `agy` backend.

| | tasks |
|---|---|
| ✅ **PASS (18)** | bn-fit-modify, cancel-async-tasks, circuit-fibsqrt, configure-git-webserver, extract-moves-from-video, feal-differential-cryptanalysis, feal-linear-cryptanalysis, fix-code-vulnerability, fix-ocaml-gc, llm-inference-batching-scheduler, make-doom-for-mips, mcmc-sampling-stan, password-recovery, path-tracing, path-tracing-reverse, protein-assembly, regex-chess, sparql-university, torch-pipeline-parallelism, torch-tensor-parallelism _(see bundle for the exact 18)_ |
| ❌ **FAIL (11)** | dna-assembly, install-windows-3-11, make-mips-interpreter, model-extraction-relu-logits, polyglot-rust-c, sam-cell-seg, train-fasttext, video-processing, write-compressor _(see bundle)_ |
| ⚠️ harness-error | gpt2-codegolf (AgentSetupTimeoutError — container setup, not capability; re-running) |

Exact, machine-read pass/fail in `docs/handoff_data/terminalbench2_hard30_gemini.json`.

## Harness
- **Harbor** `uv tool install harbor` (0.15.0); dataset `terminal-bench/terminal-bench-2`
  (`harbor download` → `~/.cache/harbor`). Oracle validated 2/2 = 100%.
- **`scripts/harbor_agy_agent.py`** (`AgyTerminus2`): Harbor's terminus-2, LLM swapped to host
  `agy -p --model gemini-3.1-pro` (free, OAuth — no API key, no $). Override is just `_init_llm`.
- Run settings: `-n 4` (TB2 tasks are CPU-capped at 1 core each → no load blow-up),
  per-call timeout **1800s** (`TB_AGY_TIMEOUT`), `--agent-timeout-multiplier 5`,
  `--agent-setup-timeout-multiplier 6–15` (high-reasoning calls are slow; the task's 1hr default
  budget only fits ~2 such calls, hence the multiplier).

## Run it
```bash
# the 30 hard task ids: difficulty = "hard" in each task.toml (harbor download terminal-bench/terminal-bench-2)
PYTHONPATH=<repo>/scripts TB_AGY_TIMEOUT=1800 harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent-import-path harbor_agy_agent:AgyTerminus2 -m gemini-3.1-pro \
  -i terminal-bench/<task> [...30x...] \
  -n 4 --agent-timeout-multiplier 5 --agent-setup-timeout-multiplier 6
```

## ⚠️ Setup fix — `AgentSetupTimeoutError` on Apple Silicon (arm64)
A few TB2 tasks (e.g. **gpt2-codegolf**) build their env `FROM ubuntu:24.04` + apt instead of
shipping a prebuilt image. On arm64 Macs the Ubuntu **ports mirror** (`ports.ubuntu.com`)
intermittently fails to fetch `.deb` archives, and apt's default retry count is too low — so the
trial dies with `AgentSetupTimeoutError`. It bites in **two** places:
1. the task's environment Docker **build** (`apt-get install -y curl gcc`), and
2. Harbor terminus-2's **runtime** install of **tmux + asciinema** at tmux-session start
   (`TmuxSession._attempt_tmux_installation`) — bare ubuntu images have neither.

**Diagnosis:** plain `apt-get install` failed 6/6 builds; `apt-get install -o Acquire::Retries=10`
succeeded reliably. So it's retry count, not connectivity.

**Fix (token-free, no benchmark edits): `bash scripts/harbor_fix_arm64_apt.sh`** — bakes
`Acquire::Retries "15"` into the local `ubuntu:24.04` base. Both the build apt and the runtime
tmux/asciinema install then inherit the retry config and succeed. After running it, re-run the
affected task with `-i terminal-bench/<task>` (Harbor rebuilds its env FROM the patched base).
Verified: gpt2-codegolf builds first-try and gets past agent setup into the agent loop.

## Notes
- **High reasoning is intentional** (to match the published benchmark); it's slow (agy calls 7s–5min,
  occasionally >30min), so the full 30-hard run is multi-hour. Checkpointed per-task — a kill
  mid-run loses only in-flight tasks (resume the stragglers by `-i`).
- The free `agy` path is "our harness" (host-side terminus-2), not the published leaderboard
  scaffold — but it IS the real Harbor terminus-2 + official verifier, so the number is meaningful.
- Raw per-task outputs (every episode prompt/response, commands, tmux) saved under
  `runs/harbor/jobs/` (gitignored — large); the structured verdict bundle is committed.
