# Terminal-Bench hard-short experiment log

Date: 2026-06-27

## Scope

We selected six short-duration Terminal-Bench v2.1 tasks as an initial agentic
coding/shell benchmark slice:

- `cancel-async-tasks`
- `fix-code-vulnerability`
- `polyglot-rust-c`
- `configure-git-webserver`
- `feal-linear-cryptanalysis`
- `password-recovery`

This is an agentic smoke test, not a stable model ranking. It is useful for
testing whether our wrappers, timeouts, Docker setup, and model-specific
behavior can handle realistic terminal tasks.

## Saved artifacts

- Task subset manifest: `results/terminal_bench_hard_short_manifest.json`
- Completed scored rows: `results/terminal_bench_hard_short_11models.jsonl`
- Local dataset export: `.data/terminal-bench-hard-short/tasks`
- Raw run directory: `runs/terminal_bench_hard_short_11models`

## Completed scores

| Model | Score | Correct tasks | Missed tasks | Input tokens | Output tokens |
|---|---:|---|---|---:|---:|
| GPT-5.5 | 5/6 | cancel async, configure git webserver, FEAL cryptanalysis, fix code vulnerability, password recovery | polyglot Rust/C | 295,342 | 14,984 |
| Gemini 3.1 Pro Preview | 5/6 | cancel async, configure git webserver, FEAL cryptanalysis, fix code vulnerability, password recovery | polyglot Rust/C | 443,750 | 22,069 |
| Claude Opus 4.8 | 3/6 | FEAL cryptanalysis, fix code vulnerability, password recovery | cancel async, configure git webserver, polyglot Rust/C | 636,892 | 33,673 |

## Interrupted / partial rows

- Grok 4.3 was skipped after repeated long, unproductive attempts. It got stuck
  on password recovery and then followed wrong paths on fix-code vulnerability
  and FEAL. No scored row should be treated as final.
- Kimi K2.6 was interrupted to control cost/time. It had one recorded failed
  task, `configure-git-webserver`, after 273,132 input tokens and 6,121 output
  tokens. It was still running FEAL and polyglot when stopped.

## Lessons

- Docker and the local Terminal-Bench subset now work.
- GPT-5.5 and Gemini 3.1 Pro both solved the same 5/6 tasks and both missed
  `polyglot-rust-c`.
- Opus 4.8 was much more token-hungry and materially worse on this slice.
- Kimi and Grok need stricter per-task controls before running a full sweep.
- The current wrapper-level timeout did not prevent long-running model/task
  combinations from consuming too much budget. Future runs should add explicit
  per-task wall-clock limits and a max-call or max-token budget inside the
  Terminal-Bench wrapper.

## Recommended next run

Run the remaining open models one at a time with:

- `--n-concurrent 1`
- one task at a time, with resume enabled
- an explicit per-task timeout lower than the current model-level timeout
- immediate JSONL append after every task, not only after every model
- a max LLM-call cap if Terminus exposes one, or a wrapper-level kill if not

Start with the five non-frontier open models most likely to complete:

- `z-ai/glm-5.2`
- `minimax/minimax-m3`
- `deepseek/deepseek-v4-pro`
- `nvidia/nvidia-nemotron-3-ultra-550b-a55b`
- `google/gemma-4-31b-it`

Then re-probe Kimi and Grok separately with lower task concurrency and stricter
timeouts before including them in a cost-sensitive panel.

## DeepSeek V4 Pro debugging notes

Date: 2026-06-29

DeepSeek V4 Pro's original hard-short rows should not be scored. The initial
calls used the generic OpenAI-compatible path and all six tasks timed out, which
looked like a model failure but was primarily an integration/configuration
problem.

Hypotheses tested:

- Model string / transport: `trustedrouter/deepseek/deepseek-v4-pro` through the
  SDK shim can call the model successfully.
- Provider route: `gmi` can complete a Terminal-Bench PONG smoke task; `tinfoil`
  timed out even on PONG and should not be used for this harness without more
  route work.
- Parser format: XML passed PONG, but on `cancel-async-tasks` DeepSeek omitted
  the outer `<response>` wrapper, so Terminus extracted no commands. JSON emits
  markdown-fenced JSON with parser warnings, but Terminus can recover and
  execute commands.
- Context budget: litellm reports DeepSeek's `max_tokens` as `8192`, and
  Terminus uses that as a context window. This triggered premature proactive
  summarization and handoff loops. The SDK shim now supports
  `TRBENCH_CONTEXT_TOKENS` and the wrapper exposes `--context-tokens`.

Probe results:

| Probe | Provider | Parser | Episodes | Result | Tokens |
|---|---|---|---:|---|---:|
| PONG smoke | gmi | XML | 3 | passed | 15,088 in / 3,243 out |
| PONG smoke | gmi | JSON | 3 | passed | 13,441 in / 2,370 out |
| PONG smoke | tinfoil | XML | 3 | timed out | not scored |
| `cancel-async-tasks` | gmi | JSON | 12 | failed one subtest | 58,706 in / 3,958 out |
| `fix-code-vulnerability` | gmi | JSON | 12 | failed; no code fix/report | 74,003 in / 1,848 out |

Recommended DeepSeek config for the next controlled run:

```bash
TRBENCH_PROVIDER=gmi TRBENCH_MAX_TOKENS=8192 \
uv run python -m trbench.evals.terminal_bench.run \
  --models deepseek/deepseek-v4-pro \
  --dataset-path .data/terminal-bench-hard-short/tasks \
  --tasks hard-short \
  --agent terminus-2 \
  --agent-kwarg parser_name=json \
  --agent-kwarg max_episodes=12 \
  --context-tokens 1000000 \
  --per-task \
  --per-task-order task-major \
  --per-task-timeout 900 \
  --output-root runs/terminal_bench_hard_short_deepseek_fixed \
  --out results/terminal_bench_hard_short_deepseek_fixed.json \
  --resume
```

This is now runnable without the old runaway timeout behavior, but both real
task probes failed under Terminus. `fix-code-vulnerability` is the stronger
signal because the model inspected the right Bottle header code path but did not
modify `bottle.py` or create the required `/app/report.jsonl` before the episode
cap. Exclude DeepSeek from the main hard-short panel for now. Treat future
DeepSeek rows as valid only if run with the SDK shim, provider pin, JSON parser,
and context override.

## 2026-06-29 handoff

Latest reliable hard-short additions:

- Grok 4.3 passed `cancel-async-tasks` and `fix-code-vulnerability`.
- Grok 4.3 timed out on `polyglot-rust-c`.
- TR Prometheus failed `cancel-async-tasks`, passed `fix-code-vulnerability`,
  and timed out / infra-failed on `polyglot-rust-c`.

We briefly tried two parallel Terminal-Bench processes to reduce wall-clock time.
That made Docker cleanup/resume state unreliable: orphan polyglot containers were
left behind and zero-token rows were written for tasks that were not actually
attempted. The active sidecar was cleaned so `--resume` will not skip those
future tasks. Suspect rows are preserved in:

- `results/terminal_bench_hard_short_quarantined_rows.jsonl`

Current pickup recommendation:

- Keep DeepSeek V4 Pro and old TR Synth excluded.
- Resume serially, not parallel, for the remaining Grok and Prometheus cells.
- Use the cleaned main sidecar:
  `results/terminal_bench_hard_short_remaining_safe_tasks.jsonl`
- Remaining real tasks:
  - Grok 4.3: `configure-git-webserver`, `feal-linear-cryptanalysis`,
    `password-recovery`
  - TR Prometheus: `configure-git-webserver`, `feal-linear-cryptanalysis`,
    `password-recovery`
