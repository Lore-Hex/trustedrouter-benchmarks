# LiveCodeBench (contamination-resistant coding)

Real code execution, graded by running each model's program against the problem's
full test suite — **pass@1**, no judge. Contamination-resistant by *time-windowing*:
LiveCodeBench ships versioned releases (`v1`…`v6`), each adding problems from a
later window, so we grade on problems published after the models' training cutoffs.

**Untrusted code runs only in a container.** Every solution is executed in a
throwaway Docker container with `--network none`, capped memory/CPU/pids, and a
per-test timeout — never in-process. The dataset's `private_test_cases` pickle is
decoded with a restricted unpickler (builtins only), so even loading the data
can't execute code.

Scope today: **stdin/stdout problems** (AtCoder/Codeforces — empty `starter_code`).
LeetCode functional problems need call-based harnessing and are filtered out for now.

```bash
export TRUSTEDROUTER_API_KEY=...        # throwaway key
# generate solutions (replay = raw model output)
python -m trbench.evals.livecodebench.run --prompt-limit 50 --out results/livecodebench.json
# grade in containers + render
python -m trbench.evals.livecodebench.score results/livecodebench.json --concurrency 6
```

Needs Docker/OrbStack with the `python:3.11-slim` image. `--version` / `--min-date`
pick the release window. Calibration: gpt-4.1, 8 AtCoder stdin problems → pass@1 75.0
(harness validated end to end; a published-comparable number needs the full
windowed set incl. LeetCode functional, a follow-up).
