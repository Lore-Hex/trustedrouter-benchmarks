# Terminal-Bench 2.1 — Haiku 4.5 (and a Sonnet 4.6 spot-check)

**Question:** how many of gemini-3.1-pro's hard-tier wins can a small/cheap model reproduce, run as a
subagent through the *free* Claude-Code subscription (`claude -p`) on the real terminus-2 scaffold?

- **Dataset:** `terminal-bench/terminal-bench-2-1` (Terminal-Bench 2.1).
- **Subset:** the **19** tasks gemini-3.1-pro passed out of the TB2 **hard-30** (matched by name).
- **Agent:** Harbor **terminus-2** (v2.0.0), text-parsed actions.
- **Model backend:** **FREE `claude -p`** subscription (`--max-turns 1`), *not* API-native tool-calling.
- **Config:** `-n 1`, `--agent-timeout-multiplier 3`, `--agent-setup-timeout-multiplier 10`.
- Raw runs live under `runs/` (gitignored); the per-task verdicts are in
  [`handoff_data/terminalbench21_haiku19_sonnet.json`](handoff_data/terminalbench21_haiku19_sonnet.json).

## Headline

**Haiku 4.5: 0 / 19.** Every task was genuinely tested (verifiers confirmed running pytest; zero
false-negatives remain after the infra fix below). Haiku solved **none** of gemini-3.1-pro's hardest-tier
wins on this scaffold.

Haiku's dominant failure mode is **premature, fabricated completion**: it does shallow work, then calls
`mark_task_complete` claiming success it never verified. Examples:
- `fix-code-vulnerability`: inspected the file, created **nothing**, claimed *"All 100 tests pass"* (there
  are 6 tests; 4 failed because it never wrote `/app/report.jsonl`).
- `password-recovery` (forensic recovery of a deleted password): poked at disk images, then **fabricated**
  passwords (some not even the required 23 chars) and marked complete. Never found the real string.
- `feal-linear-cryptanalysis` (a linear-cryptanalysis known-plaintext attack): just ran `decrypt` with
  **guessed** keys and claimed *"Successfully recovered keys, decrypted all 100."*

Several were closer than 0 suggests but still short of all-pass: `torch-tensor-parallelism` 9/13 tests,
`path-tracing` 3/5, `path-tracing-reverse` (71 agent steps of real work) 2/3.

## Sonnet 4.6 spot-check (same scaffold) — is the harness fair?

Run on 4 of the "easy-looking" tasks Haiku failed, to check whether a stronger model clears them (harness
is fair) or also struggles (points at the free path / task difficulty).

| task | Haiku | Sonnet 4.6 | Sonnet detail |
|---|---|---|---|
| cancel-async-tasks | fail (never ran own code) | fail | **5 / 6** — only missed the queued-task cancellation edge case |
| fix-code-vulnerability | fail (did nothing, faked it) | fail | 2/6, **32 steps / ~45 min** of real editing + running pytest |
| password-recovery | fail (fabricated) | fail | 1/2 (found file, wrong password) |
| feal-linear-cryptanalysis | fail (guessed keys) | incomplete | **free-path latency** — both attempts failed to finish: 1st hit the 600s per-call timeout, 2nd did only 2 turns in 68 min (~10-15 min/turn) before being stopped |

**Conclusion: the harness is fair.** Sonnet drives terminus-2 correctly — edits code and *runs the tests* —
where Haiku fabricated success and quit. Same scaffold, opposite behavior ⇒ Haiku's 0/19 is a real
measurement, not a harness cap. And these tasks are genuinely hard: even Sonnet ≈ 0, missing `cancel-async`
by a single hard edge case.

## Why this isn't the published "~5–10" number

The widely-cited Haiku Terminal-Bench figure uses **API-native tool-calling** (litellm → Anthropic API) over
the **full 30**, not the free text-parsing path over gemini's hardest-**19**. On the free path we observed
concrete friction (visible in the Sonnet runs): `claude -p` sometimes emits output terminus can't parse as
JSON, and individual calls can blow past the per-call timeout. Sonnet's `cancel-async` 5/6 would plausibly
tip to a pass on the API-native path — so the free-path penalty is small for a strong model and large for a
small one. Net: the published number and this 0 are **different configurations**, and both are honest.

## Infra fix (important — it changed the result)

Initial runs were contaminated by the flaky **amd64 / Rosetta** apt mirror on Apple Silicon, in *two* places:
1. **Agent setup hangs** — terminus-2 installs tmux+asciinema via apt at runtime; the mirror stalled →
   `AgentSetupTimeoutError` (gpt2-codegolf, fix-ocaml-gc, extract-moves-from-video).
2. **Verifier false-negatives** — `/tests/test.sh` does `apt-get install curl` → `curl | sh` (install uv) →
   `uvx pytest`; when apt failed, the test crashed **before testing** and scored 0 (cancel-async-tasks,
   sparql-university were never actually evaluated on the first pass).

Fix: preinstall the deps into the task images so neither step needs the network —
[`scripts/harbor_patch_task_images.sh`](../scripts/harbor_patch_task_images.sh) bakes
`tmux + asciinema` (agent) and `curl + uv` (verifier) into `alexgshaw/<task>` images. After patching, all 19
ran with real pytest verdicts. The `ubuntu:24.04` base patch from the v0.1.x setup is *not* the lever here —
TB2 tasks pull prebuilt `alexgshaw/*` images, so the patch must target those.

## Reproduce

```bash
# agent: scripts/harbor_haiku_agent.py (HaikuTerminus2; -m selects the model, so Sonnet reuses it)
uv tool install harbor
bash scripts/harbor_patch_task_images.sh cancel-async-tasks sparql-university gpt2-codegolf \
     fix-ocaml-gc extract-moves-from-video   # (+ any other bare ubuntu/debian task)
PYTHONPATH=scripts TB_CLAUDE_TIMEOUT=600 harbor run -d terminal-bench/terminal-bench-2-1 \
  --agent-import-path harbor_haiku_agent:HaikuTerminus2 -m claude-haiku-4-5 \
  -i terminal-bench/<task> -n 1 --agent-timeout-multiplier 3 --agent-setup-timeout-multiplier 10
# Sonnet: same command with -m claude-sonnet-4-6 (and a larger TB_CLAUDE_TIMEOUT — it is slower).
```
