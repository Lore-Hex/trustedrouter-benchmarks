# TrustedRouter Benchmarks — build report

What got built, what ran, what's left, and the decisions and learnings along the
way. Headline result tables live in the per-eval blocks in [README.md](README.md);
this is the narrative.

## Status at a glance

| Eval | Implemented | Run on the panel | Notes |
|---|---|---|---|
| IFEval | ✅ | ✅ full 541 | deterministic, no judge — cheapest |
| SimpleQA Verified | ✅ | ✅ 250-subset | no-tools; novel Chinese-panel data |
| Chinese SimpleQA | ✅ | ✅ 250-subset | Chinese-language factuality |
| Aider polyglot (Python) | ✅ | ✅ 34 exercises | real unit tests, pass@1 |
| tau2-bench | run-doc | ⏸ pending infra | agentic; wrap upstream CLI |
| LiveCodeBench | run-doc | ⏸ pending infra | needs code execution |
| Terminal-Bench 2.0 | run-doc | ⏸ pending infra | needs Docker (some images amd64-only) |

Four evals run end-to-end on the Chinese panel via TrustedRouter. The three
agentic/sandboxed ones have the exact canonical run command (pointed at the TR
endpoint) in each eval's `README.md`; running them faithfully needs Docker /
language toolchains / a code-execution sandbox, so they're documented rather
than spun up blindly here.

## Results

See the result blocks in the README (spliced in by each eval's `score`):

- IFEval — `IFEVAL_RESULTS`
- SimpleQA Verified — `SIMPLEQA_VERIFIED_RESULTS`
- Chinese SimpleQA — `CHINESE_SIMPLEQA_RESULTS`
- Aider polyglot (Python) — `AIDER_POLYGLOT_RESULTS`

<!-- REPORT_HEADLINES_START -->
_Headline numbers are filled in once the panel runs complete._
<!-- REPORT_HEADLINES_END -->

## Decisions

- **Faithful over reinvented.** Where a canonical scorer exists, vendor or drive
  it verbatim rather than reimplement: IFEval uses Google's official verifiers;
  both factuality evals use OpenAI's canonical SimpleQA grader template; Aider
  runs the exercises' real unit tests. Reimplementing scorers is where biased
  benchmarks come from.
- **Download datasets at runtime, never redistribute.** SimpleQA Verified and
  Chinese SimpleQA are fetched to a gitignored `.data/` cache. Chinese SimpleQA
  ships no license, so we publish only aggregate scores, not the data.
- **Aider polyglot = Python subset, pass@1.** The full benchmark is 225
  exercises across 6 languages and allows a 2nd attempt with test feedback.
  Running 6 toolchains in Docker wasn't worth it for a first cut, so we run the
  34 Python exercises single-attempt. That makes our numbers a strict *floor*
  vs the public leaderboard — documented, not hidden.
- **One cheap judge for both factuality evals.** `google/gemini-2.5-flash`
  grades A/B/C with the language-agnostic SimpleQA rubric (8 output tokens/grade).
  A Chinese-native grader prompt would be marginally more faithful for Chinese
  SimpleQA; flagged as a refinement.
- **Subset sizes to bound cost.** 250 questions for the factuality evals, full
  541 for IFEval (it's the cheapest), full 34 for Aider. Full-size runs are a
  `--limit`/no-flag away once the numbers look sane.
- **Excluded GPQA Diamond.** Cheap but saturated/compressed at the top — almost
  no model-distinguishing signal (per the deep-research sweep).
- **Panel = Chinese open-weight models + one frontier ref (Opus 4.8).** The
  point of the repo is the Chinese models; the ref is a sanity line.

## Learnings (things that broke, and the fix)

- **Subset scoring bug.** The first IFEval scorer divided by all 541 prompts
  even on a `--prompt-limit` smoke, so subset scores looked ~0.2%. The smoke
  caught it; fix = score only over the prompts actually run.
- **`.gitignore` false-match.** `grep '.data/'` matched the existing
  `nltk_data/` line (the `.` is a regex wildcard), so `.data/` never got added
  and the downloaded datasets got committed. Fixed with a literal match and
  `git rm --cached`. (A full history purge is pending explicit force-push
  authorization; the datasets are already public on HF, so this is cosmetic.)
- **Background output buffering.** Python block-buffers stdout to a file, so a
  long run looks silent mid-flight even though it's working; check the process,
  not the log tail.
- **Provider gaps surface as errors, not crashes.** `z-ai/glm-5.2` was
  provider-down (HTTP 5xx) during the run; the harness records the error per
  item and the model scores over what completed, with an errors column.
- **Cost discipline.** The deterministic / no-judge evals (IFEval, Aider) are
  the safe-cheap bets; the LLM-judge and agentic ones cost more, so they get
  subsets and small-N.

## Next

1. **Fusion.** Run each eval through `trustedrouter/fusion` vs the best single
   model — the unique TR story, extending the DRACO/Fusion posts.
2. **Full-size runs** (541 / 1000 / 3000 / 225) for publishable numbers, and add
   the newest flagships missing from public boards.
3. **The three agentic evals** on a Docker/amd64 host.
4. Verify Chinese SimpleQA's license before any dataset republication.
