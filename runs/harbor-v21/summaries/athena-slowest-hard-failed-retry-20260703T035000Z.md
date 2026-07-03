# Athena Slowest-Hard Failed Retry

Date: 2026-07-03
Base URL: `https://api.trustedrouter.com/v1`
Model: `trustedrouter/athena`
Dataset: Terminal-Bench v2.1 hard subset, previous Athena slowest-hard failures

## Runs

- `runs/harbor-v21/athena-slowest-hard-failed7-retry-20260703T025916Z`
- `runs/harbor-v21/athena-slowest-hard-failed4-retry-20260703T033602Z`

Both run directories include `replay_manifest.json`.

## Results

| Task | Reward | Exception | Calls | Empty responses | Advisor calls | Advisor attempts | Worker attempts | Cost |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `install-windows-3.11` | 0.0 | no | 14 | 0 | 1 | 1 | 15 | $0.4668 |
| `make-mips-interpreter` | 0.0 | no | 66 | 9 | 2 | 2 | 60 | $8.9918 |
| `protein-assembly` | n/a | yes, cancelled | 144 | 144 | 0 | 0 | 0 | $0.0000 |
| `llm-inference-batching-scheduler` | n/a | yes, cancelled | 42 | 42 | 0 | 0 | 0 | $0.0000 |

## Notes

- `install-windows-3.11` and `make-mips-interpreter` were real Athena attempts and both called advisor more than the prior baseline.
- `protein-assembly` and `llm-inference-batching-scheduler` did not reach shell execution. They repeatedly returned empty assistant content with `finish_reason: stop`, `usage: null`, and no TrustedRouter metadata.
- A tiny smoke test after the retries showed the same empty/null-usage behavior for `trustedrouter/athena`, `trustedrouter/socrates-pro-plus-1.0`, `trustedrouter/aristotle-1.0`, and `trustedrouter/openpatcher-g1`.
- Direct provider-model smoke tests through the same API key returned `402 API key spend limit exceeded` for `openai/gpt-4.1-mini` and `anthropic/claude-sonnet-4.5`, so the empty combo-model responses are likely masking the same billing/spend-limit failure.
