# Athena Terminal-Bench 2.1 Slow-Hard Repair Attempts

- Model: `trustedrouter/athena`
- Base URL: `https://api.trustedrouter.com/v1`
- Original run: `athena-slow-hard-10-20260701T150915Z`
- Repair jobs:
  - `athena-slow-hard-repair-6-20260701T184156Z`
  - `athena-slow-hard-repair2-6-20260701T204705Z`
  - `athena-slow-hard-repair3-6-20260701T230457Z`

## Harness Fix

The Harbor TrustedRouter adapter was using the SDK default request timeout (`120s`). That turned long Athena calls into Harbor `InternalError` rows:

`TrustedRouter regional endpoint unavailable: The read operation timed out`

`scripts/harbor_tr_agent.py` now passes an explicit SDK timeout/retry policy:

- default `TRUSTEDROUTER_TIMEOUT=900`
- default `TRUSTEDROUTER_MAX_RETRIES=2`
- agent kwargs: `request_timeout=...`, `request_retries=...`
- call metadata records `request_timeout` and `request_retries`

Verification: `python3 -m py_compile scripts/harbor_tr_agent.py`

## Repair Outcomes

| Task | Best repaired status | Calls | Cost | Notes |
|---|---:|---:|---:|---|
| `path-tracing` | PASS | 48 | $14.6566 | Repaired from transport `InternalError` to verifier reward `1.0`; longest successful call was 678.3s, which proves the old 120s timeout was too low. |
| `path-tracing-reverse` | stopped partial | 69 | $20.4236 | No final verifier score; reached one red-channel pixel mismatch after local compare. One call hit the new 900s timeout, then the next retry advanced. Stopped to avoid running the rest of the six-row queue unattended. |
| `train-fasttext` | still FAIL/partial | 100 | $20.3806 | Not a transport failure. Repair2 reached more training attempts but was still below target (`0.5720` in the last visible result; earlier autotune best was `0.607` vs required `0.62`). |
| `extract-moves-from-video` | still FAIL | 60 | $5.1144 | Repair1 reached 60 OCR-frame loop calls and no valid `/app/solution.txt`; not rerun under the fixed adapter before stopping repair3. |
| `write-compressor` | not re-scored after fix | 3 | $0.0242 | Old attempts were 120s transport timeouts; repair3 was stopped before this pending row. |
| `circuit-fibsqrt` | not re-scored after fix | 2 | $0.0193 | Old attempts were 120s transport timeouts; repair3 was stopped before this pending row. |

## Attempt Notes

- Repair1 kept the old SDK timeout and `stream=true`; it completed all six selected rows but still had 3 transport `InternalError`s and 3 verifier failures. Recorded cost: `$26.6360`.
- Repair2 used `stream=false` but was launched before the adapter timeout fix; `path-tracing` and `path-tracing-reverse` still hit 120s read timeouts. It was stopped while `train-fasttext` was running.
- Repair3 used the fixed adapter with `request_timeout=900`, `stream=false`, and `max_turns=120`. It cleanly passed `path-tracing`, then was stopped during `path-tracing-reverse` after 69 calls and one-pixel local mismatch.

## Current Interpretation

The transport root cause was the adapter's implicit 120s SDK timeout. Raising and logging the SDK timeout converted `path-tracing` from a transport error into a real pass. Remaining issues are now separate:

- Some Athena calls can exceed even 900s on very long rows.
- `path-tracing-reverse` is a model/local-numerics convergence issue, not the original 120s timeout.
- `train-fasttext` and `extract-moves-from-video` are verifier/correctness failures, not transport failures.
- `write-compressor` and `circuit-fibsqrt` still need targeted reruns with the fixed adapter to know their true scores.
