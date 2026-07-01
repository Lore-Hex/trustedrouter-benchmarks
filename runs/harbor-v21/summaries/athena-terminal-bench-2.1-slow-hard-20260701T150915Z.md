# Athena Terminal-Bench 2.1 Slow-Hard Run

- Job: `athena-slow-hard-10-20260701T150915Z`
- Model: `trustedrouter/athena`
- Base URL: `https://api.trustedrouter.com/v1`
- Max turns: `60`; stream: `true`; no extra prompts
- Started: `2026-07-01T08:09:16.796002`; finished: `2026-07-01T11:29:54.286094`
- Official Harbor mean: `0.4` (4/10 if API errors count as zero)
- Valid scored trials: `4/6` pass; `4` API errors
- Total recorded cost: `$19.571475`

Important caveat: four tasks ended with `InternalError: TrustedRouter regional endpoint unavailable: The read operation timed out`. These are transport errors before verifier scoring, not normal benchmark failures.

## Results

| Task | Status | Reward | Episodes | Requests | Responses | Errors | Cost | Time min | Avg call s | Advisor calls | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `train-fasttext` | FAIL | 0.0 | 60 | 60 | 60 | 0 | $4.5437 | 52.37 | 8.45 | 0 | failed verifier: accuracy 0.51 < required 0.62; model size test passed |
| `path-tracing-reverse` | ERROR |  | 32 | 39 | 31 | 8 | $4.4947 | 27.3 | 20.25 | 0 | TrustedRouter streaming InternalError: regional endpoint read timeout before verifier scoring |
| `extract-moves-from-video` | FAIL | 0.0 | 60 | 61 | 60 | 1 | $4.6685 | 31.72 | 10.66 | 0 | failed verifier: no valid /app/solution.txt after 60-call OCR attempt |
| `sam-cell-seg` | PASS | 1.0 | 30 | 31 | 30 | 1 | $2.1507 | 24.3 | 13.45 | 0 | passed verifier |
| `torch-tensor-parallelism` | PASS | 1.0 | 5 | 7 | 5 | 2 | $0.3570 | 14.26 | 34.95 | 0 | passed verifier |
| `path-tracing` | ERROR |  | 13 | 15 | 12 | 3 | $1.1253 | 12.86 | 28.67 | 0 | TrustedRouter streaming InternalError: regional endpoint read timeout before verifier scoring |
| `mcmc-sampling-stan` | PASS | 1.0 | 25 | 25 | 25 | 0 | $1.8781 | 20.05 | 14.23 | 0 | passed verifier |
| `circuit-fibsqrt` | ERROR |  | 3 | 5 | 2 | 3 | $0.0197 | 6.48 | 5.43 | 0 | TrustedRouter streaming InternalError: regional endpoint read timeout before verifier scoring |
| `bn-fit-modify` | PASS | 1.0 | 10 | 10 | 10 | 0 | $0.3193 | 2.6 | 9.44 | 0 | passed verifier |
| `write-compressor` | ERROR |  | 3 | 6 | 2 | 4 | $0.0145 | 8.65 | 11.79 | 0 | TrustedRouter streaming InternalError: regional endpoint read timeout before verifier scoring |

## Error Root Cause

- `path-tracing`, `path-tracing-reverse`, `write-compressor`, and `circuit-fibsqrt` failed with TrustedRouter streaming `InternalError` / regional endpoint read timeouts.
- `train-fasttext` reached the verifier but failed accuracy (`0.51` vs required `0.62`) while model size passed.
- `extract-moves-from-video` hit the 60-call limit after high-rate OCR work and failed verifier scoring.

## Passing Tasks

- `sam-cell-seg`: 30 calls, $2.1507, 24.3 min
- `torch-tensor-parallelism`: 5 calls, $0.3570, 14.26 min
- `mcmc-sampling-stan`: 25 calls, $1.8781, 20.05 min
- `bn-fit-modify`: 10 calls, $0.3193, 2.6 min
