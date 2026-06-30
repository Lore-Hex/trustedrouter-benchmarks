# tau3 banking_knowledge alltools comparison - 2026-06-30

20 tasks, 1 trial, tau3/tau-bench `banking_knowledge`, retrieval config `alltools` (BM25 + dense embeddings + shell), user model `openai/gpt-4.1`.

| Model | Full pass | Partial total | Wall time | Agent steps | TR calls | Advisor calls | Max-step exits | Priced chat cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xiaomi/mimo-v2.5-pro` | 10.0% (2/20) | 4.78/20 (23.9%) | 18.9m | 265 | 353 | 0 | 3 | n/a |
| `deepseek/deepseek-v4-flash` | 25.0% (5/20) | 5.92/20 (29.6%) | 19.3m | 340 | 443 | 0 | 9 | n/a |
| `trustedrouter/socrates-1.1` | 10.0% (2/20) | 4.62/20 (23.1%) | 8.2m | 280 | 372 | 0 | 3 | $5.905755 |
| `trustedrouter/aristotle-1.0` | 25.0% (5/20) | 7.42/20 (37.1%) | 18.6m | 308 | 428 | 1 | 7 | $1.470196 |

Cost note: priced chat cost is summed from `response.usage.cost_microdollars` where present in saved chat call logs. Direct Mimo/DeepSeek calls and user-simulator calls did not include priced metadata; embedding cache warmup calls are not captured in these call logs.
