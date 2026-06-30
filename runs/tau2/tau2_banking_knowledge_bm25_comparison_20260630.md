# tau2 banking_knowledge bm25 comparison - 2026-06-30

20 tasks, 1 trial, tau2 `banking_knowledge`, BM25 retrieval, user model `openai/gpt-4.1`.

| Model | Full pass | Partial macro | Calls | Advisor calls | Priced cost | Unpriced calls | Sim duration sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| `xiaomi/mimo-v2.5-pro` | 25.0% (5/20) | 35.0% | 357 | 0 | n/a | 357 | 2551.1s |
| `deepseek/deepseek-v4-flash` | 30.0% (6/20) | 36.9% | 409 | 0 | n/a | 409 | 1895.3s |
| `trustedrouter/socrates-1.1` | 10.0% (2/20) | 23.3% | 384 | 1 | $6.702605 | 123 | 1309.6s |
| `trustedrouter/aristotle-1.0` | 25.0% (5/20) | 32.5% | 440 | 0 | $1.168157 | 129 | 2224.7s |

Cost is summed from `response.usage.cost_microdollars` when present in saved TrustedRouter call logs. Direct Mimo/DeepSeek calls and the tau2 user-simulator calls in this run did not include `cost_microdollars`, so those calls are counted as unpriced rather than estimated.
