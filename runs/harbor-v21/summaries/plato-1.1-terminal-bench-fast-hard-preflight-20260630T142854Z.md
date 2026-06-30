# Plato 1.1 Terminal-Bench Fast-Hard Preflight

Generated: `2026-06-30T14:28:54Z`

Requested run: `trustedrouter/plato-1.1` on Terminal-Bench 2.1 Fast-Hard.

Status: **not run**. The production TrustedRouter chat endpoint rejects `trustedrouter/plato-1.1`, so launching Harbor would fail before any benchmark task work.

## Target Tasks

1. `feal-differential-cryptanalysis`
2. `model-extraction-relu-logits`
3. `password-recovery`
4. `feal-linear-cryptanalysis`
5. `configure-git-webserver`
6. `polyglot-rust-c`
7. `fix-code-vulnerability`
8. `cancel-async-tasks`

## Preflight Evidence

- Base URL: `https://api.trustedrouter.com/v1`
- Requested model: `trustedrouter/plato-1.1`
- Chat smoke result with both available prod keys: `400 BadRequest`
- Error text: `Model does not support chat completions: trustedrouter/plato-1.1`
- Public catalog checked at `https://trustedrouter.com/v1/models`: no `trustedrouter/plato-1.1` entry found.
- Catalog Plato entries found:
  - `trustedrouter/plato`
  - `trustedrouter/plato-1.0`
  - `trustedrouter/plato-pro`
  - `trustedrouter/plato-pro-1.0`

Control smoke calls:

| Model | Result | Response Model |
|---|---|---|
| `trustedrouter/plato` | ok | `deepseek/deepseek-v4-flash` |
| `trustedrouter/plato-1.0` | ok | `deepseek/deepseek-v4-flash` |

No Harbor replay directory was created for this requested model because the benchmark was not launched.
