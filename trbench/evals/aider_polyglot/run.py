"""Run the Aider polyglot Python subset on a TrustedRouter panel.

Single-attempt (pass@1, no test feedback). Aider's published protocol allows a
second attempt with the failing test output, so these numbers are a stricter
floor than the public leaderboard — documented in EVALS.md / REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from trbench import client
from trbench.evals.aider_polyglot.loader import load
from trbench.panel import resolve_panel

PROMPT = """\
Solve this Exercism coding exercise.

{instructions}

Edit this file and return its COMPLETE contents implementing the solution.
Return ONLY the full code for `{stub_name}` in a single ```python code block,
no explanation.

```python
{stub_code}
```"""


def run_model(model, items, *, base_url, api_key, max_tokens, timeout, concurrency) -> list[dict]:
    def one(it: dict) -> dict:
        msg = PROMPT.format(instructions=it["instructions"], stub_name=it["stub_name"], stub_code=it["stub_code"])
        r = client.chat(
            base_url=base_url, api_key=api_key, model=model,
            messages=[{"role": "user", "content": msg}],
            max_tokens=max_tokens, temperature=0.0, timeout=timeout,
        )
        row = {"model": model, "id": it["id"], "stub_name": it["stub_name"], "test_name": it["test_name"]}
        if r.get("error"):
            row["error"] = r["error"]
        else:
            row["text"] = r.get("text", "")
        return row

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = [pool.submit(one, it) for it in items]
        for f in as_completed(futs):
            out.append(f.result())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Aider polyglot (Python subset).")
    parser.add_argument("--models", default=None)
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/aider_polyglot.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    items = load(limit=args.limit)
    print(f"aider-polyglot (python): {len(models)} models x {len(items)} exercises")

    responses: list[dict] = []
    for model in models:
        print(f"  coding: {model}")
        responses.extend(
            run_model(model, items, base_url=args.base_url, api_key=api_key,
                      max_tokens=args.max_tokens, timeout=args.timeout, concurrency=args.concurrency)
        )
    result = {
        "eval": "aider_polyglot_python",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "item_count": len(items),
        "responses": sorted(responses, key=lambda r: (r["model"], r["id"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
