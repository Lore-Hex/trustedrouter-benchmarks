"""Run SimpleQA Verified (closed-book, no tools) on a TrustedRouter panel."""
from __future__ import annotations

import argparse
import json
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from trbench import client
from trbench.evals.qa_common import run_panel
from trbench.evals.simpleqa_verified.loader import load
from trbench.panel import resolve_panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SimpleQA Verified on a TrustedRouter panel.")
    parser.add_argument("--models", default=None)
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    # 32768, not 8192: verbose reasoning models (the GLM family, kimi-k2.6) spend
    # >8192 tokens thinking on hard questions and truncate before emitting a final
    # answer — the harness then records an empty/incomplete response that grades
    # NOT_ATTEMPTED, artificially sinking their score. A generous budget is free for
    # non-reasoning models (they stop at natural length).
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N questions (cheap subset).")
    parser.add_argument("--out", default="results/simpleqa_verified.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    items = load(limit=args.limit)
    print(f"simpleqa-verified: {len(models)} models x {len(items)} questions")

    responses = run_panel(
        items,
        base_url=args.base_url,
        api_key=api_key,
        models=models,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    result = {
        "eval": "simpleqa_verified",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "item_count": len(items),
        "responses": sorted(responses, key=lambda r: (r["model"], str(r["id"]))),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
