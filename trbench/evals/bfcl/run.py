"""Run BFCL v4 single-turn function-calling on a TrustedRouter panel.

Each item's function schema is handed to the model as OpenAI `tools` (via the
TrustedRouter SDK). We collect the returned tool_calls and decode them to BFCL's
`model_output` shape — [{func_name: {param: value}}] — for the vendored AST
checker. Generation only; scoring is in score.py.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from trbench import client
from trbench.evals.bfcl.loader import DEFAULT_CATEGORIES, load
from trbench.evals.bfcl.schema_convert import to_openai_tools
from trbench.panel import resolve_panel


def decode_tool_calls(tool_calls: list[dict] | None) -> list[dict]:
    """tool_calls -> BFCL model_output [{name: {param: value}}]. arguments is a
    JSON string; a member that won't parse is skipped (counts as a wrong call)."""
    out = []
    for tc in tool_calls or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        out.append({name: args})
    return out


def _answer_one(it: dict, model: str, base_url: str, api_key: str, max_tokens: int, timeout: float) -> dict:
    tools = to_openai_tools(it["functions"])
    r = client.chat(
        base_url=base_url, api_key=api_key, model=model,
        messages=it["messages"], tools=tools, max_tokens=max_tokens,
        temperature=0.0, timeout=timeout,
    )
    row = {
        "model": model, "id": it["id"], "category": it["category"],
        "checker_kind": it["checker_kind"], "ast_test_category": it["ast_test_category"],
        "functions": it["functions"], "ground_truth": it["ground_truth"],
    }
    if r.get("error"):
        row["error"] = r["error"]
    else:
        row["text"] = r.get("text", "")
        row["tool_calls"] = r.get("tool_calls")
        row["decoded"] = decode_tool_calls(r.get("tool_calls"))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BFCL v4 function-calling on a TrustedRouter panel.")
    parser.add_argument("--models", default=None)
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--categories", default=None,
                        help="Comma list (default: simple_python,multiple,parallel,parallel_multiple,irrelevance).")
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default="results/bfcl.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else DEFAULT_CATEGORIES
    items = load(categories=cats, limit_per_category=args.limit_per_category)
    print(f"bfcl: {len(models)} models × {len(items)} items ({', '.join(cats)})")

    responses: list[dict] = []
    for model in models:
        print(f"  answering: {model} ({len(items)} items)")
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = [pool.submit(_answer_one, it, model, args.base_url, api_key, args.max_tokens, args.timeout) for it in items]
            for f in as_completed(futs):
                responses.append(f.result())

    result = {
        "eval": "bfcl",
        "dataset": "ShishirPatil/gorilla BFCL v4 (single-turn AST + irrelevance)",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "categories": cats, "models": models, "item_count": len(items),
        "responses": sorted(responses, key=lambda r: (r["model"], r["id"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
