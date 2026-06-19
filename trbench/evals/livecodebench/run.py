"""Run LiveCodeBench (code generation) against a TrustedRouter model panel.

Each model writes a stdin/stdout Python program for a contamination-windowed
problem; grading happens in score.py (containerized), so the saved replay is the
raw model output and the pass@1 is reproducible from it.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from trbench import client
from trbench.evals.livecodebench.executor import gradeable
from trbench.evals.livecodebench.loader import DEFAULT_MIN_DATE, DEFAULT_VERSION, load
from trbench.panel import resolve_panel

PROMPT = (
    "{question}\n\n"
    "Write a complete Python 3 program that reads from standard input and writes the "
    "answer to standard output. Put the final program in a single ```python code block."
)


def run_model(model, items, *, base_url, api_key, max_tokens, timeout, concurrency):
    def one(item: dict) -> dict:
        r = client.chat(
            base_url=base_url, api_key=api_key, model=model,
            messages=[{"role": "user", "content": PROMPT.format(question=item["question_content"])}],
            max_tokens=max_tokens, temperature=0.0, timeout=timeout,
        )
        row = {"model": model, "id": item["id"]}
        if r.get("error"):
            row["error"] = r["error"]
        else:
            row["text"] = r.get("text", "")
        return row

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, item) for item in items]
        for fut in as_completed(futures):
            out.append(fut.result())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LiveCodeBench on a TrustedRouter panel.")
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: the panel).")
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--version", default=DEFAULT_VERSION, help="LiveCodeBench release window (v1..v6).")
    parser.add_argument("--min-date", default=DEFAULT_MIN_DATE, help="Keep problems with contest_date >= this.")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-limit", type=int, default=None, help="Use only the first N gradeable problems.")
    parser.add_argument("--out", default="results/livecodebench.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    # Only stdin/stdout problems are gradeable today; oversample the stream (some
    # problems are LeetCode-functional and get filtered) then take the first N.
    load_limit = args.prompt_limit * 3 + 20 if args.prompt_limit is not None else None
    pool = load(version=args.version, min_date=args.min_date, limit=load_limit)
    items = [it for it in pool if gradeable(it)]
    if args.prompt_limit is not None:
        items = items[: args.prompt_limit]

    responses: list[dict] = []
    for model in models:
        print(f"running livecodebench: {model} ({len(items)} problems)")
        responses.extend(
            run_model(model, items, base_url=args.base_url, api_key=api_key,
                      max_tokens=args.max_tokens, timeout=args.timeout, concurrency=args.concurrency)
        )

    result = {
        "eval": "livecodebench",
        "dataset": f"livecodebench/code_generation_lite ({args.version}, >= {args.min_date}, stdin-only)",
        "version": args.version,
        "min_date": args.min_date,
        "load_limit": load_limit,
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "item_count": len(items),
        "item_ids": [it["id"] for it in items],
        "responses": sorted(responses, key=lambda r: (r["model"], r["id"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
