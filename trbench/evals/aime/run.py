"""Run AIME 2025 against a TrustedRouter model panel.

Competition math, integer answers. Zero-shot: the model reasons and puts its
final integer in \\boxed{}, then a deterministic integer matcher checks it. No
judge, no sandbox. max_tokens defaults high because these need long solutions.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from trbench import client
from trbench.evals.aime.loader import load
from trbench.panel import resolve_panel

PROMPT_SUFFIX = (
    "\n\nThis is a competition problem with an integer answer between 0 and 999. "
    "Reason step by step, then put your final answer in \\boxed{}."
)


def run_model(
    model: str,
    items: list[dict],
    *,
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    concurrency: int,
) -> list[dict]:
    def one(item: dict) -> dict:
        r = client.chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": item["question"] + PROMPT_SUFFIX}],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
        )
        row = {"model": model, "id": item["id"], "target": item["target"]}
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
    parser = argparse.ArgumentParser(description="Run AIME 2025 on a TrustedRouter panel.")
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: the panel).")
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-limit", type=int, default=None, help="Use only the first N problems.")
    parser.add_argument("--out", default="results/aime.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    items = load(limit=args.prompt_limit)

    responses: list[dict] = []
    for model in models:
        print(f"running aime: {model} ({len(items)} problems)")
        responses.extend(
            run_model(
                model,
                items,
                base_url=args.base_url,
                api_key=api_key,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                concurrency=args.concurrency,
            )
        )

    result = {
        "eval": "aime",
        "dataset": "math-ai/aime25",
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
