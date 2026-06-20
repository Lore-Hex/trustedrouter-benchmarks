"""Generate BEAM 128K answers by feeding full conversation + probing question."""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from trbench import client
from trbench.evals.beam.loader import load
from trbench.panel import resolve_panel


def _answer_one(
    it: dict,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    messages = list(it["messages"]) + [{"role": "user", "content": it["question"]}]
    r = client.chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        timeout=timeout,
    )
    row = {
        "model": model,
        "id": it["id"],
        "conversation_id": it["conversation_id"],
        "question_type": it["question_type"],
        "question": it["question"],
        "rubric": it["rubric"],
        "ideal_response": it["ideal_response"],
        "difficulty": it.get("difficulty", ""),
    }
    if r.get("error"):
        row["error"] = r["error"]
        row["text"] = ""
    else:
        row["text"] = r.get("text", "")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BEAM 128K on a TrustedRouter panel.")
    parser.add_argument("--models", default=None)
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    # BEAM answers are expected concise — long reasoning models still need room
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=300.0)
    # Lower concurrency: each request carries a 128K context
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--questions-per-type", type=int, default=1,
        help="Questions per (conversation, type) pair. Default=1 ≈ 200 items total. Use 0 for all.",
    )
    parser.add_argument("--out", default="results/beam_128k.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    qpt = args.questions_per_type if args.questions_per_type > 0 else None
    items = load(limit=args.limit, questions_per_type=qpt)
    print(f"beam-128k: {len(models)} models × {len(items)} questions")

    responses: list[dict] = []
    for model in models:
        print(f"  answering: {model}")
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = {
                pool.submit(
                    _answer_one, it, model,
                    args.base_url, api_key, args.max_tokens, args.timeout,
                ): it
                for it in items
            }
            for done, f in enumerate(as_completed(futs), 1):
                row = f.result()
                responses.append(row)
                if done % 20 == 0:
                    print(f"    {done}/{len(items)}")

    result = {
        "eval": "beam_128k",
        "dataset": "Mohammadta/BEAM (100K split)",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "item_count": len(items),
        "responses": sorted(responses, key=lambda r: (r["model"], r["id"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
