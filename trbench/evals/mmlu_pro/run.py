"""Run MMLU-Pro against a TrustedRouter model panel.

Ten-choice knowledge/reasoning MCQ across 14 categories. Zero-shot chain-of-
thought: the model reasons then states "The answer is (X)"; scoring extracts the
letter and compares to gold. No judge model, no sandbox — deterministic.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from trbench import adapters
from trbench import client
from trbench.evals.mmlu_pro.loader import format_options, load
from trbench.panel import resolve_panel

PROMPT = (
    "{question}\n\nOptions:\n{options}\n\n"
    'Think step by step, then end your response with "The answer is (X)." '
    "where X is the letter of the correct option."
)


def run_model(
    model: str,
    items: list[dict],
    *,
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    temperature: float | None,
    extra_body: dict | None,
    concurrency: int,
) -> list[dict]:
    def one(item: dict) -> dict:
        prompt = PROMPT.format(question=item["question"], options=format_options(item["options"]))
        r = client.chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
        )
        # Carry the option count so the scorer knows the valid letter range.
        row = {"model": model, "id": item["id"], "answer": item["answer"],
               "category": item.get("category", ""), "n_options": len(item["options"])}
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
    parser = argparse.ArgumentParser(description="Run MMLU-Pro on a TrustedRouter panel.")
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: the panel).")
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    # CoT on reasoning models is long; 32768 keeps the answer from truncating
    # (same lesson as the SimpleQA/MATH panels).
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-limit", type=int, default=None, help="Use only the first N questions.")
    parser.add_argument("--adapter-file", default=None, help="JSON model adapter recommendations.")
    parser.add_argument("--out", default="results/mmlu_pro.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    adapter_specs = adapters.load_adapter_file(args.adapter_file)
    items = load(limit=args.prompt_limit)

    responses: list[dict] = []
    for model in models:
        adapter = adapters.adapter_for(
            model,
            adapter_specs,
            default_max_tokens=args.max_tokens,
            default_timeout=args.timeout,
        )
        print(
            f"running mmlu_pro: {model} ({len(items)} questions, "
            f"max_tokens={adapter.max_tokens}, timeout={adapter.timeout}, "
            f"temperature={'omitted' if adapter.temperature is None else adapter.temperature})"
        )
        responses.extend(
            run_model(
                model, items, base_url=args.base_url, api_key=api_key,
                max_tokens=adapter.max_tokens or args.max_tokens,
                timeout=adapter.timeout or args.timeout,
                temperature=adapter.temperature,
                extra_body=adapter.extra_body,
                concurrency=args.concurrency,
            )
        )

    result = {
        "eval": "mmlu_pro",
        "dataset": "TIGER-Lab/MMLU-Pro",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "adapter_file": args.adapter_file,
        "adapter_settings": {m: adapters.public_adapter_settings(m, adapter_specs) for m in models},
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
