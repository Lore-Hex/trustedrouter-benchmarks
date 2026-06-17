"""Run IFEval against a TrustedRouter model panel.

IFEval is zero-shot: the prompt IS the instruction set, no system prompt, and
the raw model output is scored by deterministic Python verifiers. Cheapest
useful eval in the repo — no judge model, no sandbox.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from trbench import client
from trbench.panel import resolve_panel

DATA_PATH = Path(__file__).parent / "data" / "input_data.jsonl"


def load_prompts(path: Path = DATA_PATH, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def run_model(
    model: str,
    prompts: list[dict],
    *,
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    concurrency: int,
) -> list[dict]:
    def one(p: dict) -> dict:
        r = client.chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": p["prompt"]}],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
        )
        row = {"model": model, "key": p["key"], "prompt": p["prompt"]}
        if r.get("error"):
            row["error"] = r["error"]
        else:
            row["text"] = r.get("text", "")
        return row

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, p) for p in prompts]
        for fut in as_completed(futures):
            out.append(fut.result())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run IFEval on a TrustedRouter panel.")
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: the panel).")
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-limit", type=int, default=None, help="Use only the first N prompts (cheap smoke).")
    parser.add_argument("--out", default="results/ifeval.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    prompts = load_prompts(limit=args.prompt_limit)

    responses: list[dict] = []
    for model in models:
        print(f"running ifeval: {model} ({len(prompts)} prompts)")
        responses.extend(
            run_model(
                model,
                prompts,
                base_url=args.base_url,
                api_key=api_key,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                concurrency=args.concurrency,
            )
        )

    result = {
        "eval": "ifeval",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "prompt_count": len(prompts),
        "responses": sorted(responses, key=lambda r: (r["model"], r["key"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
