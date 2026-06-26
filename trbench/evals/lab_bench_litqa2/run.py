"""Run a LAB-bench multiple-choice smoke test through TrustedRouter."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset

from trbench import adapters
from trbench import client

DEFAULT_MODELS = (
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.2",
    "minimax/minimax-m3",
    "deepseek/deepseek-v4-pro",
    "xiaomi/mimo-v2.5-pro",
    "nvidia/nvidia-nemotron-3-ultra-550b-a55b",
)

PROMPT = """Answer the biology question by choosing the single best option.

Question:
{question}

Options:
{options}

Return only the option letter, with no explanation."""


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def format_options(options: list[str]) -> str:
    return "\n".join(f"{letter}. {option}" for letter, option in zip(_letters(len(options)), options, strict=True))


def load_items(*, config: str, limit: int | None, cache_dir: str) -> list[dict[str, Any]]:
    ds = load_dataset("futurehouse/lab-bench", config, split="train", cache_dir=cache_dir)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    items: list[dict[str, Any]] = []
    for row in ds:
        options = [str(row["ideal"])] + [str(x) for x in row["distractors"]]
        rng = random.Random(_stable_seed(str(row["id"])))
        rng.shuffle(options)
        answer = _letters(len(options))[options.index(str(row["ideal"]))]
        items.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "options": options,
                "answer": answer,
                "ideal": str(row["ideal"]),
                "sources": row.get("sources", []),
                "is_opensource": row.get("is_opensource"),
                "tag": row.get("tag"),
            }
        )
    return items


def run_model(
    model: str,
    items: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    temperature: float | None,
    concurrency: int,
    retries: int,
    process_timeout: float | None,
    extra_body: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    def one(item: dict[str, Any]) -> dict[str, Any]:
        prompt = PROMPT.format(question=item["question"], options=format_options(item["options"]))
        r = client.chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            retries=retries,
            use_request=True,
            extra_body=extra_body,
        )
        row: dict[str, Any] = {
            "model": model,
            "id": item["id"],
            "answer": item["answer"],
            "ideal": item["ideal"],
            "latency_ms": r.get("latency_ms"),
            "attempts": r.get("attempts"),
            "usage": r.get("usage", {}),
        }
        if r.get("error"):
            row["error"] = r["error"]
        else:
            row["text"] = r.get("text", "")
        return row

    def one_process(item: dict[str, Any]) -> dict[str, Any]:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_process_call_one,
            args=(queue, model, item, base_url, api_key, max_tokens, timeout, temperature, retries, extra_body),
        )
        proc.start()
        proc.join(process_timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return {
                "model": model,
                "id": item["id"],
                "answer": item["answer"],
                "ideal": item["ideal"],
                "error": f"process_timeout_after_{process_timeout}s",
                "latency_ms": round(process_timeout * 1000) if process_timeout is not None else None,
            }
        if not queue.empty():
            return queue.get()
        return {
            "model": model,
            "id": item["id"],
            "answer": item["answer"],
            "ideal": item["ideal"],
            "error": f"worker_exited_{proc.exitcode}",
        }

    if process_timeout is not None and concurrency == 1:
        return [one_process(item) for item in items]

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, item) for item in items]
        for fut in as_completed(futures):
            out.append(fut.result())
    return out


def _process_call_one(
    queue: Any,
    model: str,
    item: dict[str, Any],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    temperature: float | None,
    retries: int,
    extra_body: dict[str, Any] | None,
) -> None:
    prompt = PROMPT.format(question=item["question"], options=format_options(item["options"]))
    r = client.chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
        use_request=True,
        extra_body=extra_body,
    )
    row: dict[str, Any] = {
        "model": model,
        "id": item["id"],
        "answer": item["answer"],
        "ideal": item["ideal"],
        "latency_ms": r.get("latency_ms"),
        "attempts": r.get("attempts"),
        "usage": r.get("usage", {}),
    }
    if r.get("error"):
        row["error"] = r["error"]
    else:
        row["text"] = r.get("text", "")
    queue.put(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LAB-bench LitQA2 on TrustedRouter models.")
    parser.add_argument("--config", default="LitQA2", help="LAB-bench config, e.g. LitQA2, ProtocolQA, DbQA, SuppQA.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model ids.")
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--cache-dir", default=".data/huggingface")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument(
        "--process-timeout",
        type=float,
        default=None,
        help="Hard wall-clock timeout per item. Used only with --concurrency 1.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt-limit", type=int, default=10)
    parser.add_argument("--extra-body", default=None, help="JSON merged into each chat completion request.")
    parser.add_argument("--adapter-file", default=None, help="JSON model adapter recommendations.")
    parser.add_argument("--out", default="results/lab_bench_litqa2_smoke_10x6.json")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    adapter_specs = adapters.load_adapter_file(args.adapter_file)
    items = load_items(config=args.config, limit=args.prompt_limit, cache_dir=args.cache_dir)

    responses: list[dict[str, Any]] = []
    for model in models:
        adapter = adapters.adapter_for(
            model,
            adapter_specs,
            default_max_tokens=args.max_tokens,
            default_timeout=args.timeout,
            base_extra_body=extra_body,
        )
        print(
            f"running lab_bench_litqa2: {model} ({len(items)} questions, "
            f"max_tokens={adapter.max_tokens}, timeout={adapter.timeout}, "
            f"temperature={'omitted' if adapter.temperature is None else adapter.temperature})",
            flush=True,
        )
        responses.extend(
            run_model(
                model,
                items,
                base_url=args.base_url,
                api_key=api_key,
                max_tokens=adapter.max_tokens or args.max_tokens,
                timeout=adapter.timeout or args.timeout,
                temperature=adapter.temperature,
                concurrency=args.concurrency,
                retries=args.retries,
                process_timeout=args.process_timeout,
                extra_body=adapter.extra_body,
            )
        )

    result = {
        "eval": f"lab_bench_{args.config.lower()}",
        "dataset": f"futurehouse/lab-bench {args.config}",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "adapter_file": args.adapter_file,
        "adapter_settings": {m: adapters.public_adapter_settings(m, adapter_specs) for m in models},
        "item_count": len(items),
        "items": items,
        "responses": sorted(responses, key=lambda r: (r["model"], r["id"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
