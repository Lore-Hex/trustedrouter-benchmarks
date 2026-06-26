#!/usr/bin/env python3
"""Probe TrustedRouter provider-pinned routes for a single model."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Any

from trustedrouter import TrustedRouter

from trbench import client


def _extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _finish_reason(data: dict[str, Any]) -> str | None:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return None


def _worker(queue: Any, args: argparse.Namespace, provider: str) -> None:
    started = time.monotonic()
    try:
        api_key = client.api_key_from_env(args.api_key)
        cli = TrustedRouter(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.request_timeout,
            max_retries=0,
        )
        data = cli.request(
            "POST",
            "/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": args.prompt}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "provider": {"only": [provider]},
            },
        )
        queue.put(
            {
                "provider": provider,
                "ok": True,
                "finish_reason": _finish_reason(data),
                "text": _extract_text(data),
                "usage": data.get("usage"),
                "wall_ms": round((time.monotonic() - started) * 1000),
            }
        )
    except Exception as exc:  # noqa: BLE001 - probe reports route behavior.
        queue.put(
            {
                "provider": provider,
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "wall_ms": round((time.monotonic() - started) * 1000),
            }
        )


def probe(args: argparse.Namespace, provider: str) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker, args=(queue, args, provider))
    proc.start()
    proc.join(args.process_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(3)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "provider": provider,
            "ok": False,
            "error": f"process_timeout_after_{args.process_timeout}s",
        }
    if not queue.empty():
        return queue.get()
    return {"provider": provider, "ok": False, "error": f"worker_exited_{proc.exitcode}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="moonshotai/kimi-k2.6")
    parser.add_argument(
        "--providers",
        default="kimi,parasail,phala,together,tinfoil,novita,fireworks,baseten,wafer",
    )
    parser.add_argument("--base-url", default=os.environ.get("TRUSTEDROUTER_BASE_URL", client.DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--process-timeout", type=float, default=75)
    parser.add_argument("--out", default="results/kimi_k26_provider_probe.json")
    parser.add_argument("--prompt", default="Return only the letter B. Do not include reasoning.")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    rows = [probe(args, provider) for provider in providers]
    for row in rows:
        text = repr(str(row.get("text", ""))[:80])
        print(
            f"{row['provider']}: ok={row.get('ok')} finish={row.get('finish_reason')} "
            f"wall_ms={row.get('wall_ms')} text={text} error={row.get('error')}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "prompt": args.prompt, "rows": rows}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
