#!/usr/bin/env python3
"""Probe provider/request-path behavior on selected mixed-packet tasks."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Any

from trbench import client
from trbench.evals.mixed_bio_packet.run import load_tasks, score_response


def _worker(queue: Any, args: argparse.Namespace, model: str, provider: str | None, task_id: str, use_request: bool) -> None:
    started = time.monotonic()
    try:
        tasks = {task["id"]: task for task in load_tasks(include_aa_lcr=False)}
        task = tasks[task_id]
        extra_body: dict[str, Any] = {}
        if provider:
            extra_body["provider"] = {"only": [provider]}
        r = client.chat(
            base_url=args.base_url,
            api_key=client.api_key_from_env(args.api_key),
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            max_tokens=args.max_tokens,
            temperature=None if args.omit_temperature else 0.0,
            timeout=args.request_timeout,
            retries=0,
            extra_body=extra_body or None,
            use_request=use_request,
        )
        text = r.get("text", "") or ""
        row = {
            "model": model,
            "provider": provider,
            "task_id": task_id,
            "use_request": use_request,
            "ok": not bool(r.get("error")),
            "error": r.get("error"),
            "chars": len(text),
            "finish_reason": r.get("finish_reason"),
            "empty_reason": r.get("empty_reason"),
            "usage": r.get("usage"),
            "latency_ms": r.get("latency_ms"),
            "wall_ms": round((time.monotonic() - started) * 1000),
            "score": score_response(task, text),
            "text_preview": text[:500],
        }
        queue.put(row)
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "model": model,
                "provider": provider,
                "task_id": task_id,
                "use_request": use_request,
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "wall_ms": round((time.monotonic() - started) * 1000),
            }
        )


def probe(args: argparse.Namespace, model: str, provider: str | None, task_id: str, use_request: bool) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker, args=(queue, args, model, provider, task_id, use_request))
    proc.start()
    proc.join(args.process_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(3)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "model": model,
            "provider": provider,
            "task_id": task_id,
            "use_request": use_request,
            "ok": False,
            "error": f"process_timeout_after_{args.process_timeout}s",
        }
    if not queue.empty():
        return queue.get()
    return {
        "model": model,
        "provider": provider,
        "task_id": task_id,
        "use_request": use_request,
        "ok": False,
        "error": f"worker_exited_{proc.exitcode}",
    }


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--providers", default="", help="Comma providers. Empty means no provider pin.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--paths", default="stream,request", help="Comma list: stream,request.")
    parser.add_argument("--base-url", default=os.environ.get("TRUSTEDROUTER_BASE_URL", client.DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--process-timeout", type=float, default=110)
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    models = _split(args.models)
    providers = [None] if not args.providers else _split(args.providers)
    tasks = _split(args.tasks)
    paths = _split(args.paths)
    use_request_values = [path == "request" for path in paths]

    rows: list[dict[str, Any]] = []
    for model in models:
        for provider in providers:
            for task_id in tasks:
                for use_request in use_request_values:
                    row = probe(args, model, provider, task_id, use_request)
                    rows.append(row)
                    score = row.get("score") or {}
                    print(
                        f"{model} provider={provider or '-'} task={task_id} "
                        f"path={'request' if use_request else 'stream'} ok={row.get('ok')} "
                        f"chars={row.get('chars')} status={score.get('status')} "
                        f"score={score.get('score')} pred={score.get('pred')} "
                        f"lat={row.get('latency_ms') or row.get('wall_ms')} err={row.get('error')}",
                        flush=True,
                    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": models,
                "providers": providers,
                "tasks": tasks,
                "paths": paths,
                "max_tokens": args.max_tokens,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
