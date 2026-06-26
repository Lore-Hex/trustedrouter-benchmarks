"""Run direct model baselines on the mixed bio expanded packet.

This is for model slugs such as trustedrouter/synth or trustedrouter/fusion.
It scores the same 16 mixed-bio tasks plus the saved 10 LitQA2 tasks used by
the synthesis experiments.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trbench import adapters, client
from trbench.evals.mixed_bio_packet.run import load_tasks, public_task, score_response, summarize
from scripts.mixed_bio_taskiq_selector_eval import load_litqa2_tasks_and_rows


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _chat_model(
    *,
    model: str,
    task: dict[str, Any],
    adapter_specs: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    retries: int,
    use_request: bool,
) -> dict[str, Any]:
    adapter = adapters.adapter_for(model, adapter_specs, default_max_tokens=max_tokens, default_timeout=timeout)
    return client.chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": str(task.get("prompt") or task.get("question") or "")}],
        max_tokens=adapter.max_tokens or max_tokens,
        temperature=adapter.temperature,
        timeout=adapter.timeout or timeout,
        extra_body=adapter.extra_body,
        retries=retries,
        use_request=use_request or adapter.use_request,
    )


def run_one(
    *,
    model: str,
    task: dict[str, Any],
    adapter_specs: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    retries: int,
    use_request: bool,
) -> dict[str, Any]:
    result = _chat_model(
        model=model,
        task=task,
        adapter_specs=adapter_specs,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        use_request=use_request,
    )
    row: dict[str, Any] = {
        "model": model,
        "id": task["id"],
        "source": task["source"],
        "text": "" if result.get("error") else str(result.get("text", "")),
        "usage": result.get("usage", {}),
        "error": result.get("error"),
        "latency_ms": result.get("latency_ms"),
        "finish_reason": result.get("finish_reason"),
        "attempts": result.get("attempts"),
    }
    if result.get("empty_reason"):
        row["empty_reason"] = result["empty_reason"]
    if not row.get("error"):
        row.update(score_response(task, row["text"]))
    return row


def payload(
    *,
    args: argparse.Namespace,
    models: list[str],
    tasks: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "eval": "mixed_bio_direct_model",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "task_count": len(tasks),
        "tasks": [public_task(t) for t in tasks],
        "responses": sorted(responses, key=lambda r: (r["model"], r["id"])),
        "summary": summarize(responses),
        "notes": [
            "Direct model call on the expanded mixed bio packet.",
            "Expanded packet is 16 mixed-bio tasks plus the saved 10 LitQA2 tasks.",
            "Scoring uses the same deterministic scorers as the synthesis experiments.",
            "Existing rows in this output file are reused on rerun.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run direct model baselines on expanded mixed bio packet.")
    parser.add_argument("--models", default="trustedrouter/synth")
    parser.add_argument("--include-litqa2", action="store_true")
    parser.add_argument("--adapter-file", action="append", default=[])
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=client.DEFAULT_RETRIES)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--use-request", action="store_true", help="Use the non-streaming request path.")
    parser.add_argument("--out", default="results/mixed_bio_reasoning_packet_26_direct_models.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = load_tasks(include_aa_lcr=False)
    if args.include_litqa2:
        litqa_tasks, _ = load_litqa2_tasks_and_rows()
        tasks.extend(litqa_tasks)

    adapter_specs: dict[str, dict[str, Any]] = {}
    for adapter_file in args.adapter_file:
        adapter_specs.update(adapters.load_adapter_file(adapter_file))

    out = Path(args.out)
    responses: list[dict[str, Any]] = []
    if out.exists():
        existing = _load_json(str(out))
        responses = list(existing.get("responses", []))
        print(f"reusing {len(responses)} rows from {out}", flush=True)

    api_key = client.api_key_from_env(args.api_key)
    existing_keys = {(str(r.get("model")), str(r.get("id"))) for r in responses}
    pending = [(model, task) for model in models for task in tasks if (model, task["id"]) not in existing_keys]
    print(f"running direct models: {len(pending)} new / {len(models) * len(tasks)} total", flush=True)

    def one(model: str, task: dict[str, Any]) -> dict[str, Any]:
        return run_one(
            model=model,
            task=task,
            adapter_specs=adapter_specs,
            base_url=args.base_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            use_request=args.use_request,
        )

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(one, model, task) for model, task in pending]
            for fut in as_completed(futures):
                responses.append(fut.result())
                result = payload(args=args, models=models, tasks=tasks, responses=responses)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = payload(args=args, models=models, tasks=tasks, responses=responses)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
