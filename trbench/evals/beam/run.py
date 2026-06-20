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
    # BEAM answers are concise, but heavy reasoners (opus-4.8) spend the budget
    # thinking and truncate to an empty final answer at 1-2K — that grades 0 and
    # wrongly sinks the strongest models. 8192 gives reasoning room; output cost is
    # tiny next to the 127K-token input anyway.
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300.0)
    # Lower concurrency: each request carries a 128K context
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--questions-per-type", type=int, default=1,
        help="Questions per (conversation, type) pair. Default=1 ≈ 200 items total. Use 0 for all.",
    )
    parser.add_argument("--out", default="results/beam_128k.json")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (model,id) pairs already in the sidecar JSONL.")
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    models = resolve_panel(args.models)
    qpt = args.questions_per_type if args.questions_per_type > 0 else None
    items = load(limit=args.limit, questions_per_type=qpt)
    print(f"beam-128k: {len(models)} models × {len(items)} questions", flush=True)

    # Per-item sidecar JSONL: each answer is appended immediately, so a killed run
    # loses nothing and --resume skips done pairs (the 127K-context run is long).
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".jsonl")
    done_keys: set[tuple[str, str]] = set()
    responses: list[dict] = []
    if args.resume and sidecar.exists():
        errors_skipped = 0
        best: dict[tuple[str, str], dict] = {}
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Re-run only TRANSPORT errors (402 credit-out, 502, timeout) — those
            # aren't the model's answer. Do NOT re-run a clean-but-empty response:
            # an empty IS the model's output (e.g. it declined to answer), and
            # BEAM's grader scores a non-responsive answer 0. Retrying empties would
            # overwrite a genuine decline with a forced (often confabulated) answer.
            if row.get("error"):
                errors_skipped += 1
                continue
            best[(row["model"], row["id"])] = row
        for key, row in best.items():
            done_keys.add(key)
            responses.append(row)
        print(f"  resume: {len(done_keys)} good answers recorded, {errors_skipped} errored rows will re-run",
              flush=True)

    sc = sidecar.open("a", encoding="utf-8")
    for model in models:
        todo = [it for it in items if (model, it["id"]) not in done_keys]
        print(f"  answering: {model} ({len(todo)}/{len(items)} to do)", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = {
                pool.submit(
                    _answer_one, it, model,
                    args.base_url, api_key, args.max_tokens, args.timeout,
                ): it
                for it in todo
            }
            for done, f in enumerate(as_completed(futs), 1):
                row = f.result()
                responses.append(row)
                sc.write(json.dumps(row, ensure_ascii=False) + "\n")
                sc.flush()
                if done % 10 == 0 or done == len(todo):
                    print(f"    {done}/{len(todo)}", flush=True)
    sc.close()

    # Filter to exactly (requested models × requested items): a resumed/--limit run
    # must not leak stale sidecar rows (a model dropped from --models, or a model's
    # extra answers from a larger prior --limit) into the result — that would score
    # models on inconsistent item sets. Keep only the current run's (model,id) grid.
    wanted = {(m, it["id"]) for m in models for it in items}
    kept = [r for r in responses if (r["model"], r["id"]) in wanted]
    result = {
        "eval": "beam_128k",
        "dataset": "Mohammadta/BEAM (100K split)",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "item_count": len(items),
        "responses": sorted(kept, key=lambda r: (r["model"], r["id"])),
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
