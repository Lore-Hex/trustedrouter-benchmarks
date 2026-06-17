"""Shared run + judge loop for short-form factuality evals (SimpleQA family).

An item is {id, question, target}. Models answer closed-book (no system prompt,
no tools — plain chat completions), and an LLM judge grades each answer with the
canonical SimpleQA rubric. Reuses across SimpleQA Verified and Chinese SimpleQA.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from trbench import client, judge


def run_panel(
    items: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    models: list[str],
    max_tokens: int = 8192,
    timeout: float = 90.0,
    concurrency: int = 8,
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for model in models:
        print(f"  answering: {model} ({len(items)} questions)")

        def one(it: dict[str, Any], model: str = model) -> dict[str, Any]:
            r = client.chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=[{"role": "user", "content": it["question"]}],
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=timeout,
            )
            row = {"model": model, "id": it["id"], "question": it["question"], "target": it["target"]}
            if r.get("error"):
                row["error"] = r["error"]
            else:
                row["text"] = r.get("text", "")
            return row

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futs = [pool.submit(one, it) for it in items]
            for f in as_completed(futs):
                responses.append(f.result())
    return responses


def score_panel(
    result: dict[str, Any],
    *,
    judge_model: str,
    base_url: str,
    api_key: str,
    concurrency: int = 8,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict]] = {}
    for r in result.get("responses", []):
        by_model.setdefault(str(r.get("model")), []).append(r)

    rows: list[dict[str, Any]] = []
    for model, resps in by_model.items():
        answered = [r for r in resps if not r.get("error")]
        errors = len(resps) - len(answered)
        grades: list[str] = []

        def judge_one(r: dict) -> str:
            return judge.grade(
                r["question"], r["target"], r.get("text", ""),
                judge_model=judge_model, base_url=base_url, api_key=api_key, timeout=timeout,
            )

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futs = [pool.submit(judge_one, r) for r in answered]
            for f in as_completed(futs):
                grades.append(f.result())

        correct = grades.count("CORRECT")
        incorrect = grades.count("INCORRECT")
        not_att = grades.count("NOT_ATTEMPTED") + grades.count("JUDGE_ERROR")
        rows.append(
            {"model": model, **judge.fscore(correct, incorrect, not_att), "completed": len(answered), "errors": errors}
        )

    rows.sort(key=lambda r: (-float(r["f1"]), -float(r["correct"]), int(r["errors"]), r["model"]))
    return rows


COLUMNS = [
    ("Model", "model"),
    ("F-score", "f1"),
    ("Correct%", "correct"),
    ("Attempted%", "attempted"),
    ("Acc|attempted", "given_attempted"),
    ("Errors", "errors"),
]
