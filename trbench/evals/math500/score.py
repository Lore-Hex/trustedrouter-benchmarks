"""Score a MATH-500 run with the vendored Hendrycks equivalence checker (no judge).

The model is asked to put its final answer in \\boxed{}. We read the last
\\boxed{}, then compare to the gold answer with `is_equiv` (LaTeX normalization).
A missing/errored/unboxed response counts as wrong. Headline is accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trbench import report
from trbench.evals.math500.equiv import is_equiv
from trbench.mathutil import boxed_answer


def score_model(responses: list[dict]) -> dict[str, Any]:
    total = len(responses)
    correct = 0
    errors = 0
    no_box = 0
    for r in responses:
        if r.get("error"):
            errors += 1
            continue
        pred = boxed_answer(str(r.get("text", "")))
        if pred is None:
            no_box += 1
            continue
        if is_equiv(pred, str(r.get("target", ""))):
            correct += 1
    return {
        "score": round(100.0 * correct / total, 1) if total else 0.0,
        "correct": correct,
        "total": total,
        "no_answer": no_box,
        "errors": errors,
    }


def summarize(result: dict[str, Any]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict]] = {}
    for r in result.get("responses", []):
        by_model.setdefault(str(r.get("model")), []).append(r)
    rows = [{"model": m, **score_model(rs)} for m, rs in by_model.items()]
    rows.sort(key=lambda r: (-float(r["score"]), int(r["errors"]), r["model"]))
    return rows


COLUMNS = [
    ("Model", "model"),
    ("Accuracy", "score"),
    ("Correct", "correct"),
    ("Total", "total"),
    ("No answer", "no_answer"),
    ("Errors", "errors"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a MATH-500 results JSON.")
    parser.add_argument("results")
    parser.add_argument("--svg", default="assets/math500.svg")
    parser.add_argument("--readme", default=None, help="If set, splice the table+chart into this README.")
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = summarize(result)
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    svg = report.svg_bar_chart(
        rows,
        score_key="score",
        max_score=100,
        title="MATH-500 on TrustedRouter",
        subtitle="Competition math, Hendrycks answer-equivalence (no judge). Higher is better.",
        label_suffix="%",
    )
    svg_path = Path(args.svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    print(f"wrote {svg_path}")

    if args.readme:
        rp = Path(args.readme)
        created = str(result.get("created_at", "unknown"))
        host = str(result.get("base_url_host", "unknown"))
        n = result.get("item_count", "?")
        block = "\n\n".join(
            [
                f"MATH-500 snapshot: `{created}` via `{host}`. {n} problems, "
                f"{len(rows)} models. Vendored Hendrycks answer-equivalence (no judge).",
                f"![MATH-500 chart]({svg_path.as_posix()})",
                table,
            ]
        )
        rp.write_text(report.splice_readme(rp.read_text(encoding="utf-8"), "MATH500_RESULTS", block), encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
