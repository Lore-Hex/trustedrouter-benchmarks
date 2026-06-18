"""Score an AIME run with a deterministic integer matcher (no judge).

AIME answers are integers 0-999. We read the integer out of the last \\boxed{};
if there is none we fall back to the last integer in the output. A missing or
errored response counts as wrong. Headline is exact-match accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from trbench import report
from trbench.mathutil import boxed_answer

_INT = re.compile(r"-?\d+")


def _to_int(s: str) -> int | None:
    s = (s or "").replace(",", "").strip()
    m = _INT.search(s)
    return int(m.group()) if m else None


def extract_pred(text: str) -> int | None:
    """The model's integer answer: prefer the last \\boxed{}, else the last
    integer anywhere in the output."""
    if not text:
        return None
    boxed = boxed_answer(text)
    if boxed is not None:
        n = _to_int(boxed)
        if n is not None:
            return n
    nums = _INT.findall(text)
    return int(nums[-1]) if nums else None


def score_model(responses: list[dict]) -> dict[str, Any]:
    total = len(responses)
    correct = 0
    errors = 0
    for r in responses:
        if r.get("error"):
            errors += 1
            continue
        gold = _to_int(str(r.get("target", "")))
        if gold is not None and extract_pred(str(r.get("text", ""))) == gold:
            correct += 1
    return {
        "score": round(100.0 * correct / total, 1) if total else 0.0,
        "correct": correct,
        "total": total,
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
    ("Errors", "errors"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an AIME results JSON.")
    parser.add_argument("results")
    parser.add_argument("--svg", default="assets/aime.svg")
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
        title="AIME 2025 on TrustedRouter",
        subtitle="Competition math, integer answers, exact match (no judge). Higher is better.",
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
                f"AIME 2025 snapshot: `{created}` via `{host}`. {n} problems, "
                f"{len(rows)} models. Deterministic integer match (no judge).",
                f"![AIME chart]({svg_path.as_posix()})",
                table,
            ]
        )
        rp.write_text(report.splice_readme(rp.read_text(encoding="utf-8"), "AIME_RESULTS", block), encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
