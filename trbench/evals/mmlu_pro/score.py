"""Score an MMLU-Pro run by extracting the chosen letter (no judge).

We pull the final answer letter out of the model's CoT — primarily from
"The answer is (X)", with the same fallbacks the official MMLU-Pro harness uses —
and compare to gold. Missing/errored/unparseable responses count as wrong.
Headline is accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from trbench import report

# Tried in order; we take the LAST match of the first pattern that hits, so a
# model that restates options before committing still scores on its final pick.
_PATTERNS = [
    re.compile(r"answer is\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"answer:\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"\(([A-J])\)\s*[.]?\s*$"),
    re.compile(r"\b([A-J])\b\s*[.)]?\s*$"),
]


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    for pat in _PATTERNS:
        matches = pat.findall(text)
        if matches:
            return matches[-1].upper()
    # last resort: the last parenthesized capital letter anywhere
    paren = re.findall(r"\(([A-J])\)", text)
    return paren[-1] if paren else None


def score_model(responses: list[dict]) -> dict[str, Any]:
    total = len(responses)
    correct = errors = no_answer = 0
    for r in responses:
        if r.get("error"):
            errors += 1
            continue
        pred = extract_letter(str(r.get("text", "")))
        if pred is None:
            no_answer += 1
            continue
        if pred == str(r.get("answer", "")).strip().upper():
            correct += 1
    return {
        "score": round(100.0 * correct / total, 1) if total else 0.0,
        "correct": correct,
        "total": total,
        "no_answer": no_answer,
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
    parser = argparse.ArgumentParser(description="Score an MMLU-Pro results JSON.")
    parser.add_argument("results")
    parser.add_argument("--svg", default="assets/mmlu_pro.svg")
    parser.add_argument("--readme", default=None, help="If set, splice the table+chart into this README.")
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = summarize(result)
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    svg = report.svg_bar_chart(
        rows, score_key="score", max_score=100,
        title="MMLU-Pro on TrustedRouter",
        subtitle="Ten-choice knowledge/reasoning MCQ, CoT letter-match (no judge). Higher is better.",
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
                f"MMLU-Pro snapshot: `{created}` via `{host}`. {n} questions, "
                f"{len(rows)} models. Zero-shot CoT, deterministic letter-match (no judge).",
                f"![MMLU-Pro chart]({svg_path.as_posix()})",
                table,
            ]
        )
        rp.write_text(report.splice_readme(rp.read_text(encoding="utf-8"), "MMLU_PRO_RESULTS", block), encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
