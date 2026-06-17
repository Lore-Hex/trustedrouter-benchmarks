"""Score a GSM8K run with a deterministic numeric matcher (no judge).

The model is asked to end with ``#### <answer>``. We read the number after the
last ``####``; if there is none we fall back to the last number anywhere in the
output (the standard "flexible extract"). A missing/errored response counts as
wrong. Headline is exact-match accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from trbench import report

_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize_num(s: str) -> str:
    """Canonicalize a numeric string: drop ``$ , %`` and trailing zeros so
    ``1,000`` / ``$1000`` / ``1000.0`` all compare equal to ``1000``."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else repr(f)


def extract_pred(text: str) -> str:
    """Pull the predicted answer out of a model's output."""
    if not text:
        return ""
    if "####" in text:
        m = _NUM.search(text.split("####")[-1])
        if m:
            return normalize_num(m.group())
    nums = _NUM.findall(text)
    return normalize_num(nums[-1]) if nums else ""


def score_model(responses: list[dict]) -> dict[str, Any]:
    total = len(responses)
    correct = 0
    errors = 0
    for r in responses:
        if r.get("error"):
            errors += 1
            continue
        if extract_pred(str(r.get("text", ""))) == normalize_num(str(r.get("target", ""))):
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
    parser = argparse.ArgumentParser(description="Score a GSM8K results JSON.")
    parser.add_argument("results")
    parser.add_argument("--svg", default="assets/gsm8k.svg")
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
        title="GSM8K on TrustedRouter",
        subtitle="Grade-school math word problems, exact-match accuracy (no judge). Higher is better.",
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
                f"GSM8K snapshot: `{created}` via `{host}`. {n} problems, "
                f"{len(rows)} models. Deterministic numeric match (no judge).",
                f"![GSM8K chart]({svg_path.as_posix()})",
                table,
            ]
        )
        rp.write_text(report.splice_readme(rp.read_text(encoding="utf-8"), "GSM8K_RESULTS", block), encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
