"""Score LAB-bench LitQA2 multiple-choice responses by letter extraction."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from trbench import report

_LETTER_RE = re.compile(r"\b([A-Z])\b")


def extract_letter(text: str, n_options: int) -> str | None:
    valid = {chr(ord("A") + i) for i in range(n_options)}
    stripped = text.strip().upper()
    if stripped in valid:
        return stripped
    matches = [m for m in _LETTER_RE.findall(stripped) if m in valid]
    return matches[-1] if matches else None


def summarize(result: dict[str, Any]) -> list[dict[str, Any]]:
    option_counts = {str(item["id"]): len(item["options"]) for item in result.get("items", [])}
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in result.get("responses", []):
        by_model.setdefault(str(row["model"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for model, responses in by_model.items():
        correct = errors = no_answer = 0
        input_tokens = output_tokens = 0
        latencies = []
        for row in responses:
            if row.get("error"):
                errors += 1
                continue
            pred = extract_letter(str(row.get("text", "")), option_counts[str(row["id"])])
            if pred is None:
                no_answer += 1
                continue
            if pred == str(row.get("answer", "")).upper():
                correct += 1
            usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
            input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            if isinstance(row.get("latency_ms"), int | float):
                latencies.append(float(row["latency_ms"]))
        total = len(responses)
        rows.append(
            {
                "model": model,
                "score": round(100 * correct / total, 1) if total else 0.0,
                "correct": correct,
                "total": total,
                "no_answer": no_answer,
                "errors": errors,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            }
        )
    rows.sort(key=lambda r: (-float(r["score"]), int(r["errors"]), r["model"]))
    return rows


COLUMNS = [
    ("Model", "model"),
    ("Accuracy", "score"),
    ("Correct", "correct"),
    ("Total", "total"),
    ("No answer", "no_answer"),
    ("Errors", "errors"),
    ("Input tok", "input_tokens"),
    ("Output tok", "output_tokens"),
    ("Avg ms", "avg_latency_ms"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a LAB-bench LitQA2 results JSON.")
    parser.add_argument("results")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = summarize(result)
    print(report.markdown_table(rows, COLUMNS))
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"source": args.results, "rows": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
