"""Score a LiveCodeBench run: extract each model's program, grade it in a
container against the problem's full test suite, report pass@1 (no judge).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from trbench import report
from trbench.evals.livecodebench.executor import extract_code, run_solution
from trbench.evals.livecodebench.loader import load


def _problems_by_id(result: dict[str, Any]) -> dict[str, dict]:
    # Same load_limit as the run, so we read the same cached problem set it graded.
    items = load(
        version=result.get("version", "v6"),
        min_date=result.get("min_date", "2024-08-01"),
        limit=result.get("load_limit"),
    )
    wanted = set(result.get("item_ids") or [])
    return {it["id"]: it for it in items if not wanted or it["id"] in wanted}


def grade(result: dict[str, Any], *, concurrency: int) -> list[dict[str, Any]]:
    problems = _problems_by_id(result)

    def grade_one(r: dict) -> tuple[str, str]:
        """-> (model, outcome) where outcome in {pass, fail, error, no_code}."""
        if r.get("error"):
            return r["model"], "error"
        prob = problems.get(r["id"])
        if prob is None:
            return r["model"], "error"
        code = extract_code(str(r.get("text", "")))
        if not code:
            return r["model"], "no_code"
        verdict = run_solution(code, prob["test_cases"])
        return r["model"], ("pass" if verdict.get("fail") is None else "fail")

    tallies: dict[str, dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(grade_one, r) for r in result.get("responses", [])]
        for fut in as_completed(futures):
            model, outcome = fut.result()
            t = tallies.setdefault(model, {"pass": 0, "fail": 0, "error": 0, "no_code": 0})
            t[outcome] += 1

    rows = []
    for model, t in tallies.items():
        total = sum(t.values())
        rows.append({
            "model": model,
            "score": round(100.0 * t["pass"] / total, 1) if total else 0.0,
            "solved": t["pass"], "total": total, "no_code": t["no_code"], "errors": t["error"],
        })
    rows.sort(key=lambda r: (-float(r["score"]), int(r["errors"]), r["model"]))
    return rows


COLUMNS = [
    ("Model", "model"),
    ("pass@1", "score"),
    ("Solved", "solved"),
    ("Total", "total"),
    ("No code", "no_code"),
    ("Errors", "errors"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a LiveCodeBench results JSON (containerized grading).")
    parser.add_argument("results")
    parser.add_argument("--concurrency", type=int, default=6, help="Parallel grading containers.")
    parser.add_argument("--svg", default="assets/livecodebench.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = grade(result, concurrency=args.concurrency)
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    svg = report.svg_bar_chart(
        rows, score_key="score", max_score=100,
        title="LiveCodeBench on TrustedRouter",
        subtitle="Contamination-windowed coding, containerized pass@1 (real execution). Higher is better.",
        label_suffix="%",
    )
    svg_path = Path(args.svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    print(f"wrote {svg_path}")

    if args.readme:
        rp = Path(args.readme)
        created = str(result.get("created_at", "unknown"))
        n = result.get("item_count", "?")
        ds = str(result.get("dataset", ""))
        block = "\n\n".join(
            [
                f"LiveCodeBench snapshot: `{created}`. {n} problems ({ds}). "
                f"Solutions executed in locked-down containers; pass@1 (no judge).",
                f"![LiveCodeBench chart]({svg_path.as_posix()})",
                table,
            ]
        )
        spliced = report.splice_readme(rp.read_text(encoding="utf-8"), "LIVECODEBENCH_RESULTS", block)
        rp.write_text(spliced, encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
