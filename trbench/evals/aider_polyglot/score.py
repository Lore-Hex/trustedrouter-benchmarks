"""Score an Aider polyglot run by running the real Exercism unit tests.

SECURITY: this executes model-generated Python via pytest in a temp dir. There
is no sandbox beyond the temp dir — run only on a throwaway machine/VM. (A
Docker-isolated runner is the right production answer; noted in REPORT.md.)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from trbench import report
from trbench.evals.aider_polyglot.loader import load


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


def run_tests(code: str, stub_name: str, test_name: str, test_code: str, timeout: float = 30.0) -> bool:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / stub_name).write_text(code, encoding="utf-8")
        (d / test_name).write_text(test_code, encoding="utf-8")
        try:
            r = subprocess.run(  # noqa: S603
                [sys.executable, "-m", "pytest", "-q", test_name],
                cwd=td, capture_output=True, timeout=timeout, text=True,
            )
            return r.returncode == 0
        except (subprocess.TimeoutExpired, Exception):  # noqa: BLE001
            return False


def summarize(result: dict[str, Any], concurrency: int = 6) -> list[dict[str, Any]]:
    ex = {e["id"]: e for e in load()}
    by_model: dict[str, list[dict]] = {}
    for r in result.get("responses", []):
        by_model.setdefault(str(r.get("model")), []).append(r)

    rows: list[dict[str, Any]] = []
    for model, resps in by_model.items():
        total = len(resps)
        errors = sum(1 for r in resps if r.get("error"))
        runnable = [r for r in resps if not r.get("error") and r["id"] in ex]

        def test_one(r: dict) -> bool:
            e = ex[r["id"]]
            return run_tests(extract_code(r.get("text", "")), e["stub_name"], e["test_name"], e["test_code"])

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            passed = sum(1 for ok in pool.map(test_one, runnable) if ok)
        rows.append(
            {
                "model": model,
                "pass_rate": round(100 * passed / total, 1) if total else 0.0,
                "passed": passed,
                "total": total,
                "errors": errors,
            }
        )
    rows.sort(key=lambda r: (-float(r["pass_rate"]), int(r["errors"]), r["model"]))
    return rows


COLUMNS = [("Model", "model"), ("Pass%", "pass_rate"), ("Passed", "passed"), ("Total", "total"), ("Errors", "errors")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an Aider polyglot (Python) run via real unit tests.")
    parser.add_argument("results")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--svg", default="assets/aider_polyglot.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = summarize(result, concurrency=args.concurrency)
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    svg = report.svg_bar_chart(
        rows, score_key="pass_rate", max_score=100,
        title="Aider polyglot (Python) on TrustedRouter",
        subtitle="Pass@1 on Exercism Python exercises, real unit tests. Higher is better.",
    )
    Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.svg).write_text(svg, encoding="utf-8")
    print(f"wrote {args.svg}")

    if args.readme:
        rp = Path(args.readme)
        created = str(result.get("created_at", "unknown"))
        n = result.get("item_count", "?")
        block = "\n\n".join(
            [
                f"Aider polyglot (Python subset) snapshot: `{created}`. {n} Exercism exercises, "
                "pass@1, real unit tests (no judge).",
                f"![Aider polyglot chart]({Path(args.svg).as_posix()})",
                table,
            ]
        )
        spliced = report.splice_readme(rp.read_text(encoding="utf-8"), "AIDER_POLYGLOT_RESULTS", block)
        rp.write_text(spliced, encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
