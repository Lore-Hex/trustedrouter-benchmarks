"""Render a saved tau2-bench result (table + chart + README splice) without
re-running the panel. run.py already computes per-model pass^1/avg_reward into
the result JSON; this just re-renders it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trbench import report
from trbench.evals.tau2.run import COLUMNS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a tau2-bench results JSON.")
    parser.add_argument("results")
    parser.add_argument("--svg", default="assets/tau2.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = [r for r in result.get("results", []) if "error" not in r]
    rows.sort(key=lambda r: (-float(r["pass1"]), -float(r["avg_reward"]), r["model"]))
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    svg = report.svg_bar_chart(
        rows, score_key="pass1", max_score=100,
        title=f"tau2-bench ({result.get('domain', 'retail')}) on TrustedRouter",
        subtitle=f"Agentic tool-use, pass^1. User sim: {result.get('user_llm', 'gpt-4.1')}. Higher is better.",
    )
    Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.svg).write_text(svg, encoding="utf-8")
    print(f"wrote {args.svg}")

    if args.readme:
        rp = Path(args.readme)
        block = "\n\n".join([
            f"tau2-bench snapshot: `{result.get('created_at', 'unknown')}`. Domain "
            f"`{result.get('domain', 'retail')}`, {result.get('num_tasks', '?')} tasks x "
            f"{result.get('num_trials', 1)} trial(s), agent vs fixed user "
            f"`{result.get('user_llm', 'gpt-4.1')}`. Metric: pass^1 (task reward == 1). "
            f"Small-subset numbers run high; the ranking is the signal.",
            f"![tau2-bench chart]({Path(args.svg).as_posix()})",
            table,
        ])
        rp.write_text(report.splice_readme(rp.read_text(encoding="utf-8"), "TAU2_RESULTS", block), encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
