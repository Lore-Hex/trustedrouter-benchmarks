"""Judge a SimpleQA Verified run and render the report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trbench import client, report
from trbench.evals.qa_common import COLUMNS, score_panel

# GPT-4.1 is the published SimpleQA Verified autorater (arXiv:2509.07968); pairing
# it with the modified grader prompt in trbench.judge is what reproduces Google's
# reported scores. Override with --judge-model for a quick/cheaper re-grade.
DEFAULT_JUDGE = "openai/gpt-4.1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge + report a SimpleQA Verified run.")
    parser.add_argument("results")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--svg", default="assets/simpleqa_verified.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = score_panel(
        result,
        judge_model=args.judge_model,
        base_url=args.base_url,
        api_key=api_key,
        concurrency=args.concurrency,
    )
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    svg = report.svg_bar_chart(
        rows,
        score_key="f1",
        max_score=100,
        title="SimpleQA Verified on TrustedRouter",
        subtitle="Closed-book factuality F-score (no tools). Judge: " + args.judge_model + ". Higher is better.",
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
                f"SimpleQA Verified snapshot: `{created}`. {n} closed-book questions, no tools. "
                f"Judge: `{args.judge_model}`. F-score = harmonic mean of accuracy and accuracy-given-attempted.",
                f"![SimpleQA Verified chart]({Path(args.svg).as_posix()})",
                table,
            ]
        )
        spliced = report.splice_readme(rp.read_text(encoding="utf-8"), "SIMPLEQA_VERIFIED_RESULTS", block)
        rp.write_text(spliced, encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
