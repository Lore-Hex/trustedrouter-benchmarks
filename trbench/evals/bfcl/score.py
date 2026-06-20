"""Score a BFCL run with the vendored canonical AST checker + render report.

AST categories (simple/multiple/parallel/parallel_multiple): the decoded
tool_calls are matched against the ground-truth possible-answer set by BFCL's own
`ast_checker`. Relevance categories (irrelevance): correct = the model made NO
tool call. Accuracy is reported per category and overall (BFCL's headline metric).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from trbench import report
from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker


def _grade_one(resp: dict) -> bool:
    if resp.get("error"):
        return False
    decoded = resp.get("decoded") or []

    if resp.get("checker_kind") == "relevance":
        # irrelevance: correct iff the model declined to call any function.
        return len(decoded) == 0

    # AST: needs at least one decoded call and a ground truth to match.
    gt = resp.get("ground_truth")
    if not decoded or not gt:
        return False
    try:
        out = ast_checker(
            resp["functions"], decoded, gt,
            Language.PYTHON, resp["ast_test_category"], resp["model"],
        )
    except Exception:  # noqa: BLE001 - a checker crash = not a valid call
        return False
    return bool(out.get("valid"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score + report a BFCL run.")
    parser.add_argument("results")
    parser.add_argument("--svg", default="assets/bfcl.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    categories = result.get("categories", [])

    # per (model, category): [n_correct, n_total]; plus per-model errors.
    by_model_cat: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    errors: dict[str, int] = defaultdict(int)
    for resp in result.get("responses", []):
        model, cat = resp["model"], resp["category"]
        if resp.get("error"):
            errors[model] += 1
        ok = _grade_one(resp)
        by_model_cat[(model, cat)][0] += int(ok)
        by_model_cat[(model, cat)][1] += 1

    rows = []
    for model in result.get("models", []):
        total_c = total_n = 0
        per_cat: dict[str, float] = {}
        for cat in categories:
            c, n = by_model_cat.get((model, cat), [0, 0])
            per_cat[cat] = round(100 * c / n, 1) if n else float("nan")
            total_c += c
            total_n += n
        rows.append({
            "model": model,
            "overall": round(100 * total_c / total_n, 1) if total_n else 0.0,
            "errors": errors.get(model, 0),
            **{cat: per_cat[cat] for cat in categories},
        })

    rows.sort(key=lambda r: -r["overall"])
    COLUMNS = [("Model", "model"), ("Overall", "overall")] + \
              [(c.replace("_python", "").replace("parallel_multiple", "par_mult"), c) for c in categories] + \
              [("Errors", "errors")]
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    svg = report.svg_bar_chart(
        rows, score_key="overall", max_score=100,
        title="BFCL v4 (function-calling) on TrustedRouter",
        subtitle="Single-turn tool-calling accuracy (BFCL's vendored AST checker). Higher is better.",
    )
    Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.svg).write_text(svg, encoding="utf-8")
    print(f"wrote {args.svg}")

    if args.readme:
        rp = Path(args.readme)
        block = "\n\n".join([
            f"BFCL v4 snapshot: `{result.get('created_at','?')}`. {result.get('item_count','?')} single-turn "
            f"items ({', '.join(categories)}). Function schema sent as OpenAI tools; graded by BFCL's own "
            f"vendored AST checker (no LLM judge). Overall = accuracy across categories.",
            f"![BFCL chart]({Path(args.svg).as_posix()})",
            table,
        ])
        rp.write_text(report.splice_readme(rp.read_text(encoding="utf-8"), "BFCL_RESULTS", block), encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
