"""Score an IFEval run with Google's canonical verifiers (vendored).

Reports the four standard IFEval numbers — prompt-level and instruction-level
accuracy under both strict and loose checking — plus their average as the
headline. A missing/errored response counts as "did not follow".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trbench import report
from trbench.evals.ifeval.run import DATA_PATH
from trbench.evals.ifeval.vendor import evaluation_lib as E

_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk  # noqa: PLC0415

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:  # noqa: BLE001, S110
            pass
    _NLTK_READY = True


def score_model(inputs: list[Any], responses: list[dict]) -> dict[str, Any]:
    _ensure_nltk()
    ptr: dict[str, str] = {}
    errors = 0
    for r in responses:
        if r.get("error"):
            errors += 1
            ptr.setdefault(r["prompt"], "")
        else:
            ptr[r["prompt"]] = r.get("text", "")
    # Any input the model never returned for at all counts as empty (a fail).
    for inp in inputs:
        ptr.setdefault(inp.prompt, "")

    n_prompt = len(inputs)
    n_inst = sum(len(inp.instruction_id_list) for inp in inputs)
    ps = pl = isf = ilf = 0
    for inp in inputs:
        strict = E.test_instruction_following_strict(inp, ptr)
        loose = E.test_instruction_following_loose(inp, ptr)
        ps += int(strict.follow_all_instructions)
        pl += int(loose.follow_all_instructions)
        isf += sum(strict.follow_instruction_list)
        ilf += sum(loose.follow_instruction_list)

    prompt_strict = 100.0 * ps / n_prompt
    prompt_loose = 100.0 * pl / n_prompt
    inst_strict = 100.0 * isf / n_inst
    inst_loose = 100.0 * ilf / n_inst
    headline = (prompt_strict + prompt_loose + inst_strict + inst_loose) / 4
    return {
        "score": round(headline, 1),
        "prompt_strict": round(prompt_strict, 1),
        "prompt_loose": round(prompt_loose, 1),
        "inst_strict": round(inst_strict, 1),
        "inst_loose": round(inst_loose, 1),
        "completed": n_prompt - errors,
        "errors": errors,
    }


def summarize(result: dict[str, Any]) -> list[dict[str, Any]]:
    all_inputs = E.read_prompt_list(str(DATA_PATH))
    # Score only over the prompts that were actually run (so --prompt-limit
    # subsets aren't divided by the full 541).
    run_keys = {r.get("key") for r in result.get("responses", [])}
    inputs = [i for i in all_inputs if i.key in run_keys] or all_inputs
    by_model: dict[str, list[dict]] = {}
    for r in result.get("responses", []):
        by_model.setdefault(str(r.get("model")), []).append(r)
    rows = [{"model": m, **score_model(inputs, rs)} for m, rs in by_model.items()]
    rows.sort(key=lambda r: (-float(r["score"]), int(r["errors"]), r["model"]))
    return rows


COLUMNS = [
    ("Model", "model"),
    ("IFEval", "score"),
    ("Prompt-strict", "prompt_strict"),
    ("Prompt-loose", "prompt_loose"),
    ("Inst-strict", "inst_strict"),
    ("Inst-loose", "inst_loose"),
    ("Errors", "errors"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an IFEval results JSON.")
    parser.add_argument("results")
    parser.add_argument("--svg", default="assets/ifeval.svg")
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
        title="IFEval on TrustedRouter",
        subtitle="Average of prompt/instruction strict+loose accuracy. Higher is better.",
        label_suffix="",
    )
    svg_path = Path(args.svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    print(f"wrote {svg_path}")

    if args.readme:
        rp = Path(args.readme)
        created = str(result.get("created_at", "unknown"))
        host = str(result.get("base_url_host", "unknown"))
        n = result.get("prompt_count", 541)
        block = "\n\n".join(
            [
                f"IFEval snapshot: `{created}` via `{host}`. {n} prompts, "
                f"{len(rows)} models. Deterministic Python verifiers (no judge).",
                f"![IFEval chart]({svg_path.as_posix()})",
                table,
            ]
        )
        rp.write_text(report.splice_readme(rp.read_text(encoding="utf-8"), "IFEVAL_RESULTS", block), encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
