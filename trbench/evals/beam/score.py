"""Judge BEAM 128K answers using rubric-based LLM scoring and render report.

Scoring follows BEAM's methodology: for each rubric item, the judge assigns
0.0 / 0.5 / 1.0. The item score is the mean across rubric items. The model
score is the mean across all items (overall) and per question-type means.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from trbench import client, report
from trbench.evals.beam.loader import QUESTION_TYPES

# BEAM's canonical judge is gpt-4.1-mini (src/llm.py: gpt_llm_obj model_name="gpt-4.1-mini",
# temperature=0) — NOT the full gpt-4.1. Matching the judge model is part of reproducing
# the paper's numbers (a bigger/different judge grades more leniently).
DEFAULT_JUDGE = "openai/gpt-4.1-mini"

# BEAM's EXACT judge prompt, vendored verbatim (see beam_grader_prompt.py). Using
# the canonical grader — not a reimplementation — is what makes our scores
# comparable to the paper. BEAM replaces ONLY <rubric_item> and <llm_response> at
# call time (it leaves <question> a literal placeholder — replicated for fidelity).
from trbench.evals.beam.beam_grader_prompt import UNIFIED_LLM_JUDGE_PROMPT  # noqa: E402


def _judge_rubric_item(
    rubric_item: str,
    response: str,
    judge_model: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> float:
    # BEAM: prompt.replace("<rubric_item>", item).replace("<llm_response>", llm_response)
    prompt = UNIFIED_LLM_JUDGE_PROMPT.replace("<rubric_item>", rubric_item).replace("<llm_response>", response)
    r = client.chat(
        base_url=base_url, api_key=api_key, model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, temperature=0.0, timeout=timeout,
    )
    text = r.get("text", "") or ""
    m = re.search(r'"score"\s*:\s*(1(?:\.0+)?|0(?:\.\d+)?)', text)
    if m:
        v = float(m.group(1))
        return v if v in (0.0, 0.5, 1.0) else 0.0
    return 0.0  # BEAM defaults a non-parseable judgment toward no-compliance


def _score_response(
    resp: dict,
    judge_model: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> dict:
    rubric = resp.get("rubric", [])
    text = resp.get("text", "") or ""
    qtype = resp.get("question_type", "")

    # Transport error (402/502/timeout) = infra, not a model answer → exclude.
    if resp.get("error"):
        return {**resp, "rubric_scores": [], "item_score": 0.0, "is_error": True}

    # Empty model response is NOT excluded: BEAM's judge scores a non-responsive
    # answer 0 (and the paper's "explicit-only" abstention means silence fails).
    # Short-circuit empties to 0 across the rubric rather than spend judge calls.
    if not text.strip():
        return {**resp, "rubric_scores": [0.0] * len(rubric), "item_score": 0.0, "is_error": False}

    raw = [
        _judge_rubric_item(item, text, judge_model, base_url, api_key, timeout)
        for item in rubric
    ]
    # BEAM aggregation: 9 rubric types floor each item with int() (0.5 → 0, binary
    # full-compliance); event_ordering keeps the float. Mean across rubric items.
    if qtype == "event_ordering":
        item_score = sum(raw) / len(raw) if raw else 0.0
    else:
        item_score = sum(int(s) for s in raw) / len(raw) if raw else 0.0
    return {**resp, "rubric_scores": raw, "item_score": item_score, "is_error": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge + report BEAM 128K run.")
    parser.add_argument("results")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--svg", default="assets/beam_128k.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    responses = result.get("responses", [])

    print(f"Judging {len(responses)} responses with {args.judge_model}…")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(_score_response, r, args.judge_model, args.base_url, api_key, args.timeout): r
            for r in responses
        }
        scored: list[dict] = []
        for done, f in enumerate(as_completed(futs), 1):
            scored.append(f.result())
            if done % 50 == 0:
                print(f"  judged {done}/{len(responses)}")

    # Aggregate per model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        by_model[s["model"]].append(s)

    rows: list[dict] = []
    for model, items in sorted(by_model.items()):
        errors = sum(1 for it in items if it.get("is_error"))
        valid = [it for it in items if not it.get("is_error")]
        overall = sum(it["item_score"] for it in valid) / len(valid) * 100 if valid else 0.0

        per_type: dict[str, float] = {}
        for qtype in QUESTION_TYPES:
            typed = [it for it in valid if it.get("question_type") == qtype]
            per_type[qtype] = sum(it["item_score"] for it in typed) / len(typed) * 100 if typed else float("nan")

        rows.append({
            "model": model,
            "overall": round(overall, 1),
            "errors": errors,
            **{k: round(v, 1) for k, v in per_type.items()},
        })

    rows.sort(key=lambda r: -r["overall"])

    # Print table
    COLUMNS = [
        ("Model", "model"),
        ("Overall", "overall"),
        ("Info Extr.", "information_extraction"),
        ("Temporal", "temporal_reasoning"),
        ("Multi-hop", "multi_session_reasoning"),
        ("Abstention", "abstention"),
        ("Errors", "errors"),
    ]
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    # SVG bar chart
    svg = report.svg_bar_chart(
        rows,
        score_key="overall",
        max_score=100,
        title="BEAM 128K on TrustedRouter",
        subtitle="Long-context memory (128K). Rubric-based score avg across 10 categories. Judge: " + args.judge_model,
    )
    Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.svg).write_text(svg, encoding="utf-8")
    print(f"wrote {args.svg}")

    if args.readme:
        rp = Path(args.readme)
        created = str(result.get("created_at", "unknown"))
        n = result.get("item_count", "?")
        block = "\n\n".join([
            f"BEAM 128K snapshot: `{created}`. {n} probing questions across 20 conversations (~128K tokens each). "
            f"Rubric-based LLM judge ({args.judge_model}). Overall = mean across 10 memory-ability categories.",
            f"![BEAM 128K chart]({Path(args.svg).as_posix()})",
            table,
        ])
        spliced = report.splice_readme(rp.read_text(encoding="utf-8"), "BEAM_128K_RESULTS", block)
        rp.write_text(spliced, encoding="utf-8")
        print(f"updated {rp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
