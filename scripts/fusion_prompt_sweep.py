"""Prompt sweep for fusion SELECTION on a 2-model panel {kimi-k2.6, glm-5.2}.

The clean-data finding: every selection strategy underperformed the best solo on
tool-calling because the selector picks WRONG on the items where the panel
disagrees. With a 2-model panel every disagreement is a binary choice, so this
isolates the one thing that matters: can a better SELECTOR PROMPT pick the right
candidate on disagreements? We sweep prompt variants (glm-5.2 as selector) and
score each vs best-solo and the 2-model oracle.

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_prompt_sweep.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from trbench import client
from trbench.evals.bfcl.loader import load
from trbench.evals.bfcl.run import decode_tool_calls
from trbench.evals.bfcl.schema_convert import to_openai_tools
from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

PANEL = ["minimax/minimax-m3", "moonshotai/kimi-k2.6", "z-ai/glm-5.2", "google/gemma-4-31b-it", "deepseek/deepseek-v4-pro"]
SELECTOR = "z-ai/glm-5.2"
CATEGORIES = ["live_parallel", "live_parallel_multiple"]
CACHE = "results/_prompt_sweep5_panel.json"

# Each prompt gets {q} (request), {schema} (function names), {opts} (candidates).
# All must end by asking for JSON {"choice": <index>}.
PROMPTS = {
    "terse": (
        "User request:\n{q}\n\nCandidates:\n{opts}\n\n"
        "Which candidate is the most correct? Reply ONLY JSON {{\"choice\": <index>}}."
    ),
    "right_fn_args": (
        "Select the most correct set of tool calls.\nAvailable functions: {schema}\n\n"
        "USER REQUEST:\n{q}\n\nCANDIDATES:\n{opts}\n\n"
        "Which candidate calls the right function(s) with the right arguments for THIS request? "
        "Reply ONLY JSON {{\"choice\": <index>}}."
    ),
    "diff_focused": (
        "Two candidates propose tool calls for a request; they may differ on which function or "
        "which argument value.\nAvailable functions: {schema}\n\nUSER REQUEST:\n{q}\n\nCANDIDATES:\n{opts}\n\n"
        "Identify exactly where the candidates DIFFER (function or argument value), decide which reading "
        "is correct for THIS request, and pick that candidate. Reply ONLY JSON {{\"choice\": <index>}}."
    ),
    "independent_then_match": (
        "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\n"
        "First, independently work out the correct tool call(s) and argument values for the request "
        "(ignore the candidates). Then compare to these candidates:\n{opts}\n\n"
        "Pick the candidate that matches your independent answer. Reply ONLY JSON {{\"choice\": <index>}}."
    ),
    "verify_each": (
        "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nCANDIDATES:\n{opts}\n\n"
        "For EACH candidate, verify: (a) correct function(s) chosen, (b) every argument value correct and "
        "literally supported by the request, (c) no missing or extra calls. Then pick the candidate that "
        "passes most strictly. Reply ONLY JSON {{\"choice\": <index>}}."
    ),
}


def canonical(d):
    return json.dumps(sorted(json.dumps(c, sort_keys=True) for c in (d or [])))


def grade(it, dec):
    if not dec or not it.get("ground_truth"):
        return False
    try:
        return bool(ast_checker(it["functions"], dec, it["ground_truth"], Language.PYTHON, it["ast_test_category"], "x")["valid"])
    except Exception:
        return False


def ask(model, it, key):
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=model, messages=it["messages"],
                    tools=to_openai_tools(it["functions"]), max_tokens=4096, temperature=0.0, timeout=180)
    return [] if r.get("error") else decode_tool_calls(r.get("tool_calls"))


def select(it, cands, template, key):
    q = it["messages"][-1]["content"]
    schema = json.dumps([f.get("name") for f in it["functions"]])
    opts = "\n".join(f"[{i}] {json.dumps(c)}" for i, c in enumerate(cands))
    prompt = template.format(q=q, schema=schema, opts=opts)
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=SELECTOR,
                    messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.0, timeout=180)
    m = re.search(r'"choice"\s*:\s*(\d+)', r.get("text", "") or "")
    idx = int(m.group(1)) if m else 0
    return cands[idx] if 0 <= idx < len(cands) else cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    key = client.api_key_from_env()
    items = load(categories=CATEGORIES, limit_per_category=args.limit)

    answers = {}
    if os.path.exists(CACHE):
        answers = json.loads(open(CACHE).read())
    if set(answers) < {it["id"] for it in items}:
        answers = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {pool.submit(ask, m, it, key): (m, it["id"]) for it in items for m in PANEL}
            for f in as_completed(futs):
                m, iid = futs[f]
                answers.setdefault(iid, {})[m] = f.result()
        open(CACHE, "w").write(json.dumps(answers))
        print("  gathered fresh 2-model panel")
    else:
        print("  reused cached 2-model panel")

    n = len(items)
    solo = {m: sum(grade(it, answers[it["id"]].get(m, [])) for it in items) for m in PANEL}
    oracle = sum(any(grade(it, answers[it["id"]].get(m, [])) for m in PANEL) for it in items)
    # items where the two disagree (the only ones a selector can move)
    disagree = [it for it in items if canonical(answers[it["id"]].get(PANEL[0], [])) != canonical(answers[it["id"]].get(PANEL[1], []))]

    print(f"\n=== 2-model panel {PANEL}  (n={n}, disagreements={len(disagree)}) ===")
    for m in PANEL:
        print(f"  solo {m:24} {100*solo[m]/n:5.1f}")
    print(f"  best solo                     {100*max(solo.values())/n:5.1f}")
    print(f"  ORACLE (either right)         {100*oracle/n:5.1f}  <- ceiling")
    print(f"\n=== selector={SELECTOR}: prompt sweep ===")
    for name, tmpl in PROMPTS.items():
        def run(it):
            cands = [answers[it["id"]].get(m, []) for m in PANEL]
            distinct = []
            seen = set()
            for c in cands:
                k = canonical(c)
                if k not in seen:
                    seen.add(k); distinct.append(c)
            if len(distinct) == 1:
                return grade(it, distinct[0])
            return grade(it, select(it, distinct, tmpl, key))
        with ThreadPoolExecutor(max_workers=3) as pool:
            correct = sum(f.result() for f in as_completed([pool.submit(run, it) for it in items]))
        print(f"  {name:24} {100*correct/n:5.1f}")


if __name__ == "__main__":
    main()
