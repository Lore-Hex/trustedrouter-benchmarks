"""Higher-N confirmation of the fusion SELECTION prompt finding.

Expands beyond the 40 parallel items to all four live (real-user-query) AST
categories, capped per category. Gathers a clean 5-model panel (low concurrency,
cached), classifies decidable items, and re-runs the top prompts to confirm
`candidates_as_hints` still hits the oracle and beats best-solo at higher N.

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_confirm.py [--limit-per-cat 40]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from trbench import client
from trbench.evals.bfcl.loader import load
from trbench.evals.bfcl.run import decode_tool_calls
from trbench.evals.bfcl.schema_convert import to_openai_tools
from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

PANEL = ["minimax/minimax-m3", "moonshotai/kimi-k2.6", "z-ai/glm-5.2", "google/gemma-4-31b-it", "deepseek/deepseek-v4-pro"]
SELECTOR = "z-ai/glm-5.2"
CATEGORIES = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
CACHE = "results/_confirm_panel.json"

PROMPTS = {
    "candidates_as_hints": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nThese candidate answers are HINTS (some may be wrong):\n{opts}\nUse them to inform your thinking, but independently decide the correct tool call(s) and argument values for the request. Then output the index of the candidate matching the correct answer. Reply ONLY JSON {{\"choice\": <index>}}.",
    "fresh_strict": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nIgnore the candidates. From the request ALONE, write the exact correct tool call(s) and argument values. THEN read these candidates:\n{opts}\nOutput the index of the candidate that exactly matches your independent answer (closest if none exact). Reply ONLY JSON {{\"choice\": <index>}}.",
    "verify_each": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nCANDIDATES:\n{opts}\n\nFor EACH candidate, verify: (a) correct function(s), (b) every argument value correct and literally supported by the request, (c) no missing/extra calls. Pick the candidate that passes most strictly. Reply ONLY JSON {{\"choice\": <index>}}.",
    "right_fn_args": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nCANDIDATES:\n{opts}\n\nWhich candidate calls the right function(s) with the right arguments for THIS request? Reply ONLY JSON {{\"choice\": <index>}}.",
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


def distinct(cands):
    out, seen = [], set()
    for c in cands:
        k = canonical(c)
        if k not in seen:
            seen.add(k); out.append(c)
    return out


def ask(model, it, key):
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=model, messages=it["messages"],
                    tools=to_openai_tools(it["functions"]), max_tokens=4096, temperature=0.0, timeout=180)
    return [] if r.get("error") else decode_tool_calls(r.get("tool_calls"))


def select(it, cands, template, key):
    q = it["messages"][-1]["content"]
    schema = json.dumps([f.get("name") for f in it["functions"]])
    opts = "\n".join(f"[{i}] {json.dumps(c)}" for i, c in enumerate(cands))
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=SELECTOR,
                    messages=[{"role": "user", "content": template.format(q=q, schema=schema, opts=opts)}],
                    max_tokens=4096, temperature=0.0, timeout=180)
    m = re.search(r'"choice"\s*:\s*(\d+)', r.get("text", "") or "")
    idx = int(m.group(1)) if m else 0
    return cands[idx] if 0 <= idx < len(cands) else cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-cat", type=int, default=40)
    args = ap.parse_args()
    key = client.api_key_from_env()
    items = load(categories=CATEGORIES, limit_per_category=args.limit_per_cat)
    ids = {it["id"] for it in items}

    answers = json.loads(open(CACHE).read()) if os.path.exists(CACHE) else {}
    todo = [(m, it) for it in items for m in PANEL if m not in answers.get(it["id"], {})]
    if todo:
        print(f"incremental gather: {len(todo)} missing (model,item) of {len(items)*len(PANEL)} (reusing {len(items)*len(PANEL)-len(todo)})", flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {pool.submit(ask, m, it, key): (m, it["id"]) for (m, it) in todo}
            done = 0
            for f in as_completed(futs):
                m, iid = futs[f]
                answers.setdefault(iid, {})[m] = f.result()
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
        open(CACHE, "w").write(json.dumps(answers))
    else:
        print("fully cached, no gather")

    n = len(items)
    solo = {m: sum(grade(it, answers[it["id"]].get(m, [])) for it in items) for m in PANEL}
    oracle = sum(any(grade(it, answers[it["id"]].get(m, [])) for m in PANEL) for it in items)
    trivial = 0
    decidable = []
    for it in items:
        cands = distinct([answers[it["id"]].get(m, []) for m in PANEL])
        flags = [grade(it, c) for c in cands]
        if all(flags):
            trivial += 1
        elif any(flags):
            decidable.append((it, cands))
    print(f"\n=== HIGHER-N confirm: n={n}  trivial={trivial}  decidable={len(decidable)}  unwinnable={n-trivial-len(decidable)} ===")
    for m in PANEL:
        print(f"  solo {m:24} {100*solo[m]/n:5.1f}")
    print(f"  best solo                     {100*max(solo.values())/n:5.1f}")
    print(f"  ORACLE                        {100*oracle/n:5.1f}")
    print(f"\n=== selector={SELECTOR}: prompt sweep (decidable-acc / implied-overall) ===")
    res = []
    for name, tmpl in PROMPTS.items():
        with ThreadPoolExecutor(max_workers=3) as pool:
            correct = sum(f.result() for f in as_completed(
                [pool.submit(lambda it, cands: grade(it, select(it, cands, tmpl, key)), it, cands) for it, cands in decidable]))
        res.append((name, 100 * correct / len(decidable), 100 * (trivial + correct) / n))
    for name, dec, ov in sorted(res, key=lambda r: -r[1]):
        print(f"  {name:22} decidable {dec:5.1f}   implied_overall {ov:5.1f}")


if __name__ == "__main__":
    main()
