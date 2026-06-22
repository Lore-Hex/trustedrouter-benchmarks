"""Fast creative-prompt sweep on the DECIDABLE subset only.

Reuses the cached clean 5-model panel. Classifies each item:
  - trivial    : every distinct candidate is correct (selector can't lose)
  - unwinnable : no candidate is correct (selector can't win — oracle 0)
  - decidable  : some candidates right, some wrong  <- the ONLY items that matter
We sweep many selector prompts on the decidable subset only (small -> fast). The
metric that matters is decidable-accuracy = fraction of decidable items where the
selector picks a correct candidate (oracle on this subset is 100% by construction).
implied_overall = (trivial + decidable_correct) / n, comparable to the full runs.

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_prompt_sweep2.py
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from trbench import client
from trbench.evals.bfcl.loader import load
from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

PANEL = ["minimax/minimax-m3", "moonshotai/kimi-k2.6", "z-ai/glm-5.2", "google/gemma-4-31b-it", "deepseek/deepseek-v4-pro"]
SELECTOR = "z-ai/glm-5.2"
CACHE = "results/_prompt_sweep5_panel.json"

PROMPTS = {
    # reference: the 85.0 winner (ignores candidates while solving)
    "fresh_strict": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nIgnore the candidates. From the request ALONE, write the exact correct tool call(s) and argument values. THEN read these candidates:\n{opts}\nOutput the index of the candidate that exactly matches your independent answer (closest if none exact). Reply ONLY JSON {{\"choice\": <index>}}.",
    # variants that FOLD THE CANDIDATES BACK IN to correct the fresh draft
    "fresh_then_crosscheck": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nStep 1 — solve it yourself: from the request alone, write the correct tool call(s) and argument values.\nStep 2 — cross-check against the candidates:\n{opts}\nDoes any candidate reveal a function or argument value you missed or got wrong? If a candidate is more correct than your draft, adopt its correction.\nStep 3 — output the index of the candidate that matches your final answer. Reply ONLY JSON {{\"choice\": <index>}}.",
    "candidates_as_hints": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nThese candidate answers are HINTS (some may be wrong):\n{opts}\nUse them to inform your thinking, but independently decide the correct tool call(s) and argument values for the request. Then output the index of the candidate matching the correct answer. Reply ONLY JSON {{\"choice\": <index>}}.",
    "fresh_reconcile": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nFirst write your own correct answer. Then for EACH candidate:\n{opts}\nnote where it agrees/differs from yours and decide whether the candidate's version is actually more correct (it may catch something you missed). Reconcile into the single correct answer, then output the index of the candidate that matches it. Reply ONLY JSON {{\"choice\": <index>}}.",
    "evidence_build": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nCANDIDATES (evidence — a value proposed by only ONE candidate can still be the right one):\n{opts}\nDetermine the correct function(s) and each correct argument value for the request, using the candidates as evidence for what's possible. Build the correct answer, then output the index of the candidate that matches it. Reply ONLY JSON {{\"choice\": <index>}}.",
    "fresh_then_pick_best_arg": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nSolve the request yourself first. Then look at where the candidates disagree:\n{opts}\nFor each point of disagreement (a function or an argument value), decide which candidate is right by re-reading the request. Output the index of the candidate correct on the most/most-important disagreements. Reply ONLY JSON {{\"choice\": <index>}}.",
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
    key = client.api_key_from_env()
    answers = json.loads(open(CACHE).read())
    items = [it for it in load(categories=["live_parallel", "live_parallel_multiple"]) if it["id"] in answers]
    n = len(items)

    trivial = unwinnable = 0
    decidable = []
    for it in items:
        cands = distinct([answers[it["id"]].get(m, []) for m in PANEL])
        flags = [grade(it, c) for c in cands]
        if all(flags):
            trivial += 1
        elif not any(flags):
            unwinnable += 1
        else:
            decidable.append((it, cands))
    print(f"n={n}: trivial(all-right)={trivial}  unwinnable(none-right)={unwinnable}  DECIDABLE={len(decidable)}")
    print("  (a few examples)")
    for it, cands in decidable[:2]:
        print(f"    decidable {it['id']}: {len(cands)} distinct candidates, "
              f"{sum(grade(it,c) for c in cands)} correct")
    print(f"  implied_overall ceiling (capture all decidable) = {100*(trivial+len(decidable))/n:.1f}")

    print(f"\n=== selector={SELECTOR}: decidable-accuracy / implied-overall (n={n}) ===")
    results = []
    for name, tmpl in PROMPTS.items():
        with ThreadPoolExecutor(max_workers=3) as pool:
            correct = sum(f.result() for f in as_completed(
                [pool.submit(lambda it, cands: grade(it, select(it, cands, tmpl, key)), it, cands) for it, cands in decidable]))
        dec_acc = 100 * correct / len(decidable)
        overall = 100 * (trivial + correct) / n
        results.append((name, dec_acc, overall))
    for name, dec_acc, overall in sorted(results, key=lambda r: -r[1]):
        print(f"  {name:22} decidable {dec_acc:5.1f}   implied_overall {overall:5.1f}")


if __name__ == "__main__":
    main()
