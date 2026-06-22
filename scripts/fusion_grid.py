"""Exhaustive grid: can ANY single-shot selection config beat best-solo on tools?

Reuses the cached 120-item 5-model panel (results/_confirm_panel.json). For each
PANEL (subset of the cached 5 -> no re-gather), each SELECTOR model, and each
top PROMPT, computes selection accuracy on that panel's decidable subset and the
implied overall, against that panel's best-solo and oracle. Results are cached per
config so the run is resumable.

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_grid.py
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from trbench import client
from trbench.evals.bfcl.loader import load
from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

K, G, M, GE, D = "moonshotai/kimi-k2.6", "z-ai/glm-5.2", "minimax/minimax-m3", "google/gemma-4-31b-it", "deepseek/deepseek-v4-pro"
PANELS = {
    "full5":    [M, K, G, GE, D],
    "strong3":  [K, G, GE],
    "strong2":  [K, GE],
    "diverse3": [K, G, D],
}
SELECTORS = [G, K, GE]
PROMPTS = {
    "verify_each": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nCANDIDATES:\n{opts}\n\nFor EACH candidate, verify: (a) correct function(s), (b) every argument value correct and literally supported by the request, (c) no missing/extra calls. Pick the candidate that passes most strictly. Reply ONLY JSON {{\"choice\": <index>}}.",
    "candidates_as_hints": "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nThese candidate answers are HINTS (some may be wrong):\n{opts}\nUse them to inform your thinking, but independently decide the correct tool call(s) and argument values for the request. Then output the index of the candidate matching the correct answer. Reply ONLY JSON {{\"choice\": <index>}}.",
}
CACHE = "results/_confirm_panel.json"
RESULTS = "results/_grid_results.json"
CATEGORIES = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]


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


def select(selector, it, cands, template, key):
    q = it["messages"][-1]["content"]
    schema = json.dumps([f.get("name") for f in it["functions"]])
    opts = "\n".join(f"[{i}] {json.dumps(c)}" for i, c in enumerate(cands))
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=selector,
                    messages=[{"role": "user", "content": template.format(q=q, schema=schema, opts=opts)}],
                    max_tokens=4096, temperature=0.0, timeout=180)
    m = re.search(r'"choice"\s*:\s*(\d+)', r.get("text", "") or "")
    idx = int(m.group(1)) if m else 0
    return cands[idx] if 0 <= idx < len(cands) else cands[0]


def panel_stats(items, answers, panel):
    solo = {m: sum(grade(it, answers[it["id"]].get(m, [])) for it in items) for m in panel}
    oracle = sum(any(grade(it, answers[it["id"]].get(m, [])) for m in panel) for it in items)
    trivial, decidable = 0, []
    for it in items:
        cands = distinct([answers[it["id"]].get(m, []) for m in panel])
        flags = [grade(it, c) for c in cands]
        if all(flags):
            trivial += 1
        elif any(flags):
            decidable.append((it, cands))
    return solo, oracle, trivial, decidable


def main():
    key = client.api_key_from_env()
    answers = json.loads(open(CACHE).read())
    items = [it for it in load(categories=CATEGORIES, limit_per_category=40) if it["id"] in answers]
    n = len(items)
    cache = json.loads(open(RESULTS).read()) if os.path.exists(RESULTS) else {}

    print(f"n={n}\n")
    for pname, panel in PANELS.items():
        solo, oracle, trivial, decidable = panel_stats(items, answers, panel)
        best_solo = 100 * max(solo.values()) / n
        print(f"=== panel {pname} {[m.split('/')[-1] for m in panel]}  "
              f"best_solo={best_solo:.1f}  oracle={100*oracle/n:.1f}  decidable={len(decidable)} ===")
        for selname in SELECTORS:
            row = []
            for pr_name, tmpl in PROMPTS.items():
                ckey = f"{pname}|{selname}|{pr_name}"
                if ckey not in cache:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        correct = sum(f.result() for f in as_completed(
                            [pool.submit(lambda it, cands: grade(it, select(selname, it, cands, tmpl, key)), it, cands)
                             for it, cands in decidable]))
                    cache[ckey] = {"overall": 100 * (trivial + correct) / n, "dec": 100 * correct / max(1, len(decidable))}
                    open(RESULTS, "w").write(json.dumps(cache))
                r = cache[ckey]
                beat = "  WIN" if r["overall"] > best_solo + 1e-9 else ("  tie" if abs(r["overall"]-best_solo) < 1e-9 else "")
                row.append(f"{pr_name}={r['overall']:.1f}(dec{r['dec']:.0f}){beat}")
            print(f"  selector {selname.split('/')[-1]:18} " + "  ".join(row))
        print()


if __name__ == "__main__":
    main()
