"""Proper two-stage fusion: JUDGE (kimi-k2.6) -> SYNTHESIZER (glm-5.2).

The established roles (not the collapsed single-call I was testing):
  - JUDGE = kimi-k2.6: reads the candidates and writes an ANALYSIS — what they
    agree on and EXACTLY where they differ (function / argument value). It does
    NOT pick a winner.
  - SYNTHESIZER = glm-5.2: reads the candidates + the judge's analysis and
    decides the final answer (picks the correct candidate).

Reuses the cached 120-item 5-model panel. Runs on each panel's decidable subset,
compares two-stage implied-overall against best-solo and the single-call number.

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_twostage.py
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
JUDGE = K       # analyzes, does NOT pick
SYNTH = G       # picks / assembles
PANELS = {
    "strong2":  [K, GE],
    "strong3":  [K, G, GE],
    "full5":    [M, K, G, GE, D],
}
CACHE = "results/_confirm_panel.json"
RESULTS = "results/_twostage_results.json"
CATEGORIES = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]

JUDGE_PROMPT = (
    "You are the JUDGE in a tool-calling panel. Several models each proposed tool calls for the request.\n"
    "Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nCANDIDATE ANSWERS:\n{opts}\n\n"
    "Write a short analysis: (1) what the candidates AGREE on, (2) EXACTLY where they DIFFER — name the "
    "specific function or argument value in dispute and what the request says about it. "
    "Do NOT choose a winner; just surface the agreements and the precise points of disagreement."
)
SYNTH_PROMPT = (
    "You are the SYNTHESIZER. Decide the correct tool calls for the request, using the judge's analysis "
    "to focus on the disputed points.\nAvailable functions: {schema}\n\nUSER REQUEST:\n{q}\n\n"
    "CANDIDATES:\n{opts}\n\nJUDGE'S ANALYSIS:\n{analysis}\n\n"
    "Resolve each point of disagreement by re-reading the request, then output the index of the candidate "
    "that is correct. Reply ONLY JSON {{\"choice\": <index>}}."
)


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


def chat(model, prompt, key):
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=model,
                    messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.0, timeout=180)
    return r.get("text", "") or ""


def two_stage(it, cands, key):
    q = it["messages"][-1]["content"]
    schema = json.dumps([f.get("name") for f in it["functions"]])
    opts = "\n".join(f"[{i}] {json.dumps(c)}" for i, c in enumerate(cands))
    analysis = chat(JUDGE, JUDGE_PROMPT.format(schema=schema, q=q, opts=opts), key)
    out = chat(SYNTH, SYNTH_PROMPT.format(schema=schema, q=q, opts=opts, analysis=analysis), key)
    m = re.search(r'"choice"\s*:\s*(\d+)', out)
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
    print(f"two-stage  JUDGE={JUDGE.split('/')[-1]} (analyze, no pick)  ->  SYNTH={SYNTH.split('/')[-1]} (pick)   n={n}\n")

    for pname, panel in PANELS.items():
        solo, oracle, trivial, decidable = panel_stats(items, answers, panel)
        best_solo = 100 * max(solo.values()) / n
        ckey = pname
        if ckey not in cache:
            with ThreadPoolExecutor(max_workers=4) as pool:
                correct = sum(f.result() for f in as_completed(
                    [pool.submit(lambda it, cands: grade(it, two_stage(it, cands, key)), it, cands) for it, cands in decidable]))
            cache[ckey] = {"overall": 100 * (trivial + correct) / n, "dec": 100 * correct / max(1, len(decidable))}
            open(RESULTS, "w").write(json.dumps(cache))
        r = cache[ckey]
        beat = "  WIN" if r["overall"] > best_solo + 1e-9 else "  (<= solo)"
        print(f"  panel {pname:8} {[m.split('/')[-1] for m in panel]}")
        print(f"    best_solo={best_solo:.1f}  oracle={100*oracle/n:.1f}  decidable={len(decidable)}")
        print(f"    TWO-STAGE implied_overall={r['overall']:.1f} (dec {r['dec']:.0f}){beat}\n")


if __name__ == "__main__":
    main()
