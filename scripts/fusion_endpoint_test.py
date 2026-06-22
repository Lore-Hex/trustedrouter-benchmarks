"""Basic DIRECT test of the real trustedrouter/fusion-code endpoint on BFCL.

No sim, no made-up selector — calls the actual fusion endpoint (judge kimi-k2.6,
synth glm-5.2 via plugin config), grades its emitted tool calls with the BFCL
checker, and compares to the solo models on the SAME items (from the cached
panel). Prints per-item timing so a wedge is visible.

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_endpoint_test.py [--n 10]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from trbench import client
from trbench.evals.bfcl.loader import load
from trbench.evals.bfcl.run import decode_tool_calls
from trbench.evals.bfcl.schema_convert import to_openai_tools
from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

PANEL = ["minimax/minimax-m3", "moonshotai/kimi-k2.6", "z-ai/glm-5.2", "google/gemma-4-31b-it", "deepseek/deepseek-v4-pro"]
CACHE = "results/_confirm_panel.json"
EXTRA = {"plugins": [{"id": "fusion",
    "analysis_models": ["minimax/minimax-m3", "moonshotai/kimi-k2.6", "z-ai/glm-5.2", "google/gemma-4-31b-it", "deepseek/deepseek-v4-pro"],
    "judge_models": ["moonshotai/kimi-k2.6"], "final_models": ["z-ai/glm-5.2"]}]}


def grade(it, dec):
    if not dec or not it.get("ground_truth"):
        return False
    try:
        return bool(ast_checker(it["functions"], dec, it["ground_truth"], Language.PYTHON, it["ast_test_category"], "x")["valid"])
    except Exception:
        return False


def call_fusion(it, key):
    t0 = time.perf_counter()
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model="trustedrouter/fusion-code",
                    messages=it["messages"], tools=to_openai_tools(it["functions"]),
                    max_tokens=2048, temperature=0.0, timeout=300, retries=1, extra_body=EXTRA)
    dt = time.perf_counter() - t0
    dec = [] if r.get("error") else decode_tool_calls(r.get("tool_calls"))
    ok = grade(it, dec)
    print(f"  {it['id']:28} {dt:6.1f}s  {'OK ' if ok else 'X  '} err={r.get('error')}", flush=True)
    return it["id"], ok, dt, r.get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    key = client.api_key_from_env()
    answers = json.loads(open(CACHE).read())
    items = [it for it in load(categories=["live_parallel", "live_parallel_multiple"]) if it["id"] in answers][:args.n]
    print(f"fusion-code (judge kimi-k2.6 -> synth glm-5.2) on {len(items)} items, concurrency 2\n", flush=True)

    res = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(call_fusion, it, key) for it in items]):
            iid, ok, dt, err = f.result()
            res[iid] = ok

    n = len(items)
    fusion = 100 * sum(res.values()) / n
    print(f"\n=== {n} items ===")
    print(f"  fusion-code              {fusion:5.1f}")
    for m in PANEL:
        s = 100 * sum(grade(it, answers[it["id"]].get(m, [])) for it in items) / n
        print(f"  solo {m:24} {s:5.1f}")
    oracle = 100 * sum(any(grade(it, answers[it["id"]].get(m, [])) for m in PANEL) for it in items) / n
    print(f"  ORACLE                   {oracle:5.1f}")


if __name__ == "__main__":
    main()
