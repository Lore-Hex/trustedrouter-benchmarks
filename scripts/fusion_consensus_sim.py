"""Local simulation of tool-call CONSENSUS fusion, to validate (before any enclave
change) that selecting over the panel's tool calls beats the best solo and
approaches the oracle — vs the current text-synthesis that lands mid-pack.

For each BFCL item: query every panel member, decode their tool calls, then score
three strategies against BFCL's vendored checker:
  - per-model solo accuracy
  - ORACLE  : item right if ANY member is right (ceiling of perfect selection)
  - MAJORITY: pick the most-common normalized answer; score it
  - JUDGE   : when there's no clear majority, an LLM judge picks the best proposal

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_consensus_sim.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from trbench import client
from trbench.evals.bfcl.loader import load
from trbench.evals.bfcl.run import decode_tool_calls
from trbench.evals.bfcl.schema_convert import to_openai_tools
from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

PANEL = [
    "minimax/minimax-m3",
    "moonshotai/kimi-k2.7-code",
    "z-ai/glm-5.2",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v4-pro",
]
JUDGE = "moonshotai/kimi-k2.7-code"
CATEGORIES = ["live_parallel", "live_parallel_multiple"]


def canonical(decoded: list[dict]) -> str:
    """Order-independent key for a set of tool calls (BFCL parallel is unordered)."""
    return json.dumps(sorted(json.dumps(c, sort_keys=True) for c in (decoded or [])))


def grade(item: dict, decoded: list[dict]) -> bool:
    if item["checker_kind"] == "relevance":
        return len(decoded) == 0
    if not decoded or not item.get("ground_truth"):
        return False
    try:
        return bool(ast_checker(item["functions"], decoded, item["ground_truth"],
                                Language.PYTHON, item["ast_test_category"], "x")["valid"])
    except Exception:
        return False


def ask(model: str, item: dict, key: str) -> list[dict]:
    tools = to_openai_tools(item["functions"])
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=model,
                    messages=item["messages"], tools=tools, max_tokens=4096,
                    temperature=0.0, timeout=180)
    return decode_tool_calls(r.get("tool_calls")) if not r.get("error") else []


def judge_pick(item: dict, proposals: list[list[dict]], key: str) -> list[dict]:
    """LLM judge picks the best distinct proposal (tiebreak when no majority)."""
    uniq = []
    seen = set()
    for p in proposals:
        c = canonical(p)
        if c not in seen:
            seen.add(c)
            uniq.append(p)
    if len(uniq) == 1:
        return uniq[0]
    q = item["messages"][-1]["content"]
    opts = "\n".join(f"[{i}] {json.dumps(p)}" for i, p in enumerate(uniq))
    prompt = (f"User request:\n{q}\n\nCandidate tool-call answers:\n{opts}\n\n"
              f"Which candidate is the most correct set of tool calls for the request? "
              f'Reply ONLY with JSON {{"choice": <index>}}.')
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=JUDGE,
                    messages=[{"role": "user", "content": prompt}], max_tokens=2048, temperature=0.0, timeout=120)
    import re
    m = re.search(r'"choice"\s*:\s*(\d+)', r.get("text", "") or "")
    idx = int(m.group(1)) if m else 0
    return uniq[idx] if 0 <= idx < len(uniq) else uniq[0]


JUDGE_SWEEP = ["deepseek/deepseek-v4-pro", "google/gemini-3.1-pro-preview",
               "openai/gpt-5.5", "z-ai/glm-5.2", "anthropic/claude-opus-4.8"]
PANEL_CACHE = "results/_consensus_panel.json"


def judge_best_every(item: dict, proposals: list[list[dict]], judge_model: str, key: str) -> list[dict]:
    """For EVERY item (not just ties): judge picks the most-correct distinct proposal."""
    uniq = []
    seen = set()
    for p in proposals:
        c = canonical(p)
        if c not in seen:
            seen.add(c)
            uniq.append(p)
    if len(uniq) == 1:
        return uniq[0]
    q = item["messages"][-1]["content"]
    schema = json.dumps([f.get("name") for f in item["functions"]])
    opts = "\n".join(f"[{i}] {json.dumps(p)}" for i, p in enumerate(uniq))
    prompt = (f"You are selecting the single most correct set of tool calls for a user request.\n"
              f"Available functions: {schema}\n\nUSER REQUEST:\n{q}\n\nCANDIDATE ANSWERS:\n{opts}\n\n"
              f"Think about which candidate calls the right function(s) with the right arguments for "
              f"THIS request. Reply ONLY with JSON: {{\"choice\": <index>}}.")
    import re
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=judge_model,
                    messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.0, timeout=180)
    m = re.search(r'"choice"\s*:\s*(\d+)', r.get("text", "") or "")
    idx = int(m.group(1)) if m else 0
    return uniq[idx] if 0 <= idx < len(uniq) else uniq[0]


def main() -> int:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--judge-sweep", action="store_true", help="Sweep judge models for best-of-N selection.")
    args = ap.parse_args()
    key = client.api_key_from_env()
    items = load(categories=CATEGORIES, limit_per_category=args.limit)
    print(f"consensus-sim: {len(items)} items x {len(PANEL)} panel members")

    # gather every member's decoded answer per item (cached so judge sweeps are cheap)
    answers: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    if os.path.exists(PANEL_CACHE):
        cached = json.loads(open(PANEL_CACHE).read())
        if set(cached.keys()) >= {it["id"] for it in items}:
            answers = defaultdict(dict, {k: v for k, v in cached.items()})
            print("  (reused cached panel answers)")
    if not answers:
        with ThreadPoolExecutor(max_workers=10) as pool:
            futs = {pool.submit(ask, m, it, key): (m, it["id"]) for it in items for m in PANEL}
            for f in as_completed(futs):
                m, iid = futs[f]
                answers[iid][m] = f.result()
        open(PANEL_CACHE, "w").write(json.dumps(answers))

    solo = Counter()
    oracle = maj = judged = 0
    for it in items:
        iid = it["id"]
        per = answers[iid]
        for m in PANEL:
            solo[m] += grade(it, per.get(m, []))
        oracle += any(grade(it, per.get(m, [])) for m in PANEL)
        # majority over normalized answers
        keys = [canonical(per.get(m, [])) for m in PANEL]
        top_key, _ = Counter(keys).most_common(1)[0]
        rep = next(per[m] for m in PANEL if canonical(per.get(m, [])) == top_key)
        maj += grade(it, rep)
        # judge: majority if a strict majority exists (>= 3/5), else judge picks
        if Counter(keys).most_common(1)[0][1] >= 3:
            judged += grade(it, rep)
        else:
            judged += grade(it, judge_pick(it, [per.get(m, []) for m in PANEL], key))

    n = len(items)
    print("\n=== live_* tool-calling: selection strategies (n=%d) ===" % n)
    for m in PANEL:
        print(f"  solo {m:30} {100*solo[m]/n:5.1f}")
    print(f"  {'-'*40}")
    print(f"  best solo                          {100*max(solo.values())/n:5.1f}")
    print(f"  MAJORITY consensus                 {100*maj/n:5.1f}")
    print(f"  MAJORITY + judge-tiebreak          {100*judged/n:5.1f}")
    print(f"  ORACLE (any member right)          {100*oracle/n:5.1f}  <- ceiling")
    print(f"  [ref] current synthesize fusion    72.5")

    if args.judge_sweep:
        print("\n=== JUDGE SWEEP: judge picks the most-correct candidate every item ===")
        for jm in JUDGE_SWEEP:
            correct = 0
            with ThreadPoolExecutor(max_workers=8) as pool:
                futs = {pool.submit(judge_best_every, it, [answers[it["id"]].get(m, []) for m in PANEL], jm, key): it
                        for it in items}
                for f in as_completed(futs):
                    it = futs[f]
                    correct += grade(it, f.result())
            print(f"  judge={jm:32} {100*correct/n:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
