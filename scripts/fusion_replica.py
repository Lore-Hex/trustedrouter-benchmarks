"""Faithful local replica of trustedrouter/fusion-code, reusing intermediates.

Reproduces the EXACT enclave pipeline (enclave-go/cmd/enclave/fusion.go):
  panel  -> reused from results/_confirm_panel.json (no re-call)
  judge  -> kimi-k2.6, real judge system prompt + fusionJudgePrompt, JSON out
            (cached per judge-variant in results/_replica_judge.json)
  synth  -> glm-5.2, original messages + real line-1023 instruction + real
            fusionPanelEvidence + judge JSON, REAL tools, emits tool calls

Because panel+judge are cached, sweeping the synth prompt only re-runs the synth.
The BASELINE judge/synth strings are copied verbatim from fusion.go so a win here
should transfer to the deployed endpoint. We calibrate the baseline replica against
the live endpoint (same panel, same items): if they match, the replica is faithful.

Run: PYTHONPATH=. .venv/bin/python scripts/fusion_replica.py [--n 6] [--calibrate]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from trbench import client
from trbench.evals.bfcl.loader import load
from trbench.evals.bfcl.run import decode_tool_calls
from trbench.evals.bfcl.schema_convert import to_openai_tools
from trbench.evals.bfcl.vendor.ast_checker import Language, ast_checker

PANEL = ["minimax/minimax-m3", "moonshotai/kimi-k2.6", "z-ai/glm-5.2", "google/gemma-4-31b-it", "deepseek/deepseek-v4-pro"]
JUDGE = "moonshotai/kimi-k2.6"
SYNTH = "z-ai/glm-5.2"
CACHE = "results/_confirm_panel.json"
JUDGE_CACHE = "results/_replica_judge.json"

# --- verbatim from enclave-go/cmd/enclave/fusion.go ---
JUDGE_SYSTEM = "You are the TrustedRouter Fusion judge. Compare panel responses and return compact JSON with keys consensus, contradictions, partial_coverage, unique_insights, blind_spots, and final_guidance. Do not write the final answer. Return only JSON; do not include chain-of-thought, hidden reasoning, or <think> blocks."
SYNTH_INSTRUCTION = "TrustedRouter Fusion panel answers and judge analysis follow. Continue solving the original task using the panel answers as primary evidence and the judge analysis as guidance. If the next correct action is a tool call, emit the tool call directly instead of describing it in text. Return visible text only when no tool call is needed. Do not include chain-of-thought, hidden reasoning, analysis, scratchpad text, <think> blocks, or internal model names unless the user asked for methodology."

JUDGE_PROMPTS = {"baseline": JUDGE_SYSTEM}
SYNTH_PROMPTS = {"baseline": SYNTH_INSTRUCTION}


def grade(it, dec):
    if not dec or not it.get("ground_truth"):
        return False
    try:
        return bool(ast_checker(it["functions"], dec, it["ground_truth"], Language.PYTHON, it["ast_test_category"], "x")["valid"])
    except Exception:
        return False


def tool_calls_text(calls):  # mirrors fusionToolCallsText
    parts = []
    for c in (calls or []):
        for name, args in c.items():
            parts.append(f"{name}({json.dumps(args) if args else '{}'})")
    return ("Proposed tool call(s): " + ", ".join(parts)) if parts else ""


def panel_evidence(ans):  # mirrors fusionPanelEvidence; panel members in PANEL order
    b = "Panel answers:\n"
    for i, m in enumerate(PANEL):
        tc = tool_calls_text(ans.get(m, []))
        b += f"\n[{i+1}] model={m}\n{tc}\n"
    return b


def chat_messages_text(messages):  # mirrors chatMessagesText
    parts = []
    for msg in messages:
        text = (msg.get("content") or "").strip()
        if text:
            parts.append(msg["role"].upper() + ": " + text)
    return "\n".join(parts)


def judge_prompt(it, ans):  # mirrors fusionJudgePrompt
    return "Original request summary:\n" + chat_messages_text(it["messages"]) + \
           "\n\nPanel responses:\n" + panel_evidence(ans)[len("Panel answers:\n"):]


def run_judge(it, ans, key, system):
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=JUDGE,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": judge_prompt(it, ans)}],
                    max_tokens=2048, temperature=0.0, timeout=180, retries=1,
                    extra_body={"response_format": {"type": "json_object"}})
    return "" if r.get("error") else (r.get("text") or "")


def run_synth(it, ans, judge_json, key, instruction):  # mirrors fusionFinalRequest (tool branch)
    new_user = instruction + "\n\n" + panel_evidence(ans) + "\n\nJudge analysis JSON:\n" + judge_json
    messages = list(it["messages"]) + [{"role": "user", "content": new_user}]
    r = client.chat(base_url=client.DEFAULT_BASE_URL, api_key=key, model=SYNTH, messages=messages,
                    tools=to_openai_tools(it["functions"]), max_tokens=2048, temperature=0.0, timeout=180, retries=1)
    return [] if r.get("error") else decode_tool_calls(r.get("tool_calls"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    key = client.api_key_from_env()
    answers = json.loads(open(CACHE).read())
    items = [it for it in load(categories=["live_parallel", "live_parallel_multiple"]) if it["id"] in answers][:args.n]
    n = len(items)
    jcache = json.loads(open(JUDGE_CACHE).read()) if os.path.exists(JUDGE_CACHE) else {}

    # Stage 1: judge per variant (cached) -- reused across all synth variants
    for jname, jsys in JUDGE_PROMPTS.items():
        todo = [it for it in items if f"{it['id']}|{jname}" not in jcache]
        if todo:
            print(f"judge[{jname}]: running {len(todo)} (reuse {n-len(todo)})", flush=True)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futs = {pool.submit(run_judge, it, answers[it["id"]], key, jsys): it["id"] for it in todo}
                for f in as_completed(futs):
                    jcache[f"{futs[f]}|{jname}"] = f.result()
            open(JUDGE_CACHE, "w").write(json.dumps(jcache))
    print(f"\nsample judge JSON [{items[0]['id']}]:\n{(jcache[items[0]['id']+'|baseline'] or '')[:400]}\n", flush=True)

    # Stage 2: synth per (judge,synth) variant -- reuses cached panel+judge
    print(f"=== faithful replica, n={n} ===")
    solo = {m: 100 * sum(grade(it, answers[it["id"]].get(m, [])) for it in items) / n for m in PANEL}
    oracle = 100 * sum(any(grade(it, answers[it["id"]].get(m, [])) for m in PANEL) for it in items) / n
    for jname in JUDGE_PROMPTS:
        for sname, instr in SYNTH_PROMPTS.items():
            def one(it):
                jj = jcache[f"{it['id']}|{jname}"]
                return grade(it, run_synth(it, answers[it["id"]], jj, key, instr))
            with ThreadPoolExecutor(max_workers=3) as pool:
                correct = sum(f.result() for f in as_completed([pool.submit(one, it) for it in items]))
            print(f"  judge={jname:10} synth={sname:10}  {100*correct/n:5.1f}")
    print("  --- references ---")
    for m in PANEL:
        print(f"  solo {m:24} {solo[m]:5.1f}")
    print(f"  ORACLE                        {oracle:5.1f}")


if __name__ == "__main__":
    main()
