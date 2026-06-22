"""Grade the agentic-fusion mock-domain workflow: solo vs fusion task success.

Success = every expected action (tau2 evaluation_criteria.actions) appears in the
episode's taken actions with matching name + argument values (transfer matches on
name only). Reports per-policy success on the action-gradeable tasks + per-task.

Run: PYTHONPATH=. .venv/bin/python scripts/grade_tau.py <workflow-output-path>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

_TF = sys.argv[2] if len(sys.argv) > 2 else "results/_tau_tasks.json"
TASKS = {t["id"]: t for t in json.load(open(_TF))}


def match(expected, taken):
    for ea in expected:
        ok = False
        for ta in (taken or []):
            if ta.get("name") != ea["name"]:
                continue
            if ea["name"] == "transfer_to_human_agents":
                ok = True; break
            def argeq(got, exp):
                if isinstance(exp, list) and isinstance(got, list):
                    return sorted(map(str, got)) == sorted(map(str, exp))
                return str(got) == str(exp)
            if all(argeq((ta.get("arguments") or {}).get(k), v) for k, v in ea["arguments"].items()):
                ok = True; break
        if not ok:
            return False
    return True


def main():
    obj = json.loads(open(sys.argv[1]).read())
    eps = obj["result"] if isinstance(obj, dict) and isinstance(obj.get("result"), list) else obj
    eps = [e for e in eps if e and e.get("task_id")]

    by = defaultdict(dict)  # task_id -> {policy: success}
    for e in eps:
        exp = TASKS[e["task_id"]]["expected_actions"]
        if not exp:
            continue  # nl-only task, not action-gradeable
        by[e["task_id"]][e["policy"]] = match(exp, e.get("actions"))

    WRITES = {"cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address",
              "modify_pending_order_payment", "modify_user_address", "exchange_delivered_order_items",
              "return_delivered_order_items", "transfer_to_human_agents"}
    wpol = defaultdict(lambda: [0, 0])
    for e in eps:
        exp_w = [a for a in TASKS[e["task_id"]]["expected_actions"] if a["name"] in WRITES]
        if not exp_w:
            continue
        wpol[e["policy"]][0] += int(match(exp_w, e.get("actions"))); wpol[e["policy"]][1] += 1

    pol_score = defaultdict(lambda: [0, 0])
    print(f"{'task':<42} {'solo':>6} {'fusion':>7}")
    for tid in sorted(by):
        row = by[tid]
        for p in ("solo", "fusion"):
            if p in row:
                pol_score[p][0] += int(row[p]); pol_score[p][1] += 1
        s = "PASS" if row.get("solo") else "fail"
        f = "PASS" if row.get("fusion") else "fail"
        flag = "  <-- fusion fixed" if (not row.get("solo") and row.get("fusion")) else ("  <-- fusion BROKE" if (row.get("solo") and not row.get("fusion")) else "")
        print(f"  {tid:<40} {s:>6} {f:>7}{flag}")
    print(f"\n=== full-sequence match (strict) ===")
    for p in ("solo", "fusion"):
        c, n = pol_score[p]
        print(f"  {p:8} {100*c/n:5.1f}  ({c}/{n})" if n else f"  {p}: n/a")
    print(f"=== writes-only match (closer to tau2 reward) ===")
    for p in ("solo", "fusion"):
        c, n = wpol[p]
        print(f"  {p:8} {100*c/n:5.1f}  ({c}/{n})" if n else f"  {p}: n/a")
    # machine-readable line for combining chunks
    print(f"COMBINE strict_solo={pol_score['solo']} strict_fusion={pol_score['fusion']} "
          f"writes_solo={wpol['solo']} writes_fusion={wpol['fusion']}")


if __name__ == "__main__":
    main()
