"""Grade a tau2 retail trajectory with tau2's REAL evaluator (run in tau2 venv).

Reward = EvaluationType.ALL (env x action x communicate), solo_mode=False — the
exact leaderboard metric. All three components are deterministic (no LLM) unless a
task's reward_basis needs NL. Build a SimulationRun from a trajectory and evaluate.

Validate first: feed the GOLD actions -> expect reward 1.0.

Run (in tau2 venv):
  cd /Users/jperla/claude/tau2-bench && .venv/bin/python <this> goldcheck 0 1 11
  cd /Users/jperla/claude/tau2-bench && .venv/bin/python <this> grade <traj.json>
"""
import json
import sys

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry

RETAIL_TASKS = "data/tau2/domains/retail/tasks.json"
_TASKS = {str(d["id"]): d for d in json.load(open(RETAIL_TASKS))}


def load_task(tid):
    return Task.model_validate(_TASKS[str(tid)])


def build_sim(tid, trajectory):
    """trajectory: ordered list of steps, each either
    {"tool_calls":[{"name","arguments"}]} or {"content": "..."}."""
    env = registry.get_env_constructor("retail")(solo_mode=False)
    msgs = [UserMessage(role="user", content="Hi, I need help with my account.")]
    cid = 0
    for step in trajectory:
        if step.get("tool_calls"):
            tcs, resps = [], []
            for tc in step["tool_calls"]:
                cid += 1
                call = ToolCall(id=f"c{cid}", name=tc["name"],
                                arguments=tc.get("arguments", {}), requestor="assistant")
                tcs.append(call)
                resps.append(env.get_response(call))  # real execution -> consistent ToolMessage
            msgs.append(AssistantMessage(role="assistant", content=None, tool_calls=tcs))
            msgs.extend(resps)
        elif step.get("content") is not None:
            msgs.append(AssistantMessage(role="assistant", content=step["content"]))
    msgs.append(AssistantMessage(role="assistant", content="Anything else I can help with?"))
    return SimulationRun(id=f"sim_{tid}", task_id=str(tid),
                         start_time="2026-01-01T00:00:00", end_time="2026-01-01T00:01:00",
                         duration=60.0, termination_reason=TerminationReason.AGENT_STOP, messages=msgs)


def grade(tid, trajectory):
    task = load_task(tid)
    sim = build_sim(tid, trajectory)
    etype = EvaluationType.ENV if __import__("os").environ.get("TAU2_DB_ONLY") else EvaluationType.ALL
    ri = evaluate_simulation(sim, task, etype, solo_mode=False, domain="retail")
    return ri, task


def gold_traj(task):
    return [{"tool_calls": [{"name": a.name, "arguments": a.arguments}]}
            for a in task.evaluation_criteria.actions]


def main():
    cmd = sys.argv[1]
    if cmd == "goldcheck":
        for tid in sys.argv[2:]:
            task = load_task(tid)
            ri, _ = grade(tid, gold_traj(task))
            basis = [str(b) for b in task.evaluation_criteria.reward_basis]
            print(f"task {tid:>3}  GOLD reward={ri.reward}  basis={basis}  breakdown={ri.reward_breakdown}")
    elif cmd == "grade":
        data = json.load(open(sys.argv[2]))  # {tid: trajectory}
        tot = 0.0
        for tid, traj in data.items():
            try:
                ri, _ = grade(tid, traj)
                r = ri.reward
                bd = ri.reward_breakdown
            except Exception as e:
                r, bd = 0.0, f"ERROR {type(e).__name__}: {str(e)[:80]}"
            tot += r
            print(f"task {tid:>3}  reward={r}  {bd}")
        print(f"\nAVG reward (pass^1) = {100*tot/len(data):.1f}  (n={len(data)})")


if __name__ == "__main__":
    main()
