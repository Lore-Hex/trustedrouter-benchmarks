"""Run tau2-bench (agentic tool-use) on a TrustedRouter panel.

tau2-bench (sierra-research/tau2-bench, MIT) is a full agent+simulated-user
framework, so we DON'T reimplement it — we drive its CLI as a subprocess, once
per panel model, with both the agent and the (fixed) user LLM routed through the
TR gateway via LiteLLM, then read its canonical per-task `reward` out of
`data/simulations/<slug>/results.json` and report avg_reward + pass^k.

LiteLLM → TR routing: a TR model id like `z-ai/glm-5` becomes the LiteLLM name
`openai/z-ai/glm-5` (the `openai/` prefix selects the OpenAI-compatible provider;
the remainder is sent verbatim as the `model`), with OPENAI_API_BASE/OPENAI_API_KEY
pointed at TR. The simulated user is a fixed strong model so only the agent varies.

Prereq: tau2-bench cloned + installed in its own venv (it needs Python 3.12 and
heavy deps we keep out of trbench). Point --tau2-home / $TAU2_HOME at it; default
is ../tau2-bench with .venv/bin/tau2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from trbench import client, report
from trbench.panel import resolve_panel

DEFAULT_TAU2_HOME = os.environ.get("TAU2_HOME", str(Path(__file__).resolve().parents[4] / "tau2-bench"))
DEFAULT_USER_LLM = "openai/gpt-4.1"  # fixed simulated user (a TR model id)
SUCCESS = 0.999  # tau2 reward is in [0,1]; a task "passes" at reward == 1


_REPO_ROOT = Path(__file__).resolve().parents[3]  # trustedrouter-benchmarks/


def _litellm_name(tr_model_id: str) -> str:
    """TR model id -> LiteLLM custom-OpenAI name (the `openai/` prefix routes to
    OPENAI_API_BASE; the rest is sent as the model). Idempotent."""
    return tr_model_id if tr_model_id.startswith("openai/") and tr_model_id.count("/") >= 2 else f"openai/{tr_model_id}"


def _model_arg(tr_model_id: str, use_sdk: bool) -> str:
    """litellm model name. --use-sdk routes through the TrustedRouter SDK (custom
    provider in trbench.tr_litellm, registered by the _shim); else openai/ base URL."""
    if use_sdk:
        return tr_model_id if tr_model_id.startswith("trustedrouter/") else f"trustedrouter/{tr_model_id}"
    return _litellm_name(tr_model_id)


def _pass_hat_k(success_count: int, num_trials: int, k: int) -> float:
    """tau2's pass^k: probability that k independently sampled trials all succeed,
    = C(success, k) / C(trials, k). For k=1 this is the per-task success rate."""
    if k > num_trials or num_trials == 0:
        return 0.0
    return math.comb(success_count, k) / math.comb(num_trials, k)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _diagnostic_partial_reward(reward_info: dict | None) -> float:
    """Partial-credit score from tau2's serialized component checks.

    tau2's official task reward is intentionally strict: the components listed in
    reward_basis are multiplied, and premature max_steps sims get no reward_info.
    For analysis we also want a softer score, analogous to "tests passed / tests":
    average every serialized component check available for the task. Components
    not serialized cannot receive credit; missing reward_info is 0.
    """
    if not reward_info:
        return 0.0
    components: list[float] = []
    db_check = reward_info.get("db_check")
    if isinstance(db_check, dict):
        components.append(float(db_check.get("db_reward") or 0.0))
    action_checks = reward_info.get("action_checks")
    if isinstance(action_checks, list) and action_checks:
        components.append(_mean([float(check.get("action_reward") or 0.0) for check in action_checks]))
    env_assertions = reward_info.get("env_assertions")
    if isinstance(env_assertions, list) and env_assertions:
        values = []
        for check in env_assertions:
            if not isinstance(check, dict):
                continue
            if "reward" in check:
                values.append(float(check.get("reward") or 0.0))
            elif "met" in check:
                values.append(1.0 if check.get("met") else 0.0)
        if values:
            components.append(_mean(values))
    communicate_checks = reward_info.get("communicate_checks")
    if isinstance(communicate_checks, list) and communicate_checks:
        components.append(_mean([1.0 if check.get("met") else 0.0 for check in communicate_checks if isinstance(check, dict)]))
    nl_assertions = reward_info.get("nl_assertions")
    if isinstance(nl_assertions, list) and nl_assertions:
        components.append(_mean([1.0 if check.get("met") else 0.0 for check in nl_assertions if isinstance(check, dict)]))
    if components:
        return _mean(components)
    return float(reward_info.get("reward") or 0.0)


def _slug(model: str, domain: str) -> str:
    return "trbench_" + re.sub(r"[^a-z0-9]+", "-", f"{model}_{domain}".lower()).strip("-")


def _run_one(*, tau2_bin: str, tau2_home: Path, domain: str, model: str, user_llm: str,
             use_sdk: bool, num_tasks: int, num_trials: int, max_steps: int, max_concurrency: int,
             max_retries: int, retry_delay: float, seed: int, env: dict[str, str],
             timeout: float, resume: bool, retrieval_config: str | None,
             retrieval_config_kwargs: str | None) -> dict:
    slug = _slug(model, domain)
    # tau2 auto-RESUMES from an existing data/simulations/<slug>/results.json, which
    # silently keeps stale infrastructure-error sims from a prior run. Clear it for a
    # clean run (the published-comparable behavior) unless --resume is requested.
    if not resume:
        shutil.rmtree(tau2_home / "data" / "simulations" / slug, ignore_errors=True)
    cmd = [
        tau2_bin, "run", "--domain", domain,
        "--agent-llm", _model_arg(model, use_sdk), "--user-llm", _model_arg(user_llm, use_sdk),
        "--num-tasks", str(num_tasks), "--num-trials", str(num_trials),
        "--max-steps", str(max_steps), "--max-concurrency", str(max_concurrency),
        "--max-retries", str(max_retries), "--retry-delay", str(retry_delay),
        "--seed", str(seed), "--save-to", slug,
    ]
    if retrieval_config:
        cmd += ["--retrieval-config", retrieval_config]
    if retrieval_config_kwargs:
        cmd += ["--retrieval-config-kwargs", retrieval_config_kwargs]
    print(f"  agent={model} domain={domain} tasks={num_tasks} trials={num_trials} ...", flush=True)
    proc = subprocess.run(cmd, cwd=str(tau2_home), env=env, capture_output=True, text=True, timeout=timeout)
    results_path = tau2_home / "data" / "simulations" / slug / "results.json"
    if not results_path.exists():
        return {"model": model, "error": f"no results.json (rc={proc.returncode}): {proc.stderr[-400:]}"}
    data = json.loads(results_path.read_text())
    sims = data.get("simulations", [])

    # reward per (task, trial); a task passes a trial when reward == 1.
    by_task: dict[str, list[float]] = {}
    partial_by_task: dict[str, list[float]] = {}
    rewards: list[float] = []
    partial_rewards: list[float] = []
    sim_errors = 0
    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None:
            sim_errors += 1
            continue
        rewards.append(float(r))
        task_id = str(s.get("task_id"))
        by_task.setdefault(task_id, []).append(float(r))
        partial = _diagnostic_partial_reward(ri)
        partial_rewards.append(partial)
        partial_by_task.setdefault(task_id, []).append(partial)

    n_tasks = len(by_task)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    partial_macro_vals = [_mean(trs) for trs in partial_by_task.values()]
    partial_macro = _mean(partial_macro_vals)
    partial_micro = _mean(partial_rewards)
    # pass^k averaged over tasks, for k = 1..num_trials
    passk: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        vals = [_pass_hat_k(sum(1 for r in trs if r >= SUCCESS), len(trs), k) for trs in by_task.values()]
        passk[k] = sum(vals) / len(vals) if vals else 0.0
    return {
        "model": model, "domain": domain, "n_tasks": n_tasks, "num_trials": num_trials,
        "avg_reward": round(100 * avg_reward, 1),
        "pass1": round(100 * passk.get(1, 0.0), 1),
        "partial_macro": round(100 * partial_macro, 1),
        "partial_micro": round(100 * partial_micro, 1),
        "passk": {k: round(100 * v, 1) for k, v in passk.items()},
        "sim_errors": sim_errors,
        "rewards_by_task": by_task,  # replay: lets the metric be recomputed
        "partial_by_task": partial_by_task,
    }


COLUMNS = [
    ("Model", "model"),
    ("pass^1", "pass1"),
    ("partial", "partial_macro"),
    ("avg_reward", "avg_reward"),
    ("Tasks", "n_tasks"),
    ("Errors", "sim_errors"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tau2-bench on a TrustedRouter panel (agentic tool-use).")
    parser.add_argument("--models", default=None, help="Panel name or comma list of TR model ids.")
    parser.add_argument("--domain", default="retail",
                        choices=["airline", "retail", "telecom", "mock", "banking_knowledge"])
    parser.add_argument("--user-llm", default=DEFAULT_USER_LLM, help="Fixed simulated-user model (TR id).")
    parser.add_argument("--eval-llm", default="openai/gpt-4.1",
                        help="Fixed NL-assertion judge model (TR id); routes tau2's hardcoded default via the shim.")
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=6,
                        help="tau2 retries on transient LLM/gateway (infrastructure) errors.")
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--retrieval-config", default=None,
                        help="Pass-through for tau2 banking_knowledge retrieval config, e.g. bm25 or alltools.")
    parser.add_argument("--retrieval-config-kwargs", default=None,
                        help="JSON string passed through to tau2 --retrieval-config-kwargs.")
    parser.add_argument("--resume", action="store_true",
                        help="Keep tau2's existing sims for a slug (default: clear for a clean run).")
    parser.add_argument("--use-sdk", dest="use_sdk", action=argparse.BooleanOptionalAction, default=True,
                        help="Route agent/user/judge calls through the TrustedRouter SDK (litellm "
                             "custom provider). --no-use-sdk falls back to the OpenAI-compatible base URL.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau2-home", default=DEFAULT_TAU2_HOME)
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--per-model-timeout", type=float, default=3600.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--svg", default="assets/tau2.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    tau2_home = Path(args.tau2_home).resolve()
    tau2_bin = str(tau2_home / ".venv" / "bin" / "tau2")
    if not Path(tau2_bin).exists():
        raise SystemExit(
            f"tau2 CLI not found at {tau2_bin}. Clone+install sierra-research/tau2-bench, pass --tau2-home.")

    models = resolve_panel(args.models)
    out = Path(args.out or f"results/tau2_{args.domain}.json")
    call_log_dir = Path("runs/tau2") / out.stem / "trustedrouter-calls"
    # Put our litellm shim on PYTHONPATH so tau2's hardcoded `gpt-4.1-2025-04-14`
    # evaluator/judge calls route through TR instead of aborting the sim. The judge
    # is fixed (not the agent under test) so grading stays consistent across models.
    shim_dir = str(Path(__file__).resolve().parent / "_shim")
    # --use-sdk also needs the repo root on PYTHONPATH so the shim can import
    # trbench.tr_litellm (the SDK custom provider). The tau2 venv then needs the
    # SDK: tau2-bench/.venv/bin/pip install /path/to/trusted-router-py.
    pp_parts = [shim_dir] + ([str(_REPO_ROOT)] if args.use_sdk else []) + [os.environ.get("PYTHONPATH", "")]
    pythonpath = os.pathsep.join(pp_parts).rstrip(os.pathsep)
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "TRBENCH_TAU2_EVAL_LLM": _model_arg(args.eval_llm, args.use_sdk),
        "TRBENCH_LITELLM_LOG_DIR": str(call_log_dir),
    }
    if args.use_sdk:
        env["TRUSTEDROUTER_API_KEY"] = api_key
        env["TRUSTEDROUTER_BASE_URL"] = args.base_url
    else:
        env["OPENAI_API_BASE"] = args.base_url
        env["OPENAI_API_KEY"] = api_key
    print(f"tau2-bench [{args.domain}]: {len(models)} agents x {args.num_tasks} tasks "
          f"x {args.num_trials} trials, user={args.user_llm}")

    rows = []
    for model in models:
        rows.append(_run_one(
            tau2_bin=tau2_bin, tau2_home=tau2_home, domain=args.domain, model=model, user_llm=args.user_llm,
            use_sdk=args.use_sdk,
            num_tasks=args.num_tasks, num_trials=args.num_trials, max_steps=args.max_steps,
            max_concurrency=args.max_concurrency, max_retries=args.max_retries, retry_delay=args.retry_delay,
            seed=args.seed, env=env, timeout=args.per_model_timeout, resume=args.resume,
            retrieval_config=args.retrieval_config,
            retrieval_config_kwargs=args.retrieval_config_kwargs,
        ))

    good = [r for r in rows if "error" not in r]
    good.sort(key=lambda r: (-float(r["pass1"]), -float(r["avg_reward"]), r["model"]))
    for r in rows:
        if "error" in r:
            print(f"  !! {r['model']}: {r['error']}")

    table = report.markdown_table(good, COLUMNS)
    print(table)
    result = {
        "eval": "tau2", "domain": args.domain, "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "user_llm": args.user_llm, "num_tasks": args.num_tasks, "num_trials": args.num_trials,
        "retrieval_config": args.retrieval_config,
        "retrieval_config_kwargs": args.retrieval_config_kwargs,
        "trustedrouter_call_log_dir": str(call_log_dir),
        "models": models, "results": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    svg = report.svg_bar_chart(
        good, score_key="pass1", max_score=100,
        title=f"tau2-bench ({args.domain}) on TrustedRouter",
        subtitle=f"Agentic tool-use, pass^1. User sim: {args.user_llm}. Higher is better.",
    )
    Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.svg).write_text(svg, encoding="utf-8")
    print(f"wrote {args.svg}")

    if args.readme:
        rp = Path(args.readme)
        block = "\n\n".join([
            f"tau2-bench snapshot: `{result['created_at']}`. Domain `{args.domain}`, "
            f"{args.num_tasks} tasks x {args.num_trials} trial(s), agent vs fixed user `{args.user_llm}`. "
            f"Metric: pass^1 (task reward == 1).",
            f"![tau2-bench chart]({Path(args.svg).as_posix()})",
            table,
        ])
        rp.write_text(report.splice_readme(rp.read_text(encoding="utf-8"), "TAU2_RESULTS", block), encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
