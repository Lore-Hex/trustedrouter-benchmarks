"""Run Terminal-Bench (agentic terminal/coding) on a TrustedRouter panel.

Terminal-Bench (laude-institute/terminal-bench, Apache-2.0) runs each task in a
Docker sandbox with an agent (Terminus) driving a real shell. We DON'T reimplement
it — we drive its `tb run` CLI as a subprocess, once per panel model, with the
agent's model routed through TR via LiteLLM, then read the canonical per-task
`is_resolved` out of `runs/<run-id>/results.json` and report accuracy.

LiteLLM → TR routing (same as tau2): a TR model id like `z-ai/glm-5` becomes the
LiteLLM name `openai/z-ai/glm-5` (the `openai/` prefix selects the OpenAI-compatible
provider; the rest is sent verbatim as `model`), with OPENAI_API_BASE/OPENAI_API_KEY
pointed at TR.

Prereq: `uv tool install terminal-bench` (provides the `tb` CLI) + Docker running.
A full 80-task run is agentic, multi-turn and Docker-heavy → expensive; default is
a curated 10-task subset to stay cheap. Some upstream task images are amd64-only,
so on Apple Silicon a few tasks may fail to build (counted as unresolved/errored).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from trbench import client, report
from trbench.panel import resolve_panel

# Curated 10-task subset: diverse (coding, git, data, sysadmin, algorithms) and
# biased toward arch-portable pure-Python/C tasks (not the qemu/kernel-build tasks,
# which are amd64-only and won't build on Apple Silicon). Fixed for reproducibility.
DEFAULT_SUBSET = [
    "hello-world",            # trivial — sanity/routing check
    "fibonacci-server",       # write + run a server
    "csv-to-parquet",         # data transform
    "fix-git",                # git repair
    "chess-best-move",        # algorithm
    "openssl-selfsigned-cert",  # crypto / sysadmin
    "polyglot-c-py",          # multi-language interop
    "count-dataset-tokens",   # data scripting
    "grid-pattern-transform",  # reasoning / transform
    "organization-json-generator",  # structured output
]

HARD_SHORT_SUBSET = [
    "cancel-async-tasks",
    "fix-code-vulnerability",
    "polyglot-rust-c",
    "configure-git-webserver",
    "feal-linear-cryptanalysis",
    "password-recovery",
]

TASK_SUBSETS = {
    "default": DEFAULT_SUBSET,
    "hard-short": HARD_SHORT_SUBSET,
}

DEFAULT_DATASET = "terminal-bench-core==0.1.1"


_REPO_ROOT = Path(__file__).resolve().parents[3]              # trustedrouter-benchmarks/
_SHIM_DIR = Path(__file__).resolve().parents[1] / "_tr_sdk_shim"  # sitecustomize that registers the SDK provider


def _litellm_name(tr_model_id: str) -> str:
    """TR model id -> LiteLLM custom-OpenAI name. The `openai/` prefix routes to
    OPENAI_API_BASE; the rest is sent as `model`. Idempotent."""
    return tr_model_id if tr_model_id.startswith("openai/") and tr_model_id.count("/") >= 2 else f"openai/{tr_model_id}"


def _model_arg(tr_model_id: str, use_sdk: bool) -> str:
    """The `--model` litellm passes to the agent. With --use-sdk (default), the
    `trustedrouter/` prefix routes every call through the TrustedRouter SDK (via
    the litellm custom provider in trbench.tr_litellm, registered by the shim);
    otherwise the generic `openai/` provider hits the OpenAI-compatible base URL."""
    if use_sdk:
        return tr_model_id if tr_model_id.startswith("trustedrouter/") else f"trustedrouter/{tr_model_id}"
    return _litellm_name(tr_model_id)


def _slug(model: str) -> str:
    return "trbench-" + re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _usage_from_debug(path: Path) -> tuple[int, int]:
    """Recover token usage from Terminus/LiteLLM debug logs.

    Timed-out tb runs often do not write results.json, but they do leave
    agent-logs/episode-*/debug.json. LiteLLM stores the provider response as a
    JSON string in original_response, including usage.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0

    candidates = [data]
    original = data.get("original_response") if isinstance(data, dict) else None
    if isinstance(original, str):
        try:
            candidates.insert(0, json.loads(original))
        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        usage = candidate.get("usage")
        if not isinstance(usage, dict):
            continue
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        return int(input_tokens or 0), int(output_tokens or 0)
    return 0, 0


def _recover_debug_usage(run_dir: Path) -> tuple[int, int, int]:
    input_tokens = 0
    output_tokens = 0
    calls = 0
    for path in run_dir.glob("**/agent-logs/episode-*/debug.json"):
        in_tok, out_tok = _usage_from_debug(path)
        if in_tok or out_tok:
            input_tokens += in_tok
            output_tokens += out_tok
            calls += 1
    return input_tokens, output_tokens, calls


def _run_one(*, tb_bin: str, model: str, model_arg: str, dataset: str, tasks: list[str], agent: str,
             n_concurrent: int, n_attempts: int, output_root: Path, env: dict[str, str],
             timeout: float, timeout_mult: float, dataset_path: str | None,
             run_suffix: str | None = None, agent_kwargs: list[str] | None = None) -> dict:
    run_id = _slug(model)
    if run_suffix:
        run_id += "-" + _slug(run_suffix).removeprefix("trbench-")
    run_dir = output_root / run_id
    shutil.rmtree(run_dir, ignore_errors=True)  # clean run

    cmd = [
        tb_bin, "run", "--agent", agent, "--model", model_arg,
        "--output-path", str(output_root), "--run-id", run_id,
        "--n-concurrent", str(n_concurrent), "--n-attempts", str(n_attempts),
        "--global-timeout-multiplier", str(timeout_mult), "--cleanup",
    ]
    if dataset_path:
        cmd += ["--dataset-path", dataset_path]
    else:
        cmd += ["--dataset", dataset]
    for t in tasks:
        cmd += ["--task-id", t]
    for kwarg in agent_kwargs or []:
        cmd += ["--agent-kwarg", kwarg]

    print(f"  agent={model} tasks={len(tasks)} ...", flush=True)
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        in_tok, out_tok, calls = _recover_debug_usage(run_dir)
        return {
            "model": model,
            "error": f"per-model timeout ({timeout}s)",
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "llm_calls": calls,
        }

    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {"model": model, "error": f"no results.json (rc={proc.returncode}): {proc.stderr[-400:]}"}

    data = json.loads(results_path.read_text())
    per_task = data.get("results", [])
    resolved = [r["task_id"] for r in per_task if r.get("is_resolved")]
    unresolved = [r["task_id"] for r in per_task if not r.get("is_resolved")]
    n = len(per_task)
    in_tok = sum(int(r.get("total_input_tokens") or 0) for r in per_task)
    out_tok = sum(int(r.get("total_output_tokens") or 0) for r in per_task)
    return {
        "model": model,
        "accuracy": round(100 * (len(resolved) / n), 1) if n else 0.0,
        "resolved": len(resolved),
        "n_tasks": n,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
    }


def _aggregate_task_rows(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        if "error" in row:
            by_model.setdefault(row["model"], []).append(row)
        else:
            by_model.setdefault(row["model"], []).append(row)

    aggregates = []
    for model, model_rows in by_model.items():
        ok_rows = [r for r in model_rows if "error" not in r]
        error_rows = [r for r in model_rows if "error" in r]
        resolved_ids: list[str] = []
        unresolved_ids: list[str] = []
        for row in ok_rows:
            resolved_ids.extend(row.get("resolved_ids", []))
            unresolved_ids.extend(row.get("unresolved_ids", []))
        n = sum(int(r.get("n_tasks") or 0) for r in ok_rows)
        resolved = len(resolved_ids)
        aggregate = {
            "model": model,
            "accuracy": round(100 * (resolved / n), 1) if n else 0.0,
            "resolved": resolved,
            "n_tasks": n,
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in ok_rows),
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in ok_rows),
            "resolved_ids": sorted(resolved_ids),
            "unresolved_ids": sorted(unresolved_ids),
        }
        if error_rows:
            aggregate["errors"] = [
                {"task": r.get("task"), "error": r.get("error")} for r in error_rows
            ]
        aggregates.append(aggregate)
    return aggregates


COLUMNS = [
    ("Model", "model"),
    ("Accuracy", "accuracy"),
    ("Resolved", "resolved"),
    ("Tasks", "n_tasks"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Terminal-Bench on a TrustedRouter panel (agentic coding).")
    parser.add_argument("--models", default=None, help="Panel name or comma list of TR model ids.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-path", default=None,
                        help="Path to a local Terminal-Bench dataset directory. Use this for "
                             "prepared upstream subsets such as .data/terminal-bench-hard-short/tasks.")
    parser.add_argument("--agent", default="terminus-2",
                        help="Terminal-Bench agent. terminus-2 (default) is robust to free-form "
                             "model output; terminus-1 demands strict JSON and fatally bails "
                             "(fatal_llm_parse_error) on models that emit prose.")
    parser.add_argument("--agent-kwarg", action="append", default=[],
                        help="Pass through an agent kwarg to `tb run`, e.g. "
                             "`--agent-kwarg temperature=0.0`. Repeatable.")
    parser.add_argument("--context-tokens", type=int, default=None,
                        help="Override litellm's context-token estimate inside the TrustedRouter "
                             "SDK shim. Useful for models whose litellm max_tokens entry is an "
                             "output cap rather than an input context window.")
    parser.add_argument("--tasks", default=None,
                        help="Comma list of task IDs, or a named subset: default, hard-short. "
                             "Default = curated 10-task subset.")
    parser.add_argument("--n-concurrent", type=int, default=2)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--per-model-timeout", type=float, default=5400.0)
    parser.add_argument("--per-task", action="store_true",
                        help="Run each model/task pair as its own tb run and append a task row "
                             "as soon as it finishes. This is slower but safer for expensive "
                             "agentic sweeps.")
    parser.add_argument("--per-task-timeout", type=float, default=None,
                        help="Subprocess timeout for each model/task run when --per-task is set. "
                             "Defaults to --per-model-timeout.")
    parser.add_argument("--per-task-order", choices=["model-major", "task-major"], default="model-major",
                        help="Execution order for --per-task. model-major runs all tasks for one model before "
                             "moving on; task-major runs each task across all models before the next task.")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default="results/terminal_bench.json")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models already in the sidecar JSONL.")
    parser.add_argument("--use-sdk", dest="use_sdk", action=argparse.BooleanOptionalAction, default=True,
                        help="Route the agent's calls through the TrustedRouter SDK (litellm custom "
                             "provider). --no-use-sdk falls back to the OpenAI-compatible base URL.")
    parser.add_argument("--svg", default="assets/terminal_bench.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    api_key = client.api_key_from_env(args.api_key)
    tb_bin = shutil.which("tb")
    if not tb_bin:
        raise SystemExit("tb CLI not found. Install: uv tool install terminal-bench (needs Docker running).")

    if args.tasks:
        task_spec = args.tasks.strip()
        tasks = list(TASK_SUBSETS[task_spec]) if task_spec in TASK_SUBSETS else [
            t.strip() for t in task_spec.split(",") if t.strip()
        ]
    else:
        tasks = list(DEFAULT_SUBSET)
    models = resolve_panel(args.models)
    output_root = Path(args.output_root).resolve()
    env = {**os.environ}
    if args.use_sdk:
        # Force the sitecustomize shim + repo root onto the tb subprocess PYTHONPATH
        # so `trustedrouter/<model>` routes through the TrustedRouter SDK. The tb
        # tool venv must have the SDK installed: `uv tool install terminal-bench
        # --with /path/to/trusted-router-py`.
        env["TRUSTEDROUTER_API_KEY"] = api_key
        env["TRUSTEDROUTER_BASE_URL"] = args.base_url
        if args.context_tokens:
            env["TRBENCH_CONTEXT_TOKENS"] = str(args.context_tokens)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_SHIM_DIR), str(_REPO_ROOT), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    else:
        env["OPENAI_API_BASE"] = args.base_url
        env["OPENAI_API_KEY"] = api_key

    print(f"terminal-bench [{args.dataset}]: {len(models)} agents x {len(tasks)} tasks "
          f"(agent={args.agent}, transport={'TR-SDK' if args.use_sdk else 'openai-base-url'})", flush=True)

    # Per-model sidecar JSONL: each model's row is appended as it finishes, so a kill
    # mid-panel keeps completed models and --resume skips them (each model run is long).
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".jsonl")

    if args.per_task:
        task_sidecar = out.with_name(out.stem + "_tasks.jsonl")
        done_pairs: set[tuple[str, str]] = set()
        task_rows = []
        if args.resume and task_sidecar.exists():
            for line in task_sidecar.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    task = row.get("task")
                    if task:
                        done_pairs.add((row["model"], task))
                    task_rows.append(row)
            print(f"  resume: {len(done_pairs)} model/task pairs already recorded", flush=True)

        timeout = args.per_task_timeout or args.per_model_timeout
        pairs = (
            [(model, task) for task in tasks for model in models]
            if args.per_task_order == "task-major"
            else [(model, task) for model in models for task in tasks]
        )
        with task_sidecar.open("a", encoding="utf-8") as sc:
            for model, task in pairs:
                if (model, task) in done_pairs:
                    continue
                row = _run_one(
                    tb_bin=tb_bin, model=model, model_arg=_model_arg(model, args.use_sdk),
                    dataset=args.dataset, tasks=[task], agent=args.agent,
                    n_concurrent=1, n_attempts=args.n_attempts, output_root=output_root,
                    env=env, timeout=timeout, timeout_mult=args.timeout_multiplier,
                    dataset_path=args.dataset_path, run_suffix=task,
                    agent_kwargs=args.agent_kwarg,
                )
                row["task"] = task
                task_rows.append(row)
                sc.write(json.dumps(row, ensure_ascii=False) + "\n")
                sc.flush()
                acc = row.get("accuracy", "err")
                print(f"    {model} / {task}: accuracy={acc}", flush=True)

        rows = _aggregate_task_rows(task_rows)
        rows.sort(key=lambda r: r["model"])
        sidecar.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

        good = [r for r in rows if "error" not in r]
        good.sort(key=lambda r: (-float(r["accuracy"]), -int(r["resolved"]), r["model"]))
        table = report.markdown_table(good, COLUMNS)
        print(table)

        result = {
            "eval": "terminal_bench", "dataset": args.dataset, "agent": args.agent,
            "created_at": datetime.now(UTC).isoformat(),
            "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
            "dataset_path": args.dataset_path,
            "per_task": True,
            "tasks": tasks, "models": models, "results": rows,
            "task_results": task_rows,
        }
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        print(f"wrote {task_sidecar}")
        return 0

    done_models: set[str] = set()
    rows = []
    if args.resume and sidecar.exists():
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done_models.add(row["model"])
                rows.append(row)
        print(f"  resume: {len(done_models)} models already recorded", flush=True)

    sc = sidecar.open("a", encoding="utf-8")
    for model in models:
        if model in done_models:
            continue
        row = _run_one(
            tb_bin=tb_bin, model=model, model_arg=_model_arg(model, args.use_sdk),
            dataset=args.dataset, tasks=tasks, agent=args.agent,
            n_concurrent=args.n_concurrent, n_attempts=args.n_attempts, output_root=output_root,
            env=env, timeout=args.per_model_timeout, timeout_mult=args.timeout_multiplier,
            dataset_path=args.dataset_path, agent_kwargs=args.agent_kwarg,
        )
        rows.append(row)
        sc.write(json.dumps(row, ensure_ascii=False) + "\n")
        sc.flush()
        acc = row.get("accuracy", "err")
        print(f"    {model}: accuracy={acc}", flush=True)
    sc.close()

    good = [r for r in rows if "error" not in r]
    good.sort(key=lambda r: (-float(r["accuracy"]), -int(r["resolved"]), r["model"]))
    for r in rows:
        if "error" in r:
            print(f"  !! {r['model']}: {r['error']}")

    table = report.markdown_table(good, COLUMNS)
    print(table)

    result = {
        "eval": "terminal_bench", "dataset": args.dataset, "agent": args.agent,
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "dataset_path": args.dataset_path,
        "tasks": tasks, "models": models, "results": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    svg = report.svg_bar_chart(
        good, score_key="accuracy", max_score=100,
        title="Terminal-Bench on TrustedRouter",
        subtitle=f"Agentic terminal/coding, {len(tasks)}-task subset, {args.agent} agent. Higher is better.",
    )
    Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.svg).write_text(svg, encoding="utf-8")
    print(f"wrote {args.svg}")

    if args.readme:
        rp = Path(args.readme)
        block = "\n\n".join([
            f"Terminal-Bench snapshot: `{result['created_at']}`. Dataset `{args.dataset}`, "
            f"{len(tasks)}-task curated subset, `{args.agent}` agent in a Docker sandbox. "
            f"Metric: accuracy (task resolved by its own unit tests).",
            f"![Terminal-Bench chart]({Path(args.svg).as_posix()})",
            table,
        ])
        readme = report.splice_readme(rp.read_text(encoding="utf-8"), "TERMINAL_BENCH_RESULTS", block)
        rp.write_text(readme, encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
