"""Prepare a local Terminal-Bench task subset from upstream original-tasks.

The current Terminal-Bench registry exposes `terminal-bench-core==0.1.1`, but a
few useful hard tasks live in upstream `original-tasks` and are not downloaded by
the registry's `head` path. This script exports only the chosen tasks into an
ignored `.data/.../tasks` directory so the repo records the selection without
vendoring benchmark data.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

HARD_SHORT_TASKS = [
    "cancel-async-tasks",
    "fix-code-vulnerability",
    "polyglot-rust-c",
    "configure-git-webserver",
    "feal-linear-cryptanalysis",
    "password-recovery",
]


UPSTREAM_URL = "https://github.com/laude-institute/terminal-bench"


def copy_subset(src_root: Path, dst_root: Path, tasks: list[str]) -> list[dict]:
    copied = []
    tasks_dir = dst_root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for task_id in tasks:
        src = src_root / "original-tasks" / task_id
        if not src.exists():
            raise FileNotFoundError(f"missing upstream task: {src}")
        dst = tasks_dir / task_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        meta = {"task_id": task_id}
        task_yaml = src / "task.yaml"
        if task_yaml.exists():
            text = task_yaml.read_text(encoding="utf-8")
            for key in [
                "difficulty",
                "category",
                "max_agent_timeout_sec",
                "max_test_timeout_sec",
                "expert_time_estimate_min",
                "junior_time_estimate_min",
            ]:
                prefix = f"{key}:"
                for line in text.splitlines():
                    if line.startswith(prefix):
                        meta[key] = line.split(":", 1)[1].strip()
                        break
        copied.append(meta)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the hard-short Terminal-Bench subset.")
    parser.add_argument("--out-dir", default=".data/terminal-bench-hard-short")
    parser.add_argument("--source", default=None, help="Existing terminal-bench checkout. Defaults to a shallow clone.")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--tasks", default=",".join(HARD_SHORT_TASKS))
    parser.add_argument("--manifest", default="results/terminal_bench_hard_short_manifest.json")
    args = parser.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    if args.source:
        src_root = Path(args.source).resolve()
        copied = copy_subset(src_root, out_dir, tasks)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src_root, text=True).strip()
    else:
        with tempfile.TemporaryDirectory() as td:
            src_root = Path(td) / "terminal-bench"
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", args.ref, UPSTREAM_URL, str(src_root)],
                check=True,
            )
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src_root, text=True).strip()
            copied = copy_subset(src_root, out_dir, tasks)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_url": UPSTREAM_URL,
        "source_ref": args.ref,
        "source_commit": commit,
        "dataset_path": str(out_dir / "tasks"),
        "subset": "hard-short",
        "tasks": copied,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'tasks'}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
