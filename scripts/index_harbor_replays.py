#!/usr/bin/env python3
"""Write a compact manifest for Harbor replay directories."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"parse_error": type(exc).__name__, "message": str(exc)}


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def file_info(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": rel(path, root),
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def call_summary(calls_dir: Path, root: Path, call_id: str) -> dict[str, Any]:
    paths = {
        "request": calls_dir / f"{call_id}.request.json",
        "response": calls_dir / f"{call_id}.response.json",
        "meta": calls_dir / f"{call_id}.meta.json",
        "stream": calls_dir / f"{call_id}.stream.jsonl",
        "live": calls_dir / f"{call_id}.live.txt",
    }
    meta = read_json(paths["meta"]) or {}
    out: dict[str, Any] = {
        "call_id": call_id,
        "started_at": meta.get("started_at"),
        "elapsed_ms": meta.get("elapsed_ms"),
        "model": meta.get("model"),
        "base_url": meta.get("base_url"),
        "attempt": meta.get("attempt"),
        "message_count": meta.get("message_count"),
        "message_chars": meta.get("message_chars"),
        "finish_reason": meta.get("finish_reason"),
        "content_chars": meta.get("content_chars"),
        "usage": meta.get("usage"),
        "trustedrouter": meta.get("trustedrouter"),
        "files": {
            kind: file_info(path, root) for kind, path in paths.items() if path.exists()
        },
    }
    if "error" in meta:
        out["error"] = {"type": meta.get("error"), "message": meta.get("message")}
    return out


def trial_summary(trial: Path, root: Path) -> dict[str, Any]:
    result = read_json(trial / "result.json") or {}
    agent_result = result.get("agent_result") or {}
    verifier_result = result.get("verifier_result") or {}
    calls_dir = trial / "agent" / "trustedrouter-calls"
    call_ids = sorted(
        {
            ".".join(path.name.split(".")[:1])
            for path in calls_dir.glob("call-*.*")
            if path.is_file()
        }
    )
    files = [p for p in trial.rglob("*") if p.is_file()]
    return {
        "trial": trial.name,
        "config_path": rel(trial / "config.json", root)
        if (trial / "config.json").exists()
        else None,
        "result_path": rel(trial / "result.json", root)
        if (trial / "result.json").exists()
        else None,
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "reward": verifier_result.get("rewards"),
        "agent_cost_usd": agent_result.get("cost_usd"),
        "agent_metadata": agent_result.get("metadata"),
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "trustedrouter_call_count": len(call_ids),
        "trustedrouter_calls": [
            call_summary(calls_dir, root, call_id) for call_id in call_ids
        ],
    }


def job_manifest(job: Path) -> dict[str, Any]:
    root = job.resolve()
    files = [p for p in root.rglob("*") if p.is_file()]
    trials = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    return {
        "schema": "harbor-replay-manifest-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "job_name": root.name,
        "job_path": str(root),
        "job_result_path": "result.json" if (root / "result.json").exists() else None,
        "job_log_path": "job.log" if (root / "job.log").exists() else None,
        "file_count": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "trials": [trial_summary(trial, root) for trial in trials],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument(
        "--output-name",
        default="replay_manifest.json",
        help="manifest filename to write inside each job directory",
    )
    args = parser.parse_args()

    output_name = Path(args.output_name)
    if output_name.name != args.output_name:
        raise SystemExit("--output-name must be a filename, not a path")

    for job in args.jobs:
        if not job.is_dir():
            raise SystemExit(f"not a directory: {job}")
        output = job / output_name
        output.write_text(
            json.dumps(job_manifest(job), indent=2, ensure_ascii=False) + "\n"
        )
        print(output)


if __name__ == "__main__":
    main()
