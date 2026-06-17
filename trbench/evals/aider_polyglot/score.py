"""Score an Aider polyglot run by running the real Exercism unit tests.

SECURITY: this executes model-generated Python. By default each test runs in a
locked-down, throwaway Docker container — ``--network none``, read-only root FS,
all capabilities dropped, ``no-new-privileges``, non-root, memory/CPU/PID limits,
``--rm``. ``--sandbox host`` falls back to running on the host (use only on a
throwaway VM) with a warning; ``--sandbox docker`` requires the container and
refuses to run unsandboxed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from trbench import report
from trbench.evals.aider_polyglot.loader import load


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


SANDBOX_IMAGE = "trbench-aider-sandbox:py312"
_SANDBOX_DOCKERFILE = "FROM python:3.12-slim\nRUN pip install --no-cache-dir pytest==8.3.4\n"
_image_ready: bool | None = None
_docker_ok: bool | None = None
_warned_host = False


def _docker_available() -> bool:
    global _docker_ok
    if _docker_ok is None:
        ok = shutil.which("docker") is not None
        if ok:
            try:
                ok = subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
            except Exception:  # noqa: BLE001
                ok = False
        _docker_ok = ok
    return _docker_ok


def _ensure_sandbox_image() -> bool:
    global _image_ready
    if _image_ready is not None:
        return _image_ready
    have = subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE], capture_output=True).returncode == 0
    if not have:
        print(f"building sandbox image {SANDBOX_IMAGE} (one-time)...")
        build = subprocess.run(
            ["docker", "build", "-q", "-t", SANDBOX_IMAGE, "-"],
            input=_SANDBOX_DOCKERFILE.encode(), capture_output=True,
        )
        have = build.returncode == 0
        if not have:
            print("  sandbox image build failed:", build.stderr.decode("utf-8", "replace")[:300])
    _image_ready = have
    return have


def _run_in_docker(workdir: str, test_name: str, timeout: float) -> bool:
    os.chmod(workdir, 0o777)  # noqa: S103 - throwaway temp dir so the unprivileged container can write __pycache__
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m",  # noqa: S108 - container tmpfs path, not a host temp file
        "--memory", "512m", "--cpus", "1.0", "--pids-limit", "256",
        "--user", "65534:65534",
        "-v", f"{workdir}:/work:rw", "-w", "/work",
        SANDBOX_IMAGE, "python", "-m", "pytest", "-q", test_name,
    ]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _run_on_host(workdir: str, test_name: str, timeout: float) -> bool:
    global _warned_host
    if not _warned_host:
        print("WARNING: running model-generated code UNSANDBOXED on the host. Use a throwaway VM.")
        _warned_host = True
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", test_name],
            cwd=workdir, capture_output=True, timeout=timeout,
        ).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def run_tests(
    code: str, stub_name: str, test_name: str, test_code: str, *, sandbox: str = "auto", timeout: float = 60.0
) -> bool:
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / stub_name).write_text(code, encoding="utf-8")
        (Path(td) / test_name).write_text(test_code, encoding="utf-8")
        if sandbox != "host" and _docker_available() and _ensure_sandbox_image():
            return _run_in_docker(td, test_name, timeout)
        if sandbox == "docker":
            raise SystemExit("--sandbox docker requested but Docker or the sandbox image is unavailable.")
        return _run_on_host(td, test_name, timeout)


def summarize(result: dict[str, Any], concurrency: int = 6, sandbox: str = "auto") -> list[dict[str, Any]]:
    # Resolve + build the sandbox image once, single-threaded, before the pool
    # (so the worker threads don't race to build it).
    if sandbox != "host" and _docker_available():
        _ensure_sandbox_image()
    ex = {e["id"]: e for e in load()}
    by_model: dict[str, list[dict]] = {}
    for r in result.get("responses", []):
        by_model.setdefault(str(r.get("model")), []).append(r)

    rows: list[dict[str, Any]] = []
    for model, resps in by_model.items():
        total = len(resps)
        errors = sum(1 for r in resps if r.get("error"))
        runnable = [r for r in resps if not r.get("error") and r["id"] in ex]

        def test_one(r: dict) -> bool:
            e = ex[r["id"]]
            return run_tests(
                extract_code(r.get("text", "")), e["stub_name"], e["test_name"], e["test_code"], sandbox=sandbox
            )

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            passed = sum(1 for ok in pool.map(test_one, runnable) if ok)
        rows.append(
            {
                "model": model,
                "pass_rate": round(100 * passed / total, 1) if total else 0.0,
                "passed": passed,
                "total": total,
                "errors": errors,
            }
        )
    rows.sort(key=lambda r: (-float(r["pass_rate"]), int(r["errors"]), r["model"]))
    return rows


COLUMNS = [("Model", "model"), ("Pass%", "pass_rate"), ("Passed", "passed"), ("Total", "total"), ("Errors", "errors")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an Aider polyglot (Python) run via real unit tests.")
    parser.add_argument("results")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--sandbox", choices=["auto", "docker", "host"], default="auto",
        help="auto: Docker if available else host (warns); docker: require the sandbox; host: unsandboxed.",
    )
    parser.add_argument("--svg", default="assets/aider_polyglot.svg")
    parser.add_argument("--readme", default=None)
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = summarize(result, concurrency=args.concurrency, sandbox=args.sandbox)
    table = report.markdown_table(rows, COLUMNS)
    print(table)

    svg = report.svg_bar_chart(
        rows, score_key="pass_rate", max_score=100,
        title="Aider polyglot (Python) on TrustedRouter",
        subtitle="Pass@1 on Exercism Python exercises, real unit tests. Higher is better.",
    )
    Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.svg).write_text(svg, encoding="utf-8")
    print(f"wrote {args.svg}")

    if args.readme:
        rp = Path(args.readme)
        created = str(result.get("created_at", "unknown"))
        n = result.get("item_count", "?")
        block = "\n\n".join(
            [
                f"Aider polyglot (Python subset) snapshot: `{created}`. {n} Exercism exercises, "
                "pass@1, real unit tests (no judge).",
                f"![Aider polyglot chart]({Path(args.svg).as_posix()})",
                table,
            ]
        )
        spliced = report.splice_readme(rp.read_text(encoding="utf-8"), "AIDER_POLYGLOT_RESULTS", block)
        rp.write_text(spliced, encoding="utf-8")
        print(f"updated {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
