"""Containerized grading for LiveCodeBench solutions.

Model-generated code is UNTRUSTED, so every solution runs in a throwaway Docker
container with no network, capped memory/CPU/pids, and a wall-clock timeout — never
in this process. One container per solution runs all of that problem's test cases
(via an in-container runner) and reports pass/fail, so we pay container startup once
per solution, not once per test case.

Scope: stdin/stdout problems (AtCoder/Codeforces — empty `starter_code`). LeetCode
functional problems (non-empty starter_code + func_name) need call-based harnessing
and are skipped for now (see `gradeable`).
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

IMAGE = "python:3.11-slim"
PER_CASE_TIMEOUT = 6  # seconds, LiveCodeBench's default per-test budget

# Runs INSIDE the container. Reads cases.json + solution.py from /work (read-only),
# runs the solution per stdin case, prints a JSON verdict to stdout. Stops at the
# first failure (pass@1 needs every case to pass).
_RUNNER_SRC = r'''
import json, subprocess, sys
cases = json.load(open("/work/cases.json"))
def norm(s):
    return "\n".join(line.rstrip() for line in str(s).strip().splitlines())
def verdict(passed, fail, why):
    print(json.dumps({"passed": passed, "total": len(cases), "fail": fail, "why": why}))
    sys.exit()
passed = 0
for i, c in enumerate(cases):
    try:
        p = subprocess.run([sys.executable, "/work/solution.py"], input=c["input"],
                           capture_output=True, text=True, timeout=PER_CASE)
    except subprocess.TimeoutExpired:
        verdict(passed, i, "timeout")
    if p.returncode != 0:
        verdict(passed, i, "runtime:" + p.stderr[-200:])
    if norm(p.stdout) != norm(c["output"]):
        verdict(passed, i, "wrong_answer")
    passed += 1
verdict(passed, None, "ok")
'''


def extract_code(text: str) -> str | None:
    """Pull the solution out of the model's reply: the last fenced code block,
    else the whole text if it looks like code."""
    if not text:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    if "def " in text or "import " in text or "input()" in text:
        return text.strip()
    return None


def gradeable(item: dict) -> bool:
    """True for stdin/stdout problems we can currently grade (no LeetCode functional)."""
    return not item.get("starter_code") and not item.get("func_name") and bool(item.get("test_cases"))


def run_solution(code: str, test_cases: list[dict], *, timeout_total: float = 240.0) -> dict:
    """Grade `code` against `test_cases` in a locked-down container. Returns
    {passed, total, fail, why} — pass@1 is fail is None."""
    runner = _RUNNER_SRC.replace("PER_CASE", str(PER_CASE_TIMEOUT))
    with tempfile.TemporaryDirectory() as d:
        Path(d, "solution.py").write_text(code, encoding="utf-8")
        Path(d, "cases.json").write_text(json.dumps(test_cases), encoding="utf-8")
        Path(d, "runner.py").write_text(runner, encoding="utf-8")
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
            "-v", f"{d}:/work:ro", IMAGE,
            "python", "/work/runner.py",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_total)
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": len(test_cases), "fail": 0, "why": "container_timeout"}
        out = (proc.stdout or "").strip().splitlines()
        for line in reversed(out):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {"passed": 0, "total": len(test_cases), "fail": 0, "why": "no_verdict:" + (proc.stderr or "")[-200:]}
