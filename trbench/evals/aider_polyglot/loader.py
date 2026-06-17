"""Loads the Python slice of Aider's polyglot benchmark (Exercism exercises).

Clones Aider-AI/polyglot-benchmark at runtime into .data/ (gitignored — not
vendored). Each exercise is a stub file + a real unit-test file + instructions;
the model fills the stub and we run the actual tests. This is the Python subset
(34 exercises) of the full 6-language, 225-exercise benchmark — see EVALS.md.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Aider-AI/polyglot-benchmark.git"
ROOT = Path(__file__).parents[3] / ".data" / "polyglot-benchmark"
PRACTICE = ROOT / "python" / "exercises" / "practice"


def ensure_repo() -> None:
    if not PRACTICE.exists():
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603, S607
            ["git", "clone", "--depth", "1", REPO_URL, str(ROOT)],
            check=True,
            capture_output=True,
        )


def load(limit: int | None = None) -> list[dict]:
    ensure_repo()
    out: list[dict] = []
    for d in sorted(PRACTICE.iterdir()):
        if not d.is_dir():
            continue
        pys = list(d.glob("*.py"))
        test = next((p for p in pys if p.name.endswith("_test.py")), None)
        stub = next((p for p in pys if not p.name.endswith("_test.py")), None)
        if not (stub and test):
            continue
        instr = []
        for f in ("instructions.md", "instructions.append.md"):
            p = d / ".docs" / f
            if p.exists():
                instr.append(p.read_text(encoding="utf-8"))
        out.append(
            {
                "id": d.name,
                "stub_name": stub.name,
                "stub_code": stub.read_text(encoding="utf-8"),
                "test_name": test.name,
                "test_code": test.read_text(encoding="utf-8"),
                "instructions": "\n\n".join(instr),
            }
        )
    return out[:limit] if limit else out
