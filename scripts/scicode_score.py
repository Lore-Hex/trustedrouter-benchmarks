"""Reconstruct SciCode generated-code files from a Workflow output, then score with the
OFFICIAL test runner (test_generated_code.py) against test_data.h5.

Run (from SciCode dir not required; paths are absolute):
  python scripts/scicode_score.py <workflow_output.json> haiku validation
"""
import json
import os
import sys
import importlib.util
from pathlib import Path

# Vendored SciCode lives at <repo>/scicode (override with SCICODE_HOME).
SCI = Path(os.environ.get("SCICODE_HOME") or (Path(__file__).resolve().parent.parent / "scicode"))
sys.path.insert(0, str(SCI / "src"))
from scicode.parse.parse import read_from_hf_dataset

OUT = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "haiku"
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "validation"
SPECIAL = {"13": 6, "62": 1, "76": 3}  # (prob_id -> skipped step) : no file, gold only as context

d = json.load(open(OUT))
eps = d["result"] if isinstance(d, dict) else d
deps_by_id = {p["problem_id"]: p["required_dependencies"] for p in read_from_hf_dataset(SPLIT)}

code_root = SCI / "eval_results" / "generated_code" / MODEL / "without_background"
if code_root.exists():
    for f in code_root.glob("*.py"):
        f.unlink()
code_root.mkdir(parents=True, exist_ok=True)

nfiles = 0
for e in eps:
    if not e:
        continue
    pid, code = e["problem_id"], e["code"]
    deps = deps_by_id[pid]
    for n in range(1, len(code) + 1):
        if SPECIAL.get(pid) == n:      # skipped step: gencode writes no file for it
            continue
        if code[n - 1] is None:
            continue
        prev = "\n".join(c if c is not None else "" for c in code[:n - 1])
        # faithful to gencode.save_response_with_steps: f'{deps}\n{prev}\n' + '\n' + python_code
        content = f"{deps}\n{prev}\n" + "\n" + code[n - 1]
        (code_root / f"{pid}.{n}.py").write_text(content, encoding="utf-8")
        nfiles += 1
print(f"wrote {nfiles} step files to {code_root}")

# run the OFFICIAL test runner with the venv python on PATH (test subprocess calls bare `python`)
os.chdir(SCI)
os.environ["PATH"] = str(SCI / ".venv" / "bin") + os.pathsep + os.environ["PATH"]
spec = importlib.util.spec_from_file_location("tgc", str(SCI / "eval/scripts/test_generated_code.py"))
tgc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tgc)
print(f"=== scoring {MODEL} on {SPLIT} (official runner) ===")
tgc.test_code(MODEL, SPLIT, "eval_results/generated_code", "eval_results/test_logs",
              "eval_results", with_background=False)
res = SCI / "eval_results" / f"{MODEL}_without_background.txt"
print("\n===== RESULT =====")
print(res.read_text())
