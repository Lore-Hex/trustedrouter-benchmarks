"""Cluster-bootstrap error bars for a SciCode fusion-vs-solo comparison. Resamples PROBLEMS
(steps within a problem are correlated, so the cluster is the problem), gives 95% CIs on each
method's subproblem rate and on the (fusion - best_solo) difference. Reuses already-scored
correct_dicts in scicode/eval_results/<model>_<bg>.json — no recomputation.

Run:  python scripts/scicode_bootstrap.py shbig_fusion sonnet_solo_big,haiku \
          "21,22,...,46" test nobg
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

SCI = Path(os.environ.get("SCICODE_HOME") or (Path(__file__).resolve().parent.parent / "scicode"))
sys.path.insert(0, str(SCI / "src"))
from scicode.parse.parse import read_from_hf_dataset

FUSION = sys.argv[1]
SOLOS = sys.argv[2].split(",")
PIDS = sys.argv[3].split(",")
SPLIT = sys.argv[4] if len(sys.argv) > 4 else "test"
BGDIR = "with_background" if (len(sys.argv) > 5 and sys.argv[5] in ("bg", "1")) else "without_background"
B = int(sys.argv[6]) if len(sys.argv) > 6 else 10000

byid = {p["problem_id"]: p for p in read_from_hf_dataset(SPLIT)}
models = [FUSION] + SOLOS
dicts = {m: json.load(open(SCI / "eval_results" / f"{m}_{BGDIR}.json")) for m in models}
# per-problem list of (pass for each model) per step
perprob = {}
for pid in PIDS:
    rows = []
    for ss in byid[pid]["sub_steps"]:
        sid = ss["step_number"]
        rows.append(tuple(sid in dicts[m].get(pid, []) for m in models))
    perprob[pid] = rows


def rate(sample, idx):
    p = t = 0
    for pid in sample:
        for r in perprob[pid]:
            t += 1
            p += r[idx]
    return p / t


nsteps = sum(len(perprob[p]) for p in PIDS)
point = [rate(PIDS, i) for i in range(len(models))]
random.seed(42)
boot = [[] for _ in models]
bootdiff = []
best_solo_idx = 1 + max(range(len(SOLOS)), key=lambda j: point[1 + j])  # strongest solo
for _ in range(B):
    samp = [random.choice(PIDS) for _ in PIDS]
    rs = [rate(samp, i) for i in range(len(models))]
    for i, r in enumerate(rs):
        boot[i].append(r)
    bootdiff.append(rs[0] - rs[best_solo_idx])


def ci(a):
    return np.percentile(a, 2.5) * 100, np.percentile(a, 97.5) * 100


print(f"=== {FUSION} vs {SOLOS} — {len(PIDS)} problems / {nsteps} steps, cluster bootstrap B={B} ===")
for i, m in enumerate(models):
    lo, hi = ci(boot[i])
    print(f"  {m:18} {100*point[i]:5.1f}%   95% CI [{lo:.1f}, {hi:.1f}]")
lo, hi = ci(bootdiff)
pgt = 100 * np.mean(np.array(bootdiff) > 0)
best = models[best_solo_idx]
sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant (CI straddles 0)"
print(f"\n  fusion - {best} (best solo): {100*(point[0]-point[best_solo_idx]):+.1f} pts   "
      f"95% CI [{lo:+.1f}, {hi:+.1f}]   P(fusion>solo)={pgt:.1f}%  -> {sig}")
