"""Merge the per-batch scored correct_dicts (each covers a disjoint set of problem_ids) into a
single all-65 correct_dict, so scicode_bootstrap.py can run over the full test split.

The batch->pid groups are fixed for the full-65 Sonnet+Haiku fusion experiment (SHBIG reused 23 +
B1 16 + B2 16 + B3 10 = 65, the exact test split). Each source dict pre-seeds all 80 problem
slots but only its own pids carry real pass-lists, and the groups are disjoint, so taking each
source's own pids is conflict-free.

Run:  python scripts/scicode_merge65.py
        -> writes scicode/eval_results/all65_fusion_without_background.json
                  scicode/eval_results/all65_sonnet_without_background.json
"""
import json
import os
import sys
from pathlib import Path

SCI = Path(os.environ.get("SCICODE_HOME") or (Path(__file__).resolve().parent.parent / "scicode"))
ER = SCI / "eval_results"
BGDIR = "without_background"

# (pids, fusion-source dict, sonnet-source dict)
GROUPS = [
    (['21','22','23','24','25','26','27','28','30','31','32','33','34','35','36','37','39','40','41','42','43','45','46'],
     "shbig_fusion", "sonnet_solo_big"),
    (['77','11','12','13','2','74','5','8','9','14','15','16','17','18','20','75'], "b1_fus", "b1_son"),
    (['48','79','76','50','52','53','54','55','56','57','58','59','60','61','62','63'], "b2_fus", "b2_son"),
    (['64','65','66','67','80','68','69','71','72','73'], "b3_fus", "b3_son"),
]


def load(m):
    return json.load(open(ER / f"{m}_{BGDIR}.json"))


fusion, sonnet = {}, {}
allpids = []
for pids, fsrc, ssrc in GROUPS:
    fd, sd = load(fsrc), load(ssrc)
    for pid in pids:
        fusion[pid] = fd.get(pid, [])
        sonnet[pid] = sd.get(pid, [])
        allpids.append(pid)

assert len(allpids) == len(set(allpids)) == 65, f"expected 65 unique pids, got {len(allpids)}/{len(set(allpids))}"
json.dump(fusion, open(ER / f"all65_fusion_{BGDIR}.json", "w"))
json.dump(sonnet, open(ER / f"all65_sonnet_{BGDIR}.json", "w"))

fsteps = sum(len(v) for v in fusion.values())
ssteps = sum(len(v) for v in sonnet.values())
print(f"merged 65 problems -> all65_fusion / all65_sonnet ({BGDIR})")
print(f"  fusion passing steps: {fsteps}   sonnet passing steps: {ssteps}")
print(f"  pids: {sorted(allpids, key=lambda x:int(x))}")
