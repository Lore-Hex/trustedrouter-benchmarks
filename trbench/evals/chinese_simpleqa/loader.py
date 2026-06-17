"""Loads OpenStellarTeam/Chinese-SimpleQA at runtime.

NOTE: the upstream repo ships no LICENSE file, so redistribution terms are
unclear. We do NOT vendor the data — it is downloaded at run time and cached
locally (gitignored). Confirm the license before republishing the dataset
itself; publishing only aggregate scores is fine.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DATA_URL = "https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA/resolve/main/chinese_simpleqa.jsonl"
CACHE = Path(__file__).parents[3] / ".data" / "chinese_simpleqa.jsonl"


def load(limit: int | None = None) -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "trbench/0.1"})
        with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
            CACHE.write_bytes(resp.read())
    items: list[dict] = []
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        items.append({"id": r["id"], "question": r["question"], "target": r["answer"]})
    return items[:limit] if limit else items
