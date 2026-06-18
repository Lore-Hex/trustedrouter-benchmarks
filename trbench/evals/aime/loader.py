"""Loads AIME 2025 (30 competition problems, integer answers 0-999) from the
math-ai/aime25 dataset at runtime. Not vendored — cached gitignored under .data/.

AIME 2025 is the freshest set, which also makes it the least contamination-prone.
Answers are integers, so scoring is exact integer match (no math-equivalence)."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DATA_URL = "https://huggingface.co/datasets/math-ai/aime25/resolve/main/test.jsonl"
CACHE = Path(__file__).parents[3] / ".data" / "aime25.jsonl"


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
        row = json.loads(line)
        items.append(
            {"id": str(row.get("id")), "question": row["problem"], "target": str(row["answer"]).strip()}
        )
    return items[:limit] if limit else items
