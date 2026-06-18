"""Loads MATH-500 (the standard 500-problem MATH test subset) from
HuggingFaceH4/MATH-500 at runtime. Not vendored — cached gitignored under .data/.

Each item carries the gold ``answer`` (a LaTeX expression), graded by the
vendored Hendrycks equivalence checker (see equiv.py)."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DATA_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
CACHE = Path(__file__).parents[3] / ".data" / "math500.jsonl"


def load(limit: int | None = None) -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "trbench/0.1"})
        with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
            CACHE.write_bytes(resp.read())
    items: list[dict] = []
    for i, line in enumerate(CACHE.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            {
                "id": str(row.get("unique_id", i)),
                "question": row["problem"],
                "target": str(row["answer"]),
            }
        )
    return items[:limit] if limit else items
