"""Loads google/simpleqa-verified at runtime (not vendored — license is the
dataset's; we only cache it locally, gitignored)."""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

DATA_URL = "https://huggingface.co/datasets/google/simpleqa-verified/resolve/main/simpleqa_verified.csv"
CACHE = Path(__file__).parents[3] / ".data" / "simpleqa_verified.csv"


def load(limit: int | None = None) -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "trbench/0.1"})
        with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
            CACHE.write_bytes(resp.read())
    items: list[dict] = []
    with CACHE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append({"id": row["original_index"], "question": row["problem"], "target": row["answer"]})
    return items[:limit] if limit else items
