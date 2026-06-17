"""Loads the GSM8K test split (1319 grade-school math word problems) from the
canonical OpenAI repo at runtime. Not vendored — cached gitignored under .data/.

Gold answers in GSM8K end with a ``#### <number>`` marker; ``target`` is that
number, normalized (commas and ``$`` stripped)."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
CACHE = Path(__file__).parents[3] / ".data" / "gsm8k_test.jsonl"


def gold_answer(answer: str) -> str:
    """Extract the gold number from a GSM8K answer field (after ``####``)."""
    return answer.split("####")[-1].strip().replace(",", "").replace("$", "")


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
            {"id": str(i), "question": row["question"], "target": gold_answer(row["answer"])}
        )
    return items[:limit] if limit else items
