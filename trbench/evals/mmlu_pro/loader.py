"""Loads MMLU-Pro (TIGER-Lab/MMLU-Pro, test split — 12,032 ten-choice questions
across 14 categories) at runtime from the HF parquet (one download; the rows API
rate-limits the 120 paginated requests a full pull needs). Cached as normalized
JSONL under .data/, so only the first pull needs pyarrow + network.

The test split is grouped by category (all `business`, then `psychology`, …), so
a contiguous first-N subset is one subject and never matches the published
overall accuracy. `load(limit=N)` therefore takes a SYSTEMATIC STRIDE across the
full set, sampling every category in proportion. Options can be fewer than ten
upstream, so letters map by position: option i -> chr(ord('A') + i)."""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

PARQUET_URL = (
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet"
)
CACHE = Path(__file__).parents[3] / ".data" / "mmlu_pro.jsonl"


def _norm(row: dict, idx: int) -> dict:
    return {
        "id": str(row.get("question_id", idx)),
        "question": row["question"],
        "options": list(row["options"]),
        "answer": str(row["answer"]).strip(),  # gold letter, e.g. "I"
        "category": row.get("category", ""),
    }


def _download_all() -> list[dict]:
    import pyarrow.parquet as pq  # heavy; only needed on the first (uncached) pull

    req = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "trbench/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        raw = resp.read()
    table = pq.read_table(io.BytesIO(raw))
    return [_norm(row, i) for i, row in enumerate(table.to_pylist())]


def load(limit: int | None = None) -> list[dict]:
    if CACHE.exists():
        full = [json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        full = _download_all()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in full) + "\n", encoding="utf-8")

    if limit is None or limit >= len(full):
        return full
    # systematic stride -> proportional coverage of every (category-grouped) section
    stride = len(full) / limit
    return [full[int(i * stride)] for i in range(limit)]


def format_options(options: list[str]) -> str:
    """A. ... B. ... — letters assigned by position."""
    return "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))
