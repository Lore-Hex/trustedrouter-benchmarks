"""Load BFCL v4 single-turn AST categories (gorilla / Berkeley Function-Calling
Leaderboard, Apache-2.0).

We use only the cheap, deterministic, no-execution categories: simple_python,
multiple, parallel, parallel_multiple (AST-checked against a ground-truth set of
acceptable values) and irrelevance (correct = the model makes NO call). Each data
item ships a function schema; we hand it to the model as an OpenAI `tools` entry
and grade the returned tool_calls with BFCL's own vendored ast_checker.

Data + ground truth are fetched from the upstream repo at runtime and cached
under .data/bfcl/. The ground-truth ("possible_answer") file is keyed by the same
id; irrelevance has no ground-truth file.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"
CACHE = Path(__file__).parents[3] / ".data" / "bfcl"

# Single-turn, deterministic categories. "irrelevance" carries function schemas
# but no ground truth — the model is supposed to decline to call.
AST_CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple"]
RELEVANCE_CATEGORIES = ["irrelevance"]
DEFAULT_CATEGORIES = AST_CATEGORIES + RELEVANCE_CATEGORIES
# Optional harder "live" (real user queries) AST variants — better discriminators.
LIVE_CATEGORIES = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]


def _fetch(rel: str) -> bytes:
    req = urllib.request.Request(f"{RAW}/{rel}", headers={"User-Agent": "trbench/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        return resp.read()


def _read_jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def _category_checker_kind(category: str) -> str:
    if "irrelevance" in category or "relevance" in category:
        return "relevance"
    return "ast"


def _ast_test_category(category: str) -> str:
    """The string BFCL's ast_checker routes on (it substring-matches
    'parallel'/'multiple'). Strip live_/_python so e.g. live_parallel_multiple
    still routes to the parallel branch."""
    return category.replace("live_", "").replace("_python", "")


def _load_category(category: str) -> list[dict]:
    cache_file = CACHE / f"{category}.jsonl"
    if cache_file.exists():
        return _read_jsonl(cache_file.read_bytes())

    data = _read_jsonl(_fetch(f"BFCL_v4_{category}.json"))
    kind = _category_checker_kind(category)
    ground: dict[str, list] = {}
    if kind == "ast":
        gt = _read_jsonl(_fetch(f"possible_answer/BFCL_v4_{category}.json"))
        ground = {row["id"]: row["ground_truth"] for row in gt}

    items = []
    for row in data:
        # question is [[{role,content},...]] — a list of turns; single-turn → [0]
        messages = row["question"][0] if row.get("question") else []
        funcs = row.get("function", [])
        if isinstance(funcs, dict):
            funcs = [funcs]
        items.append({
            "id": row["id"],
            "category": category,
            "checker_kind": kind,
            "ast_test_category": _ast_test_category(category),
            "messages": messages,
            "functions": funcs,
            "ground_truth": ground.get(row["id"]),  # None for relevance
        })

    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text("\n".join(json.dumps(it, ensure_ascii=True) for it in items) + "\n", encoding="utf-8")
    os.replace(tmp, cache_file)
    return items


def load(categories: list[str] | None = None, limit_per_category: int | None = None) -> list[dict]:
    cats = categories or DEFAULT_CATEGORIES
    out: list[dict] = []
    for cat in cats:
        items = _load_category(cat)
        if limit_per_category is not None:
            items = items[:limit_per_category]
        out.extend(items)
    return out
