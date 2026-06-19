"""Loads LiveCodeBench code_generation_lite (livecodebench/code_generation_lite,
CC license) at runtime by STREAMING the versioned JSONL — the files are huge
(test6.jsonl ~134MB) because every problem embeds its full test suite, so we read
line by line and stop once we have `limit` problems past the date cutoff rather
than downloading the whole thing.

LiveCodeBench is contamination-resistant by *time-windowing*: each release version
(test.jsonl = v1 … test6.jsonl = v6) adds problems from a later window. We default
to v6 and a recent `min_date` so the panel is graded on problems published after
the models' likely training cutoffs.

Test cases: `public_test_cases` is a JSON string; `private_test_cases` is
base64 -> zlib -> pickle (the official dataset's storage format). We decode both
into one `test_cases` list of {input, output, testtype}. testtype "stdin" =
stdin/stdout grading; a non-empty `starter_code` + `metadata.func_name` = LeetCode
functional grading. Cached normalized under .data/.

SECURITY: `private_test_cases` is a pickle from the dataset. We decode it here with
a restricted unpickler that only permits builtin containers (list/dict/str/…), so a
tampered blob can't execute arbitrary code on load. Solutions themselves run only
inside the sandbox container (see executor.py)."""
from __future__ import annotations

import base64
import io
import json
import os
import pickle
import urllib.request
import zlib
from pathlib import Path

VERSIONS = {
    "v1": "test.jsonl", "v2": "test2.jsonl", "v3": "test3.jsonl",
    "v4": "test4.jsonl", "v5": "test5.jsonl", "v6": "test6.jsonl",
}
BASE = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/{file}"
CACHE_DIR = Path(__file__).parents[3] / ".data"
DEFAULT_VERSION = "v6"
DEFAULT_MIN_DATE = "2024-08-01"  # past most 2024-era training cutoffs


class _SafeUnpickler(pickle.Unpickler):
    """Only allow plain container/scalar rebuilds — never arbitrary class/callable
    imports — so decoding the dataset's pickle can't execute code."""

    _ALLOWED = {
        ("builtins", "list"), ("builtins", "dict"), ("builtins", "tuple"),
        ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "str"),
        ("builtins", "bytes"), ("builtins", "int"), ("builtins", "float"),
        ("builtins", "bool"), ("builtins", "complex"), ("builtins", "NoneType"),
    }

    def find_class(self, module: str, name: str):  # noqa: D102
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in test-case pickle")


def _decode_public(s: str) -> list[dict]:
    return json.loads(s) if s else []


def _decode_private(s: str) -> list[dict]:
    if not s:
        return []
    obj = _SafeUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load()
    if isinstance(obj, (str, bytes)):
        obj = json.loads(obj)
    return list(obj)


def _norm(row: dict) -> dict:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    return {
        "id": str(row.get("question_id")),
        "platform": row.get("platform", ""),
        "contest_date": row.get("contest_date", ""),
        "difficulty": row.get("difficulty", ""),
        "question_content": row.get("question_content", ""),
        "starter_code": row.get("starter_code", "") or "",
        "func_name": meta.get("func_name", ""),
        "test_cases": (
            _decode_public(row.get("public_test_cases", ""))
            + _decode_private(row.get("private_test_cases", ""))
        ),
    }


def _stream(version: str, min_date: str, limit: int | None) -> list[dict]:
    url = BASE.format(file=VERSIONS[version])
    req = urllib.request.Request(url, headers={"User-Agent": "trbench/0.1"})
    out: list[dict] = []
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        for raw in resp:
            row = json.loads(raw)
            if str(row.get("contest_date", "")) < min_date:
                continue
            out.append(_norm(row))
            if limit is not None and len(out) >= limit:
                break
    return out


def load(limit: int | None = None, *, version: str = DEFAULT_VERSION, min_date: str = DEFAULT_MIN_DATE) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Limit is part of the cache key: a small `limit=N` pull must not be mistaken
    # for the full window on a later unbounded load.
    cache = CACHE_DIR / f"livecodebench_{version}_{min_date}_{limit if limit is not None else 'all'}.jsonl"
    if cache.exists():
        items = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        return items[:limit] if limit is not None else items
    items = _stream(version, min_date, limit)
    tmp = cache.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")
    os.replace(tmp, cache)
    return items
