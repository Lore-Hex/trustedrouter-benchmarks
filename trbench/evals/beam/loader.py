"""Load the BEAM 128K (100K split) long-context memory benchmark.

BEAM = "Beyond a Million Tokens" (ICLR 2026).
HuggingFace: Mohammadta/BEAM, split "100K" — 20 conversations ~128K tokens each.

Each item is one probing question bundled with the full conversation context.
The conversation is formatted as chat messages; we append the probing question
as a final user turn and ask for a single concise answer.

Caches normalised items to .data/beam_100k.jsonl after the first HF pull.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

CACHE = Path(__file__).parents[3] / ".data" / "beam_100k.jsonl"

QUESTION_TYPES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

# Probing question types vary in their "ideal answer" field name
_IDEAL_KEYS = (
    "ideal_response", "ideal_answer", "answer", "expected_answer", "correct_answer",
)


def _flatten_chat(chat: list) -> list[dict]:
    """Convert BEAM's chat field (list of batches, each a flat list of turns) to [{role, content}]."""
    msgs: list[dict] = []
    for batch in chat:
        if isinstance(batch, list):
            for turn in batch:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    msgs.append({"role": role, "content": content})
        elif isinstance(batch, dict):
            # Fallback for flat-dict format
            role = batch.get("role", "user")
            content = batch.get("content", "")
            if content:
                msgs.append({"role": role, "content": content})
    return msgs


def _get_ideal(q: dict) -> str:
    for key in _IDEAL_KEYS:
        val = q.get(key)
        if val:
            return str(val)
    return ""


def _parse_probing_questions(raw: str | dict, conv_idx: int) -> list[dict]:
    """Parse probing_questions field (Python repr or dict) into flat question list."""
    if isinstance(raw, str):
        try:
            data = ast.literal_eval(raw)  # BEAM uses single-quote Python repr, not JSON
        except (ValueError, SyntaxError):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return []
    else:
        data = raw if isinstance(raw, dict) else {}

    items = []
    for qtype, questions in data.items():
        if qtype not in QUESTION_TYPES or not isinstance(questions, list):
            continue
        for q_idx, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            question_text = q.get("question") or q.get("probing_question", "")
            rubric = q.get("rubric", [])
            ideal = _get_ideal(q)
            difficulty = q.get("difficulty", "")
            if not question_text or not rubric:
                continue
            items.append({
                "id": f"conv{conv_idx:02d}_{qtype}_{q_idx}",
                "question_type": qtype,
                "question": question_text,
                "rubric": rubric if isinstance(rubric, list) else [rubric],
                "ideal_response": ideal,
                "difficulty": difficulty,
            })
    return items


def _download_all() -> list[dict]:
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit("datasets library required: uv add datasets") from e

    print("Downloading BEAM 100K from HuggingFace (one-time)…")
    ds = load_dataset("Mohammadta/BEAM", split="100K")

    all_items: list[dict] = []
    for conv_idx, record in enumerate(ds):
        conv_id = str(record.get("conversation_id", f"conv{conv_idx:02d}"))
        messages = _flatten_chat(record.get("chat", []))

        questions = _parse_probing_questions(record.get("probing_questions", "{}"), conv_idx)

        for q in questions:
            all_items.append({
                **q,
                "conversation_id": conv_id,
                "messages": messages,
            })

    print(f"  {len(ds)} conversations → {len(all_items)} probing questions")
    return all_items


def load(limit: int | None = None, questions_per_type: int | None = None) -> list[dict]:
    """Return probing-question items with their full conversation context.

    questions_per_type: max questions per (conversation, question_type) pair.
        Default=1 gives ~200 items (cost-effective first pass).
        Pass 0 or None to use all questions (~400 items).
    """
    if questions_per_type is None:
        questions_per_type = 1

    if CACHE.exists():
        full = [json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        full = _download_all()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(
            "\n".join(json.dumps(it, ensure_ascii=True) for it in full) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, CACHE)

    if questions_per_type > 0:
        seen: dict[str, int] = {}
        sampled = []
        for it in full:
            key = f"{it['conversation_id']}::{it['question_type']}"
            seen[key] = seen.get(key, 0)
            if seen[key] < questions_per_type:
                sampled.append(it)
                seen[key] += 1
        full = sampled

    if limit is None or limit >= len(full):
        return full
    stride = len(full) / limit
    return [full[int(i * stride)] for i in range(limit)]
