"""Deterministic checks for the GSM8K answer extraction + scorer."""
from __future__ import annotations

from trbench.evals.gsm8k import score as S
from trbench.evals.gsm8k.loader import gold_answer


def test_gold_answer_strips_marker_and_formatting() -> None:
    assert gold_answer("She has 5 apples and ... so\n#### 18") == "18"
    assert gold_answer("...\n#### 1,000") == "1000"
    assert gold_answer("...\n#### $42") == "42"


def test_extract_pred_prefers_marker() -> None:
    assert S.extract_pred("lots of 7 and 9 reasoning\n#### 42") == "42"
    # marker wins even when other numbers follow earlier
    assert S.extract_pred("step 1: 100\nstep 2: 50\n#### 150") == "150"


def test_extract_pred_falls_back_to_last_number() -> None:
    assert S.extract_pred("first 3 then 4, so the answer is 12") == "12"
    assert S.extract_pred("no numbers here") == ""
    assert S.extract_pred("") == ""


def test_normalize_num_canonicalizes() -> None:
    assert S.normalize_num("1,000") == "1000"
    assert S.normalize_num("$42") == "42"
    assert S.normalize_num("18.0") == "18"
    assert S.normalize_num("50%") == "50"
    assert S.normalize_num("1000") == S.normalize_num("1,000")


def test_score_model_counts_correct_wrong_and_errors() -> None:
    responses = [
        {"model": "m", "id": "0", "target": "18", "text": "reasoning\n#### 18"},   # correct
        {"model": "m", "id": "1", "target": "1000", "text": "the answer is 1,000"},  # correct (fallback + normalize)
        {"model": "m", "id": "2", "target": "5", "text": "#### 6"},                # wrong
        {"model": "m", "id": "3", "target": "7", "error": "http_502"},             # errored = wrong
    ]
    out = S.score_model(responses)
    assert out["total"] == 4
    assert out["correct"] == 2
    assert out["errors"] == 1
    assert out["score"] == 50.0


def test_summarize_sorts_by_accuracy() -> None:
    result = {
        "responses": [
            {"model": "good", "id": "0", "target": "1", "text": "#### 1"},
            {"model": "bad", "id": "0", "target": "1", "text": "#### 2"},
        ]
    }
    rows = S.summarize(result)
    assert [r["model"] for r in rows] == ["good", "bad"]
    assert rows[0]["score"] == 100.0
    assert rows[1]["score"] == 0.0
