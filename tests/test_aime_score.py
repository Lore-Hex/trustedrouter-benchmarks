"""Deterministic checks for AIME integer extraction + scoring."""
from __future__ import annotations

from trbench.evals.aime import score as S


def test_extract_pred_prefers_boxed() -> None:
    assert S.extract_pred("lots of reasoning... \\boxed{042}") == 42
    assert S.extract_pred("first 7 then 9\nThe answer is \\boxed{204}.") == 204


def test_extract_pred_falls_back_to_last_integer() -> None:
    assert S.extract_pred("after working it out the answer is 314") == 314
    assert S.extract_pred("no number") is None
    assert S.extract_pred("") is None


def test_score_model_exact_integer_match() -> None:
    responses = [
        {"model": "m", "id": "0", "target": "42", "text": "x \\boxed{42}"},   # correct
        {"model": "m", "id": "1", "target": "7", "text": "answer is 007"},    # correct (007==7)
        {"model": "m", "id": "2", "target": "100", "text": "\\boxed{99}"},     # wrong
        {"model": "m", "id": "3", "target": "5", "error": "http_502"},         # errored = wrong
    ]
    out = S.score_model(responses)
    assert out["total"] == 4
    assert out["correct"] == 2
    assert out["errors"] == 1
    assert out["score"] == 50.0
