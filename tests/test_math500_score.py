"""Deterministic checks for MATH-500: the vendored Hendrycks equivalence checker,
\\boxed{} extraction, and scoring."""
from __future__ import annotations

from trbench.evals.math500 import score as S
from trbench.evals.math500.equiv import is_equiv
from trbench.mathutil import boxed_answer


def test_is_equiv_handles_latex_normalization() -> None:
    assert is_equiv("\\frac{1}{2}", "\\frac{1}{2}")
    assert is_equiv("\\dfrac{1}{2}", "\\frac{1}{2}")        # dfrac->frac
    assert is_equiv("1/2", "\\frac{1}{2}")                  # a/b -> frac
    assert is_equiv("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})")  # left/right + spaces
    assert is_equiv("0.5", "\\frac{1}{2}")
    assert is_equiv("9", "9")
    assert not is_equiv("\\frac{1}{2}", "\\frac{1}{3}")
    assert not is_equiv("9", "10")


def test_boxed_answer_extraction() -> None:
    assert boxed_answer("so the answer is \\boxed{\\frac{14}{3}}") == "\\frac{14}{3}"
    assert boxed_answer("two boxes \\boxed{1} then \\boxed{2}") == "2"  # last one
    assert boxed_answer("no box here") is None


def test_score_model_uses_equivalence() -> None:
    responses = [
        {"model": "m", "id": "0", "target": "\\frac{1}{2}", "text": "... \\boxed{\\dfrac{1}{2}}"},  # correct (equiv)
        {"model": "m", "id": "1", "target": "9", "text": "the answer is \\boxed{9}"},               # correct
        {"model": "m", "id": "2", "target": "p - q", "text": "\\boxed{p+q}"},                        # wrong
        {"model": "m", "id": "3", "target": "3", "text": "I think it's 3 but forgot to box"},        # no answer
        {"model": "m", "id": "4", "target": "5", "error": "http_502"},                               # errored
    ]
    out = S.score_model(responses)
    assert out["total"] == 5
    assert out["correct"] == 2
    assert out["no_answer"] == 1
    assert out["errors"] == 1
    assert out["score"] == 40.0
