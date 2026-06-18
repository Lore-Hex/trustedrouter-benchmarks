"""Offline guards for the SimpleQA Verified autorater.

No network: we assert the grader prompt carries Google's documented SimpleQA
Verified modifications (so the autorater match can't silently regress to the
original OpenAI prompt), and that the non-LLM bits of grade()/fscore() behave.
"""
from __future__ import annotations

from trbench import judge


def test_grader_prompt_has_googles_modifications():
    t = judge.GRADER_TEMPLATE
    # The numeric acceptable-range rule (the biggest score driver) and its example.
    assert "acceptable range" in t
    assert "anything between 118k and 122k" in t
    # The "force a committed direct answer" modification.
    assert "Only the direct answer matters" in t
    assert "commits to a single candidate answer" in t
    assert "without committing to a single correct answer are considered NOT_ATTEMPTED" in t
    # Format contract the parser depends on.
    for field in ("{question}", "{target}", "{predicted_answer}"):
        assert field in t
    assert 'Just return the letters "A", "B", or "C"' in t


def test_empty_prediction_is_not_attempted_without_a_judge_call():
    # No base_url/api_key reachable: an empty answer must short-circuit to
    # NOT_ATTEMPTED before any network call.
    assert judge.grade("Q", "T", "   ", judge_model="x", base_url="http://unused/v1", api_key="k") == "NOT_ATTEMPTED"


def test_fscore_is_harmonic_mean_of_overall_and_given_attempted():
    s = judge.fscore(correct=50, incorrect=30, not_attempted=20)
    assert s["correct"] == 50.0          # 50/100
    assert s["attempted"] == 80.0         # 80/100
    assert s["given_attempted"] == 62.5   # 50/80
    # harmonic mean of 0.50 and 0.625
    assert s["f1"] == 55.6


def test_fscore_handles_all_not_attempted():
    s = judge.fscore(correct=0, incorrect=0, not_attempted=10)
    assert s["f1"] == 0.0
    assert s["given_attempted"] == 0.0
