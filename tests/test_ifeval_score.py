"""Deterministic checks that the vendored IFEval verifiers are wired correctly."""
from __future__ import annotations

from trbench.evals.ifeval import score as S
from trbench.evals.ifeval.vendor import evaluation_lib as E


def _input(prompt: str, instruction_id: str, kwargs: dict) -> E.InputExample:
    return E.InputExample(key=1, instruction_id_list=[instruction_id], prompt=prompt, kwargs=[kwargs])


def test_no_comma_instruction_passes_and_fails() -> None:
    inp = _input("write without commas", "punctuation:no_comma", {})
    follows = S.score_model([inp], [{"model": "m", "key": 1, "prompt": "write without commas", "text": "no commas here at all"}])
    fails = S.score_model([inp], [{"model": "m", "key": 1, "prompt": "write without commas", "text": "yes, there are commas"}])
    assert follows["inst_strict"] == 100.0
    assert fails["inst_strict"] == 0.0


def test_errored_response_counts_as_fail() -> None:
    inp = _input("p", "punctuation:no_comma", {})
    out = S.score_model([inp], [{"model": "m", "key": 1, "prompt": "p", "error": "http_502"}])
    assert out["errors"] == 1
    assert out["score"] == 0.0


def test_word_count_instruction() -> None:
    inp = _input("write 50+ words", "length_constraints:number_words", {"num_words": 50, "relation": "at least"})
    long_text = "word " * 60
    short_text = "too short"
    assert S.score_model([inp], [{"model": "m", "key": 1, "prompt": "write 50+ words", "text": long_text}])["inst_strict"] == 100.0
    assert S.score_model([inp], [{"model": "m", "key": 1, "prompt": "write 50+ words", "text": short_text}])["inst_strict"] == 0.0
