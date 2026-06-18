"""Shared math answer-extraction helpers.

`last_boxed_only_string` and `remove_boxed` are the canonical \\boxed{} parsers
from the Hendrycks MATH repo (MIT) — see NOTICE. Both AIME and MATH-500 ask the
model to put its final answer in \\boxed{...} and read it back from here.
"""
from __future__ import annotations


def last_boxed_only_string(string: str) -> str | None:
    """Return the last ``\\boxed{...}`` (or ``\\fbox{...}``) substring, braces
    balanced, or None."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Strip the ``\\boxed{...}`` / ``\\boxed ...`` wrapper, returning the inside."""
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]
    left = "\\boxed{"
    if s[: len(left)] == left and s[-1:] == "}":
        return s[len(left) : -1]
    return s


def boxed_answer(text: str) -> str | None:
    """Convenience: the inside of the last \\boxed{} in ``text``, or None."""
    boxed = last_boxed_only_string(text or "")
    return remove_boxed(boxed) if boxed is not None else None
