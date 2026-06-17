"""Shared LLM-as-judge grading for the short-form factuality evals.

Uses OpenAI's canonical SimpleQA grader template verbatim (MIT, from
openai/simple-evals). The judge classifies a predicted answer as CORRECT /
INCORRECT / NOT_ATTEMPTED given the question and gold target, and we report the
SimpleQA F-score (harmonic mean of overall accuracy and accuracy-given-attempted).
"""
from __future__ import annotations

import re

from trbench import client

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
- Do not punish for typos in people's name if it's clearly the same name.


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

_LETTER = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def grade(
    question: str,
    target: str,
    predicted: str,
    *,
    judge_model: str,
    base_url: str,
    api_key: str,
    timeout: float = 60.0,
) -> str:
    """Returns CORRECT / INCORRECT / NOT_ATTEMPTED / JUDGE_ERROR."""
    if not (predicted or "").strip():
        return "NOT_ATTEMPTED"
    prompt = GRADER_TEMPLATE.format(question=question, target=target, predicted_answer=predicted)
    r = client.chat(
        base_url=base_url,
        api_key=api_key,
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8,
        temperature=0.0,
        timeout=timeout,
    )
    if r.get("error"):
        return "JUDGE_ERROR"
    m = re.search(r"[ABC]", (r.get("text") or "").strip().upper())
    return _LETTER.get(m.group(0), "NOT_ATTEMPTED") if m else "NOT_ATTEMPTED"


def fscore(correct: int, incorrect: int, not_attempted: int) -> dict[str, float]:
    total = correct + incorrect + not_attempted
    attempted = correct + incorrect
    overall = correct / total if total else 0.0
    given_attempted = correct / attempted if attempted else 0.0
    f1 = 2 * overall * given_attempted / (overall + given_attempted) if (overall + given_attempted) else 0.0
    return {
        "f1": round(100 * f1, 1),
        "correct": round(100 * overall, 1),
        "given_attempted": round(100 * given_attempted, 1),
        "attempted": round(100 * attempted / total, 1) if total else 0.0,
        "not_attempted": round(100 * not_attempted / total, 1) if total else 0.0,
    }
