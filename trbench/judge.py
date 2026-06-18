"""Shared LLM-as-judge grading for the short-form factuality evals.

Uses the SimpleQA Verified autorater prompt — OpenAI's SimpleQA grader (MIT, from
openai/simple-evals) as modified by Google DeepMind / Google Research for SimpleQA
Verified (arXiv:2509.07968). The modifications, relative to the original, are the
ones their paper documents: it forces a single committed answer ("only the direct
answer matters"; multiple un-committed candidates grade NOT_ATTEMPTED), and it
honors the per-target numeric *acceptable range* the dataset ships in its gold
answers (e.g. "150 (acceptable range: anything between 148 and 152)") instead of
the original's stricter "correct to the last significant figure". Matching this
prompt — and grading with GPT-4.1, the published autorater model — is what brings
our SimpleQA Verified scores in line with Google's reported numbers.

The judge classifies a predicted answer as CORRECT / INCORRECT / NOT_ATTEMPTED
given the question and gold target, and we report the SimpleQA F-score (harmonic
mean of overall accuracy and accuracy-given-attempted).
"""
from __future__ import annotations

import re

from trbench import client

# The SimpleQA Verified autorater prompt, verbatim. The {question}/{target}/
# {predicted_answer} fields and the A/B/C answer format match the original
# SimpleQA grader, so the parser below is unchanged.
GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
Predicted answer 4: It's better to refer to Obama by his full name, Barack Hussein Obama II. He has two daughters: Malia and Sasha.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Only the direct answer matters; anything else beyond what is stated in the gold target (including extra information, corrections to the question itself, etc) does not matter, as long as it does not contradict the gold target.
    - Hedging and mentioning multiple candidate answers are permissible, provided that the response eventually commits to a single candidate answer, and that answer is correct.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target.
    - A hedged statement (e.g., "it is possible that", "although i'm not sure, i think") that commits to an incorrect answer is also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
Predicted answer 5: I cannot find any information about the number of Barack Obama's children.
Predicted answer 6: The provided text does not provide any information about the number of Barack Obama's children.
Predicted answer 7: I think it's either Malia and Sasha, or Malia and Jackie, or Joey and Malia. I'm not sure which one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.
    - Hedged statements that provide multiple candidate answers without committing to a single correct answer are considered NOT_ATTEMPTED.


Also note the following things:
- For grading questions where the answer is a number, the gold target will also specify the allowed range, and any predicted answer that falls in that range should be considered correct. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k (acceptable range: anything between 118k and 122k)".
    - Predicted answers "120k", "119k", and "120,314" are all CORRECT, because they fall within the range specified in the gold target.
    - Predicted answers "100k" and "113k" are INCORRECT, because they fall outside the range specified in the gold target.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m (acceptable range: anything between 1.72 m and 1.74 m)". The predicted answer "1.74" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


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
        # Generous so a reasoning judge isn't truncated before it emits the
        # A/B/C letter (the default gpt-4.1 autorater is non-reasoning and stops
        # after ~1 token, so this costs nothing extra there).
        max_tokens=2048,
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
