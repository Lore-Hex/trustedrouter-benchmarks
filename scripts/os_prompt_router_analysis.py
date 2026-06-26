"""Prototype per-prompt OS model selector on the saved 26-row table.

This intentionally avoids benchmark-level routing. For each held-out question,
it computes features from the prompt, candidate answer text, consensus pattern,
and model priors learned from the other questions, then selects the model answer
with the highest hand-tuned probability score.
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OPEN_MODELS = {
    "Kimi": "moonshotai/kimi-k2.6",
    "GLM": "z-ai/glm-5.2",
    "Minimax": "minimax/minimax-m3",
    "DeepSeek": "deepseek/deepseek-v4-pro",
    "MiMo": "xiaomi/mimo-v2.5-pro",
    "Nemotron": "nvidia/nvidia-nemotron-3-ultra-550b-a55b",
    "Gemma": "google/gemma-4-31b-it",
}

AIIQ_IDS = {
    "Kimi": "kimi-k2.6",
    "GLM": "glm-5.2",
    "Minimax": "minimax-m3",
    "DeepSeek": "deepseek-v4-pro",
    "MiMo": "mimo-v2.5-pro",
    "Nemotron": "nvidia-nemotron-3-ultra",
    "Gemma": "gemma-4-31b",
}

AIIQ_URL = "https://www.aiiq.org/api/rankings"


def load_check_rows() -> list[dict[str, str]]:
    lines = [
        line
        for line in Path("results/expanded_bio_check_table.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


def load_aiiq_rankings() -> dict[str, Any]:
    cache = Path("results/aiiq_rankings_snapshot.json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = json.loads(urllib.request.urlopen(AIIQ_URL, timeout=30).read().decode())
    cache.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def normalized_aiiq_features(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    wanted = [
        "composite-iq",
        "scientific-reasoning-iq",
        "reliability-iq",
        "gpqa-diamond",
        "humanitys-last-exam",
        "ifbench",
    ]
    by_ranking = {r["id"]: r for r in data.get("rankings", [])}
    raw: dict[str, dict[str, float]] = {m: {} for m in OPEN_MODELS}
    for ranking_id in wanted:
        ranking = by_ranking.get(ranking_id)
        if not ranking:
            continue
        scores = {
            str(m["id"]): float(m["score"])
            for m in ranking.get("models", [])
            if isinstance(m.get("score"), int | float)
        }
        if not scores:
            continue
        values = list(scores.values())
        lo, hi = min(values), max(values)
        span = hi - lo or 1.0
        direction = ranking.get("direction", "higher_is_better")
        for model_name, aiiq_id in AIIQ_IDS.items():
            val = scores.get(aiiq_id)
            if val is None:
                raw[model_name][f"aiiq_{ranking_id}"] = 0.5
                raw[model_name][f"aiiq_{ranking_id}_present"] = 0.0
                continue
            norm = (val - lo) / span
            if direction == "lower_is_better":
                norm = 1.0 - norm
            raw[model_name][f"aiiq_{ranking_id}"] = norm
            raw[model_name][f"aiiq_{ranking_id}_present"] = 1.0

    # Task-adaptive prior: use the most relevant AIIQ benchmark/dimension.
    for model_name, feats in raw.items():
        feats["aiiq_science_prior"] = feats.get("aiiq_scientific-reasoning-iq", 0.5)
        feats["aiiq_reliability_prior"] = feats.get("aiiq_reliability-iq", 0.5)
        feats["aiiq_composite_prior"] = feats.get("aiiq_composite-iq", 0.5)
    return raw


def load_mixed() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    data = json.loads(Path("results/mixed_bio_reasoning_packet_16_open7_plus_frontier_refs_corrected.json").read_text())
    tasks = {task["id"]: task for task in data["tasks"]}
    responses = {(row["model"], row["id"]): row for row in data["responses"]}
    return tasks, responses


def load_litqa() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    files = [
        "results/lab_bench_litqa2_smoke10_6models_routed.json",
        "results/lab_bench_litqa2_gemma4_provider_deepinfra_10.json",
    ]
    items_by_uuid: dict[str, dict[str, Any]] = {}
    responses_by_uuid: dict[tuple[str, str], dict[str, Any]] = {}
    for file in files:
        data = json.loads(Path(file).read_text())
        for item in data.get("items", []):
            items_by_uuid[item["id"]] = item
        for row in data.get("responses", []):
            responses_by_uuid[(row["model"], row["id"])] = row

    # The expanded table's litqa2_N rows follow the item order from the 10-item LitQA run.
    ordered = list(json.loads(Path(files[0]).read_text())["items"])
    tasks: dict[str, dict[str, Any]] = {}
    responses: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, item in enumerate(ordered, 1):
        tid = f"litqa2_{idx}"
        tasks[tid] = {
            "id": tid,
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "answer": item.get("answer"),
            "scoring": "mcq_letter",
            "source": "litqa2",
        }
        for model in OPEN_MODELS.values():
            row = responses_by_uuid.get((model, item["id"]))
            if row:
                responses[(model, tid)] = row
    return tasks, responses


def prompt_features(task: dict[str, Any], set_name: str) -> dict[str, float]:
    text = " ".join(str(task.get(k, "")) for k in ("question", "prompt"))
    options = task.get("options") or []
    lower = text.lower()
    return {
        "is_mcq": float(bool(options) or task.get("scoring") == "mcq_letter"),
        "is_instruction": float(set_name == "IFBench" or task.get("scoring") == "ifbench_selected"),
        "has_count_constraint": float(any(w in lower for w in ("exactly", "count", "times", "keyword", "number"))),
        "has_format_constraint": float(any(w in lower for w in ("emoji", "json", "format", "sentence", "wrap"))),
        "prompt_len_log": math.log1p(len(text)),
        "n_options": float(len(options)),
        "bio_terms": float(sum(w in lower for w in ("gene", "protein", "variant", "cell", "rna", "dna", "antibiotic"))),
    }


def extract_letter(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"answer\s+is\s*\(?([A-J])\)?",
        r"final\s+answer\s*:?\s*\(?([A-J])\)?",
        r"correct\s+(?:answer|option)\s+is\s*\(?([A-J])\)?",
        r"^\s*\(?([A-J])\)?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I | re.M)
        if matches:
            return matches[-1].upper()
    return None


def answer_key(row: dict[str, Any], is_mcq: bool) -> str:
    text = str(row.get("text", "")).strip()
    if is_mcq:
        return extract_letter(text) or text[:80].strip().lower()
    return re.sub(r"\s+", " ", text.lower())[:160]


def answer_features(row: dict[str, Any], task_feat: dict[str, float]) -> dict[str, float]:
    text = str(row.get("text", ""))
    return {
        "answer_len_log": math.log1p(len(text)),
        "has_extracted_letter": float(extract_letter(text) is not None),
        "empty": float(not text.strip()),
        "has_error": float(bool(row.get("error"))),
        "finish_length": float(row.get("finish_reason") in {"length", "max_tokens"}),
        "final_format_ok": float(not task_feat["is_mcq"] or extract_letter(text) is not None),
    }


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50, min(50, x))))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def score_candidate(
    *,
    model_name: str,
    row: dict[str, Any],
    task_feat: dict[str, float],
    ans_feat: dict[str, float],
    consensus: int,
    train_rows: list[dict[str, Any]],
) -> float:
    model_train = [r for r in train_rows if r["model_name"] == model_name]
    model_correct = sum(r["correct"] for r in model_train)
    # Laplace-smoothed prior.
    prior = (model_correct + 1.0) / (len(model_train) + 2.0)

    similar = [
        r
        for r in model_train
        if r["is_mcq"] == task_feat["is_mcq"]
        and r["is_instruction"] == task_feat["is_instruction"]
        and r["has_count_constraint"] == task_feat["has_count_constraint"]
    ]
    sim_prior = (sum(r["correct"] for r in similar) + 1.0) / (len(similar) + 2.0)

    # Capability priors are intentionally weak; learned leave-one-out priors dominate.
    capability = 0.0
    if task_feat["is_instruction"]:
        capability += {"Minimax": 0.20, "Kimi": 0.12, "DeepSeek": 0.10, "Nemotron": 0.06}.get(model_name, 0.0)
    if task_feat["is_mcq"] and not task_feat["is_instruction"]:
        capability += {"MiMo": 0.10, "Nemotron": 0.08, "DeepSeek": 0.06, "Kimi": 0.04}.get(model_name, 0.0)

    logit = 0.0
    logit += 2.4 * (prior - 0.5)
    logit += 1.8 * (sim_prior - 0.5)
    logit += 0.22 * max(0, consensus - 1)
    logit += 0.30 * ans_feat["final_format_ok"]
    logit -= 1.25 * ans_feat["empty"]
    logit -= 0.90 * ans_feat["has_error"]
    logit -= 0.30 * ans_feat["finish_length"]
    logit += 0.95 * (float(row.get("aiiq_task_prior", 0.5)) - 0.5)
    logit += 0.45 * (float(row.get("aiiq_composite_prior", 0.5)) - 0.5)
    logit += capability
    return sigmoid(logit)


def vectorize_examples(examples: list[dict[str, Any]], feature_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    base_num = [
        "is_mcq",
        "is_instruction",
        "has_count_constraint",
        "has_format_constraint",
        "prompt_len_log",
        "n_options",
        "bio_terms",
        "answer_len_log",
        "has_extracted_letter",
        "empty",
        "has_error",
        "finish_length",
        "final_format_ok",
        "consensus",
        "aiiq_composite-iq",
        "aiiq_scientific-reasoning-iq",
        "aiiq_reliability-iq",
        "aiiq_gpqa-diamond",
        "aiiq_humanitys-last-exam",
        "aiiq_ifbench",
        "aiiq_science_prior",
        "aiiq_reliability_prior",
        "aiiq_composite_prior",
        "aiiq_task_prior",
    ]
    if feature_names is None:
        names = ["bias"]
        names.extend(base_num)
        names.extend(f"model={m}" for m in OPEN_MODELS)
        for m in OPEN_MODELS:
            names.extend(
                [
                    f"model={m}:is_instruction",
                    f"model={m}:is_mcq",
                    f"model={m}:has_count_constraint",
                    f"model={m}:has_format_constraint",
                    f"model={m}:consensus",
                ]
            )
    else:
        names = feature_names
    rows = []
    for ex in examples:
        vals = {"bias": 1.0}
        vals.update({name: float(ex.get(name, 0.0)) for name in base_num})
        for m in OPEN_MODELS:
            is_model = float(ex["model_name"] == m)
            vals[f"model={m}"] = is_model
            vals[f"model={m}:is_instruction"] = is_model * float(ex.get("is_instruction", 0.0))
            vals[f"model={m}:is_mcq"] = is_model * float(ex.get("is_mcq", 0.0))
            vals[f"model={m}:has_count_constraint"] = is_model * float(ex.get("has_count_constraint", 0.0))
            vals[f"model={m}:has_format_constraint"] = is_model * float(ex.get("has_format_constraint", 0.0))
            vals[f"model={m}:consensus"] = is_model * float(ex.get("consensus", 0.0))
        rows.append([vals.get(name, 0.0) for name in names])
    return np.asarray(rows, dtype=float), names


def fit_logistic(train_examples: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _ = vectorize_examples(train_examples, feature_names)
    y = np.asarray([ex["correct"] for ex in train_examples], dtype=float)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    mu[0] = 0.0
    sigma[0] = 1.0
    xs = (x - mu) / sigma
    w = np.zeros(xs.shape[1], dtype=float)
    lr = 0.08
    reg = 0.08
    for _ in range(700):
        p = sigmoid_np(xs @ w)
        grad = xs.T @ (p - y) / len(y)
        grad[1:] += reg * w[1:] / len(y)
        w -= lr * grad
    return w, mu, sigma


def logistic_scores(
    train_examples: list[dict[str, Any]],
    cands: list[dict[str, Any]],
    feature_names: list[str],
) -> list[tuple[float, dict[str, Any]]]:
    w, mu, sigma = fit_logistic(train_examples, feature_names)
    x, _ = vectorize_examples(cands, feature_names)
    p = sigmoid_np(((x - mu) / sigma) @ w)
    return sorted([(float(score), ex) for score, ex in zip(p, cands, strict=True)], key=lambda x: (-x[0], x[1]["model_name"]))


def main() -> int:
    check_rows = load_check_rows()
    aiiq_data = load_aiiq_rankings()
    aiiq_by_model = normalized_aiiq_features(aiiq_data)
    mixed_tasks, mixed_responses = load_mixed()
    litqa_tasks, litqa_responses = load_litqa()
    tasks = {**mixed_tasks, **litqa_tasks}
    responses = {**mixed_responses, **litqa_responses}

    examples: list[dict[str, Any]] = []
    for row in check_rows:
        tid = row["Question"]
        task = tasks.get(tid, {"id": tid, "question": tid, "scoring": "", "options": []})
        task_feat = prompt_features(task, row["Set"])
        keys = {}
        for model_name, model_id in OPEN_MODELS.items():
            resp = responses.get((model_id, tid), {})
            keys[model_name] = answer_key(resp, bool(task_feat["is_mcq"]))
        key_counts = Counter(k for k in keys.values() if k)
        for model_name, model_id in OPEN_MODELS.items():
            resp = responses.get((model_id, tid), {})
            ans_feat = answer_features(resp, task_feat)
            aiiq_feat = dict(aiiq_by_model.get(model_name, {}))
            if row["Set"] == "GPQA":
                aiiq_feat["aiiq_task_prior"] = aiiq_feat.get("aiiq_gpqa-diamond", 0.5)
            elif row["Set"] == "HLE":
                aiiq_feat["aiiq_task_prior"] = aiiq_feat.get("aiiq_humanitys-last-exam", 0.5)
            elif row["Set"] == "IFBench":
                aiiq_feat["aiiq_task_prior"] = aiiq_feat.get("aiiq_ifbench", 0.5)
            else:
                aiiq_feat["aiiq_task_prior"] = aiiq_feat.get("aiiq_science_prior", 0.5)
            examples.append(
                {
                    "question": tid,
                    "set": row["Set"],
                    "model_name": model_name,
                    "model_id": model_id,
                    "correct": int(row[model_name] == "✅"),
                    "answer_key": keys[model_name],
                    "consensus": key_counts.get(keys[model_name], 0),
                    **task_feat,
                    **ans_feat,
                    **aiiq_feat,
                }
            )

    feature_names = vectorize_examples(examples)[1]

    decisions = []
    logistic_decisions = []
    for heldout in check_rows:
        qid = heldout["Question"]
        train = [ex for ex in examples if ex["question"] != qid]
        cands = [ex for ex in examples if ex["question"] == qid]
        scored = [
            (
                score_candidate(
                    model_name=ex["model_name"],
                    row=ex,
                    task_feat=ex,
                    ans_feat=ex,
                    consensus=int(ex["consensus"]),
                    train_rows=train,
                ),
                ex,
            )
            for ex in cands
        ]
        scored.sort(key=lambda x: (-x[0], x[1]["model_name"]))
        best_score, best = scored[0]
        decisions.append(
            {
                "question": qid,
                "set": heldout["Set"],
                "picked_model": best["model_name"],
                "picked_score": round(best_score, 4),
                "correct": best["correct"],
                "answer_key": best["answer_key"],
                "top3": [
                    {
                        "model": ex["model_name"],
                        "score": round(score, 4),
                        "correct": ex["correct"],
                        "consensus": ex["consensus"],
                        "answer_key": ex["answer_key"],
                    }
                    for score, ex in scored[:3]
                ],
            }
        )

        learned = logistic_scores(train, cands, feature_names)
        learned_score, learned_best = learned[0]
        logistic_decisions.append(
            {
                "question": qid,
                "set": heldout["Set"],
                "picked_model": learned_best["model_name"],
                "picked_score": round(learned_score, 4),
                "correct": learned_best["correct"],
                "answer_key": learned_best["answer_key"],
                "top3": [
                    {
                        "model": ex["model_name"],
                        "score": round(score, 4),
                        "correct": ex["correct"],
                        "consensus": ex["consensus"],
                        "answer_key": ex["answer_key"],
                    }
                    for score, ex in learned[:3]
                ],
            }
        )

    total = sum(d["correct"] for d in decisions)
    logistic_total = sum(d["correct"] for d in logistic_decisions)
    by_set = defaultdict(lambda: [0, 0])
    for d in decisions:
        by_set[d["set"]][0] += int(d["correct"])
        by_set[d["set"]][1] += 1
    logistic_by_set = defaultdict(lambda: [0, 0])
    for d in logistic_decisions:
        logistic_by_set[d["set"]][0] += int(d["correct"])
        logistic_by_set[d["set"]][1] += 1

    single_scores = {
        model_name: sum(row[model_name] == "✅" for row in check_rows)
        for model_name in OPEN_MODELS
    }
    oracle = sum(any(row[m] == "✅" for m in OPEN_MODELS) for row in check_rows)
    result = {
        "method": "leave_one_question_out_prompt_router_dependency_free",
        "hand_tuned_score": total,
        "logistic_score": logistic_total,
        "total": len(decisions),
        "hand_tuned_accuracy": round(total / len(decisions), 3),
        "logistic_accuracy": round(logistic_total / len(logistic_decisions), 3),
        "by_set": {k: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in by_set.items()},
        "logistic_by_set": {
            k: {"correct": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 3)}
            for k, v in logistic_by_set.items()
        },
        "single_model_scores": single_scores,
        "best_single_score": max(single_scores.values()),
        "oracle_score": oracle,
        "hand_tuned_decisions": decisions,
        "logistic_decisions": logistic_decisions,
        "notes": [
            "One row per model/question; held-out question excluded when computing model priors.",
            "No correctness labels are used for the held-out question except final scoring.",
            "This is a dependency-free selector, not a trained gradient-boosted tree.",
        ],
    }
    out = Path("results/os_prompt_router_loo_analysis.json")
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"hand-tuned prompt-router LOO: {total}/{len(decisions)} ({total/len(decisions):.3f})")
    print(f"learned logistic prompt-router LOO: {logistic_total}/{len(logistic_decisions)} ({logistic_total/len(logistic_decisions):.3f})")
    print(f"best single OS: {max(single_scores.values())}/{len(decisions)}")
    print(f"OS oracle: {oracle}/{len(decisions)}")
    print("learned logistic by set:")
    for set_name, counts in sorted(logistic_by_set.items()):
        print(f"  {set_name}: {counts[0]}/{counts[1]}")
    print("hand-tuned by set:")
    for set_name, counts in sorted(by_set.items()):
        print(f"  {set_name}: {counts[0]}/{counts[1]}")
    print("learned logistic picks:")
    for d in logistic_decisions:
        mark = "✓" if d["correct"] else "x"
        print(f"  {mark} {d['question']:12} {d['set']:8} -> {d['picked_model']} ({d['picked_score']})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
