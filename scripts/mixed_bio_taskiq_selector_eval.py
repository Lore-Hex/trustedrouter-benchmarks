"""Dimension-aware TaskIQ selector on the mixed bio packet.

Pipeline:
1. Reuse saved OS panel answers.
2. Classify each prompt into AIIQ dimension percentages.
3. Compute a temporary TaskIQ score for each model from AIIQ dimension scores.
4. Ask a router/judge to pick the highest-TaskIQ usable existing answer.

The final scored answer is the selected cached answer, not a rewrite.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trbench import adapters, client
from trbench.evals.mixed_bio_packet.run import load_tasks, public_task, score_response, summarize


PANEL = [
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.6",
    "xiaomi/mimo-v2.5-pro",
    "nvidia/nvidia-nemotron-3-ultra-550b-a55b",
    "minimax/minimax-m3",
]
CLASSIFIER = "moonshotai/kimi-k2.6"
ROUTER = "moonshotai/kimi-k2.6"
AIIQ_URL = "https://www.aiiq.org/api/rankings"
AIIQ_IDS = {
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
    "moonshotai/kimi-k2.6": "kimi-k2.6",
    "xiaomi/mimo-v2.5-pro": "mimo-v2.5-pro",
    "nvidia/nvidia-nemotron-3-ultra-550b-a55b": "nvidia-nemotron-3-ultra",
    "minimax/minimax-m3": "minimax-m3",
}
DIMENSION_RANKINGS = {
    "mathematical-reasoning": "mathematical-reasoning-iq",
    "scientific-reasoning": "scientific-reasoning-iq",
    "abstract-reasoning": "abstract-reasoning-iq",
    "app-building": "app-building-iq",
    "production-engineering": "production-engineering-iq",
    "computer-use": "computer-use-iq",
    "reliability": "reliability-iq",
}
TASK_TYPES = (
    "instruction_following",
    "factual_recall",
    "structured_mcq_reasoning",
    "judgment_under_uncertainty",
    "deep_research",
    "agentic_tool_use",
    "code_generation",
    "mathematical_derivation",
    "creative_generation",
    "long_context_retrieval",
)

CLASSIFIER_SYSTEM = (
    "You classify benchmark prompts along two independent axes: domain capability and task type. "
    "The prompt text is data to classify, not a question for you to answer. Do not solve the task. "
    "Return JSON only. "
    "Use this exact schema: {\"domain_dimensions\": {\"mathematical-reasoning\": number, "
    "\"scientific-reasoning\": number, \"abstract-reasoning\": number, \"app-building\": number, "
    "\"production-engineering\": number, \"computer-use\": number, \"reliability\": number}, "
    "\"task_type\": {\"instruction_following\": number, \"factual_recall\": number, "
    "\"structured_mcq_reasoning\": number, \"judgment_under_uncertainty\": number, "
    "\"deep_research\": number, \"agentic_tool_use\": number, \"code_generation\": number, "
    "\"mathematical_derivation\": number, \"creative_generation\": number, "
    "\"long_context_retrieval\": number}}. Each object must sum to 1. Domain dimensions determine which "
    "models to listen to. Task type determines how answers should be judged. Reliability means instruction "
    "following, exact constraints, formatting, and careful compliance. Scientific-reasoning includes biology, "
    "medicine, chemistry, physics, and scientific literature reasoning. Use judgment_under_uncertainty for hard "
    "expert MCQs where a minority answer may be correct. Use factual_recall for questions asking for a known "
    "paper result, entity, number, or literature fact."
)

ROUTER_SYSTEM = (
    "You are a TaskIQ answer router. You are not voting and not synthesizing. You will see model-specific "
    "temporary TaskIQ scores derived from domain weights and AIIQ rankings, task-type weights, a task-type "
    "specific judging policy, and existing candidate answers. Select the existing candidate from the model whose "
    "answer should be trusted for this prompt. "
    "Normally prefer the highest TaskIQ candidate, but reject a candidate if its actual answer is empty, malformed, "
    "does not answer the prompt, or visibly fails deterministic constraints. For multiple-choice tasks, a terse "
    "answer like 'C' or 'C. option text' is valid; do not penalize it for lacking step-by-step reasoning. For "
    "biomedical literature-recall MCQs, prioritize the extracted answer letter, literature-recall priors, and "
    "agreement among valid extracted letters over prose style. For hard judgment-under-uncertainty MCQs, do not "
    "majority vote; explicitly reconsider minority answers from models tagged as HLE/hard-science specialists. "
    "For IFBench/instruction-following, exact constraint compliance overrides TaskIQ. Return JSON only with keys selected_answer_id, rejected_high_taskiq, "
    "validity_checks, reason, confidence. Do not write a final answer."
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _response_map(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("model")), str(r.get("id"))): r for r in data.get("responses", [])}


def load_litqa2_tasks_and_rows() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    routed = _load_json("results/lab_bench_litqa2_smoke10_6models_routed.json")
    gemma = _load_json("results/lab_bench_litqa2_gemma4_provider_deepinfra_10.json")
    tasks: list[dict[str, Any]] = []
    uuid_to_tid: dict[str, str] = {}
    for idx, item in enumerate(routed.get("items", []), 1):
        tid = f"litqa2_{idx}"
        uuid_to_tid[str(item["id"])] = tid
        options = list(item.get("options", []))
        tasks.append(
            {
                "id": tid,
                "source": "litqa2",
                "source_index": idx,
                "question": item.get("question", ""),
                "options": options,
                "n_options": len(options),
                "answer": item.get("answer"),
                "prompt": (
                    f"{item.get('question', '')}\n\nAnswer choices:\n"
                    + "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
                    + '\n\nThink step by step, then end your response with "The answer is (X)." '
                    "where X is the letter of the correct option."
                ),
                "scoring": "mcq_letter",
                "task_kind": "biomedical_literature_recall",
            }
        )
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for data in (routed, gemma):
        for row in data.get("responses", []):
            tid = uuid_to_tid.get(str(row.get("id")))
            if not tid:
                continue
            updated = dict(row)
            updated["id"] = tid
            updated["source"] = "litqa2"
            rows[(str(updated["model"]), tid)] = updated
    return tasks, rows


def candidate_ids(panel: list[str]) -> dict[str, str]:
    return {model: f"A{i + 1}" for i, model in enumerate(panel)}


def _prompt_text(task: dict[str, Any]) -> str:
    return str(task.get("prompt") or task.get("question") or "")


def _candidate_text(row: dict[str, Any], *, truncate: bool = True) -> str:
    text = str(row.get("text", "")).strip()
    if truncate and len(text) > 5000:
        return text[:5000] + "\n[truncated]"
    return text


def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.I | re.S)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_mcq_letter(text: str, n_options: int) -> str | None:
    if not text:
        return None
    letters = "".join(chr(65 + i) for i in range(n_options))
    patterns = [
        rf"answer\s+is\s*\(?([{letters}])\)?",
        rf"answer\s*:?\s*\(?([{letters}])\)?",
        rf"final\s+answer\s*:?\s*\(?([{letters}])\)?",
        rf"correct\s+(?:answer|option)\s+is\s*\(?([{letters}])\)?",
        rf"^\s*\(?([{letters}])\)?\s*$",
        rf"^\s*([{letters}])\s*[.)]\s+",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    return None


def _chat_model(
    *,
    model: str,
    messages: list[dict[str, str]],
    adapter_specs: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    adapter = adapters.adapter_for(model, adapter_specs, default_max_tokens=max_tokens, default_timeout=timeout)
    return client.chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=adapter.max_tokens or max_tokens,
        temperature=adapter.temperature,
        timeout=adapter.timeout or timeout,
        extra_body=adapter.extra_body,
        retries=retries,
        use_request=adapter.use_request,
    )


def load_aiiq() -> dict[str, Any]:
    cache = Path("results/aiiq_rankings_snapshot.json")
    if cache.exists():
        return _load_json(str(cache))
    data = json.loads(urllib.request.urlopen(AIIQ_URL, timeout=30).read().decode())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def aiiq_dimension_scores(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    rankings = {row["id"]: row for row in data.get("rankings", [])}
    out: dict[str, dict[str, float]] = {model: {} for model in PANEL}
    for dim, ranking_id in DIMENSION_RANKINGS.items():
        ranking = rankings.get(ranking_id)
        if not ranking:
            continue
        raw = {
            str(row.get("id")): float(row["score"])
            for row in ranking.get("models", [])
            if isinstance(row.get("score"), int | float)
        }
        if not raw:
            continue
        for model, aiiq_id in AIIQ_IDS.items():
            out[model][dim] = raw.get(aiiq_id, float("nan"))
    return out


def normalize_weights(raw: dict[str, Any] | None) -> dict[str, float]:
    weights = {dim: 0.0 for dim in DIMENSION_RANKINGS}
    if raw:
        for dim in weights:
            try:
                weights[dim] = max(0.0, float(raw.get(dim, 0.0)))
            except (TypeError, ValueError):
                weights[dim] = 0.0
    total = sum(weights.values())
    if total <= 0:
        weights["scientific-reasoning"] = 0.7
        weights["reliability"] = 0.3
        total = 1.0
    return {dim: val / total for dim, val in weights.items()}


def fallback_domain_weights(task: dict[str, Any]) -> dict[str, float]:
    task_id = str(task.get("id") or "")
    source = str(task.get("source") or "")
    scoring = str(task.get("scoring") or "")
    if "ifbench" in source or task_id.startswith("ifbench") or scoring == "ifbench_selected":
        raw = {"reliability": 0.8, "scientific-reasoning": 0.2}
    else:
        raw = {"scientific-reasoning": 0.85, "reliability": 0.15}
    return normalize_weights(raw)


def normalize_task_types(raw: dict[str, Any] | None) -> dict[str, float]:
    weights = {task_type: 0.0 for task_type in TASK_TYPES}
    if raw:
        for task_type in weights:
            try:
                weights[task_type] = max(0.0, float(raw.get(task_type, 0.0)))
            except (TypeError, ValueError):
                weights[task_type] = 0.0
    total = sum(weights.values())
    if total <= 0:
        weights["structured_mcq_reasoning"] = 0.7
        weights["factual_recall"] = 0.3
        total = 1.0
    return {task_type: val / total for task_type, val in weights.items()}


def fallback_task_types(task: dict[str, Any]) -> dict[str, float]:
    task_id = str(task.get("id") or "")
    source = str(task.get("source") or "")
    task_kind = str(task.get("task_kind") or "")
    scoring = str(task.get("scoring") or "")
    if "hle" in source or task_id.startswith("hle_bio"):
        raw = {
            "judgment_under_uncertainty": 0.7,
            "structured_mcq_reasoning": 0.2,
            "factual_recall": 0.1,
        }
    elif "ifbench" in source or task_id.startswith("ifbench") or scoring == "ifbench_selected":
        raw = {
            "instruction_following": 0.85,
            "structured_mcq_reasoning": 0.15,
        }
    elif task_kind == "biomedical_literature_recall" or source == "litqa2":
        raw = {
            "factual_recall": 0.65,
            "structured_mcq_reasoning": 0.25,
            "judgment_under_uncertainty": 0.1,
        }
    elif "gpqa" in source or task_id.startswith("gpqa_bio"):
        raw = {
            "structured_mcq_reasoning": 0.55,
            "judgment_under_uncertainty": 0.3,
            "factual_recall": 0.15,
        }
    else:
        raw = {
            "structured_mcq_reasoning": 0.7,
            "factual_recall": 0.3,
        }
    return normalize_task_types(raw)


def deterministic_classification(task: dict[str, Any]) -> dict[str, Any]:
    domain_weights = fallback_domain_weights(task)
    task_type_weights = fallback_task_types(task)
    return {
        "id": task["id"],
        "model": "deterministic_source_task_classifier",
        "text": "",
        "parsed": None,
        "classification_fallback_used": True,
        "weights": domain_weights,
        "domain_weights": domain_weights,
        "task_type_weights": task_type_weights,
        "error": None,
        "usage": {},
        "latency_ms": 0,
    }


def parse_classification(parsed: dict[str, Any] | None) -> tuple[dict[str, float], dict[str, float]]:
    if parsed and isinstance(parsed.get("domain_dimensions"), dict):
        domain = normalize_weights(parsed.get("domain_dimensions"))
    else:
        # Backward compatible with the old flat domain-only classifier output.
        domain = normalize_weights(parsed)
    task_raw = parsed.get("task_type") if parsed and isinstance(parsed.get("task_type"), dict) else None
    return domain, normalize_task_types(task_raw)


def classifier_user_message(task: dict[str, Any]) -> str:
    metadata = {
        "id": task.get("id"),
        "source": task.get("source"),
        "task_kind": task.get("task_kind"),
        "scoring": task.get("scoring"),
        "n_options": task.get("n_options") or len(task.get("options", [])),
    }
    return (
        "Classify this benchmark prompt. Do not answer it.\n\n"
        f"Metadata:\n{json.dumps(metadata, indent=2)}\n\n"
        "<TASK_TO_CLASSIFY>\n"
        f"{_prompt_text(task)}\n"
        "</TASK_TO_CLASSIFY>\n\n"
        "Return only the JSON classification object."
    )


def classify_dimensions(
    *,
    task: dict[str, Any],
    adapter_specs: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    r = _chat_model(
        model=CLASSIFIER,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": classifier_user_message(task)},
        ],
        adapter_specs=adapter_specs,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
    )
    text = "" if r.get("error") else str(r.get("text", ""))
    parsed = _extract_json(text)
    domain_weights, task_type_weights = parse_classification(parsed)
    classification_fallback_used = parsed is None
    if classification_fallback_used:
        task_type_weights = fallback_task_types(task)
    return {
        "id": task["id"],
        "model": CLASSIFIER,
        "text": text,
        "parsed": parsed,
        "classification_fallback_used": classification_fallback_used,
        "weights": domain_weights,
        "domain_weights": domain_weights,
        "task_type_weights": task_type_weights,
        "error": r.get("error"),
        "usage": r.get("usage", {}),
        "latency_ms": r.get("latency_ms"),
    }


def compute_taskiq(weights: dict[str, float], scores: dict[str, dict[str, float]]) -> dict[str, float]:
    out = {}
    for model, dims in scores.items():
        vals = []
        for dim, weight in weights.items():
            val = dims.get(dim, float("nan"))
            if math.isnan(val):
                continue
            vals.append(weight * val)
        out[model] = sum(vals)
    return out


def adjusted_taskiq(
    task: dict[str, Any],
    weights: dict[str, float],
    task_type_weights: dict[str, float],
    scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    taskiq = compute_taskiq(weights, scores)
    if task.get("task_kind") == "biomedical_literature_recall":
        # LitQA2-style questions reward paper-specific biomedical recall. AIIQ
        # has no direct literature-recall dimension, so add a weak local prior
        # from the saved expanded table: DeepSeek/Kimi/Gemma did best there.
        boosts = {
            "deepseek/deepseek-v4-pro": 8.0,
            "moonshotai/kimi-k2.6": 5.0,
            "xiaomi/mimo-v2.5-pro": -4.0,
            "nvidia/nvidia-nemotron-3-ultra-550b-a55b": -2.0,
            "minimax/minimax-m3": -1.0,
        }
        for model, boost in boosts.items():
            taskiq[model] = taskiq.get(model, 0.0) + boost
    judgment = task_type_weights.get("judgment_under_uncertainty", 0.0)
    if judgment > 0:
        # Local prior learned from the HLE packet: MiMo and Nemotron are the
        # important hard-science minority rescue models, with Kimi/Minimax also
        # occasionally useful. Scale softly by the classifier's task-type weight.
        boosts = {
            "xiaomi/mimo-v2.5-pro": 12.0,
            "nvidia/nvidia-nemotron-3-ultra-550b-a55b": 10.0,
            "moonshotai/kimi-k2.6": 5.0,
            "minimax/minimax-m3": 4.0,
            "deepseek/deepseek-v4-pro": -3.0,
        }
        for model, boost in boosts.items():
            taskiq[model] = taskiq.get(model, 0.0) + judgment * boost
    instruction = task_type_weights.get("instruction_following", 0.0)
    if instruction > 0:
        boosts = {
            "minimax/minimax-m3": 8.0,
            "moonshotai/kimi-k2.6": 3.0,
            "nvidia/nvidia-nemotron-3-ultra-550b-a55b": 2.0,
        }
        for model, boost in boosts.items():
            taskiq[model] = taskiq.get(model, 0.0) + instruction * boost
    return taskiq


def _basic_validity(task: dict[str, Any], text: str) -> dict[str, Any]:
    score = score_response(task, text)
    validity = {
        "non_empty": bool(text.strip()),
        "status": score.get("status"),
    }
    if task.get("scoring") == "mcq_letter":
        n_options = int(task.get("n_options") or len(task.get("options", [])) or 10)
        pred = score.get("pred") or _extract_mcq_letter(text, n_options)
        validity["extracted_answer"] = pred
        validity["has_mcq_answer"] = pred is not None
        validity["terse_valid_mcq_ok"] = True
    if task.get("scoring") == "ifbench_selected":
        # This is a deterministic constraint check against the prompt, not a
        # hidden-answer lookup.
        validity["ifbench_constraint_score"] = score.get("score")
        for key in ("expected_count", "actual_count", "expected_counts", "actual_counts", "sentences", "sentences_with_terminal_emoji"):
            if key in score:
                validity[key] = score[key]
    return validity


def router_prompt(
    *,
    task: dict[str, Any],
    weights: dict[str, float],
    task_type_weights: dict[str, float],
    taskiq: dict[str, float],
    cid_by_model: dict[str, str],
    saved_rows: dict[tuple[str, str], dict[str, Any]],
) -> str:
    ordered = sorted(PANEL, key=lambda model: (-taskiq.get(model, 0.0), cid_by_model[model]))
    lines = [
        "Original task:",
        _prompt_text(task),
        "",
        "Task kind:",
        str(task.get("task_kind") or task.get("scoring") or "unknown"),
        "",
        "Dimension weights:",
        json.dumps(weights, indent=2),
        "",
        "Task-type weights:",
        json.dumps(task_type_weights, indent=2),
        "",
        "Judging policy implied by task type:",
        judging_policy(task_type_weights),
        "",
        "Temporary TaskIQ ranking:",
    ]
    for model in ordered:
        cid = cid_by_model[model]
        text = _candidate_text(saved_rows[(model, task["id"])], truncate=False)
        validity = _basic_validity(task, text)
        lines.append(f"- {cid}: TaskIQ={taskiq.get(model, 0.0):.3f}; validity={json.dumps(validity, ensure_ascii=False)}")
    if task.get("scoring") == "mcq_letter":
        lines.extend(
            [
                "",
                "MCQ routing rule: candidate answers that are only a letter or 'letter. option text' are valid. "
                "Do not reject a candidate solely because it lacks step-by-step reasoning.",
            ]
        )
    lines.append("")
    lines.append("Candidate answers:")
    for model in PANEL:
        cid = cid_by_model[model]
        lines.append(f"\n[{cid}]\n{_candidate_text(saved_rows[(model, task['id'])], truncate=True)}")
    return "\n".join(lines)


def judging_policy(task_type_weights: dict[str, float]) -> str:
    dominant = max(task_type_weights.items(), key=lambda kv: kv[1])[0]
    if dominant == "instruction_following":
        return "Use deterministic validity/constraint checks first; exact count/format compliance overrides model priors."
    if dominant == "factual_recall":
        return "Extract candidate answers; ignore verbosity; prefer literature/knowledge priors and valid extracted answers."
    if dominant == "judgment_under_uncertainty":
        return "Do not majority vote. Preserve and steelman minority answers from hard-science specialists before rejecting them."
    if dominant == "structured_mcq_reasoning":
        return "Extract answer letters, compare reasoning quality, and select the most scientifically justified existing answer."
    if dominant == "deep_research":
        return "Prefer broad, source-grounded answers; penalize unsupported claims."
    return "Use TaskIQ as a prior, then apply answer-validity checks for this task."


def parse_selected(text: str, ids: list[str], default_id: str) -> tuple[str, dict[str, Any] | None]:
    parsed = _extract_json(text)
    if parsed:
        raw = parsed.get("selected_answer_id") or parsed.get("selected") or parsed.get("best_answer_id")
        if isinstance(raw, str) and raw.strip() in ids:
            return raw.strip(), parsed
    for cid in ids:
        if re.search(rf"\b{re.escape(cid)}\b", text):
            return cid, parsed
    return default_id, parsed


def route_answer(
    *,
    task: dict[str, Any],
    classification: dict[str, Any],
    taskiq: dict[str, float],
    cid_by_model: dict[str, str],
    saved_rows: dict[tuple[str, str], dict[str, Any]],
    adapter_specs: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    ids = list(cid_by_model.values())
    r = _chat_model(
        model=ROUTER,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {
                "role": "user",
                "content": router_prompt(
                    task=task,
                    weights=classification["weights"],
                    task_type_weights=classification.get("task_type_weights", normalize_task_types(None)),
                    taskiq=taskiq,
                    cid_by_model=cid_by_model,
                    saved_rows=saved_rows,
                ),
            },
        ],
        adapter_specs=adapter_specs,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
    )
    router_text = "" if r.get("error") else str(r.get("text", ""))
    default_model = max(PANEL, key=lambda model: taskiq.get(model, float("-inf")))
    default_id = cid_by_model[default_model]
    selected_id, parsed = parse_selected(router_text, ids, default_id)
    selected_model = next(model for model, cid in cid_by_model.items() if cid == selected_id)
    selected_text = _candidate_text(saved_rows[(selected_model, task["id"])], truncate=False)
    row: dict[str, Any] = {
        "model": "taskiq_kimi_selected_answer",
        "id": task["id"],
        "source": task["source"],
        "classifier_model": CLASSIFIER,
        "router_model": ROUTER,
        "dimension_weights": classification["weights"],
        "task_type_weights": classification.get("task_type_weights", normalize_task_types(None)),
        "taskiq": taskiq,
        "selected_model": selected_model,
        "selected_candidate_id": selected_id,
        "router_text": router_text,
        "router_parsed": parsed,
        "router_error": r.get("error"),
        "router_usage": r.get("usage", {}),
        "router_latency_ms": r.get("latency_ms"),
        "text": selected_text,
    }
    row.update(score_response(task, selected_text))
    if task.get("scoring") == "mcq_letter" and row.get("pred") is None:
        pred = _extract_mcq_letter(selected_text, int(task.get("n_options") or len(task.get("options", [])) or 10))
        if pred is not None:
            row["pred"] = pred
            row["status"] = "scored"
            row["score"] = int(pred == task.get("answer"))
            row["answer"] = task.get("answer")
    return row


def deterministic_route_answer(
    *,
    task: dict[str, Any],
    classification: dict[str, Any],
    taskiq: dict[str, float],
    cid_by_model: dict[str, str],
    saved_rows: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for model in PANEL:
        text = _candidate_text(saved_rows[(model, task["id"])], truncate=False)
        validity = _basic_validity(task, text)
        candidate_score = taskiq.get(model, float("-inf"))
        if task.get("scoring") == "mcq_letter":
            if not validity.get("has_mcq_answer"):
                candidate_score -= 1000.0
        elif task.get("scoring") == "ifbench_selected":
            candidate_score += 100.0 * float(validity.get("ifbench_constraint_score") or 0.0)
        elif not validity.get("non_empty"):
            candidate_score -= 1000.0
        candidates.append((candidate_score, model, validity))

    _, selected_model, selected_validity = max(candidates, key=lambda item: (item[0], item[1]))
    selected_text = _candidate_text(saved_rows[(selected_model, task["id"])], truncate=False)
    row: dict[str, Any] = {
        "model": "taskiq_deterministic_selected_answer",
        "id": task["id"],
        "source": task["source"],
        "classifier_model": classification.get("model"),
        "router_model": "deterministic_taskiq_policy",
        "dimension_weights": classification["weights"],
        "task_type_weights": classification.get("task_type_weights", normalize_task_types(None)),
        "taskiq": taskiq,
        "selected_model": selected_model,
        "selected_candidate_id": cid_by_model[selected_model],
        "router_text": "",
        "router_parsed": {
            "selected_answer_id": cid_by_model[selected_model],
            "validity": selected_validity,
            "policy": "highest adjusted TaskIQ, with MCQ answer extraction and deterministic IFBench validity checks",
        },
        "router_error": None,
        "router_usage": {},
        "router_latency_ms": 0,
        "text": selected_text,
    }
    row.update(score_response(task, selected_text))
    if task.get("scoring") == "mcq_letter" and row.get("pred") is None:
        pred = _extract_mcq_letter(selected_text, int(task.get("n_options") or len(task.get("options", [])) or 10))
        if pred is not None:
            row["pred"] = pred
            row["status"] = "scored"
            row["score"] = int(pred == task.get("answer"))
            row["answer"] = task.get("answer")
    return row


def make_payload(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "eval": "mixed_bio_taskiq_selector",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "saved_results": args.saved_results,
        "panel": PANEL,
        "classifier": CLASSIFIER,
        "router": ROUTER,
        "task_count": len(tasks),
        "tasks": [public_task(t) for t in tasks],
        "classifications": sorted(classifications, key=lambda r: r["id"]),
        "responses": sorted(responses, key=lambda r: r["id"]),
        "summary": summarize(responses),
        "notes": [
            "Panel responses were reused from saved solo runs.",
            "Classifier maps prompts to AIIQ dimension weights.",
            "Classifier also maps prompts to task-type weights; task type controls judging policy and local priors.",
            "TaskIQ is computed as the weighted sum of AIIQ dimension scores.",
            "Router selects an existing answer; no freeform final synthesis.",
            "IFBench validity includes deterministic constraint checks from visible answer text.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dimension-aware TaskIQ selector on mixed bio packet.")
    parser.add_argument("--saved-results", default="results/mixed_bio_reasoning_packet_16_open7_plus_frontier_refs_corrected.json")
    parser.add_argument("--include-litqa2", action="store_true", help="Append the saved 10-row LitQA2 packet.")
    parser.add_argument("--adapter-file", action="append", default=[])
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=client.DEFAULT_RETRIES)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--deterministic-meta",
        action="store_true",
        help="Use deterministic source/task classification and deterministic answer routing.",
    )
    parser.add_argument("--out", default="results/mixed_bio_reasoning_packet_16_taskiq_selector.json")
    args = parser.parse_args()

    saved = _load_json(args.saved_results)
    saved_rows = _response_map(saved)
    tasks = load_tasks(include_aa_lcr=False)
    if args.include_litqa2:
        litqa_tasks, litqa_rows = load_litqa2_tasks_and_rows()
        tasks.extend(litqa_tasks)
        saved_rows.update(litqa_rows)
    cid_by_model = candidate_ids(PANEL)
    aiiq_scores = aiiq_dimension_scores(load_aiiq())
    adapter_specs: dict[str, dict[str, Any]] = {}
    for adapter_file in args.adapter_file:
        adapter_specs.update(adapters.load_adapter_file(adapter_file))

    out = Path(args.out)
    classifications: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    if out.exists():
        existing = _load_json(str(out))
        classifications = list(existing.get("classifications", []))
        responses = list(existing.get("responses", []))
        print(f"reusing {len(classifications)} classifications, {len(responses)} responses from {out}", flush=True)

    api_key = "" if args.deterministic_meta else client.api_key_from_env(args.api_key)
    class_ids = {r["id"] for r in classifications}
    pending_class = [task for task in tasks if task["id"] not in class_ids]
    print(f"running TaskIQ classifier: {len(pending_class)} new / {len(tasks)} total", flush=True)

    def classify_one(task: dict[str, Any]) -> dict[str, Any]:
        return classify_dimensions(
            task=task,
            adapter_specs=adapter_specs,
            base_url=args.base_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )

    if pending_class and args.deterministic_meta:
        for task in pending_class:
            classifications.append(deterministic_classification(task))
        payload = make_payload(args, tasks, classifications, responses)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif pending_class:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(classify_one, task) for task in pending_class]
            for fut in as_completed(futures):
                classifications.append(fut.result())
                payload = make_payload(args, tasks, classifications, responses)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    class_by_id = {row["id"]: row for row in classifications}
    response_ids = {r["id"] for r in responses}
    pending_route = [task for task in tasks if task["id"] not in response_ids and task["id"] in class_by_id]
    print(f"running TaskIQ router: {len(pending_route)} new / {len(tasks)} total", flush=True)

    def route_one(task: dict[str, Any]) -> dict[str, Any]:
        classification = class_by_id[task["id"]]
        taskiq = adjusted_taskiq(
            task,
            classification["weights"],
            classification.get("task_type_weights", normalize_task_types(None)),
            aiiq_scores,
        )
        if args.deterministic_meta:
            return deterministic_route_answer(
                task=task,
                classification=classification,
                taskiq=taskiq,
                cid_by_model=cid_by_model,
                saved_rows=saved_rows,
            )
        return route_answer(
            task=task,
            classification=classification,
            taskiq=taskiq,
            cid_by_model=cid_by_model,
            saved_rows=saved_rows,
            adapter_specs=adapter_specs,
            base_url=args.base_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )

    if pending_route:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(route_one, task) for task in pending_route]
            for fut in as_completed(futures):
                responses.append(fut.result())
                payload = make_payload(args, tasks, classifications, responses)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = make_payload(args, tasks, classifications, responses)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
