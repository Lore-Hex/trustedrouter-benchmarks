"""Evaluate panel -> judge -> synthesizer variants on the mixed bio packet.

This reuses saved solo model responses as panel evidence, then runs only the
judge and synthesizer calls. Outputs are scored with the existing mixed packet
deterministic scorers.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trbench import adapters, client
from trbench.evals.mixed_bio_packet.run import load_tasks, public_task, score_response, summarize


FRONTIER_PAIR = {
    "name": "frontier_pair_gpt55_judge_opus48_synth",
    "panel": ["anthropic/claude-opus-4.8", "openai/gpt-5.5"],
    "judge": "openai/gpt-5.5",
    "synth": "anthropic/claude-opus-4.8",
}

OPEN_PARETO = {
    "name": "open_pareto_kimi_judge_glm_synth",
    "panel": [
        "moonshotai/kimi-k2.6",
        "xiaomi/mimo-v2.5-pro",
        "nvidia/nvidia-nemotron-3-ultra-550b-a55b",
        "deepseek/deepseek-v4-pro",
        "minimax/minimax-m3",
    ],
    "judge": "moonshotai/kimi-k2.6",
    "synth": "z-ai/glm-5.2",
}

OPEN_PARETO_CAPABILITY_WEIGHTED = {
    "name": "open_pareto_capability_weighted_kimi_judge_glm_synth",
    "panel": [
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "xiaomi/mimo-v2.5-pro",
        "nvidia/nvidia-nemotron-3-ultra-550b-a55b",
        "minimax/minimax-m3",
    ],
    "judge": "moonshotai/kimi-k2.6",
    "synth": "z-ai/glm-5.2",
    "capability_policy": {
        "deepseek/deepseek-v4-pro": "Strong open-model biomedical/literature reasoning coverage in the saved table.",
        "moonshotai/kimi-k2.6": "Strong minority-rescue behavior and useful instruction-reasoning signal.",
        "xiaomi/mimo-v2.5-pro": "Strongest open-model HLE Bio rescue signal in the saved table.",
        "nvidia/nvidia-nemotron-3-ultra-550b-a55b": "Good GPQA coverage and complementary HLE rescue signal.",
        "minimax/minimax-m3": "High prior weight for instruction following from broader IFBench/IFEval evidence.",
    },
}

OPEN_PARETO_MINORITY_VERIFY = {
    **OPEN_PARETO_CAPABILITY_WEIGHTED,
    "name": "open_pareto_minority_verify_kimi_judge_glm_synth",
    "judge_system": (
        "You are the judge in a model-fusion evaluation. Your main job is to prevent loss of "
        "minority-correct answers. Compare panel responses and return compact JSON only. Required keys: "
        "task_type, consensus_answer, all_distinct_candidate_answers, minority_candidates, "
        "capability_weighting, verification_plan, likely_correct_answer, constraint_risks, final_guidance. "
        "For every distinct answer proposed by only one model, include it in minority_candidates with: "
        "model, answer, strongest_supporting_evidence, strongest_objection, relevant_capability_note, "
        "and whether the synthesizer should independently verify it. Do not use simple majority vote. "
        "For instruction-following tasks, identify exact constraints/counts/formats and which candidate "
        "appears to satisfy each constraint. Do not write the final user-facing answer."
    ),
    "synth_system": (
        "You are the synthesizer in a model-fusion evaluation. Before answering, independently verify every "
        "candidate named in judge.minority_candidates against the original task. Do not simply follow the "
        "majority. If a minority candidate comes from a model with a relevant capability note and survives "
        "verification, choose it even against a majority. For instruction-following tasks, satisfy the exact "
        "requested counts/formats/keywords; the final answer itself must pass the constraints, not merely be "
        "substantively good. Do not mention the panel, judge, fusion process, or model names. For multiple-choice "
        "tasks, end exactly with 'The answer is (X).' where X is the letter."
    ),
}

VARIANTS = {
    v["name"]: v
    for v in (FRONTIER_PAIR, OPEN_PARETO, OPEN_PARETO_CAPABILITY_WEIGHTED, OPEN_PARETO_MINORITY_VERIFY)
}

JUDGE_SYSTEM = (
    "You are the judge in a model-fusion evaluation. Compare panel responses and return compact JSON "
    "with keys consensus, contradictions, unique_insights, blind_spots, likely_correct_answer, "
    "constraint_risks, capability_weighting, minority_candidates, and final_guidance. Do not write the "
    "final user-facing answer. Preserve minority answers when they are better supported than the majority. "
    "Use any supplied capability notes to weight evidence by task type instead of using simple majority vote. "
    "Return JSON only."
)

SYNTH_SYSTEM = (
    "You are the synthesizer in a model-fusion evaluation. Use the original task, the panel answers, "
    "and the judge JSON to produce the final answer. Do not mention the panel, judge, fusion process, "
    "or model names. Do not simply follow majority vote: prefer the answer supported by the models with "
    "the strongest demonstrated capability for this task type, especially when the judge identifies a "
    "well-supported minority candidate. For multiple-choice tasks, end exactly with 'The answer is (X).' "
    "where X is the letter. For instruction-following tasks, satisfy the original instruction exactly."
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _response_map(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("model")), str(r.get("id"))): r for r in data.get("responses", [])}


def _model_short(model: str) -> str:
    return model.split("/")[-1]


def _panel_evidence(task: dict[str, Any], panel_rows: list[dict[str, Any]], variant: dict[str, Any]) -> str:
    panel = variant["panel"]
    lines = [
        "Original task:",
        task["prompt"],
        "",
        "Scoring type: " + str(task.get("scoring")),
        "",
        "Capability notes for this panel:",
    ]
    capability_policy = variant.get("capability_policy") or {}
    if capability_policy:
        for model in panel:
            note = capability_policy.get(model)
            if note:
                lines.append(f"- {model}: {note}")
    else:
        lines.append("- No explicit capability notes supplied; infer from response quality and task type.")
    lines.extend([
        "",
        "Panel answers:",
    ])
    by_model = {r["model"]: r for r in panel_rows}
    for idx, model in enumerate(panel, 1):
        row = by_model.get(model, {})
        status = row.get("status", "missing")
        pred = row.get("pred")
        text = str(row.get("text", "")).strip()
        if len(text) > 5000:
            text = text[:5000] + "\n[truncated]"
        meta = f"status={status}"
        if pred is not None:
            meta += f", extracted_answer={pred}"
        lines.append(f"\n[{idx}] model={model} ({meta})\n{text}")
    return "\n".join(lines)


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


def run_variant(
    *,
    variant: dict[str, Any],
    tasks: list[dict[str, Any]],
    saved_rows: dict[tuple[str, str], dict[str, Any]],
    adapter_specs: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    retries: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    def one(task: dict[str, Any]) -> dict[str, Any]:
        panel_rows = [saved_rows[(model, task["id"])] for model in variant["panel"]]
        evidence = _panel_evidence(task, panel_rows, variant)
        judge = _chat_model(
            model=variant["judge"],
            messages=[
                {"role": "system", "content": variant.get("judge_system", JUDGE_SYSTEM)},
                {"role": "user", "content": evidence},
            ],
            adapter_specs=adapter_specs,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
        )
        judge_text = "" if judge.get("error") else str(judge.get("text", ""))
        synth_prompt = (
            evidence
            + "\n\nJudge analysis JSON:\n"
            + judge_text
            + "\n\nNow write the final answer to the original task."
        )
        synth = _chat_model(
            model=variant["synth"],
            messages=[
                {"role": "system", "content": variant.get("synth_system", SYNTH_SYSTEM)},
                {"role": "user", "content": synth_prompt},
            ],
            adapter_specs=adapter_specs,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
        )
        row: dict[str, Any] = {
            "model": variant["name"],
            "id": task["id"],
            "source": task["source"],
            "panel": variant["panel"],
            "judge_model": variant["judge"],
            "synth_model": variant["synth"],
            "judge_text": judge_text,
            "judge_usage": judge.get("usage", {}),
            "judge_error": judge.get("error"),
            "judge_latency_ms": judge.get("latency_ms"),
            "text": "" if synth.get("error") else str(synth.get("text", "")),
            "usage": synth.get("usage", {}),
            "error": synth.get("error"),
            "latency_ms": synth.get("latency_ms"),
            "finish_reason": synth.get("finish_reason"),
        }
        if synth.get("empty_reason"):
            row["empty_reason"] = synth["empty_reason"]
        if not row.get("error"):
            row.update(score_response(task, row["text"]))
        return row

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for fut in as_completed(futures):
            out.append(fut.result())
    return out


def _result_payload(
    *,
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "eval": "mixed_bio_reasoning_packet_fusion",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "saved_results": args.saved_results,
        "variants": selected,
        "task_count": len(tasks),
        "tasks": [public_task(t) for t in tasks],
        "responses": sorted(responses, key=lambda r: (r["model"], r["id"])),
        "summary": summarize(responses),
        "notes": [
            "Panel responses were reused from saved solo runs.",
            "Judge and synthesizer calls were run live.",
            "Scores use the existing mixed bio packet deterministic scorer.",
            "Existing rows in this output file are reused on rerun.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mixed bio packet fusion variants from saved panel responses.")
    parser.add_argument("--saved-results", default="results/mixed_bio_reasoning_packet_16_open7_plus_frontier_refs_corrected.json")
    parser.add_argument("--adapter-file", action="append", default=[])
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), default=[])
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=client.DEFAULT_RETRIES)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", default="results/mixed_bio_reasoning_packet_16_fusion_variants.json")
    args = parser.parse_args()

    saved = _load_json(args.saved_results)
    saved_rows = _response_map(saved)
    tasks = load_tasks(include_aa_lcr=False)
    selected = [VARIANTS[name] for name in (args.variant or sorted(VARIANTS))]

    missing = []
    for variant in selected:
        for task in tasks:
            for model in variant["panel"]:
                if (model, task["id"]) not in saved_rows:
                    missing.append((variant["name"], model, task["id"]))
    if missing:
        details = "\n".join(f"{v}: {m} {tid}" for v, m, tid in missing[:20])
        raise SystemExit(f"Missing saved panel responses:\n{details}")

    adapter_specs: dict[str, dict[str, Any]] = {}
    for adapter_file in args.adapter_file:
        adapter_specs.update(adapters.load_adapter_file(adapter_file))

    out = Path(args.out)
    responses: list[dict[str, Any]] = []
    if out.exists():
        existing = _load_json(str(out))
        responses = list(existing.get("responses", []))
        print(f"reusing {len(responses)} existing fusion rows from {out}", flush=True)

    api_key = client.api_key_from_env(args.api_key)
    for variant in selected:
        existing_keys = {(str(r.get("model")), str(r.get("id"))) for r in responses}
        pending_tasks = [task for task in tasks if (variant["name"], task["id"]) not in existing_keys]
        if not pending_tasks:
            print(f"reusing fusion variant {variant['name']} ({len(tasks)} tasks)", flush=True)
            continue
        print(
            "running fusion variant "
            f"{variant['name']}: panel={[ _model_short(m) for m in variant['panel'] ]}, "
            f"judge={_model_short(variant['judge'])}, synth={_model_short(variant['synth'])}, "
            f"new_tasks={len(pending_tasks)}",
            flush=True,
        )
        responses.extend(
            run_variant(
                variant=variant,
                tasks=pending_tasks,
                saved_rows=saved_rows,
                adapter_specs=adapter_specs,
                base_url=args.base_url,
                api_key=api_key,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                concurrency=args.concurrency,
            )
        )
        result = _result_payload(args=args, selected=selected, tasks=tasks, responses=responses)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = _result_payload(args=args, selected=selected, tasks=tasks, responses=responses)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
