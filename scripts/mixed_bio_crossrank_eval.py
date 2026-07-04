"""Cross-rank OS panel answers on the mixed bio packet.

Pipeline:
1. Reuse saved panel answers.
2. Ask each panel model to rank anonymized candidate answers.
3. Aggregate rankings with deterministic Borda count.
4. Score both the selected original answer and a constrained GLM synthesized answer.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
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
SYNTH = "z-ai/glm-5.2"
AIIQ_IDS = {
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
    "moonshotai/kimi-k2.6": "kimi-k2.6",
    "xiaomi/mimo-v2.5-pro": "mimo-v2.5-pro",
    "nvidia/nvidia-nemotron-3-ultra-550b-a55b": "nvidia-nemotron-3-ultra",
    "minimax/minimax-m3": "minimax-m3",
}

CRITIC_SYSTEM = (
    "You are a critic in a model-panel answer selection system. You will see the original task and several "
    "anonymized candidate answers. Rank the candidates by which answer is most likely correct for the original "
    "task. Do not favor style over correctness. For multiple-choice tasks, focus on the final answer and reasoning. "
    "For instruction-following tasks, explicitly check exact constraints such as counts, keywords, JSON, emoji, "
    "sentence format, and requested structure. Return JSON only with keys: ranked_answer_ids, best_answer_id, "
    "constraint_checks, reasons, confidence. ranked_answer_ids must be a list containing every candidate id exactly once."
)

SYNTH_SYSTEM = (
    "You are the finalizer in a model-panel answer selection system. You will receive the original task, the "
    "selected candidate answer, and cross-rank voting notes. Preserve the selected candidate's substantive answer. "
    "For multiple-choice tasks, output a concise answer ending exactly with 'The answer is (X).' where X is the "
    "chosen letter. For instruction-following tasks, produce an answer that satisfies the original instruction exactly; "
    "copy the selected candidate where possible and only repair obvious formatting/count issues. Do not mention models, "
    "candidate ids, judges, rankings, or this process."
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _response_map(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("model")), str(r.get("id"))): r for r in data.get("responses", [])}


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


def candidate_ids(panel: list[str]) -> dict[str, str]:
    return {model: f"A{i + 1}" for i, model in enumerate(panel)}


def _candidate_text(row: dict[str, Any]) -> str:
    text = str(row.get("text", "")).strip()
    if len(text) > 5000:
        return text[:5000] + "\n[truncated]"
    return text


def critic_prompt(task: dict[str, Any], candidates: dict[str, str]) -> str:
    parts = [
        "Original task:",
        task["prompt"],
        "",
        "Candidate answers:",
    ]
    for cid, text in candidates.items():
        parts.append(f"\n[{cid}]\n{text}")
    return "\n".join(parts)


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
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


def parse_ranking(text: str, ids: list[str]) -> tuple[list[str], dict[str, Any] | None]:
    data = _extract_json(text)
    ranking: list[str] = []
    if data:
        raw = data.get("ranked_answer_ids")
        if isinstance(raw, list):
            ranking = [str(x).strip() for x in raw]
        elif isinstance(data.get("ranking"), list):
            for item in data["ranking"]:
                if isinstance(item, dict):
                    ranking.append(str(item.get("answer_id") or item.get("id") or "").strip())
                else:
                    ranking.append(str(item).strip())
        best = data.get("best_answer_id")
        if isinstance(best, str) and best.strip() in ids and best.strip() not in ranking:
            ranking.insert(0, best.strip())
    if not ranking:
        # Fallback: first occurrence order in visible text.
        positions = []
        for cid in ids:
            m = re.search(rf"\b{re.escape(cid)}\b", text)
            if m:
                positions.append((m.start(), cid))
        ranking = [cid for _, cid in sorted(positions)]
    seen = set()
    out = []
    for cid in ranking:
        if cid in ids and cid not in seen:
            out.append(cid)
            seen.add(cid)
    out.extend(cid for cid in ids if cid not in seen)
    return out, data


def aggregate_borda(rankings: list[dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    scores = {cid: 0.0 for cid in ids}
    n = len(ids)
    for row in rankings:
        ranking = row.get("ranking") or ids
        for pos, cid in enumerate(ranking):
            if cid in scores:
                scores[cid] += n - pos
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"scores": scores, "winner": ordered[0][0], "ordered": ordered}


def _load_aiiq() -> dict[str, Any] | None:
    path = Path("results/aiiq_rankings_snapshot.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _aiiq_scores() -> dict[str, dict[str, float]]:
    data = _load_aiiq()
    if not data:
        return {model: {} for model in PANEL}
    rankings = {row["id"]: row for row in data.get("rankings", [])}
    wanted = ["composite-iq", "scientific-reasoning-iq", "reliability-iq", "gpqa-diamond", "humanitys-last-exam", "ifbench"]
    out: dict[str, dict[str, float]] = {model: {} for model in PANEL}
    for rid in wanted:
        ranking = rankings.get(rid)
        if not ranking:
            continue
        raw = {
            str(m.get("id")): float(m["score"])
            for m in ranking.get("models", [])
            if isinstance(m.get("score"), int | float)
        }
        if not raw:
            continue
        lo, hi = min(raw.values()), max(raw.values())
        span = hi - lo or 1.0
        for model, aiiq_id in AIIQ_IDS.items():
            val = raw.get(aiiq_id)
            out[model][rid] = 0.5 if val is None else (val - lo) / span
    return out


def _task_prior_key(task: dict[str, Any]) -> str:
    tid = str(task["id"])
    if tid.startswith("gpqa"):
        return "gpqa-diamond"
    if tid.startswith("hle"):
        return "humanitys-last-exam"
    if tid.startswith("ifbench"):
        return "ifbench"
    return "scientific-reasoning-iq"


def _ifbench_constraint_score(task: dict[str, Any], text: str) -> int | None:
    if task.get("scoring") != "ifbench_selected":
        return None
    score = score_response(task, text)
    raw = score.get("score")
    return int(raw) if raw is not None else None


def aggregate_weighted(
    *,
    task: dict[str, Any],
    rankings: list[dict[str, Any]],
    ids: list[str],
    cid_to_model: dict[str, str],
    candidates: dict[str, str],
    aiiq: dict[str, dict[str, float]],
) -> dict[str, Any]:
    base = aggregate_borda(rankings, ids)
    scores = {cid: float(base["scores"][cid]) for cid in ids}
    details: dict[str, dict[str, float]] = {cid: {"borda": scores[cid]} for cid in ids}
    n = len(ids)
    prior_key = _task_prior_key(task)

    for cid in ids:
        model = cid_to_model[cid]
        prior = aiiq.get(model, {}).get(prior_key, 0.5)
        composite = aiiq.get(model, {}).get("composite-iq", 0.5)
        reliability = aiiq.get(model, {}).get("reliability-iq", 0.5)
        prior_bonus = 3.0 * (prior - 0.5) + 1.0 * (composite - 0.5) + 1.0 * (reliability - 0.5)
        scores[cid] += prior_bonus
        details[cid]["aiiq_prior_bonus"] = round(prior_bonus, 4)

        text = candidates[cid]
        if not text.strip():
            scores[cid] -= 4.0
            details[cid]["empty_penalty"] = -4.0

        if_score = _ifbench_constraint_score(task, text)
        if if_score is not None:
            # IFBench has deterministic visible constraints, so let the verifier dominate.
            constraint_bonus = 12.0 if if_score == 1 else -6.0
            scores[cid] += constraint_bonus
            details[cid]["ifbench_constraint_bonus"] = constraint_bonus

        own_model = cid_to_model[cid]
        own_rows = [r for r in rankings if r.get("critic_model") == own_model]
        if own_rows:
            own_rank = own_rows[0].get("ranking") or []
            if own_rank and own_rank[0] == cid:
                scores[cid] += 1.5
                details[cid]["self_top_bonus"] = 1.5
            elif cid in own_rank[:2]:
                scores[cid] += 0.75
                details[cid]["self_top2_bonus"] = 0.75

        # Minority-rescue protection: if not many critics like it but the origin
        # model does and it has a task-relevant prior, keep it competitive.
        top2_votes = sum(cid in (r.get("ranking") or [])[:2] for r in rankings)
        if top2_votes <= 2 and own_rows and cid in (own_rows[0].get("ranking") or [])[:2] and prior >= 0.45:
            scores[cid] += 1.0
            details[cid]["minority_rescue_bonus"] = 1.0
        details[cid]["top2_votes"] = float(top2_votes)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "base_borda": base,
        "scores": scores,
        "details": details,
        "winner": ordered[0][0],
        "ordered": ordered,
        "prior_key": prior_key,
    }


def run_critic(
    *,
    critic_model: str,
    task: dict[str, Any],
    candidates: dict[str, str],
    adapter_specs: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    ids = list(candidates)
    r = _chat_model(
        model=critic_model,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": critic_prompt(task, candidates)},
        ],
        adapter_specs=adapter_specs,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
    )
    text = "" if r.get("error") else str(r.get("text", ""))
    ranking, parsed = parse_ranking(text, ids)
    return {
        "critic_model": critic_model,
        "id": task["id"],
        "ranking": ranking,
        "parsed": parsed,
        "text": text,
        "error": r.get("error"),
        "usage": r.get("usage", {}),
        "latency_ms": r.get("latency_ms"),
    }


def synthesize(
    *,
    task: dict[str, Any],
    selected_id: str,
    selected_text: str,
    aggregate: dict[str, Any],
    rankings: list[dict[str, Any]],
    adapter_specs: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    compact_rankings = [
        {
            "critic": r["critic_model"],
            "ranking": r["ranking"],
            "best": r["ranking"][0] if r.get("ranking") else None,
        }
        for r in rankings
    ]
    prompt = (
        "Original task:\n"
        + task["prompt"]
        + "\n\nSelected candidate id: "
        + selected_id
        + "\n\nSelected candidate answer:\n"
        + selected_text
        + "\n\nBorda aggregate:\n"
        + json.dumps(aggregate, ensure_ascii=False)
        + "\n\nCritic rankings:\n"
        + json.dumps(compact_rankings, ensure_ascii=False)
        + "\n\nNow produce the final answer."
    )
    return _chat_model(
        model=SYNTH,
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        adapter_specs=adapter_specs,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
    )


def make_payload(
    *,
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    selected_responses: list[dict[str, Any]],
    weighted_selected_responses: list[dict[str, Any]],
    synth_responses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "eval": "mixed_bio_crossrank",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "saved_results": args.saved_results,
        "panel": PANEL,
        "critics": PANEL,
        "synth": SYNTH,
        "task_count": len(tasks),
        "tasks": [public_task(t) for t in tasks],
        "rankings": sorted(rankings, key=lambda r: (r["id"], r["critic_model"])),
        "selected_responses": sorted(selected_responses, key=lambda r: r["id"]),
        "selected_summary": summarize(selected_responses),
        "weighted_selected_responses": sorted(weighted_selected_responses, key=lambda r: r["id"]),
        "weighted_selected_summary": summarize(weighted_selected_responses),
        "synth_responses": sorted(synth_responses, key=lambda r: r["id"]),
        "synth_summary": summarize(synth_responses),
        "notes": [
            "Panel responses were reused from saved solo runs.",
            "Candidates were anonymized as A1..A5 for critics.",
            "No saved correctness labels are included in critic or synth prompts.",
            "Selected responses score the Borda-winning original panel answer directly.",
            "Weighted selected responses score a deterministic Borda + AIIQ/local-prior + IFBench-verifier aggregator.",
            "Synth responses score GLM finalization from the selected answer and cross-rank notes.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cross-rank panel selection on mixed bio packet.")
    parser.add_argument("--saved-results", default="results/mixed_bio_reasoning_packet_16_open7_plus_frontier_refs_corrected.json")
    parser.add_argument("--adapter-file", action="append", default=[])
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=client.DEFAULT_RETRIES)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", default="results/mixed_bio_reasoning_packet_16_crossrank_open_pareto.json")
    args = parser.parse_args()

    saved = _load_json(args.saved_results)
    saved_rows = _response_map(saved)
    tasks = load_tasks(include_aa_lcr=False)
    adapter_specs: dict[str, dict[str, Any]] = {}
    for adapter_file in args.adapter_file:
        adapter_specs.update(adapters.load_adapter_file(adapter_file))

    out = Path(args.out)
    rankings: list[dict[str, Any]] = []
    selected_responses: list[dict[str, Any]] = []
    weighted_selected_responses: list[dict[str, Any]] = []
    synth_responses: list[dict[str, Any]] = []
    if out.exists():
        existing = _load_json(str(out))
        rankings = list(existing.get("rankings", []))
        selected_responses = list(existing.get("selected_responses", []))
        weighted_selected_responses = list(existing.get("weighted_selected_responses", []))
        synth_responses = list(existing.get("synth_responses", []))
        print(
            f"reusing {len(rankings)} rankings, {len(selected_responses)} selections, "
            f"{len(weighted_selected_responses)} weighted selections, "
            f"{len(synth_responses)} synth rows from {out}",
            flush=True,
        )

    api_key = client.api_key_from_env(args.api_key)
    cid_by_model = candidate_ids(PANEL)

    ranking_keys = {(r["id"], r["critic_model"]) for r in rankings}
    jobs = []
    for task in tasks:
        for model in PANEL:
            if (task["id"], model) not in ranking_keys:
                jobs.append((task, model))
    print(f"running cross-rank critics: {len(jobs)} new / {len(tasks) * len(PANEL)} total", flush=True)

    def one_rank(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
        task, critic = job
        candidates = {
            cid_by_model[model]: _candidate_text(saved_rows[(model, task["id"])])
            for model in PANEL
        }
        return run_critic(
            critic_model=critic,
            task=task,
            candidates=candidates,
            adapter_specs=adapter_specs,
            base_url=args.base_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(one_rank, job) for job in jobs]
            for fut in as_completed(futures):
                rankings.append(fut.result())
                payload = make_payload(
                    args=args,
                    tasks=tasks,
                    rankings=rankings,
                    selected_responses=selected_responses,
                    weighted_selected_responses=weighted_selected_responses,
                    synth_responses=synth_responses,
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    selected_keys = {r["id"] for r in selected_responses}
    weighted_selected_keys = {r["id"] for r in weighted_selected_responses}
    synth_keys = {r["id"] for r in synth_responses}
    aiiq = _aiiq_scores()
    rankings_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rankings:
        rankings_by_task.setdefault(row["id"], []).append(row)

    synth_jobs: list[tuple[dict[str, Any], str, str, dict[str, Any], list[dict[str, Any]], str]] = []
    cid_to_model = {cid: model for model, cid in cid_by_model.items()}
    for task in tasks:
        task_rankings = rankings_by_task.get(task["id"], [])
        if len(task_rankings) < len(PANEL):
            continue
        ids = list(cid_by_model.values())
        candidates = {
            cid_by_model[model]: _candidate_text(saved_rows[(model, task["id"])])
            for model in PANEL
        }
        aggregate = aggregate_borda(task_rankings, ids)
        selected_id = aggregate["winner"]
        selected_model = next(model for model, cid in cid_by_model.items() if cid == selected_id)
        selected_row = saved_rows[(selected_model, task["id"])]
        selected_text = _candidate_text(selected_row)
        if task["id"] not in selected_keys:
            row = {
                "model": "crossrank_borda_selected_answer",
                "id": task["id"],
                "source": task["source"],
                "selected_model": selected_model,
                "selected_candidate_id": selected_id,
                "aggregate": aggregate,
                "text": selected_text,
            }
            row.update(score_response(task, selected_text))
            selected_responses.append(row)
            selected_keys.add(task["id"])
        if task["id"] not in weighted_selected_keys:
            weighted_aggregate = aggregate_weighted(
                task=task,
                rankings=task_rankings,
                ids=ids,
                cid_to_model=cid_to_model,
                candidates=candidates,
                aiiq=aiiq,
            )
            weighted_id = weighted_aggregate["winner"]
            weighted_model = cid_to_model[weighted_id]
            weighted_text = candidates[weighted_id]
            row = {
                "model": "crossrank_weighted_selected_answer",
                "id": task["id"],
                "source": task["source"],
                "selected_model": weighted_model,
                "selected_candidate_id": weighted_id,
                "aggregate": weighted_aggregate,
                "text": weighted_text,
            }
            row.update(score_response(task, weighted_text))
            weighted_selected_responses.append(row)
            weighted_selected_keys.add(task["id"])
        if task["id"] not in synth_keys:
            synth_jobs.append((task, selected_id, selected_text, aggregate, task_rankings, selected_model))

        payload = make_payload(
            args=args,
            tasks=tasks,
            rankings=rankings,
            selected_responses=selected_responses,
            weighted_selected_responses=weighted_selected_responses,
            synth_responses=synth_responses,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"running cross-rank finalizer: {len(synth_jobs)} new / {len(tasks)} total", flush=True)

    def one_synth(job: tuple[dict[str, Any], str, str, dict[str, Any], list[dict[str, Any]], str]) -> dict[str, Any]:
        task, selected_id, selected_text, aggregate, task_rankings, selected_model = job
        r = synthesize(
            task=task,
            selected_id=selected_id,
            selected_text=selected_text,
            aggregate=aggregate,
            rankings=task_rankings,
            adapter_specs=adapter_specs,
            base_url=args.base_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
        row = {
            "model": "crossrank_borda_glm_synth",
            "id": task["id"],
            "source": task["source"],
            "selected_model": selected_model,
            "selected_candidate_id": selected_id,
            "aggregate": aggregate,
            "text": "" if r.get("error") else str(r.get("text", "")),
            "error": r.get("error"),
            "usage": r.get("usage", {}),
            "latency_ms": r.get("latency_ms"),
            "finish_reason": r.get("finish_reason"),
        }
        if r.get("empty_reason"):
            row["empty_reason"] = r["empty_reason"]
        if not row.get("error"):
            row.update(score_response(task, row["text"]))
        return row

    if synth_jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(one_synth, job) for job in synth_jobs]
            for fut in as_completed(futures):
                row = fut.result()
                synth_responses.append(row)
                payload = make_payload(
                    args=args,
                    tasks=tasks,
                    rankings=rankings,
                    selected_responses=selected_responses,
                    weighted_selected_responses=weighted_selected_responses,
                    synth_responses=synth_responses,
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = make_payload(
        args=args,
        tasks=tasks,
        rankings=rankings,
        selected_responses=selected_responses,
        weighted_selected_responses=weighted_selected_responses,
        synth_responses=synth_responses,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("selected_summary:")
    print(json.dumps(payload["selected_summary"], indent=2))
    print("weighted_selected_summary:")
    print(json.dumps(payload["weighted_selected_summary"], indent=2))
    print("synth_summary:")
    print(json.dumps(payload["synth_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
