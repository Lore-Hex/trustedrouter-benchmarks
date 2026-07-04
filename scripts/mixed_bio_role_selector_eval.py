"""Role-aware single-selector evaluation on the mixed bio packet.

This reuses saved panel answers. For each task, one selector call assigns
task-relevant model roles and chooses one existing anonymized answer. The final
scored output is the selected existing answer, not a rewrite.
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
SELECTOR = "moonshotai/kimi-k2.6"

CAPABILITY_NOTES = {
    "deepseek/deepseek-v4-pro": "Strong biomedical/literature reasoning coverage; useful for scientific argument and IFBench selected constraints.",
    "moonshotai/kimi-k2.6": "Strong minority-hypothesis and ambiguity checking signal; recovered unique IFBench cases.",
    "xiaomi/mimo-v2.5-pro": "Strongest local HLE Bio rescue signal; useful for hard biology minority answers.",
    "nvidia/nvidia-nemotron-3-ultra-550b-a55b": "Good GPQA science coverage and complementary HLE rescue signal.",
    "minimax/minimax-m3": "High external prior for instruction following and exact constraint compliance.",
}

SELECTOR_SYSTEM = (
    "You are a role-aware answer selector. You will see an original task and anonymized existing answers "
    "from several models, plus private capability notes mapping candidate ids to model strengths. First assign "
    "each candidate a task-relevant role based on the capability notes and the task. Then select the existing "
    "candidate answer most likely to be correct. Do not majority vote. For HLE/hard bio, explicitly evaluate "
    "minority answers from hard-bio specialists before rejecting them. For IFBench/instruction-following, prioritize "
    "exact constraint compliance and candidate answers that visibly satisfy counts/formats/keywords. Return JSON only "
    "with keys roles, selected_answer_id, constraint_check, minority_review, reason, confidence. Do not write a new "
    "final answer."
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _response_map(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("model")), str(r.get("id"))): r for r in data.get("responses", [])}


def _candidate_text(row: dict[str, Any]) -> str:
    text = str(row.get("text", "")).strip()
    if len(text) > 5000:
        return text[:5000] + "\n[truncated]"
    return text


def _full_candidate_text(row: dict[str, Any]) -> str:
    return str(row.get("text", "")).strip()


def candidate_ids(panel: list[str]) -> dict[str, str]:
    return {model: f"A{i + 1}" for i, model in enumerate(panel)}


def selector_prompt(task: dict[str, Any], cid_by_model: dict[str, str], saved_rows: dict[tuple[str, str], dict[str, Any]]) -> str:
    lines = [
        "Original task:",
        task["prompt"],
        "",
        "Candidate capability notes:",
    ]
    for model in PANEL:
        lines.append(f"- {cid_by_model[model]}: {CAPABILITY_NOTES[model]}")
    lines.append("")
    lines.append("Candidate answers:")
    for model in PANEL:
        cid = cid_by_model[model]
        lines.append(f"\n[{cid}]\n{_candidate_text(saved_rows[(model, task['id'])])}")
    return "\n".join(lines)


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


def parse_selected(text: str, ids: list[str]) -> tuple[str, dict[str, Any] | None]:
    data = _extract_json(text)
    if data:
        raw = data.get("selected_answer_id") or data.get("selected") or data.get("best_answer_id")
        if isinstance(raw, str) and raw.strip() in ids:
            return raw.strip(), data
    for cid in ids:
        if re.search(rf"\b{re.escape(cid)}\b", text):
            return cid, data
    return ids[0], data


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


def run_one(
    *,
    task: dict[str, Any],
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
        model=SELECTOR,
        messages=[
            {"role": "system", "content": SELECTOR_SYSTEM},
            {"role": "user", "content": selector_prompt(task, cid_by_model, saved_rows)},
        ],
        adapter_specs=adapter_specs,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
    )
    selector_text = "" if r.get("error") else str(r.get("text", ""))
    selected_id, parsed = parse_selected(selector_text, ids)
    selected_model = next(model for model, cid in cid_by_model.items() if cid == selected_id)
    selected_text = _full_candidate_text(saved_rows[(selected_model, task["id"])])
    row: dict[str, Any] = {
        "model": "role_aware_kimi_selected_answer",
        "id": task["id"],
        "source": task["source"],
        "selector_model": SELECTOR,
        "selected_model": selected_model,
        "selected_candidate_id": selected_id,
        "selector_text": selector_text,
        "selector_parsed": parsed,
        "selector_error": r.get("error"),
        "selector_usage": r.get("usage", {}),
        "selector_latency_ms": r.get("latency_ms"),
        "text": selected_text,
    }
    row.update(score_response(task, selected_text))
    return row


def make_payload(args: argparse.Namespace, tasks: list[dict[str, Any]], responses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eval": "mixed_bio_role_selector",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "saved_results": args.saved_results,
        "panel": PANEL,
        "selector": SELECTOR,
        "task_count": len(tasks),
        "tasks": [public_task(t) for t in tasks],
        "responses": sorted(responses, key=lambda r: r["id"]),
        "summary": summarize(responses),
        "notes": [
            "Panel responses were reused from saved solo runs.",
            "Selector assigns roles and picks an existing answer; no freeform final synthesis.",
            "No saved correctness labels are included in selector prompts.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run role-aware single-selector on mixed bio packet.")
    parser.add_argument("--saved-results", default="results/mixed_bio_reasoning_packet_16_open7_plus_frontier_refs_corrected.json")
    parser.add_argument("--adapter-file", action="append", default=[])
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=client.DEFAULT_RETRIES)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", default="results/mixed_bio_reasoning_packet_16_role_aware_selector.json")
    args = parser.parse_args()

    saved = _load_json(args.saved_results)
    saved_rows = _response_map(saved)
    tasks = load_tasks(include_aa_lcr=False)
    cid_by_model = candidate_ids(PANEL)
    adapter_specs: dict[str, dict[str, Any]] = {}
    for adapter_file in args.adapter_file:
        adapter_specs.update(adapters.load_adapter_file(adapter_file))

    out = Path(args.out)
    responses: list[dict[str, Any]] = []
    if out.exists():
        existing = _load_json(str(out))
        responses = list(existing.get("responses", []))
        print(f"reusing {len(responses)} role-selector rows from {out}", flush=True)

    existing_ids = {r["id"] for r in responses}
    pending = [task for task in tasks if task["id"] not in existing_ids]
    print(f"running role-aware selector: {len(pending)} new / {len(tasks)} total", flush=True)
    api_key = client.api_key_from_env(args.api_key)

    def one(task: dict[str, Any]) -> dict[str, Any]:
        return run_one(
            task=task,
            cid_by_model=cid_by_model,
            saved_rows=saved_rows,
            adapter_specs=adapter_specs,
            base_url=args.base_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(one, task) for task in pending]
            for fut in as_completed(futures):
                responses.append(fut.result())
                payload = make_payload(args, tasks, responses)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = make_payload(args, tasks, responses)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
