"""Run the mixed bio/reasoning packet.

Packet composition:
- 4 GPQA Diamond Biology multiple-choice questions
- 6 FutureHouse HLE Bio/Chem Gold biology multiple-choice questions
- 6 IFBench precise instruction-following prompts
- Optional AA-LCR long-context reasoning questions, with extracted documents included

GPQA/HLE are scored by answer-letter extraction. IFBench rows are scored by
small deterministic verifiers for the selected constraints. AA-LCR rows are
left as `needs_judge` unless a later judging pass grades them semantically.
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
from zipfile import ZipFile

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from trbench import adapters, client

OPEN7_MODELS = (
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.2",
    "minimax/minimax-m3",
    "deepseek/deepseek-v4-pro",
    "xiaomi/mimo-v2.5-pro",
    "nvidia/nvidia-nemotron-3-ultra-550b-a55b",
    "google/gemma-4-31b-it",
)

GPQA_INDEXES = (7, 10, 13, 26)
HLE_INDEXES = (0, 2, 3, 6, 36, 131)
IFBENCH_INDEXES = (0, 1, 2, 3, 17, 70)
AA_LCR_INDEXES = (8, 5)

MCQ_PROMPT = (
    "{question}\n\nAnswer choices:\n{options}\n\n"
    'Think step by step, then end your response with "The answer is (X)." '
    "where X is the letter of the correct option."
)

IF_PROMPT = "{prompt}"

AA_LCR_PROMPT = (
    "You are answering an Artificial Analysis Long Context Reasoning task.\n"
    "Use only the provided documents. Give the final answer directly and include "
    "enough supporting detail to make the reasoning auditable.\n\n"
    "Question:\n{question}\n\n"
    "Documents:\n{documents}"
)


def _letter_options(options: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {opt.strip()}" for i, opt in enumerate(options))


def _extract_answer_letter(text: str, n_options: int) -> str | None:
    if not text:
        return None
    letters = "".join(chr(65 + i) for i in range(n_options))
    patterns = [
        rf"answer\s+is\s*\(?([{letters}])\)?",
        rf"answer\s*:?\s*\(?([{letters}])\)?",
        rf"final\s+answer\s*:?\s*\(?([{letters}])\)?",
        rf"correct\s+(?:answer|option)\s+is\s*\(?([{letters}])\)?",
        rf"option\s*\(?([{letters}])\)?\s+is\s+correct",
        rf"choose\s*\(?([{letters}])\)?",
        rf"^\s*\(?([{letters}])\)?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    tail = text[-1200:]
    matches = re.findall(rf"\(([{letters}])\)\s*[.]?\s*$", tail, flags=re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].upper()
    matches = re.findall(rf"\b([{letters}])\b\s*[.)]?\s*$", tail, flags=re.IGNORECASE | re.MULTILINE)
    return matches[-1] if matches else None


def _clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _option_count_from_prompt(question: str, answer: str) -> int:
    labels = re.findall(r"(?m)^([A-Z])\.", question)
    if labels:
        return max(ord(label) - 64 for label in labels)
    return max(5, ord(answer.strip().upper()) - 64)


def _load_gpqa_tasks() -> list[dict[str, Any]]:
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    tasks: list[dict[str, Any]] = []
    for idx in GPQA_INDEXES:
        row = ds[idx]
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Deterministic rotation avoids always putting the gold at A.
        shift = idx % len(options)
        rotated = options[shift:] + options[:shift]
        answer = chr(65 + rotated.index(row["Correct Answer"]))
        tasks.append(
            {
                "id": f"gpqa_bio_{idx}",
                "source": "gpqa_diamond_biology",
                "source_index": idx,
                "question": row["Question"],
                "options": rotated,
                "n_options": len(rotated),
                "answer": answer,
                "subdomain": row.get("Subdomain", ""),
                "prompt": MCQ_PROMPT.format(question=row["Question"], options=_letter_options(rotated)),
                "scoring": "mcq_letter",
            }
        )
    return tasks


def _load_hle_tasks() -> list[dict[str, Any]]:
    ds = load_dataset("futurehouse/hle-gold-bio-chem", split="train")
    tasks: list[dict[str, Any]] = []
    for idx in HLE_INDEXES:
        row = ds[idx]
        n_options = _option_count_from_prompt(row["question"], row["answer"])
        tasks.append(
            {
                "id": f"hle_bio_{idx}",
                "source": "futurehouse_hle_gold_bio_chem",
                "source_index": idx,
                "question": row["question"],
                "answer": row["answer"],
                "n_options": n_options,
                "subdomain": row.get("raw_subject", ""),
                "prompt": (
                    f"{row['question']}\n\n"
                    'Think step by step, then end your response with "The answer is (X)." '
                    "where X is the letter of the correct option."
                ),
                "scoring": "mcq_letter",
            }
        )
    return tasks


def _load_ifbench_tasks() -> list[dict[str, Any]]:
    ds = load_dataset("allenai/IFBench_test", split="train")
    tasks: list[dict[str, Any]] = []
    for idx in IFBENCH_INDEXES:
        row = ds[idx]
        tasks.append(
            {
                "id": f"ifbench_{row['key']}",
                "source": "allenai_ifbench_test",
                "source_index": idx,
                "prompt": IF_PROMPT.format(prompt=row["prompt"]),
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": [_clean_kwargs(k) for k in row["kwargs"]],
                "scoring": "ifbench_selected",
            }
        )
    return tasks


def _aa_zip_path() -> str:
    return hf_hub_download(
        "ArtificialAnalysis/AA-LCR",
        "extracted_text/AA-LCR_extracted-text.zip",
        repo_type="dataset",
    )


def _aa_documents(row: dict[str, Any], zip_path: str) -> str:
    filenames = [f.strip() for f in str(row["data_source_filenames"]).split(";") if f.strip()]
    prefix = f"lcr/{row['document_category']}/{row['document_set_id']}/"
    chunks: list[str] = []
    with ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for filename in filenames:
            name = prefix + filename
            if name not in names:
                continue
            text = zf.read(name).decode("utf-8", errors="replace").strip()
            chunks.append(f"\n\n### {filename}\n{text}")
    return "".join(chunks).strip()


def _load_aa_lcr_tasks() -> list[dict[str, Any]]:
    ds = load_dataset("ArtificialAnalysis/AA-LCR", split="test")
    zip_path = _aa_zip_path()
    tasks: list[dict[str, Any]] = []
    for idx in AA_LCR_INDEXES:
        row = ds[idx]
        documents = _aa_documents(row, zip_path)
        tasks.append(
            {
                "id": f"aa_lcr_{row['question_id']}",
                "source": "artificialanalysis_aa_lcr",
                "source_index": idx,
                "document_category": row["document_category"],
                "document_set_id": row["document_set_id"],
                "input_tokens": row["input_tokens"],
                "question": row["question"],
                "answer": row["answer"],
                "data_source_filenames": row["data_source_filenames"],
                "prompt": AA_LCR_PROMPT.format(question=row["question"], documents=documents),
                "scoring": "needs_judge",
            }
        )
    return tasks


def load_tasks(*, include_aa_lcr: bool = True) -> list[dict[str, Any]]:
    tasks = [*_load_gpqa_tasks(), *_load_hle_tasks(), *_load_ifbench_tasks()]
    if include_aa_lcr:
        tasks.extend(_load_aa_lcr_tasks())
    return tasks


def _score_ifbench_selected(task: dict[str, Any], text: str) -> dict[str, Any]:
    instruction_ids = task.get("instruction_id_list", [])
    kwargs = task.get("kwargs", [])
    if not kwargs:
        return {"score": None, "status": "unsupported_ifbench_constraint"}
    if instruction_ids == ["count:numbers"]:
        expected = int(float(kwargs[0]["N"]))
        # Count standalone Arabic numerals, including decimals. This intentionally
        # does not count spelled-out numbers; IFBench asks for "numbers".
        actual = len(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text))
        return {"score": int(actual == expected), "status": "scored", "expected_count": expected, "actual_count": actual}
    if instruction_ids == ["format:emoji"]:
        sentences = [s.strip() for s in re.findall(r"[^.!?\n]+[.!?]+", text) if s.strip()]
        # Broad emoji coverage for common pictographs, symbols, and dingbats.
        emoji_at_end = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]\s*$")
        if not sentences and text.strip():
            sentences = [text.strip()]
        actual = sum(1 for sentence in sentences if emoji_at_end.search(sentence))
        ok = bool(sentences) and actual == len(sentences)
        return {
            "score": int(ok),
            "status": "scored",
            "expected": "emoji_at_end_of_every_sentence",
            "sentences": len(sentences),
            "sentences_with_terminal_emoji": actual,
        }
    if instruction_ids != ["count:keywords_multiple"]:
        return {"score": None, "status": "unsupported_ifbench_constraint"}
    spec = kwargs[0]
    expected = {
        spec["keyword1"]: 1,
        spec["keyword2"]: 2,
        spec["keyword3"]: 3,
        spec["keyword4"]: 5,
        spec["keyword5"]: 7,
    }
    actual = {
        keyword: len(re.findall(rf"(?<![A-Za-z]){re.escape(keyword)}(?![A-Za-z])", text, flags=re.IGNORECASE))
        for keyword in expected
    }
    ok = actual == expected
    return {"score": int(ok), "status": "scored", "expected_counts": expected, "actual_counts": actual}


def score_response(task: dict[str, Any], text: str) -> dict[str, Any]:
    if task["scoring"] == "mcq_letter":
        n_options = int(task.get("n_options") or len(task.get("options", [])) or 5)
        pred = _extract_answer_letter(text, n_options)
        status = "scored" if pred is not None else "no_answer"
        return {"score": int(pred == task["answer"]), "status": status, "pred": pred, "answer": task["answer"]}
    if task["scoring"] == "ifbench_selected":
        return _score_ifbench_selected(task, text)
    return {"score": None, "status": task["scoring"]}


def run_model(
    model: str,
    tasks: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    temperature: float | None,
    extra_body: dict[str, Any] | None,
    concurrency: int,
    retries: int,
    use_request: bool,
) -> list[dict[str, Any]]:
    def one(task: dict[str, Any]) -> dict[str, Any]:
        r = client.chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
            retries=retries,
            use_request=use_request,
        )
        row = {"model": model, "id": task["id"], "source": task["source"]}
        if r.get("error"):
            row["error"] = r["error"]
        else:
            text = r.get("text", "")
            row["text"] = text
            row.update(score_response(task, text))
            row["usage"] = r.get("usage", {})
            row["finish_reason"] = r.get("finish_reason")
            if r.get("empty_reason"):
                row["empty_reason"] = r["empty_reason"]
            row["latency_ms"] = r.get("latency_ms")
        return row

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for fut in as_completed(futures):
            out.append(fut.result())
    return out


def summarize(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in responses:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for model, rows in by_model.items():
        scored = [r for r in rows if r.get("score") is not None]
        correct = sum(int(r.get("score", 0)) for r in scored)
        summary.append(
            {
                "model": model,
                "scored_correct": correct,
                "scored_total": len(scored),
                "scored_accuracy": round(correct / len(scored), 3) if scored else None,
                "needs_judge": sum(1 for r in rows if r.get("status") == "needs_judge"),
                "no_answer": sum(1 for r in rows if r.get("status") == "no_answer"),
                "token_exhausted_empty": sum(
                    1 for r in rows if r.get("empty_reason") == "completion_token_budget_exhausted"
                ),
                "errors": sum(1 for r in rows if r.get("error")),
            }
        )
    return sorted(summary, key=lambda r: (-(r["scored_accuracy"] or 0), -r["scored_correct"], r["errors"], r["model"]))


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    hidden = {"prompt"}
    return {k: v for k, v in task.items() if k not in hidden}


def load_reused_responses(path: str | None, models: list[str], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    task_by_id = {task["id"]: task for task in tasks}
    wanted = {(model, task["id"]) for model in models for task in tasks}
    reused: list[dict[str, Any]] = []
    for row in data.get("responses", []):
        key = (str(row.get("model")), str(row.get("id")))
        if key not in wanted:
            continue
        task = task_by_id.get(key[1])
        if not task:
            continue
        updated = dict(row)
        if not updated.get("error") and "text" in updated:
            updated.update(score_response(task, str(updated.get("text", ""))))
        reused.append(updated)
    return reused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the mixed bio/reasoning packet.")
    parser.add_argument("--models", default=",".join(OPEN7_MODELS))
    parser.add_argument("--base-url", default=client.DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--adapter-file", default="results/open_panel_adapter_recommendations.json")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=client.DEFAULT_RETRIES)
    parser.add_argument("--use-request", action="store_true", help="Use direct POST instead of SDK streaming collector.")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--exclude-aa-lcr", action="store_true", help="Skip AA-LCR long-context rows.")
    parser.add_argument("--reuse-results", default=None, help="Existing result JSON to reuse matching model/task rows from.")
    parser.add_argument("--out", default="results/mixed_bio_reasoning_packet_open7.json")
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = load_tasks(include_aa_lcr=not args.exclude_aa_lcr)
    adapter_specs = adapters.load_adapter_file(args.adapter_file)

    responses: list[dict[str, Any]] = load_reused_responses(args.reuse_results, models, tasks)
    reused_keys = {(r["model"], r["id"]) for r in responses}
    if not args.manifest_only:
        api_key = client.api_key_from_env(args.api_key)
        for model in models:
            missing_tasks = [task for task in tasks if (model, task["id"]) not in reused_keys]
            if not missing_tasks:
                print(f"reusing mixed_bio_packet: {model} ({len(tasks)} tasks)")
                continue
            adapter = adapters.adapter_for(
                model,
                adapter_specs,
                default_max_tokens=args.max_tokens,
                default_timeout=args.timeout,
            )
            print(
                f"running mixed_bio_packet: {model} ({len(missing_tasks)} new / {len(tasks)} total tasks, "
                f"max_tokens={adapter.max_tokens}, timeout={adapter.timeout}, "
                f"temperature={'omitted' if adapter.temperature is None else adapter.temperature}, "
                f"use_request={args.use_request or adapter.use_request})"
            )
            responses.extend(
                run_model(
                    model,
                    missing_tasks,
                    base_url=args.base_url,
                    api_key=api_key,
                    max_tokens=adapter.max_tokens or args.max_tokens,
                    timeout=adapter.timeout or args.timeout,
                    temperature=adapter.temperature,
                    extra_body=adapter.extra_body,
                    concurrency=args.concurrency,
                    retries=args.retries,
                    use_request=args.use_request or adapter.use_request,
                )
            )

    result = {
        "eval": "mixed_bio_reasoning_packet",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url_host": urllib.parse.urlparse(args.base_url).netloc,
        "models": models,
        "adapter_file": args.adapter_file,
        "reuse_results": args.reuse_results,
        "adapter_settings": {m: adapters.public_adapter_settings(m, adapter_specs) for m in models},
        "task_count": len(tasks),
        "tasks": [public_task(t) for t in tasks],
        "responses": sorted(responses, key=lambda r: (r["model"], r["id"])),
        "summary": summarize(responses) if responses else [],
        "notes": [
            "GPQA/HLE/selected IFBench rows are deterministically scored.",
            "AA-LCR rows include extracted documents but require a semantic judge pass.",
            "AA-LCR skipped for this run." if args.exclude_aa_lcr else "AA-LCR included for this run.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
