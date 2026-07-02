"""Harbor Terminus-2 agent backed by the TrustedRouter SDK.

Run example:

  TB_TR_CONFIRM=1 PYTHONPATH=scripts uvx --from harbor --with trusted-router-py harbor run \
    -d terminal-bench/terminal-bench-2-1 \
    --agent-import-path harbor_tr_agent:TRHarborTerminus \
    -m trustedrouter/synth \
    -i sqlite-db-truncate \
    --agent-kwarg max_tokens=65536 \
    --jobs-dir runs/harbor

This mirrors scripts/tb_tr_agent.py for the legacy terminal-bench harness, but uses
Harbor's native Terminus-2 scaffold so it can run Terminal-Bench 2.x datasets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.metric import UsageInfo
from harbor.models.trial.paths import EnvironmentPaths

TR_KEY_PATH = os.environ.get("TB_TR_KEY_PATH", "~/claude/.tr_key")
TR_BASE_URL = os.environ.get("TRUSTEDROUTER_BASE_URL") or None
TR_REQUEST_TIMEOUT = float(os.environ.get("TRUSTEDROUTER_TIMEOUT", "900"))
TR_MAX_RETRIES = int(os.environ.get("TRUSTEDROUTER_MAX_RETRIES", "2"))
TERMINUS_CONTROL_KEY_SPEC_TEXT = (
    "Shell commands must end with \\n. Special keys must be sent alone without \\n, "
    'e.g. {"keystrokes":"C-c","duration":0.5}.'
)
TERMINUS_CONTROL_KEY_SPEC_TEMPLATE = TERMINUS_CONTROL_KEY_SPEC_TEXT.replace(
    "{", "{{"
).replace("}", "}}")


def _patch_terminus_prompt_template(template: str) -> str:
    """Add the tmux control-key contract to Harbor's model-facing prompt."""
    if TERMINUS_CONTROL_KEY_SPEC_TEMPLATE in template:
        return template

    for marker in (
        'The "duration" attribute specifies',
        "The `duration` attribute",
        "Task Description:",
    ):
        if marker in template:
            return template.replace(
                marker, TERMINUS_CONTROL_KEY_SPEC_TEMPLATE + "\n\n" + marker, 1
            )

    return template.rstrip() + "\n\n" + TERMINUS_CONTROL_KEY_SPEC_TEMPLATE + "\n"


def _ensure_control_key_spec(prompt: str, messages: list[dict[str, str]]) -> str:
    """Keep the terminal protocol visible after Harbor context rebuilds."""
    if TERMINUS_CONTROL_KEY_SPEC_TEXT in prompt:
        return prompt
    if any(TERMINUS_CONTROL_KEY_SPEC_TEXT in m["content"] for m in messages):
        return prompt
    return prompt.rstrip() + "\n\n" + TERMINUS_CONTROL_KEY_SPEC_TEXT


def _content_to_str(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _redacted_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return {
        "redacted": True,
        "chars": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def _request_for_replay(request: dict[str, Any]) -> dict[str, Any]:
    """Copy the request for repo-safe replay without secret prompt bodies."""
    replay = json.loads(json.dumps(request, default=str))
    for tool in replay.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        params = tool.get("parameters")
        if not isinstance(params, dict):
            continue
        for key in ("panel_prompt", "synthesis_prompt"):
            if key in params:
                params[key] = _redacted_text(params[key])
    return replay


def _usage_from_response(dumped: dict[str, Any]) -> UsageInfo | None:
    usage = dumped.get("usage") or {}
    if not isinstance(usage, dict):
        return None

    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    prompt_details = usage.get("prompt_tokens_details") or {}
    cache_tokens = int(
        prompt_details.get("cached_tokens")
        or usage.get("cache_tokens")
        or usage.get("cached_tokens")
        or 0
    )

    cost_usd = 0.0
    if usage.get("cost_microdollars") is not None:
        cost_usd = float(usage["cost_microdollars"]) / 1_000_000
    elif usage.get("cost_usd") is not None:
        cost_usd = float(usage["cost_usd"])
    elif usage.get("cost") is not None:
        cost_usd = float(usage["cost"])

    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_tokens=cache_tokens,
        cost_usd=cost_usd,
    )


def _message_content(dumped: dict[str, Any]) -> str:
    choice = (dumped.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (message.get("content") or "").strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _chunk_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _chunk_reasoning(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = delta.get(key)
        if isinstance(value, str):
            return value
    return ""


def _collect_chat_completion(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct a chat.completion object from streamed chunk frames."""
    if not chunks:
        return {
            "id": "",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
        }

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    role = "assistant"
    usage: dict[str, Any] | None = None
    trustedrouter: Any = None
    tool_calls: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        if chunk.get("trustedrouter") is not None:
            trustedrouter = chunk["trustedrouter"]

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice0 = choices[0]
        delta = choice0.get("delta") or {}
        if isinstance(delta.get("role"), str):
            role = delta["role"]
        if isinstance(delta.get("content"), str):
            text_parts.append(delta["content"])
        reasoning = _chunk_reasoning(chunk)
        if reasoning:
            reasoning_parts.append(reasoning)

        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            idx = int(tc.get("index", 0) or 0)
            slot = tool_calls.setdefault(
                idx,
                {
                    "index": idx,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tc.get("id"):
                slot["id"] = tc["id"]
            if tc.get("type"):
                slot["type"] = tc["type"]
            fn = tc.get("function")
            if isinstance(fn, dict):
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    slot["function"]["arguments"] += fn["arguments"]

        if choice0.get("finish_reason"):
            finish_reason = choice0["finish_reason"]

    last = chunks[-1]
    content = "".join(text_parts)
    message: dict[str, Any] = {
        "role": role,
        "content": content if content else (None if tool_calls else ""),
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

    result: dict[str, Any] = {
        "id": last.get("id", ""),
        "object": "chat.completion",
        "created": last.get("created", 0),
        "model": last.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or "stop",
            }
        ],
    }
    if usage is not None:
        result["usage"] = usage
    if trustedrouter is not None:
        result["trustedrouter"] = trustedrouter
    return result


class TrustedRouterHarborLLM(BaseLLM):
    """Harbor BaseLLM implementation backed by TrustedRouter chat completions."""

    def __init__(
        self,
        model: str,
        logs_dir: Path,
        temperature: float | None = None,
        max_tokens: int = 65536,
        context_limit: int = 1_000_000,
        output_limit: int | None = 65536,
        max_empty_retries: int = 2,
        panel_prompt: str | None = None,
        synthesis_prompt: str | None = None,
        advisor_depth: int | None = None,
        advisor_worker_models: list[str] | str | None = None,
        advisor_models: list[str] | str | None = None,
        advisor_max_get_advice_calls: int | None = None,
        advisor_max_tokens: int | None = None,
        advisor_timeout_ms: int | None = None,
        base_url: str | None = None,
        request_timeout: float | str | None = None,
        request_retries: int | str | None = None,
        stream: bool = True,
    ):
        super().__init__()
        if not os.environ.get("TB_TR_CONFIRM"):
            raise RuntimeError(
                "harbor_tr_agent: refusing to call TrustedRouter because it costs money. "
                "Re-run with TB_TR_CONFIRM=1 to proceed."
            )

        from trustedrouter import TrustedRouter

        api_key = os.environ.get("TRUSTEDROUTER_API_KEY")
        if not api_key:
            api_key = Path(os.path.expanduser(TR_KEY_PATH)).read_text().strip()

        self._base_url = base_url or TR_BASE_URL
        self._request_timeout = _as_optional_float(request_timeout) or TR_REQUEST_TIMEOUT
        self._request_retries = (
            _as_optional_int(request_retries)
            if request_retries is not None
            else TR_MAX_RETRIES
        )
        self._client = TrustedRouter(
            api_key=api_key,
            base_url=self._base_url,
            timeout=self._request_timeout,
            max_retries=self._request_retries,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_limit = context_limit
        self._output_limit = output_limit
        self._max_empty_retries = max_empty_retries
        self._panel_prompt = panel_prompt
        self._synthesis_prompt = synthesis_prompt
        self._advisor_depth = _as_optional_int(advisor_depth)
        self._advisor_worker_models = _as_list(advisor_worker_models)
        self._advisor_models = _as_list(advisor_models)
        self._advisor_max_get_advice_calls = _as_optional_int(advisor_max_get_advice_calls)
        self._advisor_max_tokens = _as_optional_int(advisor_max_tokens)
        self._advisor_timeout_ms = _as_optional_int(advisor_timeout_ms)
        self._stream = _as_bool(stream)
        self._calls_dir = logs_dir / "trustedrouter-calls"
        self._call_index = 0

    def _advisor_tool(self) -> dict[str, Any] | None:
        parameters: dict[str, Any] = {}
        if self._advisor_depth is not None:
            parameters["depth"] = self._advisor_depth
        if self._advisor_worker_models is not None:
            parameters["worker_models"] = self._advisor_worker_models
        if self._advisor_models is not None:
            parameters["advisor_models"] = self._advisor_models
        if self._advisor_max_get_advice_calls is not None:
            parameters["max_get_advice_calls"] = self._advisor_max_get_advice_calls
        if self._advisor_max_tokens is not None:
            parameters["advisor_max_tokens"] = self._advisor_max_tokens
        if self._advisor_timeout_ms is not None:
            parameters["advisor_timeout_ms"] = self._advisor_timeout_ms
        if not parameters:
            return None
        return {"type": "trustedrouter:advisor", "parameters": parameters}

    def _stream_label(self, call_id: int, chunk: dict[str, Any]) -> tuple[str, str]:
        tr = chunk.get("trustedrouter") or {}
        socrates = tr.get("socrates") if isinstance(tr, dict) else None
        if isinstance(socrates, dict):
            stage = socrates.get("stage") or "socrates"
            model = socrates.get("model") or chunk.get("model") or self._model
            return f"call-{call_id:04d}:{stage}:{model}", socrates.get("event") or ""
        return f"call-{call_id:04d}:{chunk.get('model') or self._model}", ""

    def _write_stream_live(
        self,
        *,
        call_id: int,
        chunk: dict[str, Any],
        live_file: Any,
    ) -> None:
        label, event = self._stream_label(call_id, chunk)
        tr = chunk.get("trustedrouter") or {}
        socrates = tr.get("socrates") if isinstance(tr, dict) else None

        text = ""
        if isinstance(socrates, dict):
            for key in ("text", "delta", "content"):
                if isinstance(socrates.get(key), str):
                    text = socrates[key]
                    break
        if not text:
            text = _chunk_text(chunk)

        reasoning = _chunk_reasoning(chunk)
        if not reasoning and isinstance(socrates, dict):
            for key in ("thinking", "reasoning", "reasoning_content"):
                if isinstance(socrates.get(key), str):
                    reasoning = socrates[key]
                    break
        if event and not text and not reasoning:
            detail = ""
            if isinstance(socrates, dict):
                raw_detail = socrates.get("detail")
                if isinstance(raw_detail, dict):
                    finish = raw_detail.get("finish_reason")
                    cost = raw_detail.get("cost_microdollars")
                    detail = f" finish={finish} cost_microdollars={cost}"
            line = f"\n[tr-stream {label}] {event}{detail}\n"
            live_file.write(line)
            live_file.flush()
            print(line, end="", flush=True)
            return

        if reasoning:
            line = f"[tr-thinking {label}] {reasoning}\n"
            live_file.write(line)
            live_file.flush()
            print(line, end="", flush=True)

        if text:
            line = f"[tr-stream {label}] {text}\n"
            live_file.write(line)
            live_file.flush()
            print(line, end="", flush=True)

    def _stream_completion(
        self,
        request: dict[str, Any],
        *,
        call_id: int,
        stream_path: Path,
        live_path: Path,
    ) -> dict[str, Any]:
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        chunks: list[dict[str, Any]] = []
        request = dict(request)
        stream_options = dict(request.get("stream_options") or {})
        stream_options.setdefault("include_usage", True)
        request["stream_options"] = stream_options

        with stream_path.open("w", encoding="utf-8") as stream_file, live_path.open(
            "w", encoding="utf-8"
        ) as live_file:
            print(
                f"\n[tr-stream call-{call_id:04d}] start model={self._model} base_url={self._base_url}\n",
                end="",
                flush=True,
            )
            for chunk in self._client.chat_completions_chunk_stream(**request):
                dumped = chunk.model_dump(mode="json")
                chunks.append(dumped)
                stream_file.write(json.dumps(dumped, ensure_ascii=False) + "\n")
                stream_file.flush()
                self._write_stream_live(
                    call_id=call_id,
                    chunk=dumped,
                    live_file=live_file,
                )
            print(f"[tr-stream call-{call_id:04d}] end\n", end="", flush=True)

        return _collect_chat_completion(chunks)

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] | None = None,
        logging_path: Path | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = [
            {"role": m.get("role", "user"), "content": _content_to_str(m.get("content"))}
            for m in (message_history or [])
        ]
        prompt = _ensure_control_key_spec(prompt, messages)
        messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        synth_parameters: dict[str, Any] = {}
        if self._panel_prompt:
            synth_parameters["panel_prompt"] = self._panel_prompt
        if self._synthesis_prompt:
            synth_parameters["synthesis_prompt"] = self._synthesis_prompt
        tools: list[dict[str, Any]] = []
        if synth_parameters:
            tools.append({"type": "trustedrouter:synth", "parameters": synth_parameters})
        advisor_tool = self._advisor_tool()
        if advisor_tool is not None:
            tools.append(advisor_tool)
        if tools:
            request["tools"] = tools
        if self._temperature is not None:
            request["temperature"] = self._temperature

        last_response: tuple[dict[str, Any], Path, int] | None = None
        for attempt in range(1, self._max_empty_retries + 2):
            self._call_index += 1
            call_id = self._call_index
            started = time.monotonic()
            started_at = datetime.now(UTC).isoformat()
            response_path = self._calls_dir / f"call-{call_id:04d}.response.json"
            request_path = self._calls_dir / f"call-{call_id:04d}.request.json"
            meta_path = self._calls_dir / f"call-{call_id:04d}.meta.json"
            stream_path = self._calls_dir / f"call-{call_id:04d}.stream.jsonl"
            live_path = self._calls_dir / f"call-{call_id:04d}.live.txt"
            manifest_path = self._calls_dir / "manifest.jsonl"

            _write_json(request_path, _request_for_replay(request))
            _append_jsonl(
                manifest_path,
                {
                    "event": "started",
                    "call_id": call_id,
                    "started_at": started_at,
                    "model": self._model,
                    "base_url": self._base_url,
                    "request_timeout": self._request_timeout,
                    "request_retries": self._request_retries,
                    "attempt": attempt,
                    "message_count": len(messages),
                    "message_chars": sum(len(m["content"]) for m in messages),
                    "request_path": request_path.name,
                    "response_path": response_path.name,
                    "meta_path": meta_path.name,
                    "stream_path": stream_path.name if self._stream else None,
                    "live_path": live_path.name if self._stream else None,
                },
            )

            try:
                if self._stream:
                    dumped = await asyncio.to_thread(
                        self._stream_completion,
                        request,
                        call_id=call_id,
                        stream_path=stream_path,
                        live_path=live_path,
                    )
                else:
                    resp = await asyncio.to_thread(
                        self._client.chat_completions, **request
                    )
                    dumped = resp.model_dump(mode="json")
            except Exception as exc:
                _write_json(
                    meta_path,
                    {
                        "started_at": started_at,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                        "model": self._model,
                        "base_url": self._base_url,
                        "request_timeout": self._request_timeout,
                        "request_retries": self._request_retries,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens,
                        "has_panel_prompt": bool(self._panel_prompt),
                        "has_synthesis_prompt": bool(self._synthesis_prompt),
                        "has_advisor_tool": self._advisor_tool() is not None,
                        "advisor_worker_models": self._advisor_worker_models,
                        "advisor_models": self._advisor_models,
                        "stream": self._stream,
                        "panel_prompt_chars": len(self._panel_prompt or ""),
                        "synthesis_prompt_chars": len(self._synthesis_prompt or ""),
                        "message_count": len(messages),
                        "message_chars": sum(len(m["content"]) for m in messages),
                        "attempt": attempt,
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "request_path": request_path.name,
                        "full_response_path": response_path.name,
                        "stream_path": stream_path.name if self._stream else None,
                        "live_path": live_path.name if self._stream else None,
                    },
                )
                _append_jsonl(
                    manifest_path,
                    {
                        "event": "error",
                        "call_id": call_id,
                        "started_at": started_at,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                        "model": self._model,
                        "base_url": self._base_url,
                        "request_timeout": self._request_timeout,
                        "request_retries": self._request_retries,
                        "attempt": attempt,
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "request_path": request_path.name,
                        "response_path": response_path.name,
                        "meta_path": meta_path.name,
                        "stream_path": stream_path.name if self._stream else None,
                        "live_path": live_path.name if self._stream else None,
                    },
                )
                raise

            elapsed_ms = round((time.monotonic() - started) * 1000)
            choice = (dumped.get("choices") or [{}])[0]
            content = _message_content(dumped)
            empty_retry = content == "" and attempt <= self._max_empty_retries

            _write_json(response_path, dumped)
            _write_json(
                meta_path,
                {
                    "started_at": started_at,
                    "elapsed_ms": elapsed_ms,
                    "model": self._model,
                    "base_url": self._base_url,
                    "request_timeout": self._request_timeout,
                    "request_retries": self._request_retries,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "has_panel_prompt": bool(self._panel_prompt),
                    "has_synthesis_prompt": bool(self._synthesis_prompt),
                    "has_advisor_tool": self._advisor_tool() is not None,
                    "advisor_worker_models": self._advisor_worker_models,
                    "advisor_models": self._advisor_models,
                    "stream": self._stream,
                    "panel_prompt_chars": len(self._panel_prompt or ""),
                    "synthesis_prompt_chars": len(self._synthesis_prompt or ""),
                    "message_count": len(messages),
                    "message_chars": sum(len(m["content"]) for m in messages),
                    "attempt": attempt,
                    "response_id": dumped.get("id"),
                    "response_model": dumped.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "content_chars": len(content),
                    "usage": dumped.get("usage"),
                    "trustedrouter": dumped.get("trustedrouter"),
                    "request_path": request_path.name,
                    "full_response_path": response_path.name,
                    "stream_path": stream_path.name if self._stream else None,
                    "live_path": live_path.name if self._stream else None,
                    "empty_response_retry": empty_retry,
                    "retry_reason": "empty_content" if empty_retry else None,
                },
            )
            _append_jsonl(
                manifest_path,
                {
                    "event": "completed",
                    "call_id": call_id,
                    "started_at": started_at,
                    "elapsed_ms": elapsed_ms,
                    "model": self._model,
                    "base_url": self._base_url,
                    "request_timeout": self._request_timeout,
                    "request_retries": self._request_retries,
                    "attempt": attempt,
                    "response_id": dumped.get("id"),
                    "response_model": dumped.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "content_chars": len(content),
                    "usage": dumped.get("usage"),
                    "empty_response_retry": empty_retry,
                    "request_path": request_path.name,
                    "response_path": response_path.name,
                    "meta_path": meta_path.name,
                    "stream_path": stream_path.name if self._stream else None,
                    "live_path": live_path.name if self._stream else None,
                },
            )

            last_response = (dumped, response_path, attempt)
            if not empty_retry:
                break

        if last_response is None:
            raise RuntimeError("TrustedRouter returned no response")

        dumped, response_path, final_attempt = last_response
        choice = (dumped.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = _message_content(dumped)
        reasoning_content = message.get("reasoning_content")
        usage = _usage_from_response(dumped)

        if logging_path is not None:
            _write_json(
                logging_path,
                {
                    "model": self._model,
                    "base_url": self._base_url,
                    "request_timeout": self._request_timeout,
                    "request_retries": self._request_retries,
                    "message_count": len(messages),
                    "message_chars": sum(len(m["content"]) for m in messages),
                    "final_attempt": final_attempt,
                    "has_panel_prompt": bool(self._panel_prompt),
                    "has_synthesis_prompt": bool(self._synthesis_prompt),
                    "has_advisor_tool": self._advisor_tool() is not None,
                    "advisor_worker_models": self._advisor_worker_models,
                    "advisor_models": self._advisor_models,
                    "stream": self._stream,
                    "panel_prompt_chars": len(self._panel_prompt or ""),
                    "synthesis_prompt_chars": len(self._synthesis_prompt or ""),
                    "response_id": dumped.get("id"),
                    "response_model": dumped.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "content_chars": len(content),
                    "usage": dumped.get("usage"),
                    "trustedrouter": dumped.get("trustedrouter"),
                    "full_response_path": str(response_path),
                    "stream_path": str(
                        response_path.with_name(
                            response_path.name.replace(".response.json", ".stream.jsonl")
                        )
                    )
                    if self._stream
                    else None,
                    "live_path": str(
                        response_path.with_name(
                            response_path.name.replace(".response.json", ".live.txt")
                        )
                    )
                    if self._stream
                    else None,
                },
            )

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model_name=dumped.get("model") or self._model,
            usage=usage,
            response_id=dumped.get("id"),
        )

    def get_model_context_limit(self) -> int:
        return self._context_limit

    def get_model_output_limit(self) -> int | None:
        return self._output_limit


class DirectAptTmuxSession(TmuxSession):
    """TmuxSession that avoids apt-get update when package lists are already present."""

    def _get_combined_install_command(
        self, system_info: dict[str, Any], tools: list[str]
    ) -> str:
        package_manager = system_info.get("package_manager")
        if package_manager != "apt-get":
            return super()._get_combined_install_command(system_info, tools)

        packages = " ".join(tools)
        direct_install = (
            "DEBIAN_FRONTEND=noninteractive "
            f"apt-get install -y --no-install-recommends {packages}"
        )
        bounded_update = (
            "DEBIAN_FRONTEND=noninteractive "
            "apt-get update "
            "-o Acquire::Retries=2 "
            "-o Acquire::http::Timeout=30 "
            "-o Acquire::https::Timeout=30"
        )
        return f"({direct_install}) || ({bounded_update} && {direct_install})"


class TRHarborTerminus(Terminus2):
    """Harbor Terminus-2 with the LLM call routed through TrustedRouter."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_tokens: int = 65536,
        context_limit: int = 1_000_000,
        output_limit: int | None = 65536,
        max_empty_retries: int = 2,
        panel_prompt: str | None = None,
        synthesis_prompt: str | None = None,
        advisor_depth: int | None = None,
        advisor_worker_models: list[str] | str | None = None,
        advisor_models: list[str] | str | None = None,
        advisor_max_get_advice_calls: int | None = None,
        advisor_max_tokens: int | None = None,
        advisor_timeout_ms: int | None = None,
        base_url: str | None = None,
        request_timeout: float | str | None = None,
        request_retries: int | str | None = None,
        stream: bool = True,
        **kwargs: Any,
    ):
        if model_name is None:
            raise ValueError("model_name is required")
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._prompt_template = _patch_terminus_prompt_template(self._prompt_template)
        self._llm = TrustedRouterHarborLLM(
            model=model_name,
            logs_dir=logs_dir,
            temperature=self._temperature,
            max_tokens=int(max_tokens),
            context_limit=int(context_limit),
            output_limit=int(output_limit) if output_limit is not None else None,
            max_empty_retries=int(max_empty_retries),
            panel_prompt=panel_prompt,
            synthesis_prompt=synthesis_prompt,
            advisor_depth=advisor_depth,
            advisor_worker_models=advisor_worker_models,
            advisor_models=advisor_models,
            advisor_max_get_advice_calls=advisor_max_get_advice_calls,
            advisor_max_tokens=advisor_max_tokens,
            advisor_timeout_ms=advisor_timeout_ms,
            base_url=base_url,
            request_timeout=request_timeout,
            request_retries=request_retries,
            stream=stream,
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        if self._record_terminal_session:
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None

        self._session = DirectAptTmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    @staticmethod
    def name() -> str:
        return "tr-harbor-terminus"
