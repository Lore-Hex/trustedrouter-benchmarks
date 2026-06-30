"""A litellm custom provider that routes `trustedrouter/<model>` through the
TrustedRouter Python SDK.

External agentic harnesses (tau2-bench, terminal-bench) drive their agent loop
with `litellm.completion(...)`. Pointing that at TR's OpenAI-compatible base URL
works, but it uses litellm's generic client, not our SDK. Registering this custom
provider makes litellm dispatch `trustedrouter/<model>` calls to the real
TrustedRouter SDK instead — so the agentic evals exercise the same client path a
TR customer uses, with NO fork of the upstream harness (the tool still just calls
`litellm.completion`).

Self-contained on purpose: it imports only `litellm` + `trustedrouter`, so it can
be dropped onto an external harness's venv/PYTHONPATH (which has neither `trbench`
nor our other deps). Call `register()` once at process start (a sitecustomize.py
on the harness PYTHONPATH does this).

Tool-calling note: the SDK's streaming `chat_completions()` drops tool_calls, so
we use the SDK client's own `request()` (a direct POST) which preserves them —
the same approach trbench.client uses.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm
from litellm import CustomLLM
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from trustedrouter import TrustedRouter

_BASE_URL = os.environ.get("TRUSTEDROUTER_BASE_URL", "https://api.trustedrouter.com/v1")
_PROVIDER = "trustedrouter"
_PASSTHROUGH = ("temperature", "max_tokens", "top_p", "tool_choice", "response_format", "stop", "seed")

_client: TrustedRouter | None = None
_log_lock = threading.Lock()
_call_seq = 0


def _sdk() -> TrustedRouter:
    global _client
    if _client is None:
        key = os.environ.get("TRUSTEDROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        _client = TrustedRouter(api_key=key, base_url=_BASE_URL, max_retries=2)
    return _client


def _write_call_log(*, started_at: str, latency_ms: float, body: dict[str, Any], data: dict[str, Any]) -> None:
    log_dir = os.environ.get("TRBENCH_LITELLM_LOG_DIR")
    if not log_dir:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    global _call_seq
    with _log_lock:
        _call_seq += 1
        call_id = _call_seq
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    record = {
        "call_id": call_id,
        "started_at": started_at,
        "latency_ms": latency_ms,
        "request": {
            "model": body.get("model"),
            "message_count": len(body.get("messages") or []),
            "has_tools": bool(body.get("tools")),
            "tool_count": len(body.get("tools") or []),
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "tool_choice": body.get("tool_choice"),
        },
        "response": {
            "id": data.get("id"),
            "model": data.get("model"),
            "usage": data.get("usage"),
            "finish_reason": choice.get("finish_reason"),
            "has_tool_calls": bool(msg.get("tool_calls")),
            "tool_call_count": len(msg.get("tool_calls") or []),
            "trustedrouter": data.get("trustedrouter"),
        },
    }
    (path / f"call-{call_id:06d}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


class TrustedRouterSDKLLM(CustomLLM):
    """litellm CustomLLM whose completion() delegates to the TrustedRouter SDK."""

    def completion(self, model: str, messages: list, *args: Any, **kwargs: Any) -> ModelResponse:
        model_response: ModelResponse = kwargs.get("model_response") or (args[3] if len(args) > 3 else ModelResponse())
        optional_params: dict = kwargs.get("optional_params") or (args[8] if len(args) > 8 else {})
        # LiteLLM strips the custom provider prefix before invoking the handler:
        # `trustedrouter/socrates-1.1` arrives as `socrates-1.1`, but TR-native
        # combo model ids keep the `trustedrouter/` namespace at the API layer.
        # Vendor-style ids such as `openai/gpt-4.1` and `z-ai/glm-5` already have
        # a slash and should pass through unchanged.
        if "/" not in model:
            model = f"trustedrouter/{model}"

        body: dict[str, Any] = {"model": model, "messages": messages}
        for k in _PASSTHROUGH:
            if optional_params.get(k) is not None:
                body[k] = optional_params[k]
        if optional_params.get("tools"):
            body["tools"] = optional_params["tools"]

        # Direct POST via the SDK client (preserves tool_calls, which the SDK's
        # streaming chat_completions assembly drops).
        started_at = datetime.now(UTC).isoformat()
        started = time.monotonic()
        data = _sdk().request("POST", "/chat/completions", json=body)
        _write_call_log(
            started_at=started_at,
            latency_ms=(time.monotonic() - started) * 1000,
            body=body,
            data=data,
        )

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        message = Message(
            role="assistant",
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls") or None,
        )
        model_response.choices = [
            Choices(index=0, message=message, finish_reason=choice.get("finish_reason") or "stop")
        ]
        model_response.model = model
        usage = data.get("usage") or {}
        model_response.usage = Usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        return model_response


def register() -> None:
    """Register the provider so litellm routes `trustedrouter/<model>` to the SDK.
    Idempotent."""
    if any(entry.get("provider") == _PROVIDER for entry in litellm.custom_provider_map):
        return
    litellm.custom_provider_map = litellm.custom_provider_map + [
        {"provider": _PROVIDER, "custom_handler": TrustedRouterSDKLLM()}
    ]


# Self-register on import so a sitecustomize.py only needs `import trbench.tr_litellm`.
register()
