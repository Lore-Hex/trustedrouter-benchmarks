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

import os
from typing import Any

import litellm
from litellm import CustomLLM
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from trustedrouter import TrustedRouter

_BASE_URL = os.environ.get("TRUSTEDROUTER_BASE_URL", "https://api.trustedrouter.com/v1")
_PROVIDER = "trustedrouter"
_PASSTHROUGH = ("temperature", "max_tokens", "top_p", "tool_choice", "response_format", "stop", "seed")

_client: TrustedRouter | None = None


def _sdk() -> TrustedRouter:
    global _client
    if _client is None:
        key = os.environ.get("TRUSTEDROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        _client = TrustedRouter(api_key=key, base_url=_BASE_URL, max_retries=2)
    return _client


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _provider_filter() -> dict[str, list[str]] | None:
    provider = os.environ.get("TRBENCH_PROVIDER")
    if not provider:
        return None
    providers = [p.strip() for p in provider.split(",") if p.strip()]
    return {"only": providers} if providers else None


def _api_model_name(model: str) -> str:
    # LiteLLM strips the custom-provider prefix before calling completion().
    # Normal catalog models still contain their org/model slash after stripping,
    # but the Synth endpoint is itself named `trustedrouter/synth`.
    return "trustedrouter/synth" if model == "synth" else model


class TrustedRouterSDKLLM(CustomLLM):
    """litellm CustomLLM whose completion() delegates to the TrustedRouter SDK."""

    def completion(self, model: str, messages: list, *args: Any, **kwargs: Any) -> ModelResponse:
        model_response: ModelResponse = kwargs.get("model_response") or (args[3] if len(args) > 3 else ModelResponse())
        optional_params: dict = kwargs.get("optional_params") or (args[8] if len(args) > 8 else {})

        body: dict[str, Any] = {"model": _api_model_name(model), "messages": messages}
        for k in _PASSTHROUGH:
            if optional_params.get(k) is not None:
                body[k] = optional_params[k]
        max_tokens = _env_int("TRBENCH_MAX_TOKENS")
        if max_tokens is not None and "max_tokens" not in body:
            body["max_tokens"] = max_tokens
        temperature = _env_float("TRBENCH_TEMPERATURE")
        if temperature is not None:
            body["temperature"] = temperature
        if optional_params.get("tools"):
            body["tools"] = optional_params["tools"]
        provider = _provider_filter()
        if provider and "provider" not in body:
            body["provider"] = provider

        # Direct POST via the SDK client (preserves tool_calls, which the SDK's
        # streaming chat_completions assembly drops).
        data = _sdk().request("POST", "/chat/completions", json=body)

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
