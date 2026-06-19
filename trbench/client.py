"""TrustedRouter gateway client for the benchmarks.

Every gateway (inference) call goes through the official TrustedRouter Python
SDK (`trusted-router-py`) — the same shipped client the SDKs and PrometheusBench
use — so the harness exercises the real client path. The thin `chat()` wrapper
keeps a stable {text, usage} shape for the scorers and owns the retry policy:
the SDK is created with max_retries=0 and this loop retries transient failures
(HTTP 5xx / 429, timeouts, connection drops) with capped exponential backoff, so
a single flaky provider route doesn't silently drop a row. (The SDK's own retry
only covers 502/503/504 via regional failover and skips 429, so we keep our own.)
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

from trustedrouter import TrustedRouter

DEFAULT_BASE_URL = "https://api.trustedrouter.com/v1"
DEFAULT_MODELS_URL = "https://trustedrouter.com/v1/models"
DEFAULT_RETRIES = 4

_API_KEY_ENV = (
    "TRUSTEDROUTER_API_KEY",
    "TRBENCH_API_KEY",
    "PROMETHEUSBENCH_API_KEY",
)


def api_key_from_env(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for name in _API_KEY_ENV:
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit(
        "Missing API key. Set TRUSTEDROUTER_API_KEY (or pass --api-key). "
        "Use a disposable key: capability prompts still route to upstream labs."
    )


def extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [c["text"] for c in content if isinstance(c, dict) and isinstance(c.get("text"), str)]
                    return "\n".join(parts)
            if isinstance(first.get("text"), str):
                return first["text"]
    return ""


# One SDK client per (base_url, api_key, timeout); httpx.Client is safe to share
# across the worker threads, and reuse keeps connection pooling.
_clients: dict[tuple[str, str, float], TrustedRouter] = {}


def _sdk_client(base_url: str, api_key: str, timeout: float) -> TrustedRouter:
    key = (base_url, api_key, timeout)
    cli = _clients.get(key)
    if cli is None:
        # max_retries=0: this module's chat() loop owns retry (incl. 429, which
        # the SDK doesn't retry) so behavior is identical across the panel.
        cli = TrustedRouter(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        _clients[key] = cli
    return cli


def chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 1536,
    temperature: float = 0.0,
    timeout: float = 120.0,
    extra_body: dict[str, Any] | None = None,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = 1.5,
) -> dict[str, Any]:
    """One chat completion via the TrustedRouter SDK. Returns {text, usage,
    latency_ms} or {error}. Transient failures retry; 4xx (other than 429) don't."""
    started = time.monotonic()
    params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    if extra_body:
        params.update(extra_body)
    # Pin the upstream provider for the whole run via env (e.g. route every
    # request through the confidential TEE tier: TRBENCH_PROVIDER=tinfoil). Sent
    # as the OpenRouter-style `provider.only` filter; a model the provider doesn't
    # serve then 400s "no route candidates" rather than silently using another.
    provider = os.environ.get("TRBENCH_PROVIDER")
    if provider and "provider" not in params:
        params["provider"] = {"only": [p.strip() for p in provider.split(",") if p.strip()]}
    cli = _sdk_client(base_url, api_key, timeout)

    last_error = "unknown"
    for attempt in range(retries + 1):
        try:
            data = cli.chat_completions(model=model, messages=messages, **params).model_dump()
            return {
                "text": extract_text(data),
                "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
                "latency_ms": round((time.monotonic() - started) * 1000),
                "attempts": attempt + 1,
            }
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "status_code", None)
            last_error = (f"http_{code}: " if isinstance(code, int) else f"{type(exc).__name__}: ") + str(exc)[:600]
            if isinstance(code, int):
                retryable = code >= 500 or code == 429  # server / rate-limit: retry; other 4xx: don't
            else:
                retryable = True  # transport / timeout / connection drop: retry
        if retryable and attempt < retries:
            time.sleep(min(retry_backoff * (2 ** attempt), 8.0))
            continue
        return {"error": last_error, "latency_ms": round((time.monotonic() - started) * 1000)}
    return {"error": last_error, "latency_ms": round((time.monotonic() - started) * 1000)}


def available_models(*, models_url: str = DEFAULT_MODELS_URL, timeout: float = 30.0) -> set[str]:
    """The live TrustedRouter catalog (a plain GET against the public catalog URL,
    a different host than the inference base_url; not a gateway/inference call)."""
    req = urllib.request.Request(models_url, headers={"User-Agent": "trbench/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        data = json.loads(response.read().decode("utf-8"))
    return {str(row.get("id")) for row in data.get("data", []) if isinstance(row, dict)}
