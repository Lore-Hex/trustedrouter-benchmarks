"""Minimal OpenAI-compatible client for the TrustedRouter gateway.

No SDK dependency on purpose: a benchmark harness should be auditable top to
bottom. Every eval in this repo talks to models through this one function.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.trustedrouter.com/v1"
DEFAULT_MODELS_URL = "https://trustedrouter.com/v1/models"

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


def _post(url: str, *, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        method="POST",
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps(body).encode("utf-8"),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


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
) -> dict[str, Any]:
    """One chat completion. Returns {text, usage, latency_ms} or {error}."""
    started = time.monotonic()
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        body.update(extra_body)
    try:
        data = _post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            body=body,
            timeout=timeout,
        )
        return {
            "text": extract_text(data),
            "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
            "latency_ms": round((time.monotonic() - started) * 1000),
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        return {"error": f"http_{exc.code}: {detail}", "latency_ms": round((time.monotonic() - started) * 1000)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "latency_ms": round((time.monotonic() - started) * 1000)}


def available_models(*, models_url: str = DEFAULT_MODELS_URL, timeout: float = 30.0) -> set[str]:
    req = urllib.request.Request(models_url, headers={"User-Agent": "trbench/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        data = json.loads(response.read().decode("utf-8"))
    return {str(row.get("id")) for row in data.get("data", []) if isinstance(row, dict)}
