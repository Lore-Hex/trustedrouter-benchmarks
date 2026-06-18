"""The retry policy in client.chat (now SDK-backed): transient failures
(HTTP 5xx / 429 / transport errors with no status_code) are retried; client
errors (4xx other than 429) are not. No network — the SDK client is monkeypatched."""
from __future__ import annotations

from trbench import client


class _SDKError(Exception):
    """Stand-in for a TrustedRouter SDK error, which carries .status_code."""

    def __init__(self, status_code: int, message: str = "err") -> None:
        super().__init__(message)
        self.status_code = status_code


class _Resp:
    def __init__(self, content: str) -> None:
        self._content = content

    def model_dump(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}], "usage": {"total_tokens": 3}}


def _run(monkeypatch, side_effects, **kw):
    calls = {"n": 0}

    class FakeClient:
        def chat_completions(self, **_kw):
            i = calls["n"]
            calls["n"] += 1
            eff = side_effects[min(i, len(side_effects) - 1)]
            if isinstance(eff, Exception):
                raise eff
            return eff

    monkeypatch.setattr(client, "_sdk_client", lambda *a, **k: FakeClient())
    result = client.chat(
        base_url="http://x/v1", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}], retry_backoff=0.0, **kw,
    )
    return result, calls["n"]


def test_retries_transient_502_then_succeeds(monkeypatch):
    result, n = _run(monkeypatch, [_SDKError(502), _SDKError(502), _Resp("hello")], retries=4)
    assert result.get("text") == "hello"
    assert result["attempts"] == 3
    assert n == 3


def test_persistent_502_gives_up_after_retries(monkeypatch):
    result, n = _run(monkeypatch, [_SDKError(502)], retries=3)
    assert "http_502" in result.get("error", "")
    assert n == 4  # 1 initial + 3 retries


def test_429_is_retried(monkeypatch):
    result, n = _run(monkeypatch, [_SDKError(429), _Resp("hello")], retries=4)
    assert result.get("text") == "hello"
    assert n == 2


def test_400_is_not_retried(monkeypatch):
    result, n = _run(monkeypatch, [_SDKError(400)], retries=4)
    assert "http_400" in result.get("error", "")
    assert n == 1  # client error: no retry


def test_transport_error_without_status_is_retried(monkeypatch):
    # e.g. an httpx timeout/connection drop surfaces with no status_code -> transient -> retry.
    result, n = _run(monkeypatch, [TimeoutError("timed out"), _Resp("hello")], retries=4)
    assert result.get("text") == "hello"
    assert n == 2
