"""The retry-until-complete behavior in client.chat: transient errors are
retried, client errors are not, and a recovered route returns a real answer.
No network — client._post is monkeypatched."""
from __future__ import annotations

import io
import urllib.error

from trbench import client


def _http_error(code: int, body: bytes = b"provider error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "err", {}, io.BytesIO(body))


def _ok():
    return {"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 3}}


def _call(monkeypatch, side_effects, **kw):
    calls = {"n": 0}

    def fake_post(*_a, **_k):
        i = calls["n"]
        calls["n"] += 1
        eff = side_effects[min(i, len(side_effects) - 1)]
        if isinstance(eff, Exception):
            raise eff
        return eff

    monkeypatch.setattr(client, "_post", fake_post)
    result = client.chat(
        base_url="http://x/v1", api_key="k", model="m",
        messages=[{"role": "user", "content": "hi"}],
        retry_backoff=0.0, **kw,
    )
    return result, calls["n"]


def test_retries_transient_502_then_succeeds(monkeypatch):
    result, n = _call(monkeypatch, [_http_error(502), _http_error(502), _ok()], retries=4)
    assert result.get("text") == "hello"
    assert result["attempts"] == 3
    assert n == 3


def test_persistent_502_gives_up_after_retries(monkeypatch):
    result, n = _call(monkeypatch, [_http_error(502)], retries=3)
    assert "http_502" in result.get("error", "")
    assert n == 4  # 1 initial + 3 retries


def test_429_is_retried(monkeypatch):
    result, n = _call(monkeypatch, [_http_error(429), _ok()], retries=4)
    assert result.get("text") == "hello"
    assert n == 2


def test_400_is_not_retried(monkeypatch):
    result, n = _call(monkeypatch, [_http_error(400)], retries=4)
    assert "http_400" in result.get("error", "")
    assert n == 1  # client error: no retry


def test_timeout_is_retried(monkeypatch):
    result, n = _call(monkeypatch, [TimeoutError("timed out"), _ok()], retries=4)
    assert result.get("text") == "hello"
    assert n == 2


def test_unexpected_exception_not_retried(monkeypatch):
    result, n = _call(monkeypatch, [ValueError("bug")], retries=4)
    assert "ValueError" in result.get("error", "")
    assert n == 1
