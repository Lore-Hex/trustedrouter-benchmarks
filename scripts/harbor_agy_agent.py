"""Harbor (Terminal-Bench 2.x) agent: drives a Gemini model through the **Antigravity CLI `agy`**
on the HOST, using Harbor's real Terminus-2 scaffold. Free path (no API $) — the Harbor port of
`scripts/tb_agy_agent.py`.

Harbor's `terminus-2` builds its LLM via `Terminus2._init_llm(...)`; we override that to return an
`agy -p`-backed `BaseLLM` instead of LiteLLM. Everything else (prompt template, JSON parser, tmux
execution, context management, official verifier) is Harbor's own.

⚠️ Requires a one-time `agy` browser sign-in on the host first (`agy models` should list models).
agy auths on the host, and Harbor tasks run in containers, so the model call must be host-side —
Terminus-2 drives the container, the completion comes from host `agy`.

Run:
  PYTHONPATH=/Users/jperla/claude/trustedrouter-benchmarks/scripts \
  harbor run -d terminal-bench/terminal-bench-2 \
    --agent-import-path harbor_agy_agent:AgyTerminus2 -m gemini-3.1-pro -l 2 -n 1
"""
import asyncio
import os
import subprocess
import tempfile
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, LLMResponse

AGY_BIN = os.environ.get("TB_AGY_BIN", "agy")
NEUTRAL_CWD = os.environ.get("TB_AGY_CWD", tempfile.gettempdir())
CALL_TIMEOUT = int(os.environ.get("TB_AGY_TIMEOUT", "1800"))  # gemini-3.1-pro HIGH reasoning is slow on big prompts


def _content_to_str(c) -> str:
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c or ""


def _flatten(message_history: list[dict[str, Any]], prompt: str) -> str:
    parts = []
    for m in message_history:
        parts.append(f"=== {(m.get('role') or 'user').upper()} ===\n{_content_to_str(m.get('content'))}")
    parts.append(f"=== USER ===\n{prompt}")
    return "\n\n".join(parts)


class AgyLLM(BaseLLM):
    """BaseLLM backed by `agy -p` (Antigravity CLI), pinned to a Gemini model. Free (host OAuth)."""

    def __init__(self, model: str = "gemini-3.1-pro", **kwargs):
        super().__init__()
        self._model = model

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        logging_path=None,
        previous_response_id=None,
        **kwargs,
    ) -> LLMResponse:
        full = _flatten(message_history, prompt)

        def _run():
            return subprocess.run(
                [AGY_BIN, "-p", full, "--model", self._model],
                cwd=NEUTRAL_CWD, capture_output=True, text=True, timeout=CALL_TIMEOUT,
            )

        proc = await asyncio.to_thread(_run)  # don't block the event loop (Harbor runs trials concurrently)
        out = (proc.stdout or "").strip()
        if not out:
            raise RuntimeError(
                f"agy -p returned empty output (rc={proc.returncode}): {(proc.stderr or '')[:400]}"
            )
        return LLMResponse(content=out, model_name=self._model)

    def get_model_context_limit(self) -> int:
        return 1_000_000  # gemini-3.x large context; we never get near it on these tasks

    def get_model_output_limit(self) -> int | None:
        return 65536


class AgyTerminus2(Terminus2):
    """Harbor Terminus-2 with the LLM call routed through the Antigravity CLI (`agy -p`)."""

    def _init_llm(self, model_name: str | None = None, **kwargs) -> BaseLLM:
        return AgyLLM(model=model_name or "gemini-3.1-pro")
