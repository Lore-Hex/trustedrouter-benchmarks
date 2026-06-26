"""Harbor (Terminal-Bench 2.x) agent: drives Haiku 4.5 through the free Claude-Code subscription
(`claude -p`) on the HOST, using Harbor's real terminus-2 scaffold. Free path (no API $) — the
Harbor port of `scripts/tb_haiku_agent.py` (and twin of `scripts/harbor_agy_agent.py`).

Override is just `Terminus2._init_llm` → a `claude -p`-backed BaseLLM. Each turn shells to
`claude -p --model claude-haiku-4-5 --max-turns 1 --output-format text` with a system-prompt
override (strips Claude Code's default prompt), no tools, single turn, neutral cwd.

Run:
  PYTHONPATH=<repo>/scripts harbor run -d terminal-bench/terminal-bench-2 \
    --agent-import-path harbor_haiku_agent:HaikuTerminus2 -m claude-haiku-4-5 \
    -i terminal-bench/<task> -n 1
"""
import asyncio
import os
import subprocess
import tempfile
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, LLMResponse

CLAUDE_BIN = os.environ.get("TB_CLAUDE_BIN", "claude")
NEUTRAL_CWD = os.environ.get("TB_CLAUDE_CWD", tempfile.gettempdir())
CALL_TIMEOUT = int(os.environ.get("TB_CLAUDE_TIMEOUT", "600"))
SYSTEM_PROMPT = (
    "You are an autonomous agent operating a Linux terminal to accomplish a task. "
    "Follow the user's instructions exactly and respond ONLY in the requested format. "
    "Output the raw response only — do NOT wrap it in markdown code fences and add no "
    "explanatory prose before or after it."
)


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


class HaikuCliLLM(BaseLLM):
    """BaseLLM backed by the free `claude -p` subscription, pinned to Haiku 4.5."""

    def __init__(self, model: str = "claude-haiku-4-5", **kwargs):
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
                [CLAUDE_BIN, "-p", full, "--model", self._model, "--max-turns", "1",
                 "--output-format", "text", "--system-prompt", SYSTEM_PROMPT,
                 "--allowed-tools", "", "--no-session-persistence"],
                cwd=NEUTRAL_CWD, capture_output=True, text=True, timeout=CALL_TIMEOUT,
            )

        # `claude -p` occasionally returns empty stdout (rc=0) on a transient hiccup — retry a few
        # times with backoff before giving up, so one blank turn doesn't error the whole task.
        last_stderr = ""
        for attempt in range(4):
            proc = await asyncio.to_thread(_run)
            out = (proc.stdout or "").strip()
            if out:
                return LLMResponse(content=out, model_name=self._model)
            last_stderr = (proc.stderr or "")[:300]
            await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"claude -p empty output after 4 tries (rc={proc.returncode}): {last_stderr}")

    def get_model_context_limit(self) -> int:
        return 200_000

    def get_model_output_limit(self) -> int | None:
        return 8192


class HaikuTerminus2(Terminus2):
    """Harbor terminus-2 with the LLM routed through the free `claude -p` (Haiku 4.5)."""

    def _init_llm(self, model_name: str | None = None, **kwargs) -> BaseLLM:
        return HaikuCliLLM(model=model_name or "claude-haiku-4-5")
