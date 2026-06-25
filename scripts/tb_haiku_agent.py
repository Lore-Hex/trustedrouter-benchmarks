"""Terminal-Bench agent that drives Haiku 4.5 through the FREE Claude-Code subscription
(`claude -p`) instead of a paid API.

It reuses Terminal-Bench's REAL Terminus-2 scaffold (prompt template, JSON/XML parser,
tmux command execution, context management) and the official grader — only the LLM backend
is swapped: each model turn shells out to `claude -p --model claude-haiku-4-5`.

⚠️ Faithfulness caveat: this is NOT the published Terminus-2-over-API setup. `claude -p` wraps
the model in the Claude-Code agent runtime; we strip the default system prompt (--system-prompt
override), disable tools (--allowed-tools ""), force a single assistant turn (--max-turns 1), and
run from a neutral cwd so no CLAUDE.md / auto-memory leaks in — but it is still "our harness," so
the score is reported as such, NOT as Artificial Analysis's 27.3% Terminal-Bench-Hard figure.

Run:
  PYTHONPATH=../scripts uv run tb run -d terminal-bench-core==0.1.1 \
    --agent-import-path tb_haiku_agent:HaikuCliTerminus \
    -t hello-world --no-livestream --no-cleanup --output-path runs/_haiku_smoke
"""
import os
import subprocess
import tempfile
from typing import Any

from terminal_bench.agents.terminus_2.terminus_2 import Terminus2
from terminal_bench.llms.base_llm import BaseLLM

CLAUDE_BIN = os.environ.get("TB_CLAUDE_BIN", "claude")
# Neutral cwd so `claude -p` does not auto-discover this repo's CLAUDE.md / auto-memory.
NEUTRAL_CWD = os.environ.get("TB_CLAUDE_CWD", tempfile.gettempdir())
CALL_TIMEOUT = int(os.environ.get("TB_CLAUDE_TIMEOUT", "180"))
SYSTEM_PROMPT = (
    "You are an autonomous agent operating a Linux terminal to accomplish a task. "
    "Follow the user's instructions exactly and respond ONLY in the requested format. "
    "Output the raw response only — do NOT wrap it in markdown code fences and add no "
    "explanatory prose before or after it."
)


def _flatten(message_history: list[dict[str, Any]], prompt: str) -> str:
    """Render the running chat into a single transcript for `claude -p` (text input is
    single-shot, so we replay the whole conversation each turn)."""
    parts = []
    for m in message_history:
        role = (m.get("role") or "user").upper()
        content = m.get("content")
        if isinstance(content, list):  # litellm content-part form
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        parts.append(f"=== {role} ===\n{content}")
    parts.append(f"=== USER ===\n{prompt}")
    return "\n\n".join(parts)


class ClaudeCliLLM(BaseLLM):
    """BaseLLM backed by the free `claude -p` subscription, pinned to Haiku 4.5."""

    def __init__(self, model: str = "claude-haiku-4-5", temperature: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._temperature = temperature

    def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        response_format=None,
        logging_path=None,
        **kwargs,
    ) -> str:
        full = _flatten(message_history, prompt)
        cmd = [
            CLAUDE_BIN,
            "-p",
            full,
            "--model", self._model,
            "--max-turns", "1",
            "--output-format", "text",
            "--system-prompt", SYSTEM_PROMPT,
            "--allowed-tools", "",
            "--no-session-persistence",
        ]
        proc = subprocess.run(
            cmd,
            cwd=NEUTRAL_CWD,
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT,
        )
        out = (proc.stdout or "").strip()
        if not out:
            # surface stderr so harness logs show why a turn was empty
            raise RuntimeError(
                f"claude -p returned empty output (rc={proc.returncode}): "
                f"{(proc.stderr or '')[:500]}"
            )
        if logging_path is not None:
            try:
                logging_path.write_text(out)
            except Exception:
                pass
        return out

    def count_tokens(self, messages: list[dict]) -> int:
        # Heuristic (~4 chars/token); avoids litellm model lookups for an id it may not know.
        total = 0
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                c = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in c
                )
            total += len(c or "")
        return total // 4


class HaikuCliTerminus(Terminus2):
    """Terminus-2, but the LLM is Haiku 4.5 via the free `claude -p` subscription."""

    def __init__(self, model_name: str = "claude-haiku-4-5", **kwargs):
        # Build the real Terminus-2 (parser, prompt template, etc.), then swap the backend.
        super().__init__(model_name=model_name, **kwargs)
        self._llm = ClaudeCliLLM(model=model_name, temperature=self._llm._temperature)

    @staticmethod
    def name() -> str:
        return "haiku-cli-terminus"

    def _count_total_tokens(self, chat) -> int:
        # Use our heuristic instead of litellm.token_counter on an unknown model id.
        return self._llm.count_tokens(chat._messages)
