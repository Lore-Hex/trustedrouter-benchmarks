"""Terminal-Bench agent that drives a Gemini model through the **Antigravity CLI (`agy`)** — the
official replacement for the now-deprecated consumer gemini-cli (Google killed the free individual
OAuth tier; `agy` uses a fresh Google sign-in).

Like `tb_haiku_agent.py`, this reuses Terminal-Bench's real Terminus-2 scaffold + official grader
and only swaps the LLM backend: each turn shells to `agy -p --model <gemini-id>` on the HOST. agy
auths on the host (OAuth), and tasks run in containers, so the model call must be host-side —
Terminus drives the container over tmux, the completion comes from host `agy`.

⚠️ One-time setup: run `agy` (no args) in a terminal and complete the browser sign-in first;
`agy models` should then list models. ⚠️ Faithfulness: this is "our harness" (agy's own agent
runtime wraps the model), not gemini-cli-as-agent — reported as such.

Run:
  PYTHONPATH=../scripts uv run tb run -d terminal-bench-core==0.1.1 \
    --agent-import-path tb_agy_agent:AgyTerminus -m gemini-3.1-pro \
    -t hello-world --no-livestream --output-path runs/_agy_smoke
"""
import os
import subprocess
import tempfile
from typing import Any

from terminal_bench.agents.terminus_2.terminus_2 import Terminus2
from terminal_bench.llms.base_llm import BaseLLM

AGY_BIN = os.environ.get("TB_AGY_BIN", "agy")
NEUTRAL_CWD = os.environ.get("TB_AGY_CWD", tempfile.gettempdir())
CALL_TIMEOUT = int(os.environ.get("TB_AGY_TIMEOUT", "240"))


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
    """BaseLLM backed by `agy -p` (Antigravity CLI), pinned to a Gemini model."""

    def __init__(self, model: str = "gemini-3.1-pro", temperature: float = 0.7, **kwargs):
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
        # agy -p: single non-interactive prompt; run from a neutral empty cwd (no project) so it
        # has nothing to act on and just answers. No --continue => each turn is stateless.
        cmd = [AGY_BIN, "-p", full, "--model", self._model]
        proc = subprocess.run(
            cmd, cwd=NEUTRAL_CWD, capture_output=True, text=True, timeout=CALL_TIMEOUT
        )
        out = (proc.stdout or "").strip()
        if not out:
            raise RuntimeError(
                f"agy -p returned empty output (rc={proc.returncode}): {(proc.stderr or '')[:500]}"
            )
        if logging_path is not None:
            try:
                logging_path.write_text(out)
            except Exception:
                pass
        return out

    def count_tokens(self, messages: list[dict]) -> int:
        return sum(len(_content_to_str(m.get("content"))) for m in messages) // 4


class AgyTerminus(Terminus2):
    """Terminus-2 with the LLM driven by the Antigravity CLI (`agy -p`)."""

    def __init__(self, model_name: str = "gemini-3.1-pro", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._llm = AgyLLM(model=model_name, temperature=self._llm._temperature)

    @staticmethod
    def name() -> str:
        return "agy-terminus"

    def _count_total_tokens(self, chat) -> int:
        return self._llm.count_tokens(chat._messages)
