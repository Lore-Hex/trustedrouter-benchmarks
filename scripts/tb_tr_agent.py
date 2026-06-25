"""Terminal-Bench agent whose model calls go through the **TrustedRouter SDK** instead of
litellm. This makes the harness TR-native: any of TR's ~230 models (anthropic/*, google/gemini-*,
the Chinese open-weight panel, etc.) can drive the real Terminus-2 scaffold over a clean API —
the faithful "Terminus-2 over an API" setup, unlike the free `claude -p` path in tb_haiku_agent.py.

It reuses Terminal-Bench's real Terminus-2 scaffold (prompt template, parser, tmux execution,
context management) and the official grader; only the LLM backend is swapped to
`trustedrouter.TrustedRouter().chat_completions(...)`.

⚠️ COST GUARD: TrustedRouter calls bill the operator's account. This module is **dormant** — it is
safe to import/commit and makes NO network call until you opt in with `TB_TR_CONFIRM=1`. Building
the agent (which a real run does) raises a clear error otherwise. (Mirrors scicode_tr_gen.py.)

Run (only with explicit go — costs $):
  TB_TR_CONFIRM=1 PYTHONPATH=../scripts uv run tb run -d terminal-bench-core==0.1.1 \
    --agent-import-path tb_tr_agent:TRTerminus -m google/gemini-3.1-pro \
    -t hello-world --no-livestream --output-path runs/_tr
"""
import os
from pathlib import Path
from typing import Any

from terminal_bench.agents.terminus_2.terminus_2 import Terminus2
from terminal_bench.llms.base_llm import BaseLLM

TR_KEY_PATH = os.environ.get("TB_TR_KEY_PATH", "~/claude/.tr_key")


def _content_to_str(c) -> str:
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c or ""


class TrustedRouterLLM(BaseLLM):
    """BaseLLM backed by the TrustedRouter SDK. Cost-guarded: refuses to construct unless
    TB_TR_CONFIRM=1 (so importing this module never spends money)."""

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 4096, **kwargs):
        super().__init__(**kwargs)
        if not os.environ.get("TB_TR_CONFIRM"):
            raise RuntimeError(
                "tb_tr_agent: refusing to call TrustedRouter (it costs money). "
                "Re-run with TB_TR_CONFIRM=1 to proceed."
            )
        from trustedrouter import TrustedRouter

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = TrustedRouter(api_key=Path(os.path.expanduser(TR_KEY_PATH)).read_text().strip())

    def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        response_format=None,
        logging_path=None,
        **kwargs,
    ) -> str:
        messages = [
            {"role": m.get("role", "user"), "content": _content_to_str(m.get("content"))}
            for m in message_history
        ]
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat_completions(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=messages,
        )
        out = (resp.choices[0].message.content or "").strip()
        if logging_path is not None:
            try:
                logging_path.write_text(out)
            except Exception:
                pass
        return out

    def count_tokens(self, messages: list[dict]) -> int:
        return sum(len(_content_to_str(m.get("content"))) for m in messages) // 4


class TRTerminus(Terminus2):
    """Terminus-2 with the LLM call routed through the TrustedRouter SDK (any TR model id)."""

    def __init__(self, model_name: str, max_tokens: int = 4096, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._llm = TrustedRouterLLM(
            model=model_name, temperature=self._llm._temperature, max_tokens=max_tokens
        )

    @staticmethod
    def name() -> str:
        return "tr-terminus"

    def _count_total_tokens(self, chat) -> int:
        return self._llm.count_tokens(chat._messages)
