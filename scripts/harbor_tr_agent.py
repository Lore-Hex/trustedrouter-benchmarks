"""Harbor Terminus-2 agent backed by the TrustedRouter SDK.

Run example:

  TB_TR_CONFIRM=1 PYTHONPATH=scripts uvx --from harbor --with trusted-router-py harbor run \
    -d terminal-bench/terminal-bench-2-1 \
    --agent-import-path harbor_tr_agent:TRHarborTerminus \
    -m trustedrouter/synth \
    -i sqlite-db-truncate \
    --agent-kwarg max_tokens=65536 \
    --jobs-dir runs/harbor

This mirrors scripts/tb_tr_agent.py for the legacy terminal-bench harness, but uses
Harbor's native Terminus-2 scaffold so it can run Terminal-Bench 2.x datasets.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.metric import UsageInfo
from harbor.models.trial.paths import EnvironmentPaths

TR_KEY_PATH = os.environ.get("TB_TR_KEY_PATH", "~/claude/.tr_key")
TR_BASE_URL = os.environ.get("TRUSTEDROUTER_BASE_URL") or None
TERMINUS_CONTROL_KEY_SPEC_TEXT = (
    "Shell commands must end with \\n. Special keys must be sent alone without \\n, "
    'e.g. {"keystrokes":"C-c","duration":0.5}.'
)
TERMINUS_CONTROL_KEY_SPEC_TEMPLATE = TERMINUS_CONTROL_KEY_SPEC_TEXT.replace(
    "{", "{{"
).replace("}", "}}")


def _patch_terminus_prompt_template(template: str) -> str:
    """Add the tmux control-key contract to Harbor's model-facing prompt."""
    if TERMINUS_CONTROL_KEY_SPEC_TEMPLATE in template:
        return template

    for marker in (
        'The "duration" attribute specifies',
        "The `duration` attribute",
        "Task Description:",
    ):
        if marker in template:
            return template.replace(
                marker, TERMINUS_CONTROL_KEY_SPEC_TEMPLATE + "\n\n" + marker, 1
            )

    return template.rstrip() + "\n\n" + TERMINUS_CONTROL_KEY_SPEC_TEMPLATE + "\n"


def _ensure_control_key_spec(prompt: str, messages: list[dict[str, str]]) -> str:
    """Keep the terminal protocol visible after Harbor context rebuilds."""
    if TERMINUS_CONTROL_KEY_SPEC_TEXT in prompt:
        return prompt
    if any(TERMINUS_CONTROL_KEY_SPEC_TEXT in m["content"] for m in messages):
        return prompt
    return prompt.rstrip() + "\n\n" + TERMINUS_CONTROL_KEY_SPEC_TEXT


def _content_to_str(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _usage_from_response(dumped: dict[str, Any]) -> UsageInfo | None:
    usage = dumped.get("usage") or {}
    if not isinstance(usage, dict):
        return None

    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    prompt_details = usage.get("prompt_tokens_details") or {}
    cache_tokens = int(
        prompt_details.get("cached_tokens")
        or usage.get("cache_tokens")
        or usage.get("cached_tokens")
        or 0
    )

    cost_usd = 0.0
    if usage.get("cost_microdollars") is not None:
        cost_usd = float(usage["cost_microdollars"]) / 1_000_000
    elif usage.get("cost_usd") is not None:
        cost_usd = float(usage["cost_usd"])
    elif usage.get("cost") is not None:
        cost_usd = float(usage["cost"])

    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_tokens=cache_tokens,
        cost_usd=cost_usd,
    )


def _message_content(dumped: dict[str, Any]) -> str:
    choice = (dumped.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (message.get("content") or "").strip()


class TrustedRouterHarborLLM(BaseLLM):
    """Harbor BaseLLM implementation backed by TrustedRouter chat completions."""

    def __init__(
        self,
        model: str,
        logs_dir: Path,
        temperature: float | None = None,
        max_tokens: int = 65536,
        context_limit: int = 1_000_000,
        output_limit: int | None = 65536,
        max_empty_retries: int = 2,
        panel_prompt: str | None = None,
        synthesis_prompt: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__()
        if not os.environ.get("TB_TR_CONFIRM"):
            raise RuntimeError(
                "harbor_tr_agent: refusing to call TrustedRouter because it costs money. "
                "Re-run with TB_TR_CONFIRM=1 to proceed."
            )

        from trustedrouter import TrustedRouter

        api_key = os.environ.get("TRUSTEDROUTER_API_KEY")
        if not api_key:
            api_key = Path(os.path.expanduser(TR_KEY_PATH)).read_text().strip()

        self._base_url = base_url or TR_BASE_URL
        self._client = TrustedRouter(api_key=api_key, base_url=self._base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_limit = context_limit
        self._output_limit = output_limit
        self._max_empty_retries = max_empty_retries
        self._panel_prompt = panel_prompt
        self._synthesis_prompt = synthesis_prompt
        self._calls_dir = logs_dir / "trustedrouter-calls"
        self._call_index = 0

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] | None = None,
        logging_path: Path | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = [
            {"role": m.get("role", "user"), "content": _content_to_str(m.get("content"))}
            for m in (message_history or [])
        ]
        prompt = _ensure_control_key_spec(prompt, messages)
        messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        synth_parameters: dict[str, Any] = {}
        if self._panel_prompt:
            synth_parameters["panel_prompt"] = self._panel_prompt
        if self._synthesis_prompt:
            synth_parameters["synthesis_prompt"] = self._synthesis_prompt
        if synth_parameters:
            request["tools"] = [
                {"type": "trustedrouter:synth", "parameters": synth_parameters}
            ]
        if self._temperature is not None:
            request["temperature"] = self._temperature

        last_response: tuple[dict[str, Any], Path, int] | None = None
        for attempt in range(1, self._max_empty_retries + 2):
            self._call_index += 1
            call_id = self._call_index
            started = time.monotonic()
            started_at = datetime.now(UTC).isoformat()
            response_path = self._calls_dir / f"call-{call_id:04d}.response.json"
            meta_path = self._calls_dir / f"call-{call_id:04d}.meta.json"

            try:
                resp = await asyncio.to_thread(self._client.chat_completions, **request)
            except Exception as exc:
                _write_json(
                    meta_path,
                    {
                        "started_at": started_at,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                        "model": self._model,
                        "base_url": self._base_url,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens,
                        "has_panel_prompt": bool(self._panel_prompt),
                        "has_synthesis_prompt": bool(self._synthesis_prompt),
                        "panel_prompt_chars": len(self._panel_prompt or ""),
                        "synthesis_prompt_chars": len(self._synthesis_prompt or ""),
                        "message_count": len(messages),
                        "message_chars": sum(len(m["content"]) for m in messages),
                        "attempt": attempt,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise

            elapsed_ms = round((time.monotonic() - started) * 1000)
            dumped = resp.model_dump(mode="json")
            choice = (dumped.get("choices") or [{}])[0]
            content = _message_content(dumped)
            empty_retry = content == "" and attempt <= self._max_empty_retries

            _write_json(response_path, dumped)
            _write_json(
                meta_path,
                {
                    "started_at": started_at,
                    "elapsed_ms": elapsed_ms,
                    "model": self._model,
                    "base_url": self._base_url,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "has_panel_prompt": bool(self._panel_prompt),
                    "has_synthesis_prompt": bool(self._synthesis_prompt),
                    "panel_prompt_chars": len(self._panel_prompt or ""),
                    "synthesis_prompt_chars": len(self._synthesis_prompt or ""),
                    "message_count": len(messages),
                    "message_chars": sum(len(m["content"]) for m in messages),
                    "attempt": attempt,
                    "response_id": dumped.get("id"),
                    "response_model": dumped.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "content_chars": len(content),
                    "usage": dumped.get("usage"),
                    "trustedrouter": dumped.get("trustedrouter"),
                    "full_response_path": response_path.name,
                    "empty_response_retry": empty_retry,
                    "retry_reason": "empty_content" if empty_retry else None,
                },
            )

            last_response = (dumped, response_path, attempt)
            if not empty_retry:
                break

        if last_response is None:
            raise RuntimeError("TrustedRouter returned no response")

        dumped, response_path, final_attempt = last_response
        choice = (dumped.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = _message_content(dumped)
        reasoning_content = message.get("reasoning_content")
        usage = _usage_from_response(dumped)

        if logging_path is not None:
            _write_json(
                logging_path,
                {
                    "model": self._model,
                    "base_url": self._base_url,
                    "message_count": len(messages),
                    "message_chars": sum(len(m["content"]) for m in messages),
                    "final_attempt": final_attempt,
                    "has_panel_prompt": bool(self._panel_prompt),
                    "has_synthesis_prompt": bool(self._synthesis_prompt),
                    "panel_prompt_chars": len(self._panel_prompt or ""),
                    "synthesis_prompt_chars": len(self._synthesis_prompt or ""),
                    "response_id": dumped.get("id"),
                    "response_model": dumped.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "content_chars": len(content),
                    "usage": dumped.get("usage"),
                    "trustedrouter": dumped.get("trustedrouter"),
                    "full_response_path": str(response_path),
                },
            )

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model_name=dumped.get("model") or self._model,
            usage=usage,
            response_id=dumped.get("id"),
        )

    def get_model_context_limit(self) -> int:
        return self._context_limit

    def get_model_output_limit(self) -> int | None:
        return self._output_limit


class DirectAptTmuxSession(TmuxSession):
    """TmuxSession that avoids apt-get update when package lists are already present."""

    def _get_combined_install_command(
        self, system_info: dict[str, Any], tools: list[str]
    ) -> str:
        package_manager = system_info.get("package_manager")
        if package_manager != "apt-get":
            return super()._get_combined_install_command(system_info, tools)

        packages = " ".join(tools)
        direct_install = (
            "DEBIAN_FRONTEND=noninteractive "
            f"apt-get install -y --no-install-recommends {packages}"
        )
        bounded_update = (
            "DEBIAN_FRONTEND=noninteractive "
            "apt-get update "
            "-o Acquire::Retries=2 "
            "-o Acquire::http::Timeout=30 "
            "-o Acquire::https::Timeout=30"
        )
        return f"({direct_install}) || ({bounded_update} && {direct_install})"


class TRHarborTerminus(Terminus2):
    """Harbor Terminus-2 with the LLM call routed through TrustedRouter."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_tokens: int = 65536,
        context_limit: int = 1_000_000,
        output_limit: int | None = 65536,
        max_empty_retries: int = 2,
        panel_prompt: str | None = None,
        synthesis_prompt: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        if model_name is None:
            raise ValueError("model_name is required")
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._prompt_template = _patch_terminus_prompt_template(self._prompt_template)
        self._llm = TrustedRouterHarborLLM(
            model=model_name,
            logs_dir=logs_dir,
            temperature=self._temperature,
            max_tokens=int(max_tokens),
            context_limit=int(context_limit),
            output_limit=int(output_limit) if output_limit is not None else None,
            max_empty_retries=int(max_empty_retries),
            panel_prompt=panel_prompt,
            synthesis_prompt=synthesis_prompt,
            base_url=base_url,
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        if self._record_terminal_session:
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None

        self._session = DirectAptTmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    @staticmethod
    def name() -> str:
        return "tr-harbor-terminus"
