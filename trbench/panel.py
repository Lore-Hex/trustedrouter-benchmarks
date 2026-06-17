"""The model panel.

The point of this repo is to run capability evals on the leading Chinese
open-weight models and publish where they actually land. These are TrustedRouter
model ids; prune to what your account can route. A few Western frontier models
are kept as a reference line.
"""
from __future__ import annotations

from collections.abc import Sequence

from trbench import client

CHINESE_PANEL: tuple[str, ...] = (
    "z-ai/glm-5.2",
    "z-ai/glm-5.1",
    "z-ai/glm-5",
    "moonshotai/kimi-k2.7-code",
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v3.2",
    "minimax/minimax-m3",
    "xiaomi/mimo-v2.5-pro",
    "xiaomi/mimo-v2.5",
    "tencent/hy3-preview",
)

FRONTIER_REFS: tuple[str, ...] = (
    "anthropic/claude-opus-4.8",
    "google/gemini-3.1-pro-preview",
    "openai/gpt-5.5",
    "anthropic/claude-haiku-4.5",  # Western budget reference — set against the cheap Chinese models
)

DEFAULT_PANEL: tuple[str, ...] = CHINESE_PANEL + FRONTIER_REFS


def resolve_panel(models_arg: str | None, *, default: Sequence[str] = DEFAULT_PANEL) -> list[str]:
    """Parse a comma list, or return the default panel."""
    if models_arg:
        return [m.strip() for m in models_arg.split(",") if m.strip()]
    return list(default)


def available_subset(models: Sequence[str], *, models_url: str = client.DEFAULT_MODELS_URL) -> list[str]:
    """Keep only the panel models the gateway currently exposes."""
    catalog = client.available_models(models_url=models_url)
    return [m for m in models if m in catalog]
