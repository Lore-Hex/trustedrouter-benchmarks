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
    # Qwen (Alibaba) — a top Chinese open-weight lab; modern flagships.
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-coder-next",
)

FRONTIER_REFS: tuple[str, ...] = (
    "anthropic/claude-opus-4.8",
    "google/gemini-3.1-pro-preview",
    "openai/gpt-5.5",
)

# Cheap, well-documented Western model used to CALIBRATE each eval against its
# published number. The frontier refs above are too new to have stable published
# per-eval scores; gemini-2.5-flash does, so it's the standing calibration anchor
# (it's also the judge for the factuality evals). Not part of the scored panel.
CALIBRATION_REF: str = "google/gemini-2.5-flash"

# Stable open-weight checkpoints (+ gpt-4o family) whose IFEval / MMLU-Pro / GSM8K
# numbers are reported in their model cards / tech reports. Because the exact
# weights are pinned, these are the cleanest anchors for confirming the harness
# reproduces published scores — unlike the frontier/Chinese flagships, which drift
# and mostly don't publish standardized per-eval numbers. Run with
# `--models "$(python -c 'from trbench.panel import CALIBRATION_ANCHORS as A; print(",".join(A))')"`.
CALIBRATION_ANCHORS: tuple[str, ...] = (
    "meta-llama/llama-3.1-8b-instruct",     # IFEval ~80, MMLU-Pro ~48, GSM8K ~84
    "meta-llama/llama-3.1-70b-instruct",    # IFEval ~87, MMLU-Pro ~66, GSM8K ~95
    "meta-llama/llama-3.3-70b-instruct",    # IFEval ~92, MMLU-Pro ~69, GSM8K ~95
    "qwen/qwen-2.5-72b-instruct",           # IFEval ~84, MMLU-Pro ~58-71, GSM8K ~96
    "qwen/qwen-2.5-7b-instruct",            # IFEval ~75, MMLU-Pro ~44, GSM8K ~85
    "openai/gpt-4o",                        # IFEval ~87, MMLU-Pro ~73, GSM8K ~96
    "openai/gpt-4o-mini",                   # IFEval ~86, MMLU-Pro ~63, GSM8K ~93
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
