"""LiteLLM routing shim for tau2-bench, auto-loaded at interpreter startup.

tau2-bench hardcodes its evaluator/default model names to OpenAI's dated string
`gpt-4.1-2025-04-14` (e.g. the NL-assertion judge in evaluator_nl_assertions.py),
with no CLI/env override. Sent to the TrustedRouter gateway that name is rejected
("Model does not support chat completions: gpt-4.1-2025-04-14") and the whole
simulation aborts as an infrastructure_error — silently dropping every task that
has an NL assertion.

This file is placed on PYTHONPATH by trbench/evals/tau2/run.py for the tau2
subprocess only. Python imports `sitecustomize` automatically at startup, so by
the time tau2 makes a call, LiteLLM's alias map already rewrites the dated names
to a TR-routable model (default `openai/openai/gpt-4.1`, overridable via
TRBENCH_TAU2_EVAL_LLM). trbench itself never imports litellm, so this is inert
outside the tau2 subprocess.
"""
import os

try:
    import litellm

    _eval = os.environ.get("TRBENCH_TAU2_EVAL_LLM", "openai/openai/gpt-4.1")
    litellm.model_alias_map = {
        **getattr(litellm, "model_alias_map", {}),
        "gpt-4.1-2025-04-14": _eval,
        "gpt-4.1": _eval,
    }
except Exception:  # never let the shim break the run
    pass
