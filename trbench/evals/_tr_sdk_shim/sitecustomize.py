"""Auto-loaded at interpreter startup (Python imports `sitecustomize`
automatically). Placed on an external agentic harness's PYTHONPATH by the eval
runners so that `trustedrouter/<model>` litellm calls route through the real
TrustedRouter SDK (see trbench.tr_litellm). Inert if trbench / litellm / the SDK
aren't importable, so it never breaks a run.
"""
try:
    import trbench.tr_litellm  # noqa: F401  - self-registers the litellm provider
except Exception:  # noqa: S110 - sitecustomize must not break harness startup
    pass

try:
    import os

    import litellm.utils as _litellm_utils

    _original_get_max_tokens = _litellm_utils.get_max_tokens

    def _get_max_tokens_with_tr_context(model: str, *args, **kwargs):
        """Return a usable context budget for TR-routed reasoning models.

        Terminal-Bench Terminus uses litellm's `get_max_tokens()` as a context
        window. For DeepSeek, litellm reports the output cap (8192), not the
        model's input context, which triggers premature Terminus handoff
        summarization. That handoff is expensive and destabilizes the agent loop.
        """

        if model in {
            "trustedrouter/deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-pro",
        }:
            return int(os.environ.get("TRBENCH_CONTEXT_TOKENS", "1000000"))
        return _original_get_max_tokens(model, *args, **kwargs)

    _litellm_utils.get_max_tokens = _get_max_tokens_with_tr_context
except Exception:  # noqa: S110 - context patch is optional best-effort setup
    pass
