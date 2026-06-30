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

# With --use-sdk, the runner names models `trustedrouter/<id>`; register the
# litellm custom provider so those route through the TrustedRouter SDK. Inert if
# trbench / the SDK aren't importable (e.g. the openai/ base-url path).
try:
    import trbench.tr_litellm  # noqa: F401  - self-registers the provider
except Exception:
    pass

# The tau3 banking_knowledge `alltools` retrieval path calls OpenAI embeddings
# directly. TR's OpenAI-compatible embeddings endpoint requires provider-
# qualified embedding ids such as `openai/text-embedding-3-large`; upstream
# defaults to the bare OpenAI id. Patch only the tau2 subprocess when requested.
try:
    _embedding_model = os.environ.get("TRBENCH_TAU2_OPENAI_EMBEDDING_MODEL")
    if _embedding_model:
        from tau2.domains.banking_knowledge import retrieval as _retrieval
        from tau2.knowledge import embeddings_cache as _embeddings_cache
        from tau2.knowledge.embedders import openai_embedder as _openai_embedder

        _retrieval.DEFAULT_DENSE_EMBEDDING_MODEL_OPENAI = _embedding_model
        for _variant in _retrieval.RETRIEVAL_VARIANTS.values():
            for _spec in (
                getattr(_variant, "kb_search", None),
                getattr(_variant, "kb_search_dense", None),
            ):
                if (
                    getattr(_spec, "embedder_type", None) == "openai"
                    and getattr(_spec, "embedder_model", None) == "text-embedding-3-large"
                ):
                    _spec.embedder_model = _embedding_model

        _orig_embedder_configs = _embeddings_cache.get_unique_embedder_configs_for_retrieval_configs

        def _tr_embedder_configs(retrieval_config_names, retrieval_config_kwargs=None):
            configs = _orig_embedder_configs(retrieval_config_names, retrieval_config_kwargs)
            patched = []
            for embedder_type, params in configs:
                params = dict(params)
                if embedder_type == "openai" and params.get("model") == "text-embedding-3-large":
                    params["model"] = _embedding_model
                patched.append((embedder_type, params))
            return patched

        _embeddings_cache.get_unique_embedder_configs_for_retrieval_configs = _tr_embedder_configs

        _orig_openai_embedder_init = _openai_embedder.OpenAIEmbedder.__init__

        def _tr_openai_embedder_init(self, model="text-embedding-ada-002", api_key=None):
            if model == "text-embedding-3-large":
                model = _embedding_model
            _orig_openai_embedder_init(self, model=model, api_key=api_key)

        _openai_embedder.OpenAIEmbedder.__init__ = _tr_openai_embedder_init
except Exception:
    pass
