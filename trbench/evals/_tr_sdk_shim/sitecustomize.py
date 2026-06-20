"""Auto-loaded at interpreter startup (Python imports `sitecustomize`
automatically). Placed on an external agentic harness's PYTHONPATH by the eval
runners so that `trustedrouter/<model>` litellm calls route through the real
TrustedRouter SDK (see trbench.tr_litellm). Inert if trbench / litellm / the SDK
aren't importable, so it never breaks a run.
"""
try:
    import trbench.tr_litellm  # noqa: F401  - self-registers the litellm provider
except Exception:
    pass
