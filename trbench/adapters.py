"""Model-specific request adapters for benchmark runs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelAdapter:
    max_tokens: int | None = None
    timeout: float | None = None
    temperature: float | None = 0.0
    extra_body: dict[str, Any] | None = None
    use_request: bool = False


def load_adapter_file(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"adapter file {path} has no object-valued 'models'")
    return {str(model): spec for model, spec in models.items() if isinstance(spec, dict)}


def adapter_for(
    model: str,
    specs: dict[str, dict[str, Any]],
    *,
    default_max_tokens: int,
    default_timeout: float,
    default_temperature: float | None = 0.0,
    base_extra_body: dict[str, Any] | None = None,
) -> ModelAdapter:
    spec = specs.get(model, {})
    extra_body: dict[str, Any] = dict(base_extra_body or {})
    spec_extra = spec.get("extra_body")
    if isinstance(spec_extra, dict):
        extra_body.update(spec_extra)
    provider = spec.get("provider")
    if isinstance(provider, str) and provider and "provider" not in extra_body:
        extra_body["provider"] = {"only": [provider]}

    omit_temperature = bool(spec.get("omit_temperature"))
    temperature: float | None
    if omit_temperature:
        temperature = None
    else:
        raw_temperature = spec.get("temperature", default_temperature)
        temperature = float(raw_temperature) if raw_temperature is not None else None

    return ModelAdapter(
        max_tokens=int(spec.get("max_tokens", default_max_tokens)),
        timeout=float(spec.get("timeout", default_timeout)),
        temperature=temperature,
        extra_body=extra_body or None,
        use_request=bool(spec.get("use_request", False)),
    )


def public_adapter_settings(model: str, specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    spec = specs.get(model, {})
    keys = ("max_tokens", "timeout", "omit_temperature", "temperature", "extra_body", "provider", "use_request")
    return {key: spec[key] for key in keys if key in spec}
