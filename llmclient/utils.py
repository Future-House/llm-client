import contextlib
from collections.abc import Callable
from inspect import signature
from typing import Any


def get_litellm_retrying_config(timeout: float = 60.0) -> dict[str, Any]:
    """Get retrying configuration for litellm.acompletion and litellm.aembedding."""
    return {"num_retries": 3, "timeout": timeout}


def prepare_args(
    func: Callable, chunk: str, name: str | None = None
) -> tuple[tuple, dict]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (chunk,), {"name": name}
    return (chunk,), {}


def partial_format(value: str, **formats: dict[str, Any]) -> str:
    """Partially format a string given a variable amount of formats."""
    for template_key, template_value in formats.items():
        with contextlib.suppress(KeyError):
            value = value.format(**{template_key: template_value})
    return value
