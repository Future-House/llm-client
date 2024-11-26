import base64
import contextlib
import io
import logging
import logging.config
from collections.abc import Callable
from inspect import iscoroutinefunction, isfunction, signature
from typing import Any

import litellm
import numpy as np
import pymupdf


def encode_image_to_base64(img: "np.ndarray") -> str:
    """Encode an image to a base64 string, to be included as an image_url in a Message."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Image processing requires the 'image' extra for 'Pillow'. Please:"
            " `pip install aviary[image]`."
        ) from e

    image = Image.fromarray(img)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return (
        f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    )


def prepare_args(
    func: Callable, chunk: str, name: str | None = None
) -> tuple[tuple, dict]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (chunk,), {"name": name}
    return (chunk,), {}


def is_coroutine_callable(obj):
    if isfunction(obj):
        return iscoroutinefunction(obj)
    elif callable(obj):  # noqa: RET505
        return iscoroutinefunction(obj.__call__)
    return False


def partial_format(value: str, **formats: dict[str, Any]) -> str:
    """Partially format a string given a variable amount of formats."""
    for template_key, template_value in formats.items():
        with contextlib.suppress(KeyError):
            value = value.format(**{template_key: template_value})
    return value


def setup_default_logs() -> None:
    """Configure logs to reasonable defaults."""
    # Trigger PyMuPDF to use Python logging
    # SEE: https://pymupdf.readthedocs.io/en/latest/app3.html#diagnostics
    pymupdf.set_messages(pylogging=True)

    # Set sane default LiteLLM logging configuration
    # SEE: https://docs.litellm.ai/docs/observability/telemetry
    litellm.telemetry = False

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            # Lower level for verbose logs
            "loggers": {
                "httpcore": {"level": "WARNING"},
                "httpx": {"level": "WARNING"},
                # SEE: https://github.com/BerriAI/litellm/issues/2256
                "LiteLLM": {"level": "WARNING"},
                "LiteLLM Router": {"level": "WARNING"},
                "LiteLLM Proxy": {"level": "WARNING"},
            },
        }
    )
