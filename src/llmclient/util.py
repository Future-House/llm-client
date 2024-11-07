import base64
import io
import contextlib

from collections.abc import Callable, Iterable
from typing import Any
from inspect import iscoroutinefunction, isfunction, signature

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

async def do_callbacks(
    callbacks: Iterable[Callable[..., Any]],
    chunk: str,
    name: str = None,
) -> None:
    for f in callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        if iscoroutinefunction(f):
            await f(*args, **kwargs)
        else:
            f(*args, **kwargs)

def prepare_args(func: Callable, chunk: str, name: str = None) -> tuple[tuple, dict]:
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
