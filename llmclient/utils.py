import asyncio
import contextlib
import logging
import logging.config
import os
from collections.abc import Awaitable, Callable, Iterable
from inspect import signature
from typing import Any, TypeVar

import litellm


def configure_llm_logs() -> None:
    """Configure log levels."""
    # Set sane default LiteLLM logging configuration
    # SEE: https://docs.litellm.ai/docs/observability/telemetry
    litellm.telemetry = False
    if (
        logging.getLevelNamesMapping().get(
            os.environ.get("LITELLM_LOG", ""), logging.WARNING
        )
        < logging.WARNING
    ):
        # If LITELLM_LOG is DEBUG or INFO, don't change the LiteLLM log levels
        litellm_loggers_config: dict[str, Any] = {}
    else:
        litellm_loggers_config = {
            "LiteLLM": {"level": "WARNING"},
            "LiteLLM Proxy": {"level": "WARNING"},
            "LiteLLM Router": {"level": "WARNING"},
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            # Lower level for httpx and LiteLLM
            "loggers": {"httpx": {"level": "WARNING"}} | litellm_loggers_config,
        }
    )


def get_litellm_retrying_config(timeout: float = 60.0) -> dict[str, Any]:
    """Get retrying configuration for litellm.acompletion and litellm.aembedding."""
    return {"num_retries": 3, "timeout": timeout}


def prepare_args(
    func: Callable, completion: str, name: str | None = None
) -> tuple[tuple, dict]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (completion,), {"name": name}
    return (completion,), {}


def partial_format(value: str, **formats: dict[str, Any]) -> str:
    """Partially format a string given a variable amount of formats."""
    for template_key, template_value in formats.items():
        with contextlib.suppress(KeyError):
            value = value.format(**{template_key: template_value})
    return value


T = TypeVar("T")


async def gather_with_concurrency(
    n: int | asyncio.Semaphore, coros: Iterable[Awaitable[T]]
) -> list[T]:
    """
    Run asyncio.gather with a concurrency limit.

    SEE:  https://stackoverflow.com/a/61478547/2392535
    """
    semaphore = asyncio.Semaphore(n) if isinstance(n, int) else n

    async def sem_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))
