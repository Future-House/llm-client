import contextvars
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

import litellm

logger = logging.getLogger(__name__)


TRACK_COSTS = contextvars.ContextVar[bool]("track_costs", default=False)
REPORT_EVERY_USD = 1.0


def set_reporting_frequency(frequency: float):
    global REPORT_EVERY_USD  # noqa: PLW0603
    REPORT_EVERY_USD = frequency


def track_costs_global(enabled: bool = True):
    TRACK_COSTS.set(enabled)


@asynccontextmanager
async def track_costs_ctx(enabled: bool = True):
    prev = TRACK_COSTS.get()
    TRACK_COSTS.set(enabled)
    try:
        yield
    finally:
        TRACK_COSTS.set(prev)


class CostTracker:
    def __init__(self):
        self.lifetime_cost_usd = 0.0
        self.last_report = 0.0

    def record(self, response: litellm.ModelResponse):
        self.lifetime_cost_usd += litellm.cost_calculator.completion_cost(
            completion_response=response
        )

        if self.lifetime_cost_usd - self.last_report > REPORT_EVERY_USD:
            logger.info(
                f"Cumulative llmclient API call cost: ${self.lifetime_cost_usd:.8f}"
            )
            self.last_report = self.lifetime_cost_usd


GLOBAL_COST_TRACKER = CostTracker()


TReturn = TypeVar("TReturn", bound=Awaitable)
TParams = ParamSpec("TParams")


def track_costs(
    func: Callable[TParams, TReturn],
) -> Callable[TParams, TReturn]:
    async def wrapped_func(*args, **kwargs):
        response = await func(*args, **kwargs)
        if TRACK_COSTS.get():
            GLOBAL_COST_TRACKER.record(response)
        return response

    return wrapped_func


class TrackedStreamWrapper:
    """Class that tracks costs as one iterates through the stream.

    Note that the following is not possible:
    ```
    async def wrap(func):
        resp: CustomStreamWrapper = await func()
        async for response in resp:
            yield response

    # This is ok
    async for resp in await litellm.acompletion(stream=True):
        print(resp


    # This is not, because we cannot await an AsyncGenerator
    async for resp in await wrap(litellm.acompletion(stream=True)):
        print(resp)
    ```

    In order for `track_costs_iter` to not change how users call functions,
    we introduce this class to wrap the stream.
    """

    def __init__(self, stream: litellm.CustomStreamWrapper):
        self.stream = stream

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        response = next(self.stream)
        if TRACK_COSTS.get():
            GLOBAL_COST_TRACKER.record(response)
        return response

    async def __anext__(self):
        response = await self.stream.__anext__()
        if TRACK_COSTS.get():
            GLOBAL_COST_TRACKER.record(response)
        return response


def track_costs_iter(
    func: Callable[TParams, TReturn],
) -> Callable[TParams, Awaitable[TrackedStreamWrapper]]:
    @wraps(func)
    async def wrapped_func(*args, **kwargs):
        return TrackedStreamWrapper(await func(*args, **kwargs))

    return wrapped_func
