from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    computed_field,
)
from typing import Any, Iterable, Union, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
from contextlib import contextmanager
from inspect import signature
from itertools import chain

import contextlib
import contextvars
import litellm
import logging
import inspect

from collections.abc import (
    Awaitable,
    Callable,
)

from aviary.core import (
    Message
)

logger = logging.getLogger(__name__)

# A context var that will be unique to threads/processes
cvar_session_id = contextvars.ContextVar[UUID | None]("session_id", default=None)

default_system_prompt = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them."
)

def prepare_args(func: Callable, chunk: str, name: str | None) -> tuple[tuple, dict]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (chunk,), {"name": name}
    return (chunk,), {}

def is_coroutine_callable(obj):
    if inspect.isfunction(obj):
        return inspect.iscoroutinefunction(obj)
    elif callable(obj):  # noqa: RET505
        return inspect.iscoroutinefunction(obj.__call__)
    return False

async def do_callbacks(
    async_callbacks: Iterable[Callable[..., Awaitable]],
    sync_callbacks: Iterable[Callable[..., Any]],
    chunk: str,
    name: str | None,
) -> None:
    for f in chain(async_callbacks, sync_callbacks):
        args, kwargs = prepare_args(f, chunk, name)
        await f(*args, **kwargs)

@contextmanager
def set_llm_session_ids(session_id: UUID):
    token = cvar_session_id.set(session_id)
    try:
        yield
    finally:
        cvar_session_id.reset(token)

class LLMResult(BaseModel):
    """A unified class to hold the result of a LLM completion, replacing two prior versions."""

    id: UUID = Field(default_factory=uuid4)
    model_config: ConfigDict = ConfigDict(populate_by_name=True)
    name: Optional[str] = None
    model: str
    text: str = "" 
    prompt_count: int = Field(default=0, description="Count of prompt tokens.")
    completion_count: int = Field(default=0, description="Count of completion tokens.")
    date: str = Field(default_factory=lambda: datetime.now().isoformat())
    seconds_to_first_token: Optional[float] = Field(
        default=0.0,
        description="Delta time (sec) to first response token's arrival."
    )
    seconds_to_last_token: float = Field(
        default=0.0,
        description="Delta time (sec) to last response token's arrival."
    )
    
    system_fingerprint: Optional[str] = Field(
        default=None, description="System fingerprint received from the LLM."
    )
    prompt: Union[str, List[dict], List[Message], None] = Field(
        default=None,
        description="Optional prompt (str) or list of serialized prompts (list[dict])."
    )
    config: Optional[dict] = None
    messages: Optional[List[Message]] = Field(
        default=None,
        description="Messages received from the LLM."
    )
    session_id: Optional[UUID] = Field(
        default_factory=cvar_session_id.get,
        description="A persistent ID to associate a group of LLMResults",
        alias="answer_id"
    )
    logprob: Optional[float] = Field(
        default=None, description="Sum of logprobs in the completion."
    )

    @property
    def prompt_and_completion_costs(self) -> tuple[float, float]:
        """Get a two-tuple of prompt tokens cost and completion tokens cost, in USD."""
        return litellm.cost_per_token(
            self.model,
            prompt_tokens=self.prompt_count,
            completion_tokens=self.completion_count,
        )

    @property
    def provider(self) -> str:
        """Get the model provider's name (e.g. 'openai', 'mistral')."""
        return litellm.get_llm_provider(self.model)[1]

    def get_supported_openai_params(self) -> Optional[List[str]]:
        """Get the supported OpenAI parameters for the model."""
        return litellm.get_supported_openai_params(self.model)

    @computed_field # type: ignore[prop-decorator]
    @property
    def cost(self) -> float:
        """Return the cost of the result in dollars."""
        if self.prompt_count and self.completion_count:
            try:
                pc = litellm.model_cost[self.model]["input_cost_per_token"]
                oc = litellm.model_cost[self.model]["output_cost_per_token"]
                return pc * self.prompt_count + oc * self.completion_count
            except KeyError:
                logger.warning(f"Could not find cost for model {self.model}.")
        return 0.0

    def __str__(self) -> str:
        return self.text
