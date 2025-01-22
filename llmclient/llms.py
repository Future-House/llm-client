import asyncio
import contextlib
import functools
import json
import logging
from abc import ABC
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
)
from enum import StrEnum
from inspect import isasyncgenfunction, isawaitable, signature
from typing import (
    Any,
    ClassVar,
    TypeAlias,
    TypeVar,
    cast,
)

import litellm
from aviary.core import (
    Message,
    Tool,
    ToolRequestMessage,
    ToolsAdapter,
    ToolSelector,
    is_coroutine_callable,
)
from litellm.types.utils import ModelResponse, ModelResponseStream
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from llmclient.constants import (
    CHARACTERS_PER_TOKEN_ASSUMPTION,
    DEFAULT_VERTEX_SAFETY_SETTINGS,
    IS_PYTHON_BELOW_312,
)
from llmclient.cost_tracker import track_costs, track_costs_iter
from llmclient.exceptions import JSONSchemaValidationError
from llmclient.rate_limiter import GLOBAL_LIMITER
from llmclient.types import LLMResult
from llmclient.utils import get_litellm_retrying_config

logger = logging.getLogger(__name__)

if not IS_PYTHON_BELOW_312:
    _DeploymentTypedDictValidator = TypeAdapter(
        list[litellm.DeploymentTypedDict],
        config=ConfigDict(arbitrary_types_allowed=True),
    )

# Yes, this is a hack, it mostly matches
# https://github.com/python-jsonschema/referencing/blob/v0.35.1/referencing/jsonschema.py#L20-L21
JSONSchema: TypeAlias = Mapping[str, Any]


class CommonLLMNames(StrEnum):
    """When you don't want to think about models, just use one from here."""

    # Use these to avoid thinking about exact versions
    GPT_4O = "gpt-4o-2024-11-20"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"

    # Use these when trying to think of a somewhat opinionated default
    OPENAI_BASELINE = "gpt-4o-2024-11-20"  # Fast and decent

    # Use these in unit testing
    OPENAI_TEST = "gpt-4o-mini-2024-07-18"  # Cheap, fast, and not OpenAI's cutting edge
    ANTHROPIC_TEST = (
        "claude-3-5-haiku-20241022"  # Cheap, fast, and not Anthropic's cutting edge
    )


def sum_logprobs(choice: litellm.utils.Choices) -> float | None:
    """Calculate the sum of the log probabilities of an LLM completion (a Choices object).

    Args:
        choice: A sequence of choices from the completion.

    Returns:
        The sum of the log probabilities of the choice.
    """
    try:
        logprob_obj = choice.logprobs
    except AttributeError:
        return None
    if isinstance(logprob_obj, dict):
        if logprob_obj.get("content"):
            return sum(
                logprob_info["logprob"] for logprob_info in logprob_obj["content"]
            )
    elif choice.logprobs.content:
        return sum(logprob_info.logprob for logprob_info in choice.logprobs.content)
    return None


def validate_json_completion(
    completion: litellm.ModelResponse,
    output_type: type[BaseModel] | TypeAdapter | JSONSchema,
) -> None:
    """Validate a completion against a JSON schema.

    Args:
        completion: The completion to validate.
        output_type: A Pydantic model, Pydantic type adapter, or a JSON schema to
            validate the completion.
    """
    try:
        for choice in completion.choices:
            if not hasattr(choice, "message") or not choice.message.content:
                continue
            # make sure it is a JSON completion, even if None
            # We do want to modify the underlying message
            # so that users of it can just parse it as expected
            choice.message.content = (
                choice.message.content.split("```json")[-1].split("```")[0] or ""
            )
            if isinstance(output_type, Mapping):  # JSON schema
                litellm.litellm_core_utils.json_validation_rule.validate_schema(
                    schema=dict(output_type), response=choice.message.content
                )
            elif isinstance(output_type, TypeAdapter):
                output_type.validate_json(choice.message.content)
            else:
                output_type.model_validate_json(choice.message.content)
    except ValidationError as err:
        raise JSONSchemaValidationError(
            "The completion does not match the specified schema."
        ) from err


def prepare_args(
    func: Callable, completion: str, name: str | None
) -> tuple[tuple, dict]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (completion,), {"name": name}
    return (completion,), {}


async def do_callbacks(
    async_callbacks: Iterable[Callable[..., Awaitable]],
    sync_callbacks: Iterable[Callable[..., Any]],
    completion: str,
    name: str | None,
) -> None:
    await asyncio.gather(
        *(
            f(*args, **kwargs)
            for f in async_callbacks
            for args, kwargs in (prepare_args(f, completion, name),)
        )
    )
    for f in sync_callbacks:
        args, kwargs = prepare_args(f, completion, name)
        f(*args, **kwargs)


class LLMModel(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    llm_type: str | None = None
    name: str
    llm_result_callback: Callable[[LLMResult], Any | Awaitable[Any]] | None = Field(
        default=None,
        description=(
            "An async callback that will be executed on each"
            " LLMResult (different than callbacks that execute on each completion)"
        ),
        exclude=True,
    )
    config: dict = Field(default_factory=dict)

    async def acompletion(self, messages: list[Message], **kwargs) -> list[LLMResult]:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def acompletion_iter(
        self, messages: list[Message], **kwargs
    ) -> AsyncIterator[LLMResult]:
        """Return an async generator that yields completions.

        Only the last tuple will be non-zero.
        """
        raise NotImplementedError
        if False:  # type: ignore[unreachable]  # pylint: disable=using-constant-test
            yield  # Trick mypy: https://github.com/python/mypy/issues/5070#issuecomment-1050834495

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # gross approximation

    def __str__(self) -> str:
        return f"{type(self).__name__} {self.name}"

    # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # > `none` means the model will not call any tool and instead generates a message.
    # > `auto` means the model can pick between generating a message or calling one or more tools.
    # > `required` means the model must call one or more tools.
    NO_TOOL_CHOICE: ClassVar[str] = "none"
    MODEL_CHOOSES_TOOL: ClassVar[str] = "auto"
    TOOL_CHOICE_REQUIRED: ClassVar[str] = "required"
    # None means we won't provide a tool_choice to the LLM API
    UNSPECIFIED_TOOL_CHOICE: ClassVar[None] = None

    async def call(  # noqa: C901, PLR0915
        self,
        messages: list[Message],
        callbacks: Iterable[Callable] | None = None,
        name: str | None = None,
        output_type: type[BaseModel] | TypeAdapter | JSONSchema | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Tool | str | None = TOOL_CHOICE_REQUIRED,
        **chat_kwargs,
    ) -> list[LLMResult]:
        """Call the LLM model with the given messages and configuration.

        messages: A list of messages to send to the language model.
        callbacks: A list of callback functions to execute
        name: Optional name for the result.
        output_type: The type of the output.
        tools: A list of tools to use.
        tool_choice: The tool choice to use.

        Results: A list of LLMResult objects containing the result of the call.

        Raises:
            ValueError: If the LLM type is unknown.
        """
        n = chat_kwargs.get("n") or self.config.get("n", 1)
        if n < 1:
            raise ValueError("Number of completions (n) must be >= 1.")

        # deal with tools
        if tools:
            chat_kwargs["tools"] = ToolsAdapter.dump_python(
                tools, exclude_none=True, by_alias=True
            )
            if tool_choice is not None:
                chat_kwargs["tool_choice"] = (
                    {
                        "type": "function",
                        "function": {"name": tool_choice.info.name},
                    }
                    if isinstance(tool_choice, Tool)
                    else tool_choice
                )

        # deal with specifying output type
        if isinstance(output_type, Mapping):  # Use structured outputs
            model_name: str = chat_kwargs.get("model") or self.name
            if not litellm.supports_response_schema(model_name, None):
                raise ValueError(f"Model {model_name} does not support JSON schema.")

            chat_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    # SEE: https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
                    "schema": dict(output_type) | {"additionalProperties": False},
                    "name": output_type["title"],  # Required by OpenAI as of 12/3/2024
                },
            }
        elif output_type is not None:  # Use JSON mode
            if isinstance(output_type, TypeAdapter):
                schema: str = json.dumps(output_type.json_schema())
            else:
                schema = json.dumps(output_type.model_json_schema())
            schema_msg = f"Respond following this JSON schema:\n\n{schema}"
            # Get the system prompt and its index, or the index to add it
            i, system_prompt = next(
                ((i, m) for i, m in enumerate(messages) if m.role == "system"),
                (0, None),
            )
            messages = [
                *messages[:i],
                (
                    system_prompt.append_text(schema_msg, inplace=False)
                    if system_prompt
                    else Message(role="system", content=schema_msg)
                ),
                *messages[i + 1 if system_prompt else i :],
            ]
            chat_kwargs["response_format"] = {"type": "json_object"}

        messages = [
            (
                m
                if not isinstance(m, ToolRequestMessage) or m.tool_calls
                # OpenAI doesn't allow for empty tool_calls lists, so downcast empty
                # ToolRequestMessage to Message here
                else Message(role=m.role, content=m.content)
            )
            for m in messages
        ]
        results: list[LLMResult] = []

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            results = await self.acompletion(messages, **chat_kwargs)
        else:
            if tools:
                raise NotImplementedError("Using tools with callbacks is not supported")
            n = chat_kwargs.get("n") or self.config.get("n", 1)
            if n > 1:
                raise NotImplementedError(
                    "Multiple completions with callbacks is not supported"
                )
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
            stream_results = await self.acompletion_iter(messages, **chat_kwargs)  # type: ignore[misc]
            text_result = []
            async for result in stream_results:
                if result.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = (
                            asyncio.get_running_loop().time() - start_clock
                        )
                    text_result.append(result.text)
                    await do_callbacks(
                        async_callbacks, sync_callbacks, result.text, name
                    )
                results.append(result)

        for result in results:
            usage = result.prompt_count, result.completion_count
            if not sum(usage):
                result.completion_count = self.count_tokens(result.text)
            result.seconds_to_last_token = (
                asyncio.get_running_loop().time() - start_clock
            )

            if self.llm_result_callback:
                possibly_awaitable_result = self.llm_result_callback(result)
                if isawaitable(possibly_awaitable_result):
                    await possibly_awaitable_result
        return results

    async def call_single(
        self,
        messages: list[Message],
        callbacks: Iterable[Callable] | None = None,
        name: str | None = None,
        output_type: type[BaseModel] | TypeAdapter | JSONSchema | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Tool | str | None = TOOL_CHOICE_REQUIRED,
    ) -> LLMResult:
        results = await self.call(
            messages, callbacks, name, output_type, tools, tool_choice, n=1
        )
        if not results:
            raise ValueError("No results returned from call")
        return results[0]


LLMModelOrChild = TypeVar("LLMModelOrChild", bound=LLMModel)


def rate_limited(
    func: Callable[
        [LLMModelOrChild, Any],
        Awaitable[ModelResponse | ModelResponseStream | list[LLMResult]]
        | AsyncIterable[LLMResult],
    ],
) -> Callable[
    [LLMModelOrChild, Any],
    Awaitable[list[LLMResult] | AsyncIterator[LLMResult]],
]:
    """Decorator to rate limit relevant methods of an LLMModel."""

    @functools.wraps(func)
    async def wrapper(
        self: LLMModelOrChild, *args: Any, **kwargs: Any
    ) -> list[LLMResult] | AsyncIterator[LLMResult]:
        if not hasattr(self, "check_rate_limit"):
            raise NotImplementedError(
                f"Model {self.name} must have a `check_rate_limit` method."
            )

        # Estimate token count based on input
        if func.__name__ in {"acompletion", "acompletion_iter"}:
            messages = args[0] if args else kwargs.get("messages", [])
            token_count = len(str(messages)) / CHARACTERS_PER_TOKEN_ASSUMPTION
        else:
            token_count = 0  # Default if method is unknown

        await self.check_rate_limit(token_count)

        # If wrapping a generator, count the tokens for each
        # portion before yielding
        if isasyncgenfunction(func):

            async def rate_limited_generator() -> AsyncGenerator[LLMResult, None]:
                async for item in func(self, *args, **kwargs):
                    token_count = 0
                    if isinstance(item, LLMResult):
                        token_count = int(
                            len(item.text or "") / CHARACTERS_PER_TOKEN_ASSUMPTION
                        )
                    await self.check_rate_limit(token_count)
                    yield item

            return rate_limited_generator()

        # We checked isasyncgenfunction above, so this must be a Awaitable
        result = await cast(Awaitable[Any], func(self, *args, **kwargs))

        if func.__name__ == "acompletion" and isinstance(result, list):
            await self.check_rate_limit(sum(r.completion_count for r in result))
        return result

    return wrapper


class PassThroughRouter(litellm.Router):  # TODO: add rate_limited
    """Router that is just a wrapper on LiteLLM's normal free functions."""

    def __init__(self, **kwargs):
        self._default_kwargs = kwargs

    async def acompletion(self, *args, **kwargs):
        return await litellm.acompletion(*args, **(self._default_kwargs | kwargs))


class LiteLLMModel(LLMModel):
    """A wrapper around the litellm library."""

    model_config = ConfigDict(extra="forbid")

    name: str = "gpt-4o-mini"
    config: dict = Field(
        default_factory=dict,
        description=(
            "Configuration of this model containing several important keys. The"
            " optional `model_list` key stores a list of all model configurations"
            " (SEE: https://docs.litellm.ai/docs/routing). The optional"
            " `router_kwargs` key is keyword arguments to pass to the Router class."
            " Inclusion of a key `pass_through_router` with a truthy value will lead"
            " to using not using LiteLLM's Router, instead just LiteLLM's free"
            f" functions (see {PassThroughRouter.__name__}). Rate limiting applies"
            " regardless of `pass_through_router` being present. The optional"
            " `rate_limit` key is a dictionary keyed by model group name with values"
            " of type limits.RateLimitItem (in tokens / minute) or valid"
            " limits.RateLimitItem string for parsing."
        ),
    )
    _router: litellm.Router | None = None

    @model_validator(mode="before")
    @classmethod
    def maybe_set_config_attribute(cls, data: dict[str, Any]) -> dict[str, Any]:
        """If a user only gives a name, make a sensible config dict for them."""
        if "config" not in data:
            data["config"] = {}
        if "name" in data and "model_list" not in data["config"]:
            data["config"] = {
                "model_list": [
                    {
                        "model_name": data["name"],
                        "litellm_params": {
                            "model": data["name"],
                            "n": data["config"].get("n", 1),
                            "temperature": data["config"].get("temperature", 0.1),
                            "max_tokens": data["config"].get("max_tokens", 4096),
                        }
                        | (
                            {}
                            if "gemini" not in data["name"]
                            else {"safety_settings": DEFAULT_VERTEX_SAFETY_SETTINGS}
                        ),
                    }
                ],
            } | data["config"]

        if "router_kwargs" not in data["config"]:
            data["config"]["router_kwargs"] = {}
        data["config"]["router_kwargs"] = (
            get_litellm_retrying_config() | data["config"]["router_kwargs"]
        )
        if not data["config"].get("pass_through_router"):
            data["config"]["router_kwargs"] = {"retry_after": 5} | data["config"][
                "router_kwargs"
            ]

        # we only support one "model name" for now, here we validate
        model_list = data["config"]["model_list"]
        if IS_PYTHON_BELOW_312:
            if not isinstance(model_list, list):
                # Work around https://github.com/BerriAI/litellm/issues/5664
                raise TypeError(f"model_list must be a list, not a {type(model_list)}.")
        else:
            # pylint: disable-next=possibly-used-before-assignment
            _DeploymentTypedDictValidator.validate_python(model_list)
        if len({m["model_name"] for m in model_list}) > 1:
            raise ValueError("Only one model name per model list is supported for now.")
        return data

    # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # > `none` means the model will not call any tool and instead generates a message.
    # > `auto` means the model can pick between generating a message or calling one or more tools.
    # > `required` means the model must call one or more tools.
    NO_TOOL_CHOICE: ClassVar[str] = "none"
    MODEL_CHOOSES_TOOL: ClassVar[str] = "auto"
    TOOL_CHOICE_REQUIRED: ClassVar[str] = "required"
    # None means we won't provide a tool_choice to the LLM API
    UNSPECIFIED_TOOL_CHOICE: ClassVar[None] = None

    def __getstate__(self):
        # Prevent _router from being pickled, SEE: https://stackoverflow.com/a/2345953
        state = super().__getstate__()
        state["__dict__"] = state["__dict__"].copy()
        state["__dict__"].pop("_router", None)
        return state

    @property
    def router(self) -> litellm.Router:
        if self._router is None:
            router_kwargs: dict = self.config.get("router_kwargs", {})
            if self.config.get("pass_through_router"):
                self._router = PassThroughRouter(**router_kwargs)
            else:
                self._router = litellm.Router(
                    model_list=self.config["model_list"], **router_kwargs
                )
        return self._router

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        if "rate_limit" in self.config:
            await GLOBAL_LIMITER.try_acquire(
                ("client", self.name),
                self.config["rate_limit"].get(self.name, None),
                weight=max(int(token_count), 1),
                **kwargs,
            )

    @rate_limited
    async def acompletion(self, messages: list[Message], **kwargs) -> list[LLMResult]:  # type: ignore[override]
        prompts = [m.model_dump(by_alias=True) for m in messages if m.content]
        completions = await track_costs(self.router.acompletion)(
            self.name, prompts, **kwargs
        )
        results: list[LLMResult] = []

        # We are not streaming here, so we can cast to list[litellm.utils.Choices]
        choices = cast(list[litellm.utils.Choices], completions.choices)
        for completion in choices:
            if completion.finish_reason == "tool_calls" or getattr(
                completion.message, "tool_calls", None
            ):
                serialized_message = completion.message.model_dump()
                serialized_message["tool_calls"] = (
                    serialized_message.get("tool_calls") or []
                )
                output_messages: list[Message | ToolRequestMessage] = [
                    ToolRequestMessage(**serialized_message)
                ]
            else:
                output_messages = [Message(**completion.message.model_dump())]
            results.append(
                LLMResult(
                    model=self.name,
                    text=completion.message.content,
                    prompt=messages,
                    messages=output_messages,
                    logprob=sum_logprobs(completion),
                    prompt_count=completions.usage.prompt_tokens,  # type: ignore[attr-defined]
                    completion_count=completions.usage.completion_tokens,  # type: ignore[attr-defined]
                    system_fingerprint=completions.system_fingerprint,
                )
            )
        return results

    @rate_limited
    async def acompletion_iter(  # type: ignore[override]
        self, messages: list[Message], **kwargs
    ) -> AsyncGenerator[LLMResult]:
        prompts = [m.model_dump(by_alias=True) for m in messages if m.content]
        stream_completions = await track_costs_iter(self.router.acompletion)(
            self.name,
            prompts,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        start_clock = asyncio.get_running_loop().time()
        result = LLMResult(model=self.name, prompt=messages)
        outputs = []
        role = None
        async for completion in stream_completions:
            delta = completion.choices[0].delta
            outputs.append(delta.content or "")
            role = delta.role or role

        text = "".join(outputs)
        result = LLMResult(
            model=self.name,
            text=text,
            prompt=messages,
            messages=[Message(role=role, content=text)],
            # TODO: Can we marginalize over all choices?
            # logprob=sum_logprobs(completion),
        )

        if text:
            result.seconds_to_first_token = (
                asyncio.get_running_loop().time() - start_clock
            )
        if hasattr(completion, "usage"):
            result.prompt_count = completion.usage.prompt_tokens
            result.completion_count = completion.usage.completion_tokens

        yield result

    def count_tokens(self, text: str) -> int:
        return litellm.token_counter(model=self.name, text=text)

    async def select_tool(
        self, *selection_args, **selection_kwargs
    ) -> ToolRequestMessage:
        """Shim to aviary.core.ToolSelector that supports tool schemae."""
        tool_selector = ToolSelector(
            model_name=self.name, acompletion=track_costs(self.router.acompletion)
        )
        return await tool_selector(*selection_args, **selection_kwargs)
