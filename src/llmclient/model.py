import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from typing import Any, AsyncIterable, ClassVar, Self, cast

import litellm
from aviary.core import (
    Tool,
    ToolRequestMessage,
    ToolsAdapter,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llmclient.constants import default_system_prompt
from llmclient.result import LLMResult
from llmclient.util import do_callbacks, is_coroutine_callable
# from llmclient.message import LLMMessage as Message
from aviary.core import Message

class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str | None
    prompt_tokens: int
    completion_tokens: int

    def __str__(self):
        return self.text

class JSONSchemaValidationError(ValueError):
    """Raised when the completion does not match the specified schema."""

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
    completion: litellm.ModelResponse, output_type: type[BaseModel]
) -> None:
    """Validate a completion against a JSON schema.

    Args:
        completion: The completion to validate.
        output_type: The Pydantic model to validate the completion against.
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
            output_type.model_validate_json(choice.message.content)
    except ValidationError as err:
        raise JSONSchemaValidationError(
            "The completion does not match the specified schema."
        ) from err

class LLMModel(BaseModel):
    """Run n completions at once, all starting from the same messages."""

    model_config = ConfigDict(extra="forbid")

    # this should keep the original model
    # if fine-tuned, this should still refer to the base model
    name: str = "unknown"
    llm_type: str | None = None
    llm_result_callback: (
        Callable[[LLMResult], None] | Callable[[LLMResult], Awaitable[None]] | None
    ) = Field(
        default=None,
        description=(
            "An async callback that will be executed on each"
            " LLMResult (different than callbacks that execute on each chunk)"
        ),
        exclude=True,
    )
    config: dict = Field(
        default={
            "model": "gpt-3.5-turbo",  # Default model should have cheap input/output for testing
            "temperature": 0.1,
        }
    )
    encoding: Any | None = None

    def __str__(self) -> str:
        return f"{type(self).__name__} {self.name}"
    
    def infer_llm_type(self) -> str:
        return "completion"
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # gross approximation
    
    async def run_prompt(
        self,
        prompt: str,
        data: dict,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
    ) -> LLMResult:
        if not self.llm_type:
            self.llm_type = self.infer_llm_type()

        if self.llm_type == "chat":
            return await self._run_chat(
                prompt, data, callbacks, name, skip_system, system_prompt
            )
        elif self.llm_type == "completion":
            return await self._run_completion(
                prompt, data, callbacks, name, skip_system, system_prompt
            )
        raise ValueError(f"Unknown llm_type {self.llm_type!r}.")
    

    async def get_result(self, usage, result, output, start_clock):
        if sum(usage) > 0:
            result.prompt_count, result.completion_count = usage
        elif output:
            result.completion_count = self.count_tokens(output)
            
        result.text = output or ""
        result.seconds_to_last_token = asyncio.get_running_loop().time() - start_clock

        if self.llm_result_callback:
            if is_coroutine_callable(self.llm_result_callback):
                await self.llm_result_callback(result)  # type: ignore[misc]
            else:
                self.llm_result_callback(result)
        return result
    
    async def add_chunk_text(self, result, async_callbacks, sync_callbacks, chunk, text_result, start_clock, name):
        if not chunk.text:
            return
        
        if result.seconds_to_first_token == 0:
            result.seconds_to_first_token = asyncio.get_running_loop().time() - start_clock

        text_result.append(chunk.text)
        await do_callbacks(
            async_callbacks, sync_callbacks, chunk.text, name
        )

    async def _run_chat(
        self,
        prompt: str,
        data: dict,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
    ) -> LLMResult:
        """Run a chat prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.
            skip_system: Set True to skip the system prompt.
            system_prompt: System prompt to use.

        Returns:
            Result of the chat.
        """
        system_message_prompt = {"role": "system", "content": system_prompt}
        human_message_prompt = {"role": "user", "content": prompt}
        messages = [
            {"role": m["role"], "content": m["content"].format(**data)}
            for m in (
                [human_message_prompt]
                if skip_system
                else [system_message_prompt, human_message_prompt]
            )
        ]
        result = LLMResult(
            model=self.name,
            name=name,
            prompt=messages,
            prompt_count=(
                sum(self.count_tokens(m["content"]) for m in messages) +
                sum(self.count_tokens(m["role"]) for m in messages)
            ),
        )

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            chunk = await self.achat(messages)
            output = chunk.text
        else:
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
            completion = await self.achat_iter(messages)  # type: ignore[misc]
            text_result = []
            async for chunk in completion:
                await self.add_chunk_text(result, async_callbacks, sync_callbacks, chunk, text_result, start_clock, name)
            output = "".join(text_result)
    
        usage = chunk.prompt_tokens, chunk.completion_tokens
        return await self.get_result(usage, result, output, start_clock)

    async def _run_completion(
        self,
        prompt: str,
        data: dict,
        callbacks: Iterable[Callable] | None = None,
        name: str | None = None,
        skip_system: bool = False,
        system_prompt: str = default_system_prompt,
    ) -> LLMResult:
        """Run a completion prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.
            skip_system: Set True to skip the system prompt.
            system_prompt: System prompt to use.

        Returns:
            Result of the completion.
        """
        formatted_prompt: str = (
            prompt if skip_system else system_prompt + "\n\n" + prompt
        ).format(**data)
        result = LLMResult(
            model=self.name,
            name=name,
            prompt=formatted_prompt,
            prompt_count=self.count_tokens(formatted_prompt),
        )

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            chunk = await self.acomplete(formatted_prompt)
            output = chunk.text
        else:
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]

            completion = self.acomplete_iter(formatted_prompt)
            text_result = []
            async for chunk in completion:
                await self.add_chunk_text(result, async_callbacks, sync_callbacks, chunk, text_result, start_clock, name)
            output = "".join(text_result)
        
        usage = chunk.prompt_tokens, chunk.completion_tokens
        return await self.get_result(usage, result, output, start_clock)
    

    @model_validator(mode="after")
    def set_model_name(self) -> Self:
        if (
            self.config.get("model") in {"gpt-3.5-turbo", None}
            and self.name != "unknown"
            or self.name != "unknown"
            and "model" not in self.config
        ):
            self.config["model"] = self.name
        elif "model" in self.config and self.name == "unknown":
            self.name = self.config["model"]
        # note we do not consider case where both are set
        # because that could be true if the model is fine-tuned
        return self
    
    async def acomplete(self, prompt: str) -> Chunk:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def acomplete_iter(self, prompt: str) -> AsyncIterable[Chunk]:  # noqa: ARG002
        """Return an async generator that yields chunks of the completion.

        Only the last tuple will be non-zero.
        """
        raise NotImplementedError

    async def achat(
        self, messages: Iterable[Message], **kwargs
    ) -> litellm.ModelResponse:
        return await litellm.acompletion(
            messages=[m.model_dump(by_alias=True) for m in messages],
            **(self.config | kwargs),
        )

    async def achat_iter(self, messages: Iterable[Message], **kwargs) -> AsyncGenerator:
        return cast(
            AsyncGenerator,
            await litellm.acompletion(
                messages=[m.model_dump(by_alias=True) for m in messages],
                stream=True,
                stream_options={
                    "include_usage": True,  # Included to get prompt token counts
                },
                **(self.config | kwargs),
            ),
        )

    # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # > `required` means the model must call one or more tools.
    TOOL_CHOICE_REQUIRED: ClassVar[str] = "required"

    async def handle_callbacks(self, tools, n, chat_kwargs, prompt, callbacks, messages, start_clock, results):
        if tools:
            raise NotImplementedError("Using tools with callbacks is not supported")
        if n > 1:
            raise NotImplementedError(
                "Multiple completions with callbacks is not supported"
            )
        result = LLMResult(model=self.name, config=chat_kwargs, prompt=prompt)

        sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
        async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
        stream_completion = await self.achat_iter(messages, **chat_kwargs)
        text_result = []
        role = "assistant"

        async for chunk in stream_completion:
            delta = chunk.choices[0].delta
            role = delta.role or role
            if delta.content:
                s = delta.content
                if result.seconds_to_first_token == 0:
                    result.seconds_to_first_token = asyncio.get_running_loop().time() - start_clock
                text_result.append(s)
                [await f(s) for f in async_callbacks]
                [f(s) for f in sync_callbacks]
            if hasattr(chunk, "usage"):
                result.prompt_count = chunk.usage.prompt_tokens

        output = "".join(text_result)
        result.completion_count = litellm.token_counter(
            model=self.name,
            text=output,
        )
        # TODO: figure out how tools stream, and log probs
        result.messages = [Message(role=role, content=output)]
        results.append(result)

    async def handle_no_callbacks(self, tools, chat_kwargs, prompt, results, output_type):
        # self.handle_no_callbacks()

        completion: litellm.ModelResponse = await self.achat(prompt, **chat_kwargs)
        if output_type:
            validate_json_completion(completion, output_type)

        for choice in completion.choices:
            if isinstance(choice, litellm.utils.StreamingChoices):
                raise NotImplementedError("Streaming is not yet supported.")

            if (
                tools is not None  # Allows for empty tools list
                or choice.finish_reason == "tool_calls"
                or (getattr(choice.message, "tool_calls", None) is not None)
            ):
                serialized_choice_message = choice.message.model_dump()
                serialized_choice_message["tool_calls"] = (
                    serialized_choice_message.get("tool_calls") or []
                )
                output_messages: list[Message | ToolRequestMessage] = [
                    ToolRequestMessage(**serialized_choice_message)
                ]
            else:
                output_messages = [Message(**choice.message.model_dump())]

            results.append(
                LLMResult(
                    model=self.name,
                    config=chat_kwargs,
                    prompt=prompt,
                    messages=output_messages,
                    logprob=sum_logprobs(choice),
                    system_fingerprint=completion.system_fingerprint,
                    # Note that these counts are aggregated over all choices
                    completion_count=completion.usage.completion_tokens,  # type: ignore[attr-defined,unused-ignore]
                    prompt_count=completion.usage.prompt_tokens,  # type: ignore[attr-defined,unused-ignore]
                )
            )

    async def call(  # noqa: C901, PLR0915
        self,
        messages: list[Message],
        callbacks: list[Callable] | None = None,
        output_type: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Tool | str | None = TOOL_CHOICE_REQUIRED,
        **chat_kwargs,
    ) -> list[LLMResult]:
        start_clock = asyncio.get_running_loop().time()

        # Deal with tools. OpenAI throws an error if tool list is empty,
        # so skip this block if tools in (None, [])
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
        if output_type is not None:
            schema = json.dumps(output_type.model_json_schema(mode="serialization"))
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

        # add static configuration to kwargs
        chat_kwargs = self.config | chat_kwargs
        n = chat_kwargs.get("n", 1)  # number of completions
        if n < 1:
            raise ValueError("Number of completions (n) must be >= 1.")

        prompt = [
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

        if callbacks is None:
            await self.handle_no_callbacks(tools, chat_kwargs, prompt, results, output_type)
        else:
            await self.handle_callbacks(tools, n, chat_kwargs, prompt, callbacks, messages, start_clock, results)

        if not results:
            # This happens in unit tests. We should probably not keep this block around
            # long-term. Previously, we would emit an empty ToolRequestMessage if
            # completion.choices were empty, so  I am replicating that here.
            results.append(
                LLMResult(
                    model=self.name,
                    config=chat_kwargs,
                    prompt=prompt,
                    messages=[ToolRequestMessage(tool_calls=[])],
                )
            )

        end_clock = asyncio.get_running_loop().time()

        for result in results:
            # Manually update prompt count if not set, which can
            # happen if the target model doesn't support 'include_usage'
            if not result.prompt_count:
                result.prompt_count = litellm.token_counter(
                    model=self.name,
                    messages=[m.model_dump() for m in result.messages],  # type: ignore[union-attr]
                )

            # update with server-side counts
            result.seconds_to_last_token = end_clock - start_clock

        return results
