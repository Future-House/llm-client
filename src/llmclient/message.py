from __future__ import annotations

from itertools import starmap
import json
import logging
import uuid

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import BaseModel, Field, field_validator, field_serializer, model_validator

from llmclient.util import encode_image_to_base64

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from logging import LogRecord

    import numpy as np

# A string to denote an invalid tool. It can be used to indicate
# an attempt to use a non-existent tool, missing/invalid parameters,
# mangled output from the LLM, etc.
INVALID_TOOL_NAME = "INVALID"


class ToolCallFunction(BaseModel):
    arguments: dict[str, Any]
    name: str

    @model_validator(mode="before")
    @classmethod
    def deserialize_args(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data["arguments"], str | None):
            if not data["arguments"]:
                data["arguments"] = {}
            else:
                try:
                    data["arguments"] = json.loads(data["arguments"])
                except json.JSONDecodeError:
                    # If the arguments are not parseable, mark this ToolCall(Function) as invalid
                    # so we can enable "learn"ing what a valid tool call looks like
                    logger.warning(
                        f"Failed to JSON load tool {data.get('name')}'s arguments"
                        f" {data['arguments']}, declaring as {INVALID_TOOL_NAME}."
                    )
                    data["name"] = INVALID_TOOL_NAME
                    data["arguments"] = {}

        return data

    @field_serializer("arguments")
    def serialize_arguments(self, arguments: dict[str, Any]) -> str:
        return json.dumps(arguments)

    def __str__(self) -> str:
        arg_str = ", ".join([f"{k}='{v}'" for k, v in self.arguments.items()])
        return f"{self.name}({arg_str})"


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction

    @staticmethod
    def generate_id() -> str:
        """Generate a tool call ID of length 9 with values in [a-zA-Z0-9]."""
        return str(uuid.uuid4()).replace("-", "")[:9]

    @classmethod
    def from_tool(cls, tool: "Tool", *args, id: str | None = None, **kwargs) -> Self:  # noqa: A002
        """Create a ToolCall from a Tool and arguments.

        The *args is packaged into the ToolCallFunction's arguments dict with best effort.
        **kwargs is what is passed to toolcall because we have to use named parameters.
        """
        # convert args to kwargs by matching them with the tool's parameters
        for i, name in enumerate(tool.info.parameters.properties.keys()):
            if i < len(args):
                kwargs[name] = args[i]
        return cls(
            id=id or cls.generate_id(),
            function=ToolCallFunction(name=tool.info.name, arguments=kwargs),
        )

    @classmethod
    def from_name(cls, function_name: str, **kwargs) -> Self:
        return cls(
            id=cls.generate_id(),
            function=ToolCallFunction(name=function_name, arguments=kwargs),
        )

    def __str__(self) -> str:
        arg_str = ", ".join([f"{k}='{v}'" for k, v in self.function.arguments.items()])
        return f"{self.function.name}({arg_str})"



class LLMMessage(BaseModel):
    DEFAULT_ROLE: ClassVar[str] = "user"
    VALID_ROLES: ClassVar[set[str]] = {
        DEFAULT_ROLE,
        "system",
        "tool",
        "assistant",
        "function",  # Prefer 'tool'
    }

    role: str = Field(
        default=DEFAULT_ROLE,
        description="Message role matching OpenAI's role conventions.",
    )
    content: str | None = Field(
        default=None,
        description=(
            "Optional message content. Can be a string or a dictionary or None. "
            "If a dictionary (for multimodal content), it will be JSON serialized. "
            "None is a sentinel value for the absence of content "
            "(different than empty string)."
        ),
    )
    content_is_json_str: bool = Field(
        default=False,
        description=(
            "Whether the content is JSON-serialized (e.g., for multiple modalities)."
        ),
        exclude=True,
        repr=False,
    )

    info: dict | None = Field(
        default=None,
        description="Optional metadata about the message.",
        exclude=True,
        repr=False,
    )
    model: str = ""

    @field_validator("role")
    @classmethod
    def check_role(cls, v: str) -> str:
        if v not in cls.VALID_ROLES:
            raise ValueError(f"Role {v} was not in {cls.VALID_ROLES}.")
        return v

    @model_validator(mode="before")
    @classmethod
    def serialize_content(cls, data):
        if isinstance(data, dict) and "content" in data:
            content = data["content"]
            if content is not None and not isinstance(content, str):
                try:
                    data["content"] = json.dumps(content)
                    data["content_is_json_str"] = True
                except TypeError as e:
                    raise ValueError(
                        "Content must be a string or JSON-serializable."
                    ) from e
        return data

    def __str__(self) -> str:
        return self.content or ""

    def model_dump(self, *args, **kwargs) -> dict:
        dump = super().model_dump(*args, **kwargs)
        if self.content_is_json_str:
            dump["content"] = json.loads(dump["content"])
        return dump

    def append_text(
        self, text: str, delim: str = "\n", inplace: bool = True
    ) -> LLMMessage:
        """Append text to the content.

        Args:
            text: The text to append.
            delim: The delimiter to use when concatenating strings.
            inplace: Whether to modify the message in place.

        Returns:
            The modified message. Note that the original message is modified and returned
            if `inplace=True` and a new message is returned otherwise.
        """
        if not self.content:
            new_content = text
        elif self.content_is_json_str:
            try:
                content_list = json.loads(self.content)
                if not isinstance(content_list, list):
                    raise TypeError("JSON content is not a list.")
                content_list.append({"type": "text", "text": text})
                new_content = json.dumps(content_list)
            except json.JSONDecodeError as e:
                raise ValueError("Content is not valid JSON.") from e
        else:
            new_content = f"{self.content}{delim}{text}"
        if inplace:
            self.content = new_content
            return self
        return self.model_copy(update={"content": new_content}, deep=True)

    @classmethod
    def create_message(
        cls,
        role: str = DEFAULT_ROLE,
        text: str | None = None,
        image: np.ndarray | None = None,
    ) -> Self:
        # Assume no image, and update to image if present
        content: str | list[dict] | None = text
        if image is not None:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_to_base64(image)},
                }
            ]
            if text is not None:
                content.append({"type": "text", "text": text})
        return cls(role=role, content=content)


class ToolRequestMessage(LLMMessage):
    role: Literal["assistant"] = Field(
        default="assistant", description="Matching LiteLLM structure."
    )
    content: str | None = None
    function_call: None = None
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="List of ToolCalls to make concurrently and independently.",
    )

    def __str__(self) -> str:
        if not self.tool_calls:
            return super().__str__()
        base_msg = f"Tool request message {self.content or ''!r}"
        if len(self.tool_calls) == 1:
            return (
                f"{base_msg} for tool calls: "
                f"{self.tool_calls[0]} [id={self.tool_calls[0].id}]"
            )
        return f"{base_msg} for tool calls: " + "; ".join([
            f"{tc!s} [id={tc.id}]" for tc in self.tool_calls
        ])


class ToolResponseMessage(LLMMessage):
    content: str = Field(
        description=(
            "Response message content, required to be a string by OpenAI/Anthropic."
        ),
    )
    role: Literal["tool"] = Field(
        default="tool", description="Matching LiteLLM structure."
    )
    name: str = Field(description="Name of the tool that was called.")
    tool_call_id: str = Field(
        description=(
            "Propagated from ToolCall.id, enabling matching response with"
            " ToolRequestMessage."
        )
    )

    @classmethod
    def from_call(cls, call: ToolCall, content: str) -> Self:
        return cls(content=content, name=call.function.name, tool_call_id=call.id)

    @classmethod
    def from_request(
        cls, request: ToolRequestMessage, contents: Iterable[str]
    ) -> list[Self]:
        return list(
            starmap(cls.from_call, zip(request.tool_calls, contents, strict=True))
        )

    def __str__(self) -> str:
        return (
            f"Tool response message {self.content!r} for tool call ID"
            f" {self.tool_call_id} of tool {self.name!r}"
        )

def join(
    msgs: Iterable[LLMMessage], delimiter: str = "\n", include_roles: bool = True
) -> str:
    return delimiter.join(
        f"{f'{m.role}: ' if include_roles else ''}{m.content or ''}" for m in msgs
    )


class MalformedMessageError(ValueError):
    """Error to throw if some aspect of a Message variant is malformed."""

    @classmethod
    def common_retryable_errors_log_filter(cls, record: LogRecord) -> bool:
        """
        Filter out common parsing failures not worth looking into from logs.

        Returns:
            False if the LogRecord should be filtered out, otherwise True to keep it.
        """
        # NOTE: match both this Exception type's name and its content, to be robust
        return not all(x in record.msg for x in (cls.__name__, EMPTY_CONTENT_BASE_MSG))


class EnvStateMessage(LLMMessage):
    """A message that contains the current state of the environment."""


# Define separately so we can filter out this message type
EMPTY_CONTENT_BASE_MSG = "No content in message"
