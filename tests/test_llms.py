import pathlib
import pickle
from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import Mock, patch

import litellm
import numpy as np
import pytest
from aviary.core import Message, Tool, ToolRequestMessage
from pydantic import BaseModel, Field, TypeAdapter, computed_field

from llmclient.exceptions import JSONSchemaValidationError
from llmclient.llms import (
    CommonLLMNames,
    LiteLLMModel,
    validate_json_completion,
)
from llmclient.types import LLMResult
from tests.conftest import VCR_DEFAULT_MATCH_ON


class TestLiteLLMModel:
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                                "logprobs": True,
                            },
                        }
                    ],
                },
                id="OpenAI-model",
            ),
            pytest.param(
                {
                    "model_name": CommonLLMNames.ANTHROPIC_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.ANTHROPIC_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.ANTHROPIC_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="Anthropic-model",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_call(self, config: dict[str, Any]) -> None:
        llm = LiteLLMModel(name=config["model_name"], config=config)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(role="user", content="What is the meaning of the universe?"),
        ]
        results = await llm.call(messages)
        assert isinstance(results, list)

        result = results[0]
        assert isinstance(result, LLMResult)
        assert isinstance(result.prompt, list)
        assert isinstance(result.prompt[1], Message)
        assert all(isinstance(msg, Message) for msg in result.prompt)
        assert len(result.prompt) == 2  # role + user messages
        assert result.prompt[1].content
        assert result.text
        assert result.logprob is None or result.logprob <= 0

        result = await llm.call_single(messages)
        assert isinstance(result, LLMResult)

    # @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON])
    @pytest.mark.asyncio
    async def test_call_w_figure(self) -> None:
        llm = LiteLLMModel(name=CommonLLMNames.GPT_4O.value)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:] = [255, 0, 0]
        messages = [
            Message(
                role="system", content="You are a detective who investigate colors"
            ),
            Message.create_message(
                role="user",
                text="What color is this square? Show me your chain of reasoning.",
                images=image,
            ),
        ]
        results = await llm.call(messages)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, LLMResult)
            assert isinstance(result.prompt, list)
            assert all(isinstance(msg, Message) for msg in result.prompt)
            assert isinstance(result.prompt[1], Message)
            assert len(result.prompt) == 2
            assert result.prompt[1].content
            assert isinstance(result.text, str)
            assert "red" in result.text.lower()
            assert result.seconds_to_last_token > 0
            assert result.prompt_count > 0
            assert result.completion_count > 0
            assert result.cost > 0

        # Also test with a callback
        async def ac(x) -> None:
            pass

        results = await llm.call(messages, [ac])
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, LLMResult)
            assert isinstance(result.prompt, list)
            assert all(isinstance(msg, Message) for msg in result.prompt)
            assert isinstance(result.prompt[1], Message)
            assert len(result.prompt) == 2
            assert result.prompt[1].content
            assert isinstance(result.text, str)
            assert "red" in result.text.lower()
            assert result.seconds_to_last_token > 0
            assert result.prompt_count > 0
            assert result.completion_count > 0
            assert result.cost > 0

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                                "logprobs": True,
                            },
                        }
                    ],
                },
                id="with-router",
            ),
            pytest.param(
                {
                    "pass_through_router": True,
                    "router_kwargs": {"temperature": 0, "max_tokens": 56},
                },
                id="without-router",
            ),
        ],
    )
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.asyncio
    async def test_call_single(self, config: dict[str, Any], subtests) -> None:
        llm = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value, config=config)

        outputs = []

        def accum(x) -> None:
            outputs.append(x)

        prompt = "The {animal} says"
        data = {"animal": "duck"}
        system_prompt = "You are a helpful assistant."
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt.format(**data)),
        ]

        completion = await llm.call_single(
            messages=messages,
            callbacks=[accum],
        )
        assert completion.model == CommonLLMNames.OPENAI_TEST.value
        assert completion.seconds_to_last_token > 0
        assert completion.prompt_count > 0
        assert completion.completion_count > 0
        assert str(completion) == "".join(outputs)
        assert completion.cost > 0

        completion = await llm.call_single(
            messages=messages,
        )
        assert completion.seconds_to_last_token > 0
        assert completion.cost > 0

        # check with mixed callbacks
        async def ac(x) -> None:
            pass

        completion = await llm.call_single(
            messages=messages,
            callbacks=[accum, ac],
        )
        assert completion.cost > 0

        with subtests.test(msg="passing-kwargs"):
            completion = await llm.call_single(
                messages=[Message(role="user", content="Tell me a very long story")],
                max_tokens=1000,
            )
            assert completion.cost > 0
            assert completion.completion_count > 100, "Expected a long completion"

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        ("config", "bypassed_router"),
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "max_tokens": 3,
                            },
                        }
                    ]
                },
                False,
                id="with-router",
            ),
            pytest.param(
                {"pass_through_router": True, "router_kwargs": {"max_tokens": 3}},
                True,
                id="without-router",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_max_token_truncation(
        self, config: dict[str, Any], bypassed_router: bool
    ) -> None:
        llm = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value, config=config)
        with patch(
            "litellm.Router.acompletion",
            side_effect=litellm.Router.acompletion,
            autospec=True,
        ) as mock_completion:
            completions = await llm.acompletion(
                [Message(content="Please tell me a story")]
            )
        if bypassed_router:
            mock_completion.assert_not_awaited()
        else:
            mock_completion.assert_awaited_once()
        assert isinstance(completions, list)
        completion = completions[0]
        assert completion.completion_count == 3
        assert completion.text
        assert len(completion.text) < 20

    def test_pickling(self, tmp_path: pathlib.Path) -> None:
        pickle_path = tmp_path / "llm_model.pickle"
        llm = LiteLLMModel(
            name=CommonLLMNames.OPENAI_TEST.value,
            config={
                "model_list": [
                    {
                        "model_name": CommonLLMNames.OPENAI_TEST.value,
                        "litellm_params": {
                            "model": CommonLLMNames.OPENAI_TEST.value,
                            "temperature": 0,
                            "max_tokens": 56,
                        },
                    }
                ]
            },
        )
        with pickle_path.open("wb") as f:
            pickle.dump(llm, f)
        with pickle_path.open("rb") as f:
            rehydrated_llm = pickle.load(f)
        assert llm.name == rehydrated_llm.name
        assert llm.config == rehydrated_llm.config
        assert llm.router.deployment_names == rehydrated_llm.router.deployment_names


class DummyOutputSchema(BaseModel):
    name: str
    age: int = Field(description="Age in years.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name_and_age(self) -> str:  # So we can test computed_field is not included
        return f"{self.name}, {self.age}"


class TestMultipleCompletion:
    NUM_COMPLETIONS: ClassVar[int] = 2
    DEFAULT_CONFIG: ClassVar[dict] = {"n": NUM_COMPLETIONS}
    MODEL_CLS: ClassVar[type[LiteLLMModel]] = LiteLLMModel

    async def call_model(self, model: LiteLLMModel, *args, **kwargs) -> list[LLMResult]:
        return await model.call(*args, **kwargs)

    @pytest.mark.parametrize(
        "model_name",
        [CommonLLMNames.GPT_35_TURBO.value, CommonLLMNames.ANTHROPIC_TEST.value],
    )
    @pytest.mark.asyncio
    async def test_acompletion(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name)
        messages = [
            Message(content="What are three things I should do today?"),
        ]
        response = await model.acompletion(messages)

        assert isinstance(response, list)
        assert len(response) == 1
        assert isinstance(response[0], LLMResult)

    @pytest.mark.parametrize(
        "model_name",
        [CommonLLMNames.OPENAI_TEST.value, CommonLLMNames.ANTHROPIC_TEST.value],
    )
    @pytest.mark.asyncio
    async def test_acompletion_iter(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name)
        messages = [Message(content="What are three things I should do today?")]
        responses = await model.acompletion_iter(messages)
        assert isinstance(responses, AsyncIterator)

        async for response in responses:
            assert isinstance(response, LLMResult)
            assert isinstance(response.prompt, list)

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize("model_name", [CommonLLMNames.GPT_35_TURBO.value])
    @pytest.mark.asyncio
    async def test_model(self, model_name: str) -> None:
        # Make model_name an arg so that TestLLMModel can parametrize it
        # only testing OpenAI, as other APIs don't support n>1
        model = self.MODEL_CLS(name=model_name, config=self.DEFAULT_CONFIG)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]
        results = await self.call_model(model, messages)
        assert len(results) == self.NUM_COMPLETIONS

        for result in results:
            assert result.prompt_count > 0
            assert result.completion_count > 0
            assert result.cost > 0
            assert result.logprob is None or result.logprob <= 0

    @pytest.mark.parametrize(
        "model_name",
        [CommonLLMNames.ANTHROPIC_TEST.value, CommonLLMNames.GPT_35_TURBO.value],
    )
    @pytest.mark.asyncio
    async def test_streaming(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name, config=self.DEFAULT_CONFIG)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]

        def callback(_) -> None:
            return

        with pytest.raises(
            NotImplementedError,
            match="Multiple completions with callbacks is not supported",
        ):
            await self.call_model(model, messages, [callback])

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_parameterizing_tool_from_arg_union(self) -> None:
        def play(move: int | None) -> None:
            """Play one turn by choosing a move.

            Args:
                move: Choose an integer to lose, choose None to win.
            """

        results = await self.call_model(
            self.MODEL_CLS(
                name=CommonLLMNames.GPT_35_TURBO.value, config=self.DEFAULT_CONFIG
            ),
            messages=[Message(content="Please win.")],
            tools=[Tool.from_function(play)],
        )
        assert len(results) == self.NUM_COMPLETIONS
        for result in results:
            assert result.messages
            assert len(result.messages) == 1
            assert isinstance(result.messages[0], ToolRequestMessage)
            assert result.messages[0].tool_calls
            assert result.messages[0].tool_calls[0].function.arguments["move"] is None

    @pytest.mark.asyncio
    @pytest.mark.vcr
    @pytest.mark.parametrize(
        ("model_name", "output_type"),
        [
            pytest.param(
                CommonLLMNames.GPT_35_TURBO.value,
                DummyOutputSchema,
                id="json-mode-base-model",
            ),
            pytest.param(
                CommonLLMNames.GPT_4O.value,
                TypeAdapter(DummyOutputSchema),
                id="json-mode-type-adapter",
            ),
            pytest.param(
                CommonLLMNames.GPT_4O.value,
                DummyOutputSchema.model_json_schema(),
                id="structured-outputs",
            ),
        ],
    )
    async def test_output_schema(
        self, model_name: str, output_type: type[BaseModel] | dict[str, Any]
    ) -> None:
        model = self.MODEL_CLS(name=model_name, config=self.DEFAULT_CONFIG)
        messages = [
            Message(
                content=(
                    "My name is Claude and I am 1 year old. What is my name and age?"
                )
            ),
        ]
        results = await self.call_model(model, messages, output_type=output_type)
        assert len(results) == self.NUM_COMPLETIONS
        for result in results:
            assert result.messages
            assert len(result.messages) == 1
            assert result.messages[0].content
            DummyOutputSchema.model_validate_json(result.messages[0].content)

    @pytest.mark.parametrize("model_name", [CommonLLMNames.OPENAI_TEST.value])
    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_text_image_message(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name, config=self.DEFAULT_CONFIG)

        # An RGB image of a red square
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:] = [255, 0, 0]  # (255 red, 0 green, 0 blue) is maximum red in RGB

        results = await self.call_model(
            model,
            messages=[
                Message.create_message(
                    text="What color is this square? Respond only with the color name.",
                    images=image,
                )
            ],
        )
        assert len(results) == self.NUM_COMPLETIONS
        for result in results:
            assert (
                result.messages is not None
            ), "Expected messages in result, but got None"
            assert (
                result.messages[-1].content is not None
            ), "Expected content in message, but got None"
            assert "red" in result.messages[-1].content.lower()

    @pytest.mark.parametrize(
        "model_name",
        [CommonLLMNames.ANTHROPIC_TEST.value, CommonLLMNames.GPT_35_TURBO.value],
    )
    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_single_completion(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name, config={"n": 1})
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]
        result = await model.call_single(messages)
        assert isinstance(result, LLMResult)

        assert isinstance(result, LLMResult)
        assert result.messages
        assert len(result.messages) == 1
        assert result.messages[0].content

        model = self.MODEL_CLS(name=model_name, config={"n": 2})
        result = await model.call_single(messages)
        assert isinstance(result, LLMResult)
        assert result.messages
        assert len(result.messages) == 1
        assert result.messages[0].content

    @pytest.mark.asyncio
    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param(CommonLLMNames.ANTHROPIC_TEST.value, id="anthropic"),
            pytest.param(CommonLLMNames.OPENAI_TEST.value, id="openai"),
        ],
    )
    async def test_multiple_completion(self, model_name: str, request) -> None:
        model = self.MODEL_CLS(name=model_name, config={"n": self.NUM_COMPLETIONS})
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]
        if request.node.callspec.id == "anthropic":
            # Anthropic does not support multiple completions
            with pytest.raises(litellm.BadRequestError, match="anthropic"):
                await model.call(messages)
        else:
            results = await model.call(messages)  # noqa: FURB120
            assert len(results) == self.NUM_COMPLETIONS

            model = self.MODEL_CLS(name=model_name, config={"n": 5})
            results = await model.call(messages, n=self.NUM_COMPLETIONS)
            assert len(results) == self.NUM_COMPLETIONS


def test_json_schema_validation() -> None:
    # Invalid JSON
    mock_completion1 = Mock()
    mock_completion1.choices = [Mock()]
    mock_completion1.choices[0].message.content = "not a json"
    # Invalid schema
    mock_completion2 = Mock()
    mock_completion2.choices = [Mock()]
    mock_completion2.choices[0].message.content = '{"name": "John", "age": "nan"}'
    # Valid schema
    mock_completion3 = Mock()
    mock_completion3.choices = [Mock()]
    mock_completion3.choices[0].message.content = '{"name": "John", "age": 30}'

    class DummyModel(BaseModel):
        name: str
        age: int

    with pytest.raises(JSONSchemaValidationError):
        validate_json_completion(mock_completion1, DummyModel)
    with pytest.raises(JSONSchemaValidationError):
        validate_json_completion(mock_completion2, DummyModel)
    validate_json_completion(mock_completion3, DummyModel)


@pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
@pytest.mark.asyncio
async def test_deepseek_model():
    llm = LiteLLMModel(
        name="deepseek/deepseek-reasoner",
        config={
            "model_list": [
                {
                    "model_name": "deepseek/deepseek-reasoner",
                    "litellm_params": {
                        "model": "deepseek/deepseek-reasoner",
                        "api_base": "https://api.deepseek.com/v1",
                    },
                }
            ]
        },
    )
    messages = [
        Message(
            role="system",
            content="Think deeply about the following question and answer it.",
        ),
        Message(content="What is the meaning of life?"),
    ]
    results = await llm.call(messages)
    for result in results:
        assert result.reasoning_content

    outputs: list[str] = []
    results = await llm.call(messages, callbacks=[outputs.append])
    for result in results:
        # TODO: Litellm is not populating provider_specific_fields in streaming mode.
        # https://github.com/BerriAI/litellm/issues/7942
        # I'm keeping this test as a reminder to fix this.
        # once the issue is fixed.
        assert not result.reasoning_content
