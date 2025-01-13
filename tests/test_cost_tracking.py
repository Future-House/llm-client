from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from aviary.core import Message

from llmclient import cost_tracking_ctx
from llmclient.cost_tracker import GLOBAL_COST_TRACKER
from llmclient.embeddings import LiteLLMEmbeddingModel
from llmclient.llms import CommonLLMNames, LiteLLMModel, MultipleCompletionLLMModel
from llmclient.types import LLMResult

from .conftest import VCR_DEFAULT_MATCH_ON


@contextmanager
def assert_costs_increased():
    """All tests in this file should increase accumulated costs."""
    initial_cost = GLOBAL_COST_TRACKER.lifetime_cost_usd
    yield
    assert GLOBAL_COST_TRACKER.lifetime_cost_usd > initial_cost


class TestLiteLLMEmbeddingCosts:
    @pytest.mark.asyncio
    async def test_embed_documents(self):
        stub_texts = ["test1", "test2"]
        with assert_costs_increased(), cost_tracking_ctx():
            model = LiteLLMEmbeddingModel(name="text-embedding-3-small", ndim=8)
            await model.embed_documents(stub_texts)


class TestLiteLLMModel:
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_name": "gpt-4o-mini",
                    "model_list": [
                        {
                            "model_name": "gpt-4o-mini",
                            "litellm_params": {
                                "model": "gpt-4o-mini",
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="chat-model",
            ),
            pytest.param(
                {
                    "model_name": "gpt-3.5-turbo-instruct",
                    "model_list": [
                        {
                            "model_name": "gpt-3.5-turbo-instruct",
                            "litellm_params": {
                                "model": "gpt-3.5-turbo-instruct",
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="completion-model",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_call(self, config: dict[str, Any]) -> None:
        with assert_costs_increased(), cost_tracking_ctx():
            llm = LiteLLMModel(name=config["model_name"], config=config)
            messages = [
                Message(role="system", content="Respond with single words."),
                Message(role="user", content="What is the meaning of the universe?"),
            ]
            await llm.call(messages)

    @pytest.mark.asyncio
    async def test_call_w_figure(self) -> None:
        async def ac(x) -> None:
            pass

        with cost_tracking_ctx():
            with assert_costs_increased():
                llm = LiteLLMModel(name="gpt-4o")
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                image[:] = [255, 0, 0]
                messages = [
                    Message(
                        role="system",
                        content="You are a detective who investigate colors",
                    ),
                    Message.create_message(
                        role="user",
                        text=(
                            "What color is this square? Show me your chain of"
                            " reasoning."
                        ),
                        images=image,
                    ),
                ]  # TODO: It's not decoding the image. It's trying to guess the color from the encoded image string.
                await llm.call(messages)

            with assert_costs_increased():
                await llm.call(messages, [ac])

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": "gpt-4o-mini",
                            "litellm_params": {
                                "model": "gpt-4o-mini",
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ]
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
    @pytest.mark.asyncio
    async def test_run_prompt(self, config: dict[str, Any]) -> None:
        with cost_tracking_ctx(), assert_costs_increased():
            llm = LiteLLMModel(name="gpt-4o-mini", config=config)

            outputs = []

            def accum(x) -> None:
                outputs.append(x)

            await llm.run_prompt(
                prompt="The {animal} says",
                data={"animal": "duck"},
                system_prompt=None,
                callbacks=[accum],
            )


class TestMultipleCompletionLLMModel:
    async def call_model(
        self, model: MultipleCompletionLLMModel, *args, **kwargs
    ) -> list[LLMResult]:
        return await model.call(*args, **kwargs)

    @pytest.mark.parametrize(
        "model_name", ["gpt-3.5-turbo", CommonLLMNames.ANTHROPIC_TEST.value]
    )
    @pytest.mark.asyncio
    async def test_achat(self, model_name: str) -> None:
        with cost_tracking_ctx():
            with assert_costs_increased():
                model = MultipleCompletionLLMModel(name=model_name)
                await model.achat(
                    messages=[
                        Message(content="What are three things I should do today?"),
                    ]
                )

            with assert_costs_increased():
                async for _ in await model.achat_iter(
                    messages=[
                        Message(content="What are three things I should do today?"),
                    ]
                ):
                    pass

    @pytest.mark.parametrize("model_name", [CommonLLMNames.OPENAI_TEST.value])
    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_text_image_message(self, model_name: str) -> None:
        with cost_tracking_ctx(), assert_costs_increased():
            model = MultipleCompletionLLMModel(name=model_name, config={"n": 2})

            # An RGB image of a red square
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            # (255 red, 0 green, 0 blue) is maximum red in RGB
            image[:] = [255, 0, 0]

            await self.call_model(
                model,
                messages=[
                    Message.create_message(
                        text=(
                            "What color is this square? Respond only with the color"
                            " name."
                        ),
                        images=image,
                    )
                ],
            )
