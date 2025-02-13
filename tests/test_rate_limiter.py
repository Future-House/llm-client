import asyncio
import time
from itertools import product
from typing import Any

import pytest
from aviary.core import Message
from limits import RateLimitItemPerSecond

from llmclient.constants import CHARACTERS_PER_TOKEN_ASSUMPTION
from llmclient.embeddings import LiteLLMEmbeddingModel
from llmclient.llms import CommonLLMNames, LiteLLMModel
from llmclient.types import LLMResult

LLM_CONFIG_W_RATE_LIMITS = [
    # following ensures that "short-form" rate limits are also supported
    # where the user doesn't specify the model_list
    {
        "name": CommonLLMNames.OPENAI_TEST.value,
        "config": {
            "rate_limit": {
                CommonLLMNames.OPENAI_TEST.value: RateLimitItemPerSecond(20, 3)
            },
        },
    },
    {
        "name": CommonLLMNames.OPENAI_TEST.value,
        "config": {
            "model_list": [
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "litellm_params": {
                        "model": CommonLLMNames.OPENAI_TEST.value,
                        "temperature": 0,
                    },
                }
            ],
            "rate_limit": {
                CommonLLMNames.OPENAI_TEST.value: RateLimitItemPerSecond(20, 1)
            },
        },
    },
    {
        "name": CommonLLMNames.OPENAI_TEST.value,
        "config": {
            "model_list": [
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "litellm_params": {
                        "model": CommonLLMNames.OPENAI_TEST.value,
                        "temperature": 0,
                    },
                }
            ],
            "rate_limit": {
                CommonLLMNames.OPENAI_TEST.value: RateLimitItemPerSecond(1_000_000, 1)
            },
        },
    },
    {
        "name": CommonLLMNames.OPENAI_TEST.value,
        "config": {
            "model_list": [
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "litellm_params": {
                        "model": CommonLLMNames.OPENAI_TEST.value,
                        "temperature": 0,
                    },
                }
            ]
        },
    },
]

RATE_LIMITER_PROMPT = "Animals make many noises. The duck says"

LLM_METHOD_AND_INPUTS = [
    {
        "method": "acompletion",
        "kwargs": {
            "messages": [Message.create_message(role="user", text=RATE_LIMITER_PROMPT)]
        },
    },
    {
        "method": "acompletion_iter",
        "kwargs": {
            "messages": [Message.create_message(role="user", text=RATE_LIMITER_PROMPT)]
        },
    },
]

rate_limit_configurations = list(
    product(LLM_CONFIG_W_RATE_LIMITS, LLM_METHOD_AND_INPUTS)
)

EMBEDDING_CONFIG_W_RATE_LIMITS = [
    {"config": {"rate_limit": RateLimitItemPerSecond(20, 5)}},
    {"config": {"rate_limit": RateLimitItemPerSecond(20, 3)}},
    {"config": {"rate_limit": RateLimitItemPerSecond(1_000_000, 1)}},
    {},
]

ACCEPTABLE_RATE_LIMIT_ERROR: float = 0.10  # 10% error margin for token estimate error


async def time_n_llm_methods(
    llm: LiteLLMModel, method: str, n: int, use_gather: bool = False, *args, **kwargs
) -> float:
    """Give the token per second rate of a method call."""
    start_time = time.time()
    outputs = []

    if not use_gather:
        for _ in range(n):
            if "iter" in method:
                outputs.extend(
                    [
                        output
                        async for output in await getattr(llm, method)(*args, **kwargs)
                    ]
                )
            else:
                outputs.append(await getattr(llm, method)(*args, **kwargs))

    else:
        outputs = await asyncio.gather(
            *[getattr(llm, method)(*args, **kwargs) for _ in range(n)]
        )

    character_count = 0
    token_count = 0

    if isinstance(outputs[0], LLMResult):
        character_count = sum(len(o.text or "") for o in outputs)
    else:
        character_count = sum(len(o) for o in outputs)

    if hasattr(outputs[0], "prompt_count"):
        token_count = sum(o.prompt_count + o.completion_count for o in outputs)

    return (
        (character_count / CHARACTERS_PER_TOKEN_ASSUMPTION)
        if token_count == 0
        else token_count
    ) / (time.time() - start_time)


@pytest.mark.parametrize("llm_config_w_rate_limits", LLM_CONFIG_W_RATE_LIMITS)
@pytest.mark.asyncio
async def test_rate_limit_on_call_single(
    llm_config_w_rate_limits: dict[str, Any],
) -> None:

    llm = LiteLLMModel(**llm_config_w_rate_limits)

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

    estimated_tokens_per_second = await time_n_llm_methods(
        llm,
        "call_single",
        3,
        messages=messages,
        callbacks=[accum],
    )

    if "rate_limit" in llm.config:
        max_tokens_per_second = (
            llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].amount
            / llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].multiples
        )
        assert estimated_tokens_per_second / max_tokens_per_second < (
            1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
        )
    else:
        assert estimated_tokens_per_second > 0

    outputs = []

    def accum2(x) -> None:
        outputs.append(x)

    estimated_tokens_per_second = await time_n_llm_methods(
        llm,
        "call_single",
        3,
        use_gather=True,
        messages=messages,
        callbacks=[accum2],
    )

    if "rate_limit" in llm.config:
        max_tokens_per_second = (
            llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].amount
            / llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].multiples
        )
        assert estimated_tokens_per_second / max_tokens_per_second < (
            1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
        )
    else:
        assert estimated_tokens_per_second > 0


@pytest.mark.parametrize(
    ("llm_config_w_rate_limits", "llm_method_kwargs"), rate_limit_configurations
)
@pytest.mark.asyncio
async def test_rate_limit_on_sequential_completion_litellm_methods(
    llm_config_w_rate_limits: dict[str, Any],
    llm_method_kwargs: dict[str, Any],
) -> None:

    llm = LiteLLMModel(**llm_config_w_rate_limits)

    estimated_tokens_per_second = await time_n_llm_methods(
        llm,
        llm_method_kwargs["method"],
        3,
        use_gather=False,
        **llm_method_kwargs["kwargs"],
    )
    if "rate_limit" in llm.config:
        max_tokens_per_second = (
            llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].amount
            / llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].multiples
        )
        assert estimated_tokens_per_second / max_tokens_per_second < (
            1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
        )
    else:
        assert estimated_tokens_per_second > 0


@pytest.mark.parametrize(
    ("llm_config_w_rate_limits", "llm_method_kwargs"), rate_limit_configurations
)
@pytest.mark.asyncio
async def test_rate_limit_on_parallel_completion_litellm_methods(
    llm_config_w_rate_limits: dict[str, Any],
    llm_method_kwargs: dict[str, Any],
) -> None:

    llm = LiteLLMModel(**llm_config_w_rate_limits)

    if "iter" not in llm_method_kwargs["method"]:
        estimated_tokens_per_second = await time_n_llm_methods(
            llm,
            llm_method_kwargs["method"],
            3,
            use_gather=True,
            **llm_method_kwargs["kwargs"],
        )
        if "rate_limit" in llm.config:
            max_tokens_per_second = (
                llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].amount
                / llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].multiples
            )
            assert estimated_tokens_per_second / max_tokens_per_second < (
                1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
            )
        else:
            assert estimated_tokens_per_second > 0


@pytest.mark.parametrize(
    "embedding_config_w_rate_limits", EMBEDDING_CONFIG_W_RATE_LIMITS
)
@pytest.mark.asyncio
async def test_embedding_rate_limits(
    embedding_config_w_rate_limits: dict[str, Any],
) -> None:

    embedding_model = LiteLLMEmbeddingModel(**embedding_config_w_rate_limits)
    texts_to_embed = ["the duck says"] * 10
    start = time.time()
    await embedding_model.embed_documents(texts=texts_to_embed, batch_size=5)
    estimated_tokens_per_second = sum(
        len(t) / CHARACTERS_PER_TOKEN_ASSUMPTION for t in texts_to_embed
    ) / (time.time() - start)

    if "rate_limit" in embedding_config_w_rate_limits:
        max_tokens_per_second = (
            embedding_config_w_rate_limits["rate_limit"].amount
            / embedding_config_w_rate_limits["rate_limit"].multiples
        )
        assert estimated_tokens_per_second / max_tokens_per_second < (
            1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
        )
    else:
        assert estimated_tokens_per_second > 0


@pytest.mark.asyncio
async def test_constructing_llm_w_multiple_models_1():
    model_name = CommonLLMNames.GPT_4O
    llm = LiteLLMModel(
        name=model_name,
        config={
            "model_list": [
                {
                    "model_name": CommonLLMNames.CLAUDE_35_SONNET,
                    "litellm_params": {
                        "model": CommonLLMNames.CLAUDE_35_SONNET,
                        "max_tokens": 4096,
                    },
                },
                {
                    "model_name": CommonLLMNames.GPT_4O,
                    "litellm_params": {
                        "model": CommonLLMNames.GPT_4O,
                        "temperature": 1,
                        "max_tokens": 4096,
                    },
                },
            ],
        },
    )
    assert llm.name == model_name

    messages = [
        Message(role="user", content="Hi"),
    ]

    results = await llm.acompletion(messages)

    assert (
        results[0].model == model_name
    ), f"The response should be generated with the selected model {model_name}"
    model_names = [model["model_name"] for model in llm.config["model_list"]]
    assert (
        model_name in model_names
    ), f"The model {model_name} should be in the model list"
    assert (
        CommonLLMNames.CLAUDE_35_SONNET in model_names
    ), f"The model {CommonLLMNames.CLAUDE_35_SONNET} should also be in the model list"


def test_constructing_llm_w_multiple_models_2():
    model_name = "gemini"
    model_list = [
        {
            "model_name": CommonLLMNames.CLAUDE_35_SONNET,
            "litellm_params": {
                "model": CommonLLMNames.CLAUDE_35_SONNET,
                "max_tokens": 4096,
            },
        },
    ]
    with pytest.raises(
        ValueError,
        match=f"Provided name '{model_name}' not found in model_list. Available models: ",
    ):
        LiteLLMModel(
            name=model_name,
            config={
                "model_list": model_list,
            },
        )


@pytest.mark.asyncio
async def test_rate_limit_on_multiple_models():
    llm = LiteLLMModel(
        config={
            "model_list": [
                {
                    "model_name": CommonLLMNames.CLAUDE_35_SONNET,
                    "litellm_params": {
                        "model": CommonLLMNames.CLAUDE_35_SONNET,
                        "max_tokens": 4096,
                    },
                },
                {
                    "model_name": CommonLLMNames.GPT_4O,
                    "litellm_params": {
                        "model": CommonLLMNames.GPT_4O,
                        "temperature": 1,
                        "max_tokens": 4096 * 2,
                    },
                },
            ],
            "rate_limit": {
                CommonLLMNames.CLAUDE_35_SONNET: "35/minute",
                CommonLLMNames.GPT_4O: "10000/minute",
            },
        }
    )

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the meaning of life?"),
    ]

    results = await llm.acompletion(messages)

    assert (
        results[0].model == CommonLLMNames.CLAUDE_35_SONNET
    ), f"The response should be generated with {CommonLLMNames.CLAUDE_35_SONNET}"
    assert (
        llm.name == CommonLLMNames.GPT_4O
    ), f"After the response was generated, we should have hit the rate limit for {CommonLLMNames.CLAUDE_35_SONNET} and changed to {CommonLLMNames.GPT_4O}"
