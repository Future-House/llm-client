[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=plastic)]()
[![tests](https://github.com/Future-House/llm-client/actions/workflows/test.yaml/badge.svg?style=plastic)](https://github.com/Future-House/llm-client)
[![PyPI version](https://badge.fury.io/py/fh-llm-client.svg?style=plastic)](https://badge.fury.io/py/fh-llm-client)

# llm-client

A Python library for interacting with Large Language Models (LLMs) through a unified interface.


## Installation
```bash
pip install fh-llm-client
```

## Quick start

```python
from llmclient import LiteLLMModel
from aviary import Message

llm = LiteLLMModel(name="gpt-4o-mini", config={})
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What is the meaning of life?"),
]
completions = await llm.call(messages)

```

## Documentation

### LLMs

An LLM is a class that inherits from `LLMModel` and implements the following methods:
- `achat`
- `achat_iter`
- `acompletion`
- `acompletion_iter`
- `check_rate_limit`

These methos are used by the base class `LLMModel` to implement the LLM interface.

#### LLMModel

An `LLMModel` implements `call`, which receives a list of `aviary.Message`s and returns a list of `LLMResult`s. `LLMModel.call` checks if the llm being used is a completion or a chat model and calls the appropriate method.

#### LiteLLMModel

`LiteLLMModel` wrapps `LiteLLM` API usage within our `LLMModel` interface. It receives a `name` parameter, which is the name of the model to use and a `config` parameter, which is a dictionary of configuration options for the model following the [LiteLLM configuration schema](https://docs.litellm.ai/docs/routing). Common parameters such as `temperature`, `max_token`, and `n` (the number of completions to return) can be passed as part of the `config` dictionary.

```python
from llmclient import LiteLLMModel

config = {
    "model_list": [
        {
            "name": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "frequency_penalty": 1.5,
            "top_p": 0.9,
            "max_tokens": 512,
        }
    ]
    'n': 5,
    'temperature': 0.1,
}

llm = LiteLLMModel(name="gpt-4o", config=config)

```

### Embeddings

This client also includes embedding models. 
An embedding model is a class that inherits from `EmbeddingModel` and implements the `embed_documents` method, which receives a list of strings and returns a list with a list of floats (the embeddings)for each string.

- `LiteLLMEmbeddingModel`
- `SparseEmbeddingModel`
- `HybridEmbeddingModel`
- `SentenceTransformerEmbeddingModel`

### Cost tracking

Cost tracking is supported in two different ways:
1. Calls to the LLM returns the token usage for each call in `LLMResult.prompt_count` and `LLMResult.completion_count`. Additionally, `LLMResult.cost` can be used to get a cost estimate for the call in USD.
2. A global cost tracker is maintained in `GLOBAL_COST_TRACKER` and can be enabled or disabled using `enable_cost_tracking()` and `cost_tracking_ctx()`.

### Rate limiting

### Tool calling
