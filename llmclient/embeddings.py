from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import litellm
import numpy as np
import tiktoken
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from llmclient.constants import CHARACTERS_PER_TOKEN_ASSUMPTION, MODEL_COST_MAP
from llmclient.rate_limiter import GLOBAL_LIMITER


def get_litellm_retrying_config(timeout: float = 60.0) -> dict[str, Any]:
    """Get retrying configuration for litellm.acompletion and litellm.aembedding."""
    return {"num_retries": 3, "timeout": timeout}


class EmbeddingModes(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingModel(ABC, BaseModel):
    name: str
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional `rate_limit` key, value must be a RateLimitItem or RateLimitItem"
            " string for parsing"
        ),
    )

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        if "rate_limit" in self.config:
            await GLOBAL_LIMITER.try_acquire(
                ("client", self.name),
                self.config["rate_limit"],
                weight=max(int(token_count), 1),
                **kwargs,
            )

    def set_mode(self, mode: EmbeddingModes) -> None:
        """Several embedding models have a 'mode' or prompt which affects output."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass


class LiteLLMEmbeddingModel(EmbeddingModel):

    name: str = Field(default="text-embedding-3-small")
    config: dict[str, Any] = Field(
        default_factory=dict,  # See below field_validator for injection of kwargs
        description=(
            "The optional `rate_limit` key's value must be a RateLimitItem or"
            " RateLimitItem string for parsing. The optional `kwargs` key is keyword"
            " arguments to pass to the litellm.aembedding function. Note that LiteLLM's"
            " Router is not used here."
        ),
    )

    @field_validator("config", mode="before")
    @classmethod
    def set_up_default_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "kwargs" not in value:
            value["kwargs"] = get_litellm_retrying_config(
                timeout=120,  # 2-min timeout seemed reasonable
            )
        return value

    def _truncate_if_large(self, texts: list[str]) -> list[str]:
        """Truncate texts if they are too large by using litellm cost map."""
        if self.name not in MODEL_COST_MAP:
            return texts
        max_tokens = MODEL_COST_MAP[self.name]["max_input_tokens"]
        # heuristic about ratio of tokens to characters
        conservative_char_token_ratio = 3
        maybe_too_large = max_tokens * conservative_char_token_ratio
        if any(len(t) > maybe_too_large for t in texts):
            try:
                enct = tiktoken.encoding_for_model("cl100k_base")
                enc_batch = enct.encode_ordinary_batch(texts)
                return [enct.decode(t[:max_tokens]) for t in enc_batch]
            except KeyError:
                return [t[: max_tokens * conservative_char_token_ratio] for t in texts]

        return texts

    async def embed_documents(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        texts = self._truncate_if_large(texts)
        N = len(texts)
        embeddings = []
        for i in range(0, N, batch_size):

            await self.check_rate_limit(
                sum(
                    len(t) / CHARACTERS_PER_TOKEN_ASSUMPTION
                    for t in texts[i : i + batch_size]
                )
            )

            response = await litellm.aembedding(
                self.name,
                input=texts[i : i + batch_size],
                **self.config.get("kwargs", {}),
            )
            embeddings.extend([e["embedding"] for e in response.data])

        return embeddings


class SparseEmbeddingModel(EmbeddingModel):
    """This is a very simple keyword search model - probably best to be mixed with others."""

    name: str = "sparse"
    ndim: int = 256
    enc: Any = Field(default_factory=lambda: tiktoken.get_encoding("cl100k_base"))

    async def embed_documents(self, texts) -> list[list[float]]:
        enc_batch = self.enc.encode_ordinary_batch(texts)
        # now get frequency of each token rel to length
        return [
            np.bincount([xi % self.ndim for xi in x], minlength=self.ndim).astype(float)  # type: ignore[misc]
            / len(x)
            for x in enc_batch
        ]


class HybridEmbeddingModel(EmbeddingModel):
    name: str = "hybrid-embed"
    models: list[EmbeddingModel]

    async def embed_documents(self, texts):
        all_embeds = await asyncio.gather(
            *[m.embed_documents(texts) for m in self.models]
        )
        return np.concatenate(all_embeds, axis=1)

    def set_mode(self, mode: EmbeddingModes) -> None:
        # Set mode for all component models
        for model in self.models:
            model.set_mode(mode)


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """An embedding model using SentenceTransformers."""

    name: str = Field(default="multi-qa-MiniLM-L6-cos-v1")
    config: dict[str, Any] = Field(default_factory=dict)
    _model: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Please install fh-llm-client[local] to use"
                " SentenceTransformerEmbeddingModel."
            ) from exc

        self._model = SentenceTransformer(self.name)

    def set_mode(self, mode: EmbeddingModes) -> None:
        # SentenceTransformer does not support different modes.
        pass

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Asynchronously embed a list of documents using SentenceTransformer.

        Args:
            texts: A list of text documents to embed.

        Returns:
            A list of embedding vectors.
        """
        # Extract additional configurations if needed
        batch_size = self.config.get("batch_size", 32)
        device = self.config.get("device", "cpu")

        # Update the model's device if necessary
        if device:
            self._model.to(device)

        # Run the synchronous encode method in a thread pool to avoid blocking the event loop.
        embeddings = await asyncio.to_thread(
            lambda: self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,  # Disabled progress bar
                batch_size=batch_size,
                device=device,
            ),
        )
        # If embeddings are returned as numpy arrays, convert them to lists.
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings


def embedding_model_factory(embedding: str, **kwargs) -> EmbeddingModel:
    """
    Factory function to create an appropriate EmbeddingModel based on the embedding string.

    Supports:
    - SentenceTransformer models prefixed with "st-" (e.g., "st-multi-qa-MiniLM-L6-cos-v1")
    - LiteLLM models (default if no prefix is provided)
    - Hybrid embeddings prefixed with "hybrid-", contains a sparse and a dense model

    Args:
        embedding: The embedding model identifier. Supports prefixes like "st-" for SentenceTransformer
                   and "hybrid-" for combining multiple embedding models.
        **kwargs: Additional keyword arguments for the embedding model.
    """
    embedding = embedding.strip()  # Remove any leading/trailing whitespace

    if embedding.startswith("hybrid-"):
        # Extract the component embedding identifiers after "hybrid-"
        dense_name = embedding[len("hybrid-") :]

        if not dense_name:
            raise ValueError(
                "Hybrid embedding must contain at least one component embedding."
            )

        # Recursively create each component embedding model
        dense_model = embedding_model_factory(dense_name, **kwargs)
        sparse_model = SparseEmbeddingModel(**kwargs)

        return HybridEmbeddingModel(models=[dense_model, sparse_model])

    if embedding.startswith("st-"):
        # Extract the SentenceTransformer model name after "st-"
        model_name = embedding[len("st-") :].strip()
        if not model_name:
            raise ValueError(
                "SentenceTransformer model name must be specified after 'st-'."
            )

        return SentenceTransformerEmbeddingModel(
            name=model_name,
            config=kwargs,
        )

    if embedding.startswith("litellm-"):
        # Extract the LiteLLM model name after "litellm-"
        model_name = embedding[len("litellm-") :].strip()
        if not model_name:
            raise ValueError("model name must be specified after 'litellm-'.")

        return LiteLLMEmbeddingModel(
            name=model_name,
            config=kwargs,
        )

    if embedding == "sparse":
        return SparseEmbeddingModel(**kwargs)

    # Default to LiteLLMEmbeddingModel if no special prefix is found
    return LiteLLMEmbeddingModel(name=embedding, config=kwargs)
