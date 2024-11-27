import pytest

from llmclient.embeddings import (
    MODEL_COST_MAP,
    HybridEmbeddingModel,
    LiteLLMEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    SparseEmbeddingModel,
    embedding_model_factory,
)


class TestLiteLLMEmbeddingModel:
    @pytest.fixture
    def embedding_model(self):
        return LiteLLMEmbeddingModel()

    def test_default_config_injection(self, embedding_model):
        # field_validator is only triggered if the attribute is passed
        embedding_model = LiteLLMEmbeddingModel(config={})

        config = embedding_model.config
        assert "kwargs" in config
        assert config["kwargs"]["timeout"] == 120

    def test_truncate_if_large_no_truncation(self, embedding_model):
        texts = ["short text", "another short text"]
        truncated_texts = embedding_model._truncate_if_large(texts)
        assert truncated_texts == texts

    def test_truncate_if_large_with_truncation(self, embedding_model, mocker):
        texts = ["a" * 10000, "b" * 10000]
        mocker.patch.dict(
            MODEL_COST_MAP, {embedding_model.name: {"max_input_tokens": 100}}
        )
        mocker.patch(
            "tiktoken.encoding_for_model",
            return_value=mocker.Mock(
                encode_ordinary_batch=lambda texts: [[1] * 1000 for _ in texts],
                decode=lambda text: "truncated text",  # noqa: ARG005
            ),
        )
        truncated_texts = embedding_model._truncate_if_large(texts)
        assert truncated_texts == ["truncated text", "truncated text"]

    def test_truncate_if_large_key_error(self, embedding_model, mocker):
        texts = ["a" * 10000, "b" * 10000]
        mocker.patch.dict(
            MODEL_COST_MAP, {embedding_model.name: {"max_input_tokens": 100}}
        )
        mocker.patch("tiktoken.encoding_for_model", side_effect=KeyError)
        truncated_texts = embedding_model._truncate_if_large(texts)
        assert truncated_texts == ["a" * 300, "b" * 300]

    @pytest.mark.asyncio
    async def test_embed_documents(self, embedding_model, mocker):
        texts = ["short text", "another short text"]
        mocker.patch(
            "llmclient.embeddings.LiteLLMEmbeddingModel._truncate_if_large",
            return_value=texts,
        )
        mocker.patch(
            "llmclient.embeddings.LiteLLMEmbeddingModel.check_rate_limit",
            return_value=None,
        )
        mock_response = mocker.Mock()
        mock_response.data = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
        mocker.patch("litellm.aembedding", return_value=mock_response)

        embeddings = await embedding_model.embed_documents(texts)
        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_embedding_model_factory_sentence_transformer() -> None:
    """Test that the factory creates a SentenceTransformerEmbeddingModel when given an 'st-' prefix."""
    embedding = "st-multi-qa-MiniLM-L6-cos-v1"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, SentenceTransformerEmbeddingModel
    ), "Factory did not create SentenceTransformerEmbeddingModel"
    assert model.name == "multi-qa-MiniLM-L6-cos-v1", "Incorrect model name assigned"

    # Test embedding functionality
    texts = ["Hello world", "Test sentence"]
    embeddings = await model.embed_documents(texts)
    assert len(embeddings) == 2, "Incorrect number of embeddings returned"
    assert all(
        isinstance(embed, list) for embed in embeddings
    ), "Embeddings are not in list format"
    assert all(len(embed) > 0 for embed in embeddings), "Embeddings should not be empty"


@pytest.mark.asyncio
async def test_embedding_model_factory_hybrid_with_sentence_transformer() -> None:
    """Test that the factory creates a HybridEmbeddingModel containing a SentenceTransformerEmbeddingModel."""
    embedding = "hybrid-st-multi-qa-MiniLM-L6-cos-v1"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, HybridEmbeddingModel
    ), "Factory did not create HybridEmbeddingModel"
    assert len(model.models) == 2, "Hybrid model should contain two component models"
    assert isinstance(
        model.models[0], SentenceTransformerEmbeddingModel
    ), "First component should be SentenceTransformerEmbeddingModel"
    assert isinstance(
        model.models[1], SparseEmbeddingModel
    ), "Second component should be SparseEmbeddingModel"

    # Test embedding functionality
    texts = ["Hello world", "Test sentence"]
    embeddings = await model.embed_documents(texts)
    assert len(embeddings) == 2, "Incorrect number of embeddings returned"
    expected_length = len((await model.models[0].embed_documents(texts))[0]) + len(
        (await model.models[1].embed_documents(texts))[0]
    )
    assert all(
        len(embed) == expected_length for embed in embeddings
    ), "Embeddings do not match expected combined length"


@pytest.mark.asyncio
async def test_embedding_model_factory_invalid_st_prefix() -> None:
    """Test that the factory raises a ValueError when 'st-' prefix is provided without a model name."""
    embedding = "st-"
    with pytest.raises(
        ValueError,
        match="SentenceTransformer model name must be specified after 'st-'.",
    ):
        embedding_model_factory(embedding)


@pytest.mark.asyncio
async def test_embedding_model_factory_unknown_prefix() -> None:
    """Test that the factory defaults to LiteLLMEmbeddingModel when an unknown prefix is provided."""
    embedding = "unknown-prefix-model"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, LiteLLMEmbeddingModel
    ), "Factory did not default to LiteLLMEmbeddingModel for unknown prefix"
    assert model.name == "unknown-prefix-model", "Incorrect model name assigned"


@pytest.mark.asyncio
async def test_embedding_model_factory_sparse() -> None:
    """Test that the factory creates a SparseEmbeddingModel when 'sparse' is provided."""
    embedding = "sparse"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, SparseEmbeddingModel
    ), "Factory did not create SparseEmbeddingModel"
    assert model.name == "sparse", "Incorrect model name assigned"


@pytest.mark.asyncio
async def test_embedding_model_factory_litellm() -> None:
    """Test that the factory creates a LiteLLMEmbeddingModel when 'litellm-' prefix is provided."""
    embedding = "litellm-text-embedding-3-small"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, LiteLLMEmbeddingModel
    ), "Factory did not create LiteLLMEmbeddingModel"
    assert model.name == "text-embedding-3-small", "Incorrect model name assigned"


@pytest.mark.asyncio
async def test_embedding_model_factory_default() -> None:
    """Test that the factory defaults to LiteLLMEmbeddingModel when no known prefix is provided."""
    embedding = "default-model"
    model = embedding_model_factory(embedding)
    assert isinstance(
        model, LiteLLMEmbeddingModel
    ), "Factory did not default to LiteLLMEmbeddingModel"
    assert model.name == "default-model", "Incorrect model name assigned"
