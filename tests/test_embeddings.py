import pytest

from llmclient.embeddings import MODEL_COST_MAP, LiteLLMEmbeddingModel


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
