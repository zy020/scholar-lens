import pytest
from unittest.mock import patch, MagicMock
from scholar_lens.core.llm_factory import LLMFactory
from scholar_lens.core.settings import LLMConfig, EmbeddingConfig, Settings


class TestLLMFactory:
    def test_create_chat_llm(self):
        config = LLMConfig(api_key="test-key", model="gpt-4o-mini")
        factory = LLMFactory(config)
        llm = factory.create_chat_llm()
        assert llm is not None
        assert llm.model_name == "gpt-4o-mini"

    def test_create_chat_llm_custom_base_url(self):
        config = LLMConfig(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="qwen2.5",
        )
        factory = LLMFactory(config)
        llm = factory.create_chat_llm()
        assert llm.openai_api_base == "http://localhost:11434/v1"

    def test_create_embeddings(self):
        config = EmbeddingConfig(api_key="test-key", model="text-embedding-3-small")
        factory = LLMFactory(config)
        emb = factory.create_embeddings()
        assert emb is not None

    def test_create_embeddings_custom_base_url(self):
        config = EmbeddingConfig(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="bge-m3",
        )
        factory = LLMFactory(config)
        emb = factory.create_embeddings()
        assert emb is not None

    def test_factory_from_settings(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("LLM_API_KEY=k\nEMBEDDING_API_KEY=k\n")
        settings = Settings(_env_file=str(env_file))
        factory = LLMFactory.from_settings(settings)
        assert factory is not None
