import os
import pytest
from scholar_lens.core.settings import Settings, LLMConfig, EmbeddingConfig


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig(api_key="test-key")
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.3

    def test_custom_values(self):
        cfg = LLMConfig(
            api_key="key",
            base_url="http://localhost:11434/v1",
            model="qwen2.5",
            temperature=0.7,
        )
        assert cfg.base_url == "http://localhost:11434/v1"


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig(api_key="test-key")
        assert cfg.model == "text-embedding-3-small"

    def test_embeds_separate_from_llm(self):
        llm = LLMConfig(api_key="llm-key")
        emb = EmbeddingConfig(api_key="emb-key", base_url="http://other/v1")
        assert llm.api_key != emb.api_key
        assert emb.base_url == "http://other/v1"


class TestSettings:
    def test_load_from_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "LLM_API_KEY=env-key\n"
            "LLM_MODEL=custom-model\n"
            "EMBEDDING_API_KEY=emb-key\n"
        )
        s = Settings(_env_file=str(env_file))
        assert s.llm.api_key == "env-key"
        assert s.llm.model == "custom-model"
        assert s.embedding.api_key == "emb-key"

    def test_reranker_optional(self):
        s = Settings(
            llm=LLMConfig(api_key="k"),
            embedding=EmbeddingConfig(api_key="k"),
        )
        assert s.reranker is None

    def test_reranker_configured(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "LLM_API_KEY=k\n"
            "EMBEDDING_API_KEY=k\n"
            "RERANKER_MODEL=bge-reranker-m3\n"
        )
        s = Settings(_env_file=str(env_file))
        assert s.reranker is not None
        assert s.reranker.model == "bge-reranker-m3"
