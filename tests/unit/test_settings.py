import pytest
from scholar_lens.core.settings import EmbeddingConfig, LLMConfig, RerankerConfig, Settings, VisionConfig


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig(api_key="test-key")
        assert cfg.model == ""

    def test_custom_values(self):
        cfg = LLMConfig(api_key="key", base_url="http://localhost:11434/v1", model="qwen2.5", temperature=0.7)
        assert cfg.base_url == "http://localhost:11434/v1"


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig(api_key="test-key")
        assert cfg.model == ""

    def test_embeds_separate_from_llm(self):
        llm = LLMConfig(api_key="llm-key")
        emb = EmbeddingConfig(api_key="emb-key", base_url="http://other/v1")
        assert llm.api_key != emb.api_key


class TestSettings:
    def test_shared_credentials_inherited(self):
        s = Settings(
            _env_file="",  # disable real .env
            api_key="shared-key",
            base_url="https://shared.com/v1",
            llm=LLMConfig(model="llm-model"),
            embedding=EmbeddingConfig(model="emb-model"),
        )
        assert s.llm.api_key == "shared-key"
        assert s.llm.base_url == "https://shared.com/v1"
        assert s.embedding.api_key == "shared-key"

    def test_per_model_override(self):
        s = Settings(
            _env_file="",
            api_key="shared-key",
            base_url="https://shared.com/v1",
            llm=LLMConfig(model="llm", api_key="llm-specific-key"),
            embedding=EmbeddingConfig(model="emb", base_url="https://emb-only.com/v1"),
        )
        assert s.llm.api_key == "llm-specific-key"
        assert s.llm.base_url == "https://shared.com/v1"
        assert s.embedding.api_key == "shared-key"
        assert s.embedding.base_url == "https://emb-only.com/v1"

    def test_reranker_and_vision_optional_when_not_configured(self):
        """Without RERANKER__MODEL/VISION__MODEL in env, they should be None."""
        s = Settings(
            _env_file="",
            api_key="sk",
            llm=LLMConfig(model="test"),
            embedding=EmbeddingConfig(model="test"),
        )
        assert s.reranker is None
        assert s.vision is None

    def test_reranker_inherits_credentials(self):
        from scholar_lens.core.settings import RerankerConfig

        s = Settings(
            _env_file="",
            api_key="sk",
            base_url="https://api.com/v1",
            llm=LLMConfig(model="llm"),
            embedding=EmbeddingConfig(model="emb"),
            reranker=RerankerConfig(model="rerank-model"),
        )
        assert s.reranker is not None
        assert s.reranker.api_key == "sk"
        assert s.reranker.base_url == "https://api.com/v1"

    def test_vision_inherits_credentials(self):
        from scholar_lens.core.settings import VisionConfig

        s = Settings(
            _env_file="",
            api_key="sk",
            base_url="https://api.com/v1",
            llm=LLMConfig(model="llm"),
            embedding=EmbeddingConfig(model="emb"),
            vision=VisionConfig(model="vision-model"),
        )
        assert s.vision is not None
        assert s.vision.api_key == "sk"

    def test_env_file_loads_shared_credentials(self, tmp_path, monkeypatch):
        for key in (
            "API_KEY",
            "BASE_URL",
            "LLM__API_KEY",
            "LLM__BASE_URL",
            "LLM__MODEL",
            "EMBEDDING__API_KEY",
            "EMBEDDING__BASE_URL",
            "EMBEDDING__MODEL",
        ):
            monkeypatch.delenv(key, raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "API_KEY=env-shared-key\n"
            "BASE_URL=https://env.com/v1\n"
            "LLM__MODEL=env-llm\n"
            "EMBEDDING__MODEL=env-emb\n"
        )
        s = Settings(_env_file=str(env_file))
        assert s.llm.api_key == "env-shared-key"
        assert s.llm.model == "env-llm"
        assert s.embedding.model == "env-emb"

    def test_flat_accessor_backward_compat(self):
        s = Settings(_env_file="", api_key="k", llm=LLMConfig(model="test-model"))
        assert s.llm_api_key == "k"
        assert s.llm_model == "test-model"

    def test_memory_llm_compression_flag_defaults_false(self):
        s = Settings(_env_file="")
        assert s.memory_llm_compression_enabled is False

    def test_optional_model_separate_empty_credentials_do_not_inherit_shared(self):
        s = Settings(
            _env_file="",
            api_key="shared-key",
            base_url="https://shared.example/v1",
            reranker=RerankerConfig(model="rerank-model", api_key="", base_url=""),
            reranker_use_separate=True,
            vision=VisionConfig(model="vision-model", api_key="", base_url=""),
            vision_use_separate=True,
        )

        assert s.reranker.api_key == ""
        assert s.reranker.base_url == ""
        assert s.vision.api_key == ""
        assert s.vision.base_url == ""
