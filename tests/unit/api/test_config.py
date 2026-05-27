"""Tests for config routes per Ticket 4."""
import pytest

pytest.importorskip("fastapi")

from scholar_lens.api.main import create_app
from scholar_lens.api.deps import get_settings
from scholar_lens.api.routes import config
from tests.unit.api.helpers import ASGITestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("LLM_QUALITY_ENABLED", "false")
    monkeypatch.setenv("VISION_ENHANCEMENT_ENABLED", "false")
    app = create_app()
    yield ASGITestClient(app)
    get_settings.cache_clear()


class TestConfigRoutes:
    def test_get_config(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "llm_model" in data
        assert "requires_restart" in data
        assert data["auto_ocr_enabled"] is True
        assert data["llm_quality_enabled"] is False
        assert data["vision_enhancement_enabled"] is False
        assert isinstance(data["reranker_active"], bool)
        assert data["reranker_mode"] in {"rule", "model"}

    def test_post_config_returns_200(self, client):
        r = client.post("/api/config", json={
            "llm_model": "test-model",
            "llm_api_key": "test-key",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["requires_restart"] is False

    def test_post_config_empty_is_ok(self, client):
        r = client.post("/api/config", json={})
        assert r.status_code == 200

    def test_reranker_and_vision_default_enabled_with_shared_credentials(self, client):
        r = client.post("/api/config", json={
            "api_key": "shared-key",
            "base_url": "https://shared.example/v1",
            "llm_model": "llm-model",
            "embedding_model": "embedding-model",
            "reranker_model": "reranker-model",
            "vision_model": "vision-model",
        })
        assert r.status_code == 200
        settings = get_settings()
        assert settings.reranker is not None
        assert settings.reranker.api_key == "shared-key"
        assert settings.reranker.base_url == "https://shared.example/v1"
        assert settings.vision is not None
        assert settings.vision.api_key == "shared-key"
        assert settings.vision.base_url == "https://shared.example/v1"

    def test_can_disable_reranker_and_vision(self, client):
        r = client.post("/api/config", json={
            "api_key": "shared-key",
            "llm_model": "llm-model",
            "embedding_model": "embedding-model",
            "reranker_enabled": False,
            "vision_enabled": False,
            "reranker_model": "reranker-model",
            "vision_model": "vision-model",
        })
        assert r.status_code == 200
        settings = get_settings()
        assert settings.reranker is None
        assert settings.vision is None

    def test_separate_flags_control_credentials(self, client):
        r = client.post("/api/config", json={
            "api_key": "shared-key",
            "base_url": "https://shared.example/v1",
            "llm_model": "llm-model",
            "embedding_model": "embedding-model",
            "reranker_model": "reranker-model",
            "reranker_use_separate": True,
            "reranker_api_key": "reranker-key",
            "reranker_base_url": "https://reranker.example/v1",
            "vision_model": "vision-model",
            "vision_use_separate": False,
            "vision_api_key": "ignored-key",
            "vision_base_url": "https://ignored.example/v1",
        })
        assert r.status_code == 200
        settings = get_settings()
        assert settings.reranker.api_key == "reranker-key"
        assert settings.reranker.base_url == "https://reranker.example/v1"
        assert settings.vision.api_key == "shared-key"
        assert settings.vision.base_url == "https://shared.example/v1"

    def test_test_connection(self, client):
        r = client.post("/api/config/test")
        assert r.status_code == 200
        assert r.json()["status"] in ("ok", "not_configured")

    def test_post_config_updates_reranker_and_vision_credentials(self, client):
        r = client.post("/api/config", json={
            "api_key": "shared-key",
            "base_url": "https://shared.example/v1",
            "reranker_api_key": "rerank-key",
            "reranker_base_url": "https://rerank.example/v1",
            "reranker_model": "rerank-model",
            "reranker_use_separate": True,
            "vision_api_key": "vision-key",
            "vision_base_url": "https://vision.example/v1",
            "vision_model": "vision-model",
            "vision_use_separate": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["reranker_available"] is True
        assert data["reranker_active"] is True
        assert data["reranker_mode"] == "model"
        assert data["vision_available"] is True

        settings = get_settings()
        assert settings.reranker is not None
        assert settings.reranker.api_key == "rerank-key"
        assert settings.reranker.base_url == "https://rerank.example/v1"
        assert settings.reranker.model == "rerank-model"
        assert settings.vision is not None
        assert settings.vision.api_key == "vision-key"
        assert settings.vision.base_url == "https://vision.example/v1"
        assert settings.vision.model == "vision-model"

    def test_post_config_updates_enhancement_policy_flags(self, client):
        r = client.post("/api/config", json={
            "llm_quality_enabled": True,
            "vision_enhancement_enabled": True,
        })

        assert r.status_code == 200
        data = r.json()
        assert data["auto_ocr_enabled"] is True
        assert data["llm_quality_enabled"] is True
        assert data["vision_enhancement_enabled"] is True

        settings = get_settings()
        assert settings.auto_ocr_enabled is True
        assert settings.llm_quality_enabled is True
        assert settings.vision_enhancement_enabled is True

    def test_post_config_flags_do_not_overwrite_existing_model_names(self, client):
        first = client.post("/api/config", json={
            "api_key": "shared-key",
            "base_url": "https://shared.example/v1",
            "llm_model": "real-answer-model",
            "embedding_model": "real-embedding-model",
        })
        assert first.status_code == 200

        second = client.post("/api/config", json={
            "llm_quality_enabled": True,
        })

        assert second.status_code == 200
        settings = get_settings()
        assert settings.llm.model == "real-answer-model"
        assert settings.embedding.model == "real-embedding-model"
        assert second.json()["llm_model"] == "real-answer-model"
        assert second.json()["embedding_model"] == "real-embedding-model"

    def test_post_config_persists_current_settings_to_env_file(self, client):
        r = client.post("/api/config", json={
            "api_key": "shared-key",
            "base_url": "https://shared.example/v1",
            "llm_model": "persisted-llm",
            "embedding_model": "persisted-embedding",
            "reranker_model": "persisted-reranker",
            "vision_model": "persisted-vision",
            "llm_quality_enabled": True,
            "vision_enhancement_enabled": True,
        })

        assert r.status_code == 200
        env_text = config.ENV_PATH.read_text(encoding="utf-8")
        assert "API_KEY='shared-key'" in env_text
        assert "BASE_URL='https://shared.example/v1'" in env_text
        assert "LLM__MODEL='persisted-llm'" in env_text
        assert "EMBEDDING__MODEL='persisted-embedding'" in env_text
        assert "RERANKER__MODEL='persisted-reranker'" in env_text
        assert "VISION__MODEL='persisted-vision'" in env_text
        assert "LLM_QUALITY_ENABLED='true'" in env_text
        assert "VISION_ENHANCEMENT_ENABLED='true'" in env_text

    def test_post_config_persists_disabled_optional_models_as_empty_values(self, client):
        r = client.post("/api/config", json={
            "api_key": "shared-key",
            "reranker_enabled": False,
            "vision_enabled": False,
        })

        assert r.status_code == 200
        env_text = config.ENV_PATH.read_text(encoding="utf-8")
        assert "RERANKER__MODEL=''" in env_text
        assert "VISION__MODEL=''" in env_text

    def test_post_config_resets_chat_runtime_and_reports_live_update(self, client, monkeypatch):
        from scholar_lens.api.routes import chat

        called = {"reset": 0}

        def fake_reset():
            called["reset"] += 1

        monkeypatch.setattr(chat, "reset_chat_runtime", fake_reset)

        r = client.post("/api/config", json={
            "api_key": "new-key",
            "base_url": "https://runtime.example/v1",
            "llm_model": "runtime-model",
            "embedding_model": "embedding-model",
        })

        assert r.status_code == 200
        data = r.json()
        assert data["requires_restart"] is False
        assert data["llm_base_url"] == "https://runtime.example/v1"
        assert data["llm_configured"] is True
        assert called["reset"] == 1

    def test_post_config_allows_clearing_existing_values(self, client):
        first = client.post("/api/config", json={
            "api_key": "shared-key",
            "base_url": "https://shared.example/v1",
            "llm_model": "model-a",
            "embedding_model": "embedding-a",
        })
        assert first.status_code == 200

        second = client.post("/api/config", json={
            "api_key": "",
            "base_url": "",
            "llm_api_key": "",
            "llm_base_url": "",
            "llm_model": "",
            "embedding_api_key": "",
            "embedding_base_url": "",
            "embedding_model": "",
            "llm_use_separate": True,
            "embedding_use_separate": True,
        })

        assert second.status_code == 200
        settings = get_settings()
        assert settings.api_key == ""
        assert settings.base_url == ""
        assert settings.llm.api_key == ""
        assert settings.llm.base_url == ""
        assert settings.llm.model == ""
        assert settings.embedding.api_key == ""
        assert settings.embedding.base_url == ""
        assert settings.embedding.model == ""
        assert second.json()["llm_configured"] is False
        assert second.json()["embedding_configured"] is False

    def test_config_reports_optional_model_available_only_when_fully_configured(self, client):
        r = client.post("/api/config", json={
            "vision_model": "vision-only-model",
            "vision_use_separate": True,
            "vision_api_key": "",
            "vision_base_url": "",
            "reranker_model": "reranker-only-model",
            "reranker_use_separate": True,
            "reranker_api_key": "",
            "reranker_base_url": "",
        })

        assert r.status_code == 200
        data = r.json()
        assert data["vision_available"] is False
        assert data["reranker_available"] is False
        assert data["reranker_active"] is False
        assert data["reranker_mode"] == "rule"
