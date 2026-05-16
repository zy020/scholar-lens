import pytest
from scholar_lens.frontend.components.config_panel import ConfigPanelState


class TestConfigPanelState:
    def test_create_defaults(self):
        panel = ConfigPanelState()
        assert panel.llm_api_key == ""
        assert panel.llm_model == "gpt-4o-mini"
        assert panel.is_configured is False

    def test_configure(self):
        panel = ConfigPanelState()
        panel.llm_api_key = "test-key"
        panel.embedding_api_key = "test-emb"
        panel.is_configured = True
        assert panel.is_configured is True

    def test_to_request_dict(self):
        panel = ConfigPanelState(
            llm_api_key="key",
            llm_model="gpt-4o",
            embedding_api_key="emb",
        )
        d = panel.to_request_dict()
        assert d["llm_api_key"] == "key"
        assert d["llm_model"] == "gpt-4o"
