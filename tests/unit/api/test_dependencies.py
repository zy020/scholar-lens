from pathlib import Path

from scholar_lens.api import deps
from scholar_lens.core.settings import EmbeddingConfig, LLMConfig, Settings


def test_document_store_uses_configured_data_dir(tmp_path, monkeypatch):
    settings = Settings(
        _env_file="",
        data_dir=tmp_path / "runtime-data",
        llm=LLMConfig(),
        embedding=EmbeddingConfig(),
    )
    monkeypatch.setattr(deps, "get_settings", lambda: settings)
    deps.get_document_store.cache_clear()

    store = deps.get_document_store()

    assert Path(store.root) == tmp_path / "runtime-data" / "documents"
