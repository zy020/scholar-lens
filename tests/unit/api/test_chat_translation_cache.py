from unittest.mock import AsyncMock

from scholar_lens.api.main import create_app
from scholar_lens.api.routes import chat
from scholar_lens.api.schemas import DocumentStatus
from scholar_lens.core.settings import EmbeddingConfig, LLMConfig, Settings
from scholar_lens.rag.document_store import DocumentStore
from tests.unit.api.helpers import ASGITestClient


class FakeResponse:
    content = "缓存后的中文翻译"


def test_section_translation_is_cached_per_document_section_and_model(tmp_path, monkeypatch):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.update_status(doc.doc_id, DocumentStatus.ready)

    settings = Settings(
        _env_file=None,
        api_key="test-key",
        llm=LLMConfig(api_key="test-key", model="translate-model"),
        embedding=EmbeddingConfig(),
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = FakeResponse()

    class FakeFactory:
        def create(self, streaming=False):
            return mock_llm

    async def noop_memory_event(*args, **kwargs):
        return None

    from scholar_lens.core import llm_factory

    monkeypatch.setattr(chat, "get_settings", lambda: settings)
    monkeypatch.setattr(chat, "get_document_store", lambda: store)
    monkeypatch.setattr(chat, "_record_memory_event", noop_memory_event)
    monkeypatch.setattr(llm_factory.ChatLLMFactory, "from_settings", lambda settings: FakeFactory())

    client = ASGITestClient(create_app())
    payload = {
        "doc_id": doc.doc_id,
        "section_id": "intro",
        "mode": "translate",
        "message": "Translate this section.\n\nSelf-attention connects tokens.",
    }

    first = client.post("/api/chat/explain", json=payload)
    second = client.post("/api/chat/explain", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["content"] == "缓存后的中文翻译"
    assert second.json()["content"] == "缓存后的中文翻译"
    assert mock_llm.ainvoke.await_count == 1
    assert (store.document_dir(doc.doc_id) / "translation_cache.json").exists()
