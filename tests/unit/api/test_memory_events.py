from types import SimpleNamespace

import pytest

from scholar_lens.api import memory_events
from scholar_lens.core.settings import LLMConfig


class FakeMemory:
    def __init__(self):
        self.calls = []

    async def record_event(self, event_type, *, doc_id="", section_id="", payload=None, summary_llm=None):
        self.calls.append({
            "event_type": event_type,
            "doc_id": doc_id,
            "section_id": section_id,
            "payload": payload,
            "summary_llm": summary_llm,
        })


@pytest.mark.asyncio
async def test_record_memory_event_omits_summary_llm_when_disabled(monkeypatch):
    settings = SimpleNamespace(
        memory_llm_compression_enabled=False,
        llm=LLMConfig(api_key="key", base_url="https://llm.example/v1", model="model"),
        backup_llm=None,
    )
    monkeypatch.setattr(memory_events, "get_settings", lambda: settings)

    memory = FakeMemory()
    await memory_events.record_memory_event(
        memory,
        "chat_question",
        doc_id="doc",
        section_id="s1",
        payload={"message": "hello"},
    )

    assert memory.calls[0]["summary_llm"] is None


@pytest.mark.asyncio
async def test_record_memory_event_passes_summary_llm_when_enabled(monkeypatch):
    settings = SimpleNamespace(
        memory_llm_compression_enabled=True,
        llm=LLMConfig(api_key="key", base_url="https://llm.example/v1", model="model"),
        backup_llm=None,
    )
    fake_llm = object()

    class FakeFactory:
        @classmethod
        def from_settings(cls, received_settings):
            assert received_settings is settings
            return cls()

        def create(self, *, config=None, streaming=False):
            assert config is settings.llm
            assert streaming is False
            return fake_llm

    monkeypatch.setattr(memory_events, "get_settings", lambda: settings)
    monkeypatch.setattr(memory_events, "ChatLLMFactory", FakeFactory)

    memory = FakeMemory()
    await memory_events.record_memory_event(memory, "chat_question", doc_id="doc")

    assert memory.calls[0]["summary_llm"] is fake_llm
