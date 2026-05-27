from scholar_lens.api.main import create_app
from scholar_lens.memory.memory_manager import MemoryManager
from tests.unit.api.helpers import ASGITestClient


class TestMemoryRoutes:
    def test_memory_snapshot_route(self, monkeypatch, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))

        async def seed():
            await mm.record_event(
                "chat_question",
                doc_id="paper_001",
                section_id="intro",
                payload={"message": "What is self-attention?"},
            )

        import asyncio

        asyncio.run(seed())
        monkeypatch.setattr("scholar_lens.api.routes.memory.get_memory_manager", lambda: mm)

        response = ASGITestClient(create_app()).get("/api/memory?doc_id=paper_001")

        assert response.status_code == 200
        data = response.json()
        assert data["core"]["current_position"] == "paper_001:intro"
        assert data["recent_events"][0]["event_type"] == "chat_question"
        assert any(item["concept"] == "self-attention" for item in data["concepts"])

    def test_clear_document_memory_route(self, monkeypatch, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))

        async def seed():
            await mm.record_event(
                "chat_question",
                doc_id="paper_001",
                payload={"message": "What is attention?"},
            )
            await mm.record_event(
                "chat_question",
                doc_id="paper_002",
                payload={"message": "What is BERT?"},
            )

        import asyncio

        asyncio.run(seed())
        monkeypatch.setattr("scholar_lens.api.routes.memory.get_memory_manager", lambda: mm)

        response = ASGITestClient(create_app()).delete("/api/memory/document?doc_id=paper_001")

        assert response.status_code == 200
        assert response.json()["status"] == "cleared"
        assert asyncio.run(mm.structured.get_concept_memory("paper_001")) == []
        assert asyncio.run(mm.structured.get_concept_memory("paper_002"))
