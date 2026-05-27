import pytest
from scholar_lens.memory.memory_manager import MemoryManager


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_create(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        assert mm is not None

    @pytest.mark.asyncio
    async def test_core_memory_access(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        cm = mm.core_memory
        assert cm.student_profile == ""

    @pytest.mark.asyncio
    async def test_update_core_position(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        mm.core_memory.update_position("doc1", "3.1")
        assert mm.core_memory.current_position == "doc1:3.1"

    @pytest.mark.asyncio
    async def test_structured_memory_roundtrip(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        await mm.structured.add_reading_record("doc1", "1", 0.9)
        records = await mm.structured.get_reading_history("doc1")
        assert len(records) == 1
        await mm.close()

    @pytest.mark.asyncio
    async def test_reflection_roundtrip(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        await mm.reflection.save_reflection("test", "Content")
        result = await mm.reflection.get_latest_reflection()
        assert "Content" in result

    @pytest.mark.asyncio
    async def test_record_event_updates_position_and_summary(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))

        await mm.record_event(
            "section_read",
            doc_id="doc1",
            section_id="intro",
            payload={"title": "Introduction"},
        )
        await mm.record_event(
            "chat_question",
            doc_id="doc1",
            section_id="intro",
            payload={"message": "What is attention?"},
        )

        assert mm.core_memory.current_position == "doc1:intro"
        assert "Read Introduction" in mm.core_memory.session_summary
        assert "Asked: What is attention?" in mm.core_memory.session_summary

    @pytest.mark.asyncio
    async def test_document_memory_hydration_replaces_current_doc_cache(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))

        mm.document.load_from_document_understanding(
            doc_id="doc1",
            l0={"intro": "Intro summary"},
            l1={"intro": "Intro overview"},
        )
        mm.document.load_from_document_understanding(
            doc_id="doc2",
            l0={"method": "Method summary"},
            l1={"method": "Method overview"},
        )

        assert mm.document.doc_id == "doc2"
        assert mm.document.get_l0_summary("intro") == ""
        assert mm.document.get_l0_summary("method") == "Method summary"

    @pytest.mark.asyncio
    async def test_record_event_updates_concept_memory(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))

        await mm.record_event(
            "chat_question",
            doc_id="paper_001",
            section_id="method",
            payload={"message": "我不理解 self-attention 公式和 positional encoding"},
        )

        concepts = await mm.structured.get_concept_memory("paper_001")
        concept_names = {item["concept"] for item in concepts}

        assert "self-attention" in concept_names
        assert "positional encoding" in concept_names
        assert all(item["status"] == "needs_review" for item in concepts)

    @pytest.mark.asyncio
    async def test_memory_snapshot_contains_core_events_and_concepts(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        await mm.record_event(
            "section_read",
            doc_id="paper_001",
            section_id="intro",
            payload={"title": "Introduction"},
        )
        await mm.record_event(
            "chat_question",
            doc_id="paper_001",
            section_id="intro",
            payload={"message": "What is graph neural network?"},
        )

        snapshot = await mm.get_snapshot(doc_id="paper_001")

        assert snapshot["core"]["current_position"] == "paper_001:intro"
        assert "Recent learning actions" in snapshot["core"]["session_summary"]
        assert snapshot["recent_events"][0]["event_type"] == "chat_question"
        assert any(item["concept"] == "graph neural network" for item in snapshot["concepts"])

    @pytest.mark.asyncio
    async def test_retrieval_hints_use_current_section_and_review_concepts(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        await mm.record_event(
            "chat_question",
            doc_id="paper_001",
            section_id="method",
            payload={"message": "我不理解 attention formula"},
        )

        hints = await mm.get_retrieval_hints("paper_001")

        assert hints["current_section_id"] == "method"
        assert "attention formula" in hints["concepts"]

    @pytest.mark.asyncio
    async def test_memory_update_graph_records_event_and_compacts_summary(self, tmp_path):
        from scholar_lens.memory.memory_graph import run_memory_update_graph

        mm = MemoryManager(data_dir=str(tmp_path))

        for idx in range(10):
            await run_memory_update_graph(
                mm,
                event_type="chat_question",
                doc_id="paper_001",
                section_id="intro",
                payload={"message": f"What is self-attention detail {idx}?"},
            )

        concepts = await mm.structured.get_concept_memory("paper_001")
        events = await mm.structured.get_learning_events("paper_001", limit=20)

        assert len(events) == 10
        assert any(item["concept"] == "self-attention" for item in concepts)
        assert mm.core_memory.current_position == "paper_001:intro"
        assert "detail 0" not in mm.core_memory.session_summary
        assert "detail 9" in mm.core_memory.session_summary

    @pytest.mark.asyncio
    async def test_memory_update_graph_uses_optional_llm_compression(self, tmp_path):
        from scholar_lens.memory.memory_graph import run_memory_update_graph

        class FakeResponse:
            content = "本轮主要围绕 self-attention 的概念和公式进行追问，需要继续复习。"

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                return FakeResponse()

        mm = MemoryManager(data_dir=str(tmp_path))
        llm = FakeLLM()

        await run_memory_update_graph(
            mm,
            event_type="chat_question",
            doc_id="paper_001",
            section_id="intro",
            payload={"message": "我不理解 self-attention 公式"},
            summary_llm=llm,
        )

        assert llm.calls == 1
        assert mm.core_memory.session_summary == "本轮主要围绕 self-attention 的概念和公式进行追问，需要继续复习。"
