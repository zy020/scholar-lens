import pytest
from scholar_lens.memory.structured_memory import StructuredMemory


@pytest.fixture
def memory(tmp_path):
    return StructuredMemory(db_path=str(tmp_path / "test.db"))


class TestStructuredMemory:
    @pytest.mark.asyncio
    async def test_add_reading_record(self, memory):
        await memory.add_reading_record(doc_id="paper_001", section_id="3.1", comprehension_score=0.8)
        records = await memory.get_reading_history("paper_001")
        assert len(records) == 1
        assert records[0]["section_id"] == "3.1"

    @pytest.mark.asyncio
    async def test_add_term_log(self, memory):
        await memory.add_term_log(term="self-attention", definition_zh="自注意力机制")
        terms = await memory.get_term_log()
        assert len(terms) == 1
        assert terms[0]["term"] == "self-attention"

    @pytest.mark.asyncio
    async def test_add_session_summary(self, memory):
        await memory.add_session_summary(doc_id="paper_001", topics="attention, transformer", difficulty="advanced")
        summaries = await memory.get_session_summaries()
        assert len(summaries) == 1

    @pytest.mark.asyncio
    async def test_add_validation_record(self, memory):
        await memory.add_validation_record(explanation_id="exp_001", passed=True, issues=[])
        records = await memory.get_validation_records()
        assert len(records) == 1
        assert records[0]["passed"] == 1

    @pytest.mark.asyncio
    async def test_multiple_records(self, memory):
        for i in range(5):
            await memory.add_reading_record(doc_id=f"doc_{i}", section_id="1", comprehension_score=0.5 + i * 0.1)
        all_records = await memory.get_all_reading_history()
        assert len(all_records) == 5

    @pytest.mark.asyncio
    async def test_close(self, memory):
        await memory.close()
