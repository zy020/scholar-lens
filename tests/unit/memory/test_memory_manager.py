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
