import pytest
from pathlib import Path
from scholar_lens.memory.reflection_memory import ReflectionMemory


class TestReflectionMemory:
    @pytest.mark.asyncio
    async def test_save_reflection(self, tmp_path):
        rm = ReflectionMemory(knowledge_dir=str(tmp_path))
        await rm.save_reflection(title="weekly_reflection", content="# Weekly Reflection\n\nLearned about attention mechanisms.")
        files = list(tmp_path.glob("reflections/*.md"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "attention" in content

    @pytest.mark.asyncio
    async def test_get_latest_reflection(self, tmp_path):
        rm = ReflectionMemory(knowledge_dir=str(tmp_path))
        await rm.save_reflection(title="r1", content="First reflection")
        await rm.save_reflection(title="r2", content="Second reflection")
        latest = await rm.get_latest_reflection()
        assert "Second" in latest

    @pytest.mark.asyncio
    async def test_get_latest_no_reflections(self, tmp_path):
        rm = ReflectionMemory(knowledge_dir=str(tmp_path))
        result = await rm.get_latest_reflection()
        assert result == ""

    @pytest.mark.asyncio
    async def test_obsidian_format(self, tmp_path):
        rm = ReflectionMemory(knowledge_dir=str(tmp_path))
        await rm.save_reflection(title="test", content="# Reflection\n\nContent here.", tags=["learning", "transformer"])
        files = list(tmp_path.glob("reflections/*.md"))
        content = files[0].read_text()
        assert "---" in content
        assert "learning" in content
