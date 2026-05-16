from __future__ import annotations

from pathlib import Path

from scholar_lens.memory.core_memory import CoreMemory
from scholar_lens.memory.structured_memory import StructuredMemory
from scholar_lens.memory.reflection_memory import ReflectionMemory


class MemoryManager:
    """Unified memory manager coordinating all four tiers."""

    def __init__(self, data_dir: str = "data") -> None:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        self.core_memory = CoreMemory()
        self.structured = StructuredMemory(db_path=str(data_path / "memory.db"))
        self.reflection = ReflectionMemory(knowledge_dir=str(data_path / "knowledge"))

    async def close(self) -> None:
        await self.structured.close()

    def get_core_context(self) -> str:
        return self.core_memory.to_context_string()
