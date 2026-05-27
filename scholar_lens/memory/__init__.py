from scholar_lens.memory.core_memory import CoreMemory
from scholar_lens.memory.document_memory import DocumentMemory
from scholar_lens.memory.reflection_memory import ReflectionMemory

__all__ = [
    "CoreMemory",
    "DocumentMemory",
    "MemoryManager",
    "ReflectionMemory",
    "StructuredMemory",
]


def __getattr__(name: str):
    if name == "MemoryManager":
        from scholar_lens.memory.memory_manager import MemoryManager

        return MemoryManager
    if name == "StructuredMemory":
        from scholar_lens.memory.structured_memory import StructuredMemory

        return StructuredMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
