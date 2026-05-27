from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from scholar_lens.core.settings import Settings
from scholar_lens.rag.document_store import DocumentStore

if TYPE_CHECKING:
    from scholar_lens.memory.memory_manager import MemoryManager

@lru_cache
def get_settings() -> Settings:
    return Settings()


_settings: Settings | None = None
_memory_manager: "MemoryManager | None" = None


def init_dependencies() -> None:
    global _settings, _memory_manager

    # P2.6: Fix ONNX Runtime deadlock in WSL2 — limit to single thread
    # RapidOCR loads 3 ONNX models, multi-threaded allocation deadlocks in WSL2
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("ONNXRUNTIME_NUM_THREADS", "1")

    _settings = get_settings()
    from scholar_lens.memory.memory_manager import MemoryManager

    _memory_manager = MemoryManager(data_dir=str(_settings.data_dir))


@lru_cache
def get_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(root=settings.data_dir / "documents")


def get_memory_manager() -> "MemoryManager":
    if _memory_manager is None:
        init_dependencies()
    if _memory_manager is None:
        raise RuntimeError("MemoryManager failed to initialize")
    return _memory_manager
