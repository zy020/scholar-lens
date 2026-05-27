from __future__ import annotations

from scholar_lens.api.deps import get_settings
from scholar_lens.rag.document_store import DocumentStore


def get_script_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(root=settings.data_dir / "documents")
