from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scholar_lens.core.llm_factory import EmbeddingFactory
from scholar_lens.parsers.models import Chunk
from scholar_lens.rag.retriever import RetrievalResult
from scholar_lens.rag.vectorstore import ScholarVectorStore

if TYPE_CHECKING:
    from scholar_lens.core.settings import Settings
    from scholar_lens.rag.document_store import DocumentStore

logger = logging.getLogger(__name__)


def embedding_configured(settings: Settings | None) -> bool:
    if settings is None:
        return False
    config = getattr(settings, "embedding", None)
    return bool(config and config.api_key and config.model)


def _default_vector_store(settings: Settings) -> ScholarVectorStore:
    return ScholarVectorStore(
        collection_name="scholar_lens_chunks",
        persist_dir=settings.data_dir / "vectorstore",
    )


def _default_embeddings(settings: Settings):
    return EmbeddingFactory.from_settings(settings).create()


def _embedding_text(chunk: Chunk) -> str:
    formula_ids = " ".join(chunk.metadata.formula_ids)
    return "\n".join(
        part
        for part in (
            chunk.text,
            chunk.metadata.contextual_prefix,
            formula_ids,
        )
        if part
    )


def index_document_chunks(
    store: DocumentStore | None,
    doc_id: str,
    chunks: list[Chunk],
    settings: Settings | None,
    embeddings=None,
    vector_store: ScholarVectorStore | None = None,
) -> bool:
    if not chunks or not embedding_configured(settings):
        return False
    try:
        embedder = embeddings or _default_embeddings(settings)
        vectors = embedder.embed_documents([_embedding_text(chunk) for chunk in chunks])
        target_store = vector_store or _default_vector_store(settings)
        target_store.delete_by_doc_id(doc_id)
        target_store.add_chunks(chunks, vectors)
        return True
    except Exception:
        logger.warning("Vector indexing failed for document %s", doc_id, exc_info=True)
        return False


def search_vector_chunks(
    doc_id: str,
    query: str,
    top_k: int,
    settings: Settings | None,
    embeddings=None,
    vector_store: ScholarVectorStore | None = None,
) -> list[RetrievalResult]:
    if not query.strip() or not embedding_configured(settings):
        return []
    try:
        embedder = embeddings or _default_embeddings(settings)
        query_embedding = embedder.embed_query(query)
        target_store = vector_store or _default_vector_store(settings)
        return target_store.query_results(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"doc_id": doc_id},
        )
    except Exception:
        logger.warning("Vector search failed for document %s", doc_id, exc_info=True)
        return []
