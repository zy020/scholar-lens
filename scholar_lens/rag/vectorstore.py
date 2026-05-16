from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scholar_lens.parsers.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class ScholarVectorStore:
    """ChromaDB vector store wrapper for ScholarLens chunks."""

    def __init__(self, collection_name: str = "scholar_lens", persist_dir: str | Path | None = None) -> None:
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        import chromadb
        if self._persist_dir:
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        else:
            self._client = chromadb.Client()
        return self._client

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        client = self._get_client()
        self._collection = client.get_or_create_collection(name=self._collection_name, metadata={"hnsw:space": "cosine"})
        return self._collection

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        collection = self._get_collection()
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._chunk_to_metadata(c) for c in chunks]
        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(self, query_embedding: list[float], top_k: int = 5, where: dict[str, Any] | None = None) -> list[Chunk]:
        collection = self._get_collection()
        kwargs: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where
        try:
            results = collection.query(**kwargs)
        except Exception as e:
            logger.warning(f"Vector store query failed: {e}")
            return []
        if not results["ids"] or not results["ids"][0]:
            return []
        chunks = []
        for i, chunk_id in enumerate(results["ids"][0]):
            text = results["documents"][0][i] if results["documents"] else ""
            metadata_dict = results["metadatas"][0][i] if results["metadatas"] else {}
            chunk = self._metadata_to_chunk(chunk_id, text, metadata_dict)
            chunks.append(chunk)
        return chunks

    def delete_by_doc_id(self, doc_id: str) -> None:
        collection = self._get_collection()
        try:
            collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            logger.warning(f"Failed to delete doc {doc_id}: {e}")

    def _chunk_to_metadata(self, chunk: Chunk) -> dict[str, Any]:
        m = chunk.metadata
        return {
            "section_id": m.section_id,
            "section_type": m.section_type,
            "chapter": m.chapter,
            "difficulty_score": m.difficulty_score,
            "has_formula": m.has_formula,
            "doc_id": m.doc_id,
            "layer": chunk.layer,
            "content_type": m.content_type,
        }

    def _metadata_to_chunk(self, chunk_id: str, text: str, meta: dict) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                section_id=meta.get("section_id", ""),
                section_type=meta.get("section_type", "prose"),
                chapter=meta.get("chapter", ""),
                difficulty_score=meta.get("difficulty_score", 0.5),
                has_formula=meta.get("has_formula", False),
                doc_id=meta.get("doc_id", ""),
            ),
            layer=meta.get("layer", "L2"),
        )
