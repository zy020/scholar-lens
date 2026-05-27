"""DocumentIndex: BM25-based chunk retrieval from DocumentStore.

Falls back to simple token-overlap scoring when rank_bm25 is unavailable.
No embedding API required.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from scholar_lens.parsers.math_normalizer import normalize_math_text
from scholar_lens.rag.retriever import RetrievalResult

if TYPE_CHECKING:
    from scholar_lens.rag.document_store import DocumentStore


QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "what", "when", "where", "which", "why",
    "with",
}


class DocumentIndex:
    def __init__(self, store: DocumentStore) -> None:
        self._store = store
        self._bm25 = None
        self._bm25_available = False
        try:
            from rank_bm25 import BM25Okapi
            self._BM25Okapi = BM25Okapi
            self._bm25_available = True
        except ImportError:
            pass

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())

    @classmethod
    def _query_text(cls, query: str) -> str:
        normalized = normalize_math_text(query)
        if normalized and normalized != query:
            return f"{query} {normalized}"
        return query

    @staticmethod
    def _search_text(chunk: dict) -> str:
        meta = chunk.get("metadata", {})
        formula_ids = meta.get("formula_ids") or []
        if isinstance(formula_ids, list):
            formula_text = " ".join(str(item) for item in formula_ids)
        else:
            formula_text = str(formula_ids)
        return " ".join(
            part
            for part in (
                chunk.get("text", ""),
                str(meta.get("contextual_prefix") or ""),
                formula_text,
            )
            if part
        )

    @classmethod
    def _core_query_tokens(cls, query: str) -> set[str]:
        return {
            token
            for token in cls._tokenize(cls._query_text(query))
            if token not in QUERY_STOPWORDS
        }

    @classmethod
    def _adjust_courseware_score(cls, query: str, chunk: dict, score: float) -> float:
        meta = chunk.get("metadata", {})
        is_slide = (
            meta.get("content_type") == "slide"
            or meta.get("section_type") == "slide"
            or str(meta.get("section_id", "")).startswith("slide_")
        )
        if not is_slide:
            return score
        core_tokens = cls._core_query_tokens(query)
        if not core_tokens:
            return score
        text = cls._search_text(chunk)
        text_tokens = set(cls._tokenize(text))
        core_overlap = core_tokens & text_tokens
        if not core_overlap:
            return score * 0.35
        header = "\n".join(line.strip() for line in text.splitlines()[:3] if line.strip())
        header_overlap = core_tokens & set(cls._tokenize(header))
        formula_boost = 0.0
        if meta.get("has_formula") and core_overlap:
            formula_boost = min(2.0, len(core_overlap) * 0.35)
        return score + len(core_overlap) * 0.5 + len(header_overlap) * 1.5 + formula_boost

    def search(
        self, doc_id: str, query: str, section_id: str = "", top_k: int = 5,
        section_only: bool = False,
    ) -> list[RetrievalResult]:
        chunks = self._store.load_chunks(doc_id)
        if not chunks:
            return []

        # If section_only, pre-filter to matching section chunks
        if section_only and section_id:
            section_chunks = [c for c in chunks if c.get("metadata", {}).get("section_id") == section_id]
            if section_chunks:
                chunks = section_chunks

        texts = [self._search_text(c) for c in chunks]

        if self._bm25_available:
            tokenized = [self._tokenize(t) for t in texts]
            bm25 = self._BM25Okapi(tokenized)
            scores = bm25.get_scores(self._tokenize(self._query_text(query)))
        else:
            # Simple token overlap fallback
            query_tokens = set(self._tokenize(self._query_text(query)))
            scores = []
            for t in texts:
                doc_tokens = set(self._tokenize(t))
                overlap = len(query_tokens & doc_tokens)
                scores.append(float(overlap))
        scores = [float(score) for score in scores]

        # Section boost: matching-section chunks get significant priority
        for i, c in enumerate(chunks):
            if section_id and c.get("metadata", {}).get("section_id") == section_id:
                scores[i] += 2.0  # strong boost ensures section chunks rank first
            scores[i] = self._adjust_courseware_score(query, c, scores[i])

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k], 1):
            c = chunks[idx]
            meta = c.get("metadata", {})
            results.append(RetrievalResult(
                chunk_id=c.get("chunk_id", str(idx)),
                text=c.get("text", ""),
                score=score,
                source="bm25" if self._bm25_available else "overlap",
                rank=rank,
                metadata=meta,
            ))
        return results


def evidence_from_results(results: list[RetrievalResult]) -> list[dict]:
    """Convert retrieval results to evidence items for the frontend."""
    from scholar_lens.api.schemas import EvidenceItem
    return [
        EvidenceItem(
            doc_id=r.metadata.get("doc_id", ""),
            section_id=r.metadata.get("section_id", ""),
            page=r.metadata.get("page_start"),
            chunk_id=r.chunk_id,
            quote=r.text[:300],
            score=round(r.score, 3),
        ).model_dump()
        for r in results
    ]
