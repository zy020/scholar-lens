from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from scholar_lens.parsers.models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source: str  # bm25 | vector | rrf_fused | rule_reranked
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Hybrid retrieval: BM25 + vector search + Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self._bm25_index = None
        self._chunks: list[Chunk] = []

    def build_bm25_index(self, chunks: list[Chunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed, BM25 retrieval disabled")
            return
        self._chunks = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self._bm25_index = BM25Okapi(tokenized)

    def bm25_search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self._bm25_index is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self._chunks[idx]
            results.append(RetrievalResult(chunk_id=chunk.chunk_id, text=chunk.text, score=float(scores[idx]), source="bm25", rank=rank, metadata=chunk.metadata.model_dump()))
        return results

    def rrf_fuse(self, result_lists: list[list[RetrievalResult]], top_k: int | None = None) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        metadata: dict[str, dict] = {}
        for results in result_lists:
            for r in results:
                rrf_score = 1.0 / (self.rrf_k + r.rank)
                scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + rrf_score
                texts[r.chunk_id] = r.text
                metadata[r.chunk_id] = r.metadata
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        if top_k:
            sorted_ids = sorted_ids[:top_k]
        return [
            RetrievalResult(chunk_id=cid, text=texts[cid], score=scores[cid], source="rrf_fused", rank=rank, metadata=metadata[cid])
            for rank, cid in enumerate(sorted_ids, 1)
        ]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())
