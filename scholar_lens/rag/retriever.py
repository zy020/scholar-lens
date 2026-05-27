from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scholar_lens.parsers.models import Chunk

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source: str  # bm25 | vector | rrf_fused | rule_reranked | fact_extracted
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Hybrid retrieval: BM25 + vector search + Reciprocal Rank Fusion.

    Cross-lingual handling: BM25 signal strength is measured per query.
    When BM25 signal is weak (cross-lingual), its RRF weight is attenuated
    to avoid polluting vector results.
    """

    def __init__(self, rrf_k: int = 60, bm25_min_signal: float = 0.05) -> None:
        self.rrf_k = rrf_k
        self.bm25_min_signal = bm25_min_signal
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
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id, text=chunk.text,
                score=float(scores[idx]), source="bm25", rank=rank,
                metadata=chunk.metadata.model_dump(),
            ))
        return results

    def rrf_fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
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
            RetrievalResult(
                chunk_id=cid, text=texts[cid], score=scores[cid],
                source="rrf_fused", rank=rank, metadata=metadata[cid],
            )
            for rank, cid in enumerate(sorted_ids, 1)
        ]

    def weighted_rrf_fuse(
        self,
        weighted_lists: list[tuple[list[RetrievalResult], float]],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """RRF fusion with per-source weights.

        Args:
            weighted_lists: List of (results, weight) tuples.
                weight=1.0 = full contribution, weight=0.0 = no contribution.
        """
        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        metadata: dict[str, dict] = {}
        for results, weight in weighted_lists:
            if weight <= 0.001:
                continue
            for r in results:
                rrf_score = weight / (self.rrf_k + r.rank)
                scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + rrf_score
                texts[r.chunk_id] = r.text
                metadata[r.chunk_id] = r.metadata
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        if top_k:
            sorted_ids = sorted_ids[:top_k]
        return [
            RetrievalResult(
                chunk_id=cid, text=texts[cid], score=scores[cid],
                source="rrf_fused", rank=rank, metadata=metadata[cid],
            )
            for rank, cid in enumerate(sorted_ids, 1)
        ]

    def _bm25_signal_strength(self, bm25_results: list[RetrievalResult]) -> float:
        """Measure BM25 signal strength. Returns 0.0 for cross-lingual queries.

        Normalizes max score by typical BM25 score range. Scores < ~5.0
        indicate poor term overlap (cross-lingual).
        """
        if not bm25_results:
            return 0.0
        max_score = max(r.score for r in bm25_results)
        # BM25 scores typically range from 0 to 20+ for good matches.
        # Scores < 1.0 mean essentially no term overlap.
        return min(max_score / 5.0, 1.0)

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        vector_results: list[RetrievalResult],
        top_k: int = 10,
        fact_boost: float = 1.3,
        bm25_results: list[RetrievalResult] | None = None,
    ) -> list[RetrievalResult]:
        """Unified hybrid search with cross-lingual-aware weighting.

        When BM25 signal is weak (cross-lingual query), BM25 weight is
        automatically attenuated to prevent noise from polluting vector results.
        fact-chunks are boosted to improve numerical/ factual query recall.

        If bm25_results is provided (pre-computed, e.g. from multi-rewrite),
        use those instead of running internal bm25_search.
        """
        # Boost fact-chunks in vector results (2.1-B)
        boosted_vector = []
        for r in vector_results:
            score = r.score
            if r.metadata.get("content_type") == "fact":
                score *= fact_boost
            boosted_vector.append(RetrievalResult(
                chunk_id=r.chunk_id, text=r.text, score=score,
                source=r.source, rank=r.rank, metadata=r.metadata,
            ))
        # Re-sort after boosting
        boosted_vector.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(boosted_vector):
            r.rank = i + 1

        bm25 = bm25_results if bm25_results is not None else self.bm25_search(query, top_k)
        signal = self._bm25_signal_strength(bm25)

        if signal < self.bm25_min_signal:
            logger.debug(
                "BM25 signal too weak (%.4f < %.4f), skipping BM25 for cross-lingual query",
                signal, self.bm25_min_signal,
            )
            return boosted_vector[:top_k]

        return self.weighted_rrf_fuse(
            [(bm25, signal), (boosted_vector, 1.0)],
            top_k=top_k,
        )

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())

    # ===== Batch 3.1: CRAG (Corrective RAG) =====

    async def check_relevance(
        self, query: str, top_chunks: list[RetrievalResult], llm: BaseChatModel,
    ) -> bool:
        """CRAG relevance check: are retrieved chunks actually relevant?

        If no chunk is relevant, the caller should fall back to LLM
        knowledge without retrieved context.
        """
        if not top_chunks:
            return False
        from langchain_core.messages import HumanMessage
        sample = "\n---\n".join(r.text[:300] for r in top_chunks[:3])
        response = await llm.ainvoke([HumanMessage(content=f"""Are the following document chunks relevant to the query?
Answer ONLY "yes" or "no".

Query: {query}

Chunks:
{sample}

Relevant (yes/no):""")])
        return "yes" in response.content.strip().lower()

    def check_relevance_sync(
        self, query: str, top_chunks: list[RetrievalResult], llm: BaseChatModel,
    ) -> bool:
        """Sync version of check_relevance for threaded evaluation."""
        if not top_chunks:
            return False
        from langchain_core.messages import HumanMessage
        sample = "\n---\n".join(r.text[:300] for r in top_chunks[:3])
        response = llm.invoke([HumanMessage(content=f"""Are the following document chunks relevant to the query?
Answer ONLY "yes" or "no".

Query: {query}

Chunks:
{sample}

Relevant (yes/no):""")])
        return "yes" in response.content.strip().lower()

    # ===== Query Rewrite (2.1-C) =====

    async def rewrite_query(
        self, query: str, llm: BaseChatModel, num_variants: int = 3,
    ) -> list[str]:
        """Rewrite a Chinese query into multiple English keyword variants.

        Multiple variants improve BM25 recall: different phrasings match different
        chunks. All variants generated in a single LLM call (no extra latency).

        Only rewrites if query contains CJK characters.
        """
        cjk = sum(1 for c in query if "一" <= c <= "鿿")
        if cjk == 0:
            return [query]

        from langchain_core.messages import HumanMessage

        response = await llm.ainvoke([
            HumanMessage(content=f"""Rewrite this Chinese academic question into {num_variants} different English keyword variants.
Each variant should use different technical terms and phrasings.
Output one variant per line, no numbering, no explanation.

Chinese: {query}
Variants:"""),
        ])
        variants = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
        logger.debug("Query rewritten: '%s' → %d variants", query[:60], len(variants))
        return variants or [query]

    def rewrite_query_sync(
        self, query: str, llm: BaseChatModel, num_variants: int = 3,
    ) -> list[str]:
        """Sync version of rewrite_query for threaded evaluation."""
        cjk = sum(1 for c in query if "一" <= c <= "鿿")
        if cjk == 0:
            return [query]
        from langchain_core.messages import HumanMessage
        response = llm.invoke([
            HumanMessage(content=f"""Rewrite this Chinese academic question into {num_variants} different English keyword variants.
Each variant should use different technical terms and phrasings.
Output one variant per line, no numbering, no explanation.

Chinese: {query}
Variants:"""),
        ])
        variants = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
        logger.debug("Query rewritten (sync): '%s' → %d variants", query[:60], len(variants))
        return variants or [query]

    def _multi_bm25_search(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        """BM25 search with multiple query variants. Takes best score per chunk."""
        if self._bm25_index is None:
            return []
        all_results: dict[str, RetrievalResult] = {}
        for query in queries:
            for r in self.bm25_search(query, top_k):
                if r.chunk_id not in all_results or r.score > all_results[r.chunk_id].score:
                    all_results[r.chunk_id] = r
        best = sorted(all_results.values(), key=lambda x: x.score, reverse=True)[:top_k]
        for i, r in enumerate(best):
            r.rank = i + 1
        return best

    # ===== Two-Stage Fact Retrieval (Problem 3 Plan C) =====

    async def retrieve_fact(
        self,
        query: str,
        top_chunks: list[RetrievalResult],
        llm: BaseChatModel,
        max_context_chunks: int = 10,
    ) -> str:
        """Two-stage retrieval: coarse search + LLM fact extraction.

        Stage 1: Use top_chunks from hybrid_search() as context (expanded to 10).
        Stage 2: Ask LLM to extract the specific fact from those chunks.

        Returns the extracted fact, or "NOT_FOUND" if the fact is not in
        the retrieved chunks.
        """
        if not top_chunks:
            return "NOT_FOUND"

        context = "\n---\n".join(
            f"[{r.chunk_id}] {r.text[:800]}" for r in top_chunks[:max_context_chunks]
        )

        from langchain_core.messages import HumanMessage

        response = await llm.ainvoke([
            HumanMessage(content=f"""Extract the precise answer to the question from the context below.
If the answer is a number, include the number and its unit (e.g., "28.4 BLEU").
If the answer is not in the context, respond with NOT_FOUND.
Do not make up information.

Context:
{context}

Question: {query}

Answer:"""),
        ])
        return response.content.strip()
