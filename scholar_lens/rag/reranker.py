from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import requests

from scholar_lens.rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)

_SECTION_WEIGHTS = {
    "method": 1.2,
    "results": 1.1,
    "prose": 1.0,
    "citation_context": 0.9,
    "references": 0.3,
}

_DIFFICULTY_TARGETS = {
    "beginner": 0.3,
    "intermediate": 0.5,
    "advanced": 0.8,
}

# Numeric/fact query patterns for intent detection
_FACT_QUERY_PATTERNS = [
    r"\b(bleu|accuracy|score|perplexity|f1|precision|recall|rouge)\b",
    r"\b(多少|几个|数值|得分|结果是多少|准确率|正确率)\b",
    r"\b\d+\.?\d*\s*(bleu|points|percent|%)\b",
]


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, results: list[RetrievalResult],
        query: str = "", student_level: str = "intermediate",
    ) -> list[RetrievalResult]:
        ...


class RuleReranker(BaseReranker):
    """Level 3: Rule-based reranking with fact-query awareness."""

    def rerank(
        self, results: list[RetrievalResult],
        query: str = "", student_level: str = "intermediate",
    ) -> list[RetrievalResult]:
        if not results:
            return []
        target_difficulty = _DIFFICULTY_TARGETS.get(student_level, 0.5)
        is_fact = self._is_fact_query(query)

        scored = []
        for r in results:
            score = r.score
            section_type = r.metadata.get("section_type", "prose")
            score *= _SECTION_WEIGHTS.get(section_type, 1.0)
            difficulty = r.metadata.get("difficulty_score", 0.5)
            diff_penalty = abs(difficulty - target_difficulty) * 0.5
            score *= (1.0 - diff_penalty)
            cross_refs = r.metadata.get("cross_refs", [])
            if isinstance(cross_refs, list) and cross_refs:
                score *= 1.05

            # Fact query boost: reward chunks containing numbers
            if is_fact:
                num_count = len(re.findall(r"\d+\.?\d*", r.text))
                if num_count >= 2:
                    score = score * 1.2 + 0.05
                if r.metadata.get("content_type") == "fact":
                    score = score * 1.3 + 0.1

            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(
                chunk_id=r.chunk_id, text=r.text, score=score,
                source="rule_reranked", rank=rank, metadata=r.metadata,
            )
            for rank, (score, r) in enumerate(scored, 1)
        ]

    def _is_fact_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in _FACT_QUERY_PATTERNS)


class ModelReranker(BaseReranker):
    """Stage 1: Model-based reranking via dedicated reranker API (e.g. Qwen3-Reranker).

    Calls /rerank endpoint with query + documents, reorders by relevance_score.
    Falls back gracefully on network errors.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 15):
        self._url = f"{base_url}/rerank"
        self._key = api_key
        self._model = model
        self._timeout = timeout

    # NOTE: uses blocking requests.post — if called from async code,
    # wrap with asyncio.to_thread() or replace with httpx.AsyncClient.
    def rerank(
        self, results: list[RetrievalResult],
        query: str = "", student_level: str = "intermediate",
    ) -> list[RetrievalResult]:
        if not results or len(results) < 2:
            return results

        docs = [r.text[:1500] for r in results]  # Truncate per doc for API limits
        try:
            resp = requests.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "query": query, "documents": docs, "top_n": len(results)},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Model reranker failed: %s, passing through original results", e)
            return results

        reranked = []
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            r = results[idx]
            reranked.append(RetrievalResult(
                chunk_id=r.chunk_id, text=r.text, score=score,
                source="model_reranked", rank=item.get("rank", len(reranked) + 1),
                metadata=r.metadata,
            ))

        # Append any results not returned by the API
        returned_ids = {item["index"] for item in data.get("results", [])}
        for i, r in enumerate(results):
            if i not in returned_ids:
                reranked.append(r)

        return reranked


class DiversityReranker(BaseReranker):
    """Ensures retrieval diversity across documents.

    Two strategies (composed):
    1. Per-doc constraint: no more than max_per_doc results from any document
    2. MMR: penalizes same-document chunks when selecting consecutive results
    """

    def __init__(self, max_per_doc: int = 3, mmr_lambda: float = 0.7):
        self.max_per_doc = max_per_doc
        self.mmr_lambda = mmr_lambda

    def rerank(
        self, results: list[RetrievalResult],
        query: str = "", student_level: str = "intermediate",
    ) -> list[RetrievalResult]:
        if not results:
            return []

        # Step 1: Hard per-doc constraint
        constrained = self._apply_diversity_constraint(results)
        if len(constrained) <= 1:
            return constrained

        # Step 2: MMR re-ranking
        return self._mmr_rerank(constrained)

    def _apply_diversity_constraint(
        self, results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Ensure no single document contributes more than max_per_doc results."""
        doc_counts: dict[str, int] = {}
        out = []
        for r in results:
            doc_id = r.metadata.get("doc_id") or r.chunk_id.rsplit("_", 1)[0]
            if doc_counts.get(doc_id, 0) < self.max_per_doc:
                out.append(r)
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        return out

    def _mmr_rerank(
        self, results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """MMR: balance relevance and document diversity.

        λ closer to 1.0 → pure relevance ranking.
        λ closer to 0.0 → maximize document diversity.
        """
        selected: list[RetrievalResult] = []
        remaining = list(results)

        while remaining and len(selected) < len(results):
            best_idx = 0
            best_score = -float("inf")

            for i, r in enumerate(remaining):
                relevance = r.score
                diversity_penalty = 0.0
                for s in selected:
                    doc_r = r.metadata.get("doc_id", "")
                    doc_s = s.metadata.get("doc_id", "")
                    if doc_r == doc_s:
                        diversity_penalty += 0.3
                    # Also penalize same-section results
                    if r.metadata.get("section_id") == s.metadata.get("section_id"):
                        diversity_penalty += 0.1

                mmr = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * diversity_penalty
                )
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return [
            RetrievalResult(
                chunk_id=r.chunk_id, text=r.text, score=r.score,
                source="diversity_reranked", rank=rank, metadata=r.metadata,
            )
            for rank, r in enumerate(selected, 1)
        ]

    def diversify_only(self, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Lightweight: apply per-doc constraint only, keep original order."""
        constrained = self._apply_diversity_constraint(results)
        return constrained[:top_k]


class RerankerPipeline:
    """Multi-stage reranking pipeline: model → rule → diversity."""

    def __init__(
        self,
        model_reranker: BaseReranker | None = None,
        llm_reranker: BaseReranker | None = None,
        diversity: bool = True,
        max_per_doc: int = 3,
        mmr_lambda: float = 0.7,
    ) -> None:
        self._model_reranker = model_reranker
        self._llm_reranker = llm_reranker
        self._rule_reranker = RuleReranker()
        self._diversity_reranker = DiversityReranker(
            max_per_doc=max_per_doc, mmr_lambda=mmr_lambda,
        ) if diversity else None
        self.level = self._determine_level()

    def _determine_level(self) -> str:
        if self._model_reranker:
            return "model"
        if self._llm_reranker:
            return "llm"
        return "rule"

    def rerank(
        self, results: list[RetrievalResult],
        query: str = "", student_level: str = "intermediate",
    ) -> list[RetrievalResult]:
        # Stage 1: Model or LLM reranker
        if self._model_reranker:
            try:
                results = self._model_reranker.rerank(results, query, student_level)
            except Exception as e:
                logger.warning(f"Model reranker failed: {e}, falling back")
        elif self._llm_reranker:
            try:
                results = self._llm_reranker.rerank(results, query, student_level)
            except Exception as e:
                logger.warning(f"LLM reranker failed: {e}, falling back")

        # Stage 2: Rule reranker (always runs — zero cost)
        results = self._rule_reranker.rerank(results, query, student_level)

        # Stage 3: Diversity reranker
        if self._diversity_reranker:
            results = self._diversity_reranker.rerank(results, query, student_level)

        return results
