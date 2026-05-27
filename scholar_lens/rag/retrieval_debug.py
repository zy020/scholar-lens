from __future__ import annotations

import time
from statistics import mean
from typing import Any

from scholar_lens.api.chat_service import (
    _expand_retrieval_context,
    _normalize_rank,
    _rewrite_query_variants,
    _search_bm25_variants,
    build_reranker_pipeline,
)
from scholar_lens.rag.document_index import DocumentIndex
from scholar_lens.rag.document_store import DocumentStore
from scholar_lens.rag.retriever import HybridRetriever, RetrievalResult
from scholar_lens.rag.vector_index import search_vector_chunks


def result_to_dict(result: RetrievalResult, *, include_text: bool = False) -> dict[str, Any]:
    metadata = result.metadata or {}
    payload = {
        "chunk_id": result.chunk_id,
        "rank": result.rank,
        "score": round(float(result.score), 6),
        "source": result.source,
        "section_id": metadata.get("section_id", ""),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "text_preview": (result.text or "")[:240],
    }
    if include_text:
        payload["text"] = result.text or ""
    return payload


def _results_to_dicts(results: list[RetrievalResult], *, include_text: bool = False) -> list[dict[str, Any]]:
    return [result_to_dict(result, include_text=include_text) for result in results]


def trace_retrieval(
    store: DocumentStore,
    doc_id: str,
    query: str,
    *,
    top_k: int = 5,
    context_k: int | None = None,
    section_id: str = "",
    section_only: bool = False,
    use_reranker: bool = True,
    student_level: str = "intermediate",
    settings=None,
    include_text: bool = False,
    intent_hint: str | None = None,
) -> dict[str, Any]:
    index = DocumentIndex(store)
    candidate_k = max(top_k * 3, top_k) if use_reranker else top_k
    effective_context_k = max(context_k if context_k is not None else top_k * 2, top_k)
    query_variants = _rewrite_query_variants(query, settings)
    bm25_results = _search_bm25_variants(
        index,
        doc_id,
        query_variants,
        section_id,
        candidate_k,
        section_only,
    )
    vector_results = search_vector_chunks(doc_id, query, candidate_k, settings)
    hybrid_results = bm25_results
    if vector_results:
        hybrid_results = HybridRetriever().hybrid_search(
            query=query,
            query_embedding=[],
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=candidate_k,
        )
    reranked_results = hybrid_results
    if use_reranker and reranked_results:
        reranked_results = build_reranker_pipeline(settings).rerank(
            reranked_results,
            query=query,
            student_level=student_level,
        )
    evidence_results = _normalize_rank(reranked_results, top_k)
    context_results = _expand_retrieval_context(
        store,
        doc_id,
        reranked_results,
        limit=effective_context_k,
        query=query,
        intent_hint=intent_hint,
    )
    return {
        "query": query,
        "doc_id": doc_id,
        "intent_hint": intent_hint or "",
        "query_variants": query_variants,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "context_k": effective_context_k,
        "stages": {
            "bm25": _results_to_dicts(bm25_results, include_text=include_text),
            "vector": _results_to_dicts(vector_results, include_text=include_text),
            "hybrid": _results_to_dicts(hybrid_results, include_text=include_text),
            "reranked": _results_to_dicts(reranked_results, include_text=include_text),
        },
        "evidence": _results_to_dicts(evidence_results, include_text=include_text),
        "context": _results_to_dicts(context_results, include_text=include_text),
    }


def _text_contains(text: str, term: str) -> bool:
    import re
    normalized_text = re.sub(r"\s+", " ", text).strip().lower()
    normalized_term = re.sub(r"\s+", " ", term).strip().lower()
    return normalized_term in normalized_text


def _matches_target(item: dict[str, Any], target: dict[str, Any]) -> bool:
    chunk_ids = set(target.get("chunk_ids", []))
    if chunk_ids and item.get("chunk_id") in chunk_ids:
        return True
    section_ids = set(target.get("section_ids", []))
    if section_ids and item.get("section_id") in section_ids:
        return True
    pages = set(target.get("pages", []))
    if pages and item.get("page_start") in pages:
        return True
    terms = [term for term in target.get("terms", []) if term]
    text = item.get("text") or item.get("text_preview", "")
    return bool(terms and any(_text_contains(text, term) for term in terms))


def _any_match(items: list[dict[str, Any]], targets: list[dict[str, Any]]) -> bool:
    return any(_matches_target(item, target) for item in items for target in targets)


def _first_match_rank(items: list[dict[str, Any]], targets: list[dict[str, Any]]) -> int | None:
    for idx, item in enumerate(items, start=1):
        if any(_matches_target(item, target) for target in targets):
            return idx
    return None


def _matched_target_indexes(items: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[int]:
    return [
        idx
        for idx, target in enumerate(targets)
        if _any_match(items, [target])
    ]


def score_trace(trace: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    targets = case.get("targets", [])
    if not targets:
        return {
            "hit_at_k": None,
            "context_hit_at_k": None,
            "mrr_at_k": None,
            "recall_at_k": None,
            "context_recall_at_k": None,
            "evidence_rank": None,
            "empty_retrieval": 1.0 if not trace.get("evidence", []) else 0.0,
            "matched_targets": [],
        }
    evidence = trace.get("evidence", [])
    context = trace.get("context", [])
    evidence_rank = _first_match_rank(evidence, targets)
    evidence_matches = _matched_target_indexes(evidence, targets)
    context_matches = _matched_target_indexes(context, targets)
    target_count = len(targets)
    return {
        "hit_at_k": 1.0 if evidence_rank is not None else 0.0,
        "context_hit_at_k": 1.0 if context_matches else 0.0,
        "mrr_at_k": round(1.0 / evidence_rank, 3) if evidence_rank is not None else 0.0,
        "recall_at_k": round(len(evidence_matches) / target_count, 3),
        "context_recall_at_k": round(len(context_matches) / target_count, 3),
        "evidence_rank": evidence_rank,
        "empty_retrieval": 1.0 if not evidence else 0.0,
        "matched_targets": context_matches,
    }


def _mean_metric(records: list[dict[str, Any]], name: str) -> float | None:
    values = [record["metrics"][name] for record in records if record["metrics"].get(name) is not None]
    return round(mean(values), 3) if values else None


def evaluate_retrieval_cases(
    store: DocumentStore,
    cases: list[dict[str, Any]],
    *,
    top_k: int = 5,
    context_k: int | None = None,
    section_only: bool = False,
    use_reranker: bool = True,
    student_level: str = "intermediate",
    settings=None,
    include_text: bool = False,
) -> dict[str, Any]:
    records = []
    for case in cases:
        started = time.perf_counter()
        trace = trace_retrieval(
            store,
            case["doc_id"],
            case["query"],
            top_k=top_k,
            context_k=context_k,
            section_id=case.get("section_id", ""),
            section_only=case.get("section_only", section_only),
            use_reranker=case.get("use_reranker", use_reranker),
            student_level=case.get("student_level", student_level),
            settings=settings,
            include_text=include_text,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        records.append({
            "id": case.get("id", ""),
            "query": case["query"],
            "doc_id": case["doc_id"],
            "trace": trace,
            "elapsed_ms": elapsed_ms,
            "metrics": score_trace(trace, case),
        })
    return {
        "records": records,
        "summary": {
            "num_cases": len(records),
            "hit_at_k": _mean_metric(records, "hit_at_k"),
            "context_hit_at_k": _mean_metric(records, "context_hit_at_k"),
            "mrr_at_k": _mean_metric(records, "mrr_at_k"),
            "recall_at_k": _mean_metric(records, "recall_at_k"),
            "context_recall_at_k": _mean_metric(records, "context_recall_at_k"),
            "empty_retrieval_rate": _mean_metric(records, "empty_retrieval"),
        },
    }
