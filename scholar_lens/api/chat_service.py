from __future__ import annotations

from dataclasses import dataclass
import logging
import re

from scholar_lens.core.llm_factory import ChatLLMFactory
from scholar_lens.rag.document_index import DocumentIndex, evidence_from_results
from scholar_lens.rag.document_store import DocumentStore
from scholar_lens.rag.reranker import ModelReranker, RerankerPipeline
from scholar_lens.rag.retriever import HybridRetriever, RetrievalResult
from scholar_lens.rag.vector_index import search_vector_chunks

logger = logging.getLogger(__name__)


@dataclass
class ChatContext:
    results: list[RetrievalResult]
    evidence: list[dict]
    context_text: str
    has_formula_evidence: bool = False
    retrieval_debug: dict | None = None


@dataclass
class ChatRetrievalOptions:
    top_k: int = 5
    section_only: bool = False
    use_reranker: bool = True
    student_level: str = "intermediate"


def _complete_reranker_config(settings):
    config = getattr(settings, "reranker", None)
    if config and config.api_key and config.base_url and config.model:
        return config
    return None


def build_reranker_pipeline(settings=None) -> RerankerPipeline:
    model_reranker = None
    config = _complete_reranker_config(settings) if settings is not None else None
    if config is not None:
        model_reranker = ModelReranker(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
        )
    return RerankerPipeline(model_reranker=model_reranker)


def _normalize_rank(results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
    normalized = []
    for rank, result in enumerate(results[:top_k], 1):
        normalized.append(RetrievalResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            source=result.source,
            rank=rank,
            metadata=result.metadata,
        ))
    return normalized


def _apply_memory_retrieval_boost(
    results: list[RetrievalResult],
    memory_hints: dict | None = None,
) -> list[RetrievalResult]:
    if not results or not memory_hints:
        return results
    current_section_id = str(memory_hints.get("current_section_id") or "")
    concepts = [
        str(item).lower().strip()
        for item in (memory_hints.get("concepts") or [])
        if str(item).strip()
    ][:8]
    if not current_section_id and not concepts:
        return results
    boosted = []
    for result in results:
        score = result.score
        text = result.text.lower()
        if current_section_id and result.metadata.get("section_id") == current_section_id:
            score = score * 1.25 + 2.0
        for concept in concepts:
            concept_tokens = _tokenize_for_overlap(concept)
            if concept_tokens and concept_tokens.issubset(_tokenize_for_overlap(text)):
                score = score * 1.2 + 2.0
                break
            if concept and concept in text:
                score = score * 1.2 + 2.0
                break
        boosted.append(RetrievalResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=score,
            source=result.source,
            rank=result.rank,
            metadata=result.metadata,
        ))
    boosted.sort(key=lambda result: result.score, reverse=True)
    return [
        RetrievalResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            source=result.source,
            rank=rank,
            metadata=result.metadata,
        )
        for rank, result in enumerate(boosted, 1)
    ]


def _contains_cjk(query: str) -> bool:
    return any("一" <= char <= "鿿" for char in query)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _tokenize_for_overlap(text: str) -> set[str]:
    return set(re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower()))


def _rewrite_query_variants(query: str, settings=None, max_variants: int = 3) -> list[str]:
    query = query.strip()
    if not query:
        return []
    if not _contains_cjk(query):
        return [query]
    configs = configured_llm_configs(settings) if settings is not None else []
    if not configs:
        return [query]
    try:
        llm = ChatLLMFactory(configs[0]).create(streaming=False)
        variants = HybridRetriever().rewrite_query_sync(query, llm, num_variants=max_variants)
        return _unique_preserve_order([query, *variants])[: max_variants + 1]
    except Exception:
        logger.warning("Query rewrite failed, falling back to original query", exc_info=True)
        return [query]


def _search_bm25_variants(
    index: DocumentIndex,
    doc_id: str,
    query_variants: list[str],
    section_id: str,
    top_k: int,
    section_only: bool,
) -> list[RetrievalResult]:
    candidates: dict[str, RetrievalResult] = {}
    for variant in query_variants:
        query_tokens = _tokenize_for_overlap(variant)
        for result in index.search(
            doc_id,
            variant,
            section_id,
            top_k=top_k,
            section_only=section_only,
        ):
            if result.score <= 0 and query_tokens:
                overlap = len(query_tokens & _tokenize_for_overlap(result.text))
                if overlap > 0:
                    result = RetrievalResult(
                        chunk_id=result.chunk_id,
                        text=result.text,
                        score=float(overlap),
                        source=result.source,
                        rank=result.rank,
                        metadata=result.metadata,
                    )
            existing = candidates.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                candidates[result.chunk_id] = result
    ranked = sorted(candidates.values(), key=lambda result: result.score, reverse=True)[:top_k]
    return [
        RetrievalResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            source=result.source,
            rank=rank,
            metadata=result.metadata,
        )
        for rank, result in enumerate(ranked, 1)
    ]


def _stored_chunk_to_result(chunk: dict, score: float, rank: int) -> RetrievalResult:
    metadata = dict(chunk.get("metadata", {}))
    return RetrievalResult(
        chunk_id=chunk.get("chunk_id", ""),
        text=chunk.get("text", ""),
        score=score,
        source="context_expanded",
        rank=rank,
        metadata=metadata,
    )


def _same_section_neighbor(hit: RetrievalResult, candidate: dict) -> bool:
    hit_section = hit.metadata.get("section_id", "")
    candidate_section = candidate.get("metadata", {}).get("section_id", "")
    return bool(hit_section and hit_section == candidate_section)


def _adjacent_slide_neighbor(hit: RetrievalResult, candidate: dict) -> bool:
    hit_page = hit.metadata.get("page_start")
    candidate_page = candidate.get("metadata", {}).get("page_start")
    if hit_page is None or candidate_page is None:
        return False
    try:
        return abs(int(candidate_page) - int(hit_page)) <= 1
    except (TypeError, ValueError):
        return False


def _is_courseware_hit(hit: RetrievalResult) -> bool:
    return (
        hit.metadata.get("content_type") == "slide"
        or hit.metadata.get("section_type") == "slide"
        or str(hit.metadata.get("section_id", "")).startswith("slide_")
    )


def _is_formula_hit(hit: RetrievalResult) -> bool:
    return bool(hit.metadata.get("has_formula") or hit.metadata.get("formula_ids"))


def _detect_question_intent(query: str, *, has_formula_evidence: bool = False) -> str:
    normalized = query.lower()
    if has_formula_evidence or re.search(r"\b(formula|equation|latex)\b|公式|方程|推导|符号|变量", normalized):
        return "formula"
    if re.search(r"\b(figure|fig\.?|caption|image|diagram|plot|chart)\b|图|图片|图表|图像|可视化", normalized):
        return "figure"
    if re.search(r"\b(outline|agenda|structure|overview|flow|pipeline)\b|大纲|结构|脉络|流程|框架|整体|组织", normalized):
        return "structure"
    if re.search(r"\b(method|approach|algorithm|architecture|model|procedure|work)\b|方法|算法|架构|模型|如何工作|怎么工作", normalized):
        return "method"
    if re.search(r"\b(which|how many|where|when|list)\b|哪个|哪些|多少|几|列出|具体|分别|参数|数值", normalized):
        return "detail"
    return "concept"


def _chunk_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    normalized = re.sub(r"[^0-9a-z一-鿿 ]+", "", normalized)
    return normalized[:320]


def _courseware_neighbor_budget(intent: str, result_rank: int, query: str) -> int:
    if not query.strip():
        return 1
    if intent in {"figure", "detail"}:
        return 0
    if intent == "formula":
        return 1 if result_rank <= 1 else 0
    if intent == "structure":
        return 1
    return 1 if result_rank <= 2 else 0


def _metadata_list(metadata: dict, key: str) -> list[str]:
    value = metadata.get(key) or []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _paper_hit_has_visual_context(hit: RetrievalResult, intent: str) -> bool:
    content_type = str(hit.metadata.get("content_type") or "")
    return intent == "figure" or content_type in {"figure", "table"} or bool(hit.metadata.get("caption"))


def _paper_reference_related(hit: RetrievalResult, candidate: dict) -> bool:
    candidate_id = str(candidate.get("chunk_id") or "")
    candidate_meta = candidate.get("metadata", {})
    hit_refs = set(_metadata_list(hit.metadata, "cross_refs") + _metadata_list(hit.metadata, "referenced_by"))
    candidate_refs = set(_metadata_list(candidate_meta, "cross_refs") + _metadata_list(candidate_meta, "referenced_by"))
    return bool(
        candidate_id in hit_refs
        or hit.chunk_id in candidate_refs
        or hit_refs.intersection(candidate_refs)
    )


def _paper_formula_related(hit: RetrievalResult, candidate: dict) -> bool:
    hit_formula_ids = set(_metadata_list(hit.metadata, "formula_ids"))
    candidate_meta = candidate.get("metadata", {})
    candidate_formula_ids = set(_metadata_list(candidate_meta, "formula_ids"))
    if hit_formula_ids and candidate_formula_ids.intersection(hit_formula_ids):
        return True
    return bool(hit.metadata.get("has_formula") and candidate_meta.get("has_formula"))


def _paper_same_section_type_related(hit: RetrievalResult, candidate: dict, intent: str) -> bool:
    if not _same_section_neighbor(hit, candidate):
        return False
    candidate_type = str(candidate.get("metadata", {}).get("section_type") or "")
    hit_type = str(hit.metadata.get("section_type") or "")
    if intent == "method":
        return candidate_type == "method" and hit_type == "method"
    if intent in {"concept", "detail"}:
        return candidate_type and candidate_type == hit_type and candidate_type != "prose"
    return False


def _detect_answer_depth(question: str) -> str:
    normalized = question.lower()
    if re.search(
        r"\b(detail|detailed|expand|deep|in depth|comprehensive|example|examples|beginner|intuitive)\b"
        r"|详细|展开|深入|完整|多讲|举例|例子|通俗|从基础|不理解|仔细|全面",
        normalized,
    ):
        return "expanded"
    return "concise"


def _answer_constraint_guidance(question: str, *, has_formula_evidence: bool = False) -> str:
    intent = _detect_question_intent(question, has_formula_evidence=has_formula_evidence)
    depth = _detect_answer_depth(question)
    intent_guidance = {
        "figure": (
            "For figure or caption questions, answer what the figure/caption directly states. "
            "In concise mode, use 1-3 sentences and do not add mechanisms, examples, or background "
            "unless the student explicitly asks for them."
        ),
        "formula": (
            "For formula questions, start with the formula's role in one sentence, then explain symbols "
            "only when supported by evidence. If derivation steps are missing, say the document does not provide them."
        ),
        "detail": (
            "For detail questions, answer directly and avoid adding background unless it is needed to disambiguate the evidence."
        ),
        "structure": (
            "For structure or outline questions, synthesize across cited evidence in document order and do not invent missing sections."
        ),
        "method": (
            "For method questions, explain the mechanism using cited method evidence and avoid adding implementation details not in evidence."
        ),
        "concept": (
            "For concept questions, give a definition plus the key role or example supported by evidence; avoid encyclopedia-style expansion by default."
        ),
    }.get(intent, "")
    if depth == "expanded":
        depth_guidance = (
            "The student explicitly asks for an expanded explanation. You may provide a fuller teaching-style answer, "
            "but keep it grounded. Separate document evidence from helpful background explanation. "
            "Do not present background knowledge as document evidence, and clearly say when a point is general background rather than stated in the document."
        )
    else:
        depth_guidance = (
            "Default to a concise answer unless the student explicitly asks for a detailed, expanded, beginner-friendly, "
            "or example-based explanation. In concise mode, answer the direct question and stop; do not add an uncited "
            "concluding implication sentence."
        )
    return f"\nAnswer-depth and grounding guidance: {depth_guidance} {intent_guidance}\n"


def _personalized_guidance(memory_context: str, student_level: str = "intermediate") -> str:
    guidance = []
    normalized = memory_context.lower()
    if student_level == "beginner" or "beginner" in normalized or "基础" in memory_context:
        guidance.append("Use a beginner-friendly explanation style before technical compression.")
    if "|||" in memory_context:
        known_terms = []
        for line in memory_context.splitlines():
            if "|||" in line:
                known_terms.extend(part.split("|||", 1)[0].strip(" ,") for part in line.split(",") if "|||" in part)
        known_terms = [term for term in known_terms if term][:6]
        if known_terms:
            guidance.append(
                "The student has seen these terms before; preserve them and avoid re-teaching basics unless asked: "
                + ", ".join(known_terms)
                + "."
            )
    if re.search(r"公式|formula|equation|推导|符号", memory_context, re.IGNORECASE):
        guidance.append("The student recently struggled with formulas; explain formula roles and symbols carefully when evidence supports it.")
    if not guidance:
        return ""
    return "Personalized teaching guidance: " + " ".join(guidance)


def _format_result_context(result: RetrievalResult, label: str) -> str:
    prefix = str(result.metadata.get("contextual_prefix") or "").strip()
    parts = [f"[{label}] {result.text[:300]}"]
    if _is_formula_hit(result) and prefix:
        parts.append(prefix[:240])
    return "\n".join(parts)


def _enrich_formula_evidence(evidence: list[dict], results: list[RetrievalResult]) -> list[dict]:
    metadata_by_chunk_id = {result.chunk_id: result.metadata for result in results}
    enriched = []
    for item in evidence:
        updated = dict(item)
        metadata = metadata_by_chunk_id.get(str(item.get("chunk_id") or ""), {})
        if metadata.get("has_formula"):
            updated["has_formula"] = True
        if metadata.get("formula_ids"):
            updated["formula_ids"] = list(metadata.get("formula_ids") or [])
        enriched.append(updated)
    return enriched


def _expand_retrieval_context(
    store: DocumentStore,
    doc_id: str,
    results: list[RetrievalResult],
    limit: int,
    query: str = "",
    intent_hint: str | None = None,
) -> list[RetrievalResult]:
    if not results or limit <= 0:
        return []
    stored_chunks = store.load_chunks(doc_id)
    if not stored_chunks:
        return results[:limit]
    intent = intent_hint or _detect_question_intent(
        query,
        has_formula_evidence=any(_is_formula_hit(result) for result in results),
    )
    index_by_id = {
        chunk.get("chunk_id", ""): idx
        for idx, chunk in enumerate(stored_chunks)
        if chunk.get("chunk_id")
    }
    expanded: list[RetrievalResult] = []
    seen: set[str] = set()
    seen_courseware_fingerprints: set[str] = set()

    def add_result(result: RetrievalResult) -> None:
        if not result.chunk_id or result.chunk_id in seen or len(expanded) >= limit:
            return
        if _is_courseware_hit(result):
            fingerprint = _chunk_fingerprint(result.text)
            if fingerprint and fingerprint in seen_courseware_fingerprints:
                seen.add(result.chunk_id)
                return
            if fingerprint:
                seen_courseware_fingerprints.add(fingerprint)
        expanded.append(result)
        seen.add(result.chunk_id)

    for result in results:
        add_result(result)
    for result in results:
        if len(expanded) >= limit:
            break
        hit_idx = index_by_id.get(result.chunk_id)
        if hit_idx is None:
            continue
        if _is_courseware_hit(result):
            neighbor_budget = _courseware_neighbor_budget(intent, result.rank, query)
            if neighbor_budget <= 0:
                continue
            added_neighbors = 0
            candidates = range(max(0, hit_idx - 2), min(len(stored_chunks), hit_idx + 3))
            for idx in candidates:
                if added_neighbors >= neighbor_budget * 2:
                    break
                candidate = stored_chunks[idx]
                candidate_id = candidate.get("chunk_id", "")
                if candidate_id == result.chunk_id or candidate_id in seen:
                    continue
                if _adjacent_slide_neighbor(result, candidate):
                    before_len = len(expanded)
                    add_result(_stored_chunk_to_result(candidate, score=result.score * 0.85, rank=len(expanded) + 1))
                    if len(expanded) > before_len:
                        added_neighbors += 1
        else:
            for idx, candidate in enumerate(stored_chunks):
                if len(expanded) >= limit:
                    break
                candidate_id = candidate.get("chunk_id", "")
                if candidate_id == result.chunk_id or candidate_id in seen:
                    continue
                related = (
                    (_paper_hit_has_visual_context(result, intent) and _paper_reference_related(result, candidate))
                    or (intent == "formula" and _same_section_neighbor(result, candidate) and _paper_formula_related(result, candidate))
                    or _paper_same_section_type_related(result, candidate, intent)
                )
                if related:
                    add_result(_stored_chunk_to_result(candidate, score=result.score * 0.9, rank=len(expanded) + 1))
            for idx in (hit_idx - 1, hit_idx + 1):
                if idx < 0 or idx >= len(stored_chunks):
                    continue
                candidate = stored_chunks[idx]
                candidate_id = candidate.get("chunk_id", "")
                if candidate_id == result.chunk_id or candidate_id in seen:
                    continue
                if _same_section_neighbor(result, candidate):
                    add_result(_stored_chunk_to_result(candidate, score=result.score * 0.85, rank=len(expanded) + 1))
    return expanded


def retrieve_chat_context(
    store: DocumentStore,
    doc_id: str,
    message: str,
    section_id: str = "",
    top_k: int = 5,
    section_only: bool = False,
    use_reranker: bool = True,
    student_level: str = "intermediate",
    settings=None,
    context_k: int | None = None,
    memory_hints: dict | None = None,
    intent_hint: str | None = None,
) -> ChatContext:
    index = DocumentIndex(store)
    initial_intent = intent_hint or _detect_question_intent(message)
    candidate_k = max(top_k * 3, top_k) if use_reranker else top_k
    if initial_intent in {"structure", "method"}:
        candidate_k = max(candidate_k, top_k * 4, 8)
    elif initial_intent in {"detail", "figure"}:
        candidate_k = max(candidate_k, top_k * 2)
    if memory_hints and (memory_hints.get("concepts") or memory_hints.get("current_section_id")):
        candidate_k = max(candidate_k, top_k * 4, 8)
    query_variants = _rewrite_query_variants(message, settings)
    results = _search_bm25_variants(
        index,
        doc_id,
        query_variants,
        section_id,
        candidate_k,
        section_only,
    )
    vector_results = search_vector_chunks(
        doc_id,
        message,
        candidate_k,
        settings,
    )
    if vector_results:
        hybrid = HybridRetriever()
        results = hybrid.hybrid_search(
            query=message,
            query_embedding=[],
            vector_results=vector_results,
            bm25_results=results,
            top_k=candidate_k,
        )
    results = _apply_memory_retrieval_boost(results, memory_hints)
    if use_reranker and results:
        pipeline = build_reranker_pipeline(settings)
        results = pipeline.rerank(results, query=message, student_level=student_level)
    resolved_intent = intent_hint or _detect_question_intent(
        message,
        has_formula_evidence=any(_is_formula_hit(result) for result in results),
    )
    if context_k is not None:
        effective_context_k = max(context_k, top_k)
    elif resolved_intent == "detail":
        effective_context_k = top_k
    elif resolved_intent == "structure":
        effective_context_k = max(top_k * 3, top_k)
    elif resolved_intent == "formula":
        effective_context_k = max(top_k + 2, top_k)
    elif any(_is_courseware_hit(result) for result in results):
        effective_context_k = max(min(top_k + 2, 6), top_k)
    else:
        effective_context_k = max(top_k * 2, top_k)
    context_results = _expand_retrieval_context(
        store,
        doc_id,
        results,
        limit=effective_context_k,
        query=message,
        intent_hint=resolved_intent,
    )
    evidence_results = _normalize_rank(results, top_k)
    evidence_ids = {result.chunk_id for result in evidence_results}
    citation_by_chunk_id = {result.chunk_id: result.rank for result in evidence_results}
    evidence = _enrich_formula_evidence(evidence_from_results(evidence_results), evidence_results)
    parts = []
    next_context_num = 1
    for result in context_results:
        if result.chunk_id in evidence_ids:
            citation = citation_by_chunk_id[result.chunk_id]
            parts.append(_format_result_context(result, str(citation)))
        else:
            parts.append(_format_result_context(result, f"context {next_context_num}"))
            next_context_num += 1
    context_text = "\n\n".join(parts) if parts else "No relevant content found."
    return ChatContext(
        results=evidence_results,
        evidence=evidence,
        context_text=context_text,
        has_formula_evidence=any(_is_formula_hit(result) for result in evidence_results),
        retrieval_debug={
            "intent": resolved_intent,
            "intent_hint": intent_hint or "",
            "query_variants": query_variants,
            "candidate_k": candidate_k,
            "top_k": top_k,
            "context_k": effective_context_k,
            "evidence_count": len(evidence_results),
            "context_count": len(context_results),
            "has_formula_evidence": any(_is_formula_hit(result) for result in evidence_results),
            "section_only": section_only,
        },
    )


async def retrieve_chat_context_async(
    store: DocumentStore,
    doc_id: str,
    message: str,
    section_id: str = "",
    top_k: int = 5,
    section_only: bool = False,
    use_reranker: bool = True,
    student_level: str = "intermediate",
    settings=None,
    context_k: int | None = None,
    memory_hints: dict | None = None,
    intent_hint: str | None = None,
) -> ChatContext:
    return retrieve_chat_context(
        store,
        doc_id,
        message,
        section_id,
        top_k=top_k,
        section_only=section_only,
        use_reranker=use_reranker,
        student_level=student_level,
        settings=settings,
        context_k=context_k,
        memory_hints=memory_hints,
        intent_hint=intent_hint,
    )


def build_no_llm_answer(evidence: list[dict]) -> str:
    if evidence:
        source = evidence[0].get("quote", "").strip()
        if source:
            return (
                "当前未配置可用的 LLM，因此先返回基于检索证据的简要提示：\n\n"
                f"最相关片段：{source[:240]}\n\n"
                "请在设置中配置 API Key 和模型后，我可以生成更完整的中文讲解。"
            )
    return "当前未配置可用的 LLM，且没有检索到可用文档证据。请先确认文档已解析完成并配置模型。"


def configured_llm_configs(settings) -> list:
    configs = []
    for config in (settings.llm, settings.backup_llm):
        if config is not None and config.api_key and config.model:
            configs.append(config)
    return configs


def build_chat_messages(
    *,
    question: str,
    context_text: str,
    section_title: str = "",
    has_formula_evidence: bool = False,
    memory_context: str = "",
    student_level: str = "intermediate",
):
    from langchain_core.messages import HumanMessage, SystemMessage
    from scholar_lens.agents.prompts import EXPLAINER_SYSTEM

    section_context = f"Current section: {section_title}\n" if section_title else ""
    formula_guidance = ""
    if has_formula_evidence:
        formula_guidance = (
            "\nFormula-answer guidance for 公式相关问题: "
            "引用原始公式; 解释变量或符号 only when supported by the evidence; "
            "说明该公式在当前页/章节中的作用; 如果证据没有给出推导过程，明确说明材料中没有足够推导信息，不要编造推导。\n"
        )
    answer_constraint_guidance = _answer_constraint_guidance(
        question,
        has_formula_evidence=has_formula_evidence,
    )
    vision_structured_guidance = ""
    if "Visual type:" in context_text:
        vision_structured_guidance = (
            "\nVision-structured evidence guidance: Some visual evidence may be a model-generated description "
            "of a page image, chart, table, diagram, or formula. Use it as cited evidence, but do not present "
            "inferred visual interpretation as verbatim source text. If chart/table/formula details are missing, say so.\n"
        )
    memory_section = ""
    if memory_context.strip():
        personalization = _personalized_guidance(memory_context, student_level=student_level)
        memory_section = (
            "Learning memory context:\n"
            f"{memory_context.strip()[:1200]}\n"
            "Memory is for personalization only. It may guide explanation style, known terms, and continuity, "
            "but document evidence remains the only source for factual claims.\n\n"
        )
        if personalization:
            memory_section += f"{personalization}\n\n"
    user_prompt = f"""{section_context}Evidence from the document:
{context_text}

{memory_section}
Student question: {question}
{formula_guidance}
{vision_structured_guidance}
{answer_constraint_guidance}

Answer in Chinese. Preserve key English terms inline when needed, but do not append a glossary, related terms, vocabulary list, or extra terminology section unless the student explicitly asks for one. Use only the evidence items that directly support your answer. Number citations using the provided evidence numbers, for example [1]. Do not cite evidence you did not use. Do not present broader implications, benefits, causes, or mechanisms as document conclusions unless they are directly stated in the cited evidence. If you add helpful general background, put it in a separate sentence explicitly labeled as general background, not as a document claim. If evidence is insufficient, say so."""
    return [
        SystemMessage(content=EXPLAINER_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
