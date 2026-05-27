from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from scholar_lens.api.schemas import DocumentAnalysisDetailResponse, SectionSummary
from scholar_lens.core.models import DocumentUnderstanding, Section, Term
from scholar_lens.rag.document_store import DocumentStore

TERM_TRANSLATIONS = {
    "RAG": "检索增强生成",
    "LLM": "大语言模型",
    "self-attention": "自注意力",
    "reranker": "重排器",
    "Transformer": "Transformer 模型",
    "embedding": "嵌入表示",
}


@dataclass
class AnalysisRunResult:
    doc_id: str
    status: str
    source: str = "unavailable"
    error: str = ""


def _chunk_section_id(chunk: dict) -> str:
    return str(chunk.get("metadata", {}).get("section_id", ""))


def _quote(text: str, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def _chunks_by_section(chunks: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        grouped.setdefault(_chunk_section_id(chunk), []).append(chunk)
    return grouped


def _section_text(section: SectionSummary, grouped: dict[str, list[dict]]) -> str:
    chunks = grouped.get(section.section_id, [])
    text = " ".join(chunk.get("text", "") for chunk in chunks)
    return text or section.gist


def _extract_terms(text: str) -> list[Term]:
    found: list[Term] = []
    lower = text.lower()
    for english, chinese in TERM_TRANSLATIONS.items():
        if english.lower() in lower:
            found.append(Term(english=english, chinese=chinese))
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9-]{2,}\b", text):
        term = match.group(0)
        if not any(item.english == term for item in found):
            found.append(Term(english=term, chinese=""))
        if len(found) >= 12:
            break
    return found[:12]


def _build_mermaid(doc_id: str, sections: list[SectionSummary]) -> str:
    lines = ["graph TD", f'  doc["{doc_id}"]']
    parent_by_level: dict[int, str] = {0: "doc"}
    for index, section in enumerate(sections[:30], start=1):
        node_id = f"s{index}"
        title = (section.title or section.section_id).replace('"', "'").replace("[", "(").replace("]", ")")
        level = max(1, int(section.level or 1))
        parent = parent_by_level.get(level - 1, "doc")
        lines.append(f'  {node_id}["{title}"]')
        lines.append(f"  {parent} --> {node_id}")
        parent_by_level[level] = node_id
        for deeper in [item for item in parent_by_level if item > level]:
            parent_by_level.pop(deeper, None)
    if not sections:
        lines.append('  empty["Document"]')
        lines.append("  doc --> empty")
    return "\n".join(lines)


def build_understanding_from_store_data(
    *,
    doc_id: str,
    doc_type: str,
    sections: list[SectionSummary],
    chunks: list[dict],
) -> DocumentUnderstanding:
    grouped = _chunks_by_section(chunks)
    model_sections = [
        Section(
            section_id=section.section_id,
            title=section.title,
            level=section.level,
            page_start=section.page_start,
            page_end=section.page_end,
        )
        for section in sections
    ]
    l0_summaries: dict[str, str] = {}
    l1_overviews: dict[str, str] = {}
    for section in sections:
        text = _section_text(section, grouped)
        l0_summaries[section.section_id] = _quote(text, 220) or section.title
        l1_overviews[section.section_id] = _quote(text, 900) or l0_summaries[section.section_id]

    all_text = "\n".join(chunk.get("text", "") for chunk in chunks[:30])
    return DocumentUnderstanding(
        doc_type=doc_type or "research_paper",
        language="en",
        difficulty="intermediate",
        estimated_reading_time=max(5, min(120, len(all_text.split()) // 180 or 5)),
        sections=model_sections,
        mermaid_map=_build_mermaid(doc_id, sections),
        key_terms=_extract_terms(all_text),
        l0_summaries=l0_summaries,
        l1_overviews=l1_overviews,
    )


def save_document_analysis(
    store: DocumentStore,
    doc_id: str,
    parsed_doc_type: str = "research_paper",
) -> DocumentUnderstanding:
    understanding = build_understanding_from_store_data(
        doc_id=doc_id,
        doc_type=parsed_doc_type,
        sections=store.load_sections(doc_id),
        chunks=store.load_chunks(doc_id),
    )
    store.save_understanding(doc_id, understanding)
    store.save_analysis_meta(doc_id, {
        "source": "parser",
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "error": "",
    })
    return understanding


def load_document_analysis(store: DocumentStore, doc_id: str) -> DocumentUnderstanding | None:
    return store.load_understanding(doc_id)


def build_analysis_response(store: DocumentStore, doc_id: str) -> DocumentAnalysisDetailResponse:
    understanding = store.load_understanding(doc_id)
    if understanding is None:
        return _build_parse_quality_analysis_response(store, doc_id)

    meta = store.load_analysis_meta(doc_id)
    return DocumentAnalysisDetailResponse(
        doc_id=doc_id,
        status="available",
        source=meta.get("source", "parser"),
        updated_at=meta.get("updated_at", ""),
        error=meta.get("error", ""),
        difficulty=understanding.difficulty,
        estimated_reading_time=understanding.estimated_reading_time,
        key_terms=[term.model_dump() for term in understanding.key_terms],
        l0_summaries=understanding.l0_summaries,
        l1_overviews=understanding.l1_overviews,
        mermaid_map=understanding.mermaid_map,
    )


def _build_parse_quality_analysis_response(store: DocumentStore, doc_id: str) -> DocumentAnalysisDetailResponse:
    doc = store.get(doc_id)
    qualities = store.load_parse_quality(doc_id)
    if doc is None or not qualities:
        return DocumentAnalysisDetailResponse(doc_id=doc_id)

    enhanced_doc = store.load_parsed_document(doc_id, enhanced=True)
    enhancement_completed = bool(
        enhanced_doc and any(getattr(page, "enhanced", False) for page in enhanced_doc.pages)
    )
    weak_items = [
        item for item in qualities
        if str(item.get("quality", "")) in {"weak", "failed"}
        or str(item.get("recommended_action", "")) in {"ocr", "vision"}
    ]
    recommended_ocr_pages = list(getattr(doc, "ocr_recommended_pages", []) or [])
    status = "enhanced_completed" if enhancement_completed else ("needs_enhancement" if weak_items or recommended_ocr_pages else "usable")
    warnings: list[str] = []
    actions: list[str] = []
    if status == "enhanced_completed":
        actions.append("解析增强已完成，当前阅读、检索和问答将使用增强后的解析结果。")
    elif status == "needs_enhancement":
        warnings.append("部分页面解析质量偏低，检索和问答可能遗漏图片、图表或扫描文本。")
    if status != "enhanced_completed" and recommended_ocr_pages:
        actions.append("系统已按当前配置完成可用的自动增强；若仍偏低，可继续使用解析增强做进一步检查。")
    if status != "enhanced_completed" and any(item.get("recommended_action") == "vision" for item in qualities):
        actions.append("部分页面建议使用 Vision 解析。")
    elif status != "enhanced_completed" and recommended_ocr_pages:
        actions.append("如仍有图表、公式或乱码页面，可配置 Vision 后进行进一步增强。")
    if not actions:
        actions.append("当前解析质量可用于基础阅读和检索。")

    page_items = []
    for item in ([] if enhancement_completed else weak_items[:8]):
        page = item.get("page_start", item.get("page", ""))
        if isinstance(page, str) and page.isdigit():
            page = int(page)
        page_items.append({
            "page": page,
            "page_label": f"第 {int(page) + 1} 页" if isinstance(page, int) else "相关页面",
            "quality": item.get("quality", "unknown"),
            "recommended_action": item.get("recommended_action", "keep"),
            "reasons": item.get("reasons", []),
            "text_score": item.get("text_score", 0),
            "overall_score": item.get("overall_score", 0),
            "text_preview": item.get("text_preview", ""),
        })

    return DocumentAnalysisDetailResponse(
        doc_id=doc_id,
        status="available",
        source="parse_quality",
        difficulty="",
        estimated_reading_time=0,
        parse_quality_status=status,
        parse_quality_message=(
            "解析增强已完成。"
            if status == "enhanced_completed"
            else (
                "解析质量偏低，部分页面可能需要进一步增强。"
                if status == "needs_enhancement"
                else "解析质量可接受。"
            )
        ),
        parse_quality_warnings=warnings,
        parse_quality_actions=actions,
        parse_quality_pages=page_items,
    )


def hydrate_memory_from_analysis(memory, understanding: DocumentUnderstanding, doc_id: str = "") -> None:
    memory.document.load_from_document_understanding(
        doc_id=doc_id,
        l0=understanding.l0_summaries,
        l1=understanding.l1_overviews,
        mermaid_map=understanding.mermaid_map,
    )
    for term in understanding.key_terms:
        if term.english and term.chinese:
            memory.core_memory.add_glossary_entry(term.english, term.chinese)


def _full_text_from_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("text", "")).strip()


def _section_dicts(sections: list[SectionSummary]) -> list[dict]:
    return [
        {
            "section_id": section.section_id,
            "id": section.section_id,
            "title": section.title,
            "level": section.level,
            "page_start": section.page_start,
            "page_end": section.page_end,
        }
        for section in sections
    ]


def _has_llm_config(settings) -> bool:
    llm = getattr(settings, "llm", None)
    return bool(llm and llm.api_key and llm.model)


async def enhance_document_analysis(
    store: DocumentStore,
    doc_id: str,
    settings=None,
    analyzer=None,
    memory_manager=None,
) -> AnalysisRunResult:
    doc = store.get(doc_id)
    if doc is None:
        raise ValueError("Document not found")

    sections = store.load_sections(doc_id)
    chunks = store.load_chunks(doc_id)

    if analyzer is None and not _has_llm_config(settings):
        return AnalysisRunResult(
            doc_id=doc_id,
            status="unavailable",
            source="unavailable",
            error="LLM is not configured. Enable an LLM to generate document analysis.",
        )

    try:
        if analyzer is None:
            from scholar_lens.agents.doc_analyzer import DocumentAnalyzerAgent
            from scholar_lens.core.llm_factory import ChatLLMFactory

            llm = ChatLLMFactory.from_settings(settings).create(config=settings.llm, streaming=False)
            analyzer = DocumentAnalyzerAgent(llm=llm)

        from scholar_lens.api.document_analysis_graph import run_document_analysis_graph

        return await run_document_analysis_graph(
            store=store,
            doc_id=doc_id,
            analyzer=analyzer,
            memory_manager=memory_manager,
        )
    except Exception as exc:
        if store.load_understanding(doc_id) is None:
            store.save_analysis_meta(doc_id, {
                "source": "unavailable",
                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "error": f"LLM analysis failed: {exc}",
            })
        return AnalysisRunResult(
            doc_id=doc_id,
            status="unavailable",
            source="unavailable",
            error=f"LLM analysis failed: {exc}",
        )
