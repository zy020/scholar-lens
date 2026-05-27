"""Document routes backed by DocumentStore.

Each uploaded document gets a directory under data/documents/{doc_id}/.
Documents transition through statuses: uploaded → parsing → ... → ready | failed.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
import re

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from scholar_lens.api.schemas import (
    DocumentAnalysisResponse,
    DocumentAnalysisDetailResponse,
    DocumentDetail,
    DocumentStatus,
    DocumentSummary,
    EnhancementApplyResponse,
    EnhancePlanResponse,
    OCREnhanceResponse,
    ParseQualityResponse,
    SectionSummary,
    VisionEnhanceResponse,
)
from scholar_lens.api.deps import get_document_store, get_memory_manager, get_settings
from scholar_lens.api.document_analysis import build_analysis_response, enhance_document_analysis
from scholar_lens.api.memory_events import record_memory_event
from scholar_lens.parsers.ocr_capabilities import detect_rapidocr_capability
from scholar_lens.rag.document_store import DocumentStore
from scholar_lens.rag.vector_index import index_document_chunks

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024
SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".pptx"}
PAPER_UPLOAD_SUFFIXES = {".pdf"}
COURSEWARE_UPLOAD_SUFFIXES = {".pdf", ".pptx"}
PPTX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

_store: DocumentStore | None = None


def _get_store() -> DocumentStore:
    return _store if _store is not None else get_document_store()


async def evaluate_parse_quality_with_llm(
    store: DocumentStore,
    doc_id: str,
    settings=None,
) -> ParseQualityResponse:
    settings = settings or get_settings()
    qualities = store.load_parse_quality(doc_id)
    if not settings.llm_api_key or not settings.llm_model:
        return ParseQualityResponse(
            doc_id=doc_id,
            source="llm",
            status="unavailable",
            qualities=qualities,
            message="LLM parse quality evaluation requires a configured LLM.",
        )
    parsed = store.load_parsed_document(doc_id)
    page_preview = []
    if parsed is not None:
        page_preview = [
            {
                "unit_id": f"page_{page.page_num}",
                "page": page.page_num,
                "text_preview": " ".join((page.text or "").split())[:600],
                "char_count": page.char_count,
            }
            for page in parsed.pages[:20]
        ]
    prompt_payload = {
        "heuristic_qualities": qualities[:20],
        "page_previews": page_preview,
    }
    from langchain_core.messages import HumanMessage
    from scholar_lens.core.llm_factory import ChatLLMFactory

    prompt = (
        "You are evaluating academic document parse quality. Return JSON only. "
        "For each unit, judge whether parsed text is usable for retrieval and QA. "
        "Schema: {\"qualities\":[{\"unit_id\":\"page_0\",\"llm_score\":0.0-1.0,"
        "\"quality\":\"good|weak|failed\",\"recommended_action\":\"keep|ocr|vision\","
        "\"llm_reason\":\"short reason\"}]}. Input:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False)}"
    )
    try:
        llm = ChatLLMFactory.from_settings(settings).create(streaming=False)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        llm_items = _parse_llm_quality_items(str(response.content))
    except Exception as exc:
        return ParseQualityResponse(
            doc_id=doc_id,
            source="llm",
            status="failed",
            qualities=qualities,
            message="LLM parse quality evaluation failed.",
            error=str(exc),
        )
    merged = _merge_llm_quality_items(qualities, llm_items)
    return ParseQualityResponse(
        doc_id=doc_id,
        source="llm",
        status="available",
        qualities=merged,
        message="LLM parse quality evaluation completed.",
    )


def _parse_llm_quality_items(content: str) -> list[dict]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        items = data.get("qualities", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _merge_llm_quality_items(qualities: list[dict], llm_items: list[dict]) -> list[dict]:
    by_unit = {str(item.get("unit_id", "")): item for item in llm_items if item.get("unit_id")}
    merged = []
    for quality in qualities:
        unit_id = str(quality.get("unit_id", ""))
        llm_item = by_unit.get(unit_id)
        if llm_item:
            updated = dict(quality)
            for key in ("llm_score", "quality", "recommended_action", "llm_reason"):
                if key in llm_item:
                    updated[key] = llm_item[key]
            merged.append(updated)
        else:
            merged.append(quality)
    known = {str(item.get("unit_id", "")) for item in merged}
    for item in llm_items:
        if str(item.get("unit_id", "")) not in known:
            merged.append(item)
    return merged


async def run_rapidocr_enhancement(
    store: DocumentStore,
    doc_id: str,
    mode: str = "auto",
) -> OCREnhanceResponse:
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    from scholar_lens.parsers.ocr_executor import OCRUnavailableError, RapidOCRExecutor

    try:
        executor = RapidOCRExecutor(prefer_gpu=mode != "cpu")
        source = store.source_path(doc_id)
        result = executor.run(source, pages=doc.ocr_recommended_pages)
    except OCRUnavailableError as exc:
        return OCREnhanceResponse(
            doc_id=doc_id,
            status="unavailable",
            message=str(exc),
            error=str(exc),
        )

    pages = [page.model_dump() for page in result.pages]
    vision_pages = [
        page["page"]
        for page in pages
        if page.get("vision_recommended")
    ]
    return OCREnhanceResponse(
        doc_id=doc_id,
        status=result.status,
        engine=result.engine,
        pages=pages,
        vision_recommended_pages=vision_pages,
        message="OCR enhancement completed.",
        error=result.error,
    )


async def run_vision_enhancement(
    store: DocumentStore,
    doc_id: str,
    pages: list[int],
    settings=None,
) -> VisionEnhanceResponse:
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    settings = settings or get_settings()
    from scholar_lens.core.settings import VisionConfig
    from scholar_lens.parsers.vision_executor import VisionEnhancementExecutor, VisionUnavailableError

    config = VisionConfig(
        api_key=settings.vision_api_key or "",
        base_url=settings.vision_base_url or "",
        model=settings.vision_model or "",
    )
    try:
        executor = VisionEnhancementExecutor(config=config)
        result = executor.run(store.source_path(doc_id), pages=pages)
    except VisionUnavailableError as exc:
        return VisionEnhanceResponse(
            doc_id=doc_id,
            status="unavailable",
            message=str(exc),
            error=str(exc),
        )

    return VisionEnhanceResponse(
        doc_id=doc_id,
        status=result.status,
        engine=result.engine,
        pages=[page.model_dump() for page in result.pages],
        message="Vision enhancement completed.",
        error=result.error,
    )


def _find_section_title(text: str, title: str) -> int:
    """Find *title* as a section heading in *text*.

    Tries line-start matches first (possibly with numeric prefix like \"3.1 Title\"),
    then falls back to plain substring search.  Returns char position or -1.
    """
    import re

    escaped = re.escape(title)
    # Line-start, optionally preceded by a section number like "3.1" or "3.1.1"
    line_pattern = re.compile(
        rf"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s+)?{escaped}\s*$",
        re.MULTILINE,
    )
    m = line_pattern.search(text)
    if m:
        return m.start()
    # Try case-insensitive at line start
    line_pattern_ci = re.compile(
        rf"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s+)?{escaped}\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = line_pattern_ci.search(text)
    if m:
        return m.start()
    # Fallback: plain string search
    pos = text.find(title)
    if pos >= 0:
        return pos
    return text.lower().find(title.lower())


def _text_diagnostics(parsed, sections: list[SectionSummary]) -> dict:
    from scholar_lens.parsers.pdf_parser import diagnose_text_quality
    return diagnose_text_quality(
        parsed.pages,
        raw_text=parsed.raw_text,
        sections=[s.model_dump() for s in sections] if sections else parsed.sections,
    )


def _summary_ocr_needed(text_quality: str, recommended_pages: list[int]) -> bool:
    if recommended_pages:
        return True
    return text_quality == "image_based"


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _courseware_section_summaries(parsed, doc_id: str) -> list[SectionSummary]:
    if parsed.pages:
        sections_by_id = {
            str(section.get("id") or section.get("section_id")): section
            for section in (parsed.sections or [])
            if section.get("id") or section.get("section_id")
        }
        summaries: list[SectionSummary] = []
        for idx, page in enumerate(parsed.pages):
            text = page.text.strip()
            if not text:
                continue
            section_id = f"slide_{page.page_num}"
            section = sections_by_id.get(section_id) or sections_by_id.get(f"slide_{idx}") or {}
            title = f"Slide {page.page_num + 1}"
            body = str(section.get("text") or "").strip() or text
            summaries.append(SectionSummary(
                section_id=section_id,
                title=title,
                level=section.get("level", 1),
                page_start=section.get("page_start", page.page_num),
                page_end=section.get("page_end", page.page_num),
                gist=body[:200],
            ))
        if summaries:
            return summaries

    sections: list[SectionSummary] = []
    for idx, section in enumerate(parsed.sections or []):
        text = str(section.get("text") or "").strip()
        if not text and not str(section.get("title") or "").strip():
            continue
        section_id = str(section.get("id") or section.get("section_id") or f"slide_{idx}")
        page_start = section.get("page_start", idx)
        try:
            slide_no = int(page_start) + 1
        except (TypeError, ValueError):
            slide_no = idx + 1
        sections.append(SectionSummary(
            section_id=section_id,
            title=f"Slide {slide_no}",
            level=section.get("level", 1),
            page_start=page_start,
            page_end=section.get("page_end", idx),
            gist=text[:200],
        ))

    return sections


def _build_sections_and_chunks(parsed, chunker, doc_id: str) -> tuple[list[SectionSummary], list]:
    toc_sections = parsed.sections or []
    if parsed.doc_subtype in {"slides_pdf", "courseware_pptx"}:
        sections = _courseware_section_summaries(parsed, doc_id)
        chunks = chunker.chunk(parsed, doc_id=doc_id)
    elif toc_sections and parsed.pages:
        sections, chunks = _chunk_by_toc(parsed, toc_sections, chunker, doc_id)
        if not chunks and parsed.raw_text:
            chunks = chunker.chunk(parsed, doc_id=doc_id)
    else:
        chunks = chunker.chunk(parsed, doc_id=doc_id)
        sections = [
            SectionSummary(
                section_id=s.get("id", s.get("section_id", str(i))),
                title=s.get("title", "") or f"Section {i + 1}",
                level=s.get("level", 1),
                page_start=s.get("page_start"),
                page_end=s.get("page_end"),
                gist=s.get("text", "")[:200] if s.get("text") else "",
            )
            for i, s in enumerate(toc_sections[:50] or [])
        ]

    if not sections and chunks:
        sections = [
            SectionSummary(
                section_id="0",
                title="Document",
                level=1,
                gist=parsed.raw_text[:200] if parsed.raw_text else "",
            )
        ]
    return sections, chunks


def _chunk_by_toc(parsed, toc_sections, chunker, doc_id: str) -> tuple[list[SectionSummary], list]:
    """Chunk by detecting TOC section-title boundaries in the parsed text.

    Instead of assigning whole pages, we search for each TOC title string
    in the raw text and split at those positions.  Each segment is assigned
    to the section whose title appears just before it, then chunked.
    """

    raw_text = parsed.raw_text
    if not raw_text:
        raw_text = "\n".join(p.text for p in (parsed.pages or []))

    # Build section summaries
    section_summaries: list[SectionSummary] = []
    for idx, s in enumerate(toc_sections):
        section_summaries.append(SectionSummary(
            section_id=f"{doc_id}_{idx + 1}",
            title=str(s.get("title", "") or f"Section {idx + 1}"),
            level=s.get("level", 1),
            page_start=s.get("page_start"),
            page_end=None,
            gist="",
        ))

    # Find each section title in the raw text.
    # Prefer matches at line-starts to avoid matching substrings in paper titles.
    boundaries: list[tuple[int, int]] = []  # (char_pos, section_idx)
    for idx, ss in enumerate(section_summaries):
        title = ss.title
        if not title:
            continue
        pos = _find_section_title(raw_text, title)
        if pos >= 0:
            boundaries.append((pos, idx))

    if not boundaries:
        return section_summaries, []

    boundaries.sort(key=lambda x: x[0])

    # Split text at boundaries and chunk each segment
    all_chunks: list = []
    for bi, (start_pos, sec_idx) in enumerate(boundaries):
        end_pos = boundaries[bi + 1][0] if bi + 1 < len(boundaries) else len(raw_text)
        seg_text = raw_text[start_pos:end_pos].strip()

        if not seg_text:
            continue

        sec_chunks = chunker._chunk_text(
            text=seg_text,
            doc_id=doc_id,
            section_id=section_summaries[sec_idx].section_id,
            section_type="prose",
            chapter=str(sec_idx),
        )
        for ci, c in enumerate(sec_chunks):
            c.chunk_id = f"{doc_id}_{section_summaries[sec_idx].section_id}_{ci}"
        all_chunks.extend(sec_chunks)

    # Preamble (before first section heading) → assign to first TOC section
    if boundaries and boundaries[0][0] > 100:
        preamble = raw_text[:boundaries[0][0]].strip()
        first_idx = 0  # always first TOC entry (Introduction / Abstract)
        first_sec_id = section_summaries[first_idx].section_id
        pre_chunks = chunker._chunk_text(
            text=preamble,
            doc_id=doc_id,
            section_id=first_sec_id,
            section_type="prose",
            chapter="0",
        )
        for ci, c in enumerate(pre_chunks):
            c.chunk_id = f"{doc_id}_{first_sec_id}_pre{ci}"
        all_chunks = pre_chunks + all_chunks

    # Update section gist from first chunk text
    sec_chunk_map: dict[str, list] = {}
    for c in all_chunks:
        sid = c.metadata.section_id
        if sid not in sec_chunk_map:
            sec_chunk_map[sid] = []
        sec_chunk_map[sid].append(c)

    for ss in section_summaries:
        sc = sec_chunk_map.get(ss.section_id, [])
        if sc:
            ss.gist = sc[0].text[:200]

    return section_summaries, all_chunks


def _forced_doc_subtype(suffix: str, document_kind: str) -> str:
    if document_kind == "paper":
        if suffix not in PAPER_UPLOAD_SUFFIXES:
            raise HTTPException(status_code=415, detail="Paper uploads accept PDF files only")
        return "research_paper"
    if document_kind == "courseware":
        if suffix not in COURSEWARE_UPLOAD_SUFFIXES:
            raise HTTPException(status_code=415, detail="Courseware uploads accept PDF or PPTX files only")
        return "courseware_pptx" if suffix == ".pptx" else "slides_pdf"
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only PDF or PPTX files are accepted")
    return "courseware_pptx" if suffix == ".pptx" else "research_paper"


def _coerce_parsed_subtype(parsed, forced_subtype: str):
    if parsed.doc_subtype == forced_subtype:
        return parsed
    return parsed.model_copy(update={"doc_subtype": forced_subtype})


async def _upload_document_with_kind(file: UploadFile, document_kind: str) -> DocumentSummary:
    filename = (file.filename or "document.pdf")
    suffix = Path(filename).suffix.lower()
    forced_subtype = _forced_doc_subtype(suffix, document_kind)

    store = _get_store()

    if store.name_exists(filename):
        raise HTTPException(status_code=409, detail=f"Document already exists: {filename}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100 MB)")

    doc = store.create_document(filename, suffix=suffix)
    source = store.save_source(doc.doc_id, content, suffix=suffix)

    # Parse synchronously
    try:
        store.update_status(doc.doc_id, DocumentStatus.parsing)
        from scholar_lens.parsers.pdf_parser import PDFParser
        from scholar_lens.parsers.ppt_parser import PPTParser
        from scholar_lens.parsers.chunker import SectionAwareChunker

        parser = PPTParser() if suffix == ".pptx" else PDFParser()
        parsed = _coerce_parsed_subtype(parser.parse(source), forced_subtype)
        store.save_parsed_document(doc.doc_id, parsed)

        store.update_status(doc.doc_id, DocumentStatus.chunking)
        chunker = SectionAwareChunker()
        sections, chunks = _build_sections_and_chunks(parsed, chunker, doc.doc_id)

        store.save_sections(doc.doc_id, sections)
        store.save_chunks(doc.doc_id, chunks)
        index_status = "keyword_only"
        try:
            index_status = "vector" if index_document_chunks(store, doc.doc_id, chunks, get_settings()) else "keyword_only"
        except Exception:
            index_status = "failed"
            logger.warning("Vector indexing failed after upload", exc_info=True)
        diagnostic_sections = sections
        if parsed.doc_subtype == "slides_pdf" and not (parsed.sections or []):
            diagnostic_sections = []
        text_diag = _text_diagnostics(parsed, diagnostic_sections)
        from scholar_lens.parsers.parse_quality import assess_parse_unit_quality, recommend_ocr_from_quality
        parse_qualities = assess_parse_unit_quality(parsed)
        store.save_parse_quality(doc.doc_id, parse_qualities)
        ocr_recommendation = recommend_ocr_from_quality(parse_qualities)
        store.update_summary(
            doc.doc_id,
            doc_type=parsed.doc_subtype,
            text_quality=text_diag["text_quality"],
            ocr_needed=_summary_ocr_needed(text_diag["text_quality"], ocr_recommendation.pages),
            ocr_recommended_pages=ocr_recommendation.pages,
            ocr_recommendation_reasons=ocr_recommendation.reasons,
            page_text_coverage=text_diag["page_text_coverage"],
            section_quality=text_diag["section_quality"],
            diagnostic_notes=text_diag["diagnostic_notes"],
            index_status=index_status,
        )
        await _auto_enhance_after_upload(store, doc.doc_id, settings=get_settings())
        store.update_status(doc.doc_id, DocumentStatus.ready)

        doc = store.get(doc.doc_id) or doc
        return doc

    except Exception as e:
        logger.exception("Document parsing failed")
        store.update_status(doc.doc_id, DocumentStatus.failed, error=str(e))
        doc = store.get(doc.doc_id) or doc
        return doc


def _vision_pages_from_policy(store: DocumentStore, doc_id: str, llm_quality_response: ParseQualityResponse | None = None) -> list[int]:
    pages: list[int] = []
    ocr_payload = store.load_ocr_enhancement(doc_id)
    for page in ocr_payload.get("vision_recommended_pages", []) if ocr_payload else []:
        if isinstance(page, int):
            pages.append(page)
        elif isinstance(page, str) and page.isdigit():
            pages.append(int(page))
    if llm_quality_response is not None:
        for item in llm_quality_response.qualities:
            if item.get("recommended_action") != "vision":
                continue
            page = item.get("page_start")
            if isinstance(page, int):
                pages.append(page)
            elif isinstance(page, str) and page.isdigit():
                pages.append(int(page))
            else:
                unit_id = str(item.get("unit_id", ""))
                match = re.search(r"(\d+)$", unit_id)
                if match:
                    pages.append(int(match.group(1)))
    return sorted(dict.fromkeys(pages))


def _payload_has_usable_enhancement_text(payload: dict, quality_key: str) -> bool:
    for page in payload.get("pages", []):
        if str(page.get(quality_key, page.get("quality", "failed"))) == "failed":
            continue
        if str(page.get("text", "")).strip():
            return True
    return False


async def _auto_enhance_after_upload(store: DocumentStore, doc_id: str, settings) -> None:
    doc = store.get(doc_id)
    if doc is None:
        return

    has_enhancement_payload = False
    llm_quality_response: ParseQualityResponse | None = None

    if bool(getattr(settings, "auto_ocr_enabled", True)) and doc.ocr_recommended_pages:
        try:
            ocr_response = await run_rapidocr_enhancement(store, doc_id, mode="auto")
            store.save_ocr_enhancement(doc_id, ocr_response)
            has_enhancement_payload = _payload_has_usable_enhancement_text(
                ocr_response.model_dump(),
                "ocr_quality",
            )
        except Exception as exc:
            logger.warning("Automatic OCR enhancement failed for %s", doc_id, exc_info=True)
            store.save_ocr_enhancement(doc_id, OCREnhanceResponse(
                doc_id=doc_id,
                status="failed",
                message="Automatic OCR enhancement failed.",
                error=str(exc),
            ))

    if bool(getattr(settings, "llm_quality_enabled", False)):
        try:
            llm_quality_response = await evaluate_parse_quality_with_llm(store, doc_id, settings=settings)
            if llm_quality_response.qualities:
                store.save_parse_quality(doc_id, llm_quality_response.qualities)
        except Exception:
            logger.warning("Automatic LLM parse quality evaluation failed for %s", doc_id, exc_info=True)

    vision_ready = bool(
        getattr(settings, "vision_enhancement_enabled", False)
        and getattr(settings, "vision_api_key", "")
        and getattr(settings, "vision_base_url", "")
        and getattr(settings, "vision_model", "")
    )
    if vision_ready:
        pages = _vision_pages_from_policy(store, doc_id, llm_quality_response=llm_quality_response)
        if pages:
            try:
                vision_response = await run_vision_enhancement(store, doc_id, pages=pages, settings=settings)
                store.save_vision_enhancement(doc_id, vision_response)
                has_enhancement_payload = has_enhancement_payload or _payload_has_usable_enhancement_text(
                    vision_response.model_dump(),
                    "vision_quality",
                )
            except Exception as exc:
                logger.warning("Automatic Vision enhancement failed for %s", doc_id, exc_info=True)
                store.save_vision_enhancement(doc_id, VisionEnhanceResponse(
                    doc_id=doc_id,
                    status="failed",
                    message="Automatic Vision enhancement failed.",
                    error=str(exc),
                ))

    if has_enhancement_payload:
        try:
            await apply_enhancement(doc_id)
        except Exception:
            logger.warning("Automatic enhancement apply failed for %s", doc_id, exc_info=True)


@router.post("/upload/paper", response_model=DocumentSummary)
async def upload_paper_document(file: UploadFile = File(...)):
    return await _upload_document_with_kind(file, document_kind="paper")


@router.post("/upload/courseware", response_model=DocumentSummary)
async def upload_courseware_document(file: UploadFile = File(...)):
    return await _upload_document_with_kind(file, document_kind="courseware")


@router.get("")
async def list_documents():
    return {"docs": _get_store().list()}


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    store = _get_store()
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    sections = store.load_sections(doc_id)
    return DocumentDetail(
        **doc.model_dump(),
        sections=sections,
    )


@router.get("/{doc_id}/sections")
async def get_sections(doc_id: str):
    store = _get_store()
    if store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"sections": store.load_sections(doc_id)}


@router.post("/{doc_id}/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(doc_id: str, force: bool = Query(False)):
    store = _get_store()
    if store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    result = await enhance_document_analysis(
        store,
        doc_id,
        settings=get_settings(),
        memory_manager=get_memory_manager(),
    )
    return DocumentAnalysisResponse(
        doc_id=result.doc_id,
        status=result.status,
        source=result.source,
        error=result.error,
    )


@router.post("/{doc_id}/enhance-plan", response_model=EnhancePlanResponse)
async def get_enhance_plan(doc_id: str):
    store = _get_store()
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    settings = get_settings()
    vision_available = bool(
        getattr(settings, "vision_model", "")
        and getattr(settings, "vision_api_key", "")
        and getattr(settings, "vision_base_url", "")
    )
    vision_enhancement_enabled = bool(getattr(settings, "vision_enhancement_enabled", False))
    ocr_capability = detect_rapidocr_capability(vision_available=vision_available)
    recommended_pages = doc.ocr_recommended_pages
    escalation_reasons = [
        "ocr_too_short_visual_high",
        "garbled_text",
        "diagram_like",
    ]
    status = "planned" if recommended_pages else "skipped"
    source = store.source_path(doc_id)
    pptx_scope_note = (
        " For PPTX, lightweight OCR/Vision only processes embedded slide images; text boxes and shapes are handled by the PPTX parser."
        if source.suffix.lower() == ".pptx"
        else ""
    )
    return EnhancePlanResponse(
        doc_id=doc_id,
        status=status,
        recommended_ocr_pages=recommended_pages,
        ocr_recommendation_reasons=doc.ocr_recommendation_reasons,
        estimated_ocr_pages=len(recommended_pages),
        ocr_engine=ocr_capability.engine,
        ocr_installed=ocr_capability.installed,
        ocr_gpu_available=ocr_capability.gpu_available,
        ocr_cpu_available=ocr_capability.cpu_available,
        ocr_recommended_mode=ocr_capability.recommended_mode,
        available_actions=ocr_capability.available_actions,
        vision_available=vision_available,
        vision_enhancement_enabled=vision_enhancement_enabled,
        vision_possible=vision_available and vision_enhancement_enabled and bool(recommended_pages),
        vision_escalation_reasons=escalation_reasons,
        message=(
            "OCR enhancement is recommended for selected pages; Vision may be used after OCR quality evaluation."
            + pptx_scope_note
            if recommended_pages
            else "No OCR enhancement is currently recommended." + pptx_scope_note
        ),
    )


@router.post("/{doc_id}/enhance/ocr", response_model=OCREnhanceResponse)
async def enhance_with_ocr(doc_id: str, mode: str = Query("auto")):
    store = _get_store()
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if not doc.ocr_recommended_pages:
        response = OCREnhanceResponse(
            doc_id=doc_id,
            status="skipped",
            pages=[],
            message="No OCR enhancement pages are currently recommended.",
        )
        store.save_ocr_enhancement(doc_id, response)
        return response

    response = await run_rapidocr_enhancement(store, doc_id, mode=mode)
    store.save_ocr_enhancement(doc_id, response)
    return response


@router.post("/{doc_id}/enhance/vision", response_model=VisionEnhanceResponse)
async def enhance_with_vision(doc_id: str):
    store = _get_store()
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    settings = get_settings()
    if not bool(getattr(settings, "vision_enhancement_enabled", True)):
        response = VisionEnhanceResponse(
            doc_id=doc_id,
            status="unavailable",
            message="Vision enhancement is not enabled.",
            error="Vision enhancement is not enabled.",
        )
        store.save_vision_enhancement(doc_id, response)
        return response
    if not settings.vision_api_key or not settings.vision_base_url or not settings.vision_model:
        response = VisionEnhanceResponse(
            doc_id=doc_id,
            status="unavailable",
            message="Vision model is not configured.",
            error="Vision model is not configured.",
        )
        store.save_vision_enhancement(doc_id, response)
        return response

    ocr_payload = store.load_ocr_enhancement(doc_id)
    pages = [
        int(page)
        for page in ocr_payload.get("vision_recommended_pages", [])
        if isinstance(page, int) or (isinstance(page, str) and page.isdigit())
    ]
    if not pages:
        pages = list(doc.ocr_recommended_pages)
    if not pages:
        response = VisionEnhanceResponse(
            doc_id=doc_id,
            status="skipped",
            pages=[],
            message="No Vision enhancement pages are currently recommended.",
        )
        store.save_vision_enhancement(doc_id, response)
        return response

    response = await run_vision_enhancement(store, doc_id, pages=pages, settings=settings)
    store.save_vision_enhancement(doc_id, response)
    return response


@router.post("/{doc_id}/enhance/apply", response_model=EnhancementApplyResponse)
async def apply_enhancement(doc_id: str):
    store = _get_store()
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    ocr_payload = store.load_ocr_enhancement(doc_id)
    vision_payload = store.load_vision_enhancement(doc_id)
    if not ocr_payload and not vision_payload:
        return EnhancementApplyResponse(
            doc_id=doc_id,
            status="missing",
            source="ocr",
            message="No OCR or Vision enhancement payload found.",
        )

    parsed = store.load_parsed_document(doc_id)
    if parsed is None:
        return EnhancementApplyResponse(
            doc_id=doc_id,
            status="missing",
            source="ocr",
            message="No parsed document artifact found.",
        )

    from scholar_lens.parsers.chunker import SectionAwareChunker
    from scholar_lens.parsers.enhancement_merge import fragments_from_ocr_payload, fragments_from_vision_payload, merge_enhancements
    from scholar_lens.parsers.parse_quality import assess_parse_unit_quality, recommend_ocr_from_quality

    fragments = fragments_from_ocr_payload(ocr_payload) + fragments_from_vision_payload(vision_payload)
    updated_pages = {
        fragment.page
        for fragment in fragments
        if fragment.quality != "failed" and fragment.text.strip()
    }
    if not updated_pages:
        return EnhancementApplyResponse(
            doc_id=doc_id,
            status="skipped",
            source="mixed" if ocr_payload and vision_payload else ("vision" if vision_payload else "ocr"),
            num_pages_updated=0,
            num_chunks=doc.num_chunks,
            message="No usable OCR or Vision enhancement text found.",
        )

    merged = merge_enhancements(parsed, fragments)

    chunker = SectionAwareChunker()
    sections, chunks = _build_sections_and_chunks(merged, chunker, doc_id)
    store.save_parsed_document(doc_id, merged, enhanced=True)
    store.save_sections(doc_id, sections)
    store.save_chunks(doc_id, chunks)
    index_status = "keyword_only"
    try:
        index_status = "vector" if index_document_chunks(store, doc_id, chunks, get_settings()) else "keyword_only"
    except Exception:
        index_status = "failed"
        logger.warning("Vector reindexing failed after enhancement apply", exc_info=True)

    parse_qualities = assess_parse_unit_quality(merged)
    store.save_parse_quality(doc_id, parse_qualities)
    ocr_recommendation = recommend_ocr_from_quality(parse_qualities)
    diagnostic_sections = sections
    if merged.doc_subtype == "slides_pdf" and not (merged.sections or []):
        diagnostic_sections = []
    text_diag = _text_diagnostics(merged, diagnostic_sections)
    store.update_summary(
        doc_id,
        doc_type=merged.doc_subtype,
        text_quality=text_diag["text_quality"],
        ocr_needed=_summary_ocr_needed(text_diag["text_quality"], ocr_recommendation.pages),
        ocr_recommended_pages=ocr_recommendation.pages,
        ocr_recommendation_reasons=ocr_recommendation.reasons,
        page_text_coverage=text_diag["page_text_coverage"],
        section_quality=text_diag["section_quality"],
        diagnostic_notes=text_diag["diagnostic_notes"],
        index_status=index_status,
    )

    return EnhancementApplyResponse(
        doc_id=doc_id,
        status="applied",
        source="mixed" if ocr_payload and vision_payload else ("vision" if vision_payload else "ocr"),
        num_pages_updated=len(updated_pages),
        num_chunks=len(chunks),
        message="Enhancement text applied and chunks regenerated.",
    )


@router.get("/{doc_id}/quality", response_model=ParseQualityResponse)
async def get_parse_quality(doc_id: str):
    store = _get_store()
    if store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    qualities = store.load_parse_quality(doc_id)
    status = "available" if qualities else "missing"
    message = "Heuristic parse quality is available." if qualities else "No parse quality artifact found."
    return ParseQualityResponse(
        doc_id=doc_id,
        source="heuristic",
        status=status,
        qualities=qualities,
        message=message,
    )


@router.post("/{doc_id}/quality/evaluate", response_model=ParseQualityResponse)
async def evaluate_parse_quality(doc_id: str, use_llm: bool = Query(False)):
    store = _get_store()
    if store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if use_llm:
        return await evaluate_parse_quality_with_llm(
            store,
            doc_id,
            settings=get_settings(),
        )

    qualities = store.load_parse_quality(doc_id)
    status = "available" if qualities else "missing"
    return ParseQualityResponse(
        doc_id=doc_id,
        source="heuristic",
        status=status,
        qualities=qualities,
        message="Heuristic parse quality returned without LLM evaluation.",
    )


@router.get("/{doc_id}/analysis", response_model=DocumentAnalysisDetailResponse)
async def get_document_analysis(doc_id: str):
    store = _get_store()
    if store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return build_analysis_response(store, doc_id)


@router.get("/{doc_id}/sections/{section_id}/text")
async def get_section_text(doc_id: str, section_id: str):
    store = _get_store()
    if store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = store.load_sections(doc_id)
    section = next((s for s in sections if s.section_id == section_id), None)
    if section is None:
        raise HTTPException(status_code=404, detail="Section not found")

    matching_chunks = [
        chunk for chunk in store.load_chunks(doc_id)
        if chunk.get("metadata", {}).get("section_id") == section_id
    ]
    text = "\n\n".join(chunk.get("text", "").strip() for chunk in matching_chunks if chunk.get("text", "").strip())
    if not text:
        parsed = store.load_parsed_document(doc_id, enhanced=True) or store.load_parsed_document(doc_id)
        if parsed is not None:
            parsed_section = next(
                (
                    item for item in (parsed.sections or [])
                    if str(item.get("id") or item.get("section_id") or "") == section_id
                ),
                None,
            )
            if parsed_section:
                text = str(parsed_section.get("text") or "").strip()
            if not text and section_id.startswith("slide_"):
                try:
                    page_num = int(section_id.removeprefix("slide_"))
                except ValueError:
                    page_num = -1
                page = next((item for item in parsed.pages if item.page_num == page_num), None)
                if page is not None:
                    text = (page.text or "").strip()
    if not text and section.gist:
        text = section.gist

    await record_memory_event(
        get_memory_manager(),
        "section_read",
        doc_id=doc_id,
        section_id=section_id,
        payload={"title": section.title, "num_chunks": len(matching_chunks)},
    )

    return {
        "doc_id": doc_id,
        "section_id": section.section_id,
        "title": section.title,
        "text": text,
        "num_chunks": len(matching_chunks),
    }


@router.get("/{doc_id}/file")
async def get_file(doc_id: str):
    store = _get_store()
    if store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    source = store.source_path(doc_id)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    media_type = PPTX_MEDIA_TYPE if source.suffix.lower() == ".pptx" else "application/pdf"
    return FileResponse(source, media_type=media_type)


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    store = _get_store()
    if store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    store.delete(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
