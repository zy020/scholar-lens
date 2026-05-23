from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from scholar_lens.api.brief_builder import build_fallback_brief, build_lecture_fallback_brief, build_low_text_brief
from scholar_lens.api.deps import get_memory_manager
from scholar_lens.api.schemas import NotesResponse, PaperBriefResponse
from scholar_lens.rag.document_store import DocumentStore

logger = logging.getLogger(__name__)
router = APIRouter()

_store = DocumentStore()
BRIEF_LLM_MIN_TIMEOUT_SECONDS = 120.0


def _brief_cache_path(doc_id: str) -> Path:
    return _store.document_dir(doc_id) / "paper_brief.json"


def _load_cached_brief(doc_id: str) -> PaperBriefResponse | None:
    path = _brief_cache_path(doc_id)
    if not path.exists():
        return None
    return PaperBriefResponse(**json.loads(path.read_text(encoding="utf-8")))


def _save_cached_brief(doc_id: str, brief: PaperBriefResponse) -> None:
    _brief_cache_path(doc_id).write_text(
        brief.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _is_lecture_doc(doc_type: str) -> bool:
    return doc_type in {"slides_pdf", "courseware_pptx", "lecture_slide", "courseware"}


def _needs_low_text_brief(text_quality: str, ocr_needed: bool) -> bool:
    return text_quality in {"image_based", "unknown"} or (text_quality == "weak" and ocr_needed)


@router.get("/{doc_id}", response_model=NotesResponse)
async def get_notes(doc_id: str):
    memory = get_memory_manager()
    core = memory.core_memory
    terms = []
    for e in core.active_glossary:
        parts = e.split("|||", 1)
        terms.append({"english": parts[0], "chinese": parts[1] if len(parts) > 1 else ""})
    return NotesResponse(
        doc_id=doc_id,
        terms=terms,
        reading_progress={},
        concept_map=core.session_summary,
    )


@router.get("/{doc_id}/brief", response_model=PaperBriefResponse)
async def get_paper_brief(doc_id: str, force: bool = Query(False)):
    doc = _store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if not force:
        cached = _load_cached_brief(doc_id)
        if cached:
            cached.source = "cached"
            return cached

    sections = _store.load_sections(doc_id)
    chunks = _store.load_chunks(doc_id)

    # Route low-text before any LLM call
    if _needs_low_text_brief(doc.text_quality, doc.ocr_needed):
        brief = build_low_text_brief(
            doc_id=doc_id,
            title=doc.name,
            doc_type=doc.doc_type,
            text_quality=doc.text_quality,
            diagnostic_notes=doc.diagnostic_notes,
        )
        _save_cached_brief(doc_id, brief)
        return brief

    from scholar_lens.api.brief_builder import build_llm_brief_prompt, build_lecture_llm_brief_prompt, parse_llm_brief_json
    from scholar_lens.api.deps import get_settings

    settings = get_settings()
    if settings.llm.api_key and settings.llm.model:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from scholar_lens.core.llm_factory import ChatLLMFactory

            brief_llm_config = settings.llm.model_copy(
                update={"request_timeout": max(settings.llm.request_timeout, BRIEF_LLM_MIN_TIMEOUT_SECONDS)}
            )
            llm = ChatLLMFactory.from_settings(settings).create(config=brief_llm_config, streaming=False)
            if _is_lecture_doc(doc.doc_type):
                prompt = build_lecture_llm_brief_prompt(doc.name, sections, chunks)
            else:
                prompt = build_llm_brief_prompt(doc.name, sections, chunks)
            response = await llm.ainvoke([
                SystemMessage(content="You produce strict JSON for academic paper understanding briefs."),
                HumanMessage(content=prompt),
            ])
            brief = parse_llm_brief_json(doc_id, doc.name, response.content)
            if _is_lecture_doc(doc.doc_type):
                brief.brief_type = "lecture"
            brief.text_quality = doc.text_quality
            brief.ocr_needed = doc.ocr_needed
            _save_cached_brief(doc_id, brief)
            return brief
        except Exception as exc:
            if _is_lecture_doc(doc.doc_type):
                brief = build_lecture_fallback_brief(
                    doc_id, doc.name, sections, chunks,
                    text_quality=doc.text_quality,
                    ocr_needed=doc.ocr_needed,
                )
            else:
                brief = build_fallback_brief(doc_id, doc.name, sections, chunks)
                brief.text_quality = doc.text_quality
                brief.ocr_needed = doc.ocr_needed
            brief.error = f"LLM brief failed; fallback used: {exc}"
            _save_cached_brief(doc_id, brief)
            return brief

    if _is_lecture_doc(doc.doc_type):
        brief = build_lecture_fallback_brief(
            doc_id, doc.name, sections, chunks,
            text_quality=doc.text_quality,
            ocr_needed=doc.ocr_needed,
        )
    else:
        brief = build_fallback_brief(doc_id, doc.name, sections, chunks)
        brief.text_quality = doc.text_quality
        brief.ocr_needed = doc.ocr_needed
    _save_cached_brief(doc_id, brief)
    return brief


@router.post("/{doc_id}/export")
async def export_obsidian(doc_id: str):
    """Batch 4: Export reading notes as Obsidian-compatible Markdown."""
    doc_id = doc_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    memory = get_memory_manager()
    core = memory.core_memory
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    files_written = []

    # 1. Reading log
    log_dir = memory.reflection._dir.parent / "reading_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{date_str}_{doc_id}.md"
    log_content = f"""---
date: {date_str}
doc_id: {doc_id}
position: {core.current_position}
---

# Reading Session — {doc_id}

## Session Summary
{core.session_summary or "First reading session."}

## Position
{core.current_position or "Not set."}

## Active Terms
"""
    for entry in core.active_glossary:
        log_content += f"- {entry}\n"
    log_path.write_text(log_content, encoding="utf-8")
    files_written.append(str(log_path))

    # 2. Glossary
    glossary_dir = memory.reflection._dir.parent / "glossary"
    glossary_dir.mkdir(parents=True, exist_ok=True)
    glossary_path = glossary_dir / f"{doc_id}_terms.md"
    glossary_content = f"# Glossary — {doc_id}\n\n"
    for entry in core.active_glossary:
        parts = entry.split("|||", 1)
        en = parts[0]
        zh = parts[1] if len(parts) > 1 else ""
        glossary_content += f"- **{en}**（{zh}）\n"
    glossary_path.write_text(glossary_content, encoding="utf-8")
    files_written.append(str(glossary_path))

    # 3. Reflection
    await memory.reflection.save_reflection(
        title=f"{doc_id}_reflection",
        content=f"# Reflection — {doc_id}\n\n"
                f"Date: {date_str}\n\n"
                f"Session summary: {core.session_summary}\n\n"
                f"Terms learned: {len(core.active_glossary)}\n\n"
                f"## Learning Patterns\n*Auto-generated after 5 sessions.*\n",
        tags=["auto", doc_id],
    )

    # 4. Concept map placeholder
    cm_dir = memory.reflection._dir.parent / "concept_maps"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cm_path = cm_dir / f"{doc_id}_concepts.md"
    cm_path.write_text(
        f"# Concept Map — {doc_id}\n\n"
        "```mermaid\ngraph TD\n"
        "  A[Document] --> B[Section 1]\n"
        "  A --> C[Section 2]\n"
        "```\n\n"
        "*Update with actual concept relationships.*\n",
        encoding="utf-8",
    )
    files_written.append(str(cm_path))

    return {"status": "exported", "files": files_written, "doc_id": doc_id}
