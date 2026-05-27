from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from scholar_lens.api.brief_builder import build_not_generated_brief, build_unavailable_brief
from scholar_lens.api.brief_graph import is_lecture_doc_type, run_brief_generation_graph
from scholar_lens.api.chat_service import configured_llm_configs
from scholar_lens.api.document_analysis import hydrate_memory_from_analysis, load_document_analysis
from scholar_lens.api.deps import get_document_store, get_memory_manager
from scholar_lens.api.memory_events import record_memory_event
from scholar_lens.api.schemas import NotesResponse, PaperBriefResponse
from scholar_lens.rag.document_store import DocumentStore

logger = logging.getLogger(__name__)
router = APIRouter()

_store: DocumentStore | None = None


def _get_store() -> DocumentStore:
    return _store if _store is not None else get_document_store()


async def _record_memory_event(memory, event_type: str, doc_id: str, section_id: str = "", payload: dict | None = None) -> None:
    await record_memory_event(
        memory,
        event_type,
        doc_id=doc_id,
        section_id=section_id,
        payload=payload or {},
    )


BRIEF_LLM_MIN_TIMEOUT_SECONDS = 120.0


def _brief_cache_path(doc_id: str) -> Path:
    return _get_store().document_dir(doc_id) / "paper_brief.json"


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


def _is_lecture_doc(doc_type: str, sections: list | None = None) -> bool:
    return is_lecture_doc_type(doc_type)


def build_concept_map_markdown(doc_id: str, sections: list) -> str:
    lines = [f"# Concept Map — {doc_id}", "", "```mermaid", "graph TD", "  doc[Document]"]
    parent_by_level: dict[int, str] = {0: "doc"}
    for index, section in enumerate(sections[:30], start=1):
        node_id = f"s{index}"
        title = getattr(section, "title", "") or getattr(section, "section_id", f"Section {index}")
        safe_title = str(title).replace('"', "'").replace("[", "(").replace("]", ")")
        level = max(1, int(getattr(section, "level", 1) or 1))
        parent_id = parent_by_level.get(level - 1, "doc")
        lines.append(f'  {node_id}["{safe_title}"]')
        lines.append(f"  {parent_id} --> {node_id}")
        parent_by_level[level] = node_id
        for deeper in [k for k in parent_by_level if k > level]:
            parent_by_level.pop(deeper, None)
    if not sections:
        lines.append('  empty["No extracted sections"]')
        lines.append("  doc --> empty")
    lines.extend(["```", ""])
    return "\n".join(lines)


@router.get("/{doc_id}", response_model=NotesResponse)
async def get_notes(doc_id: str):
    store = _get_store()
    memory = get_memory_manager()
    understanding = load_document_analysis(store, doc_id)
    if understanding is not None:
        hydrate_memory_from_analysis(memory, understanding, doc_id=doc_id)
    core = memory.core_memory
    terms = []
    for e in core.active_glossary:
        parts = e.split("|||", 1)
        terms.append({"english": parts[0], "chinese": parts[1] if len(parts) > 1 else ""})
    if understanding is not None:
        existing = {term["english"] for term in terms}
        for term in understanding.key_terms:
            if term.english and term.english not in existing:
                terms.append({"english": term.english, "chinese": term.chinese})
                existing.add(term.english)
    return NotesResponse(
        doc_id=doc_id,
        terms=terms,
        reading_progress={},
        concept_map=understanding.mermaid_map if understanding and understanding.mermaid_map else core.session_summary,
    )


@router.get("/{doc_id}/brief", response_model=PaperBriefResponse)
async def get_paper_brief(doc_id: str, force: bool = Query(False)):
    store = _get_store()
    memory = get_memory_manager()
    doc = store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if not force:
        cached = _load_cached_brief(doc_id)
        if cached:
            cached.source = "cached"
            await _record_memory_event(memory, "brief_view", doc_id=doc_id, payload={"source": "cached"})
            return cached
        sections = store.load_sections(doc_id)
        return build_not_generated_brief(
            doc_id,
            doc.name,
            brief_type="lecture" if _is_lecture_doc(doc.doc_type, sections) else "paper",
            text_quality=doc.text_quality,
            ocr_needed=doc.ocr_needed,
        )

    sections = store.load_sections(doc_id)
    chunks = store.load_chunks(doc_id)
    understanding = load_document_analysis(store, doc_id)

    from scholar_lens.api.deps import get_settings

    settings = get_settings()
    llm_configs = configured_llm_configs(settings)
    if llm_configs:
        try:
            from scholar_lens.core.llm_factory import ChatLLMFactory

            brief_llm_config = llm_configs[0].model_copy(
                update={"request_timeout": max(llm_configs[0].request_timeout, BRIEF_LLM_MIN_TIMEOUT_SECONDS)}
            )
            llm = ChatLLMFactory.from_settings(settings).create(config=brief_llm_config, streaming=False)
            brief = await run_brief_generation_graph(
                doc_id=doc_id,
                title=doc.name,
                doc_type=doc.doc_type,
                text_quality=doc.text_quality,
                ocr_needed=doc.ocr_needed,
                sections=sections,
                chunks=chunks,
                llm=llm,
            )
            _save_cached_brief(doc_id, brief)
            await _record_memory_event(memory, "brief_generate", doc_id=doc_id, payload={"brief_type": brief.brief_type, "source": brief.source})
            return brief
        except Exception as exc:
            brief = build_unavailable_brief(
                doc_id,
                doc.name,
                brief_type="lecture" if _is_lecture_doc(doc.doc_type, sections) else "paper",
                text_quality=doc.text_quality,
                ocr_needed=doc.ocr_needed,
                error=f"LLM brief failed; enhanced brief required: {exc}",
            )
            await _record_memory_event(memory, "brief_generate", doc_id=doc_id, payload={"brief_type": brief.brief_type, "source": brief.source, "error": brief.error[:300]})
            return brief

    brief = build_unavailable_brief(
        doc_id,
        doc.name,
        brief_type="lecture" if _is_lecture_doc(doc.doc_type, sections) else "paper",
        text_quality=doc.text_quality,
        ocr_needed=doc.ocr_needed,
        error="LLM brief is not configured; enhanced brief is required.",
    )
    await _record_memory_event(memory, "brief_generate", doc_id=doc_id, payload={"brief_type": brief.brief_type, "source": brief.source})
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

    # 4. Concept map from extracted sections
    cm_dir = memory.reflection._dir.parent / "concept_maps"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cm_path = cm_dir / f"{doc_id}_concepts.md"
    sections = _get_store().load_sections(doc_id)
    cm_path.write_text(build_concept_map_markdown(doc_id, sections), encoding="utf-8")
    files_written.append(str(cm_path))

    await _record_memory_event(memory, "export_obsidian", doc_id=doc_id, payload={"files": len(files_written)})

    return {"status": "exported", "files": files_written, "doc_id": doc_id}
