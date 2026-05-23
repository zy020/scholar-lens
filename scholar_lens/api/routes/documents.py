"""Document routes backed by DocumentStore.

Each uploaded document gets a directory under data/documents/{doc_id}/.
Documents transition through statuses: uploaded → parsing → ... → ready | failed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from scholar_lens.api.schemas import (
    DocumentDetail,
    DocumentStatus,
    DocumentSummary,
    SectionSummary,
)
from scholar_lens.rag.document_store import DocumentStore

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024

_store = DocumentStore()


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


@router.post("/upload", response_model=DocumentSummary)
async def upload_document(file: UploadFile = File(...)):
    filename = (file.filename or "document.pdf")
    suffix = Path(filename).suffix.lower()

    if suffix != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are accepted")

    if _store.name_exists(filename):
        raise HTTPException(status_code=409, detail=f"Document already exists: {filename}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100 MB)")

    doc = _store.create_document(filename)
    _store.save_source(doc.doc_id, content)

    # Parse synchronously
    try:
        _store.update_status(doc.doc_id, DocumentStatus.parsing)
        from scholar_lens.parsers.pdf_parser import PDFParser
        from scholar_lens.parsers.chunker import SectionAwareChunker

        source = _store.source_path(doc.doc_id)
        parser = PDFParser()
        parsed = parser.parse(source)

        _store.update_status(doc.doc_id, DocumentStatus.chunking)
        chunker = SectionAwareChunker(max_chunk_tokens=800)

        toc_sections = parsed.sections or []
        if toc_sections and parsed.pages:
            sections, chunks = _chunk_by_toc(
                parsed, toc_sections, chunker, doc.doc_id
            )
        else:
            chunks = chunker.chunk(parsed, doc_id=doc.doc_id)
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

        _store.save_sections(doc.doc_id, sections)
        _store.save_chunks(doc.doc_id, chunks)
        text_diag = _text_diagnostics(parsed, sections)
        _store.update_summary(
            doc.doc_id,
            doc_type=parsed.doc_subtype,
            text_quality=text_diag["text_quality"],
            ocr_needed=text_diag["ocr_needed"],
            page_text_coverage=text_diag["page_text_coverage"],
            section_quality=text_diag["section_quality"],
            diagnostic_notes=text_diag["diagnostic_notes"],
        )
        _store.update_status(doc.doc_id, DocumentStatus.ready)

        doc = _store.get(doc.doc_id) or doc
        return doc

    except Exception as e:
        logger.exception("Document parsing failed")
        _store.update_status(doc.doc_id, DocumentStatus.failed, error=str(e))
        doc = _store.get(doc.doc_id) or doc
        return doc


@router.get("")
async def list_documents():
    return {"docs": _store.list()}


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    doc = _store.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    sections = _store.load_sections(doc_id)
    return DocumentDetail(
        **doc.model_dump(),
        sections=sections,
    )


@router.get("/{doc_id}/sections")
async def get_sections(doc_id: str):
    if _store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"sections": _store.load_sections(doc_id)}


@router.get("/{doc_id}/sections/{section_id}/text")
async def get_section_text(doc_id: str, section_id: str):
    if _store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = _store.load_sections(doc_id)
    section = next((s for s in sections if s.section_id == section_id), None)
    if section is None:
        raise HTTPException(status_code=404, detail="Section not found")

    matching_chunks = [
        chunk for chunk in _store.load_chunks(doc_id)
        if chunk.get("metadata", {}).get("section_id") == section_id
    ]
    text = "\n\n".join(chunk.get("text", "").strip() for chunk in matching_chunks if chunk.get("text", "").strip())
    if not text and section.gist:
        text = section.gist

    return {
        "doc_id": doc_id,
        "section_id": section.section_id,
        "title": section.title,
        "text": text,
        "num_chunks": len(matching_chunks),
    }


@router.get("/{doc_id}/file")
async def get_file(doc_id: str):
    if _store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    source = _store.source_path(doc_id)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    return FileResponse(source, media_type="application/pdf")


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    if _store.get(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    _store.delete(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
