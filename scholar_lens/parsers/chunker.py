from __future__ import annotations

import re

from scholar_lens.parsers.models import Chunk, ChunkMetadata, ParsedDocument


def _estimate_tokens(text: str) -> int:
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 4


class SectionAwareChunker:
    """Section-aware chunking per spec Section 6.2."""

    def __init__(self, max_chunk_tokens: int = 800, overlap_tokens: int = 100) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: ParsedDocument, doc_id: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        if doc.sections:
            for section in doc.sections:
                section_text = section.get("text", "")
                section_id = section.get("id", "unknown")
                section_type = self._classify_section(section.get("title", ""))
                chapter = self._extract_chapter(section_id)
                if section_type == "references":
                    continue
                section_chunks = self._chunk_text(text=section_text, doc_id=doc_id, section_id=section_id, section_type=section_type, chapter=chapter)
                chunks.extend(section_chunks)
        elif doc.raw_text:
            chunks = self._chunk_text(text=doc.raw_text, doc_id=doc_id, section_id="0", section_type="prose", chapter="")
        return chunks

    def _chunk_text(self, text: str, doc_id: str, section_id: str, section_type: str, chapter: str) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []
        tokens = _estimate_tokens(text)
        if tokens <= self.max_chunk_tokens:
            return [self._make_chunk(text, doc_id, section_id, section_type, chapter, 0)]
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[Chunk] = []
        current_text = ""
        chunk_idx = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            combined = f"{current_text}\n\n{para}" if current_text else para
            if _estimate_tokens(combined) <= self.max_chunk_tokens:
                current_text = combined
            else:
                if current_text:
                    chunks.append(self._make_chunk(current_text, doc_id, section_id, section_type, chapter, chunk_idx))
                    chunk_idx += 1
                # If single paragraph exceeds limit, hard-split by sentences/chars
                if _estimate_tokens(para) > self.max_chunk_tokens:
                    sub_chunks = self._hard_split(para, doc_id, section_id, section_type, chapter, chunk_idx)
                    chunks.extend(sub_chunks)
                    chunk_idx += len(sub_chunks)
                    current_text = ""
                else:
                    current_text = para
        if current_text:
            chunks.append(self._make_chunk(current_text, doc_id, section_id, section_type, chapter, chunk_idx))
        return chunks

    def _hard_split(self, text: str, doc_id: str, section_id: str, section_type: str, chapter: str, start_idx: int) -> list[Chunk]:
        """Split oversized text by sentence boundaries, falling back to char limits."""
        max_chars = self.max_chunk_tokens * 4  # rough: 4 chars per token
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[Chunk] = []
        current = ""
        idx = start_idx
        for sent in sentences:
            if not sent.strip():
                continue
            combined = f"{current} {sent}" if current else sent
            if len(combined) <= max_chars:
                current = combined
            else:
                if current:
                    chunks.append(self._make_chunk(current.strip(), doc_id, section_id, section_type, chapter, idx))
                    idx += 1
                # If single sentence > max_chars, split by char limit
                if len(sent) > max_chars:
                    for i in range(0, len(sent), max_chars):
                        chunks.append(self._make_chunk(sent[i:i + max_chars], doc_id, section_id, section_type, chapter, idx))
                        idx += 1
                    current = ""
                else:
                    current = sent
        if current:
            chunks.append(self._make_chunk(current.strip(), doc_id, section_id, section_type, chapter, idx))
        return chunks

    def _make_chunk(self, text: str, doc_id: str, section_id: str, section_type: str, chapter: str, idx: int) -> Chunk:
        chunk_id = f"{doc_id}_{section_id}_{idx}" if doc_id else f"{section_id}_{idx}"
        return Chunk(chunk_id=chunk_id, text=text, metadata=ChunkMetadata(section_id=section_id, section_type=section_type, chapter=chapter, doc_id=doc_id), layer="L2")

    def _classify_section(self, title: str) -> str:
        title_lower = title.lower()
        if title_lower in ("references", "bibliography"):
            return "references"
        if any(kw in title_lower for kw in ("method", "approach", "model", "architecture")):
            return "method"
        if any(kw in title_lower for kw in ("result", "experiment", "evaluation")):
            return "results"
        return "prose"

    def _extract_chapter(self, section_id: str) -> str:
        parts = section_id.split(".")
        return parts[0] if parts else ""
