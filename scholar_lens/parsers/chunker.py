from __future__ import annotations

from dataclasses import dataclass
import re

from scholar_lens.core.token_tracker import estimate_tokens
from scholar_lens.parsers.math_normalizer import analyze_math_text
from scholar_lens.parsers.models import Chunk, ChunkMetadata, ParsedDocument

# Batch 3.2: Formula detection patterns
_FORMULA_PATTERNS = [
    r"\$\$.+?\$\$",           # display math $$...$$
    r"\$.+?\$",               # inline math $...$
    r"\\begin\{equation\}.*?\\end\{equation\}",  # equation environment
    r"\\begin\{align\}.*?\\end\{align\}",
]
_FORMULA_REGEX = re.compile("|".join(_FORMULA_PATTERNS), re.DOTALL)

# Batch 3.3: Cross-reference patterns
_CROSS_REF_PATTERNS = [
    r"(?:see\s+)?(?:Fig(?:ure)?\.?\s*\d+)",
    r"(?:see\s+)?(?:Eq(?:uation)?\.?\s*\(?\d+\)?)",
    r"(?:see\s+)?(?:Table\.?\s*\d+)",
    r"(?:see\s+)?(?:Section\.?\s*[\d.]+)",
]
_CROSS_REF_REGEX = re.compile("|".join(_CROSS_REF_PATTERNS), re.IGNORECASE)
COURSEWARE_DOC_TYPES = {"slides_pdf", "courseware_pptx"}


@dataclass(frozen=True)
class ChunkingPolicy:
    max_chunk_tokens: int
    overlap_tokens: int


@dataclass(frozen=True)
class SemanticBlock:
    text: str
    block_type: str = "prose"


@dataclass(frozen=True)
class SemanticStrategy:
    name: str
    split_structural_blocks_when_fit: bool = False
    bind_short_title_to_next: bool = False
    teaching_labels: tuple[str, ...] = ()


DEFAULT_CHUNKING_POLICIES = {
    "research_paper": ChunkingPolicy(max_chunk_tokens=800, overlap_tokens=100),
    "slides_pdf": ChunkingPolicy(max_chunk_tokens=500, overlap_tokens=50),
    "courseware_pptx": ChunkingPolicy(max_chunk_tokens=500, overlap_tokens=50),
}
DEFAULT_CHUNKING_POLICY = DEFAULT_CHUNKING_POLICIES["research_paper"]
PAPER_SEMANTIC_STRATEGY = SemanticStrategy(
    name="paper",
    split_structural_blocks_when_fit=True,
)
DEFAULT_SEMANTIC_STRATEGY = PAPER_SEMANTIC_STRATEGY
COURSEWARE_SEMANTIC_STRATEGY = SemanticStrategy(
    name="courseware",
    bind_short_title_to_next=True,
    teaching_labels=("definition", "theorem", "example", "proof", "note", "key idea"),
)
class SectionAwareChunker:
    """Section-aware chunking per spec Section 6.2."""

    def __init__(self, max_chunk_tokens: int | None = None, overlap_tokens: int | None = None) -> None:
        self._max_chunk_tokens_override = max_chunk_tokens
        self._overlap_tokens_override = overlap_tokens
        self.max_chunk_tokens = max_chunk_tokens or DEFAULT_CHUNKING_POLICY.max_chunk_tokens
        self.overlap_tokens = overlap_tokens or DEFAULT_CHUNKING_POLICY.overlap_tokens

    def chunk(self, doc: ParsedDocument, doc_id: str = "") -> list[Chunk]:
        self._apply_policy(doc)
        if doc.doc_subtype in COURSEWARE_DOC_TYPES:
            return self.chunk_courseware(doc, doc_id=doc_id)

        chunks: list[Chunk] = []
        has_section_text = any(str(section.get("text") or "").strip() for section in doc.sections)
        if doc.sections and has_section_text:
            for section in doc.sections:
                section_text = section.get("text", "")
                section_id = section.get("id", "unknown")
                section_type = self._classify_section(section.get("title", ""))
                chapter = self._extract_chapter(section_id)
                if section_type == "references":
                    continue
                section_chunks = self._chunk_text(
                    text=section_text,
                    doc_id=doc_id,
                    section_id=section_id,
                    section_type=section_type,
                    chapter=chapter,
                    semantic_strategy=self._semantic_strategy_for_doc(doc.doc_subtype, section_type),
                    page_start=section.get("page_start"),
                    page_end=section.get("page_end"),
                    content_source=str(section.get("content_source") or "parser"),
                    enhanced=bool(section.get("enhanced", False)),
                )
                chunks.extend(section_chunks)
        elif doc.raw_text:
            chunks = self._chunk_text(
                text=doc.raw_text,
                doc_id=doc_id,
                section_id="0",
                section_type="prose",
                chapter="",
                semantic_strategy=self._semantic_strategy_for_doc(doc.doc_subtype, "prose"),
            )
        return chunks

    def chunk_courseware(self, doc: ParsedDocument, doc_id: str = "") -> list[Chunk]:
        self._apply_policy(doc)
        chunks: list[Chunk] = []
        for idx, section in enumerate(self._courseware_sections(doc)):
            section_id = section["id"]
            title = str(section.get("title") or f"Slide {idx + 1}").strip()
            body = str(section.get("text") or "").strip()
            if body and title and not title.startswith("Slide ") and not body.startswith(title):
                text = f"{title}\n\n{body}"
            else:
                text = body or title

            slide_chunks = self._chunk_text(
                text=text,
                doc_id=doc_id,
                section_id=section_id,
                section_type="slide",
                chapter=str(idx + 1),
                semantic_strategy=self._semantic_strategy_for_doc(doc.doc_subtype, "slide"),
                page_start=section.get("page_start"),
                page_end=section.get("page_end"),
                content_source=str(section.get("content_source") or "parser"),
                enhanced=bool(section.get("enhanced", False)),
            )
            for chunk_idx, chunk in enumerate(slide_chunks):
                chunk.chunk_id = f"{doc_id}_{section_id}_{chunk_idx}" if doc_id else f"{section_id}_{chunk_idx}"
                if chunk.metadata.content_type == "text":
                    chunk.metadata.content_type = "slide"
                chunk.metadata.section_type = "slide"
                chunk.metadata.chapter = str(idx + 1)
                chunks.append(chunk)
        return chunks

    def _apply_policy(self, doc: ParsedDocument) -> None:
        policy = DEFAULT_CHUNKING_POLICIES.get(doc.doc_subtype, DEFAULT_CHUNKING_POLICY)
        if self._max_chunk_tokens_override is None:
            self.max_chunk_tokens = policy.max_chunk_tokens
        if self._overlap_tokens_override is None:
            self.overlap_tokens = policy.overlap_tokens

    def _semantic_strategy_for_doc(self, doc_subtype: str, section_type: str) -> SemanticStrategy:
        if doc_subtype == "research_paper":
            return PAPER_SEMANTIC_STRATEGY
        if doc_subtype in COURSEWARE_DOC_TYPES or section_type == "slide":
            return COURSEWARE_SEMANTIC_STRATEGY
        return PAPER_SEMANTIC_STRATEGY

    def _courseware_sections(self, doc: ParsedDocument) -> list[dict]:
        if doc.pages:
            sections_by_id = {
                str(section.get("id") or section.get("section_id")): section
                for section in (doc.sections or [])
                if section.get("id") or section.get("section_id")
            }
            slide_sections = []
            for idx, page in enumerate(doc.pages):
                text = page.text.strip()
                if not text:
                    continue
                section_id = f"slide_{page.page_num}"
                section = sections_by_id.get(section_id) or sections_by_id.get(f"slide_{idx}") or {}
                title = f"Slide {page.page_num + 1}"
                slide_sections.append({
                    "id": section_id,
                    "title": title,
                    "text": str(section.get("text") or "").strip() or text,
                    "page_start": page.page_num,
                    "page_end": page.page_num,
                    "content_source": str(section.get("content_source") or page.content_source or "parser"),
                    "enhanced": bool(section.get("enhanced", page.enhanced)),
                })
            if slide_sections:
                return slide_sections

        sections = []
        for idx, section in enumerate(doc.sections or []):
            text = str(section.get("text") or "").strip()
            original_title = str(section.get("title") or "").strip()
            if not text and not original_title:
                continue
            section_id = str(section.get("id") or section.get("section_id") or f"slide_{idx}")
            page_start = section.get("page_start")
            try:
                slide_no = int(page_start) + 1 if page_start is not None else idx + 1
            except (TypeError, ValueError):
                slide_no = idx + 1
            body = text or original_title
            if text and original_title and not text.startswith(original_title):
                body = f"{original_title}\n\n{text}"
            sections.append({
                "id": section_id,
                "title": f"Slide {slide_no}",
                "text": body,
                "page_start": page_start,
                "page_end": section.get("page_end"),
                "content_source": str(section.get("content_source") or "parser"),
                "enhanced": bool(section.get("enhanced", False)),
            })
        return sections

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        section_id: str,
        section_type: str,
        chapter: str,
        semantic_strategy: SemanticStrategy | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        content_source: str = "parser",
        enhanced: bool = False,
    ) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []
        strategy = semantic_strategy or DEFAULT_SEMANTIC_STRATEGY
        blocks = self._semantic_blocks(text, strategy)
        tokens = estimate_tokens(text)
        if tokens <= self.max_chunk_tokens and not self._should_split_when_fit(blocks, strategy):
            return [
                self._make_chunk(
                    text,
                    doc_id,
                    section_id,
                    section_type,
                    chapter,
                    0,
                    page_start=page_start,
                    page_end=page_end,
                    content_source=content_source,
                    enhanced=enhanced,
                )
            ]
        chunks: list[Chunk] = []
        current_text = ""
        chunk_idx = 0
        overlap_prefix = ""
        for block in blocks:
            block_text = block.text.strip()
            if not block_text:
                continue
            if block.block_type == "heading" and current_text:
                chunks.append(
                    self._make_chunk(
                        current_text,
                        doc_id,
                        section_id,
                        section_type,
                        chapter,
                        chunk_idx,
                        page_start=page_start,
                        page_end=page_end,
                        content_source=content_source,
                        enhanced=enhanced,
                    )
                )
                overlap_prefix = self._overlap_text(current_text)
                chunk_idx += 1
                current_text = ""
            combined = f"{current_text}\n\n{block_text}" if current_text else block_text
            if estimate_tokens(combined) <= self.max_chunk_tokens:
                current_text = combined
            else:
                if current_text:
                    chunks.append(
                        self._make_chunk(
                            current_text,
                            doc_id,
                            section_id,
                            section_type,
                            chapter,
                            chunk_idx,
                            page_start=page_start,
                            page_end=page_end,
                            content_source=content_source,
                            enhanced=enhanced,
                        )
                    )
                    overlap_prefix = self._overlap_text(current_text)
                    chunk_idx += 1
                # If single paragraph exceeds limit, hard-split by sentences/chars
                if estimate_tokens(block_text) > self.max_chunk_tokens:
                    split_prefix = self._fit_overlap_prefix(overlap_prefix, block_text)
                    split_text = f"{split_prefix} {block_text}".strip() if split_prefix else block_text
                    sub_chunks = self._hard_split(
                        split_text,
                        doc_id,
                        section_id,
                        section_type,
                        chapter,
                        chunk_idx,
                        page_start=page_start,
                        page_end=page_end,
                        content_source=content_source,
                        enhanced=enhanced,
                    )
                    chunks.extend(sub_chunks)
                    chunk_idx += len(sub_chunks)
                    overlap_prefix = self._overlap_text(sub_chunks[-1].text) if sub_chunks else ""
                    current_text = ""
                else:
                    fitted_prefix = self._fit_overlap_prefix(overlap_prefix, block_text)
                    current_text = f"{fitted_prefix}\n\n{block_text}".strip() if fitted_prefix else block_text
        if current_text:
            chunks.append(
                self._make_chunk(
                    current_text,
                    doc_id,
                    section_id,
                    section_type,
                    chapter,
                    chunk_idx,
                    page_start=page_start,
                    page_end=page_end,
                    content_source=content_source,
                    enhanced=enhanced,
                )
            )
        return chunks

    def _should_split_when_fit(self, blocks: list[SemanticBlock], strategy: SemanticStrategy) -> bool:
        if not strategy.split_structural_blocks_when_fit:
            return False
        return any(block.block_type == "heading" for block in blocks[1:])

    def _semantic_blocks(self, text: str, strategy: SemanticStrategy | None = None) -> list[SemanticBlock]:
        strategy = strategy or DEFAULT_SEMANTIC_STRATEGY
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        blocks: list[SemanticBlock] = []
        idx = 0
        while idx < len(raw_paragraphs):
            paragraph = raw_paragraphs[idx]
            block_type = self._classify_block(paragraph, strategy)

            if self._should_bind_to_next(block_type, paragraph, idx, strategy) and idx + 1 < len(raw_paragraphs):
                next_paragraph = raw_paragraphs[idx + 1]
                next_type = self._classify_block(next_paragraph, strategy)
                if next_type in self._bindable_next_types(block_type, strategy):
                    blocks.append(SemanticBlock(text=f"{paragraph}\n\n{next_paragraph}", block_type=block_type))
                    idx += 2
                    continue

            blocks.append(SemanticBlock(text=paragraph, block_type=block_type))
            idx += 1
        return blocks

    def _classify_block(self, text: str, strategy: SemanticStrategy | None = None) -> str:
        strategy = strategy or DEFAULT_SEMANTIC_STRATEGY
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "prose"
        if self._is_teaching_label(lines[0], strategy) and len(lines) <= 2:
            return "teaching_label"
        if self._is_heading(lines[0]) and len(lines) <= 2:
            return "heading"
        if all(self._is_list_item(line) for line in lines):
            return "list"
        if self._is_formula_block(text):
            return "formula"
        if self._is_caption(lines[0]):
            return "caption"
        if strategy.bind_short_title_to_next and self._is_short_slide_title(lines):
            return "slide_title"
        return "prose"

    def _should_bind_to_next(self, block_type: str, text: str, idx: int, strategy: SemanticStrategy) -> bool:
        if block_type in {"heading", "caption", "teaching_label"}:
            return True
        if block_type == "slide_title" and strategy.bind_short_title_to_next and idx == 0:
            return True
        return False

    def _bindable_next_types(self, block_type: str, strategy: SemanticStrategy) -> set[str]:
        if block_type == "caption":
            return {"prose"}
        if block_type == "heading":
            return {"prose"}
        if block_type == "teaching_label":
            return {"prose", "list"}
        if block_type == "slide_title" and strategy.bind_short_title_to_next:
            return {"prose", "list"}
        return set()

    def _is_heading(self, line: str) -> bool:
        if re.match(r"^#{1,6}\s+\S+", line):
            return True
        if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z][\w\s:/,-]{1,80}$", line):
            return True
        return False

    def _is_list_item(self, line: str) -> bool:
        return bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line))

    def _is_teaching_label(self, line: str, strategy: SemanticStrategy) -> bool:
        if not strategy.teaching_labels:
            return False
        normalized = line.strip().rstrip(":").lower()
        return normalized in strategy.teaching_labels

    def _is_short_slide_title(self, lines: list[str]) -> bool:
        if len(lines) != 1:
            return False
        line = lines[0]
        if self._is_list_item(line) or self._is_caption(line):
            return False
        return len(line) <= 80 and estimate_tokens(line) <= 16

    def _is_formula_block(self, text: str) -> bool:
        stripped = text.strip()
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        return bool(
            stripped.startswith("$$")
            or stripped.endswith("$$")
            or re.search(r"\\begin\{(?:equation|align)\}", stripped)
            or (
                len(lines) <= 3
                and any(re.match(r"^[A-Za-z][\w\s^_{}\\()+*/.,-]*=\s*.+$", line) for line in lines)
            )
        )

    def _is_caption(self, line: str) -> bool:
        return bool(re.match(r"^(?:Fig(?:ure)?\.?|Table)\s*\d+[:.]\s+", line, re.IGNORECASE))

    def _hard_split(
        self,
        text: str,
        doc_id: str,
        section_id: str,
        section_type: str,
        chapter: str,
        start_idx: int,
        page_start: int | None = None,
        page_end: int | None = None,
        content_source: str = "parser",
        enhanced: bool = False,
    ) -> list[Chunk]:
        """Split oversized text by sentence boundaries, falling back to char limits."""
        if self._can_split_by_words(text):
            return self._split_word_windows(
                text,
                doc_id,
                section_id,
                section_type,
                chapter,
                start_idx,
                page_start=page_start,
                page_end=page_end,
                content_source=content_source,
                enhanced=enhanced,
            )

        max_chars = self._max_chars_for_text(text)
        sentences = re.split(r"(?<=[.!?。！？；])\s*", text)
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
                    chunks.append(
                        self._make_chunk(
                            current.strip(),
                            doc_id,
                            section_id,
                            section_type,
                            chapter,
                            idx,
                            page_start=page_start,
                            page_end=page_end,
                            content_source=content_source,
                            enhanced=enhanced,
                        )
                    )
                    idx += 1
                # If single sentence > max_chars, split by char limit
                if len(sent) > max_chars:
                    step = self._char_step(max_chars)
                    for i in range(0, len(sent), step):
                        chunks.append(
                            self._make_chunk(
                                sent[i:i + max_chars],
                                doc_id,
                                section_id,
                                section_type,
                                chapter,
                                idx,
                                page_start=page_start,
                                page_end=page_end,
                                content_source=content_source,
                                enhanced=enhanced,
                            )
                        )
                        idx += 1
                    current = ""
                else:
                    current = sent
        if current:
            chunks.append(
                self._make_chunk(
                    current.strip(),
                    doc_id,
                    section_id,
                    section_type,
                    chapter,
                    idx,
                    page_start=page_start,
                    page_end=page_end,
                    content_source=content_source,
                    enhanced=enhanced,
                )
            )
        return chunks

    def chunk_with_facts(self, doc: ParsedDocument, doc_id: str = "", fact_max_tokens: int = 200) -> list[Chunk]:
        """Chunk document with additional fact-level chunks for number-dense regions.

        Fact chunks are smaller (default 200 tokens) and focus on sentences
        containing numerical values (scores, percentages, counts).
        They supplement, not replace, the regular section-level chunks.
        """
        regular = self.chunk(doc, doc_id)
        fact_chunks = []
        for chunk in regular:
            facts = self._extract_fact_chunks(chunk, doc_id, fact_max_tokens)
            fact_chunks.extend(facts)
        return regular + fact_chunks

    def _extract_fact_chunks(self, chunk: Chunk, doc_id: str, max_tokens: int) -> list[Chunk]:
        """Extract fact-level sub-chunks from number-dense regions."""
        sentences = re.split(r"(?<=[.!?。！？；])\s*", chunk.text)
        facts = []
        current_fact = ""
        num_count = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_nums = len(re.findall(r"\d+\.?\d*", sent))

            combined = f"{current_fact} {sent}" if current_fact else sent
            if estimate_tokens(combined) <= max_tokens:
                current_fact = combined
                num_count += sent_nums
            else:
                # Emit current fact if it has numbers
                if current_fact and num_count >= 1:
                    facts.append(self._make_fact_chunk(current_fact.strip(), chunk, doc_id, len(facts)))
                current_fact = sent
                num_count = sent_nums

        if current_fact and num_count >= 1:
            facts.append(self._make_fact_chunk(current_fact.strip(), chunk, doc_id, len(facts)))

        return facts

    def _make_fact_chunk(self, text: str, parent: Chunk, doc_id: str, idx: int) -> Chunk:
        chunk_id = f"{parent.chunk_id}_f{idx}"
        meta = parent.metadata.model_copy(update={"content_type": "fact"})
        return Chunk(chunk_id=chunk_id, text=text, metadata=meta, layer="L2")

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        section_id: str,
        section_type: str,
        chapter: str,
        idx: int,
        page_start: int | None = None,
        page_end: int | None = None,
        content_source: str = "parser",
        enhanced: bool = False,
    ) -> Chunk:
        chunk_id = f"{doc_id}_{section_id}_{idx}" if doc_id else f"{section_id}_{idx}"
        math_analysis = analyze_math_text(text)
        vision_visual_type = self._vision_visual_type(text)
        has_formula = self._has_formula(text) or math_analysis.has_formula or vision_visual_type == "formula"
        cross_refs = self._extract_cross_refs(text)
        contextual_prefix = ""
        if math_analysis.has_formula and math_analysis.formula_ids:
            contextual_prefix = f"Formula terms: {'; '.join(math_analysis.formula_ids[:4])}"
        vision_prefix = self._vision_contextual_prefix(text, vision_visual_type)
        if vision_prefix:
            contextual_prefix = f"{contextual_prefix}\n{vision_prefix}".strip()
        content_type = vision_visual_type if vision_visual_type in {"formula", "table", "chart", "diagram"} else "text"
        caption = self._vision_caption(text, vision_visual_type)
        return Chunk(
            chunk_id=chunk_id, text=text,
            metadata=ChunkMetadata(
                section_id=section_id, section_type=section_type,
                page_start=page_start, page_end=page_end,
                chapter=chapter, doc_id=doc_id,
                has_formula=has_formula,
                formula_ids=math_analysis.formula_ids,
                cross_refs=cross_refs,
                content_type=content_type,
                caption=caption,
                contextual_prefix=contextual_prefix,
                content_source=content_source,
                enhanced=enhanced,
            ),
            layer="L2",
        )

    def _overlap_text(self, text: str) -> str:
        if self.overlap_tokens <= 0:
            return ""
        words = text.split()
        if not words:
            return ""
        return " ".join(words[-self.overlap_tokens:])

    def _fit_overlap_prefix(self, overlap_prefix: str, next_text: str) -> str:
        words = overlap_prefix.split()
        while words:
            candidate = " ".join(words)
            combined = f"{candidate}\n\n{next_text}".strip()
            if estimate_tokens(combined) <= self.max_chunk_tokens:
                return candidate
            words = words[1:]
        return ""

    def _split_word_windows(
        self,
        text: str,
        doc_id: str,
        section_id: str,
        section_type: str,
        chapter: str,
        start_idx: int,
        page_start: int | None = None,
        page_end: int | None = None,
        content_source: str = "parser",
        enhanced: bool = False,
    ) -> list[Chunk]:
        words = text.split()
        chunks: list[Chunk] = []
        idx = start_idx
        start = 0
        while start < len(words):
            end = start + 1
            best_end = end
            while end <= len(words):
                candidate = " ".join(words[start:end])
                if estimate_tokens(candidate) > self.max_chunk_tokens:
                    break
                best_end = end
                end += 1
            if best_end == start:
                best_end = min(len(words), start + 1)
            chunk_text = " ".join(words[start:best_end])
            chunks.append(
                self._make_chunk(
                    chunk_text,
                    doc_id,
                    section_id,
                    section_type,
                    chapter,
                    idx,
                    page_start=page_start,
                    page_end=page_end,
                    content_source=content_source,
                    enhanced=enhanced,
                )
            )
            idx += 1
            if best_end >= len(words):
                break
            overlap = min(self.overlap_tokens, max(0, best_end - start - 1))
            start = best_end - overlap
        return chunks

    def _can_split_by_words(self, text: str) -> bool:
        return len(text.split()) > 1

    def _max_chars_for_text(self, text: str) -> int:
        cjk = sum(1 for c in text if "一" <= c <= "鿿")
        chars_per_token = 4.0 - 2.0 * cjk / len(text) if text else 4.0
        return max(1, int(self.max_chunk_tokens * chars_per_token))

    def _char_step(self, max_chars: int) -> int:
        if self.overlap_tokens <= 0:
            return max_chars
        overlap_chars = min(max_chars - 1, self.overlap_tokens * 2)
        return max(1, max_chars - overlap_chars)

    def _has_formula(self, text: str) -> bool:
        return bool(_FORMULA_REGEX.search(text))

    def _vision_visual_type(self, text: str) -> str:
        match = re.search(r"^Visual type:\s*(formula|table|chart|diagram|mixed)\s*$", text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).lower() if match else ""

    def _vision_contextual_prefix(self, text: str, visual_type: str) -> str:
        if not visual_type:
            return ""
        summaries = []
        for label in ("Formula summary", "Table summary", "Chart summary", "QA hint"):
            value = self._vision_field(text, label)
            if value:
                summaries.append(f"{label}: {value}")
        if not summaries:
            return f"Vision {visual_type} content"
        return f"Vision {visual_type} summary: " + " | ".join(summaries[:3])

    def _vision_caption(self, text: str, visual_type: str) -> str:
        if visual_type == "table":
            return self._vision_field(text, "Table summary")
        if visual_type == "chart":
            return self._vision_field(text, "Chart summary")
        if visual_type == "formula":
            return self._vision_field(text, "Formula summary")
        return ""

    def _vision_field(self, text: str, label: str) -> str:
        match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _extract_cross_refs(self, text: str) -> list[str]:
        return _CROSS_REF_REGEX.findall(text)

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

    def _first_nonempty_line(self, text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return ""
