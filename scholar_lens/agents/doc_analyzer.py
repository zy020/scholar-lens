from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from scholar_lens.agents.prompts import (
    DOC_ANALYZER_L0,
    DOC_ANALYZER_L1,
    DOC_ANALYZER_STRUCTURE,
    DOC_ANALYZER_SYSTEM,
    DOC_ANALYZER_TERMS,
)
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.circuit_breaker import CircuitBreaker
from scholar_lens.core.exceptions import CircuitOpenError
from scholar_lens.core.models import DocumentUnderstanding, Section, Term
from scholar_lens.core.utils import extract_json_from_llm_output

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from scholar_lens.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class DocumentAnalyzerAgent:
    """Document Analyzer Agent per spec Section 4.1.

    Two APIs:
    - analyze(state) — LangGraph pipeline node (reads from state.messages)
    - analyze_document(text, sections) — direct call (for upload pre-processing)
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm
        self._circuit_breaker = CircuitBreaker(name="llm-analyzer")

    # ===== LangGraph pipeline API (backward compatible) =====

    async def analyze(self, state: ScholarLensState) -> ScholarLensState:
        if not self._llm:
            state.error = "No LLM configured for Document Analyzer"
            state.current_step = "analyze"
            return state

        doc_text = ""
        for msg in state.messages:
            if "Document text:" in msg.get("content", ""):
                doc_text = msg["content"]
                break

        if not doc_text:
            state.error = "No document text found in state"
            state.current_step = "analyze"
            return state

        try:
            understanding = await self.analyze_document(doc_text, [])
            state.doc_understanding = understanding
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            state.error = f"Analysis failed: {e}"

        state.current_step = "analyze"
        return state

    # ===== Direct API (for upload pre-processing) =====

    async def analyze_document(
        self,
        doc_text: str,
        sections: list[dict] | None = None,
        memory_manager: MemoryManager | None = None,
        max_concurrent: int = 5,
    ) -> DocumentUnderstanding:
        """Analyze a document and return a complete DocumentUnderstanding.

        Args:
            doc_text: Full document text (first 8000 chars used for structure).
            sections: Optional pre-extracted sections from the parser.
            memory_manager: Optional MemoryManager for storing glossary terms.
            max_concurrent: Max parallel LLM calls for L0/L1 generation.
        """
        if not self._llm:
            raise RuntimeError("No LLM configured for Document Analyzer")

        # Stage 1: Structure + terms + mermaid (1 LLM call)
        result = await self._call_llm(doc_text[:8000])
        understanding = self._parse_result(result)

        # Stage 2: If we have pre-extracted sections, use them to override
        if sections:
            understanding.sections = self._merge_sections(understanding.sections, sections)

        # Stage 3: L0/L1 summaries (parallel across sections)
        if understanding.sections:
            understanding = await self._generate_layered_summaries(
                doc_text, understanding, max_concurrent,
            )

        # Stage 4: Glossary terms (best-effort — failure must not discard understanding)
        if memory_manager:
            try:
                await self._extract_and_store_terms(
                    doc_text[:5000], understanding, memory_manager,
                )
            except Exception:
                logger.warning("Term extraction failed, continuing with partial analysis", exc_info=True)

        return understanding

    async def _generate_layered_summaries(
        self,
        doc_text: str,
        understanding: DocumentUnderstanding,
        max_concurrent: int,
    ) -> DocumentUnderstanding:
        """Generate L0 + L1 summaries for each section in parallel."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _gen_l0(section: Section, text_snippet: str) -> tuple[str, str]:
            async with sem:
                l0 = await self._call_single_prompt(
                    DOC_ANALYZER_L0.format(title=section.title, text=text_snippet[:2000]),
                )
                l1 = await self._call_single_prompt(
                    DOC_ANALYZER_L1.format(title=section.title, text=text_snippet[:5000]),
                )
                return section.section_id, l0.strip(), l1.strip()

        def _section_text(sec: Section) -> str:
            """Extract text around section position. Uses page_start (~3000 chars/page)
            for rough positioning, falls back to title search, then doc prefix."""
            if sec.page_start is not None:
                pos = sec.page_start * 3000
                start = max(0, pos - 500)
                end = min(len(doc_text), pos + 3500)
                return doc_text[start:end]
            idx = doc_text.find(sec.title)
            if idx >= 0:
                start = max(0, idx - 200)
                end = min(len(doc_text), idx + 3000)
                return doc_text[start:end]
            return doc_text[:3000]

        tasks = []
        for sec in understanding.sections[:10]:  # max 10 sections to limit cost
            tasks.append(_gen_l0(sec, _section_text(sec)))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for item in results:
                if isinstance(item, BaseException):
                    logger.warning(f"Section summary failed: {item}")
                    continue
                sec_id, l0, l1 = item
                if l0:
                    understanding.l0_summaries[sec_id] = l0
                if l1:
                    understanding.l1_overviews[sec_id] = l1

        return understanding

    async def _extract_and_store_terms(
        self,
        doc_text: str,
        understanding: DocumentUnderstanding,
        memory_manager: MemoryManager,
    ) -> None:
        """Extract bilingual terms and store in CoreMemory."""
        result = await self._call_single_prompt(
            DOC_ANALYZER_TERMS.format(text=doc_text[:4000]),
        )
        data = extract_json_from_llm_output(result)
        if isinstance(data, list):
            for item in data[:20]:
                en = item.get("english", "")
                zh = item.get("chinese", "")
                if en and zh:
                    memory_manager.core_memory.add_glossary_entry(en, zh)
                    understanding.key_terms.append(Term(english=en, chinese=zh))

    def _merge_sections(
        self, llm_sections: list[Section], parsed_sections: list[dict],
    ) -> list[Section]:
        """Merge LLM-detected sections with parser-extracted sections."""
        if not parsed_sections:
            return llm_sections
        if not llm_sections:
            return [
                Section(
                    section_id=s.get("id", str(i)), title=s.get("title", ""),
                    level=s.get("level", 1), page_start=s.get("page_start"),
                    page_end=s.get("page_end"),
                    section_type="prose", chapter=s.get("chapter", ""),
                )
                for i, s in enumerate(parsed_sections)
            ]
        return llm_sections

    # ===== Internal helpers =====

    async def _call_llm(self, doc_text: str) -> str:
        if not await self._circuit_breaker.allow_request():
            raise CircuitOpenError("llm-analyzer", self._circuit_breaker)
        try:
            prompt = DOC_ANALYZER_STRUCTURE.format(document_text=doc_text)
            response = await asyncio.wait_for(
                self._llm.ainvoke([
                    SystemMessage(content=DOC_ANALYZER_SYSTEM),
                    HumanMessage(content=prompt),
                ]), timeout=120,
            )
            await self._circuit_breaker.record_success()
            return response.content
        except CircuitOpenError:
            raise
        except Exception:
            await self._circuit_breaker.record_failure()
            raise

    async def _call_single_prompt(self, prompt_text: str) -> str:
        """Call LLM with a simple prompt, return text content. Circuit-breaker protected."""
        if not await self._circuit_breaker.allow_request():
            raise CircuitOpenError("llm-analyzer", self._circuit_breaker)
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke([HumanMessage(content=prompt_text)]), timeout=60,
            )
            await self._circuit_breaker.record_success()
            return response.content
        except CircuitOpenError:
            raise
        except Exception as e:
            await self._circuit_breaker.record_failure()
            logger.warning(f"LLM prompt failed: {e}")
            raise

    def _parse_result(self, llm_output: str) -> DocumentUnderstanding:
        data = extract_json_from_llm_output(llm_output)
        if not data:
            raise ValueError("Document Analyzer returned non-JSON output")

        sections = [
            Section(
                section_id=s.get("section_id", s.get("id", str(i))),
                title=s.get("title", ""), level=s.get("level", 1),
                page_start=s.get("page_start"), page_end=s.get("page_end"),
                section_type=s.get("section_type", "prose"),
                difficulty=s.get("difficulty", "intermediate"),
            )
            for i, s in enumerate(data.get("sections", []))
        ]

        terms = [
            Term(
                english=t.get("english", ""), chinese=t.get("chinese", ""),
                relation_type=t.get("relation_type"),
            )
            for t in data.get("key_terms", [])
        ]

        return DocumentUnderstanding(
            doc_type=data.get("doc_type", "research_paper"),
            language=data.get("language", "en"),
            difficulty=data.get("difficulty", "intermediate"),
            estimated_reading_time=data.get("estimated_reading_time", 30),
            sections=sections, mermaid_map=data.get("mermaid_map", ""),
            key_terms=terms, prerequisites=data.get("prerequisites", []),
            l0_summaries=data.get("l0_summaries", {}),
            l1_overviews=data.get("l1_overviews", {}),
        )
