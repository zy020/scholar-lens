from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from scholar_lens.agents.prompts import DOC_ANALYZER_SYSTEM, DOC_ANALYZER_STRUCTURE
from scholar_lens.agents.state import ScholarLensState
from scholar_lens.core.models import DocumentUnderstanding, Section, Term

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class DocumentAnalyzerAgent:
    """Document Analyzer Agent per spec Section 4.1."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

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
            result = await self._call_llm(doc_text)
            understanding = self._parse_result(result)
            state.doc_understanding = understanding
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            state.error = f"Analysis failed: {e}"

        state.current_step = "analyze"
        return state

    async def _call_llm(self, doc_text: str) -> str:
        prompt = DOC_ANALYZER_STRUCTURE.format(document_text=doc_text[:5000])
        response = await self._llm.ainvoke([
            SystemMessage(content=DOC_ANALYZER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content

    def _parse_result(self, llm_output: str) -> DocumentUnderstanding:
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = llm_output

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return DocumentUnderstanding(
                doc_type="general_document", language="en", difficulty="intermediate",
                estimated_reading_time=30, sections=[Section(section_id="1", title="Document", level=1)],
                mermaid_map="",
            )

        sections = [
            Section(
                section_id=s.get("section_id", s.get("id", "1")),
                title=s.get("title", ""), level=s.get("level", 1),
                page_start=s.get("page_start"), page_end=s.get("page_end"),
                section_type=s.get("section_type", "prose"), difficulty=s.get("difficulty", "intermediate"),
            )
            for s in data.get("sections", [])
        ]

        terms = [
            Term(english=t.get("english", ""), chinese=t.get("chinese", ""), relation_type=t.get("relation_type"))
            for t in data.get("key_terms", [])
        ]

        return DocumentUnderstanding(
            doc_type=data.get("doc_type", "general_document"),
            language=data.get("language", "en"),
            difficulty=data.get("difficulty", "intermediate"),
            estimated_reading_time=data.get("estimated_reading_time", 30),
            sections=sections, mermaid_map=data.get("mermaid_map", ""),
            key_terms=terms, prerequisites=data.get("prerequisites", []),
            l0_summaries=data.get("l0_summaries", {}),
            l1_overviews=data.get("l1_overviews", {}),
        )
