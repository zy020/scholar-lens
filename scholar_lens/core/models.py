from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Section(BaseModel):
    section_id: str
    title: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    section_type: str = "prose"  # prose | method | results | references | citation_context
    difficulty: str = "intermediate"  # beginner | intermediate | advanced
    children: list[Section] = Field(default_factory=list)


class Term(BaseModel):
    english: str
    chinese: str
    definition_en: str | None = None
    definition_zh: str | None = None
    relation_type: str | None = None  # Used-for | Hyponym-of | Part-of | Compare-with


class Reference(BaseModel):
    ref_id: str
    authors: list[str] = Field(default_factory=list)
    title: str = ""
    year: int | None = None
    venue: str | None = None
    doi: str | None = None


class CitationContext(BaseModel):
    ref_id: str
    in_text: str
    surrounding_text: str
    section_id: str


class DocumentUnderstanding(BaseModel):
    doc_type: str  # research_paper | courseware | textbook_chapter
    language: str  # en | zh | mixed
    difficulty: str  # beginner | intermediate | advanced
    estimated_reading_time: int  # minutes
    sections: list[Section]
    mermaid_map: str
    key_terms: list[Term] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    l0_summaries: dict[str, str] = Field(default_factory=dict)
    l1_overviews: dict[str, str] = Field(default_factory=dict)
    references: list[Reference] = Field(default_factory=list)
    citation_contexts: list[CitationContext] = Field(default_factory=list)


class ExplanationRequest(BaseModel):
    section_id: str
    mode: str  # translate | explain | term_lookup | sentence_breakdown
    target_text: str | None = None
    student_level: str = "intermediate"
    previous_explanations: list[str] = Field(default_factory=list)


class ExplanationResult(BaseModel):
    original: str
    translation: str
    explanation: str
    related_terms: list[Term] = Field(default_factory=list)
    difficulty_level: str = "intermediate"
    source_section: str = ""
    confidence: str = "high"  # high | medium | low | unverified


class ValidationResult(BaseModel):
    passed: bool
    confidence: str  # high | medium | low
    issues: list[str] = Field(default_factory=list)
    correction: str | None = None


class StudentProfile(BaseModel):
    level: str = "intermediate"
    native_language: str = "zh"
    target_language: str = "en"
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    total_sessions: int = 0


class ReadingRecord(BaseModel):
    doc_id: str
    section_id: str
    comprehension_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
