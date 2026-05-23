from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---- Document Lifecycle ----

class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    parsing = "parsing"
    chunking = "chunking"
    summarizing = "summarizing"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"


class DocumentSummary(BaseModel):
    doc_id: str
    name: str
    status: DocumentStatus = DocumentStatus.uploaded
    doc_type: str = "unknown"
    text_quality: str = "unknown"
    ocr_needed: bool = False
    page_text_coverage: float = 0.0
    section_quality: str = "unknown"
    diagnostic_notes: list[str] = Field(default_factory=list)
    file_url: str = ""
    num_sections: int = 0
    num_chunks: int = 0
    error: str = ""


class DocumentDetail(DocumentSummary):
    sections: list[SectionSummary] = Field(default_factory=list)
    terms: list[dict] = Field(default_factory=list)


class SectionSummary(BaseModel):
    section_id: str
    title: str
    level: int = 1
    page_start: int | None = None
    page_end: int | None = None
    gist: str = ""


class EvidenceItem(BaseModel):
    doc_id: str
    section_id: str = ""
    page: int | None = None
    chunk_id: str
    quote: str
    score: float = 0.0


# ---- API Request/Response ----

class ConfigUpdateRequest(BaseModel):
    # UI-friendly aliases; backend normalizes them to the nested LLM config.
    api_key: str = ""
    base_url: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_use_separate: bool = False
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_use_separate: bool = False
    reranker_api_key: str = ""
    reranker_base_url: str = ""
    reranker_model: str | None = None
    reranker_enabled: bool = True
    reranker_use_separate: bool = False
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_model: str = ""
    vision_enabled: bool = True
    vision_use_separate: bool = False


class ConfigResponse(BaseModel):
    llm_model: str
    embedding_model: str
    reranker_available: bool = False
    vision_available: bool = False
    status: str = "configured"
    requires_restart: bool = False


class DocumentUploadResponse(BaseModel):
    doc_id: str
    doc_type: str
    num_sections: int
    num_terms: int = 0
    status: str  # "processing" | "processed" | "error"
    error: str = ""


class ExplanationRequest(BaseModel):
    message: str
    doc_id: str = ""
    section_id: str = ""
    context: str = ""


class ChatRequest(BaseModel):
    message: str
    doc_id: str = ""
    section_id: str = ""
    mode: str = "chat"  # chat | explain | translate | socratic | review


class ChatMessage(BaseModel):
    role: str  # user | assistant | system
    content: str
    timestamp: str = ""


class ExplanationResponse(BaseModel):
    original: str
    translation: str
    explanation: str
    related_terms: list[dict] = Field(default_factory=list)
    difficulty_level: str = "intermediate"
    source_section: str = ""
    confidence: str = "medium"


class NotesResponse(BaseModel):
    doc_id: str
    terms: list[dict] = Field(default_factory=list)
    reading_progress: dict = Field(default_factory=dict)
    concept_map: str = ""


# ---- Paper Brief Schemas ----

class BriefEvidence(BaseModel):
    section_id: str = ""
    section_title: str = ""
    quote: str = ""


class BriefContribution(BaseModel):
    claim: str
    why_it_matters: str = ""
    evidence: BriefEvidence | None = None


class BriefMethodStep(BaseModel):
    title: str
    explanation: str
    evidence: BriefEvidence | None = None


class BriefTerm(BaseModel):
    term: str
    explanation_zh: str
    keep_english: bool = True


class BriefReadingFocus(BaseModel):
    section_id: str
    section_title: str
    reason: str


class BriefReviewQuestion(BaseModel):
    question: str
    level: str = "basic"  # basic | deep | critical
    expected_answer_hint: str = ""


class PaperBriefResponse(BaseModel):
    doc_id: str
    title: str
    source: str = "fallback"  # fallback | llm | cached
    tldr: list[str] = Field(default_factory=list)
    problem: str = ""
    motivation: str = ""
    contributions: list[BriefContribution] = Field(default_factory=list)
    method_walkthrough: list[BriefMethodStep] = Field(default_factory=list)
    key_terms: list[BriefTerm] = Field(default_factory=list)
    reading_focus: list[BriefReadingFocus] = Field(default_factory=list)
    review_questions: list[BriefReviewQuestion] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    generated_at: str = ""
    error: str = ""
