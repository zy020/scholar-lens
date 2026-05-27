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
    ocr_recommended_pages: list[int] = Field(default_factory=list)
    ocr_recommendation_reasons: dict[str, str] = Field(default_factory=dict)
    page_text_coverage: float = 0.0
    section_quality: str = "unknown"
    diagnostic_notes: list[str] = Field(default_factory=list)
    index_status: str = "not_indexed"  # not_indexed | vector | keyword_only | failed
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
    api_key: str | None = None
    base_url: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    llm_use_separate: bool | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_use_separate: bool | None = None
    reranker_api_key: str | None = None
    reranker_base_url: str | None = None
    reranker_model: str | None = None
    reranker_enabled: bool | None = None
    reranker_use_separate: bool | None = None
    vision_api_key: str | None = None
    vision_base_url: str | None = None
    vision_model: str | None = None
    vision_enabled: bool | None = None
    vision_use_separate: bool | None = None
    llm_quality_enabled: bool | None = None
    vision_enhancement_enabled: bool | None = None
    memory_llm_compression_enabled: bool | None = None


class ConfigResponse(BaseModel):
    llm_model: str
    llm_base_url: str = ""
    llm_configured: bool = False
    embedding_model: str
    embedding_base_url: str = ""
    embedding_configured: bool = False
    reranker_available: bool = False
    reranker_model: str = ""
    reranker_base_url: str = ""
    reranker_active: bool = False
    reranker_mode: str = "rule"
    reranker_use_separate: bool = False
    vision_available: bool = False
    vision_model: str = ""
    vision_base_url: str = ""
    vision_use_separate: bool = False
    auto_ocr_enabled: bool = True
    llm_quality_enabled: bool = False
    vision_enhancement_enabled: bool = False
    memory_llm_compression_enabled: bool = False
    status: str = "configured"
    requires_restart: bool = False


class DocumentAnalysisResponse(BaseModel):
    doc_id: str
    status: str
    source: str = "unavailable"
    error: str = ""


class DocumentAnalysisDetailResponse(BaseModel):
    doc_id: str
    status: str = "missing"
    source: str = "missing"
    updated_at: str = ""
    error: str = ""
    difficulty: str = ""
    estimated_reading_time: int = 0
    key_terms: list[dict] = Field(default_factory=list)
    l0_summaries: dict[str, str] = Field(default_factory=dict)
    l1_overviews: dict[str, str] = Field(default_factory=dict)
    mermaid_map: str = ""
    parse_quality_status: str = ""
    parse_quality_message: str = ""
    parse_quality_warnings: list[str] = Field(default_factory=list)
    parse_quality_actions: list[str] = Field(default_factory=list)
    parse_quality_pages: list[dict] = Field(default_factory=list)


class EnhancePlanResponse(BaseModel):
    doc_id: str
    status: str = "skipped"
    recommended_ocr_pages: list[int] = Field(default_factory=list)
    ocr_recommendation_reasons: dict[str, str] = Field(default_factory=dict)
    estimated_ocr_pages: int = 0
    ocr_engine: str = "rapidocr"
    ocr_installed: bool = False
    ocr_gpu_available: bool = False
    ocr_cpu_available: bool = False
    ocr_recommended_mode: str = "unavailable"
    available_actions: list[str] = Field(default_factory=list)
    vision_available: bool = False
    vision_enhancement_enabled: bool = False
    vision_possible: bool = False
    vision_escalation_reasons: list[str] = Field(default_factory=list)
    page_decisions: list[dict] = Field(default_factory=list)
    message: str = ""


class OCREnhanceResponse(BaseModel):
    doc_id: str
    status: str = "skipped"
    engine: str = "rapidocr"
    pages: list[dict] = Field(default_factory=list)
    vision_recommended_pages: list[int] = Field(default_factory=list)
    message: str = ""
    error: str = ""


class VisionEnhanceResponse(BaseModel):
    doc_id: str
    status: str = "skipped"
    engine: str = "vision"
    pages: list[dict] = Field(default_factory=list)
    message: str = ""
    error: str = ""


class EnhancementApplyResponse(BaseModel):
    doc_id: str
    status: str = "missing"
    source: str = "ocr"
    num_pages_updated: int = 0
    num_chunks: int = 0
    message: str = ""
    error: str = ""


class ParseQualityResponse(BaseModel):
    doc_id: str
    source: str = "heuristic"
    status: str = "available"
    qualities: list[dict] = Field(default_factory=list)
    message: str = ""
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
    top_k: int = Field(default=5, ge=1, le=20)
    context_k: int | None = Field(default=None, ge=1, le=40)
    section_only: bool = False
    use_reranker: bool = True
    student_level: str = "intermediate"
    deep_mode: bool = False


class ChatMessage(BaseModel):
    role: str  # user | assistant | system
    content: str
    timestamp: str = ""
    evidence: list[dict] = Field(default_factory=list)


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
    source: str = "not_generated"  # not_generated | llm | cached | unavailable
    brief_type: str = "paper"  # paper | lecture
    text_quality: str = "unknown"
    ocr_needed: bool = False
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
