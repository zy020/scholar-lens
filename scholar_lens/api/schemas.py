from __future__ import annotations

from pydantic import BaseModel, Field


class ConfigUpdateRequest(BaseModel):
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str | None = None
    vision_api_key: str | None = None
    vision_model: str = "gpt-4o"


class ConfigResponse(BaseModel):
    llm_model: str
    embedding_model: str
    reranker_available: bool = False
    vision_available: bool = False
    status: str = "configured"


class DocumentUploadResponse(BaseModel):
    doc_id: str
    doc_type: str
    num_sections: int
    num_terms: int = 0
    status: str  # "processing" | "processed" | "error"
    error: str = ""


class ChatRequest(BaseModel):
    message: str
    doc_id: str = ""
    section_id: str = ""
    mode: str = "chat"  # chat | explain | translate


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
