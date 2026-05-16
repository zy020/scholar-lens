from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from scholar_lens.core.models import (
    DocumentUnderstanding,
    ExplanationResult,
    ValidationResult,
    StudentProfile,
)


class AgentStep(str, Enum):
    ANALYZE = "analyze"
    EXPLAIN = "explain"
    VALIDATE = "validate"
    TUTOR = "tutor"


class ScholarLensState(BaseModel):
    """Shared state for all agents in the LangGraph pipeline."""

    doc_id: str = ""
    file_path: str = ""
    doc_understanding: DocumentUnderstanding | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)
    current_step: str = ""
    explanation_request: str = ""
    explanation_result: ExplanationResult | None = None
    validation_result: ValidationResult | None = None
    student_profile: StudentProfile = Field(default_factory=StudentProfile)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    section_id: str = ""
    session_id: str = ""
    error: str = ""

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
