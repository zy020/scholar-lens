from scholar_lens.core.models import (
    CitationContext,
    DocumentUnderstanding,
    ExplanationRequest,
    ExplanationResult,
    ReadingRecord,
    Reference,
    Section,
    StudentProfile,
    Term,
    ValidationResult,
)
from scholar_lens.core.settings import Settings
from scholar_lens.core.circuit_breaker import CircuitBreaker, CircuitState
from scholar_lens.core.token_tracker import TokenTracker, TokenUsage
from scholar_lens.core.utils import extract_json_from_llm_output
from scholar_lens.core.exceptions import (
    ScholarLensError,
    ParsingError,
    LLMError,
    LLMRateLimitError,
    RetrievalError,
    ValidationError,
    CircuitOpenError,
)

__all__ = [
    "ChatLLMFactory",
    "CitationContext",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "DocumentUnderstanding",
    "EmbeddingFactory",
    "ExplanationRequest",
    "ExplanationResult",
    "LLMError",
    "LLMFactory",
    "LLMRateLimitError",
    "ParsingError",
    "ReadingRecord",
    "Reference",
    "RetrievalError",
    "ScholarLensError",
    "Section",
    "Settings",
    "StudentProfile",
    "Term",
    "TokenTracker",
    "TokenUsage",
    "ValidationError",
    "ValidationResult",
    "extract_json_from_llm_output",
]


def __getattr__(name: str):
    if name in {"ChatLLMFactory", "EmbeddingFactory", "LLMFactory"}:
        from scholar_lens.core.llm_factory import ChatLLMFactory, EmbeddingFactory, LLMFactory

        return {
            "ChatLLMFactory": ChatLLMFactory,
            "EmbeddingFactory": EmbeddingFactory,
            "LLMFactory": LLMFactory,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
