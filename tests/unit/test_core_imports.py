import pytest


def test_all_core_imports():
    from scholar_lens.core import (
        CitationContext,
        CircuitBreaker,
        CircuitOpenError,
        CircuitState,
        DocumentUnderstanding,
        ExplanationRequest,
        ExplanationResult,
        LLMError,
        LLMRateLimitError,
        ParsingError,
        ReadingRecord,
        Reference,
        RetrievalError,
        ScholarLensError,
        Section,
        Settings,
        StudentProfile,
        Term,
        TokenTracker,
        TokenUsage,
        ValidationError,
        ValidationResult,
    )
    assert True
