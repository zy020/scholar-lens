import pytest
from scholar_lens.core.exceptions import (
    ScholarLensError,
    ParsingError,
    LLMError,
    LLMRateLimitError,
    RetrievalError,
    ValidationError as AppValidationError,
    CircuitOpenError,
)


class TestExceptionHierarchy:
    def test_base_error(self):
        e = ScholarLensError("something went wrong")
        assert str(e) == "something went wrong"

    def test_parsing_error_is_base(self):
        assert issubclass(ParsingError, ScholarLensError)

    def test_llm_error_is_base(self):
        assert issubclass(LLMError, ScholarLensError)

    def test_rate_limit_is_llm_error(self):
        assert issubclass(LLMRateLimitError, LLMError)

    def test_retrieval_error_is_base(self):
        assert issubclass(RetrievalError, ScholarLensError)

    def test_validation_error_is_base(self):
        assert issubclass(AppValidationError, ScholarLensError)

    def test_circuit_open_error(self):
        cb = type("CB", (), {"name": "vlm"})()
        e = CircuitOpenError("vlm", cb)
        assert "vlm" in str(e)
