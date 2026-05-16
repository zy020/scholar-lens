from __future__ import annotations


class ScholarLensError(Exception):
    """Base exception for all ScholarLens errors."""


class ParsingError(ScholarLensError):
    """Error during document parsing."""


class LLMError(ScholarLensError):
    """Error from LLM invocation."""


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""


class RetrievalError(ScholarLensError):
    """Error during RAG retrieval."""


class ValidationError(ScholarLensError):
    """Error during content validation."""


class CircuitOpenError(ScholarLensError):
    """Circuit breaker is open, rejecting requests."""

    def __init__(self, service_name: str, circuit_breaker: object):
        self.service_name = service_name
        self.circuit_breaker = circuit_breaker
        super().__init__(f"Circuit breaker open for service: {service_name}")
