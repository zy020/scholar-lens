from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from scholar_lens.core.settings import EmbeddingConfig, LLMConfig

if TYPE_CHECKING:
    from scholar_lens.core.settings import Settings


class ChatLLMFactory:
    """Creates ChatOpenAI instances from LLMConfig.

    Known limitation: the returned ChatOpenAI reuses httpx connections internally.
    Under high concurrency with an unreliable API provider, TCP send-buffer
    deadlocks can accumulate in the connection pool (see eval script debugging).
    For production resilience, consider either:
    - Passing http_client=httpx.Client(timeout=...) with max_keepalive_connections=0
    - Or recreating the client after circuit breaker recovery events
    Risk is minimal with low concurrency (<10 req/s).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @classmethod
    def from_settings(cls, settings: Settings) -> ChatLLMFactory:
        return cls(settings.llm)

    def create(self, config: LLMConfig | None = None, streaming: bool = True) -> ChatOpenAI:
        cfg = config or self._config
        return ChatOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            streaming=streaming,
            request_timeout=cfg.request_timeout,
        )

    # Backward-compatible alias
    create_chat_llm = create


class EmbeddingFactory:
    """Creates OpenAIEmbeddings instances from EmbeddingConfig."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingFactory:
        return cls(settings.embedding)

    def create(self, config: EmbeddingConfig | None = None) -> OpenAIEmbeddings:
        cfg = config or self._config
        return OpenAIEmbeddings(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            dimensions=cfg.dimension,
            request_timeout=cfg.request_timeout,
        )

    # Backward-compatible alias
    create_embeddings = create


# Backward-compatible alias
LLMFactory = ChatLLMFactory
