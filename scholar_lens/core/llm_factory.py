from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from scholar_lens.core.settings import EmbeddingConfig, LLMConfig

if TYPE_CHECKING:
    from scholar_lens.core.settings import Settings


class LLMFactory:
    """Creates LLM and embedding instances from configuration."""

    def __init__(self, config: LLMConfig | EmbeddingConfig):
        self._config = config

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMFactory:
        return cls(settings.llm)

    def create_chat_llm(
        self,
        config: LLMConfig | None = None,
        streaming: bool = True,
    ) -> ChatOpenAI:
        cfg = config or self._config
        if not isinstance(cfg, LLMConfig):
            raise TypeError("create_chat_llm requires LLMConfig")
        return ChatOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            streaming=streaming,
        )

    def create_embeddings(
        self,
        config: EmbeddingConfig | None = None,
    ) -> OpenAIEmbeddings:
        cfg = config or self._config
        if not isinstance(cfg, EmbeddingConfig):
            raise TypeError("create_embeddings requires EmbeddingConfig")
        return OpenAIEmbeddings(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            model=cfg.model,
            dimensions=cfg.dimension,
        )
