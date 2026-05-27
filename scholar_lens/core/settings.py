from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    request_timeout: float = 60.0


class EmbeddingConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    dimension: int = 1536
    request_timeout: float = 30.0


class RerankerConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""


class VisionConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""


class Settings(BaseSettings):
    """ScholarLens settings with shared credentials.

    By default, all four models share the same api_key and base_url.
    Individual overrides: LLM__API_KEY, EMBEDDING__MODEL, RERANKER__BASE_URL, etc.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="allow",
    )

    # Shared credentials — all models inherit from these
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"

    # Per-model overrides (model name required; key/url optional, inherit from shared)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig | None = None
    reranker_use_separate: bool = False
    vision: VisionConfig | None = None
    vision_use_separate: bool = False
    backup_llm: LLMConfig | None = None  # Batch 6.3: cheaper fallback model
    auto_ocr_enabled: bool = True
    llm_quality_enabled: bool = False
    vision_enhancement_enabled: bool = False
    memory_llm_compression_enabled: bool = False

    # Paths
    data_dir: Path = Field(default_factory=lambda: Path(os.getcwd()) / "data")
    knowledge_dir: Path = Field(default_factory=lambda: Path(os.getcwd()) / "knowledge")

    @model_validator(mode="after")
    def _inherit_and_validate(self) -> Settings:
        """Inherit shared credentials + validate required config."""
        # Inherit
        for target in [self.llm, self.embedding]:
            if not target.api_key:
                target.api_key = self.api_key
            if not target.base_url:
                target.base_url = self.base_url
        if self.reranker is not None:
            if not self.reranker_use_separate and not self.reranker.api_key:
                self.reranker.api_key = self.api_key
            if not self.reranker_use_separate and not self.reranker.base_url:
                self.reranker.base_url = self.base_url
        if self.vision is not None:
            if not self.vision_use_separate and not self.vision.api_key:
                self.vision.api_key = self.api_key
            if not self.vision_use_separate and not self.vision.base_url:
                self.vision.base_url = self.base_url
        if self.backup_llm is not None:
            if not self.backup_llm.api_key:
                self.backup_llm.api_key = self.api_key
            if not self.backup_llm.base_url:
                self.backup_llm.base_url = self.base_url

        # P2.2: startup validation warnings
        import logging
        logger = logging.getLogger(__name__)
        if not self.llm.api_key:
            logger.warning(
                "LLM API key not configured. Set API_KEY in .env or "
                "LLM__API_KEY for per-model override."
            )
        if not self.llm.model:
            logger.warning("LLM model not configured. Set LLM__MODEL in .env.")

        return self

    # ===== Backward-compatible flat accessors =====

    @property
    def llm_api_key(self) -> str:
        return self.llm.api_key

    @property
    def llm_base_url(self) -> str:
        return self.llm.base_url

    @property
    def llm_model(self) -> str:
        return self.llm.model

    @property
    def llm_temperature(self) -> float:
        return self.llm.temperature

    @property
    def llm_max_tokens(self) -> int:
        return self.llm.max_tokens

    @property
    def embedding_api_key(self) -> str:
        return self.embedding.api_key

    @property
    def embedding_base_url(self) -> str:
        return self.embedding.base_url

    @property
    def embedding_model(self) -> str:
        return self.embedding.model

    @property
    def embedding_dimension(self) -> int:
        return self.embedding.dimension

    @property
    def reranker_model(self) -> str | None:
        return self.reranker.model if self.reranker else None

    @property
    def reranker_base_url(self) -> str | None:
        return self.reranker.base_url if self.reranker else None

    @property
    def reranker_api_key(self) -> str | None:
        return self.reranker.api_key if self.reranker else None

    @property
    def vision_api_key(self) -> str | None:
        return self.vision.api_key if self.vision else None

    @property
    def vision_base_url(self) -> str:
        return self.vision.base_url if self.vision else ""

    @property
    def vision_model(self) -> str:
        return self.vision.model if self.vision else ""
