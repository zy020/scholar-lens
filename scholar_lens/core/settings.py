from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-small"
    dimension: int = 1536


class RerankerConfig(BaseModel):
    model: str = "bge-reranker-m3"
    base_url: str | None = None
    api_key: str | None = None


class VisionConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # Embedding
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Optional: Reranker
    reranker_model: str | None = None
    reranker_base_url: str | None = None
    reranker_api_key: str | None = None

    # Optional: Vision
    vision_api_key: str | None = None
    vision_base_url: str = "https://api.openai.com/v1"
    vision_model: str = "gpt-4o"

    # Paths
    data_dir: Path = Path(os.getcwd()) / "data"
    knowledge_dir: Path = Path(os.getcwd()) / "knowledge"

    @property
    def llm(self) -> LLMConfig:
        return LLMConfig(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
            model=self.embedding_model,
            dimension=self.embedding_dimension,
        )

    @property
    def reranker(self) -> RerankerConfig | None:
        if self.reranker_model is None:
            return None
        return RerankerConfig(
            model=self.reranker_model,
            base_url=self.reranker_base_url,
            api_key=self.reranker_api_key,
        )

    @property
    def vision(self) -> VisionConfig | None:
        if self.vision_api_key is None:
            return None
        return VisionConfig(
            api_key=self.vision_api_key,
            base_url=self.vision_base_url,
            model=self.vision_model,
        )
