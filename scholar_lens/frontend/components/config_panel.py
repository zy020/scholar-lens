from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfigPanelState:
    """State for the model configuration panel.

    Per spec Section 7.4: LLM, Embedding, Reranker, Vision model configuration.
    """

    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3

    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"

    reranker_model: str = ""
    reranker_api_key: str = ""

    vision_api_key: str = ""
    vision_model: str = "gpt-4o"

    is_configured: bool = False

    def to_request_dict(self) -> dict:
        return {
            "llm_api_key": self.llm_api_key,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "embedding_api_key": self.embedding_api_key,
            "embedding_base_url": self.embedding_base_url,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model or None,
            "vision_api_key": self.vision_api_key or None,
            "vision_model": self.vision_model,
        }
