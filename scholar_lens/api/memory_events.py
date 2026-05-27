from __future__ import annotations

import logging
from typing import Any

from scholar_lens.api.chat_service import configured_llm_configs
from scholar_lens.api.deps import get_settings
from scholar_lens.core.llm_factory import ChatLLMFactory

logger = logging.getLogger(__name__)


def build_memory_summary_llm(settings):
    if not bool(getattr(settings, "memory_llm_compression_enabled", False)):
        return None
    configs = configured_llm_configs(settings)
    if not configs:
        return None
    try:
        return ChatLLMFactory.from_settings(settings).create(config=configs[0], streaming=False)
    except Exception:
        logger.warning("Memory summary LLM unavailable", exc_info=True)
        return None


async def record_memory_event(
    memory,
    event_type: str,
    *,
    doc_id: str = "",
    section_id: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    try:
        settings = get_settings()
        await memory.record_event(
            event_type,
            doc_id=doc_id,
            section_id=section_id,
            payload=payload or {},
            summary_llm=build_memory_summary_llm(settings),
        )
    except Exception:
        logger.warning("Memory event recording failed", exc_info=True)
