from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from scholar_lens.core.circuit_breaker import CircuitBreaker
from scholar_lens.parsers.models import Chunk

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_CONTEXTUAL_PROMPT = """You are an expert at document analysis. Given the overall document context and a specific chunk from that document, write a brief context prefix (1-2 sentences) that explains where this chunk fits within the broader document. This prefix will be prepended to the chunk to improve retrieval accuracy.

Document context: {document_context}

Chunk content: {chunk_text}

Context prefix:"""


class ContextualRetriever:
    """Generates contextual prefixes for chunks per Anthropic's Contextual Retrieval approach."""

    def __init__(self, llm: BaseChatModel | None = None, document_context: str = "", max_concurrent: int = 5) -> None:
        self._llm = llm
        self._document_context = document_context
        self._max_concurrent = max_concurrent
        self._circuit_breaker = CircuitBreaker(name="llm-contextual-retriever")

    async def generate_prefixes(self, chunks: list[Chunk]) -> list[Chunk]:
        if not self._llm:
            logger.info("No LLM configured, skipping contextual prefix generation")
            return chunks
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _process(chunk: Chunk) -> Chunk:
            async with sem:
                prefix = await self._generate_prefix(chunk.text)
            new_metadata = chunk.metadata.model_copy(update={"contextual_prefix": prefix})
            return chunk.model_copy(update={"metadata": new_metadata})

        results = await asyncio.gather(*[_process(c) for c in chunks], return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def _generate_prefix(self, chunk_text: str) -> str:
        if not await self._circuit_breaker.allow_request():
            return ""
        try:
            from langchain_core.messages import HumanMessage
            prompt = _CONTEXTUAL_PROMPT.format(document_context=self._document_context[:2000], chunk_text=chunk_text[:1000])
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            await self._circuit_breaker.record_success()
            return response.content.strip()
        except Exception as e:
            await self._circuit_breaker.record_failure()
            logger.warning(f"Contextual prefix generation failed: {e}")
            return ""
