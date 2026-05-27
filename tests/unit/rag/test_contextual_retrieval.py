import asyncio
from unittest.mock import AsyncMock, MagicMock
from scholar_lens.rag.contextual_retrieval import ContextualRetriever
from scholar_lens.parsers.models import Chunk, ChunkMetadata


class TestContextualRetriever:
    def test_instantiation(self):
        cr = ContextualRetriever()
        assert cr is not None

    def test_generate_contextual_prefixes(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This chunk discusses the self-attention mechanism in the Transformer architecture."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        cr = ContextualRetriever(llm=mock_llm, document_context="Paper about Transformers")
        chunks = [Chunk(chunk_id="1", text="The self-attention mechanism computes...", metadata=ChunkMetadata(section_id="3", section_type="method"), layer="L2")]
        result = asyncio.run(cr.generate_prefixes(chunks))
        assert len(result) == 1
        assert result[0].metadata.contextual_prefix != ""

    def test_no_llm_skips_prefix(self):
        cr = ContextualRetriever(llm=None, document_context="")
        chunks = [Chunk(chunk_id="1", text="Hello", metadata=ChunkMetadata(section_id="1", section_type="prose"), layer="L2")]
        result = asyncio.run(cr.generate_prefixes(chunks))
        assert result[0].metadata.contextual_prefix == ""
