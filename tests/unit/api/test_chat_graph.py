from scholar_lens.api.chat_graph import run_chat_graph, run_chat_retrieval_graph, run_chat_validation_graph
from scholar_lens.api.schemas import ChatRequest
from scholar_lens.parsers.models import Chunk, ChunkMetadata
from scholar_lens.rag.document_store import DocumentStore


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        text = "\n".join(str(getattr(message, "content", message)) for message in messages)
        self.calls.append(text)
        if "Return JSON only" in text:
            return FakeResponse('{"passed": false, "issues": ["needs clearer grounding"], "correction": "修正后的回答"}')
        return FakeResponse("初始回答")


def _store_with_chunk(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="c1",
            text="Transformer uses self-attention to connect tokens.",
            metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
        ),
    ])
    return store, doc


def _store_with_formula_chunk(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="formula",
            text="Scaled dot-product attention divides QK by sqrt(d_k).",
            metadata=ChunkMetadata(
                section_id="slide_2",
                section_type="slide",
                content_type="slide",
                page_start=2,
                doc_id=doc.doc_id,
                has_formula=True,
            ),
        ),
    ])
    return store, doc


async def test_chat_graph_light_mode_reuses_rag_without_validator(tmp_path):
    store, doc = _store_with_chunk(tmp_path)
    llm = FakeLLM()

    result = await run_chat_graph(
        store=store,
        request=ChatRequest(message="What is self-attention?", doc_id=doc.doc_id),
        llm=llm,
    )

    assert result.content == "初始回答"
    assert result.evidence[0]["chunk_id"] == "c1"
    assert result.validation is None
    assert len(llm.calls) == 1


async def test_chat_retrieval_graph_prepares_context_without_answer_llm(tmp_path):
    store, doc = _store_with_chunk(tmp_path)

    result = await run_chat_retrieval_graph(
        store=store,
        request=ChatRequest(message="What is self-attention?", doc_id=doc.doc_id),
    )

    assert result.intent == "concept"
    assert result.evidence[0]["chunk_id"] == "c1"
    assert "Transformer uses self-attention" in result.context.context_text


async def test_chat_graph_deep_mode_validates_and_applies_correction(tmp_path):
    store, doc = _store_with_chunk(tmp_path)
    llm = FakeLLM()

    result = await run_chat_graph(
        store=store,
        request=ChatRequest(message="What is self-attention?", doc_id=doc.doc_id, deep_mode=True),
        llm=llm,
    )

    assert result.content == "修正后的回答"
    assert result.validation["passed"] is False
    assert len(llm.calls) == 2


async def test_chat_validation_graph_stops_before_revision(tmp_path):
    store, doc = _store_with_chunk(tmp_path)
    llm = FakeLLM()

    result = await run_chat_validation_graph(
        store=store,
        request=ChatRequest(message="What is self-attention?", doc_id=doc.doc_id, deep_mode=True),
        llm=llm,
    )

    assert result.initial_answer == "初始回答"
    assert result.validation["passed"] is False
    assert result.validation["correction"] == "修正后的回答"
    assert len(llm.calls) == 2


async def test_chat_graph_passes_intent_into_retrieval_debug(tmp_path):
    store, doc = _store_with_formula_chunk(tmp_path)
    llm = FakeLLM()

    result = await run_chat_graph(
        store=store,
        request=ChatRequest(message="解释这个公式里的 sqrt(d_k)", doc_id=doc.doc_id),
        llm=llm,
    )

    assert result.intent == "formula"
    assert result.evidence[0]["has_formula"] is True
