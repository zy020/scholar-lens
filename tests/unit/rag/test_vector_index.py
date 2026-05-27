from scholar_lens.core.settings import EmbeddingConfig, Settings
from scholar_lens.parsers.models import Chunk, ChunkMetadata
from scholar_lens.rag.vector_index import (
    embedding_configured,
    index_document_chunks,
    search_vector_chunks,
)


class FakeEmbeddings:
    def __init__(self):
        self.document_texts = []
        self.query_texts = []

    def embed_documents(self, texts):
        self.document_texts.append(texts)
        return [[float(i + 1)] * 3 for i, _ in enumerate(texts)]

    def embed_query(self, text):
        self.query_texts.append(text)
        return [1.0, 1.0, 1.0]


class FakeVectorStore:
    def __init__(self):
        self.deleted = []
        self.added = []
        self.queries = []

    def delete_by_doc_id(self, doc_id):
        self.deleted.append(doc_id)

    def add_chunks(self, chunks, embeddings):
        self.added.append((chunks, embeddings))

    def query_results(self, query_embedding, top_k=5, where=None):
        self.queries.append((query_embedding, top_k, where))
        from scholar_lens.rag.retriever import RetrievalResult

        return [
            RetrievalResult(
                chunk_id="doc1_1_0",
                text="Vector evidence",
                score=0.9,
                source="vector",
                rank=1,
                metadata={"doc_id": "doc1", "section_id": "1"},
            )
        ]


def test_embedding_configured_requires_key_and_model():
    assert embedding_configured(Settings(_env_file="", embedding=EmbeddingConfig())) is False
    assert embedding_configured(Settings(
        _env_file="",
        embedding=EmbeddingConfig(api_key="ek", base_url="https://emb.example/v1", model="emb"),
    )) is True


def test_index_document_chunks_skips_without_embedding_config(tmp_path):
    settings = Settings(_env_file="", data_dir=tmp_path, embedding=EmbeddingConfig())
    vector_store = FakeVectorStore()
    chunks = [
        Chunk(chunk_id="c1", text="Hello", metadata=ChunkMetadata(section_id="1", doc_id="doc1")),
    ]

    indexed = index_document_chunks(None, "doc1", chunks, settings, embeddings=FakeEmbeddings(), vector_store=vector_store)

    assert indexed is False
    assert vector_store.deleted == []
    assert vector_store.added == []


def test_index_document_chunks_deletes_then_adds_embeddings(tmp_path):
    settings = Settings(
        _env_file="",
        data_dir=tmp_path,
        embedding=EmbeddingConfig(api_key="ek", base_url="https://emb.example/v1", model="emb"),
    )
    vector_store = FakeVectorStore()
    chunks = [
        Chunk(chunk_id="c1", text="Hello", metadata=ChunkMetadata(section_id="1", doc_id="doc1")),
        Chunk(chunk_id="c2", text="World", metadata=ChunkMetadata(section_id="1", doc_id="doc1")),
    ]

    indexed = index_document_chunks(None, "doc1", chunks, settings, embeddings=FakeEmbeddings(), vector_store=vector_store)

    assert indexed is True
    assert vector_store.deleted == ["doc1"]
    assert vector_store.added[0][0] == chunks
    assert vector_store.added[0][1] == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]


def test_index_document_chunks_embeds_formula_context_without_changing_stored_text(tmp_path):
    settings = Settings(
        _env_file="",
        data_dir=tmp_path,
        embedding=EmbeddingConfig(api_key="ek", base_url="https://emb.example/v1", model="emb"),
    )
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()
    chunks = [
        Chunk(
            chunk_id="c1",
            text="Formula\n𝛼 = softmax(𝒒∙𝒌)",
            metadata=ChunkMetadata(
                section_id="1",
                doc_id="doc1",
                has_formula=True,
                formula_ids=["q dot k", "alpha softmax q k"],
                contextual_prefix="Formula terms: q dot k; alpha softmax q k",
            ),
        ),
    ]

    indexed = index_document_chunks(None, "doc1", chunks, settings, embeddings=embeddings, vector_store=vector_store)

    assert indexed is True
    assert "q dot k" in embeddings.document_texts[0][0]
    assert "alpha softmax q k" in embeddings.document_texts[0][0]
    assert vector_store.added[0][0][0].text == "Formula\n𝛼 = softmax(𝒒∙𝒌)"


def test_search_vector_chunks_returns_vector_results(tmp_path):
    settings = Settings(
        _env_file="",
        data_dir=tmp_path,
        embedding=EmbeddingConfig(api_key="ek", base_url="https://emb.example/v1", model="emb"),
    )
    vector_store = FakeVectorStore()

    results = search_vector_chunks("doc1", "attention", 3, settings, embeddings=FakeEmbeddings(), vector_store=vector_store)

    assert results[0].source == "vector"
    assert vector_store.queries == [([1.0, 1.0, 1.0], 3, {"doc_id": "doc1"})]
