from scholar_lens.api.chat_service import (
    build_chat_messages,
    build_no_llm_answer,
    configured_llm_configs,
    _expand_retrieval_context,
    _rewrite_query_variants,
    _search_bm25_variants,
    retrieve_chat_context,
    retrieve_chat_context_async,
)
from scholar_lens.core.settings import EmbeddingConfig, LLMConfig, RerankerConfig, Settings
from scholar_lens.parsers.models import Chunk, ChunkMetadata
from scholar_lens.rag.document_store import DocumentStore
from scholar_lens.rag.retriever import RetrievalResult


class FakeRewriteLLM:
    def __init__(self, content="self attention\nattention mechanism"):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        class Response:
            pass

        response = Response()
        response.content = self.content
        return response


def test_retrieve_chat_context_returns_context_and_evidence(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="c1",
            text="Transformer uses self-attention to connect tokens.",
            metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
        ),
    ])

    context = retrieve_chat_context(store, doc.doc_id, "self-attention", "method")

    assert "Transformer uses self-attention" in context.context_text
    assert context.evidence[0]["chunk_id"] == "c1"
    assert context.evidence[0]["section_id"] == "method"
    assert context.retrieval_debug["intent"] == "concept"
    assert context.retrieval_debug["evidence_count"] == 1


def test_build_chat_messages_includes_memory_context_as_style_context():
    messages = build_chat_messages(
        question="What is self-attention?",
        context_text="[1] Self-attention connects tokens.",
        memory_context="Student Profile: beginner\nActive Glossary: attention|||注意力",
    )

    user_prompt = messages[-1].content

    assert "Learning memory context" in user_prompt
    assert "Student Profile: beginner" in user_prompt
    assert "Memory is for personalization only" in user_prompt


def test_build_chat_messages_adds_personalized_guidance_from_memory():
    messages = build_chat_messages(
        question="Explain formula details",
        context_text="[1] Formula evidence.",
        student_level="beginner",
        memory_context=(
            "Student Profile: prefers intuitive explanations\n"
            "Active Glossary: self-attention|||自注意力\n"
            "Session Summary: Asked: 不理解公式"
        ),
    )

    user_prompt = messages[-1].content

    assert "Personalized teaching guidance" in user_prompt
    assert "Use a beginner-friendly explanation style" in user_prompt
    assert "self-attention" in user_prompt
    assert "The student recently struggled with formulas" in user_prompt


def test_build_chat_messages_separates_general_background_from_document_claims():
    messages = build_chat_messages(
        question="What is self-attention useful for?",
        context_text="[1] Self-attention lets each token collect information from other tokens.",
    )

    user_prompt = messages[-1].content

    assert "Do not present broader implications" in user_prompt
    assert "explicitly labeled as general background" in user_prompt
    assert "do not add an uncited concluding implication sentence" in user_prompt


def test_retrieve_chat_context_marks_formula_evidence_for_explanation(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="formula",
            text="Formula\n𝛼 = softmax(𝒒∙𝒌)",
            metadata=ChunkMetadata(
                section_id="slide_4",
                section_type="slide",
                content_type="slide",
                page_start=4,
                doc_id=doc.doc_id,
                has_formula=True,
                formula_ids=["q dot k", "alpha softmax q k"],
                contextual_prefix="Formula terms: q dot k; alpha softmax q k",
            ),
        ),
    ])

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "解释 q dot k 和 alpha",
        top_k=1,
        use_reranker=False,
    )

    assert "𝛼 = softmax(𝒒∙𝒌)" in context.context_text
    assert "Formula terms: q dot k; alpha softmax q k" in context.context_text
    assert context.evidence[0]["has_formula"] is True
    assert context.evidence[0]["formula_ids"] == ["q dot k", "alpha softmax q k"]


def test_retrieve_chat_context_can_use_strict_section_only(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="method",
            text="The method section explains self-attention.",
            metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="intro",
            text="The introduction also mentions self-attention.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
    ])

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "self-attention",
        section_id="method",
        section_only=True,
    )

    assert [item["section_id"] for item in context.evidence] == ["method"]


def test_retrieve_chat_context_uses_intent_hint_for_precise_detail_context(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c1", text="The accuracy score is 94 percent.", metadata=ChunkMetadata(section_id="results", doc_id=doc.doc_id)),
        Chunk(chunk_id="c2", text="The method uses a transformer encoder.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c3", text="The discussion adds broader implications.", metadata=ChunkMetadata(section_id="discussion", doc_id=doc.doc_id)),
    ])

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "accuracy score",
        top_k=1,
        use_reranker=False,
        intent_hint="detail",
    )

    assert context.retrieval_debug["intent"] == "detail"
    assert context.retrieval_debug["context_k"] == 1
    assert "context 1" not in context.context_text


def test_retrieve_chat_context_reranks_fact_queries(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="generic",
            text="The model reports useful evaluation findings.",
            metadata=ChunkMetadata(section_id="results", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="fact",
            text="The accuracy score is 94 percent, improving from 91 percent.",
            metadata=ChunkMetadata(
                section_id="results",
                doc_id=doc.doc_id,
                content_type="fact",
            ),
        ),
    ])

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "accuracy score 94 percent",
        use_reranker=True,
        top_k=1,
    )

    assert context.results[0].chunk_id == "fact"
    assert context.results[0].source == "diversity_reranked"


def test_retrieve_chat_context_uses_hybrid_vector_results(monkeypatch, tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="bm25",
            text="Neural retrieval lexical candidate.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="other",
            text="Unrelated baseline information.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="other2",
            text="Another unrelated paragraph.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
    ])
    settings = Settings(
        _env_file="",
        data_dir=tmp_path,
        embedding=EmbeddingConfig(api_key="ek", base_url="https://emb.example/v1", model="emb"),
    )

    monkeypatch.setattr(
        "scholar_lens.api.chat_service.search_vector_chunks",
        lambda *args, **kwargs: [
            RetrievalResult(
                chunk_id="vector",
                text="Vector semantic candidate.",
                score=0.95,
                source="vector",
                rank=1,
                metadata={"section_id": "method", "doc_id": doc.doc_id},
            )
        ],
    )

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "neural retrieval",
        settings=settings,
        use_reranker=False,
        top_k=2,
    )

    assert {result.chunk_id for result in context.results} == {"bm25", "vector"}
    assert all(result.source == "rrf_fused" for result in context.results)


def test_retrieve_chat_context_falls_back_when_vector_empty(monkeypatch, tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="bm25",
            text="Self-attention lexical candidate.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
    ])
    settings = Settings(
        _env_file="",
        data_dir=tmp_path,
        embedding=EmbeddingConfig(api_key="ek", base_url="https://emb.example/v1", model="emb"),
    )
    monkeypatch.setattr("scholar_lens.api.chat_service.search_vector_chunks", lambda *args, **kwargs: [])

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "self-attention",
        settings=settings,
        use_reranker=False,
        top_k=1,
    )

    assert context.results[0].chunk_id == "bm25"
    assert context.results[0].source in {"bm25", "overlap"}


def test_retrieve_chat_context_boosts_memory_current_section(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="intro",
            text="Attention is mentioned in broad background.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="method",
            text="Attention is used here to compute token interactions.",
            metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
        ),
    ])

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "attention",
        use_reranker=False,
        top_k=1,
        memory_hints={"current_section_id": "method", "concepts": []},
    )

    assert context.results[0].chunk_id == "method"


def test_retrieve_chat_context_boosts_memory_review_concepts(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="generic",
            text="The architecture has several components.",
            metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="formula",
            text="The attention formula uses query and key vectors.",
            metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
        ),
    ])

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "architecture",
        use_reranker=False,
        top_k=1,
        memory_hints={"current_section_id": "", "concepts": ["attention formula"]},
    )

    assert context.results[0].chunk_id == "formula"


def test_retrieve_chat_context_uses_context_k_for_expanded_prompt(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c0", text="Before context explains the setup.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c1", text="Needle answer uses self-attention.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c2", text="After context explains the consequence.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
    ])

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "Needle self-attention",
        top_k=1,
        context_k=3,
        use_reranker=False,
    )

    assert [result.chunk_id for result in context.results] == ["c1"]
    assert [item["chunk_id"] for item in context.evidence] == ["c1"]
    assert "Before context explains the setup" in context.context_text
    assert "Needle answer uses self-attention" in context.context_text
    assert "After context explains the consequence" in context.context_text


def test_expand_retrieval_context_adds_same_section_neighbors_for_paper(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    chunks = [
        Chunk(chunk_id="c0", text="Before explanation.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c1", text="Main retrieved formula.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c2", text="After explanation.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c3", text="Other section.", metadata=ChunkMetadata(section_id="results", doc_id=doc.doc_id)),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(
            chunk_id="c1",
            text="Main retrieved formula.",
            score=1.0,
            source="bm25",
            rank=1,
            metadata=chunks[1].metadata.model_dump(),
        )
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=3)

    assert [result.chunk_id for result in expanded] == ["c1", "c0", "c2"]
    assert expanded[1].source == "context_expanded"


def test_expand_retrieval_context_adds_adjacent_slides_for_courseware(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    chunks = [
        Chunk(chunk_id="s0", text="Previous slide.", metadata=ChunkMetadata(section_id="slide_0", section_type="slide", content_type="slide", page_start=0, page_end=0, doc_id=doc.doc_id)),
        Chunk(chunk_id="s1", text="Current slide.", metadata=ChunkMetadata(section_id="slide_1", section_type="slide", content_type="slide", page_start=1, page_end=1, doc_id=doc.doc_id)),
        Chunk(chunk_id="s2", text="Next slide.", metadata=ChunkMetadata(section_id="slide_2", section_type="slide", content_type="slide", page_start=2, page_end=2, doc_id=doc.doc_id)),
        Chunk(chunk_id="s4", text="Far slide.", metadata=ChunkMetadata(section_id="slide_4", section_type="slide", content_type="slide", page_start=4, page_end=4, doc_id=doc.doc_id)),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(
            chunk_id="s1",
            text="Current slide.",
            score=1.0,
            source="vector",
            rank=1,
            metadata=chunks[1].metadata.model_dump(),
        )
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=4)

    assert [result.chunk_id for result in expanded] == ["s1", "s0", "s2"]


def test_expand_retrieval_context_does_not_expand_courseware_figure_questions(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    chunks = [
        Chunk(chunk_id="s0", text="Previous slide about training.", metadata=ChunkMetadata(section_id="slide_0", section_type="slide", content_type="slide", page_start=0, page_end=0, doc_id=doc.doc_id)),
        Chunk(chunk_id="s1", text="Figure: CLIP aligns image and text representations.", metadata=ChunkMetadata(section_id="slide_1", section_type="slide", content_type="slide", page_start=1, page_end=1, doc_id=doc.doc_id)),
        Chunk(chunk_id="s2", text="Next slide about unrelated deployment.", metadata=ChunkMetadata(section_id="slide_2", section_type="slide", content_type="slide", page_start=2, page_end=2, doc_id=doc.doc_id)),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(
            chunk_id="s1",
            text=chunks[1].text,
            score=1.0,
            source="vector",
            rank=1,
            metadata=chunks[1].metadata.model_dump(),
        )
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=3, query="这张 figure 说明了什么？")

    assert [result.chunk_id for result in expanded] == ["s1"]


def test_expand_retrieval_context_uses_intent_hint_over_query_text(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    chunks = [
        Chunk(chunk_id="s0", text="Previous slide about training.", metadata=ChunkMetadata(section_id="slide_0", section_type="slide", content_type="slide", page_start=0, page_end=0, doc_id=doc.doc_id)),
        Chunk(chunk_id="s1", text="CLIP aligns image and text representations.", metadata=ChunkMetadata(section_id="slide_1", section_type="slide", content_type="slide", page_start=1, page_end=1, doc_id=doc.doc_id)),
        Chunk(chunk_id="s2", text="Next slide about deployment.", metadata=ChunkMetadata(section_id="slide_2", section_type="slide", content_type="slide", page_start=2, page_end=2, doc_id=doc.doc_id)),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(
            chunk_id="s1",
            text=chunks[1].text,
            score=1.0,
            source="vector",
            rank=1,
            metadata=chunks[1].metadata.model_dump(),
        )
    ]

    expanded = _expand_retrieval_context(
        store,
        doc.doc_id,
        results,
        limit=3,
        query="CLIP",
        intent_hint="figure",
    )

    assert [result.chunk_id for result in expanded] == ["s1"]


def test_expand_retrieval_context_expands_courseware_structure_questions(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    chunks = [
        Chunk(chunk_id="s0", text="Agenda: attention, transformer, applications.", metadata=ChunkMetadata(section_id="slide_0", section_type="slide", content_type="slide", page_start=0, page_end=0, doc_id=doc.doc_id)),
        Chunk(chunk_id="s1", text="Transformer architecture overview.", metadata=ChunkMetadata(section_id="slide_1", section_type="slide", content_type="slide", page_start=1, page_end=1, doc_id=doc.doc_id)),
        Chunk(chunk_id="s2", text="Self-attention details.", metadata=ChunkMetadata(section_id="slide_2", section_type="slide", content_type="slide", page_start=2, page_end=2, doc_id=doc.doc_id)),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(
            chunk_id="s1",
            text=chunks[1].text,
            score=1.0,
            source="bm25",
            rank=1,
            metadata=chunks[1].metadata.model_dump(),
        )
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=3, query="这部分课件的结构是什么？")

    assert [result.chunk_id for result in expanded] == ["s1", "s0", "s2"]


def test_expand_retrieval_context_deduplicates_near_duplicate_courseware_slides(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("slides.pdf")
    chunks = [
        Chunk(chunk_id="s1", text="CLIP learns transferable visual models from natural language supervision.", metadata=ChunkMetadata(section_id="slide_1", section_type="slide", content_type="slide", page_start=1, page_end=1, doc_id=doc.doc_id)),
        Chunk(chunk_id="s2", text="CLIP learns transferable visual models from natural language supervision.", metadata=ChunkMetadata(section_id="slide_2", section_type="slide", content_type="slide", page_start=2, page_end=2, doc_id=doc.doc_id)),
        Chunk(chunk_id="s3", text="Contrastive pre-training aligns image and text embeddings.", metadata=ChunkMetadata(section_id="slide_3", section_type="slide", content_type="slide", page_start=3, page_end=3, doc_id=doc.doc_id)),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(chunk_id="s1", text=chunks[0].text, score=1.0, source="bm25", rank=1, metadata=chunks[0].metadata.model_dump()),
        RetrievalResult(chunk_id="s2", text=chunks[1].text, score=0.9, source="bm25", rank=2, metadata=chunks[1].metadata.model_dump()),
        RetrievalResult(chunk_id="s3", text=chunks[2].text, score=0.8, source="bm25", rank=3, metadata=chunks[2].metadata.model_dump()),
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=3, query="CLIP 是什么？")

    assert [result.chunk_id for result in expanded] == ["s1", "s3"]


def test_expand_retrieval_context_deduplicates_and_respects_limit(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    chunks = [
        Chunk(chunk_id="c0", text="Before.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c1", text="First hit.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c2", text="Second hit.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(chunk_id="c1", text="First hit.", score=1.0, source="bm25", rank=1, metadata=chunks[1].metadata.model_dump()),
        RetrievalResult(chunk_id="c2", text="Second hit.", score=0.9, source="bm25", rank=2, metadata=chunks[2].metadata.model_dump()),
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=3)

    assert [result.chunk_id for result in expanded] == ["c1", "c2", "c0"]


def test_expand_retrieval_context_adds_paper_figure_referencing_body_chunk(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    chunks = [
        Chunk(chunk_id="intro", text="Intro text.", metadata=ChunkMetadata(section_id="results", doc_id=doc.doc_id)),
        Chunk(
            chunk_id="fig1",
            text="Figure 1: Attention patterns by layer.",
            metadata=ChunkMetadata(
                section_id="results",
                content_type="figure",
                caption="Figure 1: Attention patterns by layer.",
                referenced_by=["body_fig1"],
                doc_id=doc.doc_id,
            ),
        ),
        Chunk(chunk_id="other", text="Other result paragraph.", metadata=ChunkMetadata(section_id="results", doc_id=doc.doc_id)),
        Chunk(
            chunk_id="body_fig1",
            text="Figure 1 shows that lower layers attend locally while higher layers attend globally.",
            metadata=ChunkMetadata(section_id="results", cross_refs=["fig1"], doc_id=doc.doc_id),
        ),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(chunk_id="fig1", text=chunks[1].text, score=1.0, source="bm25", rank=1, metadata=chunks[1].metadata.model_dump())
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=3, query="Figure 1 说明了什么？")

    assert [result.chunk_id for result in expanded][:2] == ["fig1", "body_fig1"]


def test_expand_retrieval_context_adds_paper_formula_explanation_chunk(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    chunks = [
        Chunk(chunk_id="before", text="Before formula.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(
            chunk_id="formula",
            text="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V.",
            metadata=ChunkMetadata(section_id="method", has_formula=True, formula_ids=["attention softmax q k v"], doc_id=doc.doc_id),
        ),
        Chunk(chunk_id="middle", text="Implementation detail.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(
            chunk_id="explain",
            text="The softmax weights determine how much each token attends to other tokens.",
            metadata=ChunkMetadata(section_id="method", section_type="method", formula_ids=["attention softmax q k v"], doc_id=doc.doc_id),
        ),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(chunk_id="formula", text=chunks[1].text, score=1.0, source="bm25", rank=1, metadata=chunks[1].metadata.model_dump())
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=3, query="解释这个 attention 公式")

    assert [result.chunk_id for result in expanded][:2] == ["formula", "explain"]


def test_expand_retrieval_context_prefers_same_section_method_chunks(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    chunks = [
        Chunk(chunk_id="setup", text="General setup.", metadata=ChunkMetadata(section_id="method", section_type="prose", doc_id=doc.doc_id)),
        Chunk(chunk_id="hit", text="The proposed method uses graph message passing.", metadata=ChunkMetadata(section_id="method", section_type="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="aside", text="Background aside.", metadata=ChunkMetadata(section_id="method", section_type="prose", doc_id=doc.doc_id)),
        Chunk(chunk_id="method_detail", text="The method aggregates neighbor states before updating node states.", metadata=ChunkMetadata(section_id="method", section_type="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="other_section", text="Unrelated experiment result.", metadata=ChunkMetadata(section_id="results", section_type="results", doc_id=doc.doc_id)),
    ]
    store.save_chunks(doc.doc_id, chunks)
    results = [
        RetrievalResult(chunk_id="hit", text=chunks[1].text, score=1.0, source="bm25", rank=1, metadata=chunks[1].metadata.model_dump())
    ]

    expanded = _expand_retrieval_context(store, doc.doc_id, results, limit=3, query="How does the proposed method work?")

    assert [result.chunk_id for result in expanded][:2] == ["hit", "method_detail"]
    assert "other_section" not in [result.chunk_id for result in expanded]


def test_rewrite_query_variants_rewrites_pure_chinese(monkeypatch):
    fake_llm = FakeRewriteLLM("self attention\nattention mechanism")
    settings = Settings(
        _env_file="",
        llm=LLMConfig(api_key="lk", base_url="https://llm.example/v1", model="chat"),
        embedding=EmbeddingConfig(),
    )
    monkeypatch.setattr(
        "scholar_lens.api.chat_service.ChatLLMFactory",
        lambda config: type("Factory", (), {"create": lambda self, streaming=False: fake_llm})(),
    )

    variants = _rewrite_query_variants("自注意力怎么计算", settings)

    assert variants == ["自注意力怎么计算", "self attention", "attention mechanism"]
    assert fake_llm.calls


def test_rewrite_query_variants_rewrites_mixed_chinese_english(monkeypatch):
    fake_llm = FakeRewriteLLM("positional encoding\nposition embedding")
    settings = Settings(
        _env_file="",
        llm=LLMConfig(api_key="lk", base_url="https://llm.example/v1", model="chat"),
        embedding=EmbeddingConfig(),
    )
    monkeypatch.setattr(
        "scholar_lens.api.chat_service.ChatLLMFactory",
        lambda config: type("Factory", (), {"create": lambda self, streaming=False: fake_llm})(),
    )

    variants = _rewrite_query_variants("positional encoding 是什么", settings)

    assert variants == ["positional encoding 是什么", "positional encoding", "position embedding"]


def test_rewrite_query_variants_skips_english_only(monkeypatch):
    settings = Settings(
        _env_file="",
        llm=LLMConfig(api_key="lk", base_url="https://llm.example/v1", model="chat"),
        embedding=EmbeddingConfig(),
    )

    variants = _rewrite_query_variants("self attention", settings)

    assert variants == ["self attention"]


def test_rewrite_query_variants_falls_back_on_failure(monkeypatch):
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("model down")

    settings = Settings(
        _env_file="",
        llm=LLMConfig(api_key="lk", base_url="https://llm.example/v1", model="chat"),
        embedding=EmbeddingConfig(),
    )
    monkeypatch.setattr(
        "scholar_lens.api.chat_service.ChatLLMFactory",
        lambda config: type("Factory", (), {"create": lambda self, streaming=False: FailingLLM()})(),
    )

    variants = _rewrite_query_variants("自注意力", settings)

    assert variants == ["自注意力"]


def test_bm25_variants_can_retrieve_english_chunk_for_chinese_query(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="attention",
            text="Self attention computes compatibility scores between query and key vectors.",
            metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="other",
            text="The appendix lists unrelated implementation details.",
            metadata=ChunkMetadata(section_id="appendix", doc_id=doc.doc_id),
        ),
    ])
    from scholar_lens.rag.document_index import DocumentIndex

    index = DocumentIndex(store)
    results = _search_bm25_variants(
        index,
        doc.doc_id,
        ["自注意力怎么计算", "self attention query key"],
        section_id="",
        top_k=2,
        section_only=False,
    )

    assert results[0].chunk_id == "attention"


def test_retrieve_chat_context_uses_rewritten_bm25_for_chinese_query(monkeypatch, tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="attention",
            text="Self attention computes compatibility scores between query and key vectors.",
            metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="other",
            text="The appendix lists unrelated implementation details.",
            metadata=ChunkMetadata(section_id="appendix", doc_id=doc.doc_id),
        ),
    ])
    settings = Settings(
        _env_file="",
        llm=LLMConfig(api_key="lk", base_url="https://llm.example/v1", model="chat"),
        embedding=EmbeddingConfig(),
    )
    monkeypatch.setattr(
        "scholar_lens.api.chat_service._rewrite_query_variants",
        lambda query, settings, max_variants=3: [query, "self attention query key"],
    )

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "自注意力怎么计算",
        settings=settings,
        use_reranker=False,
        top_k=1,
    )

    assert context.results[0].chunk_id == "attention"


def test_retrieve_chat_context_fuses_rewritten_bm25_and_vector_for_chinese_query(monkeypatch, tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="attention",
            text="Self attention computes compatibility scores between query and key vectors.",
            metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="other",
            text="The appendix lists unrelated implementation details.",
            metadata=ChunkMetadata(section_id="appendix", doc_id=doc.doc_id),
        ),
    ])
    settings = Settings(
        _env_file="",
        llm=LLMConfig(api_key="lk", base_url="https://llm.example/v1", model="chat"),
        embedding=EmbeddingConfig(api_key="ek", base_url="https://emb.example/v1", model="emb"),
    )
    monkeypatch.setattr(
        "scholar_lens.api.chat_service._rewrite_query_variants",
        lambda query, settings, max_variants=3: [query, "self attention query key"],
    )
    monkeypatch.setattr(
        "scholar_lens.api.chat_service.search_vector_chunks",
        lambda *args, **kwargs: [
            RetrievalResult(
                chunk_id="vector",
                text="Vector candidate about attention.",
                score=0.95,
                source="vector",
                rank=1,
                metadata={"section_id": "method", "doc_id": doc.doc_id},
            )
        ],
    )

    context = retrieve_chat_context(
        store,
        doc.doc_id,
        "自注意力怎么计算",
        settings=settings,
        use_reranker=False,
        top_k=2,
    )

    assert {result.chunk_id for result in context.results} == {"attention", "vector"}
    assert all(result.source == "rrf_fused" for result in context.results)


async def test_retrieve_chat_context_async_uses_configured_model_reranker(monkeypatch, tmp_path):
    calls = []

    class FakeModelReranker:
        def rerank(self, results, query="", student_level="intermediate"):
            calls.append((query, student_level, len(results)))
            return list(reversed(results))

    monkeypatch.setattr(
        "scholar_lens.api.chat_service.ModelReranker",
        lambda **kwargs: FakeModelReranker(),
    )
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(
            chunk_id="first",
            text="First candidate about retrieval.",
            metadata=ChunkMetadata(section_id="s1", doc_id=doc.doc_id),
        ),
        Chunk(
            chunk_id="second",
            text="Second candidate about retrieval.",
            metadata=ChunkMetadata(section_id="s1", doc_id=doc.doc_id),
        ),
    ])
    settings = Settings(
        _env_file="",
        llm=LLMConfig(),
        embedding=EmbeddingConfig(),
        reranker=RerankerConfig(
            api_key="rk",
            base_url="https://reranker.example/v1",
            model="rerank-model",
        ),
    )

    context = await retrieve_chat_context_async(
        store,
        doc.doc_id,
        "retrieval",
        settings=settings,
        student_level="advanced",
        top_k=1,
    )

    assert calls == [("retrieval", "advanced", 2)]
    assert context.results[0].chunk_id == "second"


def test_no_llm_answer_is_honest_without_evidence():
    answer = build_no_llm_answer([])

    assert "未配置可用的 LLM" in answer
    assert "没有检索到可用文档证据" in answer


def test_build_chat_messages_adds_formula_guidance_when_formula_evidence_present():
    messages = build_chat_messages(
        question="解释 q dot k 和 alpha",
        context_text="[1] Formula\n𝛼 = softmax(𝒒∙𝒌)\nFormula terms: q dot k; alpha softmax q k",
        has_formula_evidence=True,
    )

    user_prompt = messages[1].content
    assert "公式相关问题" in user_prompt
    assert "引用原始公式" in user_prompt
    assert "解释变量或符号" in user_prompt
    assert "不要编造推导" in user_prompt


def test_build_chat_messages_omits_formula_guidance_without_formula_evidence():
    messages = build_chat_messages(
        question="什么是 self-attention",
        context_text="[1] Self-attention relates tokens.",
    )

    assert "公式相关问题" not in messages[1].content


def test_build_chat_messages_defaults_to_concise_grounded_answer():
    messages = build_chat_messages(
        question="这张 Figure 说明了什么？",
        context_text="[1] Figure: CLIP aligns image and text representations.",
    )

    user_prompt = messages[1].content
    assert "Default to a concise answer" in user_prompt
    assert "For figure or caption questions" in user_prompt
    assert "do not add mechanisms, examples, or background" in user_prompt
    assert "unless the student explicitly asks" in user_prompt


def test_build_chat_messages_marks_vision_structured_evidence_as_model_description():
    messages = build_chat_messages(
        question="这个图表说明了什么？",
        context_text="Visual type: chart\nChart summary: Accuracy rises with data size.",
    )

    user_prompt = messages[1].content
    assert "Vision-structured evidence" in user_prompt
    assert "model-generated description" in user_prompt


def test_build_chat_messages_allows_expanded_grounded_answer_when_requested():
    messages = build_chat_messages(
        question="请详细展开解释 CLIP 的图，并举例说明",
        context_text="[1] Figure: CLIP aligns image and text representations.",
    )

    user_prompt = messages[1].content
    assert "The student explicitly asks for an expanded explanation" in user_prompt
    assert "Separate document evidence from helpful background" in user_prompt
    assert "Do not present background knowledge as document evidence" in user_prompt


def test_configured_llm_configs_uses_backup_when_primary_missing():
    settings = Settings(
        _env_file="",
        llm=LLMConfig(),
        backup_llm=LLMConfig(api_key="backup-key", model="backup-model"),
        embedding=EmbeddingConfig(),
    )

    configs = configured_llm_configs(settings)

    assert len(configs) == 1
    assert configs[0].model == "backup-model"
