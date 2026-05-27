from scholar_lens.parsers.models import Chunk, ChunkMetadata
from scholar_lens.rag.document_store import DocumentStore
from scholar_lens.rag.retrieval_debug import evaluate_retrieval_cases, trace_retrieval


def test_trace_retrieval_reports_stages_evidence_and_context(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c0", text="Before context explains the setup.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c1", text="Needle answer uses self-attention.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c2", text="After context explains the consequence.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
    ])

    trace = trace_retrieval(
        store,
        doc.doc_id,
        "Needle self-attention",
        top_k=1,
        context_k=3,
        use_reranker=False,
    )

    assert trace["query_variants"] == ["Needle self-attention"]
    assert trace["stages"]["bm25"][0]["chunk_id"] == "c1"
    assert trace["evidence"][0]["chunk_id"] == "c1"
    assert trace["context"][1]["source"] == "context_expanded"
    assert "text" not in trace["context"][0]


def test_trace_retrieval_can_include_full_text_for_eval_reuse(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c0", text="Full evidence text should be reusable by full eval.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
    ])

    trace = trace_retrieval(
        store,
        doc.doc_id,
        "Full evidence",
        top_k=1,
        context_k=1,
        use_reranker=False,
        include_text=True,
    )

    assert trace["context"][0]["text"] == "Full evidence text should be reusable by full eval."


def test_evaluate_retrieval_cases_computes_hit_and_context_hit(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c0", text="Before context explains the setup.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c1", text="Needle answer uses self-attention.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c2", text="After context explains the consequence.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
    ])
    cases = [
        {
            "id": "q1",
            "doc_id": doc.doc_id,
            "query": "Needle self-attention",
            "targets": [
                {"section_ids": ["method"]},
                {"chunk_ids": ["c0"]},
            ],
        }
    ]

    report = evaluate_retrieval_cases(
        store,
        cases,
        top_k=1,
        context_k=3,
        use_reranker=False,
    )

    assert report["summary"]["hit_at_k"] == 1.0
    assert report["summary"]["context_hit_at_k"] == 1.0
    assert report["records"][0]["metrics"]["matched_targets"] == [0, 1]


def test_evaluate_retrieval_cases_reports_ranking_and_empty_metrics(tmp_path):
    store = DocumentStore(root=tmp_path)
    doc = store.create_document("paper.pdf")
    store.save_chunks(doc.doc_id, [
        Chunk(chunk_id="c0", text="The setup chunk.", metadata=ChunkMetadata(section_id="intro", doc_id=doc.doc_id)),
        Chunk(chunk_id="c1", text="Distractor chunk.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
        Chunk(chunk_id="c2", text="Needle evidence supports the answer.", metadata=ChunkMetadata(section_id="method", doc_id=doc.doc_id)),
    ])
    cases = [
        {
            "id": "ranked",
            "doc_id": doc.doc_id,
            "query": "distractor needle evidence",
            "targets": [
                {"terms": ["Distractor chunk"]},
                {"chunk_ids": ["c0"]},
            ],
        },
        {
            "id": "empty",
            "doc_id": "missing-doc",
            "query": "zzzzzz unmatched",
            "targets": [{"terms": ["Needle evidence"]}],
        },
    ]

    report = evaluate_retrieval_cases(
        store,
        cases,
        top_k=2,
        context_k=3,
        use_reranker=False,
    )

    ranked = report["records"][0]["metrics"]
    assert ranked["hit_at_k"] == 1.0
    assert ranked["mrr_at_k"] == 0.5
    assert ranked["recall_at_k"] == 0.5
    assert ranked["context_recall_at_k"] == 0.5
    assert ranked["evidence_rank"] == 2
    assert ranked["empty_retrieval"] == 0.0
    empty = report["records"][1]["metrics"]
    assert empty["empty_retrieval"] == 1.0
    assert report["summary"]["mrr_at_k"] == 0.25
    assert report["summary"]["recall_at_k"] == 0.25
    assert report["summary"]["context_recall_at_k"] == 0.25
    assert report["summary"]["empty_retrieval_rate"] == 0.5
