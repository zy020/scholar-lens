"""End-to-end reading session test.

Complete flow: upload PDF → parse → chunk → embed → index →
student asks question → retrieve → explain → validate → tutor responds.

Uses real LLM/embedding settings and optional local documents supplied by
environment variables. These tests are skipped by default and do not require
private courseware paths to exist in a fresh clone.
"""

import os
import pytest


@pytest.mark.skipif(not os.getenv("E2E_TEST"), reason="Set E2E_TEST=1 for end-to-end session test")
class TestEndToEndSession:
    """Full reading session with real paper."""

    @pytest.mark.asyncio
    async def test_full_paper_pipeline(self):
        """Complete pipeline: parse → chunk → embed → retrieve → tutor."""
        import time, uuid, numpy as np
        from pathlib import Path
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_core.messages import HumanMessage, SystemMessage
        from scholar_lens.core.settings import Settings
        from scholar_lens.parsers.pdf_parser import PDFParser
        from scholar_lens.parsers.chunker import SectionAwareChunker
        from scholar_lens.rag.retriever import HybridRetriever, RetrievalResult
        from scholar_lens.rag.reranker import RerankerPipeline
        from scholar_lens.core.token_tracker import estimate_tokens
        from scholar_lens.agents.prompts import EXPLAINER_SYSTEM

        paper_path = os.getenv("E2E_PAPER_PATH", "")
        if not paper_path:
            pytest.skip("Set E2E_PAPER_PATH to run the paper E2E test")
        if not Path(paper_path).exists():
            pytest.skip(f"Test document not found: {paper_path}")

        s = Settings()
        llm = ChatOpenAI(api_key=s.llm.api_key, base_url=s.llm.base_url, model=s.llm.model)
        emb = OpenAIEmbeddings(api_key=s.embedding.api_key, base_url=s.embedding.base_url, model=s.embedding.model)

        # === Stage 1: Parse ===
        doc_id = str(uuid.uuid4())[:8]
        print(f"\n=== Full Pipeline: {Path(paper_path).name} (id={doc_id}) ===")

        parser = PDFParser()
        start = time.time()
        doc = parser.parse(Path(paper_path))
        print(f"  [1/4] Parse: {time.time()-start:.1f}s, {len(doc.pages)} pages, subtype={doc.doc_subtype}")

        chunker = SectionAwareChunker(max_chunk_tokens=800)
        chunks = chunker.chunk_with_facts(doc, doc_id=doc_id)
        regular = [c for c in chunks if c.metadata.content_type != "fact"]
        facts = [c for c in chunks if c.metadata.content_type == "fact"]
        print(f"  [1/4] Chunks: {len(regular)} regular + {len(facts)} fact = {len(chunks)} total")

        # === Stage 2: Embed & Index ===
        start = time.time()
        embeddings = emb.embed_documents([c.text for c in chunks])
        print(f"  [2/4] Embed: {time.time()-start:.1f}s, {len(embeddings)} vectors, dim={len(embeddings[0])}")

        retriever = HybridRetriever()
        retriever.build_bm25_index(chunks)

        # === Stage 3: Query & Retrieve ===
        queries = [
            ("ZH", "LLaVA是什么？它的主要创新点是什么？"),
            ("ZH", "LLaVA的模型架构是什么样的？"),
        ]

        pipeline = RerankerPipeline(diversity=True, max_per_doc=3)

        for lang, query in queries:
            print(f"\n  [3/4] Query: {query}")

            q_emb = emb.embed_query(query)
            q_vec = np.array(q_emb)
            scores = [float(np.dot(q_vec, np.array(ce))) for ce in embeddings]
            top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:10]
            vr = [RetrievalResult(chunk_id=chunks[i].chunk_id, text=chunks[i].text,
                    score=s, source="vector", rank=r+1, metadata=chunks[i].metadata.model_dump())
                  for r, (i, s) in enumerate(top_k)]

            bm25_signal = retriever._bm25_signal_strength(retriever.bm25_search(query, 10))
            fused = retriever.hybrid_search(query, q_emb, vr, top_k=5)
            reranked = pipeline.rerank(fused, query=query)

            top_doc = reranked[0].metadata.get("doc_id", "?")
            print(f"  [3/4] BM25 signal={bm25_signal:.2f}, top doc={top_doc}, diversity={len(set(r.metadata.get('doc_id','') for r in reranked[:5]))}")

            # === Stage 4: Tutor Response ===
            context = "\n\n".join(r.text[:500] for r in reranked[:3])
            response = await llm.ainvoke([
                SystemMessage(content=EXPLAINER_SYSTEM),
                HumanMessage(content=f"""Retrieved context from the paper:
{context}

Question: {query}

Answer in Chinese with key English terms preserved inline:"""),
            ])
            print(f"  [4/4] Tutor: {response.content[:250]}...")
            assert len(response.content) > 50

        print("  ✅ Full paper pipeline PASSED")

    @pytest.mark.asyncio
    async def test_courseware_end_to_end(self):
        """Test with courseware PDF (different format from research paper)."""
        import time, numpy as np
        from pathlib import Path
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_core.messages import HumanMessage, SystemMessage
        from scholar_lens.core.settings import Settings
        from scholar_lens.parsers.pdf_parser import PDFParser
        from scholar_lens.parsers.chunker import SectionAwareChunker
        from scholar_lens.rag.retriever import HybridRetriever, RetrievalResult
        from scholar_lens.rag.reranker import RerankerPipeline
        from scholar_lens.core.token_tracker import estimate_tokens

        courseware_path = os.getenv("E2E_COURSEWARE_PDF_PATH", "")
        if not courseware_path:
            pytest.skip("Set E2E_COURSEWARE_PDF_PATH to run the courseware PDF E2E test")
        if not Path(courseware_path).exists():
            pytest.skip(f"Courseware not found: {courseware_path}")

        s = Settings()
        llm = ChatOpenAI(api_key=s.llm.api_key, base_url=s.llm.base_url, model=s.llm.model)
        emb = OpenAIEmbeddings(api_key=s.embedding.api_key, base_url=s.embedding.base_url, model=s.embedding.model)

        print(f"\n=== E2E Courseware: {Path(courseware_path).name} ===")

        # Parse
        parser = PDFParser()
        start = time.time()
        doc = parser.parse(Path(courseware_path))
        print(f"  Parse: {time.time()-start:.1f}s, {len(doc.pages)} pages, subtype={doc.doc_subtype}")

        # Chunk
        chunker = SectionAwareChunker(max_chunk_tokens=800)
        chunks = chunker.chunk(doc, doc_id="modern_transformer")
        print(f"  Chunks: {len(chunks)}, tokens: {sum(estimate_tokens(c.text) for c in chunks)}")

        assert len(chunks) > 0
        assert doc.doc_subtype in {"slides_pdf", "research_paper"}

        # Embed & index
        embeddings = emb.embed_documents([c.text for c in chunks])
        retriever = HybridRetriever()
        retriever.build_bm25_index(chunks)

        # Query
        query = "这节课讲了哪些关于Transformer的改进？"
        q_emb = emb.embed_query(query)
        q_vec = np.array(q_emb)
        scores = [float(np.dot(q_vec, np.array(ce))) for ce in embeddings]
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:10]
        vr = [RetrievalResult(chunk_id=chunks[i].chunk_id, text=chunks[i].text,
                score=s, source="vector", rank=r+1, metadata=chunks[i].metadata.model_dump())
              for r, (i, s) in enumerate(top_k)]

        pipeline = RerankerPipeline(diversity=True, max_per_doc=3)
        fused = retriever.hybrid_search(query, q_emb, vr, top_k=5)
        reranked = pipeline.rerank(fused, query=query)

        top = reranked[0]
        print(f"  Top chunk: {top.text[:120]}...")

        # Generate answer
        context = "\n\n".join(r.text[:500] for r in reranked[:3])
        response = await llm.ainvoke([
            SystemMessage(content="You are a helpful Chinese tutor explaining course slides."),
            HumanMessage(content=f"根据以下课件内容回答问题。Context:\n{context}\n\nQuestion: {query}"),
        ])
        print(f"  Response: {response.content[:300]}...")
        assert len(response.content) > 50

        print("  ✅ Courseware E2E PASSED")
