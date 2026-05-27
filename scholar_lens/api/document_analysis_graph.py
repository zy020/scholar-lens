from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from scholar_lens.core.graph_utils import trace_graph_node
from scholar_lens.api.document_analysis import (
    AnalysisRunResult,
    _full_text_from_chunks,
    _section_dicts,
    hydrate_memory_from_analysis,
)
from scholar_lens.core.models import DocumentUnderstanding
from scholar_lens.rag.document_store import DocumentStore


class DocumentAnalysisGraphState(TypedDict, total=False):
    store: DocumentStore
    doc_id: str
    analyzer: Any
    memory_manager: Any
    sections: list
    chunks: list[dict]
    doc_text: str
    section_dicts: list[dict]
    understanding: DocumentUnderstanding
    result: AnalysisRunResult


def build_document_analysis_graph():
    graph = StateGraph(DocumentAnalysisGraphState)

    async def prepare_node(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
        async def run():
            store = state["store"]
            doc_id = state["doc_id"]
            sections = store.load_sections(doc_id)
            chunks = store.load_chunks(doc_id)
            return {
                **state,
                "sections": sections,
                "chunks": chunks,
                "doc_text": _full_text_from_chunks(chunks),
                "section_dicts": _section_dicts(sections),
            }
        return await trace_graph_node("document_analysis", "prepare", doc_id=state.get("doc_id", ""), func=run)

    async def analyze_node(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
        async def run():
            analyzer = state["analyzer"]
            understanding = await analyzer.analyze_document(
                state.get("doc_text", ""),
                state.get("section_dicts", []),
                memory_manager=state.get("memory_manager"),
            )
            return {**state, "understanding": understanding}
        return await trace_graph_node("document_analysis", "analyze", doc_id=state.get("doc_id", ""), func=run)

    async def persist_node(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
        async def run():
            store = state["store"]
            doc_id = state["doc_id"]
            store.save_understanding(doc_id, state["understanding"])
            store.save_analysis_meta(doc_id, {
                "source": "llm",
                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "error": "",
            })
            return {
                **state,
                "result": AnalysisRunResult(doc_id=doc_id, status="enhanced", source="llm"),
            }
        return await trace_graph_node("document_analysis", "persist", doc_id=state.get("doc_id", ""), func=run)

    async def hydrate_node(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
        async def run():
            memory_manager = state.get("memory_manager")
            if memory_manager is not None:
                hydrate_memory_from_analysis(memory_manager, state["understanding"], doc_id=state["doc_id"])
            return state
        return await trace_graph_node("document_analysis", "hydrate", doc_id=state.get("doc_id", ""), func=run)

    graph.add_node("prepare", prepare_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("persist", persist_node)
    graph.add_node("hydrate", hydrate_node)
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "analyze")
    graph.add_edge("analyze", "persist")
    graph.add_edge("persist", "hydrate")
    graph.add_edge("hydrate", END)
    return graph.compile()


async def run_document_analysis_graph(
    *,
    store: DocumentStore,
    doc_id: str,
    analyzer: Any,
    memory_manager=None,
) -> AnalysisRunResult:
    graph = build_document_analysis_graph()
    final = await graph.ainvoke({
        "store": store,
        "doc_id": doc_id,
        "analyzer": analyzer,
        "memory_manager": memory_manager,
    })
    return final["result"]
