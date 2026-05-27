from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from scholar_lens.core.graph_utils import invoke_llm_with_retries, trace_graph_node


class MemoryUpdateGraphState(TypedDict, total=False):
    manager: Any
    event_type: str
    doc_id: str
    section_id: str
    payload: dict[str, Any]
    concepts: list[str]
    summary_llm: Any


def _build_memory_compression_prompt(recent_actions: list[str], event_type: str, payload: dict[str, Any]) -> str:
    actions = "\n".join(f"- {item}" for item in recent_actions[-8:])
    return (
        "请把下面的学习行为压缩成一句中文学习记忆摘要。"
        "要求：面向后续个性化检索和回答；保留关键英文术语；不要编造未出现的知识点；不超过 80 个汉字。\n\n"
        f"最新事件类型：{event_type}\n"
        f"最新事件内容：{payload}\n\n"
        f"近期行为：\n{actions}"
    )


def build_memory_update_graph():
    graph = StateGraph(MemoryUpdateGraphState)

    async def update_core_node(state: MemoryUpdateGraphState) -> MemoryUpdateGraphState:
        async def run():
            manager = state["manager"]
            manager._update_core_from_event(
                state["event_type"],
                state.get("doc_id", ""),
                state.get("section_id", ""),
                state.get("payload", {}),
            )
            return state
        return await trace_graph_node("memory", "update_core", doc_id=state.get("doc_id", ""), func=run)

    async def persist_event_node(state: MemoryUpdateGraphState) -> MemoryUpdateGraphState:
        async def run():
            await state["manager"].structured.add_learning_event(
                event_type=state["event_type"],
                doc_id=state.get("doc_id", ""),
                section_id=state.get("section_id", ""),
                payload=state.get("payload", {}),
            )
            return state
        return await trace_graph_node("memory", "persist_event", doc_id=state.get("doc_id", ""), func=run)

    async def extract_concepts_node(state: MemoryUpdateGraphState) -> MemoryUpdateGraphState:
        async def run():
            concepts = state["manager"]._extract_concepts(state["event_type"], state.get("payload", {}))
            return {**state, "concepts": concepts}
        return await trace_graph_node("memory", "extract_concepts", doc_id=state.get("doc_id", ""), func=run)

    async def upsert_concepts_node(state: MemoryUpdateGraphState) -> MemoryUpdateGraphState:
        async def run():
            manager = state["manager"]
            for concept in state.get("concepts", []):
                await manager.structured.upsert_concept_memory(
                    concept=concept,
                    doc_id=state.get("doc_id", ""),
                    status=manager._concept_status(state["event_type"], state.get("payload", {})),
                    signal=state["event_type"],
                    section_id=state.get("section_id", ""),
                )
            return state
        return await trace_graph_node("memory", "upsert_concepts", doc_id=state.get("doc_id", ""), func=run)

    async def compact_summary_node(state: MemoryUpdateGraphState) -> MemoryUpdateGraphState:
        async def run():
            manager = state["manager"]
            manager._recent_actions = manager._recent_actions[-8:]
            summary_llm = state.get("summary_llm")
            if summary_llm is not None and manager._recent_actions:
                response = await invoke_llm_with_retries(
                    summary_llm,
                    [HumanMessage(content=_build_memory_compression_prompt(
                        manager._recent_actions,
                        state["event_type"],
                        state.get("payload", {}),
                    ))],
                    graph_name="memory",
                    node_name="compact_summary",
                    attempts=2,
                )
                summary = str(response.content or "").strip()
                if summary:
                    manager.core_memory.session_summary = summary[:500]
                    return state
            if manager._recent_actions:
                manager.core_memory.session_summary = "Recent learning actions: " + " | ".join(manager._recent_actions)
            return state
        return await trace_graph_node("memory", "compact_summary", doc_id=state.get("doc_id", ""), func=run)

    graph.add_node("update_core", update_core_node)
    graph.add_node("persist_event", persist_event_node)
    graph.add_node("extract_concepts", extract_concepts_node)
    graph.add_node("upsert_concepts", upsert_concepts_node)
    graph.add_node("compact_summary", compact_summary_node)
    graph.set_entry_point("update_core")
    graph.add_edge("update_core", "persist_event")
    graph.add_edge("persist_event", "extract_concepts")
    graph.add_edge("extract_concepts", "upsert_concepts")
    graph.add_edge("upsert_concepts", "compact_summary")
    graph.add_edge("compact_summary", END)
    return graph.compile()


async def run_memory_update_graph(
    manager,
    *,
    event_type: str,
    doc_id: str = "",
    section_id: str = "",
    payload: dict[str, Any] | None = None,
    summary_llm=None,
) -> None:
    graph = build_memory_update_graph()
    await graph.ainvoke({
        "manager": manager,
        "event_type": event_type,
        "doc_id": doc_id,
        "section_id": section_id,
        "payload": payload or {},
        "summary_llm": summary_llm,
    })
