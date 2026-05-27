from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from scholar_lens.core.graph_utils import invoke_llm_with_retries, trace_graph_node
from scholar_lens.api.brief_builder import (
    build_lecture_llm_brief_prompt,
    build_llm_brief_prompt,
    parse_llm_brief_json,
)
from scholar_lens.api.schemas import PaperBriefResponse, SectionSummary


LECTURE_DOC_TYPES = {"slides_pdf", "courseware_pptx", "lecture_slide", "courseware"}


class BriefGenerationGraphState(TypedDict, total=False):
    doc_id: str
    title: str
    doc_type: str
    text_quality: str
    ocr_needed: bool
    sections: list[SectionSummary]
    chunks: list[dict]
    llm: Any
    prompt: str
    system_prompt: str
    raw_response: str
    brief: PaperBriefResponse


def is_lecture_doc_type(doc_type: str) -> bool:
    return doc_type in LECTURE_DOC_TYPES


def build_brief_generation_graph():
    graph = StateGraph(BriefGenerationGraphState)

    async def prepare_node(state: BriefGenerationGraphState) -> BriefGenerationGraphState:
        async def run():
            if is_lecture_doc_type(state.get("doc_type", "")):
                prompt = build_lecture_llm_brief_prompt(state["title"], state.get("sections", []), state.get("chunks", []))
                system_prompt = "You produce strict JSON for lecture study briefs."
            else:
                prompt = build_llm_brief_prompt(state["title"], state.get("sections", []), state.get("chunks", []))
                system_prompt = "You produce strict JSON for academic paper understanding briefs."
            return {**state, "prompt": prompt, "system_prompt": system_prompt}
        return await trace_graph_node("brief", "prepare", doc_id=state.get("doc_id", ""), func=run)

    async def generate_node(state: BriefGenerationGraphState) -> BriefGenerationGraphState:
        async def run():
            response = await invoke_llm_with_retries(
                state["llm"],
                [
                    SystemMessage(content=state["system_prompt"]),
                    HumanMessage(content=state["prompt"]),
                ],
                graph_name="brief",
                node_name="generate",
                attempts=2,
            )
            return {**state, "raw_response": str(response.content)}
        return await trace_graph_node("brief", "generate", doc_id=state.get("doc_id", ""), func=run)

    async def parse_node(state: BriefGenerationGraphState) -> BriefGenerationGraphState:
        async def run():
            brief = parse_llm_brief_json(state["doc_id"], state["title"], state["raw_response"])
            if is_lecture_doc_type(state.get("doc_type", "")):
                brief.brief_type = "lecture"
            brief.text_quality = state.get("text_quality", "unknown")
            brief.ocr_needed = bool(state.get("ocr_needed", False))
            return {**state, "brief": brief}
        return await trace_graph_node("brief", "parse", doc_id=state.get("doc_id", ""), func=run)

    graph.add_node("prepare", prepare_node)
    graph.add_node("generate", generate_node)
    graph.add_node("parse", parse_node)
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "generate")
    graph.add_edge("generate", "parse")
    graph.add_edge("parse", END)
    return graph.compile()


async def run_brief_generation_graph(
    *,
    doc_id: str,
    title: str,
    doc_type: str,
    text_quality: str,
    ocr_needed: bool,
    sections: list[SectionSummary],
    chunks: list[dict],
    llm: Any,
) -> PaperBriefResponse:
    graph = build_brief_generation_graph()
    final = await graph.ainvoke({
        "doc_id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "text_quality": text_quality,
        "ocr_needed": ocr_needed,
        "sections": sections,
        "chunks": chunks,
        "llm": llm,
    })
    return final["brief"]
