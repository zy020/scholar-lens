from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from scholar_lens.api.chat_service import (
    ChatContext,
    _detect_question_intent,
    build_chat_messages,
    build_no_llm_answer,
    configured_llm_configs,
    retrieve_chat_context_async,
)
from scholar_lens.api.schemas import ChatRequest
from scholar_lens.core.graph_utils import invoke_llm_with_retries, trace_graph_node
from scholar_lens.core.llm_factory import ChatLLMFactory
from scholar_lens.core.utils import extract_json_from_llm_output
from scholar_lens.rag.document_store import DocumentStore


@dataclass
class ChatGraphResult:
    content: str
    evidence: list[dict]
    intent: str = "concept"
    validation: dict[str, Any] | None = None
    deep_mode: bool = False


@dataclass
class ChatRetrievalGraphResult:
    context: ChatContext
    evidence: list[dict]
    intent: str = "concept"


@dataclass
class ChatValidationGraphResult:
    initial_answer: str
    context: ChatContext
    evidence: list[dict]
    intent: str = "concept"
    validation: dict[str, Any] | None = None


class ChatGraphState(TypedDict, total=False):
    request: ChatRequest
    settings: Any
    store: DocumentStore
    section_title: str
    memory_context: str
    memory_hints: dict
    llm: Any
    intent: str
    context: ChatContext
    answer: str
    evidence: list[dict]
    validation: dict[str, Any]


def _configured_llm(settings, llm=None):
    if llm is not None:
        return llm
    configs = configured_llm_configs(settings)
    if not configs:
        return None
    return ChatLLMFactory.from_settings(settings).create(config=configs[0], streaming=False)


def _needs_validator(intent: str, request: ChatRequest) -> bool:
    return bool(request.deep_mode or intent in {"formula", "figure"})


def _build_validation_prompt(answer: str, context: ChatContext, request: ChatRequest) -> str:
    return (
        "You are validating a RAG answer for an academic learning assistant. "
        "Return JSON only with keys: passed(boolean), issues(array of strings), correction(string). "
        "Check whether the answer is grounded in the provided evidence, answers the question, "
        "and clearly states uncertainty when evidence is insufficient.\n\n"
        f"Question:\n{request.message}\n\n"
        f"Evidence:\n{context.context_text[:4000]}\n\n"
        f"Retrieval debug:\n{json.dumps(context.retrieval_debug or {}, ensure_ascii=False)}\n\n"
        f"Answer:\n{answer[:3000]}"
    )


def _parse_validation(raw: str) -> dict[str, Any]:
    data = extract_json_from_llm_output(raw)
    if not isinstance(data, dict):
        return {"passed": False, "issues": ["Validator returned non-JSON output."], "correction": ""}
    return {
        "passed": bool(data.get("passed", True)),
        "issues": [str(item) for item in data.get("issues", [])],
        "correction": str(data.get("correction", "") or ""),
    }


def _build_revision_prompt(answer: str, validation: dict[str, Any], context: ChatContext, request: ChatRequest) -> str:
    suggested_correction = str(validation.get("correction") or "").strip()
    return (
        "Revise the answer in Chinese using only the evidence. Preserve useful citations like [1]. "
        "Address the validator issues and avoid adding unsupported claims.\n\n"
        f"Question:\n{request.message}\n\n"
        f"Evidence:\n{context.context_text[:4000]}\n\n"
        f"Original answer:\n{answer[:3000]}\n\n"
        f"Validator issues:\n{json.dumps(validation.get('issues', []), ensure_ascii=False)}"
        + (f"\n\nValidator suggested correction:\n{suggested_correction[:2000]}" if suggested_correction else "")
    )


def build_revision_messages(
    *,
    answer: str,
    validation: dict[str, Any],
    context: ChatContext,
    request: ChatRequest,
) -> list[HumanMessage]:
    return [HumanMessage(content=_build_revision_prompt(answer, validation, context, request))]


def build_chat_graph(*, deep_mode: bool = False, revise: bool = True):
    graph = StateGraph(ChatGraphState)

    async def intent_node(state: ChatGraphState) -> ChatGraphState:
        async def run():
            request = state["request"]
            intent = _detect_question_intent(request.message)
            return {**state, "intent": intent}
        return await trace_graph_node("chat", "intent", doc_id=state["request"].doc_id, func=run)

    async def retrieve_node(state: ChatGraphState) -> ChatGraphState:
        async def run():
            request = state["request"]
            context = await retrieve_chat_context_async(
                state["store"],
                request.doc_id,
                request.message,
                request.section_id,
                top_k=request.top_k,
                context_k=request.context_k,
                section_only=request.section_only,
                use_reranker=request.use_reranker,
                student_level=request.student_level,
                settings=state.get("settings"),
                memory_hints=state.get("memory_hints") or {},
                intent_hint=state.get("intent"),
            )
            intent = str((context.retrieval_debug or {}).get("intent") or state.get("intent") or "concept")
            return {**state, "context": context, "intent": intent, "evidence": context.evidence}
        return await trace_graph_node("chat", "retrieve", doc_id=state["request"].doc_id, func=run)

    async def answer_node(state: ChatGraphState) -> ChatGraphState:
        async def run():
            request = state["request"]
            context = state["context"]
            llm = state.get("llm")
            if llm is None:
                return {**state, "answer": build_no_llm_answer(context.evidence)}
            response = await invoke_llm_with_retries(
                llm,
                build_chat_messages(
                    question=request.message,
                    context_text=context.context_text,
                    section_title=state.get("section_title", ""),
                    has_formula_evidence=context.has_formula_evidence,
                    memory_context=state.get("memory_context", ""),
                    student_level=request.student_level,
                ),
                graph_name="chat",
                node_name="answer",
                attempts=2,
            )
            return {**state, "answer": str(response.content)}
        return await trace_graph_node("chat", "answer", doc_id=state["request"].doc_id, func=run)

    async def validate_node(state: ChatGraphState) -> ChatGraphState:
        async def run():
            request = state["request"]
            llm = state.get("llm")
            if llm is None or not _needs_validator(state.get("intent", "concept"), request):
                return {**state, "validation": {"passed": True, "issues": [], "correction": ""}}
            response = await invoke_llm_with_retries(
                llm,
                [HumanMessage(content=_build_validation_prompt(
                    state.get("answer", ""),
                    state["context"],
                    request,
                ))],
                graph_name="chat",
                node_name="validate",
                attempts=2,
            )
            validation = _parse_validation(str(response.content))
            return {**state, "validation": validation}
        return await trace_graph_node("chat", "validate", doc_id=state["request"].doc_id, func=run)

    async def revise_node(state: ChatGraphState) -> ChatGraphState:
        async def run():
            validation = state.get("validation") or {}
            correction = str(validation.get("correction") or "").strip()
            if validation.get("passed", True):
                return state
            if correction:
                return {**state, "answer": correction}
            llm = state.get("llm")
            if llm is None:
                return state
            response = await invoke_llm_with_retries(
                llm,
                build_revision_messages(
                    answer=state.get("answer", ""),
                    validation=validation,
                    context=state["context"],
                    request=state["request"],
                ),
                graph_name="chat",
                node_name="revise",
                attempts=2,
            )
            return {**state, "answer": str(response.content)}
        return await trace_graph_node("chat", "revise", doc_id=state["request"].doc_id, func=run)

    graph.add_node("intent", intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("validate", validate_node)
    graph.add_node("revise", revise_node)
    graph.set_entry_point("intent")
    graph.add_edge("intent", "retrieve")
    graph.add_edge("retrieve", "answer")
    if deep_mode:
        graph.add_edge("answer", "validate")
        if revise:
            graph.add_edge("validate", "revise")
            graph.add_edge("revise", END)
        else:
            graph.add_edge("validate", END)
    else:
        graph.add_edge("answer", END)
    return graph.compile()


def build_chat_retrieval_graph():
    graph = StateGraph(ChatGraphState)

    async def intent_node(state: ChatGraphState) -> ChatGraphState:
        async def run():
            request = state["request"]
            return {**state, "intent": _detect_question_intent(request.message)}
        return await trace_graph_node("chat_retrieval", "intent", doc_id=state["request"].doc_id, func=run)

    async def retrieve_node(state: ChatGraphState) -> ChatGraphState:
        async def run():
            request = state["request"]
            context = await retrieve_chat_context_async(
                state["store"],
                request.doc_id,
                request.message,
                request.section_id,
                top_k=request.top_k,
                context_k=request.context_k,
                section_only=request.section_only,
                use_reranker=request.use_reranker,
                student_level=request.student_level,
                settings=state.get("settings"),
                memory_hints=state.get("memory_hints") or {},
                intent_hint=state.get("intent"),
            )
            intent = str((context.retrieval_debug or {}).get("intent") or state.get("intent") or "concept")
            return {**state, "context": context, "intent": intent, "evidence": context.evidence}
        return await trace_graph_node("chat_retrieval", "retrieve", doc_id=state["request"].doc_id, func=run)

    graph.add_node("intent", intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.set_entry_point("intent")
    graph.add_edge("intent", "retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()


async def run_chat_retrieval_graph(
    *,
    store: DocumentStore,
    request: ChatRequest,
    settings=None,
    memory_hints: dict | None = None,
) -> ChatRetrievalGraphResult:
    graph = build_chat_retrieval_graph()
    final = await graph.ainvoke({
        "request": request,
        "settings": settings,
        "store": store,
        "memory_hints": memory_hints or {},
    })
    context = final["context"]
    return ChatRetrievalGraphResult(
        context=context,
        evidence=final.get("evidence", []) or [],
        intent=str(final.get("intent", "concept")),
    )


async def run_chat_validation_graph(
    *,
    store: DocumentStore,
    request: ChatRequest,
    settings=None,
    section_title: str = "",
    memory_context: str = "",
    memory_hints: dict | None = None,
    llm=None,
) -> ChatValidationGraphResult:
    llm = _configured_llm(settings, llm=llm)
    graph = build_chat_graph(deep_mode=True, revise=False)
    final = await graph.ainvoke({
        "request": request,
        "settings": settings,
        "store": store,
        "section_title": section_title,
        "memory_context": memory_context,
        "memory_hints": memory_hints or {},
        "llm": llm,
    })
    context = final["context"]
    return ChatValidationGraphResult(
        initial_answer=str(final.get("answer", "")),
        context=context,
        evidence=final.get("evidence", []) or [],
        intent=str(final.get("intent", "concept")),
        validation=final.get("validation"),
    )


async def run_chat_graph(
    *,
    store: DocumentStore,
    request: ChatRequest,
    settings=None,
    section_title: str = "",
    memory_context: str = "",
    memory_hints: dict | None = None,
    llm=None,
) -> ChatGraphResult:
    llm = _configured_llm(settings, llm=llm)
    graph = build_chat_graph(deep_mode=request.deep_mode)
    final = await graph.ainvoke({
        "request": request,
        "settings": settings,
        "store": store,
        "section_title": section_title,
        "memory_context": memory_context,
        "memory_hints": memory_hints or {},
        "llm": llm,
    })
    return ChatGraphResult(
        content=str(final.get("answer", "")),
        evidence=final.get("evidence", []) or [],
        intent=str(final.get("intent", "concept")),
        validation=final.get("validation"),
        deep_mode=request.deep_mode,
    )
