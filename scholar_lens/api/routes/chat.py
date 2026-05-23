from __future__ import annotations

import json
import logging
import threading

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from scholar_lens.agents.state import ScholarLensState
from scholar_lens.agents.tutor import LearningTutorAgent
from scholar_lens.api.deps import get_memory_manager, get_settings
from scholar_lens.api.schemas import ChatRequest, ChatMessage
from scholar_lens.core.circuit_breaker import CircuitBreaker
from scholar_lens.core.exceptions import CircuitOpenError

logger = logging.getLogger(__name__)
router = APIRouter()

_tutor: LearningTutorAgent | None = None
_tutor_lock = threading.Lock()
_loader = None
_loader_lock = threading.Lock()
_api_circuit_breaker = CircuitBreaker(name="llm-api", cooldown_seconds=30.0)


def _get_tutor() -> LearningTutorAgent:
    global _tutor
    if _tutor is None:
        with _tutor_lock:
            if _tutor is None:
                settings = get_settings()
                from scholar_lens.core.llm_factory import ChatLLMFactory
                factory = ChatLLMFactory.from_settings(settings)
                llm = factory.create(streaming=True)
                memory = get_memory_manager()
                _tutor = LearningTutorAgent(llm=llm, core_memory_context=memory.get_core_context())
    return _tutor


def _get_loader():
    global _loader
    if _loader is None:
        with _loader_lock:
            if _loader is None:
                from scholar_lens.rag.layered_loader import LayeredLoader
                _loader = LayeredLoader()
    return _loader


@router.post("")
async def chat(request: ChatRequest):
    if not await _api_circuit_breaker.allow_request():
        return ChatMessage(role="assistant", content="Service temporarily unavailable. Please try again later.")

    try:
        tutor = _get_tutor()

        # P0.3: try layered loading first
        loader = _get_loader()
        layered_context = ""
        layer_used = "L2"
        if request.doc_id:
            content, layer_used = loader.resolve(section_id=request.section_id, need_detail=False)
            if content:
                layered_context = f"[{layer_used}] {content[:2000]}"

        state = ScholarLensState(
            doc_id=request.doc_id,
            section_id=request.section_id,
            messages=[
                {"role": "user", "content": request.message},
            ],
        )

        if layered_context:
            state.retrieved_chunks = [{"section_id": request.section_id, "text": layered_context}]

        result = await tutor.respond(state)

        if result.error:
            return ChatMessage(role="assistant", content=f"Error: {result.error}")

        await _api_circuit_breaker.record_success()
        last_msg = result.messages[-1]["content"] if result.messages else ""
        return ChatMessage(role="assistant", content=last_msg, timestamp="")

    except Exception:
        await _api_circuit_breaker.record_failure()
        raise


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming chat with typed events: status, token, evidence, done, error."""
    if not await _api_circuit_breaker.allow_request():
        raise CircuitOpenError("llm-api", _api_circuit_breaker)

    settings = get_settings()
    from scholar_lens.rag.document_store import DocumentStore
    from scholar_lens.rag.document_index import DocumentIndex, evidence_from_results
    from scholar_lens.api.schemas import DocumentStatus

    store = DocumentStore()
    index = DocumentIndex(store)

    def _fallback_answer(evidence: list[dict]) -> str:
        if evidence:
            source = evidence[0].get("quote", "").strip()
            if source:
                return (
                    "当前未配置可用的 LLM，因此先返回基于检索证据的简要提示：\n\n"
                    f"最相关片段：{source[:240]}\n\n"
                    "请在设置中配置 API Key 和模型后，我可以生成更完整的中文讲解。"
                )
        return "当前未配置可用的 LLM，且没有检索到可用文档证据。请先确认文档已解析完成并配置模型。"

    async def event_stream():
        try:
            # Validate document
            if request.doc_id:
                doc = store.get(request.doc_id)
                if doc is None:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Document not found'})}\n\n"
                    return
                if doc.status != DocumentStatus.ready:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Document status: {doc.status.value}'})}\n\n"
                    return

            # Retrieve evidence
            yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieving', 'message': '正在检索文档证据...'})}\n\n"
            results = index.search(request.doc_id, request.message, request.section_id, top_k=5)
            evidence = evidence_from_results(results)

            ctx_parts = []
            for i, r in enumerate(results):
                ctx_parts.append(f"[{i + 1}] {r.text[:300]}")
            ctx = "\n\n".join(ctx_parts) if ctx_parts else "No relevant content found."

            # Stream generation
            yield f"data: {json.dumps({'type': 'status', 'stage': 'generating', 'message': '正在生成回答...'})}\n\n"
            if not settings.llm.api_key or not settings.llm.model:
                full = _fallback_answer(evidence)
                yield f"data: {json.dumps({'type': 'token', 'token': full})}\n\n"
                yield f"data: {json.dumps({'type': 'evidence', 'items': evidence})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'full': full})}\n\n"
                return

            from langchain_core.messages import HumanMessage, SystemMessage
            from scholar_lens.agents.prompts import EXPLAINER_SYSTEM
            from scholar_lens.core.llm_factory import ChatLLMFactory

            factory = ChatLLMFactory.from_settings(settings)
            llm = factory.create(streaming=True)
            # Include section context
            section_context = ""
            if request.section_id:
                sections = store.load_sections(request.doc_id)
                sec = next((s for s in sections if s.section_id == request.section_id), None)
                if sec:
                    section_context = f"Current section: {sec.title}\n"

            system_msg = SystemMessage(content=EXPLAINER_SYSTEM)
            user_msg = HumanMessage(content=f"""{section_context}Evidence from the document:
{ctx}

Student question: {request.message}

Answer in Chinese. Preserve key English terms inline when needed, but do not append a glossary, related terms, vocabulary list, or extra terminology section unless the student explicitly asks for one. Use only the evidence items that directly support your answer. Number citations using the provided evidence numbers, for example [1]. Do not cite evidence you did not use. If evidence is insufficient, say so.""")
            full = ""
            async for chunk in llm.astream([system_msg, user_msg]):
                if hasattr(chunk, 'content') and chunk.content:
                    full += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk.content})}\n\n"

            await _api_circuit_breaker.record_success()
            yield f"data: {json.dumps({'type': 'evidence', 'items': evidence})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'full': full[:5000]})}\n\n"
        except GeneratorExit:
            pass
        except Exception as e:
            await _api_circuit_breaker.record_failure()
            logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/explain")
async def explain_text(request: ChatRequest):
    """Explain or translate selected text (mode=explain or translate)."""
    if request.mode not in ("explain", "translate") or not request.message:
        return ChatMessage(role="assistant", content="No text selected for explanation/translation.")

    if not await _api_circuit_breaker.allow_request():
        return ChatMessage(role="assistant", content="Service temporarily unavailable. Please try again later.")

    settings = get_settings()
    if not settings.llm.api_key or not settings.llm.model:
        if request.mode == "translate":
            return ChatMessage(
                role="assistant",
                content=(
                    "当前未配置可用的 LLM。以下是原文摘录，配置模型后可生成中文翻译：\n\n"
                    f"{request.message[:1000]}"
                ),
            )
        return ChatMessage(
            role="assistant",
            content=(
                "当前未配置可用的 LLM。选中文本如下，可先作为阅读摘录保存：\n\n"
                f"{request.message[:1000]}"
            ),
        )

    from langchain_core.messages import HumanMessage, SystemMessage
    from scholar_lens.core.llm_factory import ChatLLMFactory

    if request.mode == "translate":
        system_prompt = (
            "You are a bilingual academic translator. Translate the given text into clear Chinese. "
            "Preserve model names, dataset names, method names, acronyms, formulas, and domain-specific terms in English when Chinese would lose precision. "
            "For important technical terms, use 'English term（中文解释）' on first mention when helpful, then keep the English term. "
            "Only output the translation. Do not output related terms, glossary, notes, or explanations."
        )
        user_prompt = f"Translate this academic text to Chinese under the preservation rules:\n\n{request.message}"
    else:
        system_prompt = (
            "You are a bilingual academic content explainer. Explain the given text in Chinese for a university student. "
            "Use concise paragraphs and preserve key English terms. Do not add unrelated terms."
        )
        user_prompt = f"Explain this in Chinese:\n\n{request.message}"

    try:
        factory = ChatLLMFactory.from_settings(settings)
        llm = factory.create(streaming=False)
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        await _api_circuit_breaker.record_success()
        return ChatMessage(role="assistant", content=response.content)
    except Exception:
        await _api_circuit_breaker.record_failure()
        return ChatMessage(role="assistant", content="Explanation failed. Please try again.")
