from __future__ import annotations

import hashlib
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from scholar_lens.api.chat_graph import (
    build_revision_messages,
    run_chat_graph,
    run_chat_retrieval_graph,
    run_chat_validation_graph,
)
from scholar_lens.api.chat_service import build_chat_messages, build_no_llm_answer, configured_llm_configs
from scholar_lens.api.deps import get_document_store, get_memory_manager, get_settings
from scholar_lens.api.memory_events import record_memory_event
from scholar_lens.api.schemas import ChatRequest, ChatMessage, DocumentStatus
from scholar_lens.core.circuit_breaker import CircuitBreaker
from scholar_lens.core.exceptions import CircuitOpenError

logger = logging.getLogger(__name__)
router = APIRouter()

_runtime_generation = 0
_api_circuit_breaker = CircuitBreaker(name="llm-api", cooldown_seconds=30.0)
TRANSLATION_CACHE_PROMPT_VERSION = "translate-section-v1"


async def _stream_llm_tokens_with_initial_retry(llm_factory, messages, attempts: int = 2):
    last_error: Exception | None = None
    total_attempts = max(1, attempts)
    for attempt in range(total_attempts):
        llm = llm_factory()
        emitted = False
        try:
            async for chunk in llm.astream(messages):
                token = getattr(chunk, "content", "")
                if token:
                    emitted = True
                    yield token
            return
        except Exception as exc:
            last_error = exc
            if emitted or attempt >= total_attempts - 1:
                raise
            logger.warning("Streaming LLM failed before first token; retrying", exc_info=True)
    if last_error is not None:
        raise last_error


def reset_chat_runtime() -> None:
    global _runtime_generation
    _runtime_generation += 1


def _document_error(store, doc_id: str) -> str:
    if not doc_id:
        return ""
    doc = store.get(doc_id)
    if doc is None:
        return "Document not found"
    if doc.status != DocumentStatus.ready:
        return f"Document status: {doc.status.value}"
    return ""


def _section_title(store, doc_id: str, section_id: str) -> str:
    if not doc_id or not section_id:
        return ""
    sections = store.load_sections(doc_id)
    section = next((s for s in sections if s.section_id == section_id), None)
    return section.title if section else ""


def _translation_cache_key(request: ChatRequest, model: str) -> str:
    text_hash = hashlib.sha256(request.message.encode("utf-8")).hexdigest()
    parts = [
        request.doc_id or "",
        request.section_id or "",
        request.mode,
        model or "",
        TRANSLATION_CACHE_PROMPT_VERSION,
        text_hash,
    ]
    return "|".join(parts)


def _translation_cache_path(doc_id: str):
    if not doc_id:
        return None
    try:
        store = get_document_store()
        if store.get(doc_id) is None:
            return None
        return store.document_dir(doc_id) / "translation_cache.json"
    except Exception:
        logger.warning("Translation cache path unavailable", exc_info=True)
        return None


def _load_translation_cache(doc_id: str) -> dict:
    path = _translation_cache_path(doc_id)
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Translation cache read failed", exc_info=True)
        return {}


def _save_translation_cache(doc_id: str, cache: dict) -> None:
    path = _translation_cache_path(doc_id)
    if path is None:
        return
    try:
        path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.warning("Translation cache write failed", exc_info=True)


async def _memory_context(doc_id: str = "") -> str:
    try:
        return await get_memory_manager().get_personalization_context(doc_id=doc_id)
    except Exception:
        logger.warning("Memory context unavailable", exc_info=True)
        return ""


async def _memory_retrieval_hints(doc_id: str = "") -> dict:
    try:
        return await get_memory_manager().get_retrieval_hints(doc_id=doc_id)
    except Exception:
        logger.warning("Memory retrieval hints unavailable", exc_info=True)
        return {}


async def _record_memory_event(event_type: str, request: ChatRequest, payload: dict | None = None) -> None:
    await record_memory_event(
        get_memory_manager(),
        event_type,
        doc_id=request.doc_id,
        section_id=request.section_id,
        payload=payload or {},
    )


@router.post("")
async def chat(request: ChatRequest):
    if not await _api_circuit_breaker.allow_request():
        return ChatMessage(role="assistant", content="Service temporarily unavailable. Please try again later.")

    try:
        settings = get_settings()
        store = get_document_store()
        error = _document_error(store, request.doc_id)
        if error:
            return ChatMessage(role="assistant", content=error)
        await _record_memory_event(
            "chat_question",
            request,
            {"message": request.message[:500], "mode": request.mode, "student_level": request.student_level},
        )

        result = await run_chat_graph(
            store=store,
            request=request,
            settings=settings,
            section_title=_section_title(store, request.doc_id, request.section_id),
            memory_context=await _memory_context(request.doc_id),
            memory_hints=await _memory_retrieval_hints(request.doc_id),
        )
        await _api_circuit_breaker.record_success()
        return ChatMessage(role="assistant", content=result.content, evidence=result.evidence)

    except Exception:
        await _api_circuit_breaker.record_failure()
        raise


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming chat with typed events: status, token, evidence, done, error."""
    if not await _api_circuit_breaker.allow_request():
        raise CircuitOpenError("llm-api", _api_circuit_breaker)

    settings = get_settings()
    store = get_document_store()

    async def event_stream():
        try:
            error = _document_error(store, request.doc_id)
            if error:
                yield f"data: {json.dumps({'type': 'error', 'message': error})}\n\n"
                return
            await _record_memory_event(
                "chat_question",
                request,
                {"message": request.message[:500], "mode": request.mode, "student_level": request.student_level},
            )

            # Retrieve evidence through the LangGraph chat pipeline.
            yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieving', 'message': '正在检索文档证据...'})}\n\n"
            if request.deep_mode:
                yield f"data: {json.dumps({'type': 'status', 'stage': 'graph', 'message': '深度模式：正在编排检索、回答与校验...'})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'stage': 'intent', 'message': '深度模式：准备识别问题意图...'})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieve', 'message': '深度模式：准备扩展检索上下文...'})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'stage': 'draft', 'message': '深度模式：准备生成初始回答...'})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'stage': 'validate', 'message': '深度模式：准备校验证据一致性...'})}\n\n"
                result = await run_chat_validation_graph(
                    store=store,
                    request=request,
                    settings=settings,
                    section_title=_section_title(store, request.doc_id, request.section_id),
                    memory_context=await _memory_context(request.doc_id),
                    memory_hints=await _memory_retrieval_hints(request.doc_id),
                )
                validation = result.validation or {"passed": True, "issues": [], "correction": ""}
                if validation.get("passed", True):
                    await _api_circuit_breaker.record_success()
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'generating', 'message': '深度模式：校验通过，正在返回回答...'})}\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'token': result.initial_answer})}\n\n"
                    yield f"data: {json.dumps({'type': 'evidence', 'items': result.evidence})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'full': result.initial_answer[:5000]})}\n\n"
                    return

                llm_configs = configured_llm_configs(settings)
                fallback = str(validation.get("correction") or result.initial_answer)
                if not llm_configs:
                    await _api_circuit_breaker.record_success()
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'generating', 'message': '深度模式：校验未通过，当前未配置模型，返回校验建议...'})}\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'token': fallback})}\n\n"
                    yield f"data: {json.dumps({'type': 'evidence', 'items': result.evidence})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'full': fallback[:5000]})}\n\n"
                    return

                from scholar_lens.core.llm_factory import ChatLLMFactory

                yield f"data: {json.dumps({'type': 'status', 'stage': 'revise', 'message': '深度模式：校验发现可改进点，正在流式修订回答...'})}\n\n"
                revision_messages = build_revision_messages(
                    answer=result.initial_answer,
                    validation=validation,
                    context=result.context,
                    request=request,
                )
                full = ""
                async for token in _stream_llm_tokens_with_initial_retry(
                    lambda: ChatLLMFactory.from_settings(settings).create(config=llm_configs[0], streaming=True),
                    revision_messages,
                ):
                    full += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                if not full.strip():
                    full = fallback
                    yield f"data: {json.dumps({'type': 'token', 'token': full})}\n\n"
                await _api_circuit_breaker.record_success()
                yield f"data: {json.dumps({'type': 'evidence', 'items': result.evidence})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'full': full[:5000]})}\n\n"
                return

            retrieval = await run_chat_retrieval_graph(
                store=store,
                request=request,
                settings=settings,
                memory_hints=await _memory_retrieval_hints(request.doc_id),
            )
            yield f"data: {json.dumps({'type': 'status', 'stage': 'generating', 'message': '正在生成回答...'})}\n\n"
            llm_configs = configured_llm_configs(settings)
            if not llm_configs:
                full = build_no_llm_answer(retrieval.evidence)
                await _api_circuit_breaker.record_success()
                yield f"data: {json.dumps({'type': 'token', 'token': full})}\n\n"
                yield f"data: {json.dumps({'type': 'evidence', 'items': retrieval.evidence})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'full': full})}\n\n"
                return

            from scholar_lens.core.llm_factory import ChatLLMFactory

            llm = ChatLLMFactory.from_settings(settings).create(config=llm_configs[0], streaming=True)
            messages = build_chat_messages(
                question=request.message,
                context_text=retrieval.context.context_text,
                section_title=_section_title(store, request.doc_id, request.section_id),
                has_formula_evidence=retrieval.context.has_formula_evidence,
                memory_context=await _memory_context(request.doc_id),
                student_level=request.student_level,
            )
            full = ""
            async for chunk in llm.astream(messages):
                token = getattr(chunk, "content", "")
                if token:
                    full += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            await _api_circuit_breaker.record_success()
            yield f"data: {json.dumps({'type': 'evidence', 'items': retrieval.evidence})}\n\n"
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
    await _record_memory_event(
        f"{request.mode}_text",
        request,
        {"text_preview": request.message[:500]},
    )
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
        cache_key = ""
        if request.mode == "translate" and request.doc_id and request.section_id:
            cache_key = _translation_cache_key(request, settings.llm.model)
            cached = _load_translation_cache(request.doc_id).get(cache_key)
            if isinstance(cached, dict) and cached.get("content"):
                return ChatMessage(role="assistant", content=str(cached["content"]))

        factory = ChatLLMFactory.from_settings(settings)
        llm = factory.create(streaming=False)
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        await _api_circuit_breaker.record_success()
        content = str(response.content)
        if cache_key:
            cache = _load_translation_cache(request.doc_id)
            cache[cache_key] = {
                "doc_id": request.doc_id,
                "section_id": request.section_id,
                "mode": request.mode,
                "model": settings.llm.model,
                "prompt_version": TRANSLATION_CACHE_PROMPT_VERSION,
                "content": content,
            }
            _save_translation_cache(request.doc_id, cache)
        return ChatMessage(role="assistant", content=content)
    except Exception:
        await _api_circuit_breaker.record_failure()
        return ChatMessage(role="assistant", content="Explanation failed. Please try again.")
