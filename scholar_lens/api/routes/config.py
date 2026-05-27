from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from dotenv import set_key

from scholar_lens.api.deps import get_settings
from scholar_lens.api.schemas import ConfigResponse, ConfigUpdateRequest
from scholar_lens.core.settings import RerankerConfig, VisionConfig

router = APIRouter()
ENV_PATH = Path.cwd() / ".env"


def _value(value: str | None) -> str:
    return value or ""


def _env_bool(value: bool) -> str:
    return "true" if value else "false"


def _persist_settings_to_env(settings, env_path: Path | None = None) -> None:
    env_path = env_path or ENV_PATH
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")

    values = {
        "API_KEY": settings.api_key or "",
        "BASE_URL": settings.base_url or "",
        "LLM__API_KEY": settings.llm.api_key or "",
        "LLM__BASE_URL": settings.llm.base_url or "",
        "LLM__MODEL": settings.llm.model or "",
        "LLM__TEMPERATURE": str(settings.llm.temperature),
        "EMBEDDING__API_KEY": settings.embedding.api_key or "",
        "EMBEDDING__BASE_URL": settings.embedding.base_url or "",
        "EMBEDDING__MODEL": settings.embedding.model or "",
        "RERANKER__API_KEY": settings.reranker.api_key if settings.reranker else "",
        "RERANKER__BASE_URL": settings.reranker.base_url if settings.reranker else "",
        "RERANKER__MODEL": settings.reranker.model if settings.reranker else "",
        "RERANKER_USE_SEPARATE": _env_bool(bool(getattr(settings, "reranker_use_separate", False))),
        "VISION__API_KEY": settings.vision.api_key if settings.vision else "",
        "VISION__BASE_URL": settings.vision.base_url if settings.vision else "",
        "VISION__MODEL": settings.vision.model if settings.vision else "",
        "VISION_USE_SEPARATE": _env_bool(bool(getattr(settings, "vision_use_separate", False))),
        "AUTO_OCR_ENABLED": _env_bool(bool(getattr(settings, "auto_ocr_enabled", True))),
        "LLM_QUALITY_ENABLED": _env_bool(bool(getattr(settings, "llm_quality_enabled", False))),
        "VISION_ENHANCEMENT_ENABLED": _env_bool(bool(getattr(settings, "vision_enhancement_enabled", False))),
        "MEMORY_LLM_COMPRESSION_ENABLED": _env_bool(bool(getattr(settings, "memory_llm_compression_enabled", False))),
    }
    for key, value in values.items():
        set_key(str(env_path), key, value, quote_mode="always")


def _config_response(settings, requires_restart: bool = False) -> ConfigResponse:
    model_reranker_ready = bool(
        settings.reranker is not None
        and settings.reranker.api_key
        and settings.reranker.base_url
        and settings.reranker.model
    )
    vision_ready = bool(settings.vision_api_key and settings.vision_base_url and settings.vision_model)
    return ConfigResponse(
        llm_model=settings.llm_model or "not configured",
        llm_base_url=settings.llm_base_url or "",
        llm_configured=bool(settings.llm_api_key and settings.llm_model),
        embedding_model=settings.embedding_model or "not configured",
        embedding_base_url=settings.embedding_base_url or "",
        embedding_configured=bool(settings.embedding_api_key and settings.embedding_model),
        reranker_available=model_reranker_ready,
        reranker_model=settings.reranker_model or "",
        reranker_base_url=settings.reranker_base_url or "",
        reranker_active=model_reranker_ready,
        reranker_mode="model" if model_reranker_ready else "rule",
        reranker_use_separate=bool(getattr(settings, "reranker_use_separate", False)),
        vision_available=vision_ready,
        vision_model=settings.vision_model or "",
        vision_base_url=settings.vision_base_url or "",
        vision_use_separate=bool(getattr(settings, "vision_use_separate", False)),
        auto_ocr_enabled=bool(getattr(settings, "auto_ocr_enabled", True)),
        llm_quality_enabled=bool(getattr(settings, "llm_quality_enabled", False)),
        vision_enhancement_enabled=bool(getattr(settings, "vision_enhancement_enabled", False)),
        memory_llm_compression_enabled=bool(getattr(settings, "memory_llm_compression_enabled", False)),
        status="configured" if settings.llm_api_key else "not_configured",
        requires_restart=requires_restart,
    )


@router.get("", response_model=ConfigResponse)
async def get_config():
    settings = get_settings()
    return _config_response(settings)


@router.post("", response_model=ConfigResponse)
async def update_config(request: ConfigUpdateRequest):
    settings = get_settings()
    fields_set = request.model_fields_set

    def provided(name: str) -> bool:
        return name in fields_set

    shared_key = request.api_key if provided("api_key") else settings.api_key
    shared_base = request.base_url if provided("base_url") else settings.base_url

    # Update shared credentials
    if provided("api_key"):
        object.__setattr__(settings, "api_key", request.api_key)
    if provided("base_url"):
        object.__setattr__(settings, "base_url", request.base_url)

    # LLM — always present
    llm_use_separate = bool(request.llm_use_separate)
    llm_key = request.llm_api_key if llm_use_separate and provided("llm_api_key") else shared_key
    llm_base = request.llm_base_url if llm_use_separate and provided("llm_base_url") else shared_base
    if provided("llm_api_key") or provided("api_key"):
        settings.llm.api_key = llm_key
    if provided("llm_base_url") or provided("base_url"):
        settings.llm.base_url = llm_base
    if provided("llm_model"):
        settings.llm.model = request.llm_model
    if request.llm_temperature is not None:
        settings.llm.temperature = request.llm_temperature

    # Embedding — always present
    embedding_use_separate = bool(request.embedding_use_separate)
    emb_key = request.embedding_api_key if embedding_use_separate and provided("embedding_api_key") else shared_key
    emb_base = request.embedding_base_url if embedding_use_separate and provided("embedding_base_url") else shared_base
    if provided("embedding_api_key") or provided("api_key"):
        settings.embedding.api_key = emb_key
    if provided("embedding_base_url") or provided("base_url"):
        settings.embedding.base_url = emb_base
    if provided("embedding_model"):
        settings.embedding.model = request.embedding_model

    # Reranker — enabled/disabled
    if request.reranker_enabled is False:
        settings.reranker = None
        settings.reranker_use_separate = False
    elif request.reranker_model:
        reranker_use_separate = bool(request.reranker_use_separate)
        settings.reranker_use_separate = reranker_use_separate
        reranker_key = request.reranker_api_key if reranker_use_separate and provided("reranker_api_key") else shared_key
        reranker_base = request.reranker_base_url if reranker_use_separate and provided("reranker_base_url") else shared_base
        settings.reranker = RerankerConfig(
            model=request.reranker_model,
            api_key=_value(reranker_key),
            base_url=_value(reranker_base),
        )

    # Vision — enabled/disabled
    if request.vision_enabled is False:
        settings.vision = None
        settings.vision_use_separate = False
    elif request.vision_model:
        vision_use_separate = bool(request.vision_use_separate)
        settings.vision_use_separate = vision_use_separate
        vision_key = request.vision_api_key if vision_use_separate and provided("vision_api_key") else shared_key
        vision_base = request.vision_base_url if vision_use_separate and provided("vision_base_url") else shared_base
        settings.vision = VisionConfig(
            model=request.vision_model,
            api_key=_value(vision_key),
            base_url=_value(vision_base),
        )

    if request.llm_quality_enabled is not None:
        settings.llm_quality_enabled = bool(request.llm_quality_enabled)
    if request.vision_enhancement_enabled is not None:
        settings.vision_enhancement_enabled = bool(request.vision_enhancement_enabled)
    if request.memory_llm_compression_enabled is not None:
        settings.memory_llm_compression_enabled = bool(request.memory_llm_compression_enabled)

    from scholar_lens.api.routes.chat import reset_chat_runtime

    _persist_settings_to_env(settings)
    reset_chat_runtime()
    return _config_response(settings, requires_restart=False)


@router.post("/test")
async def test_connection():
    settings = get_settings()
    ok = bool(settings.llm_api_key)
    return {"status": "ok" if ok else "not_configured"}
