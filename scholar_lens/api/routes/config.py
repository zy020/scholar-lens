from __future__ import annotations

from fastapi import APIRouter

from scholar_lens.api.schemas import ConfigUpdateRequest, ConfigResponse

router = APIRouter()


@router.get("", response_model=ConfigResponse)
async def get_config():
    return ConfigResponse(
        llm_model="not configured",
        embedding_model="not configured",
        reranker_available=False,
        vision_available=False,
        status="not_configured",
    )


@router.put("")
async def update_config(request: ConfigUpdateRequest):
    return ConfigResponse(
        llm_model=request.llm_model,
        embedding_model=request.embedding_model,
        reranker_available=request.reranker_model is not None,
        vision_available=request.vision_api_key is not None,
        status="configured",
    )


@router.post("/test")
async def test_connection():
    return {"status": "not_implemented"}
