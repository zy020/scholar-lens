from __future__ import annotations

from fastapi import APIRouter, Query

from scholar_lens.api.deps import get_memory_manager

router = APIRouter()


@router.get("")
async def get_memory_snapshot(doc_id: str = "", limit: int = Query(default=30, ge=1, le=100)):
    return await get_memory_manager().get_snapshot(doc_id=doc_id, limit=limit)


@router.delete("/session")
async def clear_session_memory():
    await get_memory_manager().clear_session_memory()
    return {"status": "cleared", "scope": "session"}


@router.delete("/document")
async def clear_document_memory(doc_id: str):
    await get_memory_manager().clear_document_memory(doc_id)
    return {"status": "cleared", "scope": "document", "doc_id": doc_id}


@router.delete("/all")
async def clear_all_memory():
    await get_memory_manager().clear_all_memory()
    return {"status": "cleared", "scope": "all"}
