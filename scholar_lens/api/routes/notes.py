from __future__ import annotations

from fastapi import APIRouter

from scholar_lens.api.schemas import NotesResponse

router = APIRouter()


@router.get("/{doc_id}", response_model=NotesResponse)
async def get_notes(doc_id: str):
    return NotesResponse(doc_id=doc_id)
