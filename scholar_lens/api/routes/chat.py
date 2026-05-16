from __future__ import annotations

from fastapi import APIRouter

from scholar_lens.api.schemas import ChatRequest, ChatMessage

router = APIRouter()


@router.post("")
async def chat(request: ChatRequest):
    return ChatMessage(
        role="assistant",
        content="Tutor not configured yet.",
    )
