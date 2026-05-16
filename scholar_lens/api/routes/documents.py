from __future__ import annotations

from fastapi import APIRouter, UploadFile, File

from scholar_lens.api.schemas import DocumentUploadResponse

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    return DocumentUploadResponse(
        doc_id="pending",
        doc_type="unknown",
        num_sections=0,
        status="processing",
    )


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    return {"doc_id": doc_id, "status": "not_implemented"}
