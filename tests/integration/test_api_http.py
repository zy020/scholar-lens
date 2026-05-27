"""API HTTP integration tests.

Tests FastAPI endpoints via HTTP requests.
Requires server running: python -m uvicorn scholar_lens.api.main:create_app --factory
"""

import os
import pytest
import httpx


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@pytest.mark.skipif(not os.getenv("API_TEST"), reason="Set API_TEST=1 + start uvicorn server")
class TestAPIHTTP:
    """HTTP-level API tests."""

    async def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=BASE_URL, timeout=30)

    @pytest.mark.asyncio
    async def test_health_check(self):
        async with await self._client() as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
            print("  ✅ Health check")

    @pytest.mark.asyncio
    async def test_get_config(self):
        async with await self._client() as client:
            resp = await client.get("/api/config")
            assert resp.status_code == 200
            data = resp.json()
            assert "llm_model" in data
            assert "status" in data
            print(f"  ✅ Config: {data['status']}")

    @pytest.mark.asyncio
    async def test_chat_endpoint(self):
        async with await self._client() as client:
            resp = await client.post("/api/chat", json={
                "message": "What is deep learning?",
                "doc_id": "test",
                "section_id": "1",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["role"] == "assistant"
            print(f"  ✅ Chat: {data['content'][:80]}...")

    @pytest.mark.asyncio
    async def test_notes_endpoint(self):
        async with await self._client() as client:
            resp = await client.get("/api/notes/test_doc")
            assert resp.status_code == 200
            data = resp.json()
            assert data["doc_id"] == "test_doc"
            print(f"  ✅ Notes: {len(data.get('terms', []))} terms")
