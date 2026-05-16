import pytest
from fastapi.testclient import TestClient
from scholar_lens.api.main import create_app


class TestApp:
    def test_create_app(self):
        app = create_app()
        assert app is not None

    def test_health_check(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_cors_headers(self):
        app = create_app()
        client = TestClient(app)
        response = client.options("/api/config", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        assert response.status_code in (200, 204)
