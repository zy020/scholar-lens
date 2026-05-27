from scholar_lens.api.main import create_app
from tests.unit.api.helpers import ASGITestClient


class TestApp:
    def test_create_app(self):
        app = create_app()
        assert app is not None

    def test_health_check(self):
        app = create_app()
        client = ASGITestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_logging_config_includes_graph_logger(self):
        from scholar_lens.logging_config import GRAPH_LOGGER_NAME, STRUCTURED_LOGGER_NAMES

        assert GRAPH_LOGGER_NAME == "scholar_lens.graph"
        assert GRAPH_LOGGER_NAME in STRUCTURED_LOGGER_NAMES

    def test_cors_headers(self):
        app = create_app()
        client = ASGITestClient(app)
        response = client.options("/api/config", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        assert response.status_code in (200, 204)
