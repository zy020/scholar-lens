import pytest


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FlakyLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary api failure")
        return FakeResponse("ok")


async def test_trace_graph_node_logs_success(monkeypatch):
    from scholar_lens.core import graph_utils
    from scholar_lens.core.graph_utils import trace_graph_node

    logged = []
    monkeypatch.setattr(graph_utils.logger, "info", lambda message, extra=None: logged.append((message, extra)))

    async def node():
        return {"status": "ok"}

    result = await trace_graph_node("brief", "generate", doc_id="doc1", func=node)

    assert result["status"] == "ok"
    assert logged[0][0] == "graph node completed"
    assert logged[0][1]["graph_node"] == "brief.generate"


async def test_trace_graph_node_wraps_failure_with_node_name(monkeypatch):
    from scholar_lens.core import graph_utils
    from scholar_lens.core.graph_utils import GraphNodeError, trace_graph_node

    logged = []
    monkeypatch.setattr(
        graph_utils.logger,
        "warning",
        lambda message, extra=None, exc_info=False: logged.append((message, extra, exc_info)),
    )

    async def node():
        raise ValueError("bad json")

    with pytest.raises(GraphNodeError) as exc:
        await trace_graph_node("brief", "parse", doc_id="doc1", func=node)

    assert "brief.parse failed: bad json" in str(exc.value)
    assert exc.value.graph_name == "brief"
    assert exc.value.node_name == "parse"
    assert logged[0][0] == "graph node failed"
    assert logged[0][1]["graph_node"] == "brief.parse"


async def test_invoke_llm_with_retries_recovers_from_transient_error(monkeypatch):
    from scholar_lens.core import graph_utils
    from scholar_lens.core.graph_utils import invoke_llm_with_retries

    llm = FlakyLLM()
    logged = []
    monkeypatch.setattr(graph_utils.logger, "warning", lambda message, extra=None: logged.append((message, extra)))

    response = await invoke_llm_with_retries(
        llm,
        ["message"],
        graph_name="brief",
        node_name="generate",
        attempts=2,
    )

    assert response.content == "ok"
    assert llm.calls == 2
    assert logged[0][0] == "LLM node call failed"
    assert logged[0][1]["graph_node"] == "brief.generate"
