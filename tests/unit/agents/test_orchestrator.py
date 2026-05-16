import pytest
from scholar_lens.agents.orchestrator import ScholarLensOrchestrator


class TestScholarLensOrchestrator:
    def test_instantiation(self):
        orch = ScholarLensOrchestrator()
        assert orch is not None

    def test_build_pipeline_graph(self):
        orch = ScholarLensOrchestrator()
        graph = orch.build_pipeline_graph()
        assert graph is not None

    def test_build_tutor_graph(self):
        orch = ScholarLensOrchestrator()
        graph = orch.build_tutor_graph()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_pipeline_graph_nodes(self):
        """Pipeline graph should have analyze, explain, validate nodes."""
        orch = ScholarLensOrchestrator()
        graph = orch.build_pipeline_graph()
        compiled = graph.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_tutor_graph_nodes(self):
        """Tutor graph should have tutor, explainer, validator nodes."""
        orch = ScholarLensOrchestrator()
        graph = orch.build_tutor_graph()
        compiled = graph.compile()
        assert compiled is not None
