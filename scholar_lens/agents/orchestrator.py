from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from scholar_lens.agents.state import ScholarLensState
from scholar_lens.agents.doc_analyzer import DocumentAnalyzerAgent
from scholar_lens.agents.explainer import ContentExplainerAgent
from scholar_lens.agents.validator import ValidatorAgent
from scholar_lens.agents.tutor import LearningTutorAgent

logger = logging.getLogger(__name__)


class ScholarLensOrchestrator:
    """LangGraph orchestration for ScholarLens agent pipeline.

    Two graphs:
    1. Pipeline graph: upload → analyze → END (one-shot document processing)
       Only runs the Document Analyzer. Explanations are lazy/on-demand.
    2. Tutor graph: tutor → explainer → validator → tutor (interactive loop)
    """

    def __init__(
        self,
        doc_analyzer: DocumentAnalyzerAgent | None = None,
        explainer: ContentExplainerAgent | None = None,
        validator: ValidatorAgent | None = None,
        tutor: LearningTutorAgent | None = None,
    ) -> None:
        self.doc_analyzer = doc_analyzer or DocumentAnalyzerAgent()
        self.explainer = explainer or ContentExplainerAgent()
        self.validator = validator or ValidatorAgent()
        self.tutor = tutor or LearningTutorAgent()

    def build_pipeline_graph(self) -> StateGraph:
        """Build the document processing pipeline graph.

        Flow: analyze → END

        The pipeline only extracts document structure, concepts, and
        L0/L1 summaries. Explanations are generated lazily through
        the tutor graph when the student requests them.
        """
        graph = StateGraph(ScholarLensState)

        graph.add_node("analyze", self.doc_analyzer.analyze)

        graph.set_entry_point("analyze")
        graph.add_edge("analyze", END)

        return graph

    def build_tutor_graph(self) -> StateGraph:
        """Build the interactive tutor graph.

        Flow: tutor → (conditional: needs_explanation? → explainer → validator → tutor | END)
        """
        graph = StateGraph(ScholarLensState)

        graph.add_node("tutor", self.tutor.respond)
        graph.add_node("explainer", self.explainer.explain)
        graph.add_node("validator", self.validator.validate)

        graph.set_entry_point("tutor")

        def route_after_tutor(state: ScholarLensState) -> str:
            if state.explanation_request and not state.explanation_result:
                return "explainer"
            return END

        def route_after_validator(state: ScholarLensState) -> str:
            return "tutor"

        graph.add_conditional_edges(
            "tutor", route_after_tutor,
            {"explainer": "explainer", END: END},
        )
        graph.add_edge("explainer", "validator")
        graph.add_edge("validator", "tutor")

        return graph
