from scholar_lens.rag.document_store import DocumentStore
from scholar_lens.rag.document_index import DocumentIndex
from scholar_lens.rag.layered_loader import LayeredLoader
from scholar_lens.rag.retriever import HybridRetriever, RetrievalResult

__all__ = [
    "BaseReranker",
    "ContextualRetriever",
    "DiversityReranker",
    "DocumentIndex",
    "DocumentStore",
    "HybridRetriever",
    "LayeredLoader",
    "ModelReranker",
    "RerankerPipeline",
    "RetrievalResult",
    "RuleReranker",
    "ScholarVectorStore",
]


def __getattr__(name: str):
    if name == "ContextualRetriever":
        from scholar_lens.rag.contextual_retrieval import ContextualRetriever
        return ContextualRetriever
    if name in {"BaseReranker", "DiversityReranker", "ModelReranker", "RerankerPipeline", "RuleReranker"}:
        from scholar_lens.rag import reranker
        return getattr(reranker, name)
    if name == "ScholarVectorStore":
        from scholar_lens.rag.vectorstore import ScholarVectorStore
        return ScholarVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
