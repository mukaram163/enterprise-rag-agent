import logging
from typing import List, Dict, Any, Optional
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import DocumentReranker

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    def __init__(self, hybrid_retriever: Optional[HybridRetriever] = None, reranker: Optional[DocumentReranker] = None):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or DocumentReranker()

    def refresh(self):
        """Refreshes the sparse BM25 retriever index from PostgreSQL."""
        if hasattr(self.hybrid_retriever, "sparse"):
            self.hybrid_retriever.sparse.refresh()

    def retrieve(self, query: str, candidate_k: int = 10, top_k: int = 3, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves candidates using hybrid search and re-ranks them using cross-encoder models."""
        try:
            candidates = self.hybrid_retriever.search(query=query, top_k=candidate_k, user_id=user_id)
            if not candidates:
                return []

            reranked_results = self.reranker.rerank(
                query=query,
                candidate_docs=candidates,
                top_k=top_k
            )
            return reranked_results
        except Exception as e:
            logger.warning(f"Retrieval error encountered: {e}")
            return []

    def search(self, query: str, top_k: int = 3, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.retrieve(query=query, candidate_k=top_k * 2, top_k=top_k, user_id=user_id)