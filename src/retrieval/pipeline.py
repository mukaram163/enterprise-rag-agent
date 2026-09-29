from typing import List, Dict, Any
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import DocumentReranker
from src.models.document import DocumentChunk

class RetrievalPipeline:
    def __init__(self, chunks: List[DocumentChunk] = None):
        self.hybrid_retriever = HybridRetriever(chunks=chunks)
        self.reranker = DocumentReranker()

    def update_corpus(self, chunks: List[DocumentChunk]):
        """Updates the internal sparse index with fresh document chunks."""
        self.hybrid_retriever.set_chunks(chunks)

    def retrieve(self, query: str, candidate_k: int = 10, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes end-to-end retrieval:
        1. Retrieves top candidate_k results using Hybrid Search (Dense + BM25 with RRF).
        2. Re-ranks candidates using Cross-Encoder.
        3. Returns top_k most relevant chunks.
        """
        candidates = self.hybrid_retriever.search(query, top_k=candidate_k)
        reranked_results = self.reranker.rerank(query, candidates, top_k=top_k)
        return reranked_results