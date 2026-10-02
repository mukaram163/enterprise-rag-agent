import logging
from typing import List, Dict, Any, Optional
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, dense_retriever: Optional[DenseRetriever] = None, sparse_retriever: Optional[SparseRetriever] = None):
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or SparseRetriever()

    def search(self, query: str, top_k: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Combines Dense (pgvector) and Sparse (BM25) via Reciprocal Rank Fusion."""
        dense_docs = self.dense.search(query=query, top_k=top_k, user_id=user_id)
        sparse_docs = self.sparse.search(query=query, top_k=top_k, user_id=user_id)

        if not dense_docs and not sparse_docs:
            return []

        rrf_scores = {}
        doc_map = {}
        rrf_k = 60

        for rank, doc in enumerate(dense_docs):
            doc_id = doc["chunk_id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc in enumerate(sparse_docs):
            doc_id = doc["chunk_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            else:
                for key, val in doc.items():
                    if key not in doc_map[doc_id] or not doc_map[doc_id][key]:
                        doc_map[doc_id][key] = val

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = dict(doc_map[doc_id])
            doc["score"] = rrf_score
            fused_results.append(doc)

        return fused_results