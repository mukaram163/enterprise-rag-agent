from typing import List, Dict, Any
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.models.document import DocumentChunk

class HybridRetriever:
    def __init__(self, chunks: List[DocumentChunk] = None, k_rrf: int = 60):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever(chunks=chunks) if chunks else None
        self.k_rrf = k_rrf

    def set_chunks(self, chunks: List[DocumentChunk]):
        """Sets or updates corpus chunks for sparse retrieval."""
        self.sparse_retriever = SparseRetriever(chunks=chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Combines Dense and Sparse results using Reciprocal Rank Fusion (RRF)."""
        dense_results = self.dense_retriever.search(query, top_k=top_k * 2)
        
        sparse_results = []
        if self.sparse_retriever:
            sparse_results = self.sparse_retriever.search(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process dense rankings
        for rank, doc in enumerate(dense_results, start=1):
            key = f"{doc['file_name']}_p{doc['page_number']}_c{doc['chunk_index']}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.k_rrf + rank))
            doc_map[key] = doc

        # Process sparse rankings
        for rank, doc in enumerate(sparse_results, start=1):
            key = f"{doc['file_name']}_p{doc['page_number']}_c{doc['chunk_index']}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (self.k_rrf + rank))
            doc_map[key] = doc

        # Sort combined results by aggregated RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        fused_results = []
        for key in sorted_keys[:top_k]:
            doc_info = doc_map[key].copy()
            doc_info["rrf_score"] = rrf_scores[key]
            fused_results.append(doc_info)

        return fused_results