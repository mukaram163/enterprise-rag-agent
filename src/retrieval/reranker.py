from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initializes local cross-encoder model for fine-grained semantic re-ranking."""
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query: str, candidate_docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Re-ranks candidate document chunks based on cross-encoder similarity score."""
        if not candidate_docs:
            return []

        pairs = [[query, doc["text"]] for doc in candidate_docs]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(candidate_docs, scores):
            doc["rerank_score"] = float(score)

        reranked_docs = sorted(candidate_docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs[:top_k]