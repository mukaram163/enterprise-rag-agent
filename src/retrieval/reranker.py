import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class DocumentReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidate_docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if not candidate_docs:
            return []

        pairs = [[query, doc.get("text", "")] for doc in candidate_docs]
        scores = self.model.predict(pairs)

        reranked = []
        for doc, score in zip(candidate_docs, scores):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = float(score)
            reranked.append(doc_copy)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]