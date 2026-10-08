import logging
from typing import List
from sentence_transformers import CrossEncoder
from src.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading CrossEncoder model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: int = 5) -> List[DocumentChunk]:
        if not chunks:
            return []

        # Build (query, text) pairs for every merged candidate
        pairs = [[query, chunk.text] for chunk in chunks]
        
        # Predict cross-encoder scores
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Error predicting cross-encoder scores: {e}")
            scores = [-9999.0] * len(chunks)

        # Store in SEPARATE 'rerank_score' field without overwriting rrf_score
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        # Sort strictly by rerank_score descending
        reranked_chunks = sorted(
            chunks, 
            key=lambda c: c.rerank_score if c.rerank_score is not None else -9999.0, 
            reverse=True
        )

        return reranked_chunks[:top_k]