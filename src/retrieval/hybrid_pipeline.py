import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

class HybridRetriever:
    def __init__(self, chunks: List[Any]):
        self.chunks = chunks
        
        # Helper function to access fields whether chunks are objects or dicts
        def get_attr(item, key, default=""):
            if isinstance(item, dict):
                return item.get(key, default)
            return getattr(item, key, default)

        self.get_attr = get_attr
        self.texts = [get_attr(c, "text") for c in chunks]

        # 1. Initialize Dense Vector Model & In-Memory ChromaDB
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="rag_hybrid_demo")
        
        # Ingest embeddings into ChromaDB
        embeddings = self.embed_model.encode(self.texts).tolist()
        self.collection.add(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embeddings,
            metadatas=[{
                "file_name": get_attr(c, "file_name", ""),
                "page_number": get_attr(c, "page_number", 0)
            } for c in chunks],
            documents=self.texts
        )

        # 2. Initialize Sparse BM25
        tokenized_corpus = [doc.lower().split() for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Initialize Cross-Encoder Re-ranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve(self, query: str, top_k_retrieval: int = 5, top_k_reranked: int = 2) -> Dict[str, Any]:
        """Performs Dense + BM25 retrieval, Reciprocal Rank Fusion (RRF), and Cross-Encoder Re-ranking."""
        
        # --- A. Dense Vector Retrieval ---
        query_emb = self.embed_model.encode([query]).tolist()
        dense_results = self.collection.query(query_embeddings=query_emb, n_results=top_k_retrieval)
        dense_indices = [int(i) for i in dense_results["ids"][0]]

        # --- B. Sparse BM25 Retrieval ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_retrieval].tolist()

        # --- C. Reciprocal Rank Fusion (RRF) ---
        rrf_scores: Dict[int, float] = {}
        rrf_k = 60  # Standard RRF constant

        for rank, idx in enumerate(dense_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Select top candidates from RRF
        candidate_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k_retrieval]
        candidate_chunks = [self.chunks[i] for i in candidate_indices]

        # --- D. Cross-Encoder Re-ranking Pass ---
        pairs = [[query, self.get_attr(chunk, "text")] for chunk in candidate_chunks]
        rerank_scores = self.reranker.predict(pairs)

        # Attach re-ranking scores to results
        scored_candidates = []
        for chunk, score in zip(candidate_chunks, rerank_scores):
            scored_candidates.append({
                "chunk": chunk,
                "text": self.get_attr(chunk, "text"),
                "page_number": self.get_attr(chunk, "page_number"),
                "file_name": self.get_attr(chunk, "file_name"),
                "rerank_score": float(score)
            })

        # Sort by final re-ranker score
        reranked_results = sorted(scored_candidates, key=lambda x: x["rerank_score"], reverse=True)

        return {
            "dense_candidates": [self.chunks[i] for i in dense_indices],
            "bm25_candidates": [self.chunks[i] for i in bm25_indices],
            "final_reranked": reranked_results[:top_k_reranked]
        }