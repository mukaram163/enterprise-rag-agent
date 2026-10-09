import os
import logging
from typing import List, Dict, Union, Optional
from src.models.document import DocumentChunk

logger = logging.getLogger(__name__)

# Environment Flags
DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "0").lower() in ("1", "true", "yes")
EVAL_MODE = os.getenv("EVAL_MODE", "0").lower() in ("1", "true", "yes")


class HybridRetriever:
    def __init__(self, dense_retriever=None, sparse_retriever=None, reranker=None):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        
        # Fallback initialization if no reranker instance is explicitly provided
        if self.reranker is None:
            self._init_reranker()

    def _init_reranker(self):
        try:
            from src.retrieval.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            if EVAL_MODE:
                raise RuntimeError(f"EVAL_MODE=1: Reranker failed to initialize: {e}") from e
            logger.warning(f"Reranker failed to initialize: {e}. Falling back to un-reranked hybrid retrieval.")

    def _to_chunk(self, item: Union[dict, DocumentChunk]) -> DocumentChunk:
        if isinstance(item, DocumentChunk):
            return item
        
        meta = item.get("metadata") or {}
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        file_name = (
            item.get("file_name") 
            or meta.get("file_name") 
            or meta.get("source") 
            or "unknown.pdf"
        )
        page_number = int(item.get("page_number") or meta.get("page_number") or 1)
        chunk_id = str(item.get("id") or item.get("chunk_id") or meta.get("chunk_id") or f"{file_name}_p{page_number}")
        doc_id = str(item.get("doc_id") or meta.get("doc_id") or file_name)

        return DocumentChunk(
            id=chunk_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            file_id=file_name,
            file_name=file_name,
            page_number=page_number,
            text=item.get("text", ""),
            embedding=item.get("embedding", []),
            allowed_users=item.get("allowed_users", []),
            metadata=meta
        )

    def search(self, query: str, query_vector: Optional[List[float]] = None, top_k: int = 5, user_id: Optional[str] = None) -> List[DocumentChunk]:
        # Lazy initialization of default retrievers if missing
        if self.dense_retriever is None:
            from src.retrieval.dense import DenseRetriever
            self.dense_retriever = DenseRetriever()
        if self.sparse_retriever is None:
            from src.retrieval.sparse import SparseRetriever
            self.sparse_retriever = SparseRetriever()

        # Generate query_vector automatically if omitted
        if query_vector is None and hasattr(self.dense_retriever, "get_embedding"):
            query_vector = self.dense_retriever.get_embedding(query)

        # 1. Fetch raw dense & sparse results
        dense_raw = self.dense_retriever.search(query_vector=query_vector, top_k=top_k * 2, user_id=user_id) if self.dense_retriever and query_vector else []
        sparse_raw = self.sparse_retriever.search(query=query, top_k=top_k * 2) if self.sparse_retriever else []

        if DEBUG_RETRIEVAL:
            print(f"[DEBUG_RETRIEVAL] Dense retrieved: {len(dense_raw)}, Sparse retrieved: {len(sparse_raw)}")

        # Standardize candidates to DocumentChunk
        dense_chunks = [self._to_chunk(d) for d in dense_raw]
        sparse_chunks = [self._to_chunk(s) for s in sparse_raw]

        # 2. Pure RRF Calculation strictly using rank position: 1 / (60 + rank)
        rrf_map: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # Dense stream scoring
        for rank, chunk in enumerate(dense_chunks, start=1):
            cid = chunk.chunk_id or chunk.id
            chunk_map[cid] = chunk
            rrf_map[cid] = rrf_map.get(cid, 0.0) + (1.0 / (60.0 + rank))

        # Sparse stream scoring
        for rank, chunk in enumerate(sparse_chunks, start=1):
            cid = chunk.chunk_id or chunk.id
            chunk_map[cid] = chunk
            rrf_map[cid] = rrf_map.get(cid, 0.0) + (1.0 / (60.0 + rank))

        # 3. Create deduplicated merged candidate list with rrf_score
        merged_chunks = []
        for cid, score in rrf_map.items():
            chunk = chunk_map[cid]
            chunk.rrf_score = score
            merged_chunks.append(chunk)

        # Sort merged list by rrf_score descending
        merged_chunks.sort(key=lambda c: c.rrf_score or 0.0, reverse=True)

        # 4. Pass merged chunks to Cross-Encoder Reranker if configured
        reranked_chunks = None
        if self.reranker and hasattr(self.reranker, "rerank"):
            try:
                reranked_chunks = self.reranker.rerank(query=query, chunks=merged_chunks, top_k=top_k)
            except Exception as e:
                if EVAL_MODE:
                    raise RuntimeError(f"EVAL_MODE=1: Reranker execution failed during search: {e}") from e
                logger.warning(f"Reranking failed ({e}), returning top {top_k} merged chunks.")

        if reranked_chunks is None:
            if DEBUG_RETRIEVAL:
                print(f"[DEBUG_RETRIEVAL] Reranker execution skipped/failed. Returning top {top_k} merged chunks.")
            reranked_chunks = merged_chunks[:top_k]

        # 5. Print debug output if enabled
        if DEBUG_RETRIEVAL:
            print("\n==========================================")
            print("--- [MERGED TOP CHUNKS (RRF)] ---")
            for c in merged_chunks:
                rrf = c.rrf_score
                rerank = getattr(c, "rerank_score", None)
                rrf_str = f"{rrf:.6f}" if rrf is not None else "N/A"
                rerank_str = f"{rerank:.6f}" if rerank is not None else "N/A"
                print(f"ID: {c.chunk_id} | Page: {c.page_number} | RRF Score: {rrf_str} | Rerank Score: {rerank_str}")

            print("\n--- [RERANKED CHUNKS] ---")
            for c in reranked_chunks:
                rrf = c.rrf_score
                rerank = getattr(c, "rerank_score", None)
                rrf_str = f"{rrf:.6f}" if rrf is not None else "N/A"
                rerank_str = f"{rerank:.6f}" if rerank is not None else "N/A"
                print(f"ID: {c.chunk_id} | Page: {c.page_number} | RRF Score: {rrf_str} | Rerank Score: {rerank_str}")
            print("==========================================\n")

        return reranked_chunks