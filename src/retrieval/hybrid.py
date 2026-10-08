import logging
from typing import List, Dict, Union, Optional
from src.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, reranker=None):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker

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

    def search(self, query: str, query_vector: List[float], top_k: int = 5, user_id: Optional[str] = None) -> List[DocumentChunk]:
        # 1. Fetch raw dense & sparse results
        dense_raw = self.dense_retriever.search(query_vector=query_vector, top_k=top_k * 2, user_id=user_id) if self.dense_retriever else []
        sparse_raw = self.sparse_retriever.search(query=query, top_k=top_k * 2) if self.sparse_retriever else []

        # Standardize all candidates to DocumentChunk
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
        if self.reranker and hasattr(self.reranker, "rerank"):
            reranked_chunks = self.reranker.rerank(query=query, chunks=merged_chunks, top_k=top_k)
        else:
            reranked_chunks = merged_chunks[:top_k]

        # 5. Print both lists showing both scores
        print("\n==========================================")
        print("--- [MERGED TOP CHUNKS (RRF)] ---")
        for c in merged_chunks:
            rrf = c.rrf_score
            rerank = c.rerank_score
            rrf_str = f"{rrf:.6f}" if rrf is not None else "N/A"
            rerank_str = f"{rerank:.6f}" if rerank is not None else "N/A"
            print(f"ID: {c.chunk_id} | Page: {c.page_number} | RRF Score: {rrf_str} | Rerank Score: {rerank_str}")

        print("\n--- [RERANKED CHUNKS] ---")
        for c in reranked_chunks:
            rrf = c.rrf_score
            rerank = c.rerank_score
            rrf_str = f"{rrf:.6f}" if rrf is not None else "N/A"
            rerank_str = f"{rerank:.6f}" if rerank is not None else "N/A"
            print(f"ID: {c.chunk_id} | Page: {c.page_number} | RRF Score: {rrf_str} | Rerank Score: {rerank_str}")
        print("==========================================\n")

        return reranked_chunks