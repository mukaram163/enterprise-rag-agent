import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.ingestion.schema import get_db_connection
from src.models.document import DocumentChunk

logger = logging.getLogger(__name__)

class SparseRetriever:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.refresh()

    def refresh(self):
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT chunk_id, file_id, file_name, text, page_number, chunk_index, allowed_users, extra_metadata
                FROM document_chunks;
            """)
            rows = cur.fetchall()
            cur.close()
            conn.close()

            self.chunks = []
            corpus = []
            for row in rows:
                chunk = DocumentChunk(
                    chunk_id=row[0],
                    file_id=row[1],
                    file_name=row[2],
                    text=row[3],
                    page_number=row[4],
                    chunk_index=row[5],
                    allowed_users=row[6] or [],
                    extra_metadata=row[7] or {}
                )
                self.chunks.append(chunk)
                corpus.append(chunk.text.lower().split())

            if corpus:
                self.bm25 = BM25Okapi(corpus)
                logger.info(f"SparseRetriever loaded {len(corpus)} chunks for BM25 search.")
            else:
                self.bm25 = None
                logger.info("SparseRetriever found 0 documents in PostgreSQL database.")
        except Exception as e:
            logger.error(f"SparseRetriever database initialization failed: {e}")
            self.bm25 = None
            raise RuntimeError(f"Failed to refresh SparseRetriever corpus from database: {e}")

    def search(self, query: str, top_k: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            chunk = self.chunks[idx]

            # ACL Check: Restricted documents require matching user_id
            if chunk.allowed_users and len(chunk.allowed_users) > 0:
                if not user_id or user_id not in chunk.allowed_users:
                    continue

            results.append({
                "chunk_id": chunk.chunk_id,
                "file_id": chunk.file_id,
                "file_name": chunk.file_name,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "allowed_users": chunk.allowed_users,
                "extra_metadata": chunk.extra_metadata or {},
                "score": float(scores[idx])
            })

            if len(results) >= top_k:
                break

        return results