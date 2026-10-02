import logging
from typing import List, Dict, Any, Optional
from src.ingestion.schema import get_db_connection
from src.ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()

    def search(self, query: str, top_k: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            if user_id:
                # Match chunks if allowed_users is NULL/empty OR user_id is in allowed_users
                cur.execute("""
                    SELECT chunk_id, file_id, file_name, text, page_number, chunk_index, allowed_users, extra_metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM document_chunks
                    WHERE (allowed_users IS NULL OR cardinality(allowed_users) = 0 OR %s = ANY(allowed_users))
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                """, (query_embedding, user_id, query_embedding, top_k))
            else:
                # Match only unconstrained public chunks
                cur.execute("""
                    SELECT chunk_id, file_id, file_name, text, page_number, chunk_index, allowed_users, extra_metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM document_chunks
                    WHERE (allowed_users IS NULL OR cardinality(allowed_users) = 0)
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                """, (query_embedding, query_embedding, top_k))

            rows = cur.fetchall()
        finally:
            cur.close()
            conn.close()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row[0],
                "file_id": row[1],
                "file_name": row[2],
                "text": row[3],
                "page_number": row[4],
                "chunk_index": row[5],
                "allowed_users": row[6] if row[6] is not None else [],
                "extra_metadata": row[7] if row[7] is not None else {},
                "score": float(row[8]) if row[8] is not None else 0.0
            })

        return results