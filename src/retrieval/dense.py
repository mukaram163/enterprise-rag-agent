import json
from typing import List, Optional, Generator
import psycopg2
from psycopg2.extras import RealDictCursor
from src.models.document import DocumentChunk

class DenseRetriever:
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None
    ) -> List[dict]:
        if not query_vector:
            raise ValueError("query_vector cannot be empty.")

        vector_str = str(query_vector) if isinstance(query_vector, list) else query_vector

        if user_id:
            sql = """
                SELECT id, doc_id, text, embedding, allowed_users, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM document_chunks
                WHERE (
                    allowed_users IS NULL 
                    OR cardinality(allowed_users) = 0 
                    OR %s = ANY(allowed_users)
                )
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params = (vector_str, user_id, vector_str, top_k)
        else:
            sql = """
                SELECT id, doc_id, text, embedding, allowed_users, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM document_chunks
                WHERE (allowed_users IS NULL OR cardinality(allowed_users) = 0)
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params = (vector_str, vector_str, top_k)

        with psycopg2.connect(self.db_connection_string) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        chunks = []
        for row in rows:
            meta = row.get("metadata", {}) or {}
            file_name = meta.get("source") or meta.get("file_name") or row.get("doc_id", "multi_page_doc.pdf")
            page_number = meta.get("page_number", 1)

            chunks.append({
                "id": row.get("id"),
                "chunk_id": row.get("id"),
                "doc_id": row.get("doc_id"),
                "text": row.get("text"),
                "file_name": file_name,
                "page_number": page_number,
                "metadata": meta,
                "score": float(row["score"]) if row.get("score") is not None else 0.0
            })

        return chunks

    def run_query(
        self,
        query: str,
        query_vector: List[float],
        user_id: Optional[str] = None
    ) -> str:
        results = self.search(query_vector=query_vector, user_id=user_id)
        if not results:
            return "No relevant context found for your query."
        
        context_str = "\n".join([f"- {chunk['text']}" for chunk in results])
        return f"Retrieved context for '{query}':\n{context_str}"

    def run_stream(
        self,
        query: str,
        query_vector: List[float],
        user_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        if not query_vector:
            yield "Error: Vector embedding generation returned empty results for your query."
            return

        results = self.search(query_vector=query_vector, user_id=user_id)
        if not results:
            yield "No relevant context found for your query."
            return

        yield f"Retrieved {len(results)} relevant chunks:\n\n"
        for chunk in results:
            yield f"- {chunk['text']}\n"
