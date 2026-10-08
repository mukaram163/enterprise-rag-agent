import json
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import Json

class DocumentIndexer:
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieves all stored chunks from document_chunks table."""
        sql = "SELECT id, text, metadata FROM document_chunks;"
        with psycopg2.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        all_chunks = []
        for row in rows:
            meta = row[2]
            if meta is None:
                meta = {}
            elif isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            all_chunks.append({
                "id": row[0],
                "text": row[1],
                "metadata": meta
            })
        return all_chunks

    def reindex_document(
        self,
        doc_id: str,
        chunks_data: List[Dict[str, Any]]
    ) -> int:
        """
        Atomically deletes old chunks for doc_id and inserts updated chunks within a single transaction.
        `chunks_data` contains dicts with keys: id, text, embedding, allowed_users, metadata.
        """
        sql_delete = "DELETE FROM document_chunks WHERE doc_id = %s;"
        sql_insert = """
            INSERT INTO document_chunks (id, doc_id, text, embedding, allowed_users, metadata)
            VALUES (%s, %s, %s, %s::vector, %s, %s);
        """

        with psycopg2.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                # Delete existing chunks
                cur.execute(sql_delete, (doc_id,))
                
                # Insert new chunks atomically
                for chunk in chunks_data:
                    cur.execute(sql_insert, (
                        chunk["id"],
                        doc_id,
                        chunk["text"],
                        chunk["embedding"],
                        chunk.get("allowed_users", []),
                        Json(chunk.get("metadata", {}))
                    ))
        
        return len(chunks_data)