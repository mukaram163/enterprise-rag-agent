import json
import logging
from typing import List, Dict, Any
from src.ingestion.schema import get_db_connection
from src.models.document import DocumentChunk
from src.ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class PostgresVectorIndexer:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()

    def index_document(self, file_id: str, chunks: List[DocumentChunk]) -> int:
        """
        Safely synchronizes chunks for a single file_id.
        - If file_id produces chunks: upserts new/updated chunks and deletes old chunks for file_id.
        - If file_id produces 0 chunks: deletes all existing chunks for file_id.
        - If file_id ingestion failed before calling this method: this method is NOT called, preserving old chunks.
        """
        if not file_id:
            raise ValueError("file_id must be provided for safe document indexing.")

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            if not chunks:
                # Document produced 0 chunks: remove previous chunks for this file_id
                cur.execute("DELETE FROM document_chunks WHERE file_id = %s;", (file_id,))
                conn.commit()
                logger.info(f"File '{file_id}' produced 0 chunks. Removed existing chunks from database.")
                return 0

            texts = [c.text for c in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)

            active_chunk_ids = []
            for chunk, embedding in zip(chunks, embeddings):
                active_chunk_ids.append(chunk.chunk_id)
                meta_json = json.dumps(chunk.extra_metadata or {})

                cur.execute("""
                    INSERT INTO document_chunks 
                    (chunk_id, file_id, file_name, text, page_number, chunk_index, embedding, allowed_users, extra_metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE 
                    SET file_id = EXCLUDED.file_id,
                        file_name = EXCLUDED.file_name,
                        text = EXCLUDED.text,
                        page_number = EXCLUDED.page_number,
                        chunk_index = EXCLUDED.chunk_index,
                        embedding = EXCLUDED.embedding,
                        allowed_users = EXCLUDED.allowed_users,
                        extra_metadata = EXCLUDED.extra_metadata;
                """, (
                    chunk.chunk_id,
                    chunk.file_id,
                    chunk.file_name,
                    chunk.text,
                    chunk.page_number,
                    chunk.chunk_index,
                    embedding,
                    chunk.allowed_users or [],
                    meta_json
                ))

            # Delete stale chunks belonging ONLY to this specific file_id
            cur.execute("""
                DELETE FROM document_chunks 
                WHERE file_id = %s AND NOT (chunk_id = ANY(%s));
            """, (file_id, active_chunk_ids))

            conn.commit()
            logger.info(f"Successfully indexed {len(chunks)} chunks for file_id '{file_id}'.")
            return len(chunks)

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed indexing file_id '{file_id}': {e}")
            raise e
        finally:
            cur.close()
            conn.close()