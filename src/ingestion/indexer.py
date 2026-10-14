import json
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self.drive_connector = None
        self.embedder = None

    def ingest_drive_file(self, file_id: str, file_name: str) -> int:
        """Downloads, chunks, embeds, and indexes a PDF from Google Drive page by page."""
        try:
            from src.ingestion.loader import GoogleDriveLoader
            from src.ingestion.splitter import TextSplitter
            from src.ingestion.embeddings import EmbeddingGenerator

            loader = GoogleDriveLoader(connector=self.drive_connector)
            pages = loader.load_pdf_pages(file_id=file_id, file_name=file_name)
            if not pages:
                logger.warning(f"Drive file {file_name} is empty or could not be loaded.")
                return 0

            splitter = TextSplitter()
            chunks = []
            for page_doc in pages:
                chunks.extend(splitter.split_document(page_doc))
            if not chunks:
                return 0

            embedder = self.embedder or EmbeddingGenerator()
            texts = [c.text for c in chunks]
            embeddings = embedder.generate_embeddings(texts)
            chunks_data = []
            for c, emb in zip(chunks, embeddings):
                chunks_data.append({
                    "id": getattr(c, "id", None) or f"{file_id}_{getattr(c, 'chunk_id', 0)}",
                    "text": c.text,
                    "embedding": emb,
                    "allowed_users": getattr(c, "allowed_users", []),
                    "metadata": {
                        "file_name": file_name,
                        "source": file_name,
                        "page_number": getattr(c, "page_number", 1),
                    }
                })
            inserted = self.reindex_document(doc_id=file_id, chunks_data=chunks_data)
            logger.info(f"Successfully indexed Drive file {file_name}: {inserted} chunks inserted.")
            return inserted
        except Exception as e:
            logger.error(f"Failed to ingest Drive file {file_name} ({file_id}): {e}")
            return 0


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