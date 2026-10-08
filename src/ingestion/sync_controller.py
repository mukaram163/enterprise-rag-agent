import os
import tempfile
import json
from typing import List, Dict, Any, Optional
from src.connectors.google_drive import GoogleDriveConnector
from src.ingestion.drive_watcher import DriveWatcher
from src.ingestion.loader import GoogleDriveLoader
from src.ingestion.splitter import TextSplitter
from src.ingestion.embeddings import EmbeddingGenerator
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


class SyncController:
    """Coordinates incremental index updates based on Google Drive changes."""

    def __init__(
        self,
        connector: GoogleDriveConnector,
        watcher: DriveWatcher,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        loader: Optional[GoogleDriveLoader] = None,
        splitter: Optional[TextSplitter] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        self.connector = connector
        self.watcher = watcher
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.loader = loader or GoogleDriveLoader(connector=connector)
        self.splitter = splitter or TextSplitter()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def remove_deleted_documents(self, deleted_ids: List[str]) -> int:
        """Removes chunks associated with deleted Drive document IDs from retrievers."""
        if not deleted_ids:
            return 0

        removed_count = 0
        deleted_set = set(deleted_ids)

        # 1. Purge deleted doc IDs from SparseRetriever (BM25)
        if hasattr(self.sparse_retriever, "doc_chunks"):
            original_len = len(self.sparse_retriever.doc_chunks)
            self.sparse_retriever.doc_chunks = [
                chunk for chunk in self.sparse_retriever.doc_chunks
                if isinstance(chunk, dict) and chunk.get("doc_id") not in deleted_set
            ]
            removed_count = original_len - len(self.sparse_retriever.doc_chunks)
            if removed_count > 0 and hasattr(self.sparse_retriever, "build_index"):
                self.sparse_retriever.build_index(self.sparse_retriever.doc_chunks)

        # 2. Purge deleted doc IDs from DenseRetriever
        if hasattr(self.dense_retriever, "doc_chunks"):
            self.dense_retriever.doc_chunks = [
                chunk for chunk in self.dense_retriever.doc_chunks
                if isinstance(chunk, dict) and chunk.get("doc_id") not in deleted_set
            ]
            if removed_count > 0 and hasattr(self.dense_retriever, "rebuild_index"):
                self.dense_retriever.rebuild_index()

        return removed_count

    def process_upserted_file(self, file_meta: Dict[str, Any]) -> int:
        """Downloads, splits, embeds, and indexes a single added/modified document."""
        file_id = file_meta.get("id")
        file_name = file_meta.get("name", "Unknown")
        mime_type = file_meta.get("mimeType", "application/pdf")

        if not file_id:
            return 0

        # Purge existing chunks if this file was updated
        self.remove_deleted_documents([file_id])

        # Load document content via GoogleDriveLoader
        doc = self.loader.load_file(file_id=file_id, file_name=file_name, mime_type=mime_type)
        if not doc or not doc.page_content.strip():
            return 0

        # Split document into chunks
        chunks = self.splitter.split_document(doc)
        if not chunks:
            return 0

        # Extract text from DocumentChunk instances
        chunk_texts = [getattr(c, "text", str(c)) for c in chunks]

        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(chunk_texts)

        # Update Dense Retriever
        if hasattr(self.dense_retriever, "add_chunks") and embeddings:
            self.dense_retriever.add_chunks(chunks, embeddings)

        # Update Sparse Retriever
        if hasattr(self.sparse_retriever, "add_chunks"):
            self.sparse_retriever.add_chunks(chunks)

        return len(chunks)

    def sync(self) -> Dict[str, Any]:
        """Performs incremental synchronization based on Drive delta changes."""
        delta = self.watcher.fetch_changes()

        upserted_files = delta.get("upserted", [])
        deleted_ids = delta.get("deleted", [])

        removed_chunks = self.remove_deleted_documents(deleted_ids)

        added_chunks = 0
        for file_meta in upserted_files:
            added_chunks += self.process_upserted_file(file_meta)

        return {
            "upserted_files": len(upserted_files),
            "deleted_files": len(deleted_ids),
            "added_chunks": added_chunks,
            "removed_chunks": removed_chunks,
            "new_page_token": delta.get("new_page_token")
        }


if __name__ == "__main__":
    connector = GoogleDriveConnector()

    # Safely retrieve service instance from connector
    if hasattr(connector, "get_service"):
        service = connector.get_service()
    elif hasattr(connector, "get_drive_service"):
        service = connector.get_drive_service()
    elif hasattr(connector, "service"):
        service = connector.service
    else:
        service = getattr(connector, "_service", None)

    watcher = DriveWatcher(drive_service=service)

    splitter = TextSplitter()
    embedder = EmbeddingGenerator()
    loader = GoogleDriveLoader(connector=connector)

    try:
        dense = DenseRetriever(db_connection_string="sqlite:///:memory:")
    except Exception:
        dense = None

    try:
        sparse = SparseRetriever(chunks=[])
    except Exception:
        sparse = SparseRetriever()

    controller = SyncController(
        connector=connector,
        watcher=watcher,
        dense_retriever=dense,
        sparse_retriever=sparse,
        loader=loader,
        splitter=splitter,
        embedding_generator=embedder,
    )

    # Ensure .drive_sync_state.json contains valid JSON token state
    if not os.path.exists(".drive_sync_state.json") or os.path.getsize(".drive_sync_state.json") == 0:
        start_token = watcher.get_start_page_token()
        if hasattr(watcher, "save_token"):
            watcher.save_token(start_token)
        else:
            with open(".drive_sync_state.json", "w", encoding="utf-8") as f:
                json.dump({"saved_page_token": start_token}, f, indent=2)

    print("Executing incremental synchronization test...")
    summary = controller.sync()
    print("Sync complete! Result summary:")
    print(summary)