from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class DocumentChunk(BaseModel):
    """
    Represents a single text chunk extracted from a document with strict metadata tracking.
    """
    chunk_id: str = Field(..., description="Unique ID for the chunk (e.g., doc_123_p4_c1)")
    file_id: str = Field(..., description="Unique file ID from Google Drive")
    file_name: str = Field(..., description="Original filename (e.g., Policy_2026.pdf)")
    page_number: int = Field(..., description="Exact page number where this chunk originates")
    text: str = Field(..., description="The actual text content of the chunk")
    chunk_index: int = Field(..., description="Index position of the chunk within the document")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    extra_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Converts chunk metadata into a payload dictionary for Qdrant storage."""
        return {
            "chunk_id": self.chunk_id,
            "file_id": self.file_id,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "created_at": self.created_at,
            **self.extra_metadata
        }

class SearchResult(BaseModel):
    """
    Represents a ranked context chunk retrieved during query processing.
    """
    chunk: DocumentChunk
    score: float = Field(..., description="Relevance score assigned by vector search or reranker")
    retrieval_source: str = Field(..., description="Source technique (e.g., 'dense', 'sparse', 'reranked')")