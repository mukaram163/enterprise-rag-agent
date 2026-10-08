import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """
    Represents a single text chunk extracted from a document.
    """

    id: Optional[str] = Field(
        default=None,
        description="Database entry primary key ID"
    )

    doc_id: Optional[str] = Field(
        default=None,
        description="Associated document ID"
    )

    chunk_id: Optional[str] = Field(
        default=None,
        description="Unique ID for the chunk"
    )

    file_id: Optional[str] = Field(
        default=None,
        description="Unique file ID"
    )

    file_name: Optional[str] = Field(
        default="unknown.pdf",
        description="Original filename"
    )

    page_number: Optional[int] = Field(
        default=0,
        description="Page number where this chunk originates"
    )

    chunk_index: Optional[int] = Field(
        default=0,
        description="Index position of the chunk within the document"
    )

    text: str = Field(
        ...,
        description="Actual text content of the chunk"
    )

    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector representation of text"
    )

    allowed_users: Optional[List[str]] = Field(
        default_factory=list,
        description="Users authorized to retrieve this chunk"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="General metadata dictionary"
    )

    extra_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the chunk was created"
    )

    score: Optional[float] = Field(
        default=0.0,
        description="Relevance score assigned by search or reranker"
    )

    rrf_score: Optional[float] = Field(
        default=None,
        description="Reciprocal Rank Fusion score"
    )

    rerank_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder reranking score"
    )

    @field_validator("embedding", mode="before")
    @classmethod
    def parse_embedding(cls, v: Any) -> Optional[List[float]]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return None
        return v

    def model_post_init(self, __context: Any) -> None:
        """Fallback initialization for mandatory field aliases."""
        if not self.chunk_id:
            self.chunk_id = self.id or "chunk_unknown"
        if not self.file_id:
            self.file_id = self.doc_id or "file_unknown"
        if not self.id:
            self.id = self.chunk_id


class SearchResult(BaseModel):
    """
    Represents a ranked context chunk retrieved during query processing.
    """

    chunk: DocumentChunk

    score: float = Field(
        ...,
        description="Relevance score assigned by vector search or reranker"
    )

    retrieval_source: str = Field(
        ...,
        description="Source technique (e.g., dense, sparse, reranked)"
    )