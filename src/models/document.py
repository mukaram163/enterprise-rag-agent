from datetime import datetime, timezone
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Represents a single text chunk extracted from a document.
    """

    chunk_id: str = Field(
        ...,
        description="Unique ID for the chunk"
    )

    file_id: str = Field(
        ...,
        description="Unique file ID"
    )

    file_name: str = Field(
        ...,
        description="Original filename"
    )

    page_number: int = Field(
        ...,
        description="Page number where this chunk originates"
    )

    text: str = Field(
        ...,
        description="Actual text content of the chunk"
    )

    chunk_index: int = Field(
        ...,
        description="Index position of the chunk within the document"
    )

    allowed_users: List[str] = Field(
        default_factory=list,
        description="Users authorized to retrieve this chunk"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the chunk was created"
    )

    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


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