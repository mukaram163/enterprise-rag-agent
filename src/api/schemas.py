"""Request models for the API."""
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    query_vector: List[float]
    user_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None


class IngestChunk(BaseModel):
    id: str
    text: str
    embedding: List[float]
    allowed_users: Optional[List[str]] = []
    metadata: Optional[dict] = {}


class IngestRequest(BaseModel):
    doc_id: str
    chunks: List[IngestChunk]
    reindex: bool = False
