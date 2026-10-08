from typing import List, Optional
from rank_bm25 import BM25Okapi
from src.models.document import DocumentChunk

class SparseRetriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.fit(chunks)

    def fit(self, chunks: List[DocumentChunk]):
        """Builds or rebuilds the BM25 index with current chunks."""
        self.chunks = chunks
        self.corpus_tokens = [chunk.text.lower().split() for chunk in chunks] if chunks else []
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(
        self,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[DocumentChunk]:
        """
        Scores via BM25 in-memory and enforces fail-closed ACL post-filtering.
        Preserves original file_name, page_number, chunk_id, and doc_id.
        """
        if not self.bm25 or not self.chunks or not query.strip():
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Rank candidate indices descending by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        filtered_results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            
            # Optional score cutoff
            if score < min_score:
                continue

            if len(filtered_results) >= top_k:
                break

            chunk = self.chunks[idx]
            allowed_users = getattr(chunk, "allowed_users", []) or []

            # ACL Logic Validation
            is_public = len(allowed_users) == 0
            is_authorized = bool(user_id and user_id in allowed_users)

            if is_public or is_authorized:
                # Return updated DocumentChunk instance preserving all attributes and setting BM25 score
                chunk_copy = DocumentChunk(
                    id=chunk.id,
                    doc_id=getattr(chunk, "doc_id", getattr(chunk, "file_name", "")),
                    chunk_id=getattr(chunk, "chunk_id", chunk.id),
                    file_id=getattr(chunk, "file_id", getattr(chunk, "file_name", "unknown.pdf")),
                    file_name=getattr(chunk, "file_name", "unknown.pdf"),
                    page_number=getattr(chunk, "page_number", 0),
                    text=chunk.text,
                    embedding=getattr(chunk, "embedding", []),
                    allowed_users=allowed_users,
                    metadata=getattr(chunk, "metadata", {}),
                    score=score
                )
                filtered_results.append(chunk_copy)

        return filtered_results