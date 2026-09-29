from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.models.document import DocumentChunk

class SparseRetriever:
    def __init__(self, chunks: List[DocumentChunk] = None):
        self.chunks: List[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        if chunks:
            self.index_chunks(chunks)

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenization."""
        return text.lower().split()

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Indexes document chunks into BM25 engine."""
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs BM25 keyword search against indexed chunks."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored_chunks = list(zip(self.chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored_chunks[:top_k]:
            results.append({
                "score": float(score),
                "text": chunk.text,
                "file_name": chunk.file_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            })
        return results