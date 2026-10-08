import os
import json
from typing import Generator, Dict, Any, List, Optional

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.generation.generator import GeminiGenerator


class RAGChain:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        generator: GeminiGenerator
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.generator = generator

    def _retrieve_and_fuse(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None
    ):
        # Retrieve candidate sets
        dense_docs = self.dense.search(query_vector=query_vector, top_k=top_k, user_id=user_id)
        sparse_docs = self.sparse.search(query=query, top_k=top_k, user_id=user_id)

        # Reciprocal Rank Fusion (RRF) handling both dicts and DocumentChunk objects
        rrf_scores: Dict[str, float] = {}
        doc_map = {}

        for rank, doc in enumerate(dense_docs):
            doc_id = doc["id"] if isinstance(doc, dict) else getattr(doc, "id", str(rank))
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60 + rank + 1))

        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.id if hasattr(doc, "id") else (doc.get("id") if isinstance(doc, dict) else str(rank))
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60 + rank + 1))

        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:top_k]]

    def run_rag_pipeline(
        self,
        query: str,
        query_vector: List[float],
        user_id: Optional[str] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Executes non-streaming RAG pipeline for evaluation and debugging.
        Returns dictionary formatted for evaluation benchmarks (e.g., RAGAS / TruLens).
        """
        context_chunks = self._retrieve_and_fuse(query, query_vector, top_k=top_k, user_id=user_id)
        context_texts = [
            chunk["text"] if isinstance(chunk, dict) else getattr(chunk, "text", str(chunk))
            for chunk in context_chunks
        ]
        
        answer_text = self.generator.generate_response(query, context_chunks)

        return {
            "question": query,
            "answer": answer_text,
            "contexts": context_texts
        }

    def run_query(self, query: str, query_vector: List[float], user_id: Optional[str] = None) -> str:
        context_chunks = self._retrieve_and_fuse(query, query_vector, user_id=user_id)
        return self.generator.generate_response(query, context_chunks)

    def run_stream(self, query: str, query_vector: List[float], user_id: Optional[str] = None) -> Generator[str, None, None]:
        context_chunks = self._retrieve_and_fuse(query, query_vector, user_id=user_id)
        return self.generator.generate_stream(query, context_chunks)


def _fetch_document_chunks(indexer) -> List[Any]:
    from src.models.document import DocumentChunk

    raw_chunks = indexer.get_all_chunks()
    chunks = []
    for c in raw_chunks:
        meta = c.get("metadata") if isinstance(c, dict) else getattr(c, "metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        file_name = (
            meta.get("file_name") 
            or meta.get("source") 
            or (c.get("file_name") if isinstance(c, dict) else getattr(c, "file_name", None))
            or "unknown.pdf"
        )
        
        raw_page = meta.get("page_number")
        if raw_page is None:
            raw_page = c.get("page_number") if isinstance(c, dict) else getattr(c, "page_number", 0)
            
        try:
            page_number = int(raw_page) if raw_page is not None else 0
        except (ValueError, TypeError):
            page_number = 0

        chunk_id = c.get("id") if isinstance(c, dict) else getattr(c, "id", "")
        doc_id = (c.get("doc_id") if isinstance(c, dict) else getattr(c, "doc_id", None)) or file_name

        chunk_obj = DocumentChunk(
            id=chunk_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            file_id=file_name,
            file_name=file_name,
            page_number=page_number,
            text=c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""),
            embedding=c.get("embedding", []) if isinstance(c, dict) else getattr(c, "embedding", []),
            allowed_users=c.get("allowed_users", []) if isinstance(c, dict) else getattr(c, "allowed_users", []),
            metadata=meta
        )
        chunks.append(chunk_obj)
    return chunks


def run_rag_pipeline_standalone(query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone wrapper to embed the query, load database chunks, initialize retrievers, 
    and execute RAGChain.run_rag_pipeline.
    """
    from src.ingestion.indexer import DocumentIndexer
    from src.ingestion.embeddings import EmbeddingGenerator

    db_conn = (
        os.getenv("DATABASE_URL")
        or os.getenv("DB_CONNECTION_STRING")
        or os.getenv("POSTGRES_URL")
        or "postgresql://rag_user:rag_password@localhost:5432/enterprise_rag"
    )

    # 1. Generate Query Vector
    embedder = EmbeddingGenerator()
    query_embeddings = embedder.generate_embeddings([query])
    query_vector = query_embeddings[0] if query_embeddings else [0.0] * 384

    # 2. Initialize Database Indexer & Fetch Chunks for Sparse Search
    indexer = DocumentIndexer(db_connection_string=db_conn)
    all_chunks = _fetch_document_chunks(indexer)

    # 3. Instantiate Retrievers & Generator
    dense = DenseRetriever(db_connection_string=db_conn)
    sparse = SparseRetriever(chunks=all_chunks)
    generator = GeminiGenerator(api_key=os.getenv("GEMINI_API_KEY", ""))

    # 4. Construct RAGChain and execute
    chain = RAGChain(dense_retriever=dense, sparse_retriever=sparse, generator=generator)
    return chain.run_rag_pipeline(query=query, query_vector=query_vector, user_id=user_id)