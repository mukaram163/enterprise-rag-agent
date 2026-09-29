from typing import List, Dict, Any
from qdrant_client import QdrantClient
from src.ingestion.embeddings import EmbeddingGenerator

class DenseRetriever:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "enterprise_rag"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedder = EmbeddingGenerator()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Encodes query and retrieves top_k matching chunks from Qdrant."""
        query_vector = self.embedder.generate_embeddings([query])[0]
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        
        retrieved_docs = []
        for point in response.points:
            retrieved_docs.append({
                "score": point.score,
                "text": point.payload.get("text"),
                "file_name": point.payload.get("file_name"),
                "page_number": point.payload.get("page_number"),
                "chunk_index": point.payload.get("chunk_index"),
            })
            
        return retrieved_docs