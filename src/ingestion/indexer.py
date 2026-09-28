from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.models.document import DocumentChunk

class VectorIndexer:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "enterprise_rag"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int = 384):
        """Creates the Qdrant collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Collection '{self.collection_name}' created successfully.")

    def index_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Indexes document chunks and their corresponding embeddings into Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks must match the number of embeddings.")

        if not chunks:
            return

        vector_size = len(embeddings[0])
        self.ensure_collection(vector_size=vector_size)

        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "file_id": chunk.file_id,
                        "file_name": chunk.file_name,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Successfully indexed {len(points)} points into '{self.collection_name}'.")